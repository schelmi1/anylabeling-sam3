from __future__ import annotations

import copy
import importlib.resources as pkg_resources
import gc
import os
import pathlib
import shutil
import tempfile
import urllib.request
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from huggingface_hub import snapshot_download
from pydantic import BaseModel, Field

from anylabeling.configs import auto_labeling as auto_labeling_configs
from anylabeling.services.auto_labeling.sam2_onnx import SegmentAnything2ONNX
from anylabeling.services.auto_labeling.sam3_onnx import SegmentAnything3ONNX


class Mark(BaseModel):
    type: Literal["point", "rectangle"]
    data: list[int]
    label: int = 1


class LoadModelRequest(BaseModel):
    model_name: str


class InferRequest(BaseModel):
    model_name: str
    image_id: str
    marks: list[Mark] = Field(default_factory=list)
    text_prompt: str | None = None
    confidence_threshold: float = 0.5


class ResetRequest(BaseModel):
    model_name: str | None = None
    clear_images: bool = True


class RuntimeOptions(BaseModel):
    cuda_low_mem: bool = True
    disable_tensorrt: bool = True
    cap_cuda_mem: bool = True
    tiny_embed_cache: bool = True
    disable_preload: bool = True


class AnnotationItem(BaseModel):
    name: str
    points: list[list[int]]


class AutosaveAnnotationsRequest(BaseModel):
    image_id: str | None = None
    model: str | None = None
    annotations: list[AnnotationItem] = Field(default_factory=list)
    reason: str | None = None


class WorkdirRequest(BaseModel):
    workdir: str


class ModelService:
    def __init__(self) -> None:
        self.model_defs = self._load_model_defs()
        self.loaded_models: dict[str, dict[str, Any]] = {}
        self.images: dict[str, np.ndarray] = {}
        self.image_meta: dict[str, dict[str, Any]] = {}
        self.runtime_options = RuntimeOptions()
        default_workdir = os.environ.get(
            "ANYLABELING_WEBAPP_WORKDIR",
            str(Path.home() / "anylabeling_data" / "webapp_workspace"),
        )
        self.workdir = Path()
        self.uploads_dir = Path()
        self.autosave_dir = Path()
        self.set_workdir(default_workdir)
        self.apply_runtime_options(self.runtime_options, unload_models=False)

    def set_workdir(self, workdir: str) -> dict[str, str]:
        try:
            root = Path(workdir).expanduser().resolve()
            root.mkdir(parents=True, exist_ok=True)
            images_dir = root / "images"
            images_dir.mkdir(parents=True, exist_ok=True)
            autosave_dir = root / "_autosave"
            autosave_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # pragma: no cover - defensive path handling
            raise HTTPException(status_code=400, detail=f"Invalid workdir: {exc}") from exc

        self.workdir = root
        self.uploads_dir = images_dir
        self.autosave_dir = autosave_dir
        return {
            "workdir": str(self.workdir),
            "images_dir": str(self.uploads_dir),
            "autosave_dir": str(self.autosave_dir),
        }

    def get_workdir(self) -> dict[str, str]:
        return {
            "workdir": str(self.workdir),
            "images_dir": str(self.uploads_dir),
            "autosave_dir": str(self.autosave_dir),
        }

    def _load_model_defs(self) -> dict[str, dict[str, Any]]:
        with pkg_resources.open_text(auto_labeling_configs, "models.yaml") as f:
            all_models = yaml.safe_load(f)

        model_defs: dict[str, dict[str, Any]] = {}
        for model in all_models:
            name = model.get("name", "")
            if model.get("type") != "segment_anything":
                continue
            if not ("sam2" in name or "sam3" in name):
                continue

            cfg = copy.deepcopy(model)
            model_dir = Path.home() / "anylabeling_data" / "models" / name
            model_dir.mkdir(parents=True, exist_ok=True)
            config_file = model_dir / "config.yaml"
            cfg["config_file"] = str(config_file)

            if config_file.exists():
                with open(config_file, encoding="utf-8-sig") as f:
                    local_cfg = yaml.safe_load(f) or {}
                cfg.update(local_cfg)
            else:
                cfg["has_downloaded"] = False
                with open(config_file, "w", encoding="utf-8") as f:
                    yaml.dump(cfg, f)

            model_defs[name] = cfg

        return model_defs

    @staticmethod
    def _find_model_folder(extract_root: Path) -> Path:
        for root, _, files in os.walk(extract_root):
            if "config.yaml" in files:
                return Path(root)
        raise RuntimeError("Could not find config.yaml in extracted model")

    def _download_zip(self, tmp_dir: Path, download_url: str) -> Path:
        zip_model_path = tmp_dir / "model.zip"
        urllib.request.urlretrieve(download_url, str(zip_model_path))
        extract_dir = tmp_dir / "extract"
        with zipfile.ZipFile(zip_model_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
        return self._find_model_folder(extract_dir)

    def _download_hf(self, tmp_dir: Path, download_url: str, model_cfg: dict[str, Any]) -> Path:
        repo_id = download_url.replace("https://huggingface.co/", "").strip("/")
        repo_id = "/".join(repo_id.split("/")[:2])
        extract_dir = tmp_dir / "extract"
        snapshot_download(repo_id=repo_id, local_dir=str(extract_dir))
        cfg_path = extract_dir / "config.yaml"
        if not cfg_path.exists():
            with open(cfg_path, "w", encoding="utf-8") as f:
                yaml.dump(model_cfg, f)
        return extract_dir

    def _ensure_downloaded(self, model_cfg: dict[str, Any]) -> dict[str, Any]:
        model_cfg = copy.deepcopy(model_cfg)
        config_file = Path(model_cfg["config_file"])
        model_dir = config_file.parent

        needs_download = not model_cfg.get("has_downloaded", False)
        needs_download = needs_download or "encoder_model_path" not in model_cfg
        needs_download = needs_download or "decoder_model_path" not in model_cfg

        if not needs_download:
            enc = model_dir / model_cfg["encoder_model_path"]
            dec = model_dir / model_cfg["decoder_model_path"]
            if not enc.exists() or not dec.exists():
                needs_download = True

        if not needs_download:
            return model_cfg

        download_url = model_cfg.get("download_url")
        if not download_url:
            raise RuntimeError(f"Model {model_cfg['name']} is not downloaded and has no download_url")

        tmp_dir = Path(tempfile.mkdtemp(prefix="anylabeling_webapp_"))
        try:
            if download_url.endswith(".zip"):
                model_folder = self._download_zip(tmp_dir, download_url)
            elif download_url.startswith("https://huggingface.co"):
                model_folder = self._download_hf(tmp_dir, download_url, model_cfg)
            else:
                raise RuntimeError(f"Unsupported model download URL: {download_url}")

            if model_dir.exists():
                shutil.rmtree(model_dir)
            shutil.move(str(model_folder), str(model_dir))

            cfg_file = model_dir / "config.yaml"
            with open(cfg_file, encoding="utf-8-sig") as f:
                downloaded_cfg = yaml.safe_load(f) or {}
            downloaded_cfg["has_downloaded"] = True
            downloaded_cfg["config_file"] = str(cfg_file)
            with open(cfg_file, "w", encoding="utf-8") as f:
                yaml.dump(downloaded_cfg, f)

            self.model_defs[model_cfg["name"]].update(downloaded_cfg)
            return self.model_defs[model_cfg["name"]]
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def list_models(self) -> list[dict[str, Any]]:
        out = []
        for name, cfg in sorted(self.model_defs.items()):
            out.append(
                {
                    "name": name,
                    "display_name": cfg.get("display_name", name),
                    "is_sam3": "sam3" in name or "language_encoder_path" in cfg,
                    "has_downloaded": bool(cfg.get("has_downloaded", False)),
                }
            )
        return out

    def load_model(self, model_name: str) -> dict[str, Any]:
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]

        cfg = self.model_defs.get(model_name)
        if not cfg:
            raise HTTPException(status_code=404, detail=f"Unknown model: {model_name}")

        cfg = self._ensure_downloaded(cfg)
        model_dir = Path(cfg["config_file"]).parent
        enc = model_dir / cfg["encoder_model_path"]
        dec = model_dir / cfg["decoder_model_path"]

        if not enc.exists() or not dec.exists():
            raise HTTPException(
                status_code=500,
                detail=f"Model files missing for {model_name}. Expected {enc} and {dec}",
            )

        is_sam3 = "sam3" in model_name or "language_encoder_path" in cfg
        if is_sam3:
            lang_path = cfg.get("language_encoder_path")
            lang_abs = str(model_dir / lang_path) if lang_path else None
            model = SegmentAnything3ONNX(str(enc), str(dec), lang_abs)
        else:
            model = SegmentAnything2ONNX(str(enc), str(dec))

        handle = {
            "name": model_name,
            "cfg": cfg,
            "model": model,
            "is_sam3": is_sam3,
            "embedding_cache": {},
        }
        self.loaded_models[model_name] = handle
        return handle

    @staticmethod
    def _safe_filename(name: str) -> str:
        cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in name)
        return cleaned or "image.png"

    @staticmethod
    def _coerce_annotations(raw: Any) -> list[dict[str, Any]]:
        if not isinstance(raw, list):
            return []
        out: list[dict[str, Any]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "AUTOLABEL_OBJECT")).strip() or "AUTOLABEL_OBJECT"
            points = item.get("points", [])
            if not isinstance(points, list):
                continue
            norm_points: list[list[int]] = []
            for pt in points:
                if not isinstance(pt, (list, tuple)) or len(pt) < 2:
                    continue
                try:
                    x = int(round(float(pt[0])))
                    y = int(round(float(pt[1])))
                except Exception:
                    continue
                norm_points.append([x, y])
            if len(norm_points) < 3:
                continue
            out.append({"name": name, "points": norm_points})
        return out

    def _load_sidecar_annotations(self, image_path: Path) -> list[dict[str, Any]]:
        sidecar = image_path.with_suffix(".json")
        legacy_sidecar = image_path.with_name(f"{image_path.stem}_annotations.json")
        target = sidecar if sidecar.exists() else legacy_sidecar
        if not target.exists():
            return []
        try:
            import json
            with open(target, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return self._coerce_annotations(data.get("annotations", []))
            return self._coerce_annotations(data)
        except Exception:
            return []

    def upload_image(self, image_bytes: bytes, filename: str | None = None) -> dict[str, Any]:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode image")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_id = str(uuid.uuid4())

        original_name = (filename or "").strip()
        safe_name = self._safe_filename(original_name) if original_name else "image.png"
        upload_path = self.uploads_dir / safe_name
        with open(upload_path, "wb") as f:
            f.write(image_bytes)
        loaded_annotations = self._load_sidecar_annotations(upload_path)

        self.images[image_id] = rgb
        self.image_meta[image_id] = {
            "uploaded_path": str(upload_path),
            "original_filename": original_name or None,
        }
        return {
            "image_id": image_id,
            "width": int(rgb.shape[1]),
            "height": int(rgb.shape[0]),
            "uploaded_path": str(upload_path),
            "annotations": loaded_annotations,
            "loaded_annotation_count": len(loaded_annotations),
        }

    @staticmethod
    def _masks_to_polygons(masks: np.ndarray) -> list[list[list[int]]]:
        polygons: list[list[list[int]]] = []
        if masks is None:
            return polygons

        arr = np.asarray(masks)
        if arr.ndim == 2:
            arr = arr[None, :, :]
        while arr.ndim > 3:
            arr = arr[:, 0]

        for mask in arr:
            if mask.dtype != np.uint8:
                mask_u8 = np.where(mask > 0, 255, 0).astype(np.uint8)
            else:
                mask_u8 = mask
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if contour.shape[0] < 3:
                    continue
                poly = contour.reshape(-1, 2).astype(int).tolist()
                polygons.append(poly)
        return polygons

    def infer(self, req: InferRequest) -> dict[str, Any]:
        handle = self.load_model(req.model_name)
        image = self.images.get(req.image_id)
        if image is None:
            raise HTTPException(status_code=404, detail=f"Unknown image_id: {req.image_id}")

        text_prompt = (req.text_prompt or "visual").strip() or "visual"
        emb_key = (req.image_id, text_prompt if handle["is_sam3"] else "")
        embedding_cache = handle["embedding_cache"]

        if emb_key not in embedding_cache:
            if handle["is_sam3"]:
                embedding_cache[emb_key] = handle["model"].encode(image, text_prompt=text_prompt)
            else:
                embedding_cache[emb_key] = handle["model"].encode(image)

        embedding = embedding_cache[emb_key]
        marks = [m.model_dump() for m in req.marks]

        if handle["is_sam3"]:
            masks = handle["model"].predict_masks(
                embedding,
                marks,
                confidence_threshold=req.confidence_threshold,
            )
        else:
            masks = handle["model"].predict_masks(embedding, marks)

        polygons = self._masks_to_polygons(masks)
        return {
            "num_polygons": len(polygons),
            "polygons": polygons,
        }

    def unload(self, model_name: str | None = None, clear_images: bool = True) -> dict[str, Any]:
        if model_name:
            handle = self.loaded_models.pop(model_name, None)
            if handle is not None:
                handle["embedding_cache"].clear()
                del handle
            unloaded = 1
        else:
            unloaded = len(self.loaded_models)
            self.loaded_models.clear()

        if clear_images:
            self.images.clear()
            self.image_meta.clear()

        # Help free native allocations held by ONNX Runtime sessions.
        gc.collect()

        return {
            "unloaded_models": unloaded,
            "cleared_images": bool(clear_images),
        }

    def get_runtime_options(self) -> RuntimeOptions:
        return self.runtime_options

    def apply_runtime_options(
        self, options: RuntimeOptions, unload_models: bool = True
    ) -> dict[str, Any]:
        self.runtime_options = options

        os.environ["ANYLABELING_CUDA_LOW_MEM"] = "1" if options.cuda_low_mem else "0"
        os.environ["ANYLABELING_ENABLE_TENSORRT"] = (
            "0" if options.disable_tensorrt else "1"
        )
        if options.cap_cuda_mem:
            os.environ["ANYLABELING_CUDA_MEM_LIMIT_MB"] = "9000"
        else:
            os.environ.pop("ANYLABELING_CUDA_MEM_LIMIT_MB", None)
        os.environ["ANYLABELING_EMBED_CACHE_SIZE"] = "1" if options.tiny_embed_cache else "10"
        os.environ["ANYLABELING_PRELOAD_SIZE"] = "0" if options.disable_preload else "7"

        unloaded = 0
        if unload_models:
            unloaded = self.unload(model_name=None, clear_images=False)["unloaded_models"]

        return {
            "runtime_options": self.runtime_options.model_dump(),
            "unloaded_models": unloaded,
        }

    def autosave_annotations(self, req: AutosaveAnnotationsRequest) -> dict[str, Any]:
        out_path: Path
        if req.image_id and req.image_id in self.image_meta:
            image_path = Path(self.image_meta[req.image_id]["uploaded_path"])
            out_path = image_path.with_suffix(".json")
        else:
            image_part = (req.image_id or "session").strip() or "session"
            safe_image_part = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in image_part)
            out_path = self.autosave_dir / f"{safe_image_part}.json"
        payload = {
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "image_id": req.image_id,
            "model": req.model,
            "reason": req.reason,
            "annotation_count": len(req.annotations),
            "annotations": [a.model_dump() for a in req.annotations],
        }
        with open(out_path, "w", encoding="utf-8") as f:
            import json
            json.dump(payload, f, indent=2)
        return {
            "ok": True,
            "path": str(out_path),
            "annotation_count": len(req.annotations),
        }


service = ModelService()

app = FastAPI(title="AnyLabeling Browser App", version="0.1.0")

FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(FRONTEND_DIR / "index.html"))


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/models")
def models() -> dict[str, Any]:
    return {"models": service.list_models()}


@app.post("/api/model/load")
def load_model(req: LoadModelRequest) -> dict[str, Any]:
    handle = service.load_model(req.model_name)
    return {
        "model_name": handle["name"],
        "is_sam3": bool(handle["is_sam3"]),
    }


@app.post("/api/image/upload")
async def upload_image(file: UploadFile = File(...)) -> dict[str, Any]:
    data = await file.read()
    return service.upload_image(data, filename=file.filename)


@app.post("/api/infer")
def infer(req: InferRequest) -> dict[str, Any]:
    return service.infer(req)


@app.post("/api/reset")
def reset(req: ResetRequest) -> dict[str, Any]:
    return service.unload(model_name=req.model_name, clear_images=req.clear_images)


@app.get("/api/runtime")
def get_runtime() -> dict[str, Any]:
    return {"runtime_options": service.get_runtime_options().model_dump()}


@app.post("/api/runtime")
def apply_runtime(options: RuntimeOptions) -> dict[str, Any]:
    return service.apply_runtime_options(options, unload_models=True)


@app.get("/api/workdir")
def get_workdir() -> dict[str, str]:
    return service.get_workdir()


@app.post("/api/workdir")
def set_workdir(req: WorkdirRequest) -> dict[str, str]:
    return service.set_workdir(req.workdir)


@app.post("/api/annotations/autosave")
def autosave_annotations(req: AutosaveAnnotationsRequest) -> dict[str, Any]:
    return service.autosave_annotations(req)
