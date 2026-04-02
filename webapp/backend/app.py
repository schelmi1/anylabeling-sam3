from __future__ import annotations

import copy
import importlib.resources as pkg_resources
import gc
import json
import os
import pathlib
import re
import shutil
import subprocess
import tempfile
import threading
import urllib.request
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
import yaml
import onnxruntime
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from huggingface_hub import snapshot_download
from pydantic import BaseModel, Field

from anylabeling.configs import auto_labeling as auto_labeling_configs
from anylabeling.services.auto_labeling.sam2_onnx import SegmentAnything2ONNX
from anylabeling.services.auto_labeling.sam3_onnx import SegmentAnything3ONNX

ALLOWED_METACLIP2_MODELS = {
    "facebook/metaclip-2-worldwide-s16",
    "facebook/metaclip-2-worldwide-s16-384",
}
ALLOWED_BLIP2_MODELS = {
    "Salesforce/blip2-opt-2.7b",
    "Salesforce/blip2-opt-2.7b-coco",
}


class MetaCLIP2HF:
    def __init__(self, model_id: str):
        if model_id not in ALLOWED_METACLIP2_MODELS:
            raise ValueError(f"Unsupported MetaCLIP2 model_id: {model_id}")
        try:
            import torch
            from PIL import Image
            from transformers import (
                AutoImageProcessor,
                AutoModel,
                AutoProcessor,
                AutoTokenizer,
            )
        except ImportError as exc:
            raise RuntimeError(
                "MetaCLIP2 requires `torch`, `transformers`, and `Pillow` in the webapp env."
            ) from exc

        self.model_id = model_id
        self._torch = torch
        self._image_cls = Image
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.trust_remote_code = True
        self.processor = None
        self.tokenizer = None
        self.image_processor = None
        try:
            # Prefer AutoProcessor, but force slow tokenizer path for CLIP
            # compatibility with newer transformers versions.
            self.processor = AutoProcessor.from_pretrained(
                model_id,
                use_fast=False,
                trust_remote_code=self.trust_remote_code,
            )
        except Exception:
            # Fallback: compose CLIPProcessor manually with slow tokenizer.
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_id,
                    use_fast=False,
                    from_slow=True,
                    trust_remote_code=self.trust_remote_code,
                )
            except Exception as exc:
                raise RuntimeError(
                    "MetaCLIP2 tokenizer initialization failed. "
                    "Install tokenizer deps in this env: "
                    "`pip install sentencepiece` (and restart server). "
                    f"Original error: {exc}"
                ) from exc
            self.tokenizer = tokenizer
            self.image_processor = AutoImageProcessor.from_pretrained(
                model_id,
                trust_remote_code=self.trust_remote_code,
            )
        self.model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=self.trust_remote_code,
        ).to(self.device)
        self.model.eval()

    @staticmethod
    def _build_prompt_text(label: str) -> str:
        text = label.strip()
        if not text:
            return text
        lowered = text.lower()
        if lowered.startswith("an image showing"):
            return text
        return f"an image showing {text}"

    def predict(self, image_rgb: np.ndarray, labels: list[str]) -> dict[str, Any]:
        raw_labels = [v.strip() for v in labels if v and v.strip()]
        if not raw_labels:
            return {"top_label": None, "top_score": None, "scores": []}
        prompt_labels = [self._build_prompt_text(v) for v in raw_labels]
        pil_image = self._image_cls.fromarray(image_rgb)
        if self.processor is not None:
            inputs = self.processor(
                text=prompt_labels,
                images=pil_image,
                return_tensors="pt",
                padding=True,
            )
        else:
            text_inputs = self.tokenizer(
                prompt_labels,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            image_inputs = self.image_processor(
                images=pil_image,
                return_tensors="pt",
            )
            inputs = {**text_inputs, **image_inputs}
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self._torch.no_grad():
            outputs = self.model(**inputs)
            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            probs = (100.0 * image_embeds @ text_embeds.T).softmax(dim=-1)[0]
        top_idx = int(probs.argmax().item())
        scores = [
            {"label": raw_labels[i], "score": float(probs[i].item())}
            for i in range(len(raw_labels))
        ]
        scores.sort(key=lambda x: x["score"], reverse=True)
        return {
            "top_label": raw_labels[top_idx],
            "top_score": float(probs[top_idx].item()),
            "scores": scores,
        }

    def unload(self):
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "processor"):
            del self.processor
        if hasattr(self, "_torch") and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()


class BLIP2HF:
    def __init__(self, model_id: str):
        if model_id not in ALLOWED_BLIP2_MODELS:
            raise ValueError(f"Unsupported BLIP2 model_id: {model_id}")
        try:
            import torch
            from PIL import Image
            from transformers import Blip2ForConditionalGeneration, Blip2Processor
            from keybert import KeyBERT
        except ImportError as exc:
            raise RuntimeError(
                "BLIP2 requires `torch`, `transformers`, `Pillow`, and `keybert` in the webapp env."
            ) from exc

        self.model_id = model_id
        self._torch = torch
        self._image_cls = Image
        self._keybert_cls = KeyBERT
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = Blip2Processor.from_pretrained(model_id, trust_remote_code=True)
        model_dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=model_dtype,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()
        self.keyword_model = self._keybert_cls(model="all-MiniLM-L6-v2")

    @staticmethod
    def _build_prompt_text(label: str) -> str:
        text = label.strip()
        if not text:
            return text
        lowered = text.lower()
        if lowered.startswith("an image showing"):
            return text
        return f"an image showing {text}"

    def predict(self, image_rgb: np.ndarray, labels: list[str]) -> dict[str, Any]:
        _ = labels  # BLIP2 caption mode does not require user-provided labels.
        pil_image = self._image_cls.fromarray(image_rgb)
        inputs = self.processor(images=pil_image, return_tensors="pt")
        moved_inputs = {}
        for k, v in inputs.items():
            if hasattr(v, "dtype") and getattr(v.dtype, "is_floating_point", False):
                moved_inputs[k] = v.to(self.device, dtype=self.model.dtype)
            else:
                moved_inputs[k] = v.to(self.device)
        with self._torch.no_grad():
            generated_ids = self.model.generate(
                **moved_inputs,
                max_new_tokens=32,
            )
        caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        if not caption:
            caption = "unlabeled"
        # Stage 2: extract semantic labels from BLIP2 caption via KeyBERT.
        normalized_caption = re.sub(r"\s+", " ", caption).strip()
        kw = self.keyword_model.extract_keywords(
            normalized_caption,
            keyphrase_ngram_range=(1, 2),
            stop_words="english",
            top_n=5,
            use_mmr=True,
            diversity=0.4,
        )
        keyword_scores: list[dict[str, Any]] = []
        if isinstance(kw, list):
            for item in kw:
                if not isinstance(item, (list, tuple)) or len(item) < 2:
                    continue
                label = str(item[0]).strip()
                try:
                    score_val = float(item[1])
                except Exception:
                    continue
                if not label:
                    continue
                keyword_scores.append({"label": label, "score": score_val})
        if not keyword_scores:
            keyword_scores = [{"label": normalized_caption, "score": 1.0}]
        total = sum(max(0.0, float(v["score"])) for v in keyword_scores)
        if total <= 0:
            total = 1.0
        scores = [
            {"label": v["label"], "score": float(max(0.0, float(v["score"])) / total)}
            for v in keyword_scores
        ]
        scores.sort(key=lambda x: x["score"], reverse=True)
        top = scores[0]
        return {
            "top_label": top["label"],
            "top_score": float(top["score"]),
            "scores": scores,
            "blip2_caption": normalized_caption,
            "blip2_keywords": [v["label"] for v in scores],
        }

    def unload(self):
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "processor"):
            del self.processor
        if hasattr(self, "keyword_model"):
            del self.keyword_model
        if hasattr(self, "_torch") and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()


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


class DatasetLoadRequest(BaseModel):
    folder_path: str
    recursive: bool = True
    copy_into_workdir: bool = False


class FolderDialogRequest(BaseModel):
    initial_path: str | None = None


class DatasetSelectRequest(BaseModel):
    item_id: str


class DatasetBatchInferRequest(BaseModel):
    model_name: str
    labels_text: str
    min_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    item_ids: list[str] | None = None


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
        self.dataset_items: dict[str, dict[str, Any]] = {}
        self.dataset_jobs: dict[str, dict[str, Any]] = {}
        self.dataset_jobs_lock = threading.Lock()
        self.active_model_name: str | None = None
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
            model_type = model.get("type")
            if model_type not in {"segment_anything", "metaclip2", "blip2"}:
                continue
            if model_type == "segment_anything" and not ("sam2" in name or "sam3" in name):
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
                # Keep built-in identity/type authoritative from models.yaml so
                # stale local config files cannot silently remap model families.
                cfg["name"] = name
                cfg["type"] = model_type
                if model_type in {"metaclip2", "blip2"} and model.get("hf_model_id"):
                    cfg["hf_model_id"] = model["hf_model_id"]
            else:
                cfg["has_downloaded"] = bool(cfg.get("has_downloaded", False))
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
        if model_cfg.get("type") in {"metaclip2", "blip2"}:
            # Transformers-based models are resolved directly via HF model_id.
            model_cfg["has_downloaded"] = True
            return model_cfg

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
                    "type": cfg.get("type", "segment_anything"),
                    "has_downloaded": bool(cfg.get("has_downloaded", False)),
                }
            )
        return out

    def load_model(self, model_name: str) -> dict[str, Any]:
        if model_name in self.loaded_models:
            self.active_model_name = model_name
            return self.loaded_models[model_name]

        cfg = self.model_defs.get(model_name)
        if not cfg:
            raise HTTPException(status_code=404, detail=f"Unknown model: {model_name}")

        cfg = self._ensure_downloaded(cfg)
        model_type = cfg.get("type")
        if model_type in {"metaclip2", "blip2"}:
            hf_model_id = cfg.get("hf_model_id", "")
            try:
                if model_type == "metaclip2":
                    model = MetaCLIP2HF(hf_model_id)
                else:
                    model = BLIP2HF(hf_model_id)
            except Exception as exc:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to initialize {model_type} model '{hf_model_id}': {exc}",
                ) from exc
            is_sam3 = False
        else:
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
            "type": model_type,
            "embedding_cache": {},
        }
        self.loaded_models[model_name] = handle
        self.active_model_name = model_name
        return handle

    @staticmethod
    def _is_gpu_provider(provider_name: str) -> bool:
        p = str(provider_name or "")
        return p in {
            "CUDAExecutionProvider",
            "TensorrtExecutionProvider",
            "ROCMExecutionProvider",
            "DmlExecutionProvider",
        }

    def _collect_onnx_providers_from_handle(self, handle: dict[str, Any]) -> list[str]:
        model = handle.get("model")
        if model is None:
            return []

        providers: list[str] = []
        seen: set[str] = set()

        def add_session_provider_list(session_obj: Any) -> None:
            if session_obj is None or not hasattr(session_obj, "get_providers"):
                return
            try:
                session_providers = session_obj.get_providers() or []
            except Exception:
                session_providers = []
            for p in session_providers:
                if p not in seen:
                    seen.add(p)
                    providers.append(p)

        # Common layouts across SAM2/SAM3 wrappers.
        add_session_provider_list(getattr(model, "session", None))
        add_session_provider_list(getattr(getattr(model, "encoder", None), "session", None))
        add_session_provider_list(getattr(getattr(model, "decoder", None), "session", None))
        add_session_provider_list(getattr(getattr(model, "image_encoder", None), "session", None))
        add_session_provider_list(getattr(getattr(model, "language_encoder", None), "session", None))
        return providers

    def get_compute_status(self, model_name: str | None = None) -> dict[str, Any]:
        requested = (model_name or "").strip() or None
        selected = requested or self.active_model_name
        handle = self.loaded_models.get(selected) if selected else None

        if handle is None:
            available = onnxruntime.get_available_providers() or ["CPUExecutionProvider"]
            compute_device = (
                "cuda"
                if any(self._is_gpu_provider(p) for p in available)
                else "cpu"
            )
            return {
                "loaded": False,
                "requested_model_name": requested,
                "model_name": selected,
                "model_type": None,
                "compute_device": compute_device,
                "ai_runtime": "Not loaded",
                "providers": available,
                "active_provider": (available[0] if available else None),
            }

        model_type = str(handle.get("type", "")).lower()
        if model_type in {"metaclip2", "blip2"}:
            model_obj = handle.get("model")
            device = str(getattr(model_obj, "device", "cpu")).lower()
            return {
                "loaded": True,
                "requested_model_name": requested,
                "model_name": handle.get("name"),
                "model_type": model_type,
                "compute_device": "cuda" if "cuda" in device else "cpu",
                "ai_runtime": "Transformers (PyTorch)",
                "providers": [device],
                "active_provider": device,
            }

        providers = self._collect_onnx_providers_from_handle(handle)
        if not providers:
            providers = onnxruntime.get_available_providers() or ["CPUExecutionProvider"]
        return {
            "loaded": True,
            "requested_model_name": requested,
            "model_name": handle.get("name"),
            "model_type": model_type or "segment_anything",
            "compute_device": (
                "cuda"
                if any(self._is_gpu_provider(p) for p in providers)
                else "cpu"
            ),
            "ai_runtime": "ONNX Runtime",
            "providers": providers,
            "active_provider": (providers[0] if providers else None),
        }

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

    @staticmethod
    def _labelme_shape_to_polygon(shape: dict[str, Any]) -> list[list[int]] | None:
        points = shape.get("points", [])
        if not isinstance(points, list) or len(points) < 2:
            return None
        shape_type = str(shape.get("shape_type", "polygon") or "polygon").lower()

        def _pt(v: Any) -> list[int] | None:
            if not isinstance(v, (list, tuple)) or len(v) < 2:
                return None
            try:
                return [int(round(float(v[0]))), int(round(float(v[1])))]
            except Exception:
                return None

        if shape_type == "rectangle":
            p1 = _pt(points[0])
            p2 = _pt(points[1]) if len(points) > 1 else None
            if not p1 or not p2:
                return None
            x1, y1 = p1
            x2, y2 = p2
            left, right = min(x1, x2), max(x1, x2)
            top, bottom = min(y1, y2), max(y1, y2)
            return [[left, top], [right, top], [right, bottom], [left, bottom]]

        # Default polygon path.
        poly: list[list[int]] = []
        for p in points:
            c = _pt(p)
            if c is None:
                continue
            poly.append(c)
        if len(poly) < 3:
            return None
        return poly

    def _coerce_labelme_annotations(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        shapes = data.get("shapes", [])
        if not isinstance(shapes, list):
            return []
        out: list[dict[str, Any]] = []
        for shape in shapes:
            if not isinstance(shape, dict):
                continue
            name = str(shape.get("label", "AUTOLABEL_OBJECT")).strip() or "AUTOLABEL_OBJECT"
            poly = self._labelme_shape_to_polygon(shape)
            if not poly:
                continue
            out.append({"name": name, "points": poly})
        return out

    @staticmethod
    def _image_extensions() -> set[str]:
        return {
            ".jpg",
            ".jpeg",
            ".jpe",
            ".jfif",
            ".png",
            ".bmp",
            ".dib",
            ".tif",
            ".tiff",
            ".webp",
            ".gif",
            ".ppm",
            ".pgm",
            ".pbm",
            ".pnm",
        }

    def _is_image_file(self, path: Path) -> bool:
        if path.suffix.lower() in self._image_extensions():
            return True
        # Fallback: allow formats OpenCV can read even when extension is uncommon.
        try:
            if cv2.haveImageReader(str(path)):
                return True
            img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            return img is not None
        except Exception:
            return False

    def _parse_annotations_from_json_file(self, json_path: Path) -> list[dict[str, Any]]:
        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                # AnyLabeling autosave format
                if "annotations" in data:
                    return self._coerce_annotations(data.get("annotations", []))
                # LabelMe format
                if "shapes" in data:
                    return self._coerce_labelme_annotations(data)
            if isinstance(data, list):
                return self._coerce_annotations(data)
        except Exception:
            return []
        return []

    @staticmethod
    def _extract_image_label_flags(flags: Any) -> list[str]:
        if not isinstance(flags, dict):
            return []
        out: set[str] = set()
        label_like_keys = {
            "label",
            "labels",
            "class",
            "classes",
            "category",
            "categories",
            "blip2tag",
            "blip2_keywords",
            "metaclip2_assigned_label",
            "metaclip2_top_label",
            "blip2_assigned_label",
            "blip2_top_label",
            "classification_assigned_label",
            "classification_top_label",
        }
        for key, value in flags.items():
            k = str(key).strip()
            if not k:
                continue
            k_lower = k.lower()
            if isinstance(value, bool):
                if value:
                    out.add(k)
                continue
            if isinstance(value, (int, float)):
                if float(value) != 0.0:
                    out.add(k)
                continue
            if isinstance(value, str):
                v = value.strip()
                if not v:
                    continue
                if k_lower in label_like_keys or k_lower.endswith("_label"):
                    out.add(v)
                continue
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, str) and item.strip():
                        out.add(item.strip())
        return sorted(out, key=str.lower)

    @staticmethod
    def _read_json_file(path: Path) -> dict[str, Any]:
        if path.exists():
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return data
            except Exception:
                pass
        return {}

    @staticmethod
    def _write_json_file(path: Path, data: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _build_labelme_stub(self, image_path: Path) -> dict[str, Any]:
        width = 0
        height = 0
        try:
            img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if img is not None:
                height, width = int(img.shape[0]), int(img.shape[1])
        except Exception:
            width, height = 0, 0
        return {
            "version": "5.0.1",
            "flags": {},
            "shapes": [],
            "imagePath": image_path.name,
            "imageData": None,
            "imageHeight": height,
            "imageWidth": width,
        }

    def _load_sidecar_annotations(self, image_path: Path) -> list[dict[str, Any]]:
        sidecar = image_path.with_suffix(".json")
        legacy_sidecar = image_path.with_name(f"{image_path.stem}_annotations.json")
        target = sidecar if sidecar.exists() else legacy_sidecar
        if not target.exists():
            return []
        return self._parse_annotations_from_json_file(target)

    def _create_image_session(
        self,
        image_bytes: bytes,
        filename: str,
        *,
        stored_path: Path | None = None,
        preloaded_annotations: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode image")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_id = str(uuid.uuid4())

        self.images[image_id] = rgb
        self.image_meta[image_id] = {
            "uploaded_path": str(stored_path) if stored_path else None,
            "original_filename": filename or None,
        }
        anns = preloaded_annotations if preloaded_annotations is not None else []
        return {
            "image_id": image_id,
            "width": int(rgb.shape[1]),
            "height": int(rgb.shape[0]),
            "uploaded_path": str(stored_path) if stored_path else None,
            "annotations": anns,
            "loaded_annotation_count": len(anns),
        }

    def upload_image(self, image_bytes: bytes, filename: str | None = None) -> dict[str, Any]:
        original_name = (filename or "").strip()
        safe_name = self._safe_filename(original_name) if original_name else "image.png"
        upload_path = self.uploads_dir / safe_name
        with open(upload_path, "wb") as f:
            f.write(image_bytes)
        loaded_annotations = self._load_sidecar_annotations(upload_path)
        return self._create_image_session(
            image_bytes,
            safe_name,
            stored_path=upload_path,
            preloaded_annotations=loaded_annotations,
        )

    def load_dataset_folder(
        self, folder_path: str, recursive: bool = True, copy_into_workdir: bool = False
    ) -> dict[str, Any]:
        root = Path(folder_path).expanduser().resolve()
        if not root.exists() or not root.is_dir():
            raise HTTPException(status_code=400, detail=f"Invalid folder: {root}")

        # Workdir always follows currently loaded dataset root.
        self.set_workdir(str(root))

        self.dataset_items.clear()
        # Workdir now always equals loaded folder; copying into workdir is
        # no longer meaningful and can create duplicate/self-copy issues.
        copy_into_workdir = False
        all_image_label_flags: set[str] = set()
        image_paths = root.rglob("*") if recursive else root.glob("*")
        count = 0
        scanned_files = 0
        image_candidates = 0
        for image_path in sorted(image_paths):
            if not image_path.is_file():
                continue
            scanned_files += 1
            if not self._is_image_file(image_path):
                continue
            image_candidates += 1

            target_image = image_path
            target_json = image_path.with_suffix(".json")
            if copy_into_workdir:
                safe_name = self._safe_filename(image_path.name)
                target_image = self.uploads_dir / safe_name
                shutil.copy2(image_path, target_image)
                source_json = image_path.with_suffix(".json")
                if source_json.exists():
                    target_json = target_image.with_suffix(".json")
                    shutil.copy2(source_json, target_json)
                else:
                    target_json = target_image.with_suffix(".json")

            annotations = (
                self._parse_annotations_from_json_file(target_json)
                if target_json.exists()
                else []
            )
            classification_label = None
            classification_score = None
            classification_scores: list[dict[str, Any]] = []
            metaclip2_scores: list[dict[str, Any]] = []
            blip2_scores: list[dict[str, Any]] = []
            image_label_flags: list[str] = []
            if target_json.exists():
                json_data = self._read_json_file(target_json)
                flags = json_data.get("flags", {}) if isinstance(json_data, dict) else {}
                image_label_flags = self._extract_image_label_flags(flags)
                if isinstance(flags, dict):
                    label = (
                        flags.get("classification_top_label")
                        or flags.get("metaclip2_top_label")
                        or flags.get("blip2_top_label")
                    )
                    score = (
                        flags.get("classification_top_probability")
                        if flags.get("classification_top_probability") is not None
                        else (
                            flags.get("metaclip2_top_probability")
                            if flags.get("metaclip2_top_probability") is not None
                            else flags.get("blip2_top_probability")
                        )
                    )
                    raw_scores = (
                        flags.get("classification_scores")
                        or flags.get("metaclip2_scores")
                        or flags.get("blip2_scores")
                    )
                    raw_metaclip2_scores = flags.get("metaclip2_scores")
                    raw_blip2_scores = flags.get("blip2_scores")
                    if isinstance(label, str) and label.strip():
                        classification_label = label.strip()
                    try:
                        if score is not None:
                            classification_score = float(score)
                    except Exception:
                        classification_score = None
                    if isinstance(raw_scores, list):
                        for entry in raw_scores:
                            if not isinstance(entry, dict):
                                continue
                            lbl = str(entry.get("label", "")).strip()
                            sc = entry.get("score")
                            if not lbl:
                                continue
                            try:
                                sc_val = float(sc)
                            except Exception:
                                continue
                            classification_scores.append({"label": lbl, "score": sc_val})
                    if isinstance(raw_metaclip2_scores, list):
                        for entry in raw_metaclip2_scores:
                            if not isinstance(entry, dict):
                                continue
                            lbl = str(entry.get("label", "")).strip()
                            sc = entry.get("score")
                            if not lbl:
                                continue
                            try:
                                sc_val = float(sc)
                            except Exception:
                                continue
                            metaclip2_scores.append({"label": lbl, "score": sc_val})
                    if isinstance(raw_blip2_scores, list):
                        for entry in raw_blip2_scores:
                            if not isinstance(entry, dict):
                                continue
                            lbl = str(entry.get("label", "")).strip()
                            sc = entry.get("score")
                            if not lbl:
                                continue
                            try:
                                sc_val = float(sc)
                            except Exception:
                                continue
                            blip2_scores.append({"label": lbl, "score": sc_val})
            for v in image_label_flags:
                all_image_label_flags.add(v)
            item_id = str(uuid.uuid4())
            self.dataset_items[item_id] = {
                "item_id": item_id,
                "name": target_image.name,
                "image_path": str(target_image),
                "json_path": str(target_json) if target_json.exists() else None,
                "annotation_count": len(annotations),
                "annotations": annotations,
                "classification_label": classification_label,
                "classification_score": classification_score,
                "classification_scores": classification_scores,
                "metaclip2_scores": metaclip2_scores,
                "blip2_scores": blip2_scores,
                "image_label_flags": image_label_flags,
            }
            count += 1

        items = [
            {
                "item_id": item["item_id"],
                "name": item["name"],
                "annotation_count": item["annotation_count"],
                "has_json": item["json_path"] is not None,
                "classification_label": item.get("classification_label"),
                "classification_score": item.get("classification_score"),
                "classification_scores": item.get("classification_scores", []),
                "metaclip2_scores": item.get("metaclip2_scores", []),
                "blip2_scores": item.get("blip2_scores", []),
                "image_label_flags": item.get("image_label_flags", []),
                "preview_url": f"/api/dataset/image/{item['item_id']}",
            }
            for item in self.dataset_items.values()
        ]
        return {
            "folder_path": str(root),
            "count": count,
            "scanned_files": scanned_files,
            "image_candidates": image_candidates,
            "available_image_label_flags": sorted(all_image_label_flags, key=str.lower),
            "items": items,
        }

    def get_dataset_image(self, item_id: str) -> Path:
        item = self.dataset_items.get(item_id)
        if not item:
            raise HTTPException(status_code=404, detail=f"Unknown dataset item: {item_id}")
        path = Path(item["image_path"])
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"Image file missing: {path}")
        return path

    def select_dataset_item(self, item_id: str) -> dict[str, Any]:
        item = self.dataset_items.get(item_id)
        if not item:
            raise HTTPException(status_code=404, detail=f"Unknown dataset item: {item_id}")
        image_path = Path(item["image_path"])
        if not image_path.exists():
            raise HTTPException(status_code=404, detail=f"Image file missing: {image_path}")
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        return self._create_image_session(
            image_bytes,
            image_path.name,
            stored_path=image_path,
            preloaded_annotations=item.get("annotations", []),
        )

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

        if handle.get("type") in {"metaclip2", "blip2"}:
            labels = []
            if handle.get("type") != "blip2":
                labels = [v.strip() for v in (req.text_prompt or "").split(",") if v.strip()]
            out = handle["model"].predict(image, labels)
            out.update({"num_polygons": 0, "polygons": []})
            return out

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
            masks = handle["model"].predict_masks(
                embedding,
                marks,
                confidence_threshold=req.confidence_threshold,
            )

        polygons = self._masks_to_polygons(masks)
        return {
            "num_polygons": len(polygons),
            "polygons": polygons,
        }

    def _dataset_items_payload(self) -> list[dict[str, Any]]:
        return [
            {
                "item_id": item["item_id"],
                "name": item["name"],
                "annotation_count": item["annotation_count"],
                "has_json": item["json_path"] is not None,
                "classification_label": item.get("classification_label"),
                "classification_score": item.get("classification_score"),
                "classification_scores": item.get("classification_scores", []),
                "metaclip2_scores": item.get("metaclip2_scores", []),
                "blip2_scores": item.get("blip2_scores", []),
                "image_label_flags": item.get("image_label_flags", []),
                "preview_url": f"/api/dataset/image/{item['item_id']}",
            }
            for item in self.dataset_items.values()
        ]

    def infer_dataset_metaclip(
        self,
        req: DatasetBatchInferRequest,
        *,
        progress_cb: Any | None = None,
    ) -> dict[str, Any]:
        selected_ids: list[str]
        if req.item_ids:
            selected_ids = [iid for iid in req.item_ids if iid in self.dataset_items]
            if not selected_ids:
                raise HTTPException(
                    status_code=400,
                    detail="No valid dataset items selected for batch inference.",
                )
        else:
            selected_ids = list(self.dataset_items.keys())

        updated = 0
        total = len(selected_ids)
        if progress_cb is not None:
            progress_cb(
                done=0,
                total=total,
                item_name=None,
                input_size=None,
                message=f"Initializing: loading model weights ({req.model_name})...",
            )

        handle = self.load_model(req.model_name)
        if handle.get("type") not in {"metaclip2", "blip2"}:
            raise HTTPException(
                status_code=400,
                detail="Gallery batch infer currently supports only metaclip2 and blip2 models.",
            )
        if progress_cb is not None:
            progress_cb(
                done=0,
                total=total,
                item_name=None,
                input_size=None,
                message="Initializing: preparing inference pipeline...",
            )
        labels = []
        if handle.get("type") != "blip2":
            labels = [v.strip() for v in (req.labels_text or "").split(",") if v.strip()]
            if not labels:
                raise HTTPException(
                    status_code=400,
                    detail="Provide at least one label in text prompt (comma-separated).",
                )
        if progress_cb is not None:
            progress_cb(
                done=0,
                total=total,
                item_name=None,
                input_size=None,
                message=f"Running inference on {total} image(s)...",
            )

        for idx, item_id in enumerate(selected_ids, start=1):
            item = self.dataset_items[item_id]
            image_path = Path(item["image_path"])
            bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if bgr is None:
                if progress_cb is not None:
                    progress_cb(
                        done=idx,
                        total=total,
                        item_name=item.get("name"),
                        input_size=None,
                        message=f"Skipped unreadable image: {item.get('name', 'unknown')}",
                    )
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            out = handle["model"].predict(rgb, labels)
            top_label = out.get("top_label")
            top_score = out.get("top_score")
            scores_raw = out.get("scores", [])
            scores_norm: list[dict[str, Any]] = []
            if isinstance(scores_raw, list):
                for entry in scores_raw:
                    if not isinstance(entry, dict):
                        continue
                    lbl = str(entry.get("label", "")).strip()
                    if not lbl:
                        continue
                    try:
                        sc_val = float(entry.get("score", 0.0))
                    except Exception:
                        continue
                    scores_norm.append({"label": lbl, "score": sc_val})

            try:
                score_val = float(top_score) if top_score is not None else 0.0
            except Exception:
                score_val = 0.0
            pass_threshold = score_val >= float(req.min_probability)
            assigned_label = str(top_label) if (top_label and pass_threshold) else ""

            json_path = Path(item["json_path"]) if item.get("json_path") else image_path.with_suffix(".json")
            labelme = self._read_json_file(json_path)
            if not labelme:
                labelme = self._build_labelme_stub(image_path)
            if not isinstance(labelme.get("flags"), dict):
                labelme["flags"] = {}
            labelme["imagePath"] = image_path.name
            flags = labelme["flags"]
            family = str(handle.get("type"))
            flags["classification_family"] = family
            flags["classification_model"] = req.model_name
            flags["classification_labels"] = labels
            flags["classification_top_label"] = str(top_label) if top_label else ""
            flags["classification_top_probability"] = float(score_val)
            flags["classification_scores"] = scores_norm
            flags["classification_score_map"] = {s["label"]: float(s["score"]) for s in scores_norm}
            flags["classification_threshold"] = float(req.min_probability)
            flags["classification_pass_threshold"] = bool(pass_threshold)
            flags["classification_assigned_label"] = assigned_label
            flags["classification_updated_at"] = datetime.now(timezone.utc).isoformat()
            # Backward-compatible family-prefixed fields.
            flags[f"{family}_model"] = req.model_name
            flags[f"{family}_labels"] = labels
            flags[f"{family}_top_label"] = str(top_label) if top_label else ""
            flags[f"{family}_top_probability"] = float(score_val)
            flags[f"{family}_scores"] = scores_norm
            flags[f"{family}_score_map"] = {s["label"]: float(s["score"]) for s in scores_norm}
            flags[f"{family}_threshold"] = float(req.min_probability)
            flags[f"{family}_pass_threshold"] = bool(pass_threshold)
            flags[f"{family}_assigned_label"] = assigned_label
            flags[f"{family}_updated_at"] = datetime.now(timezone.utc).isoformat()
            if family == "blip2":
                flags["blip2tag"] = str(top_label) if top_label else ""
                flags["blip2_caption"] = str(out.get("blip2_caption", "") or "")
                raw_keywords = out.get("blip2_keywords", [])
                flags["blip2_keywords"] = [
                    str(v).strip() for v in raw_keywords if str(v).strip()
                ]
            self._write_json_file(json_path, labelme)

            item["json_path"] = str(json_path)
            item["classification_label"] = str(top_label) if top_label else None
            item["classification_score"] = float(score_val)
            item["classification_scores"] = scores_norm
            family = str(handle.get("type"))
            if family == "metaclip2":
                item["metaclip2_scores"] = scores_norm
            elif family == "blip2":
                item["blip2_scores"] = scores_norm
            item["image_label_flags"] = self._extract_image_label_flags(flags)
            updated += 1
            if progress_cb is not None:
                progress_cb(
                    done=idx,
                    total=total,
                    item_name=item.get("name"),
                    input_size={"width": int(rgb.shape[1]), "height": int(rgb.shape[0])},
                    message=f"Processed {idx}/{total}: {item.get('name', 'unknown')}",
                )
        items = self._dataset_items_payload()
        return {
            "ok": True,
            "updated_count": updated,
            "labels_count": len(labels),
            "model_name": req.model_name,
            "min_probability": float(req.min_probability),
            "items": items,
        }

    def start_dataset_infer_classifier_job(self, req: DatasetBatchInferRequest) -> dict[str, Any]:
        job_id = str(uuid.uuid4())
        if req.item_ids:
            selected_total = len([iid for iid in req.item_ids if iid in self.dataset_items])
        else:
            selected_total = len(self.dataset_items)
        with self.dataset_jobs_lock:
            self.dataset_jobs[job_id] = {
                "job_id": job_id,
                "status": "running",
                "done": 0,
                "total": selected_total,
                "message": "Queued",
                "error": None,
                "updated_count": 0,
                "items": None,
                "model_name": req.model_name,
                "current_item_name": None,
                "current_input_size": None,
            }

        req_data = req.model_dump()

        def _update_job(**kwargs: Any) -> None:
            with self.dataset_jobs_lock:
                job = self.dataset_jobs.get(job_id)
                if not job:
                    return
                job.update(kwargs)

        def _worker() -> None:
            try:
                local_req = DatasetBatchInferRequest(**req_data)
                result = self.infer_dataset_metaclip(
                    local_req,
                    progress_cb=lambda done, total, item_name, input_size, message: _update_job(
                        done=int(done),
                        total=int(total),
                        current_item_name=str(item_name) if item_name else None,
                        current_input_size=input_size if isinstance(input_size, dict) else None,
                        message=str(message or ""),
                    ),
                )
                _update_job(
                    status="done",
                    done=selected_total,
                    total=selected_total,
                    message="Completed",
                    error=None,
                    updated_count=int(result.get("updated_count", 0)),
                    items=result.get("items"),
                )
            except Exception as exc:
                _update_job(status="error", error=str(exc), message="Failed")

        thread = threading.Thread(target=_worker, name=f"classifier-gallery-{job_id}", daemon=True)
        thread.start()
        return {"job_id": job_id, "status": "running", "done": 0, "total": selected_total}

    def get_dataset_infer_classifier_job(self, job_id: str) -> dict[str, Any]:
        with self.dataset_jobs_lock:
            job = self.dataset_jobs.get(job_id)
            if not job:
                raise HTTPException(status_code=404, detail=f"Unknown job_id: {job_id}")
            return dict(job)

    def unload(self, model_name: str | None = None, clear_images: bool = True) -> dict[str, Any]:
        if model_name:
            handle = self.loaded_models.pop(model_name, None)
            if handle is not None:
                handle["embedding_cache"].clear()
                del handle
            unloaded = 1
            if self.active_model_name == model_name:
                self.active_model_name = None
        else:
            unloaded = len(self.loaded_models)
            self.loaded_models.clear()
            self.active_model_name = None

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

app = FastAPI(title="AnyLabeling-Next", version="0.1.0")

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


@app.get("/api/model/compute")
def model_compute(model_name: str | None = None) -> dict[str, Any]:
    return service.get_compute_status(model_name=model_name)


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


@app.post("/api/dataset/load")
def load_dataset(req: DatasetLoadRequest) -> dict[str, Any]:
    return service.load_dataset_folder(
        req.folder_path,
        recursive=bool(req.recursive),
        copy_into_workdir=bool(req.copy_into_workdir),
    )


@app.post("/api/dialog/folder")
def open_folder_dialog(req: FolderDialogRequest) -> dict[str, Any]:
    initial_dir = None
    if req.initial_path:
        try:
            candidate = Path(req.initial_path).expanduser().resolve()
            if candidate.exists() and candidate.is_dir():
                initial_dir = str(candidate)
        except Exception:
            initial_dir = None
    if not initial_dir:
        initial_dir = str(Path.home())

    selected = ""
    native_picker_available = bool(shutil.which("zenity") or shutil.which("kdialog"))
    native_picker_cancelled = False
    native_picker_failed = False

    # Prefer desktop-native pickers first so the dialog matches normal
    # file chooser look/feel on Linux desktop environments.
    if shutil.which("zenity"):
        try:
            proc = subprocess.run(
                [
                    "zenity",
                    "--file-selection",
                    "--directory",
                    "--title=Select Dataset Folder",
                    f"--filename={initial_dir.rstrip('/')}/",
                ],
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            if proc.returncode == 0:
                selected = (proc.stdout or "").strip()
            elif proc.returncode in (1, 5):  # cancel/close
                selected = ""
                native_picker_cancelled = True
            else:
                native_picker_failed = True
        except Exception:
            native_picker_failed = True

    if not selected and shutil.which("kdialog"):
        try:
            proc = subprocess.run(
                [
                    "kdialog",
                    "--getexistingdirectory",
                    initial_dir,
                    "--title",
                    "Select Dataset Folder",
                ],
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            if proc.returncode == 0:
                selected = (proc.stdout or "").strip()
            elif proc.returncode in (1, 255):  # cancel/close
                selected = ""
                native_picker_cancelled = True
            else:
                native_picker_failed = True
        except Exception:
            native_picker_failed = True

    # If a native picker exists and user canceled it, keep it canceled.
    # Do not fall through into Tk, which can pop a second "old-style" dialog.
    if not selected and native_picker_available and native_picker_cancelled:
        return {"folder_path": ""}

    # Fallback only when no native picker exists, or native picker failed.
    if not selected and (not native_picker_available or native_picker_failed):
        try:
            import tkinter as tk
            from tkinter import filedialog
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Folder dialog is unavailable in this environment: {exc}",
            )

        root = None
        try:
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            selected = filedialog.askdirectory(
                title="Select Dataset Folder",
                initialdir=initial_dir,
                mustexist=True,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=f"Failed to open folder dialog: {exc}"
            )
        finally:
            if root is not None:
                try:
                    root.destroy()
                except Exception:
                    pass

    folder_path = str(Path(selected).resolve()) if selected else ""
    return {"folder_path": folder_path}


@app.get("/api/dataset/image/{item_id}")
def dataset_image(item_id: str) -> FileResponse:
    return FileResponse(str(service.get_dataset_image(item_id)))


@app.post("/api/dataset/select")
def select_dataset_item(req: DatasetSelectRequest) -> dict[str, Any]:
    return service.select_dataset_item(req.item_id)


@app.post("/api/dataset/infer_metaclip")
def infer_dataset_metaclip(req: DatasetBatchInferRequest) -> dict[str, Any]:
    return service.infer_dataset_metaclip(req)


@app.post("/api/dataset/infer_metaclip/start")
def start_dataset_infer_metaclip(req: DatasetBatchInferRequest) -> dict[str, Any]:
    return service.start_dataset_infer_classifier_job(req)


@app.get("/api/dataset/infer_metaclip/progress/{job_id}")
def get_dataset_infer_metaclip_progress(job_id: str) -> dict[str, Any]:
    return service.get_dataset_infer_classifier_job(job_id)


@app.post("/api/dataset/infer_classifier")
def infer_dataset_classifier(req: DatasetBatchInferRequest) -> dict[str, Any]:
    return service.infer_dataset_metaclip(req)


@app.post("/api/dataset/infer_classifier/start")
def start_dataset_infer_classifier(req: DatasetBatchInferRequest) -> dict[str, Any]:
    return service.start_dataset_infer_classifier_job(req)


@app.get("/api/dataset/infer_classifier/progress/{job_id}")
def get_dataset_infer_classifier_progress(job_id: str) -> dict[str, Any]:
    return service.get_dataset_infer_classifier_job(job_id)


@app.post("/api/annotations/autosave")
def autosave_annotations(req: AutosaveAnnotationsRequest) -> dict[str, Any]:
    return service.autosave_annotations(req)
