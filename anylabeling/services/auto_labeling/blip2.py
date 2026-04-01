import logging

from PIL import Image
from PyQt6.QtCore import QCoreApplication

from anylabeling.views.labeling.utils.opencv import qt_img_to_rgb_cv_img

from .model import Model
from .registry import ModelRegistry
from .types import AutoLabelingResult

ALLOWED_BLIP2_MODELS = {
    "Salesforce/blip2-itm-vit-g",
    "Salesforce/blip2-itm-vit-g-coco",
}


@ModelRegistry.register("blip2")
class BLIP2(Model):
    """BLIP2 model family using Hugging Face Transformers."""

    class Meta:
        required_config_names = [
            "type",
            "name",
            "display_name",
            "hf_model_id",
        ]
        widgets = ["button_run", "label_prompt", "edit_prompt"]
        output_modes = {
            "rectangle": QCoreApplication.translate("Model", "Rectangle"),
        }
        default_output_mode = "rectangle"

    def __init__(self, model_config, on_message) -> None:
        super().__init__(model_config, on_message)

        self.text_prompt = ""
        self.model_id = self.config["hf_model_id"]
        if self.model_id not in ALLOWED_BLIP2_MODELS:
            raise ValueError(
                QCoreApplication.translate(
                    "Model",
                    "Unsupported BLIP2 model_id: {model_id}",
                ).format(model_id=self.model_id)
            )

        try:
            import torch
            from transformers import Blip2ForImageTextRetrieval, Blip2Processor
        except ImportError as exc:
            raise ImportError(
                "BLIP2 requires `transformers` and `torch`. "
                "Install extras with: pip install -e \".[blip2]\""
            ) from exc

        self._torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = Blip2Processor.from_pretrained(
            self.model_id,
            trust_remote_code=True,
        )
        self.model = Blip2ForImageTextRetrieval.from_pretrained(
            self.model_id,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()
        self.on_message(
            QCoreApplication.translate(
                "Model",
                "Loaded BLIP2 model on {device}: {model_id}",
            ).format(device=self.device, model_id=self.model_id)
        )

    def set_text_prompt(self, text):
        self.text_prompt = (text or "").strip()

    def _parse_labels(self):
        labels = [v.strip() for v in self.text_prompt.split(",")]
        return [v for v in labels if v]

    @staticmethod
    def _build_prompt_text(label: str) -> str:
        text = label.strip()
        if not text:
            return text
        lowered = text.lower()
        if lowered.startswith("an image showing"):
            return text
        return f"an image showing {text}"

    def _extract_scores(self, outputs):
        t = self._torch
        candidate = None
        for attr in ("itm_score", "logits_per_image", "logits"):
            candidate = getattr(outputs, attr, None)
            if candidate is not None:
                break
        if candidate is None and isinstance(outputs, (tuple, list)) and outputs:
            candidate = outputs[0]
        if candidate is None:
            raise RuntimeError("BLIP2 output did not include retrieval logits")
        if candidate.ndim == 2 and candidate.shape[-1] == 2:
            return t.softmax(candidate, dim=-1)[:, 1]
        if candidate.ndim == 2 and candidate.shape[0] == 1:
            candidate = candidate[0]
        if candidate.ndim == 2 and candidate.shape[1] == 1:
            candidate = candidate[:, 0]
        if candidate.ndim == 1:
            return t.sigmoid(candidate)
        return t.sigmoid(candidate.reshape(-1))

    def predict_shapes(self, image, image_path=None):
        if image is None:
            return AutoLabelingResult([], replace=False)

        labels = self._parse_labels()
        if not labels:
            self.on_message(
                QCoreApplication.translate(
                    "Model",
                    "BLIP2: enter comma-separated labels in text prompt.",
                )
            )
            return AutoLabelingResult([], replace=False)

        try:
            rgb = qt_img_to_rgb_cv_img(image, image_path)
            pil_image = Image.fromarray(rgb)
            prompt_labels = [self._build_prompt_text(v) for v in labels]
            inputs = self.processor(
                images=pil_image,
                text=prompt_labels,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with self._torch.no_grad():
                outputs = self.model(**inputs, use_image_text_matching_head=True)
                pos_scores = self._extract_scores(outputs)
                probs = pos_scores / pos_scores.sum().clamp_min(1e-8)

            best_idx = int(probs.argmax().item())
            best_label = labels[best_idx]
            best_score = float(probs[best_idx].item())
            self.on_message(
                QCoreApplication.translate(
                    "Model",
                    "BLIP2 top label: {label} ({score:.3f})",
                ).format(label=best_label, score=best_score)
            )
        except Exception as e:  # noqa
            logging.warning("BLIP2 inference failed")
            logging.warning(e)
            self.on_message(
                QCoreApplication.translate("Model", "BLIP2 inference failed: {error}").format(
                    error=str(e)
                )
            )

        # Classifier family reports scores only; no shape generation.
        return AutoLabelingResult([], replace=False)

    def unload(self):
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "processor"):
            del self.processor
        if hasattr(self, "_torch") and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
