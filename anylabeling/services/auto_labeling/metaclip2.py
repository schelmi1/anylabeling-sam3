import logging

import numpy as np
from PIL import Image
from PyQt6.QtCore import QCoreApplication

from anylabeling.views.labeling.utils.opencv import qt_img_to_rgb_cv_img

from .model import Model
from .registry import ModelRegistry
from .types import AutoLabelingResult

ALLOWED_METACLIP2_MODELS = {
    "facebook/metaclip-2-worldwide-s16",
    "facebook/metaclip-2-worldwide-s16-384",
}


@ModelRegistry.register("metaclip2")
class MetaCLIP2(Model):
    """MetaCLIP2 model family using Hugging Face Transformers."""

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
        if self.model_id not in ALLOWED_METACLIP2_MODELS:
            raise ValueError(
                QCoreApplication.translate(
                    "Model",
                    "Unsupported MetaCLIP2 model_id: {model_id}",
                ).format(model_id=self.model_id)
            )

        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "MetaCLIP2 requires `transformers` and `torch`. "
                "Install extras with: pip install -e \".[metaclip2]\""
            ) from exc

        self._torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModel.from_pretrained(self.model_id).to(self.device)
        self.model.eval()
        self.on_message(
            QCoreApplication.translate(
                "Model",
                "Loaded MetaCLIP2 model on {device}: {model_id}",
            ).format(device=self.device, model_id=self.model_id)
        )

    def set_text_prompt(self, text):
        self.text_prompt = (text or "").strip()

    def _parse_labels(self):
        labels = [v.strip() for v in self.text_prompt.split(",")]
        return [v for v in labels if v]

    def predict_shapes(self, image, image_path=None):
        if image is None:
            return AutoLabelingResult([], replace=False)

        labels = self._parse_labels()
        if not labels:
            self.on_message(
                QCoreApplication.translate(
                    "Model",
                    "MetaCLIP2: enter comma-separated labels in text prompt.",
                )
            )
            return AutoLabelingResult([], replace=False)

        try:
            rgb = qt_img_to_rgb_cv_img(image, image_path)
            pil_image = Image.fromarray(rgb)
            inputs = self.processor(
                text=labels,
                images=pil_image,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with self._torch.no_grad():
                outputs = self.model(**inputs)
                image_embeds = outputs.image_embeds
                text_embeds = outputs.text_embeds
                image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
                text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
                probs = (100.0 * image_embeds @ text_embeds.T).softmax(dim=-1)[0]

            best_idx = int(probs.argmax().item())
            best_label = labels[best_idx]
            best_score = float(probs[best_idx].item())
            self.on_message(
                QCoreApplication.translate(
                    "Model",
                    "MetaCLIP2 top label: {label} ({score:.3f})",
                ).format(label=best_label, score=best_score)
            )
        except Exception as e:  # noqa
            logging.warning("MetaCLIP2 inference failed")
            logging.warning(e)
            self.on_message(
                QCoreApplication.translate(
                    "Model", "MetaCLIP2 inference failed: {error}"
                ).format(error=str(e))
            )

        # Current integration focuses on model-family support and score reporting.
        # We do not force shape generation for CLIP-style classifiers.
        return AutoLabelingResult([], replace=False)

    def unload(self):
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "processor"):
            del self.processor
        if hasattr(self, "_torch") and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
