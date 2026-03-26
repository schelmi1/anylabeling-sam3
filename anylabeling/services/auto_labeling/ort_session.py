import os

import onnxruntime

from anylabeling.app_info import __preferred_device__


def _is_true(env_name: str) -> bool:
    return os.getenv(env_name, "").strip().lower() in {"1", "true", "yes", "on"}


def get_onnx_providers() -> list[str]:
    """Return execution providers in the desired priority order.

    Defaults in GPU mode:
    - Prefer CUDA
    - Keep CPU as fallback
    - Skip TensorRT unless explicitly enabled
    """
    available = onnxruntime.get_available_providers()

    # Optional manual override, e.g.:
    # ANYLABELING_ONNX_PROVIDERS=CUDAExecutionProvider,CPUExecutionProvider
    override = os.getenv("ANYLABELING_ONNX_PROVIDERS", "").strip()
    if override:
        providers = [p.strip() for p in override.split(",") if p.strip()]
        return [p for p in providers if p in available] or available

    if __preferred_device__ == "GPU":
        providers: list[str] = []
        enable_tensorrt = _is_true("ANYLABELING_ENABLE_TENSORRT")
        force_cuda = _is_true("ANYLABELING_FORCE_CUDA")

        if enable_tensorrt and "TensorrtExecutionProvider" in available:
            providers.append("TensorrtExecutionProvider")

        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        elif force_cuda:
            raise RuntimeError(
                "ANYLABELING_FORCE_CUDA is set, but CUDAExecutionProvider is unavailable. "
                f"Available providers: {available}"
            )

        if not force_cuda and "CPUExecutionProvider" in available:
            providers.append("CPUExecutionProvider")

        if providers:
            return providers

    return available or ["CPUExecutionProvider"]


def create_onnx_session(path: str) -> onnxruntime.InferenceSession:
    providers = get_onnx_providers()
    mem_limit_mb = os.getenv("ANYLABELING_CUDA_MEM_LIMIT_MB", "").strip()
    low_mem_mode = _is_true("ANYLABELING_CUDA_LOW_MEM")
    if not mem_limit_mb:
        if not low_mem_mode:
            return onnxruntime.InferenceSession(path, providers=providers)
        mem_limit_bytes = None
    else:
        try:
            mem_limit_bytes = int(float(mem_limit_mb) * 1024 * 1024)
        except ValueError:
            mem_limit_bytes = None

    configured: list[str | tuple[str, dict[str, int]]] = []
    for p in providers:
        if p == "CUDAExecutionProvider":
            cuda_options: dict[str, str | int] = {}
            if mem_limit_bytes is not None:
                cuda_options["gpu_mem_limit"] = mem_limit_bytes
            if low_mem_mode:
                # Reduce peak workspace and arena growth to avoid OOM on 16GB cards.
                cuda_options["cudnn_conv_use_max_workspace"] = "0"
                cuda_options["arena_extend_strategy"] = "kSameAsRequested"
            configured.append((p, cuda_options) if cuda_options else p)
        else:
            configured.append(p)

    return onnxruntime.InferenceSession(path, providers=configured)
