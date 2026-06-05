# utils/device.py

import torch


def _get_xla_device():
    """
    Essaie de récupérer un device XLA/TPU.
    Retourne None si XLA n'est pas disponible.
    """
    try:
        import torch_xla.core.xla_model as xm

        device = xm.xla_device()

        # Petit test réel : créer un tensor sur XLA
        _ = torch.zeros(1, device=device)

        return device
    except Exception:
        return None


def get_device(requested="auto"):
    """
    Sélectionne le meilleur device disponible.

    requested:
        - "auto" : TPU/XLA si disponible, sinon CUDA, sinon CPU
        - "xla"  : force TPU/XLA, fallback CPU si indisponible
        - "cuda" : force CUDA, fallback CPU si indisponible
        - "cpu"  : force CPU
    """
    if requested is None:
        requested = "auto"

    requested = str(requested).lower().strip()

    if requested == "cpu":
        return torch.device("cpu"), "forced-cpu"

    if requested == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda"), "forced-cuda"
        return torch.device("cpu"), "cuda unavailable, fallback to cpu"

    if requested in {"xla", "tpu"}:
        xla_device = _get_xla_device()
        if xla_device is not None:
            return xla_device, "forced-xla"
        return torch.device("cpu"), "xla unavailable, fallback to cpu"

    if requested != "auto":
        return torch.device("cpu"), f"unknown device '{requested}', fallback to cpu"

    # Auto selection: TPU -> CUDA -> CPU
    xla_device = _get_xla_device()
    if xla_device is not None:
        return xla_device, "auto-xla"

    if torch.cuda.is_available():
        return torch.device("cuda"), "auto-cuda"

    return torch.device("cpu"), "auto-cpu"