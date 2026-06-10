#src/utils/device.py


import torch


def get_device(requested="auto"):
    if requested is None:
        requested = "auto"

    requested = str(requested).lower()
    if requested in {"cpu", "cuda"}:
        if requested == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu"), "cuda unavailable, fallback to cpu"
        return torch.device(requested), "forced"

    if requested == "xla":
        try:
            import torch_xla.core.xla_model as xm
        except Exception:
            return torch.device("cpu"), "xla unavailable, fallback to cpu"
        return xm.xla_device(), "forced"

    # auto selection
    try:
        import torch_xla.core.xla_model as xm
        return xm.xla_device(), "auto-xla"
    except Exception:
        pass

    if torch.cuda.is_available():
        return torch.device("cuda"), "auto-cuda"

    return torch.device("cpu"), "auto-cpu"
