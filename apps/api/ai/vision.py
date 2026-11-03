"""Image classification with explainability.

Tier 1: your fine-tuned CNN from ml/train_cnn.py (data/models/cnn.pt).
Tier 2: ImageNet-pretrained ResNet18 (works with zero training).
Tier 3: colour/texture heuristic (works with zero torch).

Includes Grad-CAM. A heatmap showing *why* the model decided is the single
highest-value-per-line feature you can show a judging panel.
"""
from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

import numpy as np

from core.config import settings

_state: dict[str, Any] = {"model": None, "classes": None, "tier": "heuristic", "device": "cpu"}


def _device():
    import torch

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _tfm():
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def load() -> dict[str, Any]:
    """Idempotent. Call at startup to pay the import cost before the demo."""
    if _state["model"] is not None:
        return available()
    try:
        import torch
        from torchvision import models

        dev = _device()
        ckpt_path = Path(settings.VISION_WEIGHTS)
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            classes = ckpt.get("classes", [])
            arch = ckpt.get("arch", "resnet18")
            net = getattr(models, arch)(weights=None)
            net.fc = torch.nn.Linear(net.fc.in_features, len(classes))
            net.load_state_dict(ckpt["state_dict"])
            _state.update(model=net.to(dev).eval(), classes=classes, tier="finetuned", device=str(dev))
        else:
            net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            from torchvision.models import ResNet18_Weights

            _state.update(
                model=net.to(dev).eval(),
                classes=list(ResNet18_Weights.DEFAULT.meta["categories"]),
                tier="imagenet",
                device=str(dev),
            )
    except Exception as e:  # torch missing / no weights downloaded / offline
        _state.update(model=None, classes=None, tier="heuristic", device="cpu", error=str(e)[:160])
    return available()


def available() -> dict[str, Any]:
    return {
        "ok": True,
        "tier": _state["tier"],
        "device": _state["device"],
        "n_classes": len(_state["classes"]) if _state["classes"] else 0,
        "weights_path": settings.VISION_WEIGHTS,
        "weights_present": Path(settings.VISION_WEIGHTS).exists(),
    }


def _heuristic(img_bytes: bytes) -> dict[str, Any]:
    """No-ML fallback: colour statistics. Honest, deterministic, never crashes."""
    try:
        from PIL import Image

        im = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((64, 64))
        a = np.asarray(im, dtype="float32") / 255.0
        r, g, b = a[..., 0].mean(), a[..., 1].mean(), a[..., 2].mean()
        edge = float(np.abs(np.diff(a.mean(axis=2), axis=0)).mean())
        labels = {"vegetation": g - (r + b) / 2, "water_or_sky": b - (r + g) / 2, "soil_or_damage": r - (g + b) / 2, "texture_heavy": edge * 4}
        tot = sum(max(v, 0.001) for v in labels.values())
        ranked = sorted(((k, max(v, 0.001) / tot) for k, v in labels.items()), key=lambda x: -x[1])
        return {"tier": "heuristic", "predictions": [{"label": k, "confidence": round(float(v), 4)} for k, v in ranked]}
    except Exception as e:
        return {"tier": "heuristic", "predictions": [], "error": str(e)[:160]}


def predict(img_bytes: bytes, topk: int = 3, explain: bool = False) -> dict[str, Any]:
    load()
    if _state["model"] is None:
        return _heuristic(img_bytes)
    try:
        import torch
        from PIL import Image

        im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        x = _tfm()(im).unsqueeze(0).to(_state["device"])
        net, classes = _state["model"], _state["classes"]

        with torch.no_grad():
            probs = torch.softmax(net(x), dim=1)[0]
        k = min(topk, probs.numel())
        conf, idx = torch.topk(probs, k)
        preds = [{"label": classes[i], "confidence": round(float(c), 4)} for c, i in zip(conf.tolist(), idx.tolist())]

        out: dict[str, Any] = {"tier": _state["tier"], "device": _state["device"], "predictions": preds}
        if explain:
            heat, err = _gradcam(net, x, int(idx[0]), im)
            out["heatmap_png_b64"] = heat
            if err:
                out["explain_error"] = err
        return out
    except Exception as e:
        res = _heuristic(img_bytes)
        res["degraded_from"] = str(e)[:160]
        return res


def _gradcam(net, x, class_idx: int, original) -> tuple[str | None, str | None]:
    """Grad-CAM over the last conv block -> base64 PNG overlay.

    Returns (image_b64, error). Errors are reported, never swallowed: a missing
    heatmap you do not know about is a surprise on stage.
    """
    try:
        import torch
        from PIL import Image

        feats: dict[str, Any] = {}
        grads: dict[str, Any] = {}
        target = net.layer4[-1]
        h1 = target.register_forward_hook(lambda m, i, o: feats.__setitem__("v", o.detach()))
        h2 = target.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("v", go[0].detach()))

        try:
            net.zero_grad(set_to_none=True)
            net(x)[0, class_idx].backward()
        finally:
            h1.remove()
            h2.remove()

        if "v" not in feats or "v" not in grads:
            return None, "hooks did not fire"

        # Channel weights = spatially averaged gradients (the Grad-CAM recipe).
        w = grads["v"].mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((w * feats["v"]).sum(dim=1, keepdim=True))
        cam = torch.nn.functional.interpolate(
            cam, size=(original.height, original.width), mode="bilinear", align_corners=False
        )
        cam = cam[0, 0].float().cpu().numpy()

        # np.ptp(), not cam.ptp(): NumPy 2.0 removed the ndarray method.
        spread = float(np.ptp(cam))
        cam = (cam - cam.min()) / (spread + 1e-8) if spread > 0 else np.zeros_like(cam)

        # Blue -> green -> red ramp. Reads as a heatmap to anyone, including judges.
        r = np.clip(1.6 * cam - 0.3, 0, 1)
        g = np.clip(1.5 - 2.4 * np.abs(cam - 0.5), 0, 1)
        b = np.clip(1.2 - 1.9 * cam, 0, 1)
        heat = np.stack([r, g, b], axis=-1) * 255.0

        base = np.asarray(original.convert("RGB"), dtype="float32")
        alpha = (0.25 + 0.45 * cam)[..., None]          # highlight hot regions more
        blend = np.clip(base * (1 - alpha) + heat * alpha, 0, 255).astype("uint8")

        buf = io.BytesIO()
        Image.fromarray(blend).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode(), None
    except Exception as e:
        return None, f"{type(e).__name__}: {str(e)[:180]}"
