"""Transfer-learning image classifier. Apple-Silicon (MPS) aware.

WHY TRANSFER LEARNING: in a 24-hour hackathon you will have 50-500 images per
class, not 50,000. Training from scratch cannot work at that size. Freezing an
ImageNet backbone and retraining only the head gets you 85-95% on an easy
2-4 class problem in under 5 minutes.

DATA LAYOUT -- just folders, one per class:
    data/raw/images/
        healthy/      img001.jpg ...
        diseased/     img042.jpg ...

RUN:
    python ml/train_cnn.py --data data/raw/images --epochs 8
    python ml/train_cnn.py --data data/raw/images --epochs 12 --unfreeze   # a bit more accuracy, slower

Writes data/models/cnn.pt which apps/api/ai/vision.py picks up automatically --
no code change needed. Restart the API and the endpoint serves your model.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms

ROOT = Path(__file__).resolve().parents[1]


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_loaders(data_dir: Path, batch: int, val_split: float, workers: int):
    # Augmentation is what buys you accuracy on a tiny dataset. Do not skip it.
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.25, 0.25, 0.2, 0.05),
        transforms.RandomRotation(12),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    full = datasets.ImageFolder(data_dir, transform=train_tf)
    n_val = max(1, int(len(full) * val_split))
    n_train = len(full) - n_val
    train_ds, val_ds = random_split(full, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # val set must not be augmented
    val_base = datasets.ImageFolder(data_dir, transform=eval_tf)
    val_ds.dataset = val_base

    return (
        DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=workers),
        DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=workers),
        full.classes,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder with one subfolder per class")
    ap.add_argument("--arch", default="resnet18", choices=["resnet18", "resnet34", "resnet50"])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--unfreeze", action="store_true", help="also fine-tune layer4 (slower, usually better)")
    ap.add_argument("--out", default=str(ROOT / "data" / "models" / "cnn.pt"))
    args = ap.parse_args()

    dev = device()
    data_dir = Path(args.data)
    if not data_dir.exists():
        raise SystemExit(f"No such folder: {data_dir}\nMake one subfolder per class and put images inside.")

    train_dl, val_dl, classes = build_loaders(data_dir, args.batch, args.val_split, args.workers)
    print(f"device={dev}  classes={classes}  train={len(train_dl.dataset)}  val={len(val_dl.dataset)}")
    if len(classes) < 2:
        raise SystemExit("Need at least 2 class folders.")

    weights = {"resnet18": models.ResNet18_Weights, "resnet34": models.ResNet34_Weights,
               "resnet50": models.ResNet50_Weights}[args.arch].DEFAULT
    net = getattr(models, args.arch)(weights=weights)

    # Freeze the backbone; retrain the head. This is the whole trick.
    for p in net.parameters():
        p.requires_grad = False
    if args.unfreeze:
        for p in net.layer4.parameters():
            p.requires_grad = True
    net.fc = nn.Linear(net.fc.in_features, len(classes))
    net = net.to(dev)

    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)

    best_acc, best_state = 0.0, None
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        net.train()
        tr_loss = tr_correct = tr_n = 0
        for x, y in train_dl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            out = net(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * y.size(0)
            tr_correct += (out.argmax(1) == y).sum().item()
            tr_n += y.size(0)
        sched.step()

        net.eval()
        va_correct = va_n = 0
        per_class = {c: [0, 0] for c in classes}   # [correct, total]
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(dev), y.to(dev)
                pred = net(x).argmax(1)
                va_correct += (pred == y).sum().item()
                va_n += y.size(0)
                for yi, pi in zip(y.tolist(), pred.tolist()):
                    per_class[classes[yi]][1] += 1
                    per_class[classes[yi]][0] += int(yi == pi)

        tr_acc, va_acc = tr_correct / max(tr_n, 1), va_correct / max(va_n, 1)
        history.append({"epoch": epoch, "train_acc": round(tr_acc, 4), "val_acc": round(va_acc, 4),
                        "train_loss": round(tr_loss / max(tr_n, 1), 4)})
        print(f"epoch {epoch:2d}/{args.epochs}  train_acc={tr_acc:.3f}  val_acc={va_acc:.3f}  "
              f"loss={tr_loss / max(tr_n, 1):.3f}  {time.time() - t0:.1f}s")

        if va_acc >= best_acc:
            best_acc = va_acc
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            best_per_class = {c: round(v[0] / v[1], 3) if v[1] else None for c, v in per_class.items()}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "classes": classes, "arch": args.arch,
                "val_acc": best_acc, "per_class_acc": best_per_class, "history": history}, out_path)

    report = {"val_acc": round(best_acc, 4), "per_class_acc": best_per_class,
              "classes": classes, "arch": args.arch, "epochs": args.epochs, "history": history}
    (out_path.parent / "cnn_report.json").write_text(json.dumps(report, indent=2))

    print(f"\nSaved {out_path}  best val_acc={best_acc:.3f}")
    print(f"Per-class accuracy: {best_per_class}")
    print("\nQuote val_acc and per-class accuracy on your slide. Never quote train accuracy.")
    print("Restart the API and /ai/vision/predict serves this model automatically.")


if __name__ == "__main__":
    main()
