#!/usr/bin/env bash
# Run this TONIGHT on hotel/home wifi. Venue wifi will be hostile.
# Caches every model weight locally so the demo needs zero downloads.
set -u
cd "$(dirname "$0")/.." || exit 1
source .venv/bin/activate 2>/dev/null || true

echo "==> caching sentence-transformers embedder (RAG)"
python - <<'PY'
try:
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    print("   dim:", m.get_sentence_embedding_dimension(), "OK")
except Exception as e:
    print("   FAILED:", e)
PY

echo "==> caching ResNet18 ImageNet weights (vision zero-shot + transfer-learning init)"
python - <<'PY'
try:
    from torchvision import models
    models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    print("   resnet18 + efficientnet_b0 cached OK")
except Exception as e:
    print("   FAILED:", e)
PY

echo "==> cache locations / sizes"
du -sh ~/.cache/huggingface 2>/dev/null
du -sh ~/.cache/torch 2>/dev/null
echo "==> done"
