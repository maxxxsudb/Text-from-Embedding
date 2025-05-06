#!/usr/bin/env bash
pip install -r requirements.txt -q
python - <<'PY'
import torch, transformers, sentence_transformers, datasets, tqdm, sacrebleu, os, json, textwrap, numpy
print("✓ Python requirements installed (torch, transformers, sentence-transformers, datasets, tqdm, sacrebleu, numpy)")
print(f"✓ PyTorch version: {torch.__version__}")
print(f"✓ Transformers version: {transformers.__version__}")
PY
