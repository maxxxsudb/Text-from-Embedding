#!/usr/bin/env bash
pip install -r requirements.txt -q
python - <<'PY'
import torch, transformers, sentence_transformers, os, json, textwrap
print("✓  Requirements installed")
PY
