#!/usr/bin/env python3
"""Export BGE embedding model to ONNX format for lightweight server deployment.

Usage:
    uv run python scripts/export_onnx.py

Requires: optimum[onnxruntime] (dev dependency, not needed at runtime)

The exported model goes to models/bge-small-onnx/ and needs to be SCP'd to
the Lightsail server along with the tokenizer files.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
ST_MODEL_PATH = MODEL_DIR / "BAAI_bge-small-en-v1.5"
ONNX_OUTPUT_DIR = MODEL_DIR / "bge-small-onnx"


def main() -> None:
    if not ST_MODEL_PATH.exists():
        print(f"Error: sentence-transformers model not found at {ST_MODEL_PATH}")
        print(
            'Run `uv run python -c "from libtrails.embeddings import get_model; get_model()"` first'
        )
        sys.exit(1)

    try:
        from optimum.onnxruntime import ORTModelForFeatureExtraction
        from transformers import AutoTokenizer
    except ImportError:
        print("Error: optimum[onnxruntime] not installed.")
        print("Install with: uv add --dev 'optimum[onnxruntime]'")
        sys.exit(1)

    print(f"Exporting {ST_MODEL_PATH} to ONNX...")
    model = ORTModelForFeatureExtraction.from_pretrained(str(ST_MODEL_PATH), export=True)
    model.save_pretrained(str(ONNX_OUTPUT_DIR))

    tokenizer = AutoTokenizer.from_pretrained(str(ST_MODEL_PATH))
    tokenizer.save_pretrained(str(ONNX_OUTPUT_DIR))

    onnx_file = ONNX_OUTPUT_DIR / "model.onnx"
    size_mb = onnx_file.stat().st_size / 1024 / 1024
    print(f"Exported to {ONNX_OUTPUT_DIR}/")
    print(f"ONNX model size: {size_mb:.1f} MB")
    print()
    print("To deploy to server:")
    print(
        f"  scp -r {ONNX_OUTPUT_DIR} ubuntu@52.25.145.220:/home/ubuntu/projects/libtrails/models/"
    )


if __name__ == "__main__":
    main()
