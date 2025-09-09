#!/usr/bin/env python3
"""
Quick check for a local Ollama embedding model.

Usage:
  python scripts/check_ollama_embedding.py --model nomic-embed-text --text "hello world"

Environment:
  OLLAMA_HOST (optional) default: http://localhost:11434
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import requests


def main() -> int:
    parser = argparse.ArgumentParser(description="Check access to Ollama embedding model on localhost")
    parser.add_argument("--model", default="nomic-embed-text", help="Embedding model name (e.g., nomic-embed-text, mxbai-embed-large)")
    parser.add_argument("--text", default="hello world", help="Text to embed for the health check")
    parser.add_argument("--host", default=os.getenv("OLLAMA_HOST", "http://localhost:11434"), help="Ollama host, overrides OLLAMA_HOST env")
    parser.add_argument("--timeout", type=float, default=8.0, help="HTTP timeout in seconds")
    args = parser.parse_args()

    url = args.host.rstrip("/") + "/api/embeddings"
    payload = {"model": args.model, "prompt": args.text}

    try:
        resp = requests.post(url, json=payload, timeout=args.timeout)
    except requests.exceptions.ConnectionError as e:
        print(f"FAILED: Cannot connect to {url}\n- Is Ollama running? Try: 'ollama serve'\n- Default port 11434 must be accessible\n- Host used: {args.host}\nDetails: {e}")
        return 2
    except requests.exceptions.Timeout:
        print(f"FAILED: Request to {url} timed out after {args.timeout}s")
        return 3
    except Exception as e:  # noqa: BLE001
        print(f"FAILED: Unexpected error calling {url}: {e}")
        return 4

    if resp.status_code != 200:
        # Common case: model not found
        tip = ""
        try:
            data = resp.json()
            msg = data.get("error") or data
            if isinstance(msg, str) and "model" in msg.lower() and "not" in msg.lower():
                tip = f"\nTip: pull the model first: ollama pull {args.model}"
        except Exception:  # noqa: BLE001
            msg = resp.text
        print(f"FAILED: HTTP {resp.status_code} from Ollama: {msg}{tip}")
        return 5

    try:
        data: dict[str, Any] = resp.json()
    except ValueError:
        print("FAILED: Response was not valid JSON")
        return 6

    embedding = data.get("embedding")
    if not isinstance(embedding, list) or not embedding:
        print(f"FAILED: No embedding returned. Response: {data}")
        return 7

    dim = len(embedding)
    preview = ", ".join(f"{v:.4f}" for v in embedding[:5])
    print(
        "SUCCESS: Connected to Ollama and received an embedding\n"
        f"- Host: {args.host}\n"
        f"- Model: {args.model}\n"
        f"- Text: {args.text!r}\n"
        f"- Dimensions: {dim}\n"
        f"- Preview: [{preview}, ...]"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

