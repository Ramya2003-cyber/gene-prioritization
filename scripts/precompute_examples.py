"""Precompute example results into frontend/examples/ so the site works instantly.

Usage (from the repo root, with LLM_API_KEY or GEMINI_API_KEY set for LLM mode):
    python scripts/precompute_examples.py "ferroptosis" "Alzheimer's disease" --k 10
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")
from app import pipeline  # noqa: E402

OUT = ROOT / "frontend" / "examples"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("topics", nargs="+")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    index_path = OUT / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else []
    for topic in args.topics:
        result = pipeline.run(topic, args.k, lambda stage, msg: print(f"  [{stage}] {msg}"))
        name = re.sub(r"[^a-z0-9]+", "_", topic.lower()).strip("_") + ".json"
        (OUT / name).write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
        index = [e for e in index if e["file"] != name] + [{"topic": result["topic"], "file": name}]
        print(f"{topic}: {result['mode']}, {len(result['genes'])} genes -> {name}")
    index_path.write_text(json.dumps(index, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
