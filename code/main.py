"""Message notification router: entry point.

    python code/main.py                     route dataset/messages.csv to dataset/output.csv
    python code/main.py --no-model          deterministic only, no API calls
    python code/main.py --limit 10          smoke test on the first 10 rows
    python code/main.py --refresh-media     re-run the vision and audio pass, ignoring cache
"""

from __future__ import annotations

import argparse
import collections
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from router import pipeline  # noqa: E402
from router.context import Dataset  # noqa: E402


def load_env(repo_root: str) -> None:
    """Read .env if present. Secrets come from the environment only, never the repo."""
    path = os.path.join(repo_root, ".env")
    if not os.path.exists(path):
        return
    # utf-8-sig: shell redirection on Windows writes a BOM that would corrupt the first key.
    with open(path, encoding="utf-8-sig") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def main() -> int:
    parser = argparse.ArgumentParser(description="Route WhatsApp messages to notify, digest or mute.")
    parser.add_argument("--dataset", default="dataset", help="dataset directory")
    parser.add_argument("--input", default="messages.csv", help="input CSV inside the dataset directory")
    parser.add_argument("--output", default=None, help="output CSV path")
    parser.add_argument("--no-model", action="store_true", help="deterministic layers only")
    parser.add_argument("--refresh-media", action="store_true", help="ignore the media cache")
    parser.add_argument("--limit", type=int, default=0, help="route only the first N rows")
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_env(repo_root)

    dataset_dir = args.dataset if os.path.isabs(args.dataset) else os.path.join(repo_root, args.dataset)
    output_path = args.output or os.path.join(dataset_dir, "output.csv")

    started = time.time()
    print(f"Loading dataset from {dataset_dir}")
    dataset = Dataset(dataset_dir)
    rows = dataset.incoming(args.input)
    if args.limit:
        rows = rows[: args.limit]
    print(f"Routing {len(rows)} messages")

    model_fn = None
    media_fn = None
    if not args.no_model:
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            print("  ! GEMINI_API_KEY not set, falling back to deterministic routing")
        else:
            from router.media import MediaReader  # noqa: PLC0415
            from router.model import GeminiRouter  # noqa: PLC0415

            cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
            reader = MediaReader(dataset, cache_dir, refresh=args.refresh_media)
            media_fn = reader.enrich
            model_fn = GeminiRouter(cache_dir=cache_dir).route

    decisions = pipeline.route_all(dataset, rows, model_fn=model_fn, media_fn=media_fn)
    pipeline.write_output(decisions, output_path)

    actions = collections.Counter(d.action for d in decisions)
    types = collections.Counter(d.message_type for d in decisions)
    sources = collections.Counter(d.source for d in decisions)
    with_evidence = sum(1 for d in decisions if d.evidence_message_ids != "none")

    print(f"\nWrote {len(decisions)} rows to {output_path} in {time.time() - started:.1f}s")
    print(f"  actions:  {dict(actions)}")
    print(f"  types:    {dict(types)}")
    print(f"  decided by: {dict(sources)}")
    print(f"  evidence: {with_evidence}/{len(decisions)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
