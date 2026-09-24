"""Evaluate the router against the 30 solved rows in sample_messages.csv.

No scorer is provided with the challenge, and these 30 rows are the only labelled data that
exists. This script is the entire feedback loop.

    python code/evaluation/main.py                deterministic layers only
    python code/evaluation/main.py --model        include the Gemini pass
    python code/evaluation/main.py --errors       print every miss with its context

Read the numbers with care: 30 rows means roughly plus or minus 9 points of noise on any
accuracy figure. This is a regression guard against large mistakes, not a leaderboard.
"""

from __future__ import annotations

import argparse
import collections
import csv
import os
import sys

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(CODE_DIR)
sys.path.insert(0, CODE_DIR)

from router import pipeline  # noqa: E402
from router.context import Dataset  # noqa: E402
from router.rules import Decision  # noqa: E402

ACTIONS = ("notify", "digest", "mute")


def load_env() -> None:
    path = os.path.join(REPO_ROOT, ".env")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8-sig") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def confusion(pairs: list[tuple[str, str]]) -> str:
    """Rows are truth, columns are prediction."""
    counts: dict[tuple[str, str], int] = collections.Counter(pairs)
    width = max(len(a) for a in ACTIONS) + 2
    header = " " * (width + 8) + "".join(a.rjust(width) for a in ACTIONS)
    lines = [header]
    for truth in ACTIONS:
        cells = "".join(str(counts.get((truth, pred), 0)).rjust(width) for pred in ACTIONS)
        total = sum(counts.get((truth, p), 0) for p in ACTIONS)
        lines.append(f"  true {truth.ljust(8)}{cells}   (n={total})")
    return "\n".join(lines)


def evaluate(decisions: list[Decision], truth: list[dict[str, str]], show_errors: bool) -> float:
    by_id = {d.message_id: d for d in decisions}

    action_pairs: list[tuple[str, str]] = []
    action_hits = type_hits = both_hits = 0
    evidence_exact = evidence_overlap = evidence_gold_count = 0
    conf_correct: list[float] = []
    conf_wrong: list[float] = []
    source_stats: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    misses: list[tuple[dict[str, str], Decision]] = []

    for row in truth:
        pred = by_id.get(row["message_id"])
        if pred is None:
            continue
        gold_action, gold_type = row["action"].strip(), row["message_type"].strip()
        action_ok = pred.action == gold_action
        type_ok = pred.message_type == gold_type

        action_pairs.append((gold_action, pred.action))
        action_hits += action_ok
        type_hits += type_ok
        both_hits += action_ok and type_ok
        (conf_correct if action_ok else conf_wrong).append(pred.confidence)

        stat = source_stats[pred.source]
        stat[0] += action_ok
        stat[1] += 1

        gold_ev = {e.strip() for e in row["evidence_message_ids"].split(";") if e.strip() and e.strip() != "none"}
        pred_ev = {e.strip() for e in pred.evidence_message_ids.split(";") if e.strip() and e.strip() != "none"}
        if gold_ev:
            evidence_gold_count += 1
            if gold_ev == pred_ev:
                evidence_exact += 1
            if gold_ev & pred_ev:
                evidence_overlap += 1

        if not action_ok and show_errors:
            misses.append((row, pred))

    n = len(action_pairs)
    if not n:
        print("No overlapping message_ids between predictions and truth.")
        return 0.0

    print(f"\n{'=' * 72}\nEvaluation on {n} solved samples\n{'=' * 72}")
    print(f"  action accuracy         {action_hits}/{n}  {action_hits / n:.1%}")
    print(f"  message_type accuracy   {type_hits}/{n}  {type_hits / n:.1%}")
    print(f"  both correct            {both_hits}/{n}  {both_hits / n:.1%}")

    if evidence_gold_count:
        print(
            f"  evidence exact match    {evidence_exact}/{evidence_gold_count}  "
            f"{evidence_exact / evidence_gold_count:.1%}"
        )
        print(
            f"  evidence any overlap    {evidence_overlap}/{evidence_gold_count}  "
            f"{evidence_overlap / evidence_gold_count:.1%}"
        )

    print("\n  action confusion")
    print(confusion(action_pairs))

    print("\n  confidence calibration")
    if conf_correct:
        print(f"    mean when correct     {sum(conf_correct) / len(conf_correct):.3f}")
    if conf_wrong:
        print(f"    mean when wrong       {sum(conf_wrong) / len(conf_wrong):.3f}")
    allconf = conf_correct + conf_wrong
    print(f"    range                 {min(allconf):.2f} to {max(allconf):.2f}  (samples use 0.78 to 0.91)")

    print("\n  accuracy by deciding layer")
    for source, (hits, total) in sorted(source_stats.items()):
        print(f"    {source.ljust(10)} {hits}/{total}  {hits / total:.0%}")

    if misses:
        print(f"\n{'-' * 72}\n  {len(misses)} action misses\n{'-' * 72}")
        for row, pred in misses:
            text = (row["message_text"] or "").replace("\n", " ")[:110]
            print(f"\n  {row['message_id']}  [{row['conversation_type']}] media={row['media_type'] or 'text'}")
            print(f"    text:  {text}")
            print(f"    gold:  {row['action']:<7} / {row['message_type']}")
            print(f"    pred:  {pred.action:<7} / {pred.message_type}   ({pred.source}, {pred.reason_id})")
            print(f"    reason: {pred.reason}")

    return action_hits / n


def main() -> int:
    parser = argparse.ArgumentParser(description="Score the router against the solved samples.")
    parser.add_argument("--dataset", default=os.path.join(REPO_ROOT, "dataset"))
    parser.add_argument("--model", action="store_true", help="include the Gemini pass")
    parser.add_argument("--errors", action="store_true", help="print every action miss")
    args = parser.parse_args()

    load_env()
    dataset = Dataset(args.dataset)
    rows = dataset.incoming("sample_messages.csv")

    model_fn = media_fn = None
    if args.model:
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            print("  ! GEMINI_API_KEY not set, evaluating deterministic layers only")
        else:
            from router.media import MediaReader  # noqa: PLC0415
            from router.model import GeminiRouter  # noqa: PLC0415

            cache_dir = os.path.join(CODE_DIR, "cache")
            media_fn = MediaReader(dataset, cache_dir).enrich
            model_fn = GeminiRouter(cache_dir=cache_dir).route

    decisions = pipeline.route_all(dataset, rows, model_fn=model_fn, media_fn=media_fn, progress=False)
    evaluate(decisions, rows, args.errors)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
