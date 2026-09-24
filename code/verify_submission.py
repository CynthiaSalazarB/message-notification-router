"""Check output.csv against the submission contract before submitting.

Matching the harness exactly is the cheapest way to avoid losing points, and a malformed CSV
throws away every point the router earned. Run this last, every time.

    python code/verify_submission.py
"""

from __future__ import annotations

import argparse
import collections
import csv
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REQUIRED_COLUMNS = ["message_id", "action", "message_type", "reason", "confidence", "evidence_message_ids"]
VALID_ACTIONS = {"notify", "digest", "mute"}
VALID_TYPES = {
    "personal", "urgent", "event", "payment", "business_update",
    "promotion", "greeting", "forward", "spam", "scam", "unknown",
}


def check(output_path: str, messages_path: str, history_path: str) -> int:
    errors: list[str] = []
    warnings: list[str] = []

    with open(messages_path, encoding="utf-8", newline="") as fh:
        expected_ids = [r["message_id"] for r in csv.DictReader(fh)]
    with open(history_path, encoding="utf-8", newline="") as fh:
        known_history = {r["message_id"] for r in csv.DictReader(fh)}

    with open(output_path, encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, [])
        rows = list(reader)

    if header != REQUIRED_COLUMNS:
        errors.append(f"header mismatch\n    expected {REQUIRED_COLUMNS}\n    found    {header}")

    seen: list[str] = []
    for i, row in enumerate(rows, start=2):
        if len(row) != len(REQUIRED_COLUMNS):
            errors.append(f"line {i}: expected {len(REQUIRED_COLUMNS)} fields, found {len(row)}")
            continue
        message_id, action, message_type, reason, confidence, evidence = row
        seen.append(message_id)

        if action not in VALID_ACTIONS:
            errors.append(f"line {i}: invalid action {action!r}")
        if message_type not in VALID_TYPES:
            errors.append(f"line {i}: invalid message_type {message_type!r}")
        if not reason.strip():
            errors.append(f"line {i}: empty reason")

        try:
            value = float(confidence)
            if not 0.0 <= value <= 1.0:
                errors.append(f"line {i}: confidence {value} outside 0 to 1")
            elif not 0.7 <= value <= 0.95:
                warnings.append(f"line {i}: confidence {value} outside the sample range of 0.78 to 0.91")
        except ValueError:
            errors.append(f"line {i}: confidence {confidence!r} is not a number")

        if evidence.strip() != "none":
            for eid in evidence.split(";"):
                eid = eid.strip()
                if not eid:
                    errors.append(f"line {i}: empty evidence id in {evidence!r}")
                elif eid not in known_history:
                    errors.append(f"line {i}: evidence {eid!r} is not a real message_history id")

    missing = [m for m in expected_ids if m not in set(seen)]
    extra = [m for m in seen if m not in set(expected_ids)]
    duplicates = [m for m, n in collections.Counter(seen).items() if n > 1]

    if missing:
        errors.append(f"{len(missing)} message_ids missing from output: {missing[:5]}")
    if extra:
        errors.append(f"{len(extra)} unexpected message_ids in output: {extra[:5]}")
    if duplicates:
        errors.append(f"{len(duplicates)} duplicated message_ids: {duplicates[:5]}")
    if len(rows) != len(expected_ids):
        errors.append(f"row count {len(rows)} does not match messages.csv count {len(expected_ids)}")

    print(f"Checking {output_path}")
    print(f"  rows            {len(rows)} (messages.csv has {len(expected_ids)})")
    print(f"  columns         {'ok' if header == REQUIRED_COLUMNS else 'MISMATCH'}")

    if rows and not errors:
        actions = collections.Counter(r[1] for r in rows if len(r) == 6)
        types = collections.Counter(r[2] for r in rows if len(r) == 6)
        with_evidence = sum(1 for r in rows if len(r) == 6 and r[5] != "none")
        print(f"  actions         {dict(actions)}")
        print(f"  types           {dict(types)}")
        print(f"  with evidence   {with_evidence}/{len(rows)}")

    for w in warnings[:10]:
        print(f"  warning: {w}")
    if len(warnings) > 10:
        print(f"  ... and {len(warnings) - 10} more warnings")

    if errors:
        print(f"\nFAILED with {len(errors)} error(s):")
        for e in errors[:20]:
            print(f"  - {e}")
        return 1

    print("\nPASS: output.csv satisfies the submission contract.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate output.csv against the submission contract.")
    parser.add_argument("--dataset", default=os.path.join(REPO_ROOT, "dataset"))
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    output = args.output or os.path.join(args.dataset, "output.csv")
    return check(
        output,
        os.path.join(args.dataset, "messages.csv"),
        os.path.join(args.dataset, "message_history.csv"),
    )


if __name__ == "__main__":
    raise SystemExit(main())
