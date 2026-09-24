"""The routing pipeline: context, media, rules, model, evidence, row.

Order matters and is the whole architecture:

  1. context   deterministic join of the dataset CSVs
  2. media     images and voice notes resolved to text, cached by media_id
  3. safety    authoritative rules the model is not allowed to override
  4. rules     confident preference rules (opt-outs, mutes, repetition)
  5. model     Gemini, for the ambiguous remainder only
  6. baseline  deterministic fallback when the model is off or fails
  7. evidence  deterministic retrieval, never model-chosen
"""

from __future__ import annotations

import csv
import os
from typing import Callable, Iterable

from . import baseline, evidence, rules
from .context import Dataset, MessageContext
from .rules import Decision

OUTPUT_COLUMNS = ["message_id", "action", "message_type", "reason", "confidence", "evidence_message_ids"]

# Confidence stays inside the band observed in the solved samples (0.78 to 0.91).
# Emitting 0.99 or 0.5 would be miscalibrated against the only ground truth we have.
CONF_MIN, CONF_MAX = 0.78, 0.91


def clamp_confidence(value: float) -> float:
    return round(min(CONF_MAX, max(CONF_MIN, float(value))), 2)


def route_one(
    ctx: MessageContext,
    model_fn: Callable[[MessageContext], Decision | None] | None = None,
) -> Decision:
    decision, authoritative = rules.apply(ctx)

    if not authoritative:
        if decision is None and model_fn is not None:
            try:
                decision = model_fn(ctx)
            except Exception as exc:  # noqa: BLE001 - never let one message kill the run
                print(f"  ! model failed on {ctx.message_id}: {exc}")
                decision = None

    decision = baseline.ensure(decision, ctx)
    decision.confidence = clamp_confidence(decision.confidence)
    decision.evidence_message_ids = evidence.select(ctx, decision.reason_id)
    return decision


def route_all(
    dataset: Dataset,
    rows: Iterable[dict[str, str]],
    model_fn: Callable[[MessageContext], Decision | None] | None = None,
    media_fn: Callable[[MessageContext], None] | None = None,
    progress: bool = True,
) -> list[Decision]:
    decisions: list[Decision] = []
    rows = list(rows)
    for i, row in enumerate(rows, 1):
        ctx = dataset.build(row)
        if media_fn is not None and ctx.is_media:
            media_fn(ctx)
        decision = route_one(ctx, model_fn)
        decisions.append(decision)
        if progress and (i % 10 == 0 or i == len(rows)):
            print(f"  routed {i}/{len(rows)}")
    return decisions


def write_output(decisions: list[Decision], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(OUTPUT_COLUMNS)
        for d in decisions:
            writer.writerow(
                [d.message_id, d.action, d.message_type, d.reason, f"{d.confidence:.2f}", d.evidence_message_ids]
            )
