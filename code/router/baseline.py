"""Deterministic fallback classifier.

Two jobs. It is the answer for every message the rules layer did not resolve when no model
is available (`--no-model`), and it is the safety net when a model call fails or returns
something unusable. It is also the baseline the evaluator scores against, so we can always
tell how much the model is actually adding over structured signals alone.
"""

from __future__ import annotations

import re

from . import reasons
from .context import MessageContext
from .rules import (
    GREETING,
    MARKETING,
    NEGATED_URGENCY,
    URGENCY_PRESSURE,
    Decision,
    _decide,
    _dismiss_rate,
    _mentions_user,
)

TRANSACTIONAL = re.compile(
    r"\b(order|delivery|shipped|packed|dispatch|out for delivery|tracking|appointment|"
    r"booking|reservation|ticket|invoice|receipt|statement|refund|payment (received|"
    r"successful|confirmation)|prescription|claim|pickup)\b",
    re.I,
)
PAYMENT_ASK = re.compile(
    r"\b(send (the )?money|transfer|pay (the |your )?(dues|fee|bill|amount)|"
    r"pending (payment|dues)|maintenance (fee|dues)|collect (the )?amount|upi|"
    r"due (today|tomorrow|on)|reminder to pay)\b",
    re.I,
)
EVENTISH = re.compile(
    r"\b(meeting|meet at|schedule|reschedule|form|circular|notice|event|function|"
    r"celebration|practice|rehearsal|deadline|submit|register|rsvp|venue|"
    r"tomorrow|today|tonight|this (weekend|sunday|saturday)|next (week|sunday))\b",
    re.I,
)
ASKS_USER = re.compile(
    r"(\?|\bcan you\b|\bcould you\b|\bplease (call|send|confirm|check|join|share)\b|"
    r"\blet me know\b|\bneed (your )?(help|input|confirmation)\b|\bare you\b)",
    re.I,
)
URGENT_WORDS = re.compile(
    r"\b(urgent|asap|emergency|immediately|right away|escalation|outage|down|"
    r"hospital|accident|leak|blocked|cancelled|delayed|early|last[- ]minute|"
    r"heads[- ]up|quick|now)\b",
    re.I,
)
SELLING = re.compile(r"\b(selling|for sale|price|rs\.?\s*\d|₹\s*\d|pickup|brand new|barely used)\b", re.I)


def classify(ctx: MessageContext) -> Decision:
    text = ctx.effective_text
    is_admin_sender = ctx.sender_role_in_group == "admin"
    mentioned = _mentions_user(ctx)
    # "Nothing urgent" and "no rush" contain urgency keywords and mean the opposite.
    urgent = bool(URGENT_WORDS.search(text) or URGENCY_PRESSURE.search(text)) and not NEGATED_URGENCY.search(text)

    # ---- business conversations ----
    if ctx.conversation_type == "business":
        marketing = bool(MARKETING.search(text))
        transactional = bool(TRANSACTIONAL.search(text))

        if marketing and not transactional:
            if ctx.opted_out_of_promotions or not ctx.has_business_relationship:
                return _decide(ctx, "x07", "promotion", 0.82)
            if _dismiss_rate(ctx.business_opened_30d, ctx.business_dismissed_30d) >= 0.6:
                return _decide(ctx, "s18", "promotion", 0.83)
            if ctx.allows_promotions:
                return _decide(ctx, "s08", "promotion", 0.79)
            return _decide(ctx, "s13", "promotion", 0.79)

        if transactional and ctx.business_verified:
            if ctx.has_business_relationship and ctx.business_activity_180d > 0:
                if ctx.in_dnd:
                    return _decide(ctx, "x03", "business_update", 0.82)
                kind = "event" if re.search(r"appointment|booking|pickup|ticket", text, re.I) else "business_update"
                return _decide(ctx, "s05" if kind == "event" else "s04", kind, 0.86)
            return _decide(ctx, "s12", "business_update", 0.81)

        if ctx.business_verified:
            return _decide(ctx, "s16", "business_update", 0.8)
        return _decide(ctx, "x07", "spam", 0.81)

    # ---- greetings and forwards ----
    if GREETING.search(text) and not mentioned:
        if ctx.forwarded_count >= 3:
            return _decide(ctx, "s19", "greeting", 0.83)
        return _decide(ctx, "s10", "greeting", 0.82)

    if ctx.forwarded_count >= 3 and not mentioned:
        return _decide(ctx, "x04", "forward", 0.8)

    # ---- direct asks ----
    if mentioned and ASKS_USER.search(text):
        if urgent:
            return _decide(ctx, "s03" if ctx.group_type == "work" else "s07", "urgent", 0.85)
        return _decide(ctx, "s06", "personal", 0.84)

    if ctx.conversation_type == "personal":
        if not ctx.sender_history:
            return _decide(ctx, "s17", "unknown", 0.81)
        if urgent and ASKS_USER.search(text):
            return _decide(ctx, "s07", "urgent", 0.84)
        if PAYMENT_ASK.search(text):
            return _decide(ctx, "x01", "payment", 0.8)
        return _decide(ctx, "s15", "personal", 0.8)

    # ---- group conversations ----
    if ctx.group_muted_by_user and not mentioned:
        return _decide(ctx, "x09", "personal", 0.82)

    if PAYMENT_ASK.search(text):
        return _decide(ctx, "x01" if is_admin_sender else "s09", "payment", 0.81)

    if urgent and (is_admin_sender or ctx.group_type in {"work", "school", "society"}):
        if ctx.group_type == "school":
            return _decide(ctx, "s02", "event", 0.86)
        if ctx.group_type == "work":
            return _decide(ctx, "s03", "urgent", 0.85)
        return _decide(ctx, "s01", "urgent", 0.85)

    if SELLING.search(text):
        return _decide(ctx, "s13", "promotion", 0.82)

    if EVENTISH.search(text):
        return _decide(ctx, "s09", "event", 0.83)

    return _decide(ctx, "s11", "personal", 0.8)


def ensure(decision: Decision | None, ctx: MessageContext) -> Decision:
    if decision is not None:
        return decision
    fallback = classify(ctx)
    fallback.source = "baseline"
    return fallback
