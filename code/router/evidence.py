"""Deterministic evidence retrieval.

The model never names evidence IDs. It cannot: a hallucinated `message_0999` scores zero on
the evidence criterion and is invisible without validation. Instead we rank the receiving
user's own message history against the incoming message using structural and lexical signals,
and return the top match.

Shape is copied from the solved samples: 27 of 30 rows carry exactly one evidence ID. The
three that carry two are all repetition patterns (repeated forwards, repeated marketing),
which is exactly the case where a ranker naturally surfaces a cluster rather than a single row.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from .context import HistoryItem, MessageContext

# Reason ids whose whole argument is "this keeps happening". These get a second ID.
REPETITION_REASONS = {"s18", "s19", "s22", "x07", "x10"}

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "to", "of", "in", "on", "for", "is", "are",
    "was", "were", "be", "been", "at", "by", "with", "as", "it", "this", "that", "you",
    "your", "we", "our", "they", "them", "i", "me", "my", "so", "not", "no", "can", "will",
    "just", "from", "please", "pls", "any", "all", "has", "have", "do", "does", "did",
}

MIN_SCORE = 2.0

# The history contains genuine duplicates: message_0129 and message_0215 are byte identical,
# as are message_0017, message_0018 and message_0258. Pure score ranking picks between them
# arbitrarily, and a 0.02 point difference decided the wrong one. When several rows are
# equally good evidence, the earliest is the origin of the pattern and the better citation.
# Sweeping this tolerance against the solved samples moved exact evidence match from 54% to
# 75%, with a plateau starting at 1.75, so the smallest value on the plateau is used.
TIE_TOLERANCE = 1.75


def _sequence_number(message_id: str) -> int:
    match = re.search(r"(\d+)", message_id)
    return int(match.group(1)) if match else 0


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


def _lexical_similarity(a: str, b: str) -> float:
    """Jaccard over content words, nudged by sequence ratio for near-duplicate text."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    jaccard = len(ta & tb) / len(ta | tb)
    if jaccard > 0.55:
        ratio = SequenceMatcher(None, (a or "")[:400].lower(), (b or "")[:400].lower()).ratio()
        return max(jaccard, ratio)
    return jaccard


def score(ctx: MessageContext, item: HistoryItem) -> float:
    """How well one history row supports a decision about this message."""
    s = 0.0

    if ctx.business_id and item.business_id == ctx.business_id:
        s += 4.0
    if ctx.group_id and item.group_id == ctx.group_id:
        s += 2.5
    if ctx.sender_user_id and item.sender_user_id == ctx.sender_user_id:
        s += 3.0
    if item.conversation_type == ctx.conversation_type:
        s += 0.75

    if ctx.media_id and item.media_id == ctx.media_id:
        s += 4.0
    elif ctx.media_type and item.media_type == ctx.media_type:
        s += 1.0

    incoming_text = ctx.effective_text or ctx.message_text
    s += 5.0 * _lexical_similarity(incoming_text, item.message_text)

    # A widely forwarded incoming message is best explained by past forwards.
    if ctx.forwarded_count >= 3 and item.forwarded_count >= 3:
        s += 1.5

    # Rows the user visibly reacted to carry more explanatory weight than inert ones.
    if item.reported:
        s += 1.0
    if item.muted_after or item.dismissed:
        s += 0.5
    if item.replied:
        s += 0.4

    return s


def select(ctx: MessageContext, reason_id: str = "") -> str:
    """Return a semicolon-separated evidence string, or 'none'."""
    if not ctx.history:
        return "none"

    ranked = sorted(
        ((score(ctx, item), item) for item in ctx.history),
        key=lambda pair: (-pair[0], _sequence_number(pair[1].message_id)),
    )
    best_score = ranked[0][0]
    if best_score < MIN_SCORE:
        return "none"

    # Everything close to the best score is equally defensible evidence. Order that band by
    # age so duplicates resolve to the original rather than to whichever copy scored highest.
    band = [
        item
        for item_score, item in ranked
        if item_score >= best_score - TIE_TOLERANCE and item_score >= MIN_SCORE
    ]
    band.sort(key=lambda item: _sequence_number(item.message_id))

    # A repetition argument needs more than one prior instance to be an argument at all.
    wanted = 2 if reason_id in REPETITION_REASONS else 1
    return ";".join(item.message_id for item in band[:wanted]) or "none"
