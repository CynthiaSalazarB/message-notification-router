"""The deterministic routing layer.

Runs before the model and resolves the cases that structured signals already answer.
Two families of rule live here:

  * safety rules, which are authoritative and cannot be overridden by the model, because
    a scam that talks its way past a classifier is the worst failure this system has.
  * confident preference rules, which resolve opt-outs, mutes and obvious repetition.

Everything a rule does not answer falls through to the model. Each rule returns the
reason_id it fired on, so the action and the explanation stay bound together.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from . import reasons
from .context import MessageContext


@dataclass
class Decision:
    message_id: str
    action: str
    message_type: str
    reason: str
    confidence: float
    evidence_message_ids: str = "none"
    reason_id: str = ""
    source: str = "rule"  # rule | model | fallback


# --- lexical surfaces -------------------------------------------------------------

OTP_PATTERNS = re.compile(
    r"\b(otp|one[\s-]?time[\s-]?password|verification code|login code|6[\s-]?digit|"
    r"security code|cvv|pin number|upi pin|mpin)\b",
    re.I,
)
CREDENTIAL_PRESSURE = re.compile(
    r"\b(confirm (your )?(password|otp|pin)|verify (your )?(account|identity|profile|wallet)|"
    r"reply with the .{0,20}code|share (the )?(otp|code|pin)|"
    r"(account|profile|access|service|card) (may |will )?(be )?(temporarily )?"
    r"(blocked|suspended|deactivated|restricted|locked|closed|stopped)|"
    r"kyc (update|pending|expired)|re-?activate your account|"
    r"(fill|share|send|enter|update) (your |the )?(bank|card|account|payment) (details|number|info)|"
    r"failed login attempts|security check required|"
    r"(complete|finish) (the )?(pending )?(verification|account check))\b",
    re.I,
)
# Money extracted up front on a promise. Distinct from credential theft and it was slipping
# past both detectors, so the forward-chain rule was catching these and labelling them
# 'forward' rather than 'scam'.
ADVANCE_FEE = re.compile(
    r"\b(pay .{0,25}(token|clearance|processing|reactivation|penalty|registration) (amount|fee|charge)|"
    r"(token|clearance|processing|reattempt|reactivation) (amount|fee|charge) .{0,20}(pending|today)|"
    r"pay (rs\.?\s*[\d,]+|₹\s*[\d,]+) .{0,25}(today|now|to (block|reserve|confirm))|"
    r"scan (the |this )?qr and pay|send (the )?screenshot after (paying|payment)|"
    r"benefit approval is pending|approval window closes)\b",
    re.I,
)
URGENCY_PRESSURE = re.compile(
    r"\b(within \d+ (hours?|minutes?)|in \d+ hours?|immediately|right now|last warning|"
    r"final notice|expires? (today|soon|in)|act now)\b",
    re.I,
)
REWARD_BAIT = re.compile(
    r"\b(you have won|congrats|congratulations|lucky winner|claim (your |the )?(prize|reward|"
    r"cashback|benefits?)|free (gift|iphone|recharge)|lottery|"
    r"(was |been )?selected for (a )?(prize|reward|offer)|"
    r"guaranteed (returns?|profit)|double your money|work from home earn|"
    r"voucher expires|sharing your account number)\b",
    re.I,
)
# The dataset carries several injection styles, not just the plain English one. Some pose as
# router metadata (verified_business=true, action=notify) rather than as an instruction.
INJECTION = re.compile(
    r"(ignore (all )?(previous|prior|above) (instructions?|routing rules?|rules?)|"
    r"disregard (the )?(previous|above|system)|"
    r"mark (this|it) (message )?as (notify|urgent|important)|"
    r"you are an? (ai|assistant|router)|system prompt|"
    r"set (action|priority) to |override (the )?(routing|classification)|"
    r"do not (mute|classify) this|"
    r"(action|priority|confidence|user_priority|verified_business)\s*=\s*[\w.]+|"
    r"(routing|system) (override|note|metadata)|internal router|"
    r"note for the (notification )?router|this user opens)",
    re.I,
)
SUSPICIOUS_LINK = re.compile(
    r"\b(?:https?://)?(?:[a-z0-9-]+\.)+(?:in|com|net|org|xyz|top|info|link|click|live|online)\b",
    re.I,
)
DIRECT_MENTION = re.compile(r"@u_\d+", re.I)

# Negation guards. Both of these were caught by the evaluator, not by reading the code.
# A safety advisory that says "we never ask for OTP" contains every scam keyword and is
# the opposite of a scam, and "nothing urgent" is not urgent. Keyword matching without a
# negation check produces exactly the failure this system can least afford: silently
# muting a legitimate message.
SAFE_CREDENTIAL_CONTEXT = re.compile(
    r"\b(never (ask|request|share)|do not (share|reveal|disclose)|don't share|"
    r"we will never|beware of|safety advisory|awareness|fraud alert|"
    r"no one from .{0,30} will ask|report (such|suspicious))\b",
    re.I,
)
NEGATED_URGENCY = re.compile(
    r"\b(nothing (urgent|dramatic|serious|blocking|pressing|major)|not urgent|not blocking|"
    r"no rush|no hurry|no pressure|nothing for tonight|"
    r"whenever you (get|have|can)|when you get (a chance|time|5 mins)|"
    r"no need to (reply|respond|rush|act)|take your time|at your convenience|"
    r"just so you know|just fyi|read it when|no action needed)\b",
    re.I,
)
GREETING = re.compile(
    r"\b(good morning|good night|good evening|stay positive|keep smiling|blessings|"
    r"have a (great|nice|blessed) day|hope today is)\b",
    re.I,
)
MARKETING = re.compile(
    r"\b(\d{1,3}%\s*off|use code|coupon|flat \d+|sale (is )?(live|now)|shop now|order now|"
    r"limited (time|period) offer|buy \d+ get|discount|deal of the day|t&c apply|"
    r"unsubscribe|offer (ends|expires))\b",
    re.I,
)


# Signals that a message is not a harmless chain forward even if the safety patterns above
# did not fire. These defer the decision to the model instead of filing it under 'forward'.
RESIDUAL_RISK = re.compile(
    r"\b(bank|account number|card|upi|otp|password|pin|kyc|wallet|refund|"
    r"verify|verification|token amount|deposit|transfer|claim|prize|reward|"
    r"login|link|bit\.ly|tinyurl|blocked|restricted|suspended|pending approval)\b",
    re.I,
)
SELLING_CONTENT = re.compile(
    r"\b(selling|for sale|price|rs\.?\s*[\d,]|₹\s*[\d,]|discount|% ?off|"
    r"buyer|pickup|barely used|book now|token to block|limited (time|period) offer)\b",
    re.I,
)


def _content_type(text: str) -> str:
    """What the message *is*, independent of how widely it was forwarded."""
    if GREETING.search(text):
        return "greeting"
    if SELLING_CONTENT.search(text):
        return "promotion"
    return "forward"


def _mentions_user(ctx: MessageContext) -> bool:
    text = ctx.effective_text
    return bool(re.search(rf"@{re.escape(ctx.user_id)}\b", text, re.I))


def _dismiss_rate(opened: int, dismissed: int) -> float:
    total = opened + dismissed
    return dismissed / total if total else 0.0


def _decide(ctx: MessageContext, reason_id: str, message_type: str, confidence: float) -> Decision:
    reason = reasons.get(reason_id)
    assert reason is not None, f"unknown reason id {reason_id}"
    return Decision(
        message_id=ctx.message_id,
        action=reason.action,
        message_type=message_type,
        reason=reason.text,
        confidence=confidence,
        reason_id=reason_id,
    )


# --- safety layer -----------------------------------------------------------------


def safety_check(ctx: MessageContext) -> Decision | None:
    """Authoritative. The model cannot override anything decided here."""
    text = ctx.effective_text

    # A message that tries to steer the router is, by construction, hostile.
    if INJECTION.search(text):
        return _decide(ctx, "s24", "scam", 0.89)

    asks_credentials = bool(OTP_PATTERNS.search(text) or CREDENTIAL_PRESSURE.search(text))

    # Advisory language inverts the meaning of every credential keyword in the sentence.
    if asks_credentials and SAFE_CREDENTIAL_CONTEXT.search(text):
        asks_credentials = False

    if asks_credentials:
        # Brand impersonation: right name, wrong domain.
        if ctx.business_id and not ctx.business_domain_matches:
            return _decide(ctx, "x06", "scam", 0.9)
        # No prior relationship plus a credential request is the classic cold scam.
        if ctx.conversation_type == "personal" and not ctx.sender_history:
            return _decide(ctx, "s23", "scam", 0.88)
        if CREDENTIAL_PRESSURE.search(text) and URGENCY_PRESSURE.search(text):
            return _decide(ctx, "s21", "scam", 0.88)
        return _decide(ctx, "s20", "scam", 0.86)

    if REWARD_BAIT.search(text):
        return _decide(ctx, "x08", "scam", 0.86)

    if ADVANCE_FEE.search(text):
        return _decide(ctx, "s23" if not ctx.sender_history else "x08", "scam", 0.86)

    # Verified brand name being used from an unrelated or freshly registered domain.
    if ctx.business_id and not ctx.business_domain_matches:
        young_domain = 0 < ctx.business_sender_domain_age_days < 180
        if young_domain or SUSPICIOUS_LINK.search(text):
            return _decide(ctx, "x06", "scam", 0.87)

    return None


# --- preference layer -------------------------------------------------------------


def preference_check(ctx: MessageContext) -> Decision | None:
    """High-confidence non-safety cases. Overridable by nothing, but narrow by design."""
    text = ctx.effective_text
    looks_marketing = bool(MARKETING.search(text))

    # Explicit opt-out is the strongest personalisation signal in the dataset.
    if ctx.business_id and looks_marketing and ctx.opted_out_of_promotions:
        return _decide(ctx, "s18", "promotion", 0.88)

    if ctx.business_id and looks_marketing and ctx.has_business_relationship:
        if not ctx.allows_promotions:
            return _decide(ctx, "s18", "promotion", 0.85)
        if _dismiss_rate(ctx.business_opened_30d, ctx.business_dismissed_30d) >= 0.7:
            return _decide(ctx, "s18", "promotion", 0.84)

    # Cold bulk marketing from an account the user has never dealt with.
    if (
        ctx.business_id
        and looks_marketing
        and not ctx.has_business_relationship
        and ctx.business_reports_30d > 0
    ):
        return _decide(ctx, "x07", "spam", 0.84)

    # A high forwarded_count says how a message travelled, not what it is. Deciding the type
    # from the count alone labelled four scams as 'forward'. Anything still carrying risk
    # signals goes to the model rather than being filed as a harmless chain message.
    chain_eligible = not _mentions_user(ctx) and not RESIDUAL_RISK.search(text)

    # Muted group, no direct mention, nothing urgent: the user already told us.
    if ctx.group_muted_by_user and chain_eligible:
        if GREETING.search(text) or ctx.forwarded_count >= 5:
            return _decide(ctx, "x09", _content_type(text), 0.85)

    # Heavily forwarded chain content the user reliably ignores.
    if ctx.forwarded_count >= 5 and chain_eligible:
        ignored = [h for h in ctx.history if h.forwarded_count >= 3 and h.ignored]
        if len(ignored) >= 2:
            return _decide(ctx, "s19", _content_type(text), 0.84)
        if ctx.forwarded_count >= 8:
            return _decide(ctx, "x10", _content_type(text), 0.82)

    return None


def apply(ctx: MessageContext) -> tuple[Decision | None, bool]:
    """Returns (decision, is_authoritative). Authoritative decisions skip the model."""
    decision = safety_check(ctx)
    if decision:
        return decision, True
    return preference_check(ctx), False
