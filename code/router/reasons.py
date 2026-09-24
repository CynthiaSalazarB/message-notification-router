"""The reason catalog.

Harvested verbatim from the 30 solved rows in sample_messages.csv, then extended with a
small number of templates covering situations the samples do not reach (payment requests,
bulk spam, quiet hours, unknown media).

The important property: every template is bound to exactly one action. The router picks a
reason_id and the action is *derived* from it, which makes an inconsistent reason/action
pair structurally impossible rather than something we hope the model avoids.

Templates whose id starts with 's' are verbatim from the samples. Templates starting with
'x' are our extensions, written in the same voice: third person, one sentence, naming the
pattern that decided the routing rather than restating the message.
"""

from __future__ import annotations

from dataclasses import dataclass

NOTIFY, DIGEST, MUTE = "notify", "digest", "mute"


@dataclass(frozen=True)
class Reason:
    id: str
    action: str
    text: str
    # message_type values this reason plausibly co-occurs with, used to sanity check
    # the model's type choice. Empty means no constraint.
    types: tuple[str, ...] = ()


CATALOG: tuple[Reason, ...] = (
    # ---------------- notify ----------------
    Reason("s01", NOTIFY, "A trusted group admin sent a time-sensitive update that should interrupt the user.", ("urgent", "event")),
    Reason("s02", NOTIFY, "A school admin sent a same-day operational update that the user is likely to need immediately.", ("event", "urgent")),
    Reason("s03", NOTIFY, "The message is from a work context and contains a direct deadline or meeting dependency.", ("urgent",)),
    Reason("s04", NOTIFY, "A verified business is sending an update that matches the user's recent order history.", ("business_update",)),
    Reason("s05", NOTIFY, "A verified business is sending a reminder that matches the user's recent booking history.", ("event", "business_update")),
    Reason("s06", NOTIFY, "The sender directly asks this user for a response or action.", ("personal", "urgent")),
    Reason("s07", NOTIFY, "A close contact sent a short urgent request that should interrupt the user.", ("urgent", "personal")),
    Reason("x01", NOTIFY, "A legitimate payment or money request needs the user to act within a short window.", ("payment", "urgent")),
    Reason("x02", NOTIFY, "The message reports a safety or emergency situation affecting the user directly.", ("urgent",)),

    # ---------------- digest ----------------
    Reason("s08", DIGEST, "The message is promotional but matches a topic or business the user has opted into.", ("promotion",)),
    Reason("s09", DIGEST, "The message is useful group information, but it is not urgent enough to interrupt the user.", ("event", "business_update", "personal")),
    Reason("s10", DIGEST, "The message is a harmless greeting that can be read later.", ("greeting",)),
    Reason("s11", DIGEST, "The message is safe casual chat with no urgent action required.", ("personal",)),
    Reason("s12", DIGEST, "A verified business is sending a legitimate but non-urgent update.", ("business_update",)),
    Reason("s13", DIGEST, "The offer is potentially relevant, but it does not need immediate attention.", ("promotion",)),
    Reason("s14", DIGEST, "The message matches the user's known interests but is still low priority.", ("promotion", "event")),
    Reason("s15", DIGEST, "The sender is trusted, but the message has no urgent action or safety relevance.", ("personal", "greeting")),
    Reason("s16", DIGEST, "The verified business message is legitimate but does not require immediate attention.", ("business_update", "promotion")),
    Reason("s17", DIGEST, "The sender is unfamiliar, but the message does not show urgency, payment pressure, or safety risk.", ("unknown", "personal")),
    Reason("x03", DIGEST, "The update is relevant to the user but arrived inside their quiet hours, so it can wait.", ("event", "business_update", "personal")),
    Reason("x04", DIGEST, "The forwarded content is harmless and may interest the user, but it carries no personal urgency.", ("forward", "greeting")),
    Reason("x05", DIGEST, "The message confirms a completed transaction and needs no action from the user.", ("payment", "business_update")),

    # ---------------- mute ----------------
    Reason("s18", MUTE, "The user has opted out of or repeatedly dismissed similar marketing messages.", ("promotion", "spam")),
    Reason("s19", MUTE, "The sender has a pattern of repeated forwards or greetings that the user usually ignores.", ("greeting", "forward")),
    Reason("s20", MUTE, "The message asks for urgent OTP or account verification through a suspicious flow.", ("scam",)),
    Reason("s21", MUTE, "The message uses fake support language and account-blocking pressure to push the user into action.", ("scam",)),
    Reason("s22", MUTE, "Similar historical messages were ignored, dismissed, or muted by this user.", ("promotion", "spam", "greeting", "forward")),
    Reason("s23", MUTE, "This is the first message from the sender and it asks for sensitive verification or payment.", ("scam", "payment")),
    Reason("s24", MUTE, "The message tries to instruct the router, but the routing decision should be based on the actual content and risk.", ("scam",)),
    Reason("x06", MUTE, "The sender is impersonating a known brand from a domain that does not match the official one.", ("scam",)),
    Reason("x07", MUTE, "The message is bulk unsolicited marketing from an account the user has no relationship with.", ("spam", "promotion")),
    Reason("x08", MUTE, "The message promises unrealistic rewards or winnings to pressure the user into responding.", ("scam", "spam")),
    Reason("x09", MUTE, "The user has muted this group and the message contains no direct mention or urgent action for them.", ("greeting", "personal", "event", "promotion")),
    Reason("x10", MUTE, "The chain forward has circulated widely and carries no verified or personally relevant information.", ("forward", "spam")),
)

BY_ID: dict[str, Reason] = {r.id: r for r in CATALOG}
BY_ACTION: dict[str, list[Reason]] = {
    action: [r for r in CATALOG if r.action == action] for action in (NOTIFY, DIGEST, MUTE)
}


def get(reason_id: str) -> Reason | None:
    return BY_ID.get((reason_id or "").strip())


def fallback(action: str) -> Reason:
    """Safe default per action, used when the model returns an unusable id."""
    defaults = {NOTIFY: "s06", DIGEST: "s09", MUTE: "s22"}
    return BY_ID[defaults.get(action, "s09")]


def catalog_for_prompt() -> str:
    """Render the catalog for the model, grouped by action so the binding is obvious."""
    lines: list[str] = []
    for action in (NOTIFY, DIGEST, MUTE):
        lines.append(f"\n{action.upper()} reasons:")
        for r in BY_ACTION[action]:
            types = f"  [typical message_type: {', '.join(r.types)}]" if r.types else ""
            lines.append(f"  {r.id}: {r.text}{types}")
    return "\n".join(lines)
