"""Join the dataset CSVs into one feature record per message.

This is the deterministic half of the system. Everything downstream (rules, model,
evidence) reads a MessageContext and never touches the CSVs again. Keeping the join
in one place means the rules layer and the model layer are guaranteed to be looking
at exactly the same view of the world.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Any

DATASET_DIR = os.environ.get("DATASET_DIR", "dataset")


def _read(name: str) -> list[dict[str, str]]:
    path = os.path.join(DATASET_DIR, name)
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _int(value: str | None, default: int = 0) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(value.strip(), fmt)
        except ValueError:
            continue
    return None


def _in_dnd(window: str, when: datetime | None) -> bool:
    """A do-not-disturb window like '22:00-07:00' wraps past midnight."""
    if not window or when is None or "-" not in window:
        return False
    try:
        start_s, end_s = window.split("-", 1)
        sh, sm = (int(x) for x in start_s.strip().split(":"))
        eh, em = (int(x) for x in end_s.strip().split(":"))
    except ValueError:
        return False
    start, end, now = time(sh, sm), time(eh, em), when.time()
    if start <= end:
        return start <= now < end
    return now >= start or now < end


@dataclass
class HistoryItem:
    """One past message plus how this user reacted to it."""

    message_id: str
    user_id: str
    conversation_type: str
    group_id: str
    business_id: str
    sender_user_id: str
    created_at: str
    message_text: str
    media_type: str
    media_id: str
    forwarded_count: int
    opened: int = 0
    replied: int = 0
    dismissed: int = 0
    muted_after: int = 0
    reported: int = 0
    reaction_time_minutes: int = 0

    @property
    def ignored(self) -> bool:
        return self.dismissed == 1 or self.muted_after == 1 or self.opened == 0


@dataclass
class MessageContext:
    """Everything known about one incoming message, for one specific user."""

    message_id: str
    user_id: str
    conversation_type: str
    group_id: str
    business_id: str
    sender_user_id: str
    created_at: str
    message_text: str
    media_type: str
    media_id: str
    forwarded_count: int

    # receiving user
    dnd_window: str = ""
    in_dnd: bool = False
    user_opened_30d: int = 0
    user_replied_30d: int = 0
    user_dismissed_30d: int = 0
    user_reported_30d: int = 0
    user_daily_notifications: float = 0.0
    user_daily_dismiss_rate: float = 0.0

    # group side
    group_name: str = ""
    group_type: str = ""
    group_member_count: int = 0
    group_messages_30d: int = 0
    user_role_in_group: str = ""
    group_muted_by_user: bool = False
    user_group_read_30d: int = 0
    user_group_replies_30d: int = 0
    user_group_dismissed_30d: int = 0
    sender_role_in_group: str = ""

    # business side
    business_display_name: str = ""
    business_brand_name: str = ""
    business_category: str = ""
    business_verified: bool = False
    business_domain_matches: bool = True
    business_official_domain: str = ""
    business_sender_domain: str = ""
    business_account_age_days: int = 0
    business_sender_domain_age_days: int = 0
    business_reports_30d: int = 0
    has_business_relationship: bool = False
    why_user_knows_account: str = ""
    allows_promotions: bool = False
    opted_out_of_promotions: bool = False
    business_activity_180d: int = 0
    business_opened_30d: int = 0
    business_dismissed_30d: int = 0
    business_replied_30d: int = 0
    last_business_activity_at: str = ""

    # history
    history: list[HistoryItem] = field(default_factory=list)
    sender_history: list[HistoryItem] = field(default_factory=list)

    # media, filled in by the media pass
    media_description: str = ""
    media_transcript: str = ""

    @property
    def is_media(self) -> bool:
        return bool(self.media_type)

    @property
    def effective_text(self) -> str:
        """Text the router should reason over, including anything read out of media."""
        parts = [self.message_text or ""]
        if self.media_transcript:
            parts.append(f"[voice note transcript] {self.media_transcript}")
        if self.media_description:
            parts.append(f"[image content] {self.media_description}")
        return "\n".join(p for p in parts if p.strip()).strip()


class Dataset:
    """Loads every CSV once and builds MessageContext objects on demand."""

    def __init__(self, dataset_dir: str | None = None) -> None:
        global DATASET_DIR
        if dataset_dir:
            DATASET_DIR = dataset_dir

        self.users = {r["user_id"]: r for r in _read("users.csv")}
        self.groups = {r["group_id"]: r for r in _read("groups.csv")}
        self.businesses = {r["business_id"]: r for r in _read("business_accounts.csv")}

        self.group_members: dict[tuple[str, str], dict[str, str]] = {
            (r["group_id"], r["user_id"]): r for r in _read("group_members.csv")
        }
        self.user_business: dict[tuple[str, str], dict[str, str]] = {
            (r["user_id"], r["business_id"]): r for r in _read("user_business_history.csv")
        }

        self.images = {r["image_id"]: r["file_path"] for r in _read("images.csv")}
        self.voice_notes = {r["voice_note_id"]: r["file_path"] for r in _read("voice_notes.csv")}

        self._build_daily_summary()
        self._build_history()

    def _build_daily_summary(self) -> None:
        totals: dict[str, list[int]] = {}
        for row in _read("daily_notification_summary.csv"):
            uid = row["user_id"]
            sent, dismissed = _int(row["notifications_sent"]), _int(row["notifications_dismissed"])
            acc = totals.setdefault(uid, [0, 0, 0])
            acc[0] += sent
            acc[1] += dismissed
            acc[2] += 1
        self.daily_load = {
            uid: (sent / days if days else 0.0, dismissed / sent if sent else 0.0)
            for uid, (sent, dismissed, days) in totals.items()
        }

    def _build_history(self) -> None:
        events = {(r["user_id"], r["message_id"]): r for r in _read("message_events.csv")}
        self.history_by_user: dict[str, list[HistoryItem]] = {}
        self.history_by_id: dict[str, HistoryItem] = {}

        for row in _read("message_history.csv"):
            ev = events.get((row["user_id"], row["message_id"]), {})
            item = HistoryItem(
                message_id=row["message_id"],
                user_id=row["user_id"],
                conversation_type=row["conversation_type"],
                group_id=row["group_id"],
                business_id=row["business_id"],
                sender_user_id=row["sender_user_id"],
                created_at=row["created_at"],
                message_text=row["message_text"],
                media_type=row["media_type"],
                media_id=row["media_id"],
                forwarded_count=_int(row["forwarded_count"]),
                opened=_int(ev.get("message_opened")),
                replied=_int(ev.get("message_replied")),
                dismissed=_int(ev.get("notification_dismissed")),
                muted_after=_int(ev.get("muted_after_message")),
                reported=_int(ev.get("message_reported")),
                reaction_time_minutes=_int(ev.get("reaction_time_minutes")),
            )
            self.history_by_user.setdefault(item.user_id, []).append(item)
            self.history_by_id[item.message_id] = item

    def media_path(self, media_type: str, media_id: str) -> str | None:
        if not media_id:
            return None
        rel = self.images.get(media_id) if media_type == "image" else self.voice_notes.get(media_id)
        if not rel:
            return None
        # Paths in the CSVs are relative to the dataset root.
        candidate = os.path.join(DATASET_DIR, rel)
        return candidate if os.path.exists(candidate) else None

    def build(self, row: dict[str, str]) -> MessageContext:
        ctx = MessageContext(
            message_id=row["message_id"],
            user_id=row["user_id"],
            conversation_type=row["conversation_type"],
            group_id=row.get("group_id", ""),
            business_id=row.get("business_id", ""),
            sender_user_id=row.get("sender_user_id", ""),
            created_at=row.get("created_at", ""),
            message_text=row.get("message_text", "") or "",
            media_type=row.get("media_type", "") or "",
            media_id=row.get("media_id", "") or "",
            forwarded_count=_int(row.get("forwarded_count")),
        )
        when = _parse_dt(ctx.created_at)

        user = self.users.get(ctx.user_id, {})
        ctx.dnd_window = user.get("do_not_disturb_window", "")
        ctx.in_dnd = _in_dnd(ctx.dnd_window, when)
        ctx.user_opened_30d = _int(user.get("messages_opened_30d"))
        ctx.user_replied_30d = _int(user.get("messages_replied_30d"))
        ctx.user_dismissed_30d = _int(user.get("notifications_dismissed_30d"))
        ctx.user_reported_30d = _int(user.get("messages_reported_30d"))
        load, dismiss_rate = self.daily_load.get(ctx.user_id, (0.0, 0.0))
        ctx.user_daily_notifications = round(load, 2)
        ctx.user_daily_dismiss_rate = round(dismiss_rate, 3)

        if ctx.group_id:
            group = self.groups.get(ctx.group_id, {})
            ctx.group_name = group.get("group_name", "")
            ctx.group_type = group.get("group_type", "")
            ctx.group_member_count = _int(group.get("member_count"))
            ctx.group_messages_30d = _int(group.get("messages_30d"))

            membership = self.group_members.get((ctx.group_id, ctx.user_id), {})
            ctx.user_role_in_group = membership.get("role", "")
            ctx.group_muted_by_user = _int(membership.get("group_muted_by_user")) == 1
            ctx.user_group_read_30d = _int(membership.get("messages_read_30d"))
            ctx.user_group_replies_30d = _int(membership.get("replies_sent_30d"))
            ctx.user_group_dismissed_30d = _int(membership.get("notifications_dismissed_30d"))

            if ctx.sender_user_id:
                sender = self.group_members.get((ctx.group_id, ctx.sender_user_id), {})
                ctx.sender_role_in_group = sender.get("role", "")

        if ctx.business_id:
            biz = self.businesses.get(ctx.business_id, {})
            ctx.business_display_name = biz.get("display_name", "")
            ctx.business_brand_name = biz.get("brand_name", "")
            ctx.business_category = biz.get("category", "")
            ctx.business_verified = _int(biz.get("verified")) == 1
            ctx.business_official_domain = biz.get("official_domain", "")
            ctx.business_sender_domain = biz.get("domain_used_by_sender", "")
            ctx.business_domain_matches = (
                ctx.business_official_domain.strip().lower()
                == ctx.business_sender_domain.strip().lower()
            )
            ctx.business_account_age_days = _int(biz.get("account_age_days"))
            ctx.business_sender_domain_age_days = _int(biz.get("domain_used_by_sender_age_days"))
            ctx.business_reports_30d = _int(biz.get("user_reports_30d"))

            rel = self.user_business.get((ctx.user_id, ctx.business_id))
            if rel:
                ctx.has_business_relationship = True
                ctx.why_user_knows_account = rel.get("why_user_knows_account", "")
                ctx.allows_promotions = _int(rel.get("allows_promotions")) == 1
                ctx.opted_out_of_promotions = bool(rel.get("promotions_opted_out_at", "").strip())
                ctx.business_activity_180d = _int(rel.get("activity_count_180d"))
                ctx.business_opened_30d = _int(rel.get("messages_opened_30d"))
                ctx.business_dismissed_30d = _int(rel.get("messages_dismissed_30d"))
                ctx.business_replied_30d = _int(rel.get("messages_replied_30d"))
                ctx.last_business_activity_at = rel.get("last_activity_at", "")

        ctx.history = self.history_by_user.get(ctx.user_id, [])
        if ctx.sender_user_id:
            ctx.sender_history = [h for h in ctx.history if h.sender_user_id == ctx.sender_user_id]

        return ctx

    def incoming(self, filename: str = "messages.csv") -> list[dict[str, str]]:
        return _read(filename)


def to_feature_dict(ctx: MessageContext) -> dict[str, Any]:
    """Compact, model-facing view. Deliberately excludes raw message text, which is
    passed separately inside an untrusted-content boundary."""
    d = {
        "conversation_type": ctx.conversation_type,
        "forwarded_count": ctx.forwarded_count,
        "media_type": ctx.media_type or "text",
        "sent_during_user_quiet_hours": ctx.in_dnd,
        "user_daily_notification_load": ctx.user_daily_notifications,
        "user_daily_dismiss_rate": ctx.user_daily_dismiss_rate,
    }
    if ctx.group_id:
        d.update(
            group_name=ctx.group_name,
            group_type=ctx.group_type,
            group_member_count=ctx.group_member_count,
            group_messages_30d=ctx.group_messages_30d,
            user_role_in_group=ctx.user_role_in_group,
            sender_role_in_group=ctx.sender_role_in_group or "member",
            group_muted_by_user=ctx.group_muted_by_user,
            user_read_in_group_30d=ctx.user_group_read_30d,
            user_replies_in_group_30d=ctx.user_group_replies_30d,
            user_dismissed_in_group_30d=ctx.user_group_dismissed_30d,
        )
    if ctx.business_id:
        d.update(
            business_name=ctx.business_display_name,
            business_category=ctx.business_category,
            business_verified=ctx.business_verified,
            sender_domain_matches_official=ctx.business_domain_matches,
            business_account_age_days=ctx.business_account_age_days,
            business_user_reports_30d=ctx.business_reports_30d,
            user_has_relationship=ctx.has_business_relationship,
            why_user_knows_account=ctx.why_user_knows_account or "no prior relationship",
            allows_promotions=ctx.allows_promotions,
            opted_out_of_promotions=ctx.opted_out_of_promotions,
            user_opened_from_business_30d=ctx.business_opened_30d,
            user_dismissed_from_business_30d=ctx.business_dismissed_30d,
        )
    return d
