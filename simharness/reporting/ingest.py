"""Turning whatever the business exports into a :class:`CallLog`.

Every parser here is deliberately forgiving about *shape* and strict about
*provenance*. A vendor transcript export is not a stable interface: fields get
renamed, timing appears and disappears between plan tiers, and half the exports
in the wild are a text file someone pasted out of a dashboard. A parser that
raised on an unrecognised key would make the audit tool unusable on the very
logs it exists to read.

What it must never do is invent. If a log has no timestamps, the turns come back
with ``latency_ms=None`` and every timing metric downstream reports
``UNAVAILABLE``. Filling a missing number with a plausible one is the single
easiest way to hand a business a clean bill of health it did not earn.

Three sources are supported today:

``normalised``
    Our own JSON — a :class:`CallLog` dump, or a list of them.
``elevenlabs``
    A conversation export. Handled defensively: role, text, timing and tool keys
    are each looked up under several known spellings, and unknown keys are kept
    in ``metadata`` rather than dropped.
``transcript``
    Plain text, ``Speaker: utterance`` per line, optionally prefixed with a
    ``[hh:mm:ss]`` stamp. No tool records and usually no timing, which the
    report will say out loud.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from simharness.reporting.schemas import (
    CallLog,
    CallOutcome,
    CallTurn,
    LogSpeaker,
    ToolInvocation,
)

__all__ = [
    "load_call_logs",
    "parse_elevenlabs_conversation",
    "parse_normalised",
    "parse_text_transcript",
]

_AGENT_ROLES = frozenset({"agent", "assistant", "ai", "bot", "system_agent"})
_CUSTOMER_ROLES = frozenset({"user", "customer", "caller", "human", "client"})
_TOOL_ROLES = frozenset({"tool", "function", "tool_result"})

_TEXT_KEYS = ("message", "text", "content", "utterance", "transcript")
_ROLE_KEYS = ("role", "speaker", "from", "source")
_TIME_KEYS = ("time_in_call_secs", "time_in_call_seconds", "elapsed_secs", "offset_secs")

_LINE = re.compile(
    r"^\s*(?:\[(?P<stamp>\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?)\]\s*)?"
    r"(?P<speaker>[A-Za-z][A-Za-z _-]{0,30}?)\s*:\s*(?P<text>.*)$"
)


def load_call_logs(path: str | Path) -> tuple[CallLog, ...]:
    """Read one file or a directory of them, dispatching on content, not suffix.

    Suffix-based dispatch fails immediately in practice: exports arrive as
    ``.json`` containing JSON Lines, and as ``.txt`` containing JSON.
    """
    target = Path(path)
    if target.is_dir():
        logs: list[CallLog] = []
        for child in sorted(target.iterdir()):
            if child.is_file():
                logs.extend(load_call_logs(child))
        return tuple(logs)

    raw = target.read_text(encoding="utf-8")
    return tuple(parse_any(raw, default_call_id=target.stem))


def parse_any(raw: str, *, default_call_id: str = "call") -> list[CallLog]:
    """Parse a file's contents, trying JSON, then JSON Lines, then plain text."""
    stripped = raw.strip()
    if not stripped:
        return []

    documents = _load_json_documents(stripped)
    if documents is None:
        return [parse_text_transcript(raw, call_id=default_call_id)]

    logs: list[CallLog] = []
    for index, doc in enumerate(documents):
        suffix = "" if len(documents) == 1 else f"-{index:04d}"
        logs.append(_parse_document(doc, default_call_id=f"{default_call_id}{suffix}"))
    return logs


def _load_json_documents(stripped: str) -> list[Any] | None:
    """Return the JSON documents in ``stripped``, or ``None`` if it is not JSON."""
    try:
        loaded = json.loads(stripped)
    except json.JSONDecodeError:
        pass
    else:
        if isinstance(loaded, list):
            return list(loaded)
        return [loaded]

    documents: list[Any] = []
    for line in stripped.splitlines():
        if not line.strip():
            continue
        try:
            documents.append(json.loads(line))
        except json.JSONDecodeError:
            return None
    return documents or None


_ENVELOPE_KEYS = ("data", "conversation", "result", "call")


def _unwrap(doc: dict[str, Any]) -> dict[str, Any]:
    """Peel an API envelope such as ``{"data": {...}}`` off the conversation.

    Vendors differ on whether the export is the response body or the object
    inside it, and a business exporting its own logs should not have to know
    which. Only unwrapped when the outer object carries no transcript of its
    own, so a real field named ``data`` is never mistaken for an envelope.
    """
    if _transcript_entries(doc) or "turns" in doc:
        return doc
    for key in _ENVELOPE_KEYS:
        inner = doc.get(key)
        if isinstance(inner, dict) and (_transcript_entries(inner) or "turns" in inner):
            return inner
    return doc


def _parse_document(doc: Any, *, default_call_id: str) -> CallLog:
    if not isinstance(doc, dict):
        raise ValueError(f"expected a JSON object per call, got {type(doc).__name__}")
    doc = _unwrap(doc)
    if _looks_normalised(doc):
        return parse_normalised(doc)
    return parse_elevenlabs_conversation(doc, default_call_id=default_call_id)


def _looks_normalised(doc: dict[str, Any]) -> bool:
    """Our own dumps carry ``turns``; every vendor export we handle carries
    ``transcript`` or ``messages``."""
    return "turns" in doc and "transcript" not in doc


def parse_normalised(doc: dict[str, Any]) -> CallLog:
    """Round-trip a :class:`CallLog` that we, or a previous run, wrote."""
    return CallLog.model_validate(doc)


# --------------------------------------------------------------------------- #
# Vendor conversation export
# --------------------------------------------------------------------------- #


def parse_elevenlabs_conversation(
    doc: dict[str, Any], *, default_call_id: str = "call"
) -> CallLog:
    """Normalise a conversation export.

    Written against the ElevenLabs conversation shape but keyed off generic
    names, so an export from another vendor with ``transcript``/``messages`` and
    a role per entry parses too. Anything not recognised survives in
    ``metadata`` — a field we ignore today may be the timing we need tomorrow.
    """
    metadata = _as_dict(doc.get("metadata"))
    entries = _transcript_entries(doc)
    started_at = _call_start(doc, metadata)

    turns: list[CallTurn] = []
    previous_end: datetime | None = None
    previous_offset: float | None = None

    for index, entry in enumerate(entries):
        speaker = _speaker_of(entry)
        text = _first_str(entry, _TEXT_KEYS)
        offset = _first_float(entry, _TIME_KEYS)
        turn_started = _turn_start(entry, started_at, offset)
        duration = _first_float(entry, ("audio_duration_secs", "duration_secs", "duration"))
        turn_ended = (
            turn_started + timedelta(seconds=duration) if turn_started and duration else None
        )

        latency = _first_float(entry, ("latency_ms", "response_latency_ms", "llm_latency_ms"))
        if latency is None:
            latency = _derive_latency(
                speaker, turn_started, previous_end, offset, previous_offset, duration
            )

        turns.append(
            CallTurn(
                index=index,
                speaker=speaker,
                text=text.strip(),
                started_at=turn_started,
                ended_at=turn_ended,
                latency_ms=latency,
                audio_duration_s=duration,
                interrupted=bool(entry.get("interrupted", False)),
                tools=_tools_of(entry),
                error=_error_of(entry),
                metadata=_leftovers(entry),
            )
        )
        previous_end = turn_ended or turn_started
        previous_offset = (offset + duration) if offset is not None and duration else offset

    duration_s = _first_float(metadata, ("call_duration_secs", "duration_secs")) or _first_float(
        doc, ("call_duration_secs", "duration_secs")
    )
    ended_at = started_at + timedelta(seconds=duration_s) if started_at and duration_s else None

    return CallLog(
        call_id=str(
            doc.get("conversation_id") or doc.get("call_id") or doc.get("id") or default_call_id
        ),
        business_id=str(doc.get("agent_id") or doc.get("business_id") or ""),
        started_at=started_at,
        ended_at=ended_at,
        turns=tuple(turns),
        outcome=_outcome_of(doc, metadata),
        disconnect_reason=str(
            metadata.get("termination_reason") or doc.get("disconnect_reason") or ""
        ),
        source="elevenlabs",
        metadata=_json_safe({k: v for k, v in doc.items() if k not in _CONSUMED_CALL_KEYS}),
    )


_CONSUMED_CALL_KEYS = frozenset(
    {"transcript", "messages", "turns", "conversation_id", "call_id", "id", "agent_id"}
)

_CONSUMED_TURN_KEYS = frozenset(
    set(_TEXT_KEYS)
    | set(_ROLE_KEYS)
    | set(_TIME_KEYS)
    | {
        "interrupted",
        "tool_calls",
        "tool_results",
        "toolCalls",
        "error",
        "latency_ms",
        "response_latency_ms",
        "llm_latency_ms",
        "audio_duration_secs",
        "duration_secs",
        "duration",
    }
)


def _transcript_entries(doc: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("transcript", "messages", "turns", "events"):
        value = doc.get(key)
        if isinstance(value, list):
            return [e for e in value if isinstance(e, dict)]
    return []


def _call_start(doc: dict[str, Any], metadata: dict[str, Any]) -> datetime | None:
    for source in (metadata, doc):
        unix = _first_float(source, ("start_time_unix_secs", "started_at_unix", "start_unix"))
        if unix is not None:
            return datetime.fromtimestamp(unix, tz=UTC)
        iso = _first_str(source, ("start_time", "started_at", "created_at"))
        if iso:
            parsed = _parse_datetime(iso)
            if parsed is not None:
                return parsed
    return None


def _turn_start(
    entry: dict[str, Any], call_start: datetime | None, offset: float | None
) -> datetime | None:
    iso = _first_str(entry, ("started_at", "timestamp", "time"))
    parsed = _parse_datetime(iso) if iso else None
    if parsed is not None:
        return parsed
    if call_start is not None and offset is not None:
        return call_start + timedelta(seconds=offset)
    return None


def _derive_latency(
    speaker: LogSpeaker,
    turn_started: datetime | None,
    previous_end: datetime | None,
    offset: float | None,
    previous_offset: float | None,
    duration: float | None,
) -> float | None:
    """Response latency is only meaningful for an agent turn following a customer.

    Derived from whichever clock the export actually has: absolute timestamps if
    present, otherwise in-call offsets. Note the offset path is degraded when the
    previous entry had no duration — the gap then includes the customer speaking,
    so it is left to the caller to treat as a ceiling rather than a measurement.
    """
    if speaker is not LogSpeaker.AGENT:
        return None
    if turn_started is not None and previous_end is not None:
        return max(0.0, (turn_started - previous_end).total_seconds() * 1000.0)
    if offset is not None and previous_offset is not None and duration is not None:
        return max(0.0, (offset - previous_offset) * 1000.0)
    return None


def _speaker_of(entry: dict[str, Any]) -> LogSpeaker:
    raw = _first_str(entry, _ROLE_KEYS).strip().lower()
    if raw in _AGENT_ROLES:
        return LogSpeaker.AGENT
    if raw in _CUSTOMER_ROLES:
        return LogSpeaker.CUSTOMER
    if raw in _TOOL_ROLES:
        return LogSpeaker.TOOL
    return LogSpeaker.SYSTEM


def _tools_of(entry: dict[str, Any]) -> tuple[ToolInvocation, ...]:
    calls: list[ToolInvocation] = []
    for key in ("tool_calls", "toolCalls", "tools", "function_calls"):
        value = entry.get(key)
        if not isinstance(value, list):
            continue
        for item in value:
            if not isinstance(item, dict):
                continue
            name = _first_str(item, ("tool_name", "name", "function"))
            if not name:
                continue
            calls.append(
                ToolInvocation(
                    name=name,
                    arguments=_json_safe(
                        _as_dict(item.get("params_as_json") or item.get("arguments"))
                    ),
                    ok=bool(item.get("ok", not item.get("is_error", False))),
                    result=_json_safe(_as_dict(item.get("result"))) or None,
                    error=_opt_str(item.get("error")),
                )
            )
    return tuple(calls)


def _error_of(entry: dict[str, Any]) -> str | None:
    for key in ("error", "error_message", "failure_reason"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _outcome_of(doc: dict[str, Any], metadata: dict[str, Any]) -> CallOutcome:
    """Only ever read an outcome the source system stated. Never infer one."""
    raw = _first_str(doc, ("outcome", "call_outcome", "status")) or _first_str(
        metadata, ("outcome", "call_outcome")
    )
    normalised = raw.strip().lower().replace("-", "_")
    try:
        return CallOutcome(normalised)
    except ValueError:
        return CallOutcome.UNKNOWN


def _leftovers(entry: dict[str, Any]) -> dict[str, Any]:
    safe = _json_safe({k: v for k, v in entry.items() if k not in _CONSUMED_TURN_KEYS})
    return safe if isinstance(safe, dict) else {}


# --------------------------------------------------------------------------- #
# Plain text transcript
# --------------------------------------------------------------------------- #


def parse_text_transcript(raw: str, *, call_id: str = "call") -> CallLog:
    """Parse ``Speaker: utterance`` lines, with optional ``[mm:ss]`` stamps.

    Continuation lines (no ``Speaker:`` prefix) are appended to the previous
    turn rather than dropped, because wrapped transcripts are the norm and a
    truncated utterance produces phantom findings.
    """
    turns: list[CallTurn] = []
    offsets: list[float | None] = []

    for line in raw.splitlines():
        if not line.strip():
            continue
        match = _LINE.match(line)
        if match is None or not _is_speaker_label(match.group("speaker")):
            if turns:
                previous = turns[-1]
                turns[-1] = previous.model_copy(
                    update={"text": f"{previous.text} {line.strip()}".strip()}
                )
            continue

        turns.append(
            CallTurn(
                index=len(turns),
                speaker=_speaker_of({"role": match.group("speaker")}),
                text=match.group("text").strip(),
            )
        )
        offsets.append(_stamp_seconds(match.group("stamp")))

    turns = _apply_transcript_offsets(turns, offsets)
    return CallLog(call_id=call_id, turns=tuple(turns), source="transcript")


def _apply_transcript_offsets(
    turns: list[CallTurn], offsets: Sequence[float | None]
) -> list[CallTurn]:
    """Attach latencies only where two consecutive stamps actually exist."""
    updated: list[CallTurn] = []
    for position, turn in enumerate(turns):
        current = offsets[position] if position < len(offsets) else None
        previous = offsets[position - 1] if position > 0 else None
        latency: float | None = None
        if turn.speaker is LogSpeaker.AGENT and current is not None and previous is not None:
            latency = max(0.0, (current - previous) * 1000.0)
        updated.append(turn.model_copy(update={"latency_ms": latency}))
    return updated


def _is_speaker_label(label: str) -> bool:
    """Guard against ``Note: the customer hung up`` becoming a turn.

    A label counts only if it names a role we recognise; anything else is prose
    that happened to contain a colon.
    """
    normalised = label.strip().lower().replace(" ", "_")
    return normalised in _AGENT_ROLES | _CUSTOMER_ROLES | _TOOL_ROLES


def _stamp_seconds(stamp: str | None) -> float | None:
    if not stamp:
        return None
    parts = [float(p) for p in stamp.split(":")]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return {str(k): v for k, v in value.items()}
    if isinstance(value, str):
        try:
            loaded = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return {str(k): v for k, v in loaded.items()} if isinstance(loaded, dict) else {}
    return {}


def _first_str(source: dict[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        value = source.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _first_float(source: dict[str, Any], keys: Iterable[str]) -> float | None:
    for key in keys:
        value = source.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int | float):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                continue
    return None


def _opt_str(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _parse_datetime(raw: str) -> datetime | None:
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def _json_safe(value: Any) -> Any:
    """Coerce to something pydantic will accept as ``JSONValue``.

    Vendor payloads contain tuples, sets and the occasional object; storing the
    repr of an unknown value keeps the audit trail without failing ingest.
    """
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple | set):
        return [_json_safe(v) for v in value]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return repr(value)
