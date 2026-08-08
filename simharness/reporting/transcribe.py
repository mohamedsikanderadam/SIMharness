"""Turning call recordings into auditable transcripts.

The third input the audit was always meant to accept. A business with no
transcript export still has the recordings, and until now those could not be
graded at all.

Recordings are, in one respect, the *best* input: word-level timings come from
the audio itself rather than from whatever a vendor chose to log, so response
latency and dead air are measured rather than reported unavailable.

Two honest limits, both surfaced rather than papered over:

- Diarisation labels speakers ``speaker_0``/``speaker_1``; it does not know
  which one works for the business. :func:`call_log_from_words` assumes whoever
  speaks first does — the business answers the phone — and
  ``agent_speaker`` overrides it when that is wrong.
- Transcription is lossy. A misheard price is indistinguishable from a
  misquoted one, so :attr:`CallLog.source` records that this came from STT and
  the report says findings on a transcribed call should be confirmed against
  the audio.

Nothing here is imported by :mod:`.ingest`: that path stays offline and
deterministic, and transcription is an explicit, billable step.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

from simharness.reporting.schemas import CallLog, CallTurn, LogSpeaker

__all__ = [
    "AUDIO_SUFFIXES",
    "ElevenLabsTranscriber",
    "TranscribedWord",
    "Transcriber",
    "call_log_from_words",
    "transcribe_recordings",
]

#: What we will attempt. Anything else is left for the caller to convert.
AUDIO_SUFFIXES = frozenset({".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm", ".mp4"})


class TranscribedWord:
    """One word with its speaker and its position in the audio.

    A tiny class rather than the vendor's object so the rest of the module never
    imports an SDK and a test can construct one in a line.
    """

    __slots__ = ("end", "speaker", "start", "text")

    def __init__(self, text: str, start: float, end: float, speaker: str) -> None:
        self.text = text
        self.start = start
        self.end = end
        self.speaker = speaker


class Transcriber(Protocol):
    """Anything that can turn an audio file into diarised, timed words."""

    def transcribe(self, path: Path) -> tuple[TranscribedWord, ...]: ...


def transcribe_recordings(
    path: str | Path, transcriber: Transcriber, *, agent_speaker: str = ""
) -> tuple[CallLog, ...]:
    """Transcribe one recording, or every recording in a directory.

    Files are processed in name order so a batch is reproducible, and the file
    stem becomes the call ID — which is what a business will recognise, their
    recordings being named after their calls.
    """
    root = Path(path)
    files = (
        sorted(p for p in root.iterdir() if p.suffix.lower() in AUDIO_SUFFIXES)
        if root.is_dir()
        else [root]
    )
    return tuple(
        call_log_from_words(
            transcriber.transcribe(file),
            call_id=file.stem,
            agent_speaker=agent_speaker,
            started_at=datetime.fromtimestamp(file.stat().st_mtime, UTC),
        )
        for file in files
    )


def call_log_from_words(
    words: tuple[TranscribedWord, ...],
    *,
    call_id: str,
    agent_speaker: str = "",
    started_at: datetime | None = None,
) -> CallLog:
    """Group diarised words into turns, with latency taken from the silence.

    A turn ends when the speaker changes. The gap between one turn ending and
    the next beginning is exactly what the reliability metrics want, and unlike
    a vendor's ``latency_ms`` field it includes the time the caller spent
    listening to nothing.

    ``started_at`` anchors the audio offsets to a wall clock. Every timing metric
    is a *difference* between two turns, so dead air and latency are exact
    whatever anchor is used - but an anchor is required before they can be
    computed at all, and the file's own timestamp is a better guess than none.
    """
    spoken = [w for w in words if w.text.strip()]
    if not spoken:
        return CallLog(call_id=call_id, source="stt")

    anchor = started_at or datetime.fromtimestamp(0, UTC)

    agent = agent_speaker or spoken[0].speaker

    groups: list[list[TranscribedWord]] = []
    for word in spoken:
        if groups and groups[-1][-1].speaker == word.speaker:
            groups[-1].append(word)
        else:
            groups.append([word])

    turns: list[CallTurn] = []
    for index, group in enumerate(groups):
        previous_end = groups[index - 1][-1].end if index else None
        start = group[0].start
        turns.append(
            CallTurn(
                index=index,
                speaker=LogSpeaker.AGENT if group[0].speaker == agent else LogSpeaker.CUSTOMER,
                text=" ".join(w.text.strip() for w in group).strip(),
                started_at=anchor + timedelta(seconds=start),
                ended_at=anchor + timedelta(seconds=group[-1].end),
                latency_ms=(
                    None if previous_end is None else round((start - previous_end) * 1000, 1)
                ),
                audio_duration_s=round(group[-1].end - start, 3),
                metadata={"diarised_speaker": group[0].speaker},
            )
        )

    return CallLog(
        call_id=call_id,
        started_at=turns[0].started_at,
        ended_at=turns[-1].ended_at,
        turns=tuple(turns),
        source="stt",
        metadata={
            "transcribed": True,
            "note": "Speech recognised from audio. Confirm any finding against the recording.",
        },
    )


class ElevenLabsTranscriber:
    """Diarised transcription via ElevenLabs Scribe.

    The import is local and the client lazy, so importing the reporting package
    needs neither the SDK nor a key.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model_id: str = "scribe_v1",
        language_code: str | None = None,
    ) -> None:
        self._api_key = api_key
        self._model_id = model_id
        self._language_code = language_code
        self._client: Any = None

    def transcribe(self, path: Path) -> tuple[TranscribedWord, ...]:
        client = self._ensure_client()
        with path.open("rb") as audio:
            result = client.speech_to_text.convert(
                file=audio,
                model_id=self._model_id,
                diarize=True,
                timestamps_granularity="word",
                **({"language_code": self._language_code} if self._language_code else {}),
            )
        return tuple(
            TranscribedWord(
                text=word.text,
                start=float(word.start),
                end=float(word.end),
                speaker=word.speaker_id or "speaker_0",
            )
            for word in result.words
            if word.type == "word"
        )

    def _ensure_client(self) -> Any:
        if self._client is None:
            import os

            from elevenlabs.client import ElevenLabs

            key = self._api_key or os.environ.get("ELEVENLABS_API_KEY")
            if not key:
                raise RuntimeError("ELEVENLABS_API_KEY is not set")
            self._client = ElevenLabs(api_key=key)
        return self._client
