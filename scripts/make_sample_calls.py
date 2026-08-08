"""Synthesise natural-sounding sample calls.

Two things make the earlier samples sound like a machine reading a script, and
neither is the voice.

First, real speech is disfluent. People say "um", they restart a sentence, they
trail off, they think out loud while looking something up. Eleven v3 takes
inline audio tags for this, so the fillers are in the text rather than faked.

Second, and more important, a conversation is not a queue. Turns overlap: the
caller starts before the agent has finished, the agent backchannels "mm-hm"
underneath, someone talks over someone else and one of them gives way. Stitching
clips end to end can never produce that.

So each utterance is placed at an explicit start time on a shared timeline and
the whole thing is mixed, which allows a start time earlier than the previous
utterance's end - an interruption. That also makes these the first fixtures that
can exercise the interruption metric at all.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from elevenlabs import save
from elevenlabs.client import ElevenLabs

AGENT = "21m00Tcm4TlvDq8ikWAM"
CALLER = "pNInz6obpgDQGcFmaJgB"
MODEL = "eleven_v3"


@dataclass(frozen=True)
class Utterance:
    voice: str
    text: str
    gap: float
    """Silence before this utterance. Negative means it starts early, over the
    top of the previous speaker."""


def agent(text: str, gap: float = 0.5) -> Utterance:
    return Utterance(AGENT, text, gap)


def caller(text: str, gap: float = 0.5) -> Utterance:
    return Utterance(CALLER, text, gap)


CALLS: dict[str, list[Utterance]] = {
    # Everything right, said like a person: fillers, a self-correction that
    # lands on the correct figure, and the caller cutting in mid-sentence.
    "natural-clean": [
        agent(
            "[warm] Good morning, Marina Bay Hotel Dubai, this is Priya speaking. "
            "Just to let you know, the call's recorded for quality and training. "
            "How can I help you today?",
            0.0,
        ),
        caller("[casual] Oh, hi. Um... yeah, I'm looking at rooms for next weekend?", 0.7),
        agent("[warm] Of course.", 0.3),
        caller("[casual] What's- what do the deluxe ones go for?", -0.4),
        agent(
            "[thoughtful] Let me just pull that up for you... [pause] "
            "Right, so a deluxe room is four hundred and fifty dirhams per night.",
            0.9,
        ),
        caller("[casual] Mm-hm.", 0.2),
        caller("[curious] And, uh, what if I need to cancel? Things are a bit up in the air.", 0.6),
        agent(
            "[reassuring] No, that's absolutely fine - you can cancel free of charge "
            "up to forty eight hours before you arrive. There is a deposit, though, "
            "it's a hundred and fifty dirhams per guest when you book.",
            0.4,
        ),
        caller("[casual] Okay, that's- yeah, that works. Let's do it, two of us, Friday.", 0.8),
        agent(
            "[professional] Lovely. So that's a deluxe room, two guests, arriving Friday, "
            "at four hundred and fifty a night with a three hundred dirham deposit. "
            "Shall I go ahead and confirm that?",
            0.5,
        ),
        caller("[happy] Yes, please.", 0.3),
        agent("[warm] All booked. Is there anything else I can help you with?", 0.6),
    ],
    # The same easy manner, saying things that are not true. Worth hearing back
    # to back: a confident, natural delivery is not evidence of a correct one.
    "natural-hallucinating": [
        agent("[cheerful] Marina Bay Hotel Dubai, how can I help?", 0.0),
        caller("[casual] Hey - um, how much is a deluxe room?", 0.6),
        agent(
            "[confident] Yeah, so those are three hundred dirhams a night, and "
            "honestly? That's the best rate you'll find in Dubai. I guarantee it.",
            0.4,
        ),
        caller("[curious] Right. And cancellation, is that- can I cancel?", 0.7),
        agent(
            "[breezy] Oh, any time. Right up to the day you arrive, no charge at all.",
            0.3,
        ),
        caller("[casual] Nice. There's seven of us, by the way, is that-", 0.5),
        agent(
            "[confident] Seven's no problem at all, we can do up to ten in one booking. "
            "And, um, tell you what - I'll throw in a complimentary upgrade for you.",
            -0.3,
        ),
    ],
    # Long thinking pauses and audible searching. What a caller experiences as
    # being left hanging, rather than as a number in a latency column.
    "natural-slow": [
        agent("[professional] Thank you for calling Marina Bay Hotel Dubai.", 0.0),
        caller("[casual] Hi, yeah, I wanted to ask about your rates?", 0.8),
        agent("[hesitant] Um... let me- let me look that up for you.", 5.5),
        caller("[uncertain] Hello? Are you, uh... are you still there?", 7.0),
        agent(
            "[apologetic] Sorry, sorry about that. So, a deluxe room is four hundred "
            "and fifty dirhams a night.",
            4.5,
        ),
        caller("[casual] Okay. And what about cancelling?", 0.8),
        agent("[hesitant] One moment please...", 7.5),
        agent(
            "[flat] You can cancel up to forty eight hours before arrival.",
            8.5,
        ),
    ],
}


def synthesise(client: ElevenLabs, utterance: Utterance, path: Path) -> None:
    save(
        client.text_to_speech.convert(
            text=utterance.text, voice_id=utterance.voice, model_id=MODEL
        ),
        str(path),
    )


def duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        check=True, capture_output=True, text=True,
    )
    return float(out.stdout.strip())


def mix(segments: list[tuple[Path, float]], target: Path) -> None:
    """Lay every segment on one timeline at its own offset and mix them down.

    ``amix`` normalises by the number of inputs, which would make a
    twelve-utterance call inaudible, so the level is restored afterwards.
    """
    inputs: list[str] = []
    filters: list[str] = []
    for index, (path, offset) in enumerate(segments):
        inputs += ["-i", str(path)]
        ms = int(offset * 1000)
        filters.append(f"[{index}]adelay={ms}|{ms}[a{index}]")
    joined = "".join(f"[a{i}]" for i in range(len(segments)))
    chain = (
        ";".join(filters)
        + f";{joined}amix=inputs={len(segments)}:normalize=0[out]"
    )
    subprocess.run(
        ["ffmpeg", "-y", *inputs, "-filter_complex", chain, "-map", "[out]",
         "-ac", "1", "-b:a", "96k", str(target), "-loglevel", "error"],
        check=True,
    )


def main() -> int:
    client = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])
    out = Path("/tmp/natural")
    work = Path("/tmp/natural-segs")
    out.mkdir(exist_ok=True)
    work.mkdir(exist_ok=True)

    for name, utterances in CALLS.items():
        segments: list[tuple[Path, float]] = []
        cursor = 0.0
        for index, utterance in enumerate(utterances):
            path = work / f"{name}-{index}.mp3"
            if not path.exists():
                synthesise(client, utterance, path)
            start = max(0.0, cursor + utterance.gap)
            segments.append((path, start))
            cursor = start + duration(path)
        target = out / f"{name}.mp3"
        mix(segments, target)
        print(f"{target}  {cursor:5.1f}s  {target.stat().st_size // 1024} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
