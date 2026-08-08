"""Smoke test for the Dubai hotel mock client and optional ElevenLabs TTS."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from simharness.adapters.booking_client import MockBookingClient
from simharness.adapters.voice_client import VoiceClient
from simharness.schemas import AgentRequest, AgentTurnView, Speaker


def _load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key, value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a short Dubai hotel voice/text demo.")
    parser.add_argument(
        "--voice",
        action="store_true",
        help="Also call ElevenLabs TTS for every client response (requires ELEVENLABS_API_KEY).",
    )
    parser.add_argument("--output-dir", default=".", help="Where to save mp3 files.")
    args = parser.parse_args()

    _load_env(Path(".env"))
    _load_env(Path("secrets.env"))

    client = MockBookingClient()
    voice = VoiceClient(client) if args.voice else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    user_lines = [
        "Hello, I want to book a room in Dubai for two nights.",
        "What is the price per night?",
        "Great. I'd like to make a reservation, please.",
        "Yes, please confirm the booking.",
    ]

    history: list[AgentTurnView] = []
    for i, user_text in enumerate(user_lines, start=1):
        print(f"USER: {user_text}")

        request = AgentRequest(
            episode_id="dubai-voice-0",
            turn_index=i,
            history=tuple(history),
            tools=(),
            brief="You are a helpful hotel receptionist.",
        )
        response = client.respond(request)
        print(f"AGENT: {response.text}\n")

        if voice is not None:
            out = output_dir / f"response_{i:02d}.mp3"
            try:
                path = voice.speak(response.text, out)
                print(f"[TTS] saved to {path}\n")
            except RuntimeError as exc:
                print(f"[TTS] skipped: {exc}\n")

        history.append(AgentTurnView(speaker=Speaker.USER, text=user_text))
        history.append(AgentTurnView(speaker=Speaker.AGENT, text=response.text))

    if voice is None:
        print("Run with --voice to synthesize the client responses with ElevenLabs.")


if __name__ == "__main__":
    main()
