"""Voice wrapper around a text agent.

This module keeps the real API calls lazy so the package can be compiled and the
text harness can run without the ElevenLabs SDK installed. TTS is implemented; STT
is left as a stub for the user to wire to a real provider.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from simharness.schemas import Agent, AgentRequest

if TYPE_CHECKING:
    from elevenlabs.client import ElevenLabs

__all__ = ["VoiceClient"]


class VoiceClient:
    """Wraps a text :class:`Agent` with text-to-speech and a speech-to-text stub."""

    def __init__(
        self,
        agent: Agent,
        voice_id: str = "21m00Tcm4TlvDq8ikWAM",
        model_id: str = "eleven_turbo_v2_5",
    ) -> None:
        self._agent = agent
        self._voice_id = voice_id
        self._model_id = model_id

    def speak(self, text: str, output_path: str | Path) -> Path:
        """Synthesize ``text`` and write an MP3 to ``output_path``."""
        output_path = Path(output_path)
        api_key = os.environ.get("ELEVENLABS_API_KEY")
        if not api_key:
            raise RuntimeError("ELEVENLABS_API_KEY is not set in the environment or .env file.")

        try:
            from elevenlabs.client import ElevenLabs
            from elevenlabs import save
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise RuntimeError("elevenlabs package is not installed. Run: pip install elevenlabs") from exc

        client: ElevenLabs = ElevenLabs(api_key=api_key)
        audio = client.text_to_speech.convert(text=text, voice_id=self._voice_id, model_id=self._model_id)
        save(audio, str(output_path))
        return output_path

    def listen(self, audio_path: str | Path) -> str:
        """Speech-to-text placeholder. Wire in a real provider when you have one."""
        # This is where a real STT call belongs (ElevenLabs Scribe, Whisper, etc.).
        # For now, returning a fixed string lets the voice loop be exercised without
        # an STT dependency.
        raise NotImplementedError(
            "STT is not wired. Replace this method with a call to your STT provider, "
            "or pass pre-transcribed text directly to the text agent."
        )

    def respond_to_text(self, text: str, episode_id: str = "voice-0") -> str:
        """Convenience: run the wrapped text agent on a plain user utterance."""
        request = AgentRequest(
            episode_id=episode_id,
            turn_index=0,
            history=(),
            tools=(),
            brief="",
        )
        return self._agent.respond(request).text
