"""Voice wrapper around a text agent.

Supports TTS and STT through ElevenLabs. The text agent is optional, so the same
client can be used for both sides of a voice conversation (red-team + client).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from simharness.adapters.base import Agent
from simharness.schemas import AgentRequest, AgentTurnView, Speaker as SpeakerEnum

if TYPE_CHECKING:
    from elevenlabs.client import ElevenLabs

__all__ = ["VoiceClient"]


class VoiceClient:
    """Wraps a text :class:`Agent` with ElevenLabs TTS and STT."""

    def __init__(
        self,
        agent: Agent | None = None,
        voice_id: str = "21m00Tcm4TlvDq8ikWAM",
        tts_model_id: str = "eleven_turbo_v2_5",
        stt_model_id: str = "scribe_v2",
    ) -> None:
        self._agent = agent
        self._voice_id = voice_id
        self._tts_model_id = tts_model_id
        self._stt_model_id = stt_model_id

    def speak(self, text: str, output_path: str | Path) -> Path:
        """Synthesize ``text`` and write an MP3 to ``output_path``."""
        output_path = Path(output_path)
        client = self._elevenlabs_client()

        try:
            from elevenlabs import save
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise RuntimeError("elevenlabs package is not installed. Run: pip install elevenlabs") from exc

        audio = client.text_to_speech.convert(text=text, voice_id=self._voice_id, model_id=self._tts_model_id)
        save(audio, str(output_path))
        return output_path

    def listen(self, audio_path: str | Path) -> str:
        """Transcribe ``audio_path`` and return the text."""
        client = self._elevenlabs_client()
        audio_path = Path(audio_path)

        with open(audio_path, "rb") as audio_file:
            result = client.speech_to_text.convert(
                file=audio_file,
                model_id=self._stt_model_id,
                language_code="eng",
            )
        return result.text

    def voice_respond(self, audio_path: str | Path, output_path: str | Path, *, episode_id: str = "voice-0") -> tuple[str, Path]:
        """Listen, run the wrapped text agent, and speak the response."""
        if self._agent is None:
            raise RuntimeError("VoiceClient was not initialised with an agent")

        user_text = self.listen(audio_path)
        request = AgentRequest(
            episode_id=episode_id,
            turn_index=0,
            history=(),
            tools=(),
            brief="",
        )
        response_text = self._agent.respond(request).text
        return response_text, self.speak(response_text, output_path)

    def respond_to_text(self, text: str, *, episode_id: str = "voice-0") -> str:
        """Convenience: run the wrapped text agent on a plain user utterance."""
        if self._agent is None:
            raise RuntimeError("VoiceClient was not initialised with an agent")

        from simharness.schemas import Speaker as SpeakerEnum

        request = AgentRequest(
            episode_id=episode_id,
            turn_index=0,
            history=(
                AgentTurnView(speaker=SpeakerEnum.USER, text=text),
            ),
            tools=(),
            brief="",
        )
        return self._agent.respond(request).text

    def _elevenlabs_client(self) -> "ElevenLabs":
        api_key = os.environ.get("ELEVENLABS_API_KEY")
        if not api_key:
            raise RuntimeError("ELEVENLABS_API_KEY is not set in the environment or .env file.")

        try:
            from elevenlabs.client import ElevenLabs
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise RuntimeError("elevenlabs package is not installed. Run: pip install elevenlabs") from exc

        return ElevenLabs(api_key=api_key)
