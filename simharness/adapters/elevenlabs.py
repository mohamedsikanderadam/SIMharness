"""ElevenLabs Agents, over the text-only WebSocket.

**Not an HTTP dialect.** `HTTPAgent` is request/response; this is a connection
held open for the length of a call, which is a different shape and so a sibling
adapter rather than a `build`/`parse` pair. The `Agent` protocol is one method,
which is why that costs nothing anywhere else — the runner cannot tell.

Three facts from the API docs that make this viable as an eval channel:

- `conversation_config_override.conversation.text_only` disables audio
  processing entirely. The vendor's own docs give the reason to use it as
  avoiding audio pricing; for us it also means no STT/TTS in the loop, so the
  measurement is of *their agent* and not of a speech stack we introduced.
- Text conversations draw on a concurrency pool separate from voice, so a sweep
  does not eat the limits a live deployment is relying on.
- The noise wrapper still applies. We corrupt the text before it is sent, which
  models the ASR the real caller would be speaking through — a deliberate
  substitution, and one worth stating: it is *our* error model, not theirs.

Wire shape:
    connect  wss://api.elevenlabs.io/v1/convai/conversation?agent_id=...
    send     {"type": "user_message", "text": "..."}
    receive  {"type": "agent_response",
              "agent_response_event": {"agent_response": "..."}}

Everything else on the socket — audio chunks, interim transcripts, VAD, ping —
is skipped rather than treated as an error, because a chatty protocol is normal
and an adapter that falls over on an unrecognised frame is useless.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from typing import Any

from simharness.schemas import AgentRequest, AgentResponse, JSONObject, Speaker

DEFAULT_URL = "wss://api.elevenlabs.io/v1/convai/conversation"

#: Frames that are part of a healthy conversation and carry nothing we score.
IGNORED = frozenset(
    {
        "conversation_initiation_metadata",
        "audio",
        "user_transcript",
        "internal_tentative_agent_response",
        "vad_score",
        "asr_initiation_metadata",
        "agent_response_correction",
    }
)


class ElevenLabsAgent:
    """An ElevenLabs agent under test. Satisfies the `Agent` protocol.

    One socket per episode: opened on the first turn, closed when the episode id
    changes. Carrying a connection across episodes would let call N+1 inherit
    N's context — the same defect the runner's `episode_id` fix closed for
    `HTTPAgent`.
    """

    def __init__(
        self,
        agent_id: str,
        *,
        api_key: str | None = None,
        url: str = DEFAULT_URL,
        timeout_s: float = 30.0,
        connect: Any = None,
    ) -> None:
        self.agent_id = agent_id
        self.api_key = api_key or os.environ.get("ELEVENLABS_API_KEY")
        self.url = url
        self.timeout_s = timeout_s
        if connect is None:
            from websockets.sync.client import connect as _connect

            connect = _connect
        self._connect = connect
        self._episode: str | None = None
        self._socket: Any = None
        self.latencies_ms: list[float] = []
        self.errors = 0
        self.turns = 0

    # -- connection --------------------------------------------------------- #

    def _open(self) -> Any:
        headers = {"xi-api-key": self.api_key} if self.api_key else {}
        socket = self._connect(
            f"{self.url}?agent_id={self.agent_id}",
            additional_headers=headers,
            open_timeout=self.timeout_s,
        )
        socket.send(
            json.dumps(
                {
                    "type": "conversation_initiation_client_data",
                    "conversation_config_override": {
                        # No audio anywhere in the loop — the harness's whole
                        # premise, and the vendor's own recommendation for cost.
                        "conversation": {"text_only": True}
                    },
                }
            )
        )
        return socket

    def close(self) -> None:
        if self._socket is not None:
            with contextlib.suppress(Exception):
                self._socket.close()  # closing an already-dead socket is not news
            self._socket = None

    # -- the one method ----------------------------------------------------- #

    def respond(self, request: AgentRequest) -> AgentResponse:
        if request.episode_id != self._episode:
            self.close()
            self._episode = request.episode_id
            try:
                self._socket = self._open()
            except Exception as exc:
                self.errors += 1
                return AgentResponse(text="", error=f"connect failed: {exc}")

        latest = next(
            (
                view.text
                for view in reversed(request.history)
                if view.speaker is Speaker.USER and view.text.strip()
            ),
            "",
        )
        if not latest:
            return AgentResponse(text="", error="nothing to say")

        started = time.monotonic()
        try:
            self._socket.send(json.dumps({"type": "user_message", "text": latest}))
            text = self._await_reply()
        except Exception as exc:
            self.errors += 1
            self.close()
            return AgentResponse(
                text="",
                error=f"{type(exc).__name__}: {exc}",
                latency_ms=(time.monotonic() - started) * 1000,
            )

        latency_ms = (time.monotonic() - started) * 1000
        self.latencies_ms.append(latency_ms)
        self.turns += 1
        if not text:
            self.errors += 1
            return AgentResponse(text="", error="no agent_response", latency_ms=latency_ms)
        return AgentResponse(text=text, latency_ms=latency_ms)

    def _await_reply(self) -> str:
        """Read frames until the agent speaks, skipping the rest.

        Bounded by wall clock, not by frame count: a socket that streams audio
        chunks forever would otherwise hang the whole sweep on one turn.
        """
        deadline = time.monotonic() + self.timeout_s
        while time.monotonic() < deadline:
            remaining = max(0.1, deadline - time.monotonic())
            raw = self._socket.recv(timeout=remaining)
            try:
                frame: JSONObject = json.loads(raw)
            except (TypeError, ValueError):
                continue
            kind = frame.get("type")
            if kind == "ping":
                event = frame.get("ping_event")
                if isinstance(event, dict):
                    self._socket.send(
                        json.dumps({"type": "pong", "event_id": event.get("event_id")})
                    )
                continue
            if kind in IGNORED:
                continue
            if kind == "agent_response":
                event = frame.get("agent_response_event")
                if isinstance(event, dict):
                    said = event.get("agent_response")
                    if isinstance(said, str) and said.strip():
                        return said.strip()
        return ""

    @property
    def p95_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        ordered = sorted(self.latencies_ms)
        return ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]
