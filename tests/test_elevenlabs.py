"""The ElevenLabs text-only WebSocket adapter, against a fake socket.

Hermetic: no network, no agent id, no key. The point is that the protocol
handling is proven before anyone spends a hackathon hour debugging it live.
"""

from __future__ import annotations

import json
from typing import Any

from simharness.adapters.elevenlabs import ElevenLabsAgent
from simharness.runner import run_episode
from simharness.scenarios.blackbox import blackbox_scenario
from simharness.schemas import AgentRequest, AgentTurnView, NoiseConfig, Speaker
from simharness.simulator.base import ScriptedSimulator
from simharness.world.builders import WORLD_BUILDERS
from simharness.world.factsheet import world_from_facts

FACTS: dict[str, Any] = {
    "business_id": "ws-clinic",
    "name": "WS Clinic",
    "currency": "AED",
    "catalogue": [{"sku": "CONSULT", "name": "Consultation", "unit_price": 25000}],
    "policies": {"cancellation_window_hours": 24, "max_party_size": 1},
    "opening_hours": {"open": "08:00", "close": "20:00"},
    "slots": {"days": 2, "start_hour": 9, "count": 3, "capacity": 2},
}


class FakeSocket:
    """Replays a scripted frame stream and records what was sent."""

    def __init__(self, frames: list[dict[str, Any]]) -> None:
        self.frames = list(frames)
        self.sent: list[dict[str, Any]] = []
        self.closed = False

    def send(self, raw: str) -> None:
        self.sent.append(json.loads(raw))

    def recv(self, timeout: float | None = None) -> str:
        if not self.frames:
            raise TimeoutError("no more frames")
        return json.dumps(self.frames.pop(0))

    def close(self) -> None:
        self.closed = True


def _connector(sockets: list[FakeSocket]) -> Any:
    opened: list[FakeSocket] = []

    def connect(url: str, **kwargs: Any) -> FakeSocket:
        socket = sockets[min(len(opened), len(sockets) - 1)]
        socket.url = url  # type: ignore[attr-defined]
        socket.kwargs = kwargs  # type: ignore[attr-defined]
        opened.append(socket)
        return socket

    connect.opened = opened  # type: ignore[attr-defined]
    return connect


def _reply(text: str) -> dict[str, Any]:
    return {"type": "agent_response", "agent_response_event": {"agent_response": text}}


def _request(text: str = "Hello there.", episode: str = "e1") -> AgentRequest:
    return AgentRequest(
        episode_id=episode,
        turn_index=0,
        history=(AgentTurnView(speaker=Speaker.USER, text=text),),
        tools=(),
    )


# --------------------------------------------------------------------------- #
# Protocol
# --------------------------------------------------------------------------- #


def test_text_only_is_requested_on_connect() -> None:
    """Without this the vendor processes audio — which costs money and puts a
    speech stack inside a measurement that is supposed to be about their agent."""
    socket = FakeSocket([_reply("Hi.")])
    agent = ElevenLabsAgent("agent_x", api_key="k", connect=_connector([socket]))
    agent.respond(_request())

    init = socket.sent[0]
    assert init["type"] == "conversation_initiation_client_data"
    assert init["conversation_config_override"]["conversation"]["text_only"] is True


def test_agent_id_and_key_are_sent() -> None:
    socket = FakeSocket([_reply("Hi.")])
    connect = _connector([socket])
    ElevenLabsAgent("agent_x", api_key="secret", connect=connect).respond(_request())
    assert "agent_id=agent_x" in socket.url  # type: ignore[attr-defined]
    assert socket.kwargs["additional_headers"]["xi-api-key"] == "secret"  # type: ignore[attr-defined]


def test_the_user_turn_is_sent_as_a_user_message() -> None:
    socket = FakeSocket([_reply("Certainly.")])
    agent = ElevenLabsAgent("a", connect=_connector([socket]))
    response = agent.respond(_request("A table for six."))
    assert socket.sent[1] == {"type": "user_message", "text": "A table for six."}
    assert response.text == "Certainly."


def test_chatty_frames_are_skipped_not_fatal() -> None:
    """A real socket interleaves transcripts, VAD and audio. An adapter that
    treats an unrecognised frame as an error is useless against a live vendor."""
    socket = FakeSocket(
        [
            {"type": "conversation_initiation_metadata"},
            {"type": "vad_score", "vad_score_event": {"vad_score": 0.9}},
            {"type": "user_transcript", "user_transcription_event": {"user_transcript": "hi"}},
            {"type": "audio", "audio_event": {"audio_base_64": "AAAA"}},
            {"type": "some_future_frame_we_have_never_seen"},
            _reply("We open at eight."),
        ]
    )
    agent = ElevenLabsAgent("a", connect=_connector([socket]))
    assert agent.respond(_request()).text == "We open at eight."


def test_ping_is_answered() -> None:
    socket = FakeSocket(
        [
            {"type": "ping", "ping_event": {"event_id": 42}},
            _reply("Still here."),
        ]
    )
    agent = ElevenLabsAgent("a", connect=_connector([socket]))
    agent.respond(_request())
    assert {"type": "pong", "event_id": 42} in socket.sent


# --------------------------------------------------------------------------- #
# Failure and isolation
# --------------------------------------------------------------------------- #


def test_a_silent_vendor_becomes_an_error_not_a_hang() -> None:
    socket = FakeSocket([])
    agent = ElevenLabsAgent("a", timeout_s=0.3, connect=_connector([socket]))
    response = agent.respond(_request())
    assert response.error and not response.text
    assert agent.errors == 1


def test_a_failed_connection_does_not_raise() -> None:
    def refuse(url: str, **kwargs: Any) -> Any:
        raise ConnectionRefusedError("vendor down")

    agent = ElevenLabsAgent("a", connect=refuse)
    response = agent.respond(_request())
    assert "connect failed" in str(response.error)
    assert agent.errors == 1


def test_each_episode_gets_a_fresh_socket() -> None:
    """Reusing a connection would let call N+1 inherit N's context — the same
    defect the runner's episode_id fix closed for HTTPAgent."""
    first, second = FakeSocket([_reply("One.")]), FakeSocket([_reply("Two.")])
    connect = _connector([first, second])
    agent = ElevenLabsAgent("a", connect=connect)
    agent.respond(_request(episode="ep-1"))
    agent.respond(_request(episode="ep-2"))
    assert len(connect.opened) == 2  # type: ignore[attr-defined]
    assert first.closed, "the first episode's socket was left open"


# --------------------------------------------------------------------------- #
# End to end through the real runner and verifier
# --------------------------------------------------------------------------- #


def test_a_full_episode_scores_against_the_fact_sheet() -> None:
    WORLD_BUILDERS["ws-facts"] = lambda _seed: world_from_facts(FACTS)
    scenario = blackbox_scenario(
        scenario_id="blackbox", title="WS Clinic", world_builder="ws-facts"
    )
    socket = FakeSocket([_reply("A consultation is 250 dirhams.")] * 40)
    agent = ElevenLabsAgent("a", connect=_connector([socket]))

    trajectory, card = run_episode(
        scenario=scenario,
        agent=agent,
        persona="rushed_booker",
        seed=2,
        noise=NoiseConfig(target_wer=0.1),
        simulator=ScriptedSimulator(),
    )
    assert trajectory.turns
    assert card.reward.scalar is not None
    assert agent.turns > 0 and agent.p95_latency_ms >= 0
