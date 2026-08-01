"""The LLM provider clients, exercised against a mock — no key, no network.

These are the cases that would otherwise only fail against a live API, which is
the worst place to discover them.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from simharness.adapters.llm import AnthropicAgent
from simharness.personas import get_persona
from simharness.schemas import (
    AgentRequest,
    Disclosure,
    SimulatorConfig,
    SimulatorContext,
    SimulatorInternalState,
    ToolName,
)
from simharness.simulator.providers.anthropic_provider import AnthropicSimulator
from simharness.world import World, build_world

SAMPLING_PARAMS = {"temperature", "top_p", "top_k"}
UTTERANCE_JSON = '{"utterance": "Hi.", "terminate": false, "termination": null}'


def _response(
    *,
    text: str = "",
    tool_uses: list[tuple[str, dict]] | None = None,
    stop_reason: str = "end_turn",
) -> Any:
    response = MagicMock()
    response.stop_reason = stop_reason
    response.usage.input_tokens = 100
    response.usage.output_tokens = 20
    response.usage.cache_read_input_tokens = 0
    response.usage.cache_creation_input_tokens = 0
    blocks = []
    if text:
        block = MagicMock()
        block.type = "text"
        block.text = text
        blocks.append(block)
    for index, (name, arguments) in enumerate(tool_uses or []):
        block = MagicMock()
        block.type = "tool_use"
        block.id = f"toolu_{index}"
        block.name = name
        block.input = arguments
        blocks.append(block)
    response.content = blocks
    return response


def _client(response: Any) -> MagicMock:
    client = MagicMock()
    client.beta.messages.create.return_value = response
    return client


def _sim_context(persona_id: str = "haggler") -> SimulatorContext:
    return SimulatorContext(
        persona=get_persona(persona_id),
        internal_state=SimulatorInternalState(patience_remaining=5),
        history=(),
        turn_index=0,
        seed=1,
    )


def _agent_request() -> AgentRequest:
    world = World(build_world("vendor", 1), (ToolName.GET_PRICE, ToolName.APPLY_DISCOUNT))
    return AgentRequest(
        episode_id="e1", turn_index=0, history=(), tools=world.specs(), brief="You sell software."
    )


# --------------------------------------------------------------------------- #
# The structural guarantee
# --------------------------------------------------------------------------- #


def test_withheld_facts_never_enter_the_prompt() -> None:
    """The no-leak guarantee cannot rest on the model's discretion.

    The haggler's real budget ceiling is WITHHELD, so it is filtered out before
    the request is built — the model is never in a position to reveal it.
    """
    persona = get_persona("haggler")
    secret = next(f.value for f in persona.hidden_facts if f.disclosure is Disclosure.WITHHELD)
    client = _client(
        _response(text='{"utterance": "Hi.", "terminate": false, "termination": null}')
    )
    AnthropicSimulator(SimulatorConfig(), client=client).generate(_sim_context())

    sent = str(client.beta.messages.create.call_args.kwargs)
    assert secret not in sent, "a withheld fact reached the provider request"
    # ...while a fact it is allowed to volunteer does reach it.
    assert "Twenty seats" in sent


# --------------------------------------------------------------------------- #
# Request shape — these all 400 or silently misbehave against the live API
# --------------------------------------------------------------------------- #


def test_no_sampling_parameters_are_sent() -> None:
    """`temperature` / `top_p` / `top_k` were removed from current models and
    return a 400. A config carrying one would fail every single call."""
    for build, run in (
        (
            lambda c: AnthropicSimulator(SimulatorConfig(), client=c),
            lambda o: o.generate(_sim_context()),
        ),
        (lambda c: AnthropicAgent(client=c), lambda o: o.respond(_agent_request())),
    ):
        client = _client(_response(text=UTTERANCE_JSON))
        run(build(client))
        assert not SAMPLING_PARAMS & set(client.beta.messages.create.call_args.kwargs)


def test_agent_passes_the_worlds_tool_schemas_verbatim() -> None:
    client = _client(_response(text="One moment."))
    agent = AnthropicAgent(client=client)
    request = _agent_request()
    agent.respond(request)

    sent = {t["name"] for t in client.beta.messages.create.call_args.kwargs["tools"]}
    assert sent == {spec.name.value for spec in request.tools}


def test_system_prompt_carries_a_cache_breakpoint() -> None:
    """The brief is byte-identical on every turn of every episode. Without a
    breakpoint it is re-billed at full price on each one."""
    client = _client(_response(text="One moment."))
    AnthropicAgent(client=client).respond(_agent_request())
    system = client.beta.messages.create.call_args.kwargs["system"]
    assert any("cache_control" in block for block in system)


def test_first_message_is_always_a_user_turn() -> None:
    """The API rejects a conversation that opens with an assistant turn."""
    client = _client(
        _response(text='{"utterance": "Hi.", "terminate": false, "termination": null}')
    )
    AnthropicSimulator(SimulatorConfig(), client=client).generate(_sim_context())
    assert client.beta.messages.create.call_args.kwargs["messages"][0]["role"] == "user"


# --------------------------------------------------------------------------- #
# Response handling
# --------------------------------------------------------------------------- #


def test_refusal_is_handled_before_content_is_read() -> None:
    """A refusal arrives as a 200 with empty content. Indexing content[0]
    unconditionally is the standard way this crashes in production."""
    agent = AnthropicAgent(client=_client(_response(stop_reason="refusal")))
    response = agent.respond(_agent_request())
    assert response.error == "policy refusal"
    assert agent.refusals == 1

    simulator = AnthropicSimulator(
        SimulatorConfig(), client=_client(_response(stop_reason="refusal"))
    )
    output = simulator.generate(_sim_context())
    assert output.terminate and simulator.refusals == 1


def test_tool_use_blocks_become_tool_calls() -> None:
    client = _client(_response(text="Checking.", tool_uses=[("get_price", {"sku": "SEAT-PRO"})]))
    response = AnthropicAgent(client=client).respond(_agent_request())
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].name is ToolName.GET_PRICE
    assert response.tool_calls[0].arguments == {"sku": "SEAT-PRO"}


def test_unknown_tool_names_are_counted_not_crashed() -> None:
    client = _client(_response(text="ok", tool_uses=[("teleport", {})]))
    agent = AnthropicAgent(client=client)
    response = agent.respond(_agent_request())
    assert agent.malformed_tool_calls == 1
    assert response.tool_calls == ()


def test_malformed_simulator_json_ends_the_episode_visibly() -> None:
    """A body the schema should have prevented must not take the run down —
    but it must not pass silently either."""
    simulator = AnthropicSimulator(SimulatorConfig(), client=_client(_response(text="not json")))
    output = simulator.generate(_sim_context())
    assert output.terminate and output.termination == "harness_error"


def test_cost_is_accounted_against_a_named_price_table() -> None:
    agent = AnthropicAgent(client=_client(_response(text="One moment.")))
    agent.respond(_agent_request())
    assert agent.usd > 0
    assert agent.price_table_id.startswith("anthropic-1p-")
    assert agent.usage.total == 120
