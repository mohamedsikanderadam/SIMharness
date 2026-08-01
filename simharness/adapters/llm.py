"""An LLM behind the `Agent` protocol — the policy the eval actually measures.

This is the first thing in the harness that puts a real model where a real
agent goes, which makes it the first honest test of two questions the scripted
path cannot answer: does a model emit usable tool calls at a workable rate, and
does the claim grammar bind anything against phrasing we did not write.

Tool calls arrive as structured `tool_use` blocks, so there is no parsing
convention to get wrong — the world's `ToolSpec` JSON Schemas are handed over
verbatim and come back as validated argument objects.

Thinking stays on deliberately. With thinking disabled, current models sometimes
write a tool call into visible text instead of emitting a `tool_use` block: the
turn succeeds, the call never runs, nothing errors, and in an agentic loop that
text pollutes every later turn. Lower effort is the cheap lever, not disabled
thinking.
"""

from __future__ import annotations

from typing import Any

from simharness.pricing import PRICE_TABLE_ID, estimate
from simharness.schemas import (
    AgentRequest,
    AgentResponse,
    JSONObject,
    Speaker,
    TokenUsage,
    ToolCall,
    ToolName,
    ToolSpec,
)

SYSTEM_SUFFIX = """
You are on a live phone call. The caller's words reach you through a noisy line,
so numbers may arrive wrong — confirm anything that matters before you act on it.

Reply with what you say out loud. Use the tools to look things up and to write to
the system; do not claim you have done something you have not done through a tool,
and do not state a price, a policy, or an availability you have not looked up.
""".strip()


def _tool_params(spec: ToolSpec) -> dict[str, Any]:
    """The world's JSON Schema, with the field the API requires added."""
    params = dict(spec.parameters)
    params.setdefault("additionalProperties", False)
    return params


class AnthropicAgent:
    """An Anthropic model under test. Satisfies `simharness.adapters.base.Agent`."""

    def __init__(
        self,
        model: str = "claude-opus-5",
        effort: str = "medium",
        max_tokens: int = 4096,
        client: Any = None,
    ) -> None:
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self._client = client
        self.usd = 0.0
        self.usage = TokenUsage()
        self.price_table_id = PRICE_TABLE_ID
        self.refusals = 0
        self.malformed_tool_calls = 0
        self._turn_key: tuple[str, int] | None = None
        self._messages: list[dict[str, Any]] = []

    def respond(self, request: AgentRequest) -> AgentResponse:
        self._sync_messages(request)

        response = self._client.beta.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            system=[
                {
                    "type": "text",
                    "text": f"{request.brief}\n\n{SYSTEM_SUFFIX}",
                    # The brief and the tool list are byte-identical across every
                    # turn of every episode on this scenario. Caching them is the
                    # difference between paying for the preamble once and paying
                    # for it on every turn of a twelve-turn call.
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            output_config={"effort": self.effort},
            tools=[
                {
                    "name": spec.name.value,
                    "description": spec.description,
                    "input_schema": _tool_params(spec),
                }
                for spec in request.tools
            ],
            messages=self._messages,
        )
        self._account(response)

        if response.stop_reason == "refusal":
            self.refusals += 1
            return AgentResponse(text="", error="policy refusal")

        said = " ".join(b.text for b in response.content if b.type == "text").strip()
        calls: list[ToolCall] = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            try:
                name = ToolName(block.name)
            except ValueError:
                self.malformed_tool_calls += 1
                continue
            arguments: JSONObject = dict(block.input) if isinstance(block.input, dict) else {}
            calls.append(ToolCall(call_id=block.id, name=name, arguments=arguments))

        # Record the assistant turn so the follow-up call (with tool results)
        # continues the same conversation rather than restarting it.
        self._messages.append({"role": "assistant", "content": response.content})

        if not said and not calls:
            return AgentResponse(text="", error="empty response")
        return AgentResponse(
            text=said,
            tool_calls=tuple(calls),
            usage=TokenUsage(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
            ),
        )

    # -- internals ---------------------------------------------------------- #

    def _sync_messages(self, request: AgentRequest) -> None:
        """Rebuild on a new turn; append tool results inside one.

        The runner re-invokes the agent with `pending_tool_results` until it
        speaks, so a turn is several API calls with one growing message list.
        """
        key = (request.episode_id, request.turn_index)
        if key != self._turn_key:
            self._turn_key = key
            self._messages = self._from_history(request)
            return
        if request.pending_tool_results:
            self._messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": result.call_id,
                            "content": str(result.data if result.ok else result.error),
                            "is_error": not result.ok,
                        }
                        for result in request.pending_tool_results
                    ],
                }
            )

    @staticmethod
    def _from_history(request: AgentRequest) -> list[dict[str, Any]]:
        """`AgentTurnView` is all we get — speaker and text, nothing else.

        Tool results from *earlier* turns are deliberately not replayed: the
        agent's own words already carry whatever it learned, and re-sending old
        tool blocks would need ids this view does not expose. That is the type
        doing its job, not a gap.
        """
        messages: list[dict[str, Any]] = []
        for view in request.history:
            role = "user" if view.speaker is Speaker.USER else "assistant"
            text = view.text.strip()
            if not text:
                continue
            if messages and messages[-1]["role"] == role:
                messages[-1]["content"] += f"\n{text}"
            else:
                messages.append({"role": role, "content": text})
        if not messages or messages[0]["role"] != "user":
            messages.insert(0, {"role": "user", "content": "[The call connects.]"})
        if messages[-1]["role"] != "user":
            messages.append({"role": "user", "content": "[The caller waits.]"})
        return messages

    def _account(self, response: Any) -> None:
        usage = response.usage
        self.usage = TokenUsage(
            prompt_tokens=self.usage.prompt_tokens + usage.input_tokens,
            completion_tokens=self.usage.completion_tokens + usage.output_tokens,
        )
        self.usd += estimate(
            self.model,
            usage.input_tokens,
            usage.output_tokens,
            getattr(usage, "cache_read_input_tokens", 0) or 0,
            getattr(usage, "cache_creation_input_tokens", 0) or 0,
        )
