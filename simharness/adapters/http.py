"""Agents we do not own, reached over HTTP.

The eval product. `AnthropicAgent` measures a model we control; this measures a
vendor's deployed agent — Vapi, ElevenLabs, Retell, a bare webhook — and the
runner cannot tell the difference. That was the claim in ARCHITECTURE.md §2 and
this module is where it gets paid.

**Dialects, because every vendor's wire format is different.** The transport,
retry, timing and failure handling are shared; the two vendor-specific bits are
`build` (turn an `AgentRequest` into a request body) and `parse` (turn a
response body into text plus tool calls). A new vendor is a ten-line dialect,
not a new adapter.

**What you lose against a black-box agent, stated plainly.** A third-party agent
calls *its* backend, not ours. Its tool calls never reach our world, so the
ledger stays empty and the required-record and forbidden-mutation checks have
nothing to read. What survives is the half of the verifier that works on the
transcript: claim grounding against ground truth you supply, evidence
requirements, and termination. Score a black-box agent with a scenario that
declares no `required_records` — see `scripts/eval_http_agent.py`.

The exception worth chasing: if the vendor will point a *test* instance's
webhooks at a URL you control, the world comes back and so does the whole
verifier. That is the difference between "did it say true things" and "did it do
the right thing", and it is worth an email.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from simharness.schemas import (
    AgentRequest,
    AgentResponse,
    JSONObject,
    Speaker,
    ToolCall,
    ToolName,
)

Build = Callable[[AgentRequest, str | None], JSONObject]
Parse = Callable[[JSONObject], tuple[str, tuple[ToolCall, ...]]]


# --------------------------------------------------------------------------- #
# Dialects
# --------------------------------------------------------------------------- #


def _history_as_messages(request: AgentRequest) -> list[JSONObject]:
    """OpenAI-shaped history. Most vendors accept something close to this."""
    return [
        {
            "role": "user" if view.speaker is Speaker.USER else "assistant",
            "content": view.text,
        }
        for view in request.history
        if view.text.strip()
    ]


def build_generic(request: AgentRequest, session_id: str | None) -> JSONObject:
    """`{session_id, message, messages: [...]}` — the common denominator.

    `message` carries the latest caller turn on its own because a surprising
    number of webhook agents read only that and ignore history.
    """
    messages = _history_as_messages(request)
    latest = next(
        (
            view.text
            for view in reversed(request.history)
            if view.speaker is Speaker.USER and view.text.strip()
        ),
        "",
    )
    body: JSONObject = {"message": latest, "messages": list(messages)}
    if session_id:
        body["session_id"] = session_id
    return body


def parse_generic(payload: JSONObject) -> tuple[str, tuple[ToolCall, ...]]:
    """Pull the agent's speech out of whatever the vendor called the field.

    Tool calls are parsed when present and simply absent when not — a vendor
    that runs its own backend reports nothing here, and that is expected rather
    than an error.
    """
    text = ""
    for key in ("reply", "text", "message", "output", "response", "content", "answer"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            text = value.strip()
            break
        if isinstance(value, dict):
            inner = value.get("content") or value.get("text")
            if isinstance(inner, str) and inner.strip():
                text = inner.strip()
                break

    calls: list[ToolCall] = []
    raw = payload.get("tool_calls") or payload.get("toolCalls") or []
    if isinstance(raw, list):
        for index, item in enumerate(raw):
            if not isinstance(item, dict):
                continue
            name = item.get("name") or item.get("function")
            arguments = item.get("arguments") or item.get("input") or {}
            if not isinstance(name, str) or not isinstance(arguments, dict):
                continue
            try:
                tool = ToolName(name)
            except ValueError:
                continue
            call_id = item.get("id")
            calls.append(
                ToolCall(
                    call_id=call_id if isinstance(call_id, str) else f"http-{index}",
                    name=tool,
                    arguments=arguments,
                )
            )
    return text, tuple(calls)


# --------------------------------------------------------------------------- #
# The adapter
# --------------------------------------------------------------------------- #


class HTTPAgent:
    """A deployed agent, reached over HTTP. Satisfies the `Agent` protocol."""

    def __init__(
        self,
        endpoint: str,
        *,
        headers: dict[str, str] | None = None,
        timeout_s: float = 30.0,
        retries: int = 2,
        build: Build = build_generic,
        parse: Parse = parse_generic,
        session_key: str | None = None,
        client: Any = None,
    ) -> None:
        self.endpoint = endpoint
        self.timeout_s = timeout_s
        self.retries = retries
        self._build = build
        self._parse = parse
        self.session_key = session_key
        """Response field carrying a conversation id the vendor wants echoed
        back. Set it when the vendor is stateful; leave None when it is not."""
        if client is None:
            import httpx

            client = httpx.Client(timeout=timeout_s, headers=headers or {})
        self._client = client
        self._episode: str | None = None
        self._session_id: str | None = None
        self.latencies_ms: list[float] = []
        self.errors = 0

    def respond(self, request: AgentRequest) -> AgentResponse:
        if request.episode_id != self._episode:
            # A new episode is a new call. Never carry a vendor session across
            # one, or episode N+1 inherits N's context and the seeds stop
            # meaning anything.
            self._episode = request.episode_id
            self._session_id = None

        body = self._build(request, self._session_id)
        started = time.monotonic()
        payload, error = self._post(body)
        latency_ms = (time.monotonic() - started) * 1000
        self.latencies_ms.append(latency_ms)

        if error is not None:
            self.errors += 1
            return AgentResponse(text="", error=error, latency_ms=latency_ms)

        if self.session_key and isinstance(payload.get(self.session_key), str):
            self._session_id = str(payload[self.session_key])

        text, calls = self._parse(payload)
        if not text and not calls:
            self.errors += 1
            return AgentResponse(text="", error="no speech in response", latency_ms=latency_ms)
        return AgentResponse(text=text, tool_calls=calls, latency_ms=latency_ms)

    def _post(self, body: JSONObject) -> tuple[JSONObject, str | None]:
        """Returns (payload, error). One flaky call must not end the run — a
        failed turn is recorded as an agent error and the episode continues."""
        last = "unknown error"
        for attempt in range(self.retries + 1):
            try:
                response = self._client.post(self.endpoint, json=body)
            except Exception as exc:
                last = f"{type(exc).__name__}: {exc}"
            else:
                if response.status_code >= 500:
                    last = f"HTTP {response.status_code}"
                elif response.status_code >= 400:
                    # A 4xx is our bug, not theirs — retrying will not fix it.
                    return {}, f"HTTP {response.status_code}: {response.text[:200]}"
                else:
                    try:
                        parsed = response.json()
                    except Exception:
                        return {}, f"non-JSON response: {response.text[:200]}"
                    return (parsed if isinstance(parsed, dict) else {"text": parsed}), None
            if attempt < self.retries:
                time.sleep(0.5 * (2**attempt))
        return {}, last

    @property
    def p95_latency_ms(self) -> float:
        """Voice agents are judged on latency as much as on correctness, and
        the mean hides the turns that make a call feel broken."""
        if not self.latencies_ms:
            return 0.0
        ordered = sorted(self.latencies_ms)
        return ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]
