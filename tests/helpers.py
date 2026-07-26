"""Episode construction for tests, standing in for the Phase 4 runner.

Deliberately not part of the package: the runner will build trajectories the same
way, and if these tests depended on the runner they would stop being able to tell
a verifier bug from a runner bug.
"""

from __future__ import annotations

from datetime import datetime

from simharness.schemas import (
    CostSummary,
    EpisodeSeeds,
    JSONObject,
    Scenario,
    Speaker,
    TerminationReason,
    TokenUsage,
    ToolCall,
    ToolName,
    Trajectory,
    Turn,
)
from simharness.world import World, build_world

FIXED_CREATED_AT = datetime(2026, 3, 10, 10, 0)


class Episode:
    """Builds a `Trajectory` by driving a real `World`, one turn at a time."""

    def __init__(self, scenario: Scenario, seed: int = 0) -> None:
        self.scenario = scenario
        self.world = World(
            build_world(scenario.world_builder, scenario.world_seed), scenario.enabled_tools
        )
        self.seeds = EpisodeSeeds.derive(seed, scenario.scenario_id, "test-persona", 0)
        self.initial = self.world.snapshot(0)
        self.turns: list[Turn] = []

    def user(self, text: str, *, heard: str | None = None) -> Episode:
        """``heard`` is what the agent receives after ASR corruption."""
        self.turns.append(
            Turn(
                index=len(self.turns),
                speaker=Speaker.USER,
                text=text,
                delivered_text=heard if heard is not None else text,
            )
        )
        return self

    def agent(
        self,
        text: str,
        *,
        calls: list[tuple[ToolName, JSONObject]] | None = None,
        tokens: tuple[int, int] = (0, 0),
    ) -> Episode:
        index = len(self.turns)
        tool_calls: list[ToolCall] = []
        results = []
        for position, (name, arguments) in enumerate(calls or []):
            call = ToolCall(call_id=f"t{index}-{position}", name=name, arguments=arguments)
            tool_calls.append(call)
            results.append(self.world.execute(call, index))
        self.turns.append(
            Turn(
                index=index,
                speaker=Speaker.AGENT,
                text=text,
                delivered_text=text,
                tool_calls=tool_calls,
                tool_results=results,
                usage=TokenUsage(prompt_tokens=tokens[0], completion_tokens=tokens[1]),
            )
        )
        return self

    def finish(self, termination: TerminationReason = TerminationReason.SATISFIED) -> Trajectory:
        user_turns = sum(1 for t in self.turns if t.speaker is Speaker.USER)
        agent_usage = [t.usage for t in self.turns if t.usage is not None]
        return Trajectory(
            episode_id=f"{self.scenario.scenario_id}-test",
            scenario_id=self.scenario.scenario_id,
            persona_id="test-persona",
            seeds=self.seeds,
            config_digest="test",
            harness_version="0.1.0",
            created_at=FIXED_CREATED_AT,
            turns=self.turns,
            initial_world=self.initial,
            final_world=self.world.snapshot(len(self.turns)),
            termination=termination,
            cost=CostSummary(
                turns=user_turns,
                agent_tokens=TokenUsage(
                    prompt_tokens=sum(u.prompt_tokens for u in agent_usage),
                    completion_tokens=sum(u.completion_tokens for u in agent_usage),
                ),
            ),
        )

    def first_slot_id(self, date: str = "2026-03-12") -> str:
        return next(
            slot.slot_id
            for slot in self.world.state.business.calendar
            if slot.starts_at.date().isoformat() == date
        )
