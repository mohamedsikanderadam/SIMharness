"""The episode loop — the only module that knows about all the others.

    reset world -> (simulator turn -> noise -> agent -> tool calls mutate world)
    until terminal or patience exhausted -> verify -> Trajectory + Scorecard

Two details that are easy to get wrong and expensive to get wrong:

**Tool calls happen inside a turn, not as turns of their own.** The agent is
re-invoked with its pending tool results until it produces speech, so
`max_turns` means the same thing whether the agent is tool-happy or not.

**The customer hears itself, the agent hears the line.** A user turn records
`text` (what was said) and `delivered_text` (what arrived). `AgentTurnView` reads
the second and `SimulatorTurnView` reads the first, so a caller who said "six"
pushes back when the agent repeats "sixty" — which is the entire reason the
noise sweep measures anything.
"""

from __future__ import annotations

from datetime import datetime

from simharness.adapters.base import Agent
from simharness.noise.channel import corrupt
from simharness.personas.library import get_persona
from simharness.schemas import (
    AgentRequest,
    AgentTurnView,
    CostSummary,
    EpisodeSeeds,
    NoiseConfig,
    Persona,
    RewardConfig,
    Scenario,
    Scorecard,
    SimulatorContext,
    SimulatorInternalState,
    SimulatorTurnView,
    Speaker,
    TerminationReason,
    TokenUsage,
    ToolCall,
    ToolResult,
    Trajectory,
    Turn,
)
from simharness.simulator.base import ScriptedSimulator, Simulator
from simharness.verifier import verify
from simharness.world import World, build_world

HARNESS_VERSION = "0.1.0"
EPOCH = datetime(2026, 3, 10, 10, 0)
"""Trajectories are stamped with a fixed time, not `now()`. A replay that
disagrees with its original only on a timestamp still fails a digest comparison,
and the timestamp carries no information the episode does not already have."""


def run_episode(
    *,
    scenario: Scenario,
    agent: Agent,
    persona: Persona | str,
    seed: int,
    episode_index: int = 0,
    noise: NoiseConfig | None = None,
    simulator: Simulator | None = None,
    reward: RewardConfig | None = None,
    max_tool_iterations: int = 6,
) -> tuple[Trajectory, Scorecard]:
    """Run one episode and score it."""
    persona = get_persona(persona) if isinstance(persona, str) else persona
    simulator = simulator or ScriptedSimulator()
    noise = noise or NoiseConfig()
    seeds = EpisodeSeeds.derive(seed, scenario.scenario_id, persona.persona_id, episode_index)

    world = World(build_world(scenario.world_builder, seed), scenario.enabled_tools)
    initial = world.snapshot(0)
    turns: list[Turn] = []
    sim_state = SimulatorInternalState(patience_remaining=persona.patience_turns)
    termination: TerminationReason | None = None
    user_turns = 0

    while termination is None:
        if user_turns >= scenario.max_turns:
            termination = TerminationReason.MAX_TURNS
            break

        output = simulator.generate(
            SimulatorContext(
                persona=persona,
                internal_state=sim_state,
                history=tuple(
                    SimulatorTurnView.from_turn(t)
                    for t in turns
                    if t.speaker in (Speaker.USER, Speaker.AGENT)
                ),
                turn_index=len(turns),
                seed=seeds.for_turn("simulator", len(turns)),
            )
        )
        sim_state = output.internal_state

        delivered, trace = corrupt(
            output.utterance,
            noise,
            seeds.for_turn("noise", len(turns)),
            len(turns),
            persona.speech,
        )
        turns.append(
            Turn(
                index=len(turns),
                speaker=Speaker.USER,
                text=output.utterance,
                delivered_text=delivered,
                noise=trace,
                internal_state=sim_state,
            )
        )
        user_turns += 1

        if output.terminate:
            termination = output.termination
            break

        turns.append(_agent_turn(agent, scenario, world, turns, max_tool_iterations))

    final = world.snapshot(len(turns))
    trajectory = Trajectory(
        episode_id=f"{scenario.scenario_id}:{persona.persona_id}:{seed}:{episode_index}",
        scenario_id=scenario.scenario_id,
        persona_id=persona.persona_id,
        seeds=seeds,
        config_digest=(reward or RewardConfig()).digest,
        harness_version=HARNESS_VERSION,
        created_at=EPOCH,
        turns=turns,
        initial_world=initial,
        final_world=final,
        termination=termination or TerminationReason.MAX_TURNS,
        cost=_cost(turns, user_turns),
    )
    scorecard = verify(
        initial=initial,
        final=final,
        scenario=scenario,
        trajectory=trajectory,
        config=reward,
    )
    return trajectory, scorecard


def _agent_turn(
    agent: Agent, scenario: Scenario, world: World, turns: list[Turn], max_iterations: int
) -> Turn:
    """One agent turn, including its inner tool loop."""
    index = len(turns)
    history = tuple(AgentTurnView.from_turn(t) for t in turns if t.speaker is not Speaker.SYSTEM)
    pending: tuple[ToolResult, ...] = ()
    all_calls: list[ToolCall] = []
    all_results: list[ToolResult] = []
    text = ""
    usage = TokenUsage()

    for _ in range(max_iterations):
        response = agent.respond(
            AgentRequest(
                episode_id="",
                turn_index=index,
                history=history,
                tools=world.specs(),
                brief=scenario.agent_brief,
                pending_tool_results=pending,
            )
        )
        if response.usage:
            usage = TokenUsage(
                prompt_tokens=usage.prompt_tokens + response.usage.prompt_tokens,
                completion_tokens=usage.completion_tokens + response.usage.completion_tokens,
            )
        if response.tool_calls:
            results = [world.execute(call, index) for call in response.tool_calls]
            all_calls.extend(response.tool_calls)
            all_results.extend(results)
            pending = tuple(results)
        if response.text:
            text = response.text
            break
        if not response.tool_calls:
            break

    return Turn(
        index=index,
        speaker=Speaker.AGENT,
        text=text,
        delivered_text=text,
        tool_calls=list(all_calls),
        tool_results=all_results,
        usage=usage,
    )


def _cost(turns: list[Turn], user_turns: int) -> CostSummary:
    agent_usage = [t.usage for t in turns if t.speaker is Speaker.AGENT and t.usage]
    return CostSummary(
        turns=user_turns,
        agent_tokens=TokenUsage(
            prompt_tokens=sum(u.prompt_tokens for u in agent_usage),
            completion_tokens=sum(u.completion_tokens for u in agent_usage),
        ),
    )
