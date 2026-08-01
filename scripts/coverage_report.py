"""Run real models through the harness and report what the scripted path can't.

Two questions this answers and nothing else in the repo does:

1. **Does a model emit usable tool calls?** A policy that narrates tool use in
   prose instead of emitting `tool_use` blocks scores zero on every scenario for
   reasons that have nothing to do with the reward.
2. **What does `claim_coverage` read against phrasing we did not write?** The
   claim grammar has only ever been measured against agent text authored in this
   repo. If coverage collapses on real output, the reward is weaker than it
   looks and training against it would optimise straight into the gap.

    uv run --python .venv/bin/python scripts/coverage_report.py --episodes 20

Needs ANTHROPIC_API_KEY. Prints a per-run cost summary; nothing is written
unless --out is given.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simharness.adapters.llm import AnthropicAgent
from simharness.personas import SCENARIO_PERSONAS
from simharness.runner import run_episode
from simharness.scenarios import SCENARIOS
from simharness.schemas import ClaimVerdict, NoiseConfig, SimulatorConfig, Speaker
from simharness.simulator.providers.anthropic_provider import AnthropicSimulator


def _pairs() -> list[tuple[str, str]]:
    return [
        (scenario_id, persona_id)
        for scenario_id in sorted(SCENARIOS)
        for persona_id in SCENARIO_PERSONAS[scenario_id]
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--wer", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--agent-model", default="claude-opus-5")
    parser.add_argument("--simulator-model", default="claude-opus-5")
    parser.add_argument("--agent-effort", default="medium")
    parser.add_argument("--simulator-effort", default="low")
    parser.add_argument("--out", type=Path, default=None, help="write trajectories as JSONL")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set. Export it and re-run.", file=sys.stderr)
        return 2

    simulator = AnthropicSimulator(
        SimulatorConfig(model=args.simulator_model, effort=args.simulator_effort)
    )
    agent = AnthropicAgent(model=args.agent_model, effort=args.agent_effort)
    noise = NoiseConfig(target_wer=args.wer)
    pairs = _pairs()

    rows: list[dict[str, Any]] = []
    handle = args.out.open("w", encoding="utf-8") if args.out else None

    for index in range(args.episodes):
        scenario_id, persona_id = pairs[index % len(pairs)]
        scenario = SCENARIOS[scenario_id]
        try:
            trajectory, card = run_episode(
                scenario=scenario,
                agent=agent,
                persona=persona_id,
                seed=args.seed + index,
                episode_index=index,
                noise=noise,
                simulator=simulator,
            )
        except Exception as exc:
            print(f"  [{index:2}] {scenario_id}/{persona_id}: FAILED {type(exc).__name__}: {exc}")
            rows.append({"scenario": scenario_id, "error": str(exc)})
            continue

        agent_turns = [t for t in trajectory.turns if t.speaker is Speaker.AGENT]
        with_tools = [t for t in agent_turns if t.tool_calls]
        unparsed = sum(1 for c in card.claim_checks if c.verdict is ClaimVerdict.UNPARSED)
        rows.append(
            {
                "scenario": scenario_id,
                "persona": persona_id,
                "passed": card.passed,
                "reward": card.reward.scalar,
                "coverage": card.claim_coverage,
                "claims": len(card.claim_checks),
                "unparsed": unparsed,
                "agent_turns": len(agent_turns),
                "turns_with_tools": len(with_tools),
                "tool_calls": sum(len(t.tool_calls) for t in agent_turns),
                "failures": [f.value for f in card.failures],
                "termination": str(card.termination),
            }
        )
        print(
            f"  [{index:2}] {scenario_id:18} {persona_id:16} "
            f"pass={card.passed!s:5} reward={card.reward.scalar:+.2f} "
            f"cov={card.claim_coverage:.2f} tools={rows[-1]['tool_calls']:2} "
            f"{rows[-1]['failures']}"
        )
        if handle:
            handle.write(trajectory.model_dump_json() + "\n")

    if handle:
        handle.close()

    ok = [r for r in rows if "error" not in r]
    if not ok:
        print("\nNo episodes completed.")
        return 1

    coverages = [r["coverage"] for r in ok]
    turns_with_tools = sum(r["turns_with_tools"] for r in ok)
    agent_turns = sum(r["agent_turns"] for r in ok)

    print("\n" + "=" * 72)
    print(f"episodes            : {len(ok)} completed, {len(rows) - len(ok)} failed")
    print(f"pass rate           : {sum(r['passed'] for r in ok) / len(ok):.2f}")
    print(f"mean reward         : {statistics.mean(r['reward'] for r in ok):+.3f}")
    print(f"reward std          : {statistics.pstdev([r['reward'] for r in ok]):.3f}")
    print()
    print("--- the two questions this run exists to answer ---")
    print(
        f"tool-call rate      : {turns_with_tools}/{agent_turns} agent turns "
        f"({turns_with_tools / max(1, agent_turns):.0%}) emitted a tool_use block"
    )
    print(f"malformed tool calls: {agent.malformed_tool_calls}")
    print(
        f"claim coverage      : mean {statistics.mean(coverages):.2f}, "
        f"min {min(coverages):.2f}, max {max(coverages):.2f}"
    )
    print(
        f"unparsed claims     : {sum(r['unparsed'] for r in ok)} of "
        f"{sum(r['claims'] for r in ok)} total"
    )
    if statistics.mean(coverages) < 0.6:
        print()
        print("  ⚠ Coverage is well below what the authored transcripts showed.")
        print("    The claim grammar is not binding real phrasing. Training against")
        print("    this reward would optimise into the gap — extend the grammar first.")
    print()
    print("--- cost ---")
    print(f"price table         : {agent.price_table_id}")
    print(f"agent               : {agent.usage.total:>8} tok  ${agent.usd:.4f}")
    print(f"simulator           : {simulator.usage.total:>8} tok  ${simulator.usd:.4f}")
    print(f"total               : {'':>8}       ${agent.usd + simulator.usd:.4f}")
    print(f"per episode         : {'':>8}       ${(agent.usd + simulator.usd) / len(ok):.4f}")
    if agent.refusals or simulator.refusals:
        print(f"refusals            : agent {agent.refusals}, simulator {simulator.refusals}")
    if args.out:
        print(f"\ntrajectories        : {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
