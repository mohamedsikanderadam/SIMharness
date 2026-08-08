"""Point the harness at a deployed voice agent and get a scorecard.

    python scripts/eval_http_agent.py \
        --endpoint https://vendor.example/agent \
        --facts examples/dubai_clinic.json \
        --episodes 10

What it measures, and what it cannot: the agent's backend is not ours, so the
mutation ledger stays empty and "did it actually book the appointment" is not
answerable from here. What IS answerable, and is what the scorecard reports:
every price/hours/policy claim checked against the fact sheet, whether the
critical number was confirmed back on a noisy line, how the call ended, and
per-turn latency. See simharness/adapters/http.py for how to get the other half
back (point a test instance's webhooks at a URL you control).

Needs ANTHROPIC_API_KEY for the simulated caller. The agent under test costs
whatever the vendor charges.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simharness.adapters.http import HTTPAgent
from simharness.personas import get_persona
from simharness.runner import run_episode
from simharness.scenarios.blackbox import blackbox_scenario
from simharness.schemas import ClaimVerdict, NoiseConfig, SimulatorConfig, Speaker
from simharness.simulator.providers.anthropic_provider import AnthropicSimulator
from simharness.world.builders import WORLD_BUILDERS
from simharness.world.factsheet import load_facts, world_from_facts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--persona", default="rushed_booker")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--wer", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--confirm-value", type=int, default=None)
    parser.add_argument("--header", action="append", default=[], metavar="K:V")
    parser.add_argument("--session-key", default=None)
    parser.add_argument("--simulator-model", default="claude-opus-5")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set (needed for the simulated caller).", file=sys.stderr)
        return 2

    facts = load_facts(args.facts)
    builder_name = f"factsheet:{facts.get('business_id', 'under-test')}"
    WORLD_BUILDERS[builder_name] = lambda _seed: world_from_facts(facts)

    scenario = blackbox_scenario(
        scenario_id="blackbox",
        title=f"Black-box eval: {facts.get('name', args.endpoint)}",
        world_builder=builder_name,
        confirm_value=args.confirm_value,
    )
    headers = dict(h.split(":", 1) for h in args.header)
    agent = HTTPAgent(
        args.endpoint,
        headers={k: v.strip() for k, v in headers.items()},
        session_key=args.session_key,
    )
    simulator = AnthropicSimulator(SimulatorConfig(model=args.simulator_model))

    rows: list[dict[str, Any]] = []
    handle = args.out.open("w", encoding="utf-8") if args.out else None
    for index in range(args.episodes):
        trajectory, card = run_episode(
            scenario=scenario,
            agent=agent,
            persona=get_persona(args.persona),
            seed=args.seed + index,
            episode_index=index,
            noise=NoiseConfig(target_wer=args.wer),
            simulator=simulator,
        )
        bad = [
            c
            for c in card.claim_checks
            if c.verdict in (ClaimVerdict.INCORRECT, ClaimVerdict.UNGROUNDED)
        ]
        rows.append(
            {
                "reward": card.reward.scalar,
                "passed": card.passed,
                "coverage": card.claim_coverage,
                "bad_claims": len(bad),
                "termination": str(card.termination),
                "failures": [f.value for f in card.failures],
                "turns": sum(1 for t in trajectory.turns if t.speaker is Speaker.USER),
            }
        )
        print(
            f"  [{index:2}] reward={card.reward.scalar:+.2f} cov={card.claim_coverage:.2f} "
            f"false_claims={len(bad)} {rows[-1]['termination']:20} {rows[-1]['failures']}"
        )
        for claim in bad:
            print(
                f"        ✗ {claim.kind}: {claim.surface!r} "
                f"(truth={claim.ground_truth}, verdict={claim.verdict})"
            )
        if handle:
            handle.write(trajectory.model_dump_json() + "\n")
    if handle:
        handle.close()

    print("\n" + "=" * 72)
    print(f"episodes            : {len(rows)}")
    print(f"clean calls         : {sum(r['passed'] for r in rows)}/{len(rows)}")
    print(f"false claims        : {sum(r['bad_claims'] for r in rows)} across all calls")
    print(f"claim coverage      : {statistics.mean(r['coverage'] for r in rows):.2f}")
    print(f"mean caller turns   : {statistics.mean(r['turns'] for r in rows):.1f}")
    print(f"agent p95 latency   : {agent.p95_latency_ms:.0f} ms")
    print(f"transport errors    : {agent.errors}")
    print(f"simulator cost      : ${simulator.usd:.4f} ({simulator.price_table_id})")
    tally: dict[str, int] = {}
    for row in rows:
        for failure in row["failures"]:
            tally[failure] = tally.get(failure, 0) + 1
    if tally:
        print("\nfailure modes:")
        for name, count in sorted(tally.items(), key=lambda kv: -kv[1]):
            print(f"  {count:3}x  {name}")
    print("\nNote: no world state. 'Did it actually complete the task' is NOT")
    print("measured here — only what the transcript can prove.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
