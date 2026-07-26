"""Assembling the reward vector.

The component list is fixed and always the same length, including when cost
shaping is disabled — a reward vector whose shape depends on config is a logging
format that breaks halfway through a training run.
"""

from __future__ import annotations

from simharness.schemas import (
    CostSummary,
    RewardBreakdown,
    RewardComponent,
    RewardConfig,
)


def cost_pressure(cost: CostSummary, config: RewardConfig) -> tuple[float, dict[str, float]]:
    """Normalised agent expense, in units of "one reference episode".

    Only the agent's own turns and tokens count. The simulator's spend is real
    money and is reported, but charging the policy for it would teach it to make
    its *counterpart* terse.
    """
    turn_ratio = cost.turns / config.cost_reference_turns
    token_ratio = cost.agent_tokens.total / config.cost_reference_tokens
    raw = min(3.0, 0.5 * turn_ratio + 0.5 * token_ratio)
    return raw, {"turn_ratio": round(turn_ratio, 4), "token_ratio": round(token_ratio, 4)}


def build_reward(
    *,
    task_success: bool,
    field_accuracy: float,
    forbidden_hits: int,
    claim_accuracy: float,
    clean_termination: bool,
    cost: CostSummary,
    config: RewardConfig,
) -> RewardBreakdown:
    raw_cost, cost_detail = cost_pressure(cost, config)
    components = (
        RewardComponent(
            name="task_success",
            raw=1.0 if task_success else 0.0,
            weight=config.w_task_success,
        ),
        RewardComponent(
            name="field_accuracy",
            raw=field_accuracy,
            weight=config.w_field_accuracy,
        ),
        RewardComponent(
            name="forbidden_mutation",
            raw=1.0 if forbidden_hits else 0.0,
            weight=config.w_forbidden_mutation,
            detail={"hits": forbidden_hits},
        ),
        RewardComponent(
            name="claim_accuracy",
            raw=claim_accuracy,
            weight=config.w_claim_accuracy,
        ),
        RewardComponent(
            name="termination",
            raw=1.0 if clean_termination else 0.0,
            weight=config.w_termination,
        ),
        RewardComponent(
            name="cost",
            raw=raw_cost if config.cost_shaping_enabled else 0.0,
            weight=config.w_cost,
            detail={**cost_detail, "enabled": config.cost_shaping_enabled},
        ),
    )
    return RewardBreakdown.from_components(components, config)
