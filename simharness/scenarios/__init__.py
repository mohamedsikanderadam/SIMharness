"""The shipped scenarios.

Lives under the package rather than at the repository root so that
``python -m simharness run --scenario booking`` works for an installed wheel,
not only from a checkout.

Adding one: build a `Scenario`, register it below, and — if it needs a different
starting world — add a builder in :mod:`simharness.world.builders`.
"""

from typing import Final

from simharness.scenarios.booking import BOOKING
from simharness.scenarios.refund import REFUND_ADVERSARY
from simharness.scenarios.reschedule import RESCHEDULE
from simharness.scenarios.sales_discount import SALES_DISCOUNT
from simharness.schemas import Scenario

SCENARIOS: Final[dict[str, Scenario]] = {
    scenario.scenario_id: scenario
    for scenario in (BOOKING, RESCHEDULE, REFUND_ADVERSARY, SALES_DISCOUNT)
}


def get_scenario(scenario_id: str) -> Scenario:
    try:
        return SCENARIOS[scenario_id]
    except KeyError:
        known = ", ".join(sorted(SCENARIOS))
        raise KeyError(f"unknown scenario {scenario_id!r}; known: {known}") from None


__all__ = [
    "BOOKING",
    "REFUND_ADVERSARY",
    "RESCHEDULE",
    "SALES_DISCOUNT",
    "SCENARIOS",
    "get_scenario",
]
