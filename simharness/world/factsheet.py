"""Build a world from a JSON fact sheet — the on-ramp for a business we don't own.

For a black-box agent the world stops being a mutable store and becomes a
*fact sheet*: the published truths its claims get checked against. Prices, hours,
policies, the diary. Nothing writes to it, because the vendor's agent writes to
the vendor's backend.

That makes onboarding a new business a data task, not a code task — which is the
difference between evaluating one agent and evaluating ten. Write the JSON from
the customer's own website or price list, and every number the agent says out
loud is now checkable.

    {
      "business_id": "dha-clinic",
      "name": "Example Clinic",
      "catalogue": [{"sku": "CONSULT", "name": "Consultation", "unit_price": 25000}],
      "policies": {"cancellation_window_hours": 24, "max_party_size": 1},
      "opening_hours": {"open": "08:00", "close": "20:00", "closed_weekday": 4},
      "slots": {"days": 7, "start_hour": 9, "count": 8, "capacity": 2}
    }

Money is minor units throughout — 25000 is AED 250.00. Getting this wrong is the
single easiest way to make every price claim read as a hallucination.
"""

from __future__ import annotations

import json
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Any, Final

from simharness.schemas import (
    AvailabilitySlot,
    BusinessConfig,
    CatalogueItem,
    OpeningHours,
    Policies,
    WorldState,
)

PINNED_NOW: Final = datetime(2026, 3, 10, 10, 0)
"""Tuesday morning. Fixed rather than seeded: every claim about dates, days of
the week and opening hours has to mean the same thing in every episode, or two
transcripts stop being comparable.

Previously imported from ``world.builders``, which the red-team pivot removed —
this is the only definition left, so it lives here now.
"""


def _hours(spec: dict[str, Any]) -> tuple[OpeningHours, ...]:
    opens = time.fromisoformat(str(spec.get("open", "09:00")))
    closes = time.fromisoformat(str(spec.get("close", "17:00")))
    closed_weekday = spec.get("closed_weekday")
    return tuple(
        OpeningHours(weekday=day, opens=opens, closes=closes, closed=day == closed_weekday)
        for day in range(7)
    )


def _slots(spec: dict[str, Any]) -> tuple[AvailabilitySlot, ...]:
    days = int(spec.get("days", 7))
    start_hour = int(spec.get("start_hour", 9))
    count = int(spec.get("count", 8))
    capacity = int(spec.get("capacity", 2))
    slots: list[AvailabilitySlot] = []
    for day in range(1, days + 1):
        base = (PINNED_NOW + timedelta(days=day)).replace(
            hour=start_hour, minute=0, second=0, microsecond=0
        )
        for step in range(count):
            starts = base + timedelta(hours=step)
            slots.append(
                AvailabilitySlot(
                    slot_id=f"SL-{starts:%m%d-%H%M}", starts_at=starts, capacity=capacity
                )
            )
    return tuple(slots)


def world_from_facts(facts: dict[str, Any], now: datetime | None = None) -> WorldState:
    """A world whose only job is to be the truth the agent is measured against."""
    policies = dict(facts.get("policies", {}))
    business = BusinessConfig(
        business_id=str(facts.get("business_id", "under-test")),
        name=str(facts.get("name", "Business Under Test")),
        timezone=str(facts.get("timezone", "Asia/Dubai")),
        catalogue=tuple(
            CatalogueItem(
                sku=str(item["sku"]),
                name=str(item.get("name", item["sku"])),
                unit_price=int(item["unit_price"]),
                currency=str(item.get("currency", facts.get("currency", "AED"))),
            )
            for item in facts.get("catalogue", [])
        ),
        opening_hours=_hours(facts.get("opening_hours", {})),
        policies=Policies(
            cancellation_window_hours=int(policies.get("cancellation_window_hours", 24)),
            deposit_required_from_party_size=int(
                policies.get("deposit_required_from_party_size", 1)
            ),
            deposit_per_head=int(policies.get("deposit_per_head", 0)),
            refund_window_hours=int(policies.get("refund_window_hours", 24)),
            max_party_size=int(policies.get("max_party_size", 1)),
            discount_authority=int(policies.get("discount_authority", 0)),
        ),
        calendar=_slots(facts.get("slots", {})),
    )
    return WorldState(business=business, now=now or PINNED_NOW)


def load_facts(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        loaded: dict[str, Any] = json.load(handle)
    return loaded
