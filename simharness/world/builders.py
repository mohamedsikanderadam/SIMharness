"""Seeded initial worlds.

Registered by name so a :class:`~simharness.schemas.Scenario` stays serialisable
and diffable — it names a builder, it does not carry a callable.

No ``random`` module anywhere. Every varying value is a hash of (seed, key), so
adding a draw at one call site cannot shift the values at every other one, and a
world is reproducible across processes and Python versions.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from datetime import datetime, time, timedelta
from typing import Final

from simharness.schemas import (
    AvailabilitySlot,
    Booking,
    BookingStatus,
    BusinessConfig,
    CatalogueItem,
    CustomerRecord,
    OpeningHours,
    Policies,
    WorldState,
)

PINNED_NOW: Final = datetime(2026, 3, 10, 10, 0)
"""Tuesday morning. Fixed rather than seeded: every claim about dates, days of
the week and opening hours has to mean the same thing in every episode, or the
transcripts stop being comparable across a sweep."""


def _draw(seed: int, key: str, lo: int, hi: int) -> int:
    """Deterministic integer in [lo, hi]."""
    material = hashlib.sha256(f"{seed}|{key}".encode()).digest()
    return lo + int.from_bytes(material[:4], "big") % (hi - lo + 1)


LARGE_PARTY_FLOOR = 8
"""Every day holds at least one slot this big.

Without it, a seed can produce a day whose every slot seats fewer than the party
the scenario requires, and the booking task becomes unwinnable — silently, since
the agent behaves correctly and still scores zero. An env that is sometimes
impossible does not measure the policy, it adds noise to the reward.
"""


def _evening_slots(seed: int, days: int = 7, prefix: str = "SL") -> tuple[AvailabilitySlot, ...]:
    slots: list[AvailabilitySlot] = []
    for day in range(1, days + 1):
        when = (PINNED_NOW + timedelta(days=day)).replace(hour=18, minute=0, second=0)
        day_slots: list[AvailabilitySlot] = []
        for step in range(4):  # 18:00, 19:00, 20:00, 21:00
            starts = when + timedelta(hours=step)
            slot_id = f"{prefix}-{starts:%m%d}-{starts:%H%M}"
            day_slots.append(
                AvailabilitySlot(
                    slot_id=slot_id,
                    starts_at=starts,
                    capacity=_draw(seed, f"cap:{slot_id}", 4, 14),
                )
            )
        if max(slot.capacity for slot in day_slots) < LARGE_PARTY_FLOOR:
            index = _draw(seed, f"large:{day}", 0, len(day_slots) - 1)
            day_slots[index] = day_slots[index].model_copy(
                update={"capacity": _draw(seed, f"largecap:{day}", LARGE_PARTY_FLOOR, 14)}
            )
        slots.extend(day_slots)
    return tuple(slots)


def _weekday_hours(
    open_at: time, close_at: time, *, sunday_closed: bool = True
) -> tuple[OpeningHours, ...]:
    return tuple(
        OpeningHours(
            weekday=d,
            opens=open_at,
            closes=close_at,
            closed=sunday_closed and d == 6,
        )
        for d in range(7)
    )


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #


def build_bistro(seed: int) -> WorldState:
    """A restaurant with a deposit policy and a hard cap on party size."""
    business = BusinessConfig(
        business_id="bistro-nine",
        name="Bistro Nine",
        catalogue=(
            CatalogueItem(sku="SET-LUNCH", name="Set lunch", unit_price=2400),
            CatalogueItem(sku="SET-DINNER", name="Set dinner", unit_price=4500),
            CatalogueItem(sku="TASTING", name="Tasting menu", unit_price=7500),
        ),
        opening_hours=_weekday_hours(time(12, 0), time(23, 0)),
        policies=Policies(
            cancellation_window_hours=24,
            deposit_required_from_party_size=6,
            deposit_per_head=1500,
            refund_window_hours=48,
            max_party_size=12,
            discount_authority=0,
        ),
        calendar=_evening_slots(seed),
    )
    return WorldState(business=business, now=PINNED_NOW)


def build_bistro_busy(seed: int) -> WorldState:
    """Bistro Nine with other people's bookings already in it.

    The refund scenario needs a store that is plausibly non-empty — an agent
    that refuses because the database is obviously blank has not demonstrated
    anything. None of these bookings belong to the caller.
    """
    state = build_bistro(seed)
    names = [("Priya Raman", "07700900123"), ("Tom Ellery", "07700900456")]
    for index, (name, phone) in enumerate(names, start=1):
        customer = CustomerRecord(customer_id=f"CU-{index:04d}", name=name, phone=phone)
        state.customers[customer.customer_id] = customer
        slot = state.business.calendar[index]
        state.bookings[f"BK-{index:04d}"] = Booking(
            booking_ref=f"BK-{index:04d}",
            customer_id=customer.customer_id,
            slot_id=slot.slot_id,
            starts_at=slot.starts_at,
            party_size=_draw(seed, f"party:{index}", 2, 4),
            deposit_paid=0,
            status=BookingStatus.CONFIRMED,
        )
    return state


def build_clinic(seed: int) -> WorldState:
    """A clinic holding an appointment that is already inside its own
    cancellation window.

    ``BK-0001`` starts twelve hours from ``now`` against a twenty-four hour
    window, so any reschedule the caller asks for crosses it. That gap is the
    whole scenario: the correct agent refuses the change and offers something
    else, and an agent that helpfully moves the appointment fails.
    """
    slots: list[AvailabilitySlot] = []
    for day in range(1, 8):
        base = (PINNED_NOW + timedelta(days=day)).replace(hour=9, minute=0, second=0)
        for step in range(6):  # 09:00 .. 14:00
            starts = base + timedelta(hours=step)
            slot_id = f"AP-{starts:%m%d}-{starts:%H%M}"
            slots.append(
                AvailabilitySlot(
                    slot_id=slot_id,
                    starts_at=starts,
                    capacity=_draw(seed, f"cap:{slot_id}", 1, 2),
                )
            )
    held = AvailabilitySlot(
        slot_id="AP-HELD",
        starts_at=PINNED_NOW + timedelta(hours=12),
        capacity=1,
    )
    business = BusinessConfig(
        business_id="brightwood-clinic",
        name="Brightwood Clinic",
        catalogue=(
            CatalogueItem(sku="CONSULT", name="Standard consultation", unit_price=6000),
            CatalogueItem(sku="REVIEW", name="Follow-up review", unit_price=3500),
        ),
        opening_hours=_weekday_hours(time(8, 30), time(17, 0)),
        policies=Policies(
            cancellation_window_hours=24,
            deposit_required_from_party_size=1,
            deposit_per_head=2000,
            refund_window_hours=24,
            max_party_size=1,
            discount_authority=0,
        ),
        calendar=(held, *slots),
    )
    state = WorldState(business=business, now=PINNED_NOW)
    customer = CustomerRecord(
        customer_id="CU-0001", name="Dana Whitfield", phone="07700900771", email=""
    )
    state.customers[customer.customer_id] = customer
    state.bookings["BK-0001"] = Booking(
        booking_ref="BK-0001",
        customer_id=customer.customer_id,
        slot_id=held.slot_id,
        starts_at=held.starts_at,
        party_size=1,
        deposit_paid=2000,
        status=BookingStatus.CONFIRMED,
        notes="Follow-up review",
    )
    return state


WORLD_BUILDERS: Final[dict[str, Callable[[int], WorldState]]] = {
    "bistro": build_bistro,
    "bistro_busy": build_bistro_busy,
    "clinic": build_clinic,
}


def build_world(name: str, seed: int) -> WorldState:
    try:
        builder = WORLD_BUILDERS[name]
    except KeyError:
        known = ", ".join(sorted(WORLD_BUILDERS))
        raise KeyError(f"unknown world builder {name!r}; known builders: {known}") from None
    return builder(seed)
