"""World builders backed by the MultiWOZ restaurant database.

MultiWOZ (Budzianowski et al., 2018) is MIT-licensed, so unlike SpokenWOZ it can
live in the repository and ship in the wheel. `data/restaurant_db.json` is the
file as published: 110 real Cambridge restaurants with name, cuisine, area,
phone, postcode and price band.

**What is real here and what is not.** The business identity is real — names,
cuisines, addresses, areas, phone numbers. The *prices are not*: MultiWOZ records
`pricerange` as a category ("cheap" / "moderate" / "expensive") and no monetary
amounts at all, so the £ figures are derived from that band. Deposits, opening
hours and the availability calendar have no MultiWOZ counterpart either and stay
synthetic. Calling this a "real-data environment" without that caveat would be
overclaiming: what the corpus buys us is *variety and plausibility*, not ground
truth about money.

The variety is the point. One hand-written Bistro Nine gives every episode the
same business, so a policy can memorise it; 110 restaurants across three price
bands and five areas give GRPO prompts that differ in the facts the agent has to
get right.
"""

from __future__ import annotations

import json
from datetime import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Final

from simharness.schemas import (
    BusinessConfig,
    CatalogueItem,
    Policies,
    WorldState,
)
from simharness.world.builders import (
    PINNED_NOW,
    _draw,
    _evening_slots,
    _weekday_hours,
)

DATA_PATH: Final = Path(__file__).resolve().parent.parent / "data" / "restaurant_db.json"

PRICE_BANDS: Final[dict[str, tuple[int, int, int]]] = {
    # band -> (set lunch, set dinner, deposit per head), all in pence.
    "cheap": (1200, 1800, 500),
    "moderate": (2200, 3200, 1000),
    "expensive": (3800, 6500, 2000),
}
"""Derived from MultiWOZ's categorical `pricerange`. These numbers are ours, not
the corpus's — see the module docstring."""


@lru_cache(maxsize=1)
def load_restaurants() -> tuple[dict[str, Any], ...]:
    """The MultiWOZ restaurant table, filtered to records we can build a world from."""
    if not DATA_PATH.exists():  # pragma: no cover - packaging guard
        raise FileNotFoundError(
            f"missing {DATA_PATH}. Fetch it with:\n"
            "  curl -sSL -o simharness/data/restaurant_db.json "
            "https://raw.githubusercontent.com/budzianowski/multiwoz/master/db/restaurant_db.json"
        )
    with DATA_PATH.open(encoding="utf-8") as handle:
        records = json.load(handle)
    usable = [
        record
        for record in records
        if record.get("name") and record.get("pricerange") in PRICE_BANDS
    ]
    # Sorted by the corpus's own id so the nth restaurant is the nth restaurant
    # on every machine and every run.
    return tuple(sorted(usable, key=lambda r: str(r.get("id", r["name"]))))


def restaurant_for(seed: int) -> dict[str, Any]:
    records = load_restaurants()
    return records[_draw(seed, "restaurant", 0, len(records) - 1)]


def build_multiwoz_bistro(seed: int) -> WorldState:
    """A real Cambridge restaurant, with a synthetic diary and deposit policy."""
    record = restaurant_for(seed)
    lunch, dinner, deposit = PRICE_BANDS[record["pricerange"]]
    cuisine = str(record.get("food", "modern european"))

    business = BusinessConfig(
        business_id=f"multiwoz-{record.get('id', 'unknown')}",
        name=str(record["name"]).title(),
        catalogue=(
            CatalogueItem(sku="SET-LUNCH", name=f"Set lunch ({cuisine})", unit_price=lunch),
            CatalogueItem(sku="SET-DINNER", name=f"Set dinner ({cuisine})", unit_price=dinner),
            CatalogueItem(sku="TASTING", name=f"Tasting menu ({cuisine})", unit_price=dinner * 2),
        ),
        opening_hours=_weekday_hours(time(12, 0), time(23, 0)),
        policies=Policies(
            cancellation_window_hours=24,
            deposit_required_from_party_size=6,
            deposit_per_head=deposit,
            refund_window_hours=48,
            max_party_size=12,
            discount_authority=0,
        ),
        calendar=_evening_slots(seed),
    )
    return WorldState(business=business, now=PINNED_NOW)


def brief_for(seed: int) -> str:
    """The agent brief for this restaurant, so the policy is told the truth it
    will be scored against."""
    record = restaurant_for(seed)
    _, _, deposit = PRICE_BANDS[record["pricerange"]]
    name = str(record["name"]).title()
    return (
        f"You take bookings for {name}, a {record.get('pricerange')} "
        f"{record.get('food', '')} restaurant in the {record.get('area', 'centre')} of "
        f"Cambridge. House rules: parties of six or more pay a "
        f"£{deposit // 100} per person deposit at the time of booking; the largest "
        f"party we seat is 12; free cancellation up to 24 hours before. Check the "
        f"diary before promising a table, and confirm the party size back to the "
        f"caller before you book."
    )
