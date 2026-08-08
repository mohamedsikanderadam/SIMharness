"""Red-team demo grounded in a Context.dev extraction of a real business.

The extracted facts become the red team's ground truth. The same facts are
restated as the client's beliefs with one field corrupted, so the red team is
looking for something it was not told the location of.
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime, time

from scripts._demo import LIE_FIELDS, client_beliefs_with_lie, load_env
from simharness.adapters.contextdev_client import ContextDevClient
from simharness.runner import run_red_team_episode
from simharness.schemas import (
    BusinessConfig,
    CatalogueItem,
    OpeningHours,
    Policies,
)

#: Used when the page states no rate at all, so the demo still has something to
#: probe. Printed when it applies — a fabricated fact sheet that looks extracted
#: is worse than an obviously empty one.
_DEFAULT_PRICE = 20000


def _amount(text: str | None, default: int) -> int:
    if not text or not (match := re.search(r"[\d,]+(?:\.\d+)?", text)):
        return default
    return int(float(match.group().replace(",", "")) * 100)


def _hours(text: str | None, default: int) -> int:
    if not text or not (match := re.search(r"[\d,]+(?:\.\d+)?", text)):
        return default
    value = float(match.group().replace(",", ""))
    return int(value * 24 if "day" in text.lower() else value)


def _clock(text: str | None, default: time) -> time:
    if not text:
        return default
    for fmt in ("%I:%M %p", "%I:%M%p", "%I %p", "%H:%M", "%H:%M:%S"):
        try:
            return datetime.strptime(text.strip(), fmt).time()
        except ValueError:
            continue
    return default


def _extract_business(url: str, domain: str) -> BusinessConfig:
    client = ContextDevClient()
    brand = client.brand_retrieve(domain=domain).get("brand", {})
    data = client.extract_hotel_facts(url=url).get("data", {})

    print(f"=== Context.dev brand: {brand.get('title', domain)} ===")
    print("\n=== Context.dev facts ===")
    for key, value in data.items():
        print(f"  {key}: {value}")

    price = _amount(data.get("price_per_night"), default=_DEFAULT_PRICE)
    if data.get("price_per_night") is None:
        print(f"  (no price on the page; assuming £{_DEFAULT_PRICE / 100:.2f})")

    deposit_text = data.get("deposit_policy") or ""
    if "%" in deposit_text:
        match = re.search(r"[\d.]+", deposit_text)
        deposit = int(price * float(match.group()) / 100) if match else int(price * 0.2)
    else:
        deposit = _amount(deposit_text, default=int(price * 0.2))
        if not deposit_text:
            print("  (no deposit on the page; assuming 20% of the rate)")

    cancellation = _hours(data.get("cancellation_policy"), default=24)

    return BusinessConfig(
        business_id=domain.replace(".", "-"),
        name=data.get("name") or brand.get("title") or "Demo Hotel",
        timezone="Europe/London",
        catalogue=(CatalogueItem(sku="ROOM", name="Room", unit_price=price),),
        opening_hours=(
            OpeningHours(
                weekday=0,
                opens=_clock(data.get("check_in_time"), time(15, 0)),
                closes=_clock(data.get("check_out_time"), time(11, 0)),
            ),
        ),
        policies=Policies(
            cancellation_window_hours=cancellation,
            deposit_required_from_party_size=1,
            deposit_per_head=deposit,
            refund_window_hours=cancellation,
            max_party_size=4,
            discount_authority=0,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Context.dev red-team demo.")
    parser.add_argument(
        "--url",
        default="https://www.atlantis.com/dubai/atlantis-the-palm",
        help="Page to extract public facts from.",
    )
    parser.add_argument("--domain", default="atlantis.com", help="Domain for brand lookup.")
    parser.add_argument(
        "--lie-field",
        default="set lunch",
        choices=LIE_FIELDS,
        help="Which fact the client gets wrong.",
    )
    parser.add_argument("--lie-value", default=None, help="Override the lie with a sentence.")
    parser.add_argument("--max-turns", type=int, default=10, help="Max red-team turns.")
    args = parser.parse_args()

    load_env()

    business = _extract_business(args.url, args.domain)
    beliefs = client_beliefs_with_lie(business, args.lie_field, args.lie_value)

    print("\n=== Red-team ground truth ===")
    print(business.model_dump_json(indent=2))
    print("\n=== Client beliefs (one field corrupted) ===")
    print(beliefs.model_dump_json(indent=2))

    result = run_red_team_episode(
        business=business, client_beliefs=beliefs, max_turns=args.max_turns
    )

    print(f"\nCracked: {result.cracked}")
    print(f"Discrepancies: {result.casefile.discrepancies}")
    print(f"Confirmed: {result.casefile.confirmed_facts}")
    for turn in result.transcript:
        print(f"{turn.speaker.value}: {turn.text}")


if __name__ == "__main__":
    main()
