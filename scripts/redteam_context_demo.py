"""Red-team demo that uses Context.dev to ground-truth both the judge and the judged.

The public Context.dev facts become the red team's ground truth (BusinessConfig).
The same facts are turned into the client's private beliefs (ClientBeliefs), with
one field optionally altered to simulate a wrong or outdated internal policy.

This keeps the red team blind to the client's actual knowledge, exactly as
FIRST_PRINCIPLES.md requires.
"""

from __future__ import annotations

import argparse
import os
import re
from datetime import datetime, time
from pathlib import Path

from simharness.adapters.contextdev_client import ContextDevClient
from simharness.runner import run_red_team_episode
from simharness.schemas import (
    BusinessConfig,
    CatalogueItem,
    ClientBeliefs,
    OpeningHours,
    Policies,
)
from simharness.simulator.adaptive_redteam import AdaptiveRedTeamSimulator


def _load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key, value)


def _parse_amount(text: str | None, default: int = 0) -> int:
    if not text:
        return default
    match = re.search(r"[\d,]+(?:\.\d+)?", text)
    if not match:
        return default
    value = float(match.group(0).replace(",", ""))
    return int(value * 100)


def _parse_percentage(text: str | None) -> float:
    if not text:
        return 0.0
    match = re.search(r"[\d,]+(?:\.\d+)?", text)
    if not match:
        return 0.0
    return float(match.group(0).replace(",", ""))


def _parse_hours(text: str | None, default: int = 24) -> int:
    if not text:
        return default
    match = re.search(r"[\d,]+(?:\.\d+)?", text)
    if not match:
        return default
    value = float(match.group(0).replace(",", ""))
    if any(u in text.lower() for u in ("day", "days")):
        value *= 24
    return int(value)


def _parse_time(text: str | None, default: time) -> time:
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
    brand = client.brand_retrieve(domain=domain)
    facts = client.extract_hotel_facts(url=url)

    print("=== Context.dev brand ===")
    print(brand.get("brand", {}).get("title", domain))

    print("\n=== Context.dev facts ===")
    for key, value in facts.get("data", {}).items():
        print(f"  {key}: {value}")

    price = _parse_amount(facts.get("data", {}).get("price_per_night"), default=20000)
    if price == 20000:
        print("  (price_per_night missing; using default £200.00)")

    deposit_text = facts.get("data", {}).get("deposit_policy") or ""
    if "%" in deposit_text:
        deposit_per_head = int(price * _parse_percentage(deposit_text) / 100)
    else:
        deposit_per_head = _parse_amount(deposit_text, default=int(price * 0.2))
        if deposit_per_head == int(price * 0.2):
            print("  (deposit_policy missing; using default 20% of price)")

    cancellation_hours = _parse_hours(
        facts.get("data", {}).get("cancellation_policy"), default=24
    )

    check_in = _parse_time(
        facts.get("data", {}).get("check_in_time"), default=time(15, 0)
    )
    check_out = _parse_time(
        facts.get("data", {}).get("check_out_time"), default=time(11, 0)
    )

    return BusinessConfig(
        business_id=domain.replace(".", "-"),
        name=facts.get("data", {}).get("name")
        or brand.get("brand", {}).get("title")
        or "Demo Hotel",
        timezone="Europe/London",
        catalogue=(CatalogueItem(sku="ROOM", name="Room", unit_price=price),),
        opening_hours=(OpeningHours(weekday=0, opens=check_in, closes=check_out),),
        policies=Policies(
            cancellation_window_hours=cancellation_hours,
            deposit_required_from_party_size=1,
            deposit_per_head=deposit_per_head,
            refund_window_hours=cancellation_hours,
            max_party_size=4,
            discount_authority=0,
        ),
    )


def _client_beliefs_from_business(
    business: BusinessConfig,
    lie_field: str | None = "set lunch",
    lie_value: str | None = None,
) -> ClientBeliefs:
    facts: dict[str, str] = {}

    deposit = business.policies.deposit_per_head / 100
    facts["deposit"] = f"The deposit is £{deposit:.2f} per person."

    if business.catalogue:
        price = business.catalogue[0].unit_price / 100
        facts["set lunch"] = f"The set lunch is £{price:.2f}."

    if business.opening_hours:
        h = business.opening_hours[0]
        facts["opening hours"] = (
            f"Our opening hours are {h.opens:%H:%M} to {h.closes:%H:%M}."
        )

    facts["cancellation"] = (
        f"The cancellation window is {business.policies.cancellation_window_hours} hours."
    )

    facts["party"] = (
        f"The party size we can seat is {business.policies.max_party_size}."
    )

    aliases: dict[str, str] = {
        "price": "set lunch",
        "catalogue": "set lunch",
        "hours": "opening hours",
        "party_size": "party",
        "capacity": "party",
    }
    field = aliases.get(lie_field, lie_field) if lie_field else None

    if field and field in facts:
        if lie_value:
            facts[field] = lie_value
        else:
            if field == "deposit":
                facts[field] = f"The deposit is £{deposit * 1.5:.2f} per person."
            elif field == "set lunch" and business.catalogue:
                price = business.catalogue[0].unit_price / 100
                facts[field] = f"The set lunch is £{price * 1.5:.2f}."
            elif field == "opening hours" and business.opening_hours:
                h = business.opening_hours[0]
                # Shift later to avoid matching the true value.
                facts[field] = "Our opening hours are 09:00 to 21:00."
            elif field == "cancellation":
                hours = business.policies.cancellation_window_hours
                facts[field] = f"The cancellation window is {hours * 2} hours."
            elif field == "party":
                size = business.policies.max_party_size
                facts[field] = f"The party size we can seat is {size + 3}."

    return ClientBeliefs(facts=facts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Context.dev red-team demo.")
    parser.add_argument(
        "--url",
        default="https://www.atlantis.com/dubai/atlantis-the-palm",
        help="Hotel or business page to extract public facts from.",
    )
    parser.add_argument(
        "--domain",
        default="atlantis.com",
        help="Domain for brand lookup.",
    )
    parser.add_argument(
        "--lie-field",
        default="set lunch",
        choices=["deposit", "set lunch", "opening hours", "cancellation", "party"],
        help="Which fact the client's internal knowledge gets wrong.",
    )
    parser.add_argument(
        "--lie-value",
        default=None,
        help="Override the lie with a full sentence.",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=10,
        help="Max red-team turns.",
    )
    parser.add_argument(
        "--adaptive",
        action="store_true",
        help="Use the adaptive (claim-extraction) red team instead of the scripted one.",
    )
    args = parser.parse_args()

    _load_env(Path(".env"))
    _load_env(Path("secrets.env"))

    business = _extract_business(args.url, args.domain)
    client_beliefs = _client_beliefs_from_business(
        business, lie_field=args.lie_field, lie_value=args.lie_value
    )

    print("\n=== Red-team ground truth (BusinessConfig) ===")
    print(business.model_dump_json(indent=2))
    print("\n=== Client private beliefs (ClientBeliefs) ===")
    print(client_beliefs.model_dump_json(indent=2))

    red_team = None
    if args.adaptive:
        red_team = AdaptiveRedTeamSimulator(ground_truth=business, max_turns=args.max_turns)

    result = run_red_team_episode(
        business=business,
        client_beliefs=client_beliefs,
        max_turns=args.max_turns,
        red_team=red_team,
    )

    print(f"\nCracked: {result.cracked}")
    print(f"Discrepancies: {result.casefile.discrepancies}")
    print(f"Confirmed: {result.casefile.confirmed_facts}")
    for turn in result.transcript:
        print(f"{turn.speaker.value}: {turn.text}")


if __name__ == "__main__":
    main()
