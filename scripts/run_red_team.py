"""Manual smoke test for the red-team caller and simulated client."""

from datetime import time

from simharness.schemas import (
    BusinessConfig,
    CatalogueItem,
    ClientBeliefs,
    OpeningHours,
    Policies,
)
from simharness.runner import run_red_team_episode


def main() -> None:
    business = BusinessConfig(
        business_id="bistro-nine",
        name="Bistro Nine",
        opening_hours=tuple(
            OpeningHours(weekday=d, opens=time(12, 0), closes=time(23, 0))
            for d in range(7)
        ),
        policies=Policies(
            cancellation_window_hours=24,
            deposit_required_from_party_size=6,
            deposit_per_head=1500,
            refund_window_hours=48,
            max_party_size=12,
            discount_authority=0,
        ),
        catalogue=(
            CatalogueItem(sku="SET-LUNCH", name="Set lunch", unit_price=2400),
            CatalogueItem(sku="SET-DINNER", name="Set dinner", unit_price=4500),
        ),
    )

    # The client believes the deposit is £20.00 instead of the ground-truth £15.00.
    client_beliefs = ClientBeliefs(
        facts={
            "deposit": "The deposit is £20.00 per person.",
        }
    )

    result = run_red_team_episode(
        business=business,
        client_beliefs=client_beliefs,
        max_turns=4,
    )

    print("Cracked:", result.cracked)
    print("Discrepancies:", result.casefile.discrepancies)
    print("Confirmed:", result.casefile.confirmed_facts)
    for turn in result.transcript:
        print(f"{turn.speaker.value}: {turn.text}")


if __name__ == "__main__":
    main()
