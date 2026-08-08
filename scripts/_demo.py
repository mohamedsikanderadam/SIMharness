"""Shared plumbing for the red-team demo scripts.

Each demo sources its ground truth differently — a Context.dev extraction, a
Devin research session, a literal — but they all then need the same two things:
credentials off disk, and a client whose knowledge is right except in one place.
"""

from __future__ import annotations

import os
from pathlib import Path

from simharness.schemas import BusinessConfig, ClientBeliefs

__all__ = ["LIE_FIELDS", "client_beliefs_with_lie", "load_env"]

#: Fields a demo can corrupt. Keys match the Analyst's target fields, except
#: "party", which is the client's own wording for "party size".
LIE_FIELDS = ("deposit", "set lunch", "opening hours", "cancellation", "party")


def load_env(*paths: str) -> None:
    """Read ``KEY=value`` lines into the environment without overriding it.

    Real environment variables win, so a demo can be pointed somewhere else for
    one run without editing the file.
    """
    for path in paths or (".env", "secrets.env"):
        file = Path(path)
        if not file.exists():
            continue
        for line in file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key, value)


def client_beliefs_with_lie(
    business: BusinessConfig,
    lie_field: str | None = None,
    lie_value: str | None = None,
) -> ClientBeliefs:
    """Restate the fact sheet as client beliefs, corrupting at most one field.

    Everything else is true, so a red team that reports more than one
    discrepancy is reporting a false positive.
    """
    deposit = business.policies.deposit_per_head / 100
    cancellation = business.policies.cancellation_window_hours
    party = business.policies.max_party_size

    facts = {
        "deposit": f"The deposit is £{deposit:.2f} per person.",
        "cancellation": f"The cancellation window is {cancellation} hours.",
        "party": f"The party size we can seat is {party}.",
    }
    if business.catalogue:
        facts["set lunch"] = f"The set lunch is £{business.catalogue[0].unit_price / 100:.2f}."
    if business.opening_hours:
        hours = business.opening_hours[0]
        facts["opening hours"] = (
            f"Our opening hours are {hours.opens:%H:%M} to {hours.closes:%H:%M}."
        )

    if lie_field is None or lie_field not in facts:
        return ClientBeliefs(facts=facts)

    if lie_value is not None:
        facts[lie_field] = lie_value
    elif lie_field == "deposit":
        facts[lie_field] = f"The deposit is £{deposit * 1.5:.2f} per person."
    elif lie_field == "set lunch":
        facts[lie_field] = f"The set lunch is £{business.catalogue[0].unit_price / 100 * 1.5:.2f}."
    elif lie_field == "opening hours":
        facts[lie_field] = "Our opening hours are 09:00 to 21:00."
    elif lie_field == "cancellation":
        facts[lie_field] = f"The cancellation window is {cancellation * 2} hours."
    elif lie_field == "party":
        facts[lie_field] = f"The party size we can seat is {party + 3}."

    return ClientBeliefs(facts=facts)
