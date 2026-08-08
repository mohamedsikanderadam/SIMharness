"""Assembling the fact sheet the agent is graded against.

The audit is only ever as good as the truth behind it, so provenance is tracked
per fact rather than per sheet. Two sources, in strict precedence:

1. the business's own :class:`BusinessConfig` — authoritative, and never
   overwritten;
2. Context.dev, scraped from the business's public web presence — used *only*
   to fill keys the business left empty.

That ordering is the whole design. A scrape that disagreed with the config would
be the scrape being stale, not the business being wrong, so it must not win. And
a scraped fact is still worth auditing against: an agent contradicting the
business's own published website is a finding either way — but it is marked
:attr:`FactSource.CONTEXT_DEV` so an owner disputing it knows where to look.

Nothing here calls the network unless a provider is passed in. The default path
is offline and deterministic.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, ClassVar, Protocol

from simharness.reporting.schemas import BusinessFact, FactSheet, FactSource
from simharness.schemas import BusinessConfig

__all__ = [
    "ContextDevFactProvider",
    "FactProvider",
    "build_fact_sheet",
    "facts_from_business_config",
]

_WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


class FactProvider(Protocol):
    """Anything that can supply public facts for a business.

    One method, so the Context.dev client, a cached JSON blob, and a test double
    are interchangeable and none of the grading code imports an HTTP library.
    """

    def public_facts(self, *, business_id: str) -> dict[str, str]: ...


def build_fact_sheet(
    config: BusinessConfig,
    *,
    provider: FactProvider | None = None,
    required_keys: tuple[str, ...] = (),
) -> FactSheet:
    """Derive facts from ``config``, then ask ``provider`` for what is missing.

    ``required_keys`` names facts the audit needs regardless of whether the
    config models them — ``check_in_time`` for a hotel, say. A required key that
    neither source supplies is still emitted, with source
    :attr:`FactSource.ABSENT`, so the gaps section can name it instead of the
    report quietly not checking it.
    """
    facts: dict[str, BusinessFact] = {f.key: f for f in facts_from_business_config(config)}
    scraped_at: datetime | None = None

    wanted = set(required_keys) | set(facts)
    missing = {key for key in wanted if key not in facts or not facts[key].value}

    if provider is not None and missing:
        supplied = provider.public_facts(business_id=config.business_id)
        scraped_at = datetime.now(UTC)
        for key in sorted(missing):
            value = supplied.get(key, "").strip()
            if not value:
                continue
            facts[key] = BusinessFact(
                key=key,
                label=_label_for(key),
                value=value,
                currency=config.catalogue[0].currency if config.catalogue else "",
                source=FactSource.CONTEXT_DEV,
                source_detail="context.dev",
                aliases=_aliases_for(key),
            )

    for key in sorted(wanted - set(facts)):
        facts[key] = BusinessFact(
            key=key, label=_label_for(key), value="", source=FactSource.ABSENT
        )

    return FactSheet(
        business_id=config.business_id,
        business_name=config.name,
        currency=config.catalogue[0].currency if config.catalogue else "",
        timezone=config.timezone,
        facts=tuple(facts[key] for key in sorted(facts)),
        scraped_at=scraped_at,
    )


def facts_from_business_config(config: BusinessConfig) -> tuple[BusinessFact, ...]:
    """Flatten a :class:`BusinessConfig` into checkable statements.

    Money keeps its ``minor_units`` alongside the display string so a later
    comparison is integer-exact. Comparing the rendered ``"AED 250.00"`` against
    an agent that said ``"250 dirhams"`` would miss; comparing 25000 does not.
    """
    currency = config.catalogue[0].currency if config.catalogue else ""
    policies = config.policies
    facts: list[BusinessFact] = []

    for item in config.catalogue:
        facts.append(
            BusinessFact(
                key=f"price:{item.sku}",
                label=f"Price — {item.name}",
                value=_money(item.unit_price, item.currency),
                minor_units=item.unit_price,
                currency=item.currency,
                source=FactSource.BUSINESS_CONFIG,
                source_detail="catalogue",
                aliases=(item.name.lower(), item.sku.lower()),
            )
        )

    open_days = [h for h in config.opening_hours if not h.closed]
    if open_days:
        facts.append(
            BusinessFact(
                key="opening_hours",
                label="Opening hours",
                value=_hours_summary(config.opening_hours),
                source=FactSource.BUSINESS_CONFIG,
                source_detail="opening_hours",
                aliases=("opening hours", "what time", "open", "close", "closing"),
            )
        )

    closed_days = [_WEEKDAYS[h.weekday] for h in config.opening_hours if h.closed]
    if closed_days:
        facts.append(
            BusinessFact(
                key="closed_days",
                label="Closed days",
                value=", ".join(closed_days),
                source=FactSource.BUSINESS_CONFIG,
                source_detail="opening_hours",
                aliases=("closed", "shut"),
            )
        )

    facts.append(
        BusinessFact(
            key="cancellation_window",
            label="Cancellation window",
            value=f"{policies.cancellation_window_hours} hours",
            source=FactSource.BUSINESS_CONFIG,
            source_detail="policies",
            aliases=("cancel", "cancellation"),
        )
    )
    facts.append(
        BusinessFact(
            key="refund_window",
            label="Refund window",
            value=f"{policies.refund_window_hours} hours",
            source=FactSource.BUSINESS_CONFIG,
            source_detail="policies",
            aliases=("refund", "money back"),
        )
    )
    facts.append(
        BusinessFact(
            key="deposit",
            label="Deposit per head",
            value=_money(policies.deposit_per_head, currency),
            minor_units=policies.deposit_per_head,
            currency=currency,
            source=FactSource.BUSINESS_CONFIG,
            source_detail="policies",
            aliases=("deposit", "upfront", "hold"),
        )
    )
    facts.append(
        BusinessFact(
            key="max_party_size",
            label="Maximum party size",
            value=str(policies.max_party_size),
            source=FactSource.BUSINESS_CONFIG,
            source_detail="policies",
            aliases=("party size", "how many people", "group size", "guests"),
        )
    )
    facts.append(
        BusinessFact(
            key="discount_authority",
            label="Discount the agent may grant",
            value=_money(policies.discount_authority, currency),
            minor_units=policies.discount_authority,
            currency=currency,
            source=FactSource.BUSINESS_CONFIG,
            source_detail="policies",
            aliases=("discount", "off", "reduction"),
        )
    )
    return tuple(facts)


class ContextDevFactProvider:
    """Fills gaps from Context.dev, mapping its output onto our fact keys.

    The client is constructed lazily and the import is local, so importing the
    reporting package neither needs an API key nor pulls in ``httpx``.
    """

    #: Context.dev extraction field -> our fact key.
    FIELD_MAP: ClassVar[dict[str, str]] = {
        "cancellation_policy": "cancellation_window",
        "deposit_policy": "deposit",
        "check_in_time": "check_in_time",
        "check_out_time": "check_out_time",
        "price_per_night": "price:ROOM",
        "location": "location",
    }

    def __init__(self, url: str, *, api_key: str | None = None, fact_check: bool = True) -> None:
        self._url = url
        self._api_key = api_key
        self._fact_check = fact_check

    def public_facts(self, *, business_id: str) -> dict[str, str]:
        from simharness.adapters.contextdev_client import ContextDevClient

        client = ContextDevClient(api_key=self._api_key)
        extracted = client.extract_hotel_facts(url=self._url, fact_check=self._fact_check)
        return self._map(extracted)

    def _map(self, extracted: dict[str, Any]) -> dict[str, str]:
        payload = extracted.get("data") if isinstance(extracted.get("data"), dict) else extracted
        if not isinstance(payload, dict):
            return {}
        facts: dict[str, str] = {}
        for field, key in self.FIELD_MAP.items():
            value = payload.get(field)
            if isinstance(value, str) and value.strip():
                facts[key] = value.strip()
        return facts


def _money(minor_units: int, currency: str) -> str:
    amount = f"{minor_units / 100:.2f}"
    return f"{currency} {amount}".strip() if currency else amount


def _hours_summary(hours: tuple[Any, ...]) -> str:
    """Collapse a week of hours into one line when every open day matches.

    Most businesses keep one schedule, and "09:00 to 17:00" is what the agent
    would actually say. Only when days genuinely differ is the per-day form worth
    the noise.
    """
    open_days = [h for h in hours if not h.closed]
    if not open_days:
        return "closed"
    first = open_days[0]
    if all(h.opens == first.opens and h.closes == first.closes for h in open_days):
        return f"{first.opens:%H:%M} to {first.closes:%H:%M}"
    return "; ".join(f"{_WEEKDAYS[h.weekday]} {h.opens:%H:%M}-{h.closes:%H:%M}" for h in open_days)


_LABELS: dict[str, str] = {
    "cancellation_window": "Cancellation window",
    "deposit": "Deposit",
    "check_in_time": "Check-in time",
    "check_out_time": "Check-out time",
    "location": "Location",
    "opening_hours": "Opening hours",
    "refund_window": "Refund window",
    "max_party_size": "Maximum party size",
    "discount_authority": "Discount the agent may grant",
}

_ALIASES: dict[str, tuple[str, ...]] = {
    "check_in_time": ("check in", "check-in", "checkin", "arrive"),
    "check_out_time": ("check out", "check-out", "checkout", "leave by"),
    "location": ("where are you", "address", "located"),
}


def _label_for(key: str) -> str:
    if key in _LABELS:
        return _LABELS[key]
    if key.startswith("price:"):
        return f"Price — {key.split(':', 1)[1]}"
    return key.replace("_", " ").capitalize()


def _aliases_for(key: str) -> tuple[str, ...]:
    if key in _ALIASES:
        return _ALIASES[key]
    if key.startswith("price:"):
        return (key.split(":", 1)[1].lower(), "price", "cost", "rate")
    return (key.replace("_", " "),)
