"""Token prices, so cost accounting is a fact rather than an estimate.

`CostSummary.price_table_id` is stamped on every run: a dollar figure produced
under one price table and compared against another is not a comparison. Prices
move, so the table carries its own date and the id changes when the numbers do.

USD per million tokens, Anthropic first-party API rates as of 2026-06-24.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

PRICE_TABLE_ID: Final = "anthropic-1p-2026-06-24"


@dataclass(frozen=True)
class ModelPrice:
    """Per-million-token prices, plus the cache multipliers.

    Cache reads bill at ~0.1x input and 5-minute writes at ~1.25x, which is why
    the agent brief and tool schemas are worth a breakpoint: they are identical
    on every turn of every episode.
    """

    input_per_mtok: float
    output_per_mtok: float
    cache_read_multiplier: float = 0.1
    cache_write_multiplier: float = 1.25

    def cost(
        self,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> float:
        million = 1_000_000
        return (
            input_tokens * self.input_per_mtok / million
            + output_tokens * self.output_per_mtok / million
            + cache_read_tokens * self.input_per_mtok * self.cache_read_multiplier / million
            + cache_write_tokens * self.input_per_mtok * self.cache_write_multiplier / million
        )


PRICES: Final[dict[str, ModelPrice]] = {
    "claude-opus-5": ModelPrice(5.00, 25.00),
    "claude-fable-5": ModelPrice(10.00, 50.00),
    "claude-opus-4-8": ModelPrice(5.00, 25.00),
    # Sonnet 5 carries an introductory rate through 2026-08-31 ($2/$10); the
    # standard rate is recorded here so a cost figure does not silently become
    # wrong the day the promotion ends.
    "claude-sonnet-5": ModelPrice(3.00, 15.00),
    "claude-sonnet-4-6": ModelPrice(3.00, 15.00),
    "claude-haiku-4-5": ModelPrice(1.00, 5.00),
}


def price_for(model: str) -> ModelPrice | None:
    """None rather than a guess: an unpriced model reports $0 and says so via
    ``price_table_id``, which is honest. A guessed price is not."""
    return PRICES.get(model)


def estimate(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float:
    price = price_for(model)
    if price is None:
        return 0.0
    return price.cost(input_tokens, output_tokens, cache_read_tokens, cache_write_tokens)
