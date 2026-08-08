"""Client for the Context.dev API.

Brand and web extraction endpoints are exposed as plain methods so any module
can enrich a business fact sheet without importing an SDK.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

__all__ = ["ContextDevClient"]


class ContextDevClient:
    """Thin wrapper over the Context.dev v1 HTTP API."""

    BASE_URL = "https://api.context.dev/v1"

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.environ.get("CONTEXT_DEV_API_KEY")
        if not self._api_key:
            raise RuntimeError("CONTEXT_DEV_API_KEY is not set")
        self._client = httpx.Client(
            base_url=self.BASE_URL,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            timeout=120.0,
        )

    def brand_retrieve(self, *, domain: str) -> dict[str, Any]:
        """Resolve a domain into brand metadata (logos, description, industry, etc.)."""
        resp = self._client.post(
            "/brand/retrieve",
            json={"type": "by_domain", "domain": domain},
        )
        return _unwrap(resp)

    def extract_hotel_facts(self, *, url: str, fact_check: bool = False) -> dict[str, Any]:
        """Crawl a hotel page and extract a structured fact sheet."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "description": {"type": "string"},
                "location": {"type": "string"},
                "price_per_night": {"type": ["string", "null"]},
                "currency": {"type": ["string", "null"]},
                "deposit_policy": {"type": ["string", "null"]},
                "cancellation_policy": {"type": ["string", "null"]},
                "check_in_time": {"type": ["string", "null"]},
                "check_out_time": {"type": ["string", "null"]},
                "amenities": {
                    "type": ["array", "null"],
                    "items": {"type": "string"},
                },
            },
            "required": ["name", "location"],
            "additionalProperties": False,
        }

        resp = self._client.post(
            "/web/extract",
            json={
                "url": url,
                "schema": schema,
                "instructions": (
                    "Extract the hotel's public pricing, deposit, and cancellation "
                    "policies as they are stated on the page. If a value is not "
                    "explicitly present, return null."
                ),
                "factCheck": fact_check,
                "maxDepth": 0,
                "maxPages": 1,
            },
        )
        return _unwrap(resp)


def _unwrap(resp: httpx.Response) -> dict[str, Any]:
    if resp.status_code >= 400:
        raise RuntimeError(f"Context.dev error {resp.status_code}: {resp.text}")
    return resp.json()
