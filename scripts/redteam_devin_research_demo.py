"""Red-team demo where a Devin session does the reconnaissance.

Devin is given a URL, reads the site, and returns a structured fact sheet. That
becomes the red team's ground truth; the client then gets the same facts with
one field corrupted.

Devin reads past the landing page, so it recovers terms-page facts (cancellation
windows, check-in times) that a single-page extraction returns as null.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime
from datetime import time as clock
from typing import Any, cast

import httpx

from scripts._demo import LIE_FIELDS, client_beliefs_with_lie, load_env
from simharness.runner import run_red_team_episode
from simharness.schemas import BusinessConfig, CatalogueItem, OpeningHours, Policies

BASE_URL = "https://api.devin.ai/v3"

_PROMPT = """Research this URL: {url}

Extract the business / hotel facts and save them to a file called
`business_facts.json` in the session workspace. Use exactly this JSON shape:

{{
  "name": "...",
  "location": "...",
  "price_per_night_gbp_pence": 20000,
  "deposit_per_head_gbp_pence": 4000,
  "cancellation_window_hours": 24,
  "check_in_time": "15:00",
  "check_out_time": "11:00",
  "max_party_size": 4,
  "amenities": ["..."]
}}

All money values must be in British pence (so £200.00 is 20000).
Times must be HH:MM in 24-hour format.
If the first page does not state price, deposit, cancellation, check-in/out or
party size, search the same site for a booking flow, terms and conditions, or
FAQ page that does. Convert AED/USD to GBP using a reasonable exchange rate.
Do not ask me for clarification. Save the JSON file, print its contents, and
finish.
"""


class DevinClient:
    """Thin wrapper over the Devin v3 session endpoints used by this demo."""

    def __init__(self, api_key: str, org_id: str | None = None) -> None:
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._org_id = org_id or self._get("/self")["org_id"]

    def _get(self, path: str) -> Any:
        resp = httpx.get(f"{BASE_URL}{path}", headers=self._headers, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def _org(self, path: str) -> Any:
        return self._get(f"/organizations/{self._org_id}{path}")

    def start(self, prompt: str, devin_mode: str) -> str:
        resp = httpx.post(
            f"{BASE_URL}/organizations/{self._org_id}/sessions",
            headers={**self._headers, "Content-Type": "application/json"},
            json={"prompt": prompt, "devin_mode": devin_mode},
            timeout=60,
        )
        resp.raise_for_status()
        session = resp.json()
        print(f"  Session: {session['url']}", flush=True)
        return cast(str, session["session_id"])

    def wait(self, session_id: str, *, interval: int = 10, timeout: int = 600) -> None:
        """Block until the session finishes or turns the question back on us.

        ``waiting_for_user`` is a terminal state for our purposes: nobody is
        going to answer, and Devin has usually already produced the artefact.
        """
        for _ in range(timeout // interval):
            session = self._org(f"/sessions/{session_id}")
            status, detail = session.get("status"), session.get("status_detail", "")
            print(f"  Devin status: {status} ({detail})", flush=True)
            if status in ("exit", "error", "suspended") or detail == "waiting_for_user":
                return
            time.sleep(interval)
        raise TimeoutError(f"Devin session {session_id} did not finish within {timeout}s.")

    def attachment(self, session_id: str, name: str) -> bytes | None:
        for item in self._org(f"/sessions/{session_id}/attachments"):
            if item.get("name") != name:
                continue
            # The download endpoint 307s to a presigned S3 URL. Follow it by
            # hand: forwarding the bearer token to S3 breaks the signature.
            url = (
                f"{BASE_URL}/organizations/{self._org_id}"
                f"/attachments/{item['attachment_id']}/{name}"
            )
            resp = httpx.get(url, headers=self._headers, timeout=60)
            if resp.is_redirect:
                return httpx.get(resp.headers["location"], timeout=60).content
            resp.raise_for_status()
            return resp.content
        return None

    def last_json_block(self, session_id: str) -> dict[str, Any] | None:
        """Fall back to the transcript when no artefact was attached."""
        for message in reversed(self._org(f"/sessions/{session_id}/messages")["items"]):
            if message.get("source") != "devin":
                continue
            match = re.search(r"```(?:json)?\n(.*?)\n```", message.get("message", ""), re.DOTALL)
            if match:
                try:
                    return cast(dict[str, Any], json.loads(match.group(1)))
                except json.JSONDecodeError:
                    continue
        return None


def _research(
    client: DevinClient, url: str, devin_mode: str, session_id: str | None
) -> dict[str, Any]:
    if session_id:
        print(f"Reusing Devin session {session_id}...", flush=True)
    else:
        print(f"Starting Devin research session for {url}...", flush=True)
        session_id = client.start(_PROMPT.format(url=url), devin_mode)

    client.wait(session_id)

    if data := client.attachment(session_id, "business_facts.json"):
        return cast(dict[str, Any], json.loads(data))
    if facts := client.last_json_block(session_id):
        return facts
    raise RuntimeError(f"Devin produced no fact sheet. Inspect session {session_id}.")


def _clock(text: str | None, default: clock) -> clock:
    if not text:
        return default
    for fmt in ("%H:%M", "%I:%M %p", "%I:%M%p", "%I %p"):
        try:
            return datetime.strptime(text.strip(), fmt).time()
        except ValueError:
            continue
    return default


def _business_from(facts: dict[str, Any]) -> BusinessConfig:
    price = int(facts.get("price_per_night_gbp_pence") or 20000)
    cancellation = int(facts.get("cancellation_window_hours") or 24)

    return BusinessConfig(
        business_id="devin-research",
        name=facts.get("name") or "Devin Research Business",
        timezone="Europe/London",
        catalogue=(CatalogueItem(sku="ROOM", name="Room", unit_price=price),),
        opening_hours=(
            OpeningHours(
                weekday=0,
                opens=_clock(facts.get("check_in_time"), clock(15, 0)),
                closes=_clock(facts.get("check_out_time"), clock(11, 0)),
            ),
        ),
        policies=Policies(
            cancellation_window_hours=cancellation,
            deposit_required_from_party_size=1,
            deposit_per_head=int(facts.get("deposit_per_head_gbp_pence") or int(price * 0.2)),
            refund_window_hours=cancellation,
            max_party_size=int(facts.get("max_party_size") or 4),
            discount_authority=0,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Devin research red-team demo.")
    parser.add_argument(
        "--url",
        default="https://www.atlantis.com/dubai/atlantis-the-palm",
        help="Page for Devin to research.",
    )
    parser.add_argument(
        "--lie-field",
        default="set lunch",
        choices=LIE_FIELDS,
        help="Which fact the client gets wrong.",
    )
    parser.add_argument("--lie-value", default=None, help="Override the lie with a sentence.")
    parser.add_argument("--max-turns", type=int, default=10, help="Max red-team turns.")
    parser.add_argument(
        "--devin-mode", default="fast", choices=["fast", "lite"], help="Devin session mode."
    )
    parser.add_argument(
        "--session-id",
        default=None,
        help="Reuse a finished Devin session instead of spending on a new one.",
    )
    args = parser.parse_args()

    load_env()

    api_key = os.environ.get("DEVIN_API_KEY")
    if not api_key:
        raise RuntimeError("DEVIN_API_KEY is not set. Put it in secrets.env or the environment.")

    client = DevinClient(api_key, os.environ.get("DEVIN_ORG_ID"))
    facts = _research(client, args.url, args.devin_mode, args.session_id)

    print("\n=== Devin extracted facts ===")
    print(json.dumps(facts, indent=2))

    business = _business_from(facts)
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
