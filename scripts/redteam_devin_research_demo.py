"""Red-team demo where Devin does the reconnaissance.

Devin is given a public URL and asked to extract a structured business fact
sheet. That fact sheet becomes the red team's ground truth. The script then
injects one lie into the client and runs the red team to see if it exposes it.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime
from datetime import time as dt_time
from pathlib import Path
from typing import Any, cast

import httpx

from simharness.runner import run_red_team_episode
from simharness.schemas import (
    BusinessConfig,
    CatalogueItem,
    ClientBeliefs,
    OpeningHours,
    Policies,
)
from simharness.simulator.adaptive_redteam import AdaptiveRedTeamSimulator
from simharness.simulator.redteam import RedTeamSimulator

BASE_URL = "https://api.devin.ai/v3"


def _load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key, value)


def _devin_org_id(api_key: str) -> str:
    """Look up the org_id for the service user if it is not in the environment."""
    resp = httpx.get(
        f"{BASE_URL}/self",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30,
    )
    resp.raise_for_status()
    return cast(str, resp.json()["org_id"])


def _create_devin_session(
    api_key: str,
    org_id: str,
    prompt: str,
    *,
    devin_mode: str = "fast",
) -> dict[str, Any]:
    """Start a Devin session and return the session object."""
    resp = httpx.post(
        f"{BASE_URL}/organizations/{org_id}/sessions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "prompt": prompt,
            "devin_mode": devin_mode,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return cast(dict[str, Any], resp.json())


def _poll_devin_session(
    api_key: str,
    org_id: str,
    session_id: str,
    *,
    poll_interval: int = 10,
    max_wait: int = 600,
) -> dict[str, Any]:
    """Poll the session until it exits, errors, suspends, or asks for input."""
    headers = {"Authorization": f"Bearer {api_key}"}
    for _ in range(max_wait // poll_interval):
        resp = httpx.get(
            f"{BASE_URL}/organizations/{org_id}/sessions/{session_id}",
            headers=headers,
            timeout=30,
        )
        resp.raise_for_status()
        data = cast(dict[str, Any], resp.json())
        status = data.get("status")
        detail = data.get("status_detail", "")
        print(f"  Devin status: {status} ({detail})", flush=True)
        if status in ("exit", "error", "suspended") or detail == "waiting_for_user":
            return data
        time.sleep(poll_interval)
    raise TimeoutError(f"Devin session {session_id} did not finish in time.")


def _download_attachment(
    api_key: str, org_id: str, session_id: str, name: str
) -> bytes | None:
    """Find and download a named attachment from the session via the download endpoint."""
    headers = {"Authorization": f"Bearer {api_key}"}
    resp = httpx.get(
        f"{BASE_URL}/organizations/{org_id}/sessions/{session_id}/attachments",
        headers=headers,
        timeout=30,
    )
    resp.raise_for_status()
    body = resp.json()
    items = body.get("items", []) if isinstance(body, dict) else body
    for item in items:
        if item.get("name") == name:
            attachment_id = item.get("attachment_id")
            if not attachment_id:
                return None
            # The download endpoint 307-redirects to a presigned S3 URL. Follow
            # it by hand: the presigned URL is already authorised, and forwarding
            # the bearer token to S3 breaks the signature.
            redirect = httpx.get(
                f"{BASE_URL}/organizations/{org_id}/attachments/{attachment_id}/{name}",
                headers=headers,
                timeout=60,
            )
            if redirect.status_code in (301, 302, 303, 307, 308):
                location = redirect.headers["location"]
                return httpx.get(location, timeout=60).content
            redirect.raise_for_status()
            return redirect.content
    return None


def _extract_json_from_messages(
    api_key: str, org_id: str, session_id: str
) -> dict[str, Any] | None:
    """Find the latest devin message with a JSON code block and parse it."""
    headers = {"Authorization": f"Bearer {api_key}"}
    resp = httpx.get(
        f"{BASE_URL}/organizations/{org_id}/sessions/{session_id}/messages",
        headers=headers,
        timeout=30,
    )
    resp.raise_for_status()
    body = resp.json()
    items = body.get("items", []) if isinstance(body, dict) else body
    for message in reversed(items):
        if message.get("source") != "devin":
            continue
        content = message.get("message", "")
        # Find a JSON code block.
        match = re.search(r"```(?:json)?\n(.*?)\n```", content, re.DOTALL)
        if not match:
            continue
        try:
            return cast(dict[str, Any], json.loads(match.group(1)))
        except json.JSONDecodeError:
            continue
    return None


def _devin_extract_business(
    url: str,
    api_key: str,
    org_id: str,
    *,
    devin_mode: str = "fast",
    session_id: str | None = None,
) -> dict[str, Any]:
    """Ask Devin to research the URL and return a JSON fact sheet."""

    prompt = f"""Research this URL: {url}

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

    if session_id:
        print(f"Reusing Devin session {session_id}...", flush=True)
    else:
        print(f"Starting Devin research session for {url}...", flush=True)
        session = _create_devin_session(api_key, org_id, prompt, devin_mode=devin_mode)
        session_id = session["session_id"]
        print(f"  Session: {session.get('url')}", flush=True)

    _poll_devin_session(api_key, org_id, session_id)

    data = _download_attachment(api_key, org_id, session_id, "business_facts.json")
    if data is not None:
        return cast(dict[str, Any], json.loads(data.decode("utf-8")))

    # Session may have asked a question; try to extract JSON from its messages.
    facts = _extract_json_from_messages(api_key, org_id, session_id)
    if facts is not None:
        return facts

    raise RuntimeError(
        "Devin did not produce business_facts.json. Check the session in the Devin UI."
    )


def _amount_from_text(text: str | None) -> int:
    if not text:
        return 0
    text = text.strip().lower().replace(",", "")
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if not match:
        return 0
    value = float(match.group(1))
    return int(value * 100)


def _hours_from_text(text: str | None, default: int = 24) -> int:
    if not text:
        return default
    text = text.strip().lower().replace(",", "")
    match = re.search(r"(\d+)", text)
    if not match:
        return default
    return int(match.group(1))


def _time_from_text(text: str | None, default: dt_time) -> dt_time:
    if not text:
        return default
    text = text.strip()
    for fmt in ("%H:%M", "%I:%M %p", "%I:%M%p", "%I %p"):
        try:
            return datetime.strptime(text, fmt).time()
        except ValueError:
            pass
    parts = text.split(":")
    if len(parts) == 2:
        try:
            return dt_time(int(parts[0]), int(parts[1]))
        except ValueError:
            pass
    return default


def _business_from_devin_json(facts: dict[str, Any]) -> BusinessConfig:
    """Turn the JSON Devin returned into a BusinessConfig."""
    price = int(facts.get("price_per_night_gbp_pence") or 20000)
    deposit = int(facts.get("deposit_per_head_gbp_pence") or int(price * 0.2))
    cancellation = int(facts.get("cancellation_window_hours") or 24)
    check_in = _time_from_text(facts.get("check_in_time"), dt_time(15, 0))
    check_out = _time_from_text(facts.get("check_out_time"), dt_time(11, 0))
    max_party = int(facts.get("max_party_size") or 4)

    return BusinessConfig(
        business_id="devin-research",
        name=facts.get("name") or "Devin Research Business",
        opening_hours=(
            OpeningHours(weekday=0, opens=check_in, closes=check_out),
        ),
        catalogue=(
            CatalogueItem(
                sku="ROOM",
                name="Room" if not facts.get("amenities") else facts["amenities"][0],
                unit_price=price,
            ),
        ),
        policies=Policies(
            cancellation_window_hours=cancellation,
            deposit_required_from_party_size=1,
            deposit_per_head=deposit,
            refund_window_hours=cancellation,
            max_party_size=max_party,
            discount_authority=0,
        ),
    )


def _client_beliefs_from_business(
    business: BusinessConfig,
    lie_field: str,
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
    field = aliases.get(lie_field, lie_field)

    if field in facts:
        if field == "deposit":
            facts[field] = f"The deposit is £{deposit * 1.5:.2f} per person."
        elif field == "set lunch" and business.catalogue:
            price = business.catalogue[0].unit_price / 100
            facts[field] = f"The set lunch is £{price * 1.5:.2f}."
        elif field == "opening hours" and business.opening_hours:
            facts[field] = "Our opening hours are 09:00 to 21:00."
        elif field == "cancellation":
            hours = business.policies.cancellation_window_hours
            facts[field] = f"The cancellation window is {hours * 2} hours."
        elif field == "party":
            size = business.policies.max_party_size
            facts[field] = f"The party size we can seat is {size + 3}."

    return ClientBeliefs(facts=facts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Devin research red-team demo.")
    parser.add_argument(
        "--url",
        default="https://www.atlantis.com/dubai/atlantis-the-palm",
        help="Public page for Devin to research.",
    )
    parser.add_argument(
        "--lie-field",
        default="set lunch",
        choices=["deposit", "set lunch", "opening hours", "cancellation", "party"],
        help="Which fact the client gets wrong.",
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
        help="Use the adaptive (claim-extraction) red team.",
    )
    parser.add_argument(
        "--devin-mode",
        default="fast",
        choices=["fast", "lite"],
        help="Devin session mode (fast or lite are cheaper).",
    )
    parser.add_argument(
        "--session-id",
        default=None,
        help="Reuse an existing Devin session instead of starting a new one.",
    )
    args = parser.parse_args()

    _load_env(Path(".env"))
    _load_env(Path("secrets.env"))

    api_key = os.environ.get("DEVIN_API_KEY")
    if not api_key:
        raise RuntimeError("DEVIN_API_KEY is not set. Put it in secrets.env or the environment.")

    org_id = os.environ.get("DEVIN_ORG_ID") or _devin_org_id(api_key)

    facts = _devin_extract_business(
        args.url,
        api_key,
        org_id,
        devin_mode=args.devin_mode,
        session_id=args.session_id,
    )
    print("\n=== Devin extracted facts ===")
    print(json.dumps(facts, indent=2))

    business = _business_from_devin_json(facts)
    client_beliefs = _client_beliefs_from_business(business, args.lie_field)

    print("\n=== Red-team ground truth ===")
    print(business.model_dump_json(indent=2))
    print("\n=== Client private beliefs ===")
    print(client_beliefs.model_dump_json(indent=2))

    red_team = (
        AdaptiveRedTeamSimulator(ground_truth=business, max_turns=args.max_turns)
        if args.adaptive
        else RedTeamSimulator(ground_truth=business, max_turns=args.max_turns)
    )

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
