"""Fetch real business facts from Context.dev.

Example:
    PYTHONPATH=. .venv/bin/python scripts/fetch_context_demo.py \
        --domain atlantis.com \
        --url "https://www.atlantis.com/dubai/atlantis-the-palm"
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from simharness.adapters.contextdev_client import ContextDevClient


def _load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key, value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default="atlantis.com", help="Domain for brand lookup.")
    parser.add_argument(
        "--url",
        default="https://www.atlantis.com/dubai/atlantis-the-palm",
        help="Hotel or business page to extract facts from.",
    )
    args = parser.parse_args()

    _load_env(Path(".env"))
    _load_env(Path("secrets.env"))

    client = ContextDevClient()

    print(f"=== Brand: {args.domain} ===")
    brand = client.brand_retrieve(domain=args.domain)
    print(json.dumps(brand, indent=2))

    print(f"\n=== Extracted facts: {args.url} ===")
    facts = client.extract_hotel_facts(url=args.url)
    print(json.dumps(facts, indent=2))


if __name__ == "__main__":
    main()
