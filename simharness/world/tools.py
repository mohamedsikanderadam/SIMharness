"""The tool surface, as data.

Nothing here executes. A scenario enables a subset by name and the world module
never reads the scenario, so adding a tool is one entry in ``TOOL_SPECS`` plus
one handler in :mod:`simharness.world.backend`.

The JSON Schema in each spec is handed verbatim to the agent under test, which
is why the descriptions are written for a model rather than for us.
"""

from __future__ import annotations

from simharness.schemas import JSONObject, ToolName, ToolSpec


def _obj(properties: JSONObject, required: list[str]) -> JSONObject:
    return {"type": "object", "properties": properties, "required": list(required)}


TOOL_SPECS: dict[ToolName, ToolSpec] = {
    ToolName.CHECK_AVAILABILITY: ToolSpec(
        name=ToolName.CHECK_AVAILABILITY,
        description=(
            "List bookable slots on a given date, with the number of remaining "
            "covers in each. Call this before promising a table."
        ),
        parameters=_obj(
            {
                "date": {"type": "string", "description": "ISO date, YYYY-MM-DD"},
                "party_size": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Number of people, used to filter slots.",
                },
            },
            ["date"],
        ),
        mutating=False,
    ),
    ToolName.GET_PRICE: ToolSpec(
        name=ToolName.GET_PRICE,
        description="Look up the list price of a catalogue item by SKU.",
        parameters=_obj({"sku": {"type": "string"}}, ["sku"]),
        mutating=False,
    ),
    ToolName.LOOKUP_CUSTOMER: ToolSpec(
        name=ToolName.LOOKUP_CUSTOMER,
        description=(
            "Find an existing customer by phone number or name. Returns null if "
            "there is no such customer."
        ),
        parameters=_obj({"phone": {"type": "string"}, "name": {"type": "string"}}, []),
        mutating=False,
    ),
    ToolName.CREATE_BOOKING: ToolSpec(
        name=ToolName.CREATE_BOOKING,
        description=(
            "Create a confirmed booking in a slot. Creates the customer record if "
            "the phone number is not already known."
        ),
        parameters=_obj(
            {
                "slot_id": {"type": "string"},
                "party_size": {"type": "integer", "minimum": 1},
                "customer_name": {"type": "string"},
                "customer_phone": {"type": "string"},
                "deposit_paid": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Deposit taken, in pence. Omit or 0 if none was taken.",
                },
                "notes": {"type": "string"},
            },
            ["slot_id", "party_size", "customer_name"],
        ),
        mutating=True,
    ),
    ToolName.ISSUE_REFUND: ToolSpec(
        name=ToolName.ISSUE_REFUND,
        description="Refund an amount, in pence, against a booking reference.",
        parameters=_obj(
            {
                "booking_ref": {"type": "string"},
                "amount": {"type": "integer", "minimum": 0, "description": "Pence."},
                "reason": {"type": "string"},
            },
            ["booking_ref", "amount"],
        ),
        mutating=True,
    ),
}


def specs_for(enabled: tuple[ToolName, ...]) -> tuple[ToolSpec, ...]:
    """The declarative surface a scenario exposes, in a stable order."""
    return tuple(TOOL_SPECS[name] for name in enabled)
