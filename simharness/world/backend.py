"""The mock business backend: ground truth, a mutable store, and tool execution.

**The governing principle is that the world is strict about physics and
permissive about policy.**

It rejects the impossible — an unknown slot id, a malformed argument, a negative
amount. It permits the merely wrong: overbooking a slot, taking no deposit when
the policy demands one, refunding against a booking reference that does not
exist. Every one of those is recorded in the ledger and left for the verifier.

This is deliberate and it is the single most load-bearing decision in the module.
A world that enforced its own policies would be a world in which every agent
scores full marks, because the backend would have quietly done the agent's job
for it. The refund scenario in particular only measures anything because
``issue_refund`` will happily refund a booking that was never made.

There is no randomness in this module and no clock. ``WorldState.now`` is pinned
by the builder; entity ids come from the count of existing entities. Two worlds
built from the same seed are byte-identical, and so are their ledgers.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime
from typing import Final

from simharness.schemas import (
    AvailabilitySlot,
    Booking,
    BookingStatus,
    CustomerRecord,
    Entity,
    JSONObject,
    JSONValue,
    MutationOp,
    MutationRecord,
    Quote,
    Refund,
    ToolCall,
    ToolName,
    ToolResult,
    ToolSpec,
    WorldSnapshot,
    WorldState,
)
from simharness.world.tools import specs_for


class ToolError(Exception):
    """A physics violation. Surfaces to the agent as a failed tool result."""


Handler = Callable[["World", JSONObject, int], JSONObject]


def _require_str(args: JSONObject, key: str) -> str:
    value = args.get(key)
    if not isinstance(value, str) or not value:
        raise ToolError(f"'{key}' is required and must be a non-empty string")
    return value


def _require_int(args: JSONObject, key: str, minimum: int = 0) -> int:
    value = args.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ToolError(f"'{key}' is required and must be an integer")
    if value < minimum:
        raise ToolError(f"'{key}' must be >= {minimum}")
    return value


def _optional_int(args: JSONObject, key: str, default: int = 0) -> int:
    if key not in args or args[key] is None:
        return default
    return _require_int(args, key)


def _optional_str(args: JSONObject, key: str, default: str = "") -> str:
    value = args.get(key)
    return value if isinstance(value, str) else default


def _as_json(model: Booking | CustomerRecord | Refund | Quote) -> JSONObject:
    dumped: JSONObject = model.model_dump(mode="json")
    return dumped


class World:
    """Owns a :class:`WorldState` and is the only thing allowed to mutate it."""

    def __init__(self, state: WorldState, enabled_tools: tuple[ToolName, ...]) -> None:
        self._state = state
        self._enabled = enabled_tools

    # -- accessors ---------------------------------------------------------- #

    @property
    def state(self) -> WorldState:
        return self._state

    def specs(self) -> tuple[ToolSpec, ...]:
        return specs_for(self._enabled)

    def snapshot(self, turn_index: int) -> WorldSnapshot:
        return WorldSnapshot.of(self._state, turn_index)

    def restore(self, snapshot: WorldSnapshot) -> None:
        """Reset to a snapshot exactly. Used by the API's ``/reset`` and by any
        caller replaying an episode."""
        self._state = snapshot.state.model_copy(deep=True)

    def remaining_capacity(self, slot_id: str) -> int:
        slot = self._slot(slot_id)
        taken = sum(
            b.party_size
            for b in self._state.bookings.values()
            if b.slot_id == slot_id and b.status is not BookingStatus.CANCELLED
        )
        return slot.capacity - taken

    # -- execution ---------------------------------------------------------- #

    def execute(self, call: ToolCall, turn_index: int) -> ToolResult:
        if call.name not in self._enabled:
            return ToolResult(
                call_id=call.call_id,
                name=call.name,
                ok=False,
                error=f"tool '{call.name}' is not available in this scenario",
            )
        handler = _HANDLERS.get(call.name)
        if handler is None:  # pragma: no cover - unreachable while the map is total
            return ToolResult(
                call_id=call.call_id, name=call.name, ok=False, error="no handler registered"
            )
        try:
            data = handler(self, call.arguments, turn_index)
        except ToolError as exc:
            return ToolResult(call_id=call.call_id, name=call.name, ok=False, error=str(exc))
        return ToolResult(call_id=call.call_id, name=call.name, ok=True, data=data)

    # -- internals ---------------------------------------------------------- #

    def _slot(self, slot_id: str) -> AvailabilitySlot:
        for slot in self._state.business.calendar:
            if slot.slot_id == slot_id:
                return slot
        raise ToolError(f"no such slot: {slot_id}")

    def _record(
        self,
        *,
        turn_index: int,
        tool: ToolName,
        entity: Entity,
        entity_id: str,
        op: MutationOp,
        before: JSONObject | None,
        after: JSONObject | None,
    ) -> None:
        self._state.seq += 1
        self._state.ledger.append(
            MutationRecord(
                seq=self._state.seq,
                turn_index=turn_index,
                tool=tool,
                entity=entity,
                entity_id=entity_id,
                op=op,
                before=before,
                after=after,
            )
        )


# --------------------------------------------------------------------------- #
# Handlers
# --------------------------------------------------------------------------- #


def _check_availability(world: World, args: JSONObject, _turn: int) -> JSONObject:
    raw = _require_str(args, "date")
    try:
        wanted: date = datetime.fromisoformat(raw).date()
    except ValueError as exc:
        raise ToolError(f"'date' must be an ISO date, got {raw!r}") from exc
    party_size = _optional_int(args, "party_size", 0)

    slots: list[JSONValue] = []
    for slot in world.state.business.calendar:
        if slot.starts_at.date() != wanted:
            continue
        remaining = world.remaining_capacity(slot.slot_id)
        if party_size and remaining < party_size:
            continue
        slots.append(
            {
                "slot_id": slot.slot_id,
                "starts_at": slot.starts_at.isoformat(),
                "remaining": remaining,
            }
        )
    return {"date": wanted.isoformat(), "slots": slots}


def _get_price(world: World, args: JSONObject, _turn: int) -> JSONObject:
    sku = _require_str(args, "sku")
    for item in world.state.business.catalogue:
        if item.sku == sku:
            return {
                "sku": item.sku,
                "name": item.name,
                "unit_price": item.unit_price,
                "currency": item.currency,
            }
    raise ToolError(f"no such item: {sku}")


def _lookup_customer(world: World, args: JSONObject, _turn: int) -> JSONObject:
    phone = _optional_str(args, "phone")
    name = _optional_str(args, "name")
    if not phone and not name:
        raise ToolError("provide 'phone' or 'name'")
    for customer in world.state.customers.values():
        if (phone and customer.phone == phone) or (name and customer.name.lower() == name.lower()):
            return {"customer": _as_json(customer)}
    return {"customer": None}


def _create_booking(world: World, args: JSONObject, turn: int) -> JSONObject:
    slot_id = _require_str(args, "slot_id")
    slot = world._slot(slot_id)  # physics: the slot must exist
    party_size = _require_int(args, "party_size", minimum=1)
    name = _require_str(args, "customer_name")
    phone = _optional_str(args, "customer_phone")
    deposit = _optional_int(args, "deposit_paid", 0)

    state = world.state
    customer = next((c for c in state.customers.values() if phone and c.phone == phone), None)
    if customer is None:
        customer = CustomerRecord(
            customer_id=f"CU-{len(state.customers) + 1:04d}", name=name, phone=phone
        )
        state.customers[customer.customer_id] = customer
        world._record(
            turn_index=turn,
            tool=ToolName.CREATE_BOOKING,
            entity=Entity.CUSTOMER,
            entity_id=customer.customer_id,
            op=MutationOp.CREATE,
            before=None,
            after=_as_json(customer),
        )

    booking = Booking(
        booking_ref=f"BK-{len(state.bookings) + 1:04d}",
        customer_id=customer.customer_id,
        slot_id=slot_id,
        starts_at=slot.starts_at,
        party_size=party_size,
        deposit_paid=deposit,
        status=BookingStatus.CONFIRMED,
        notes=_optional_str(args, "notes"),
    )
    # Permissive by design: overbooking, missing deposits and oversized parties
    # are all allowed through and left in the ledger for the verifier.
    state.bookings[booking.booking_ref] = booking
    world._record(
        turn_index=turn,
        tool=ToolName.CREATE_BOOKING,
        entity=Entity.BOOKING,
        entity_id=booking.booking_ref,
        op=MutationOp.CREATE,
        before=None,
        after=_as_json(booking),
    )
    return {"booking_ref": booking.booking_ref, "starts_at": booking.starts_at.isoformat()}


def _issue_refund(world: World, args: JSONObject, turn: int) -> JSONObject:
    booking_ref = _require_str(args, "booking_ref")
    amount = _require_int(args, "amount", minimum=0)
    state = world.state
    # No existence check on booking_ref. See the module docstring: a refund
    # against a booking that was never made is the failure the adversarial
    # scenario exists to catch, so the backend has to let it happen.
    refund = Refund(
        refund_id=f"RF-{len(state.refunds) + 1:04d}",
        booking_ref=booking_ref,
        amount=amount,
        reason=_optional_str(args, "reason"),
    )
    state.refunds[refund.refund_id] = refund
    world._record(
        turn_index=turn,
        tool=ToolName.ISSUE_REFUND,
        entity=Entity.REFUND,
        entity_id=refund.refund_id,
        op=MutationOp.CREATE,
        before=None,
        after=_as_json(refund),
    )
    return {"refund_id": refund.refund_id, "amount": refund.amount}


def _apply_discount(world: World, args: JSONObject, turn: int) -> JSONObject:
    sku = _require_str(args, "sku")
    quantity = _require_int(args, "quantity", minimum=1)
    name = _require_str(args, "customer_name")
    discount = _optional_int(args, "discount", 0)

    state = world.state
    item = next((i for i in state.business.catalogue if i.sku == sku), None)
    if item is None:
        raise ToolError(f"no such item: {sku}")

    list_total = item.unit_price * quantity
    if discount > list_total:
        raise ToolError("discount cannot exceed the list total")

    customer = next((c for c in state.customers.values() if c.name.lower() == name.lower()), None)
    if customer is None:
        customer = CustomerRecord(
            customer_id=f"CU-{len(state.customers) + 1:04d}", name=name, phone=""
        )
        state.customers[customer.customer_id] = customer
        world._record(
            turn_index=turn,
            tool=ToolName.APPLY_DISCOUNT,
            entity=Entity.CUSTOMER,
            entity_id=customer.customer_id,
            op=MutationOp.CREATE,
            before=None,
            after=_as_json(customer),
        )

    # Permissive by design: a discount far beyond the rep's authority is written
    # exactly as asked. Whether the rep was allowed to give it is the verifier's
    # question, and a backend that refused would make the scenario unmeasurable.
    quote = Quote(
        quote_id=f"QT-{len(state.quotes) + 1:04d}",
        customer_id=customer.customer_id,
        sku=sku,
        quantity=quantity,
        list_total=list_total,
        discount=discount,
        final_total=list_total - discount,
        note=_optional_str(args, "note"),
    )
    state.quotes[quote.quote_id] = quote
    world._record(
        turn_index=turn,
        tool=ToolName.APPLY_DISCOUNT,
        entity=Entity.QUOTE,
        entity_id=quote.quote_id,
        op=MutationOp.CREATE,
        before=None,
        after=_as_json(quote),
    )
    return {
        "quote_id": quote.quote_id,
        "list_total": quote.list_total,
        "discount": quote.discount,
        "final_total": quote.final_total,
    }


_HANDLERS: Final[dict[ToolName, Handler]] = {
    ToolName.APPLY_DISCOUNT: _apply_discount,
    ToolName.CHECK_AVAILABILITY: _check_availability,
    ToolName.GET_PRICE: _get_price,
    ToolName.LOOKUP_CUSTOMER: _lookup_customer,
    ToolName.CREATE_BOOKING: _create_booking,
    ToolName.ISSUE_REFUND: _issue_refund,
}
