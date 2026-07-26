"""The world is seeded, restorable, strict about physics and permissive about policy."""

from __future__ import annotations

import pytest

from simharness.schemas import Entity, MutationOp, ToolCall, ToolName, digest_of
from simharness.world import World, build_world


def _bistro(seed: int = 7) -> World:
    return World(
        build_world("bistro", seed),
        (ToolName.CHECK_AVAILABILITY, ToolName.CREATE_BOOKING, ToolName.GET_PRICE),
    )


def _call(world: World, name: ToolName, arguments: dict[str, object], turn: int = 1) -> object:
    return world.execute(ToolCall(call_id="c", name=name, arguments=arguments), turn)


def test_same_seed_builds_an_identical_world() -> None:
    assert digest_of(build_world("bistro", 7)) == digest_of(build_world("bistro", 7))


def test_different_seeds_build_different_worlds() -> None:
    assert digest_of(build_world("bistro", 7)) != digest_of(build_world("bistro", 8))


def test_all_builders_are_deterministic() -> None:
    for name in ("bistro", "bistro_busy", "clinic"):
        assert digest_of(build_world(name, 3)) == digest_of(build_world(name, 3)), name


def test_snapshot_restore_is_exact() -> None:
    world = _bistro()
    before = world.snapshot(0)
    slot = world.state.business.calendar[0].slot_id
    _call(world, ToolName.CREATE_BOOKING, {"slot_id": slot, "party_size": 2, "customer_name": "A"})
    assert world.state.bookings
    world.restore(before)
    assert not world.state.bookings
    assert world.snapshot(0).digest == before.digest


def test_snapshot_is_not_a_live_view() -> None:
    """A snapshot taken before a mutation must not observe that mutation."""
    world = _bistro()
    snap = world.snapshot(0)
    slot = world.state.business.calendar[0].slot_id
    _call(world, ToolName.CREATE_BOOKING, {"slot_id": slot, "party_size": 2, "customer_name": "A"})
    assert snap.state.bookings == {}
    assert digest_of(snap.state) == snap.digest


def test_physics_violations_are_rejected() -> None:
    world = _bistro()
    unknown = _call(
        world, ToolName.CREATE_BOOKING, {"slot_id": "NOPE", "party_size": 2, "customer_name": "A"}
    )
    assert not unknown.ok and "NOPE" in str(unknown.error)  # type: ignore[attr-defined]
    malformed = _call(world, ToolName.GET_PRICE, {})
    assert not malformed.ok  # type: ignore[attr-defined]
    assert not world.state.ledger


def test_policy_violations_are_permitted_and_recorded() -> None:
    """The world must let the agent be wrong, or the verifier measures nothing."""
    world = _bistro()
    slot = world.state.business.calendar[0]
    capacity = world.remaining_capacity(slot.slot_id)
    result = _call(
        world,
        ToolName.CREATE_BOOKING,
        {"slot_id": slot.slot_id, "party_size": capacity + 40, "customer_name": "A"},
    )
    assert result.ok  # type: ignore[attr-defined]
    assert world.remaining_capacity(slot.slot_id) < 0
    ops = [(m.entity, m.op) for m in world.state.ledger]
    assert (Entity.BOOKING, MutationOp.CREATE) in ops


def test_refund_against_a_nonexistent_booking_is_permitted() -> None:
    """Scenario 3 only measures anything because this is allowed through."""
    world = World(build_world("bistro_busy", 1), (ToolName.ISSUE_REFUND,))
    result = _call(world, ToolName.ISSUE_REFUND, {"booking_ref": "BK-9999", "amount": 5000})
    assert result.ok  # type: ignore[attr-defined]
    assert [m.entity for m in world.state.ledger] == [Entity.REFUND]


def test_disabled_tools_are_refused() -> None:
    world = World(build_world("bistro", 1), (ToolName.GET_PRICE,))
    result = _call(world, ToolName.ISSUE_REFUND, {"booking_ref": "BK-0001", "amount": 1})
    assert not result.ok  # type: ignore[attr-defined]
    assert not world.state.ledger


def test_availability_reflects_bookings() -> None:
    world = _bistro()
    slot = world.state.business.calendar[0]
    before = world.remaining_capacity(slot.slot_id)
    _call(
        world,
        ToolName.CREATE_BOOKING,
        {"slot_id": slot.slot_id, "party_size": 2, "customer_name": "A"},
    )
    assert world.remaining_capacity(slot.slot_id) == before - 2


def test_unknown_builder_names_itself() -> None:
    with pytest.raises(KeyError, match="known builders"):
        build_world("does-not-exist", 1)
