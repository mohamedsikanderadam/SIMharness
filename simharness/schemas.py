"""All data that crosses a module boundary in simharness.

This module is the contract. It depends on nothing inside the package and on
nothing outside it except pydantic and the stdlib, so every other module can
import it without creating a cycle and without dragging in a provider SDK.

Three invariants are enforced here structurally rather than by convention,
because they are the ones that quietly destroy a reward signal if they rot:

1. **No hidden-state leak.** The agent-under-test is handed
   :class:`AgentRequest`, whose history is a tuple of :class:`AgentTurnView`.
   That view carries a speaker and a string and forbids extra fields, so there
   is no field on it through which a persona goal, an internal simulator state,
   or a world snapshot could travel. :meth:`AgentTurnView.from_turn` is the only
   sanctioned projection.
2. **Money is exact.** Every monetary amount is an integer count of minor units
   (pence, cents, fils). A verifier that compares floats is a verifier that
   emits false negatives at 3am.
3. **The scalar is not a bare float and cannot drift from its parts.**
   :class:`RewardBreakdown` recomputes its own scalar from the weighted
   components and rejects any mismatch.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, time
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import TypeAliasType

# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #

JSONValue = TypeAliasType(
    "JSONValue",
    "str | int | float | bool | list[JSONValue] | dict[str, JSONValue] | None",
)
"""Anything that survives a JSON round trip. Used for tool arguments and results.

Spelled with ``TypeAliasType`` rather than a plain alias because pydantic needs a
real recursive reference to build a schema for it; a self-referencing plain alias
expands until the recursion limit.
"""

JSONObject = dict[str, JSONValue]

MinorUnits = Annotated[int, Field(description="Money as whole minor units, e.g. 2350 == 23.50")]
Probability = Annotated[float, Field(ge=0.0, le=1.0)]
Seed = Annotated[int, Field(ge=0, lt=2**63)]


class Frozen(BaseModel):
    """Base for anything that must not be mutated after construction."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class Mutable(BaseModel):
    """Base for live episode state."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


def digest_of(model: BaseModel) -> str:
    """Content hash of a model, stable across processes.

    pydantic serialises fields in declaration order, so ``model_dump_json`` is
    canonical for a fixed schema version. Used for world snapshots and for the
    config digest that stamps every trajectory.
    """
    return hashlib.sha256(model.model_dump_json().encode("utf-8")).hexdigest()[:32]


# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #


class Speaker(StrEnum):
    USER = "user"
    AGENT = "agent"
    TOOL = "tool"
    SYSTEM = "system"


class TerminationReason(StrEnum):
    """How an episode ended. The first three are chosen by the simulator; the
    rest are imposed by the harness and are all, for scoring purposes, unclean."""

    SATISFIED = "satisfied"
    GAVE_UP = "gave_up"
    HUNG_UP_ANGRY = "hung_up_angry"
    PATIENCE_EXHAUSTED = "patience_exhausted"
    MAX_TURNS = "max_turns"
    AGENT_ERROR = "agent_error"
    HARNESS_ERROR = "harness_error"


CLEAN_TERMINATIONS: frozenset[TerminationReason] = frozenset(
    {TerminationReason.SATISFIED, TerminationReason.GAVE_UP}
)
"""``GAVE_UP`` is clean because on the refusal scenario a correct agent *should*
leave the customer unsatisfied. Anger, timeouts and crashes are never clean."""


class ToolName(StrEnum):
    CHECK_AVAILABILITY = "check_availability"
    CREATE_BOOKING = "create_booking"
    LOOKUP_CUSTOMER = "lookup_customer"
    ISSUE_REFUND = "issue_refund"
    GET_PRICE = "get_price"
    APPLY_DISCOUNT = "apply_discount"


class Entity(StrEnum):
    BOOKING = "booking"
    CUSTOMER = "customer"
    REFUND = "refund"
    QUOTE = "quote"


class MutationOp(StrEnum):
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"


class Temperament(StrEnum):
    BRISK = "brisk"
    CHATTY = "chatty"
    ANXIOUS = "anxious"
    IRRITABLE = "irritable"
    FLAT = "flat"


class Disclosure(StrEnum):
    """When a persona will part with a fact it knows."""

    VOLUNTEERED = "volunteered"
    ON_DIRECT_QUESTION = "on_direct_question"
    WITHHELD = "withheld"


class AdversarialFlag(StrEnum):
    SEEKS_UNAUTHORISED_DISCOUNT = "seeks_unauthorised_discount"
    ASSERTS_FALSE_PRIOR_BOOKING = "asserts_false_prior_booking"
    PRESSURES_POLICY_EXCEPTION = "pressures_policy_exception"
    INVENTS_STAFF_PROMISE = "invents_staff_promise"


class ClaimKind(StrEnum):
    """Categories of factual assertion the verifier will try to check."""

    PRICE = "price"
    HOURS = "hours"
    POLICY = "policy"
    AVAILABILITY = "availability"
    BOOKING_REF = "booking_ref"


class ClaimVerdict(StrEnum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    UNGROUNDED = "ungrounded"
    """Well-formed and parseable, but no tool result in this episode supports it."""
    UNPARSED = "unparsed"
    """A numeric or entity token the grammar could not bind to a claim. Counted
    against coverage, never against the score. See DESIGN_NOTE.md."""


class EvidenceKind(StrEnum):
    """Transcript-level requirements, all grounded in the tool ledger rather
    than in sentiment or keyword matching."""

    OFFERED_ALTERNATIVE_SLOT = "offered_alternative_slot"
    STATED_POLICY_CORRECTLY = "stated_policy_correctly"
    CONFIRMED_CRITICAL_NUMBER = "confirmed_critical_number"
    CHECKED_RECORDS = "checked_records"
    """Required a successful call to a named tool. Without this, a scenario whose
    only criterion is "do not mutate" is passed by an agent that says nothing and
    does nothing — stonewalling scores the same as diligence, and the null policy
    beats an honest one that looked and then explained."""


class Severity(StrEnum):
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"
    INFO = "info"


class FailureTag(StrEnum):
    """The eval product's vocabulary. One tag == one thing to tell a customer."""

    HALLUCINATED_PRICE = "hallucinated_price"
    HALLUCINATED_POLICY = "hallucinated_policy"
    HALLUCINATED_AVAILABILITY = "hallucinated_availability"
    MISSTATED_BOOKING_RECORD = "misstated_booking_record"
    """Said something about the customer's records that the records do not
    support — including asserting there is no booking without ever looking."""
    BOOKED_UNAVAILABLE_SLOT = "booked_unavailable_slot"
    WRONG_PARTY_SIZE = "wrong_party_size"
    SKIPPED_DEPOSIT = "skipped_deposit"
    COMPLIED_WITH_DISALLOWED_REQUEST = "complied_with_disallowed_request"
    REFUNDED_WITHOUT_BOOKING = "refunded_without_booking"
    GRANTED_UNAUTHORISED_DISCOUNT = "granted_unauthorised_discount"
    NO_COMPLIANT_QUOTE = "no_compliant_quote"
    """No quote was written that a manager would sign off. Distinct from granting
    an unauthorised discount: an agent that quoted nothing at all did not concede
    margin, it just failed to sell."""
    MISHEARD_NUMBER_UNCONFIRMED = "misheard_number_unconfirmed"
    NO_ALTERNATIVE_OFFERED = "no_alternative_offered"
    REFUSED_WITHOUT_CHECKING = "refused_without_checking"
    ABANDONED_CUSTOMER = "abandoned_customer"
    ENRAGED_CUSTOMER = "enraged_customer"
    EXCEEDED_TURN_BUDGET = "exceeded_turn_budget"


class NoiseOpKind(StrEnum):
    HOMOPHONE = "homophone"
    FILLER = "filler"
    DROP = "drop"
    TRUNCATE = "truncate"
    DIGIT = "digit"


# --------------------------------------------------------------------------- #
# World: immutable ground truth
# --------------------------------------------------------------------------- #


class CatalogueItem(Frozen):
    sku: str
    name: str
    unit_price: MinorUnits
    currency: str = "GBP"


class OpeningHours(Frozen):
    weekday: Annotated[int, Field(ge=0, le=6, description="0 == Monday")]
    opens: time
    closes: time
    closed: bool = False


class Policies(Frozen):
    """Every field here is a fact an agent can get wrong out loud."""

    cancellation_window_hours: Annotated[int, Field(ge=0)]
    deposit_required_from_party_size: Annotated[int, Field(ge=1)]
    deposit_per_head: MinorUnits
    refund_window_hours: Annotated[int, Field(ge=0)]
    max_party_size: Annotated[int, Field(ge=1)]
    discount_authority: MinorUnits = 0
    """Largest discount an agent may grant unilaterally. Zero means none."""


class AvailabilitySlot(Frozen):
    slot_id: str
    starts_at: datetime
    capacity: Annotated[int, Field(ge=0)]


class BusinessConfig(Frozen):
    """The half of the world an agent may read but never write."""

    business_id: str
    name: str
    timezone: str = "Europe/London"
    catalogue: tuple[CatalogueItem, ...] = ()
    opening_hours: tuple[OpeningHours, ...] = ()
    policies: Policies
    calendar: tuple[AvailabilitySlot, ...] = ()


# --------------------------------------------------------------------------- #
# World: mutable store
# --------------------------------------------------------------------------- #


class BookingStatus(StrEnum):
    CONFIRMED = "confirmed"
    CANCELLED = "cancelled"
    PENDING_DEPOSIT = "pending_deposit"


class Booking(Mutable):
    booking_ref: str
    customer_id: str
    slot_id: str
    starts_at: datetime
    party_size: Annotated[int, Field(ge=1)]
    deposit_paid: MinorUnits = 0
    status: BookingStatus = BookingStatus.CONFIRMED
    notes: str = ""


class CustomerRecord(Mutable):
    customer_id: str
    name: str
    phone: str
    email: str = ""
    notes: str = ""


class Refund(Mutable):
    refund_id: str
    booking_ref: str
    amount: MinorUnits
    reason: str = ""


class Quote(Mutable):
    """A priced offer written into the CRM.

    ``discount`` is the field the whole sales scenario turns on: the backend will
    write any discount the agent asks for, and whether that discount was within
    the rep's authority is a question for the verifier, not the database.
    """

    quote_id: str
    customer_id: str
    sku: str
    quantity: Annotated[int, Field(ge=1)]
    list_total: MinorUnits
    discount: MinorUnits = 0
    final_total: MinorUnits = 0
    note: str = ""


class MutationRecord(Frozen):
    """Append-only ledger entry. The forbidden-mutation check reads this, not a
    diff of the two snapshots, so a create-then-delete cannot hide."""

    seq: int
    turn_index: int
    tool: ToolName
    entity: Entity
    entity_id: str
    op: MutationOp
    before: JSONObject | None = None
    after: JSONObject | None = None


class WorldState(Mutable):
    """Live episode state: frozen ground truth plus everything the agent wrote.

    ``now`` is pinned at reset and advanced only by the runner, never by the
    wall clock — a verifier that consults ``datetime.now()`` is a verifier whose
    replays disagree with themselves.
    """

    business: BusinessConfig
    now: datetime
    bookings: dict[str, Booking] = Field(default_factory=dict)
    customers: dict[str, CustomerRecord] = Field(default_factory=dict)
    refunds: dict[str, Refund] = Field(default_factory=dict)
    quotes: dict[str, Quote] = Field(default_factory=dict)
    ledger: list[MutationRecord] = Field(default_factory=list)
    seq: int = 0
    """Monotonic counter backing both ledger ordering and generated ids, so
    booking refs are a function of the seed rather than of a random source."""


class WorldSnapshot(Frozen):
    """A deep copy taken at a boundary, with a content hash.

    The verifier receives snapshots, never the live :class:`WorldState`, which
    is how "the primary score is a pure function" is enforced rather than
    merely asserted.
    """

    state: WorldState
    digest: str
    taken_at_turn: int

    @classmethod
    def of(cls, state: WorldState, turn_index: int) -> Self:
        copied = state.model_copy(deep=True)
        return cls(state=copied, digest=digest_of(copied), taken_at_turn=turn_index)


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #


class ToolSpec(Frozen):
    """Declarative tool surface. Scenarios enable a subset by name; nothing in
    the world module reads the scenario, so adding a tool is one entry here plus
    one handler."""

    name: ToolName
    description: str
    parameters: JSONObject
    """JSON Schema for the arguments object, handed verbatim to the agent."""
    mutating: bool


class ToolCall(Frozen):
    call_id: str
    name: ToolName
    arguments: JSONObject


class ToolResult(Frozen):
    call_id: str
    name: ToolName
    ok: bool
    data: JSONObject | None = None
    error: str | None = None

    @model_validator(mode="after")
    def _one_of(self) -> Self:
        if self.ok == (self.error is not None):
            raise ValueError("a tool result is either ok with data or not ok with an error")
        return self


# --------------------------------------------------------------------------- #
# Persona
# --------------------------------------------------------------------------- #


class HiddenFact(Frozen):
    """One thing the customer knows. ``key`` is what the verifier and the
    fidelity probes refer to it by; ``value`` is what the simulator may say."""

    key: str
    value: str
    disclosure: Disclosure
    asked_by: tuple[str, ...] = ()
    """Lowercase substrings that count as a direct question about this fact.
    Only consulted for ``ON_DIRECT_QUESTION``."""


class HiddenGoal(Frozen):
    """What the customer is actually trying to achieve.

    Kept separate from :class:`Scenario.success` on purpose: the goal drives the
    simulator's behaviour, the scenario's success criteria drive the reward. On
    the refusal scenario they deliberately disagree — the customer wants the
    reschedule, and the correct agent does not give it to them.
    """

    summary: str
    target: JSONObject = Field(default_factory=dict)
    satisfied_when: str = ""
    """Prose the simulator uses to decide it is done. Never read by the verifier."""


class SpeechProfile(Frozen):
    """Per-persona modulation of the noise wrapper. A mumbler is not a different
    ASR, it is the same ASR having a worse time."""

    wer_multiplier: Annotated[float, Field(ge=0.0, le=3.0)] = 1.0
    truncation_bias: Probability = 0.0
    digit_error_bias: Probability = 0.0


class Persona(Frozen):
    persona_id: str
    display_name: str
    temperament: Temperament
    hidden_goal: HiddenGoal
    hidden_facts: tuple[HiddenFact, ...] = ()
    patience_turns: Annotated[int, Field(ge=1)]
    interruption_prob: Probability = 0.0
    topic_change_prob: Probability = 0.0
    verbosity: Annotated[int, Field(ge=1, le=5)] = 3
    adversarial_flags: tuple[AdversarialFlag, ...] = ()
    speech: SpeechProfile = SpeechProfile()
    style_notes: str = ""
    opening: str = ""
    """The first thing this customer says. The scripted simulator speaks it
    verbatim; an LLM simulator uses it to anchor voice and opening posture."""
    satisfied_markers: tuple[str, ...] = ()
    """Lowercase substrings that mean this customer got what they came for.

    Declared per persona rather than hardcoded in the simulator: what counts as
    satisfaction is a property of the goal, and a haggler hearing "I can approve"
    is done in a way a diner hearing it is not."""
    escalations: tuple[str, ...] = ()
    """What they say when told no, in order. Empty for a cooperative persona —
    an escalation ladder is what makes an adversarial flag mean something rather
    than being a label nothing reads."""

    def facts_by_disclosure(self, level: Disclosure) -> tuple[HiddenFact, ...]:
        return tuple(f for f in self.hidden_facts if f.disclosure is level)


# --------------------------------------------------------------------------- #
# Scenario
# --------------------------------------------------------------------------- #


class FieldMatch(Frozen):
    """One expected field value. ``path`` is a dotted path into the record."""

    path: str
    equals: JSONValue = None
    at_least: float | None = None
    at_most: float | None = None

    @model_validator(mode="after")
    def _one_predicate(self) -> Self:
        given = sum(x is not None for x in (self.equals, self.at_least, self.at_most))
        if given != 1:
            raise ValueError(f"{self.path}: exactly one predicate must be set")
        return self


class RequiredRecord(Frozen):
    entity: Entity
    matches: tuple[FieldMatch, ...]
    count: Annotated[int, Field(ge=1)] = 1
    tag: FailureTag | None = None
    """What to call it when this record is missing. Omitted, the verifier guesses
    from the field paths, which is fine for bookings and wrong for anything it
    has not met — a scenario knows its own failure mode better than a heuristic
    does."""


class ForbiddenMutation(Frozen):
    """A mutation that must not appear in the ledger. ``where`` narrows by
    entity id; omitted means any instance is forbidden."""

    entity: Entity
    op: MutationOp | None = None
    where_entity_id: str | None = None
    where_matches: tuple[FieldMatch, ...] = ()
    """Field predicates against the mutation's ``after`` payload. Without these a
    prohibition can only say "no quotes at all", when what the business actually
    forbids is "no quote *discounted beyond the rep's authority*" — the write
    itself is fine, its contents are not."""
    tag: FailureTag


class EvidenceRequirement(Frozen):
    kind: EvidenceKind
    detail: JSONObject = Field(default_factory=dict)
    tag: FailureTag


class SuccessCriteria(Frozen):
    required_records: tuple[RequiredRecord, ...] = ()
    forbidden_mutations: tuple[ForbiddenMutation, ...] = ()
    required_evidence: tuple[EvidenceRequirement, ...] = ()
    claim_scope: tuple[ClaimKind, ...] = (
        ClaimKind.PRICE,
        ClaimKind.HOURS,
        ClaimKind.POLICY,
        ClaimKind.AVAILABILITY,
    )


class Scenario(Frozen):
    scenario_id: str
    title: str
    description: str
    world_builder: str
    """Name of the registered builder that produces the initial world. A string
    rather than a callable so a scenario stays serialisable and diffable."""
    world_seed: Seed
    enabled_tools: tuple[ToolName, ...]
    success: SuccessCriteria
    max_turns: Annotated[int, Field(ge=1)] = 20
    """Counted in user turns, so the budget means the same thing whether or not
    the agent burns tool calls."""
    opening_speaker: Literal[Speaker.AGENT, Speaker.USER] = Speaker.AGENT
    """Inbound calls open with the business greeting, which is what a webhook
    agent expects. Set to USER for scenarios that start mid-conversation."""
    agent_brief: str = ""
    """What the business tells its agent. Handed to LocalPolicyAgent as a system
    prompt; ignored by HTTPAgent, whose prompt belongs to whoever we are testing."""


# --------------------------------------------------------------------------- #
# Noise
# --------------------------------------------------------------------------- #


class NoiseOp(Frozen):
    kind: NoiseOpKind
    token_index: int
    before: str
    after: str


class NoiseConfig(Frozen):
    target_wer: Probability = 0.0
    op_weights: dict[NoiseOpKind, float] = Field(
        default_factory=lambda: {
            NoiseOpKind.HOMOPHONE: 0.35,
            NoiseOpKind.DIGIT: 0.25,
            NoiseOpKind.DROP: 0.20,
            NoiseOpKind.FILLER: 0.15,
            NoiseOpKind.TRUNCATE: 0.05,
        }
    )
    max_truncation_fraction: Probability = 0.3
    phoneme_distance_max: Annotated[int, Field(ge=0, le=4)] = 1
    locale: str = "en"
    """Extension point. Code-switching and Arabic transliteration hook in here;
    not implemented — see ARCHITECTURE.md."""


class NoiseTrace(Frozen):
    """Enough to reproduce and to explain a corruption after the fact."""

    seed: Seed
    target_wer: Probability
    measured_wer: float
    ops: tuple[NoiseOp, ...] = ()


# --------------------------------------------------------------------------- #
# Simulator
# --------------------------------------------------------------------------- #


class SimulatorInternalState(Mutable):
    """Never crosses the adapter boundary. Persisted in the trajectory because
    replay and the leak test both need it."""

    revealed_fact_keys: list[str] = Field(default_factory=list)
    patience_remaining: int
    mood: Annotated[float, Field(ge=-1.0, le=1.0)] = 0.0
    goal_progress: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0
    scratchpad: str = ""
    """The simulator's private reasoning. The single most dangerous field in the
    codebase; :class:`AgentTurnView` exists to make it unreachable."""


class SimulatorTurnView(Frozen):
    """History as the *customer* remembers it — the mirror of
    :class:`AgentTurnView`.

    The asymmetry between the two is the entire noise model stated in one line:
    this view reads ``turn.text``, the agent's view reads ``turn.delivered_text``.
    A caller who mishears "fifteen" as "fifty" is not corrected by the customer's
    memory, because the customer knows perfectly well what they said. Feeding the
    simulator the corrupted text instead would make it apologise for the ASR's
    mistakes, which is both unrealistic and a quiet source of reward inflation.
    """

    speaker: Literal[Speaker.USER, Speaker.AGENT]
    text: str

    @classmethod
    def from_turn(cls, turn: Turn) -> Self:
        if turn.speaker not in (Speaker.USER, Speaker.AGENT):
            raise ValueError("only user and agent turns are visible to the simulator")
        return cls(speaker=turn.speaker, text=turn.text)


class SimulatorContext(Frozen):
    """Everything a simulator is allowed to condition on.

    Note what is absent: no :class:`WorldState`, no :class:`SuccessCriteria`, no
    tool results. A simulator that could see the success criteria would steer the
    agent toward them, and every scenario would pass.
    """

    persona: Persona
    internal_state: SimulatorInternalState
    history: tuple[SimulatorTurnView, ...]
    turn_index: int
    seed: Seed


class SimulatorOutput(Frozen):
    """Return type of every simulator. Only ``utterance`` continues downstream."""

    utterance: str
    internal_state: SimulatorInternalState
    terminate: bool = False
    termination: TerminationReason | None = None
    usage: TokenUsage | None = None

    @model_validator(mode="after")
    def _termination_agrees(self) -> Self:
        if self.terminate and self.termination is None:
            raise ValueError("terminate=True requires a termination reason")
        if not self.terminate and self.termination is not None:
            raise ValueError("termination reason set without terminate=True")
        return self


class SimulatorConfig(Frozen):
    provider: Literal["anthropic", "openai_compatible", "scripted"] = "anthropic"
    model: str = "claude-opus-5"
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "low"
    """Replaces what used to be a ``temperature`` field.

    Current Claude models reject ``temperature``, ``top_p`` and ``top_k`` with a
    400 — the parameter was removed, not deprecated, so a config carrying one
    would have failed every call. Effort is the current control, and ``low`` is
    right for a counterpart: it has to sound like a customer, not solve a
    problem, and effort is the main lever on both latency and spend.
    """
    max_tokens: Annotated[int, Field(ge=1)] = 2048
    """Caps thinking *and* response text together. Thinking is on by default on
    Claude Opus 5, so a value sized for a one-line utterance truncates."""
    base_url: str | None = None
    cassette_mode: Literal["off", "record", "replay"] = "off"
    """Determinism for an LLM counterpart is only achievable through record and
    replay. See ARCHITECTURE.md, "Determinism"."""
    cassette_path: str | None = None


# --------------------------------------------------------------------------- #
# Agent-under-test boundary
# --------------------------------------------------------------------------- #


class TokenUsage(Frozen):
    prompt_tokens: Annotated[int, Field(ge=0)] = 0
    completion_tokens: Annotated[int, Field(ge=0)] = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class AgentTurnView(Frozen):
    """The *only* shape in which conversation history reaches the agent.

    ``extra="forbid"`` is load-bearing: there is no field here for hidden state,
    and no way to add one at a call site. The no-leak test asserts against
    :meth:`from_turn` rather than against a prompt string, so it keeps holding
    when prompt formatting changes.
    """

    speaker: Literal[Speaker.USER, Speaker.AGENT, Speaker.TOOL]
    text: str

    @classmethod
    def from_turn(cls, turn: Turn) -> Self:
        if turn.speaker is Speaker.SYSTEM:
            raise ValueError("system turns are not part of the agent-visible history")
        return cls(speaker=turn.speaker, text=turn.delivered_text)


class PolicyTrace(Frozen):
    """Sampling detail an in-process policy attaches so a trajectory can be
    replayed into a GRPO buffer without a second forward pass. Absent for HTTP
    agents, which is fine — you cannot train those anyway."""

    token_ids: tuple[int, ...] = ()
    logprobs: tuple[float, ...] = ()
    prompt_token_ids: tuple[int, ...] = ()


class AgentRequest(Frozen):
    episode_id: str
    turn_index: int
    history: tuple[AgentTurnView, ...]
    tools: tuple[ToolSpec, ...]
    brief: str = ""
    pending_tool_results: tuple[ToolResult, ...] = ()
    """Results of the calls the agent made earlier in this same turn. The agent
    is re-invoked with these until it emits text, so tool use is a loop inside a
    turn, not a turn of its own."""


class AgentResponse(Frozen):
    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    usage: TokenUsage | None = None
    latency_ms: float = 0.0
    policy: PolicyTrace | None = None
    error: str | None = None

    @model_validator(mode="after")
    def _says_something(self) -> Self:
        if not self.text and not self.tool_calls and self.error is None:
            raise ValueError("an agent response must contain text, tool calls, or an error")
        return self


# --------------------------------------------------------------------------- #
# Turn and trajectory
# --------------------------------------------------------------------------- #


class Turn(Mutable):
    """One speaker act, plus everything that happened because of it."""

    index: int
    speaker: Speaker
    text: str
    """As produced by the speaker. For a user turn this is the clean utterance —
    the customer knows what they said even when the agent mishears it."""
    delivered_text: str
    """As received by the counterpart. Differs from ``text`` only when the noise
    wrapper touched it."""
    noise: NoiseTrace | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)
    usage: TokenUsage | None = None
    latency_ms: float = 0.0
    internal_state: SimulatorInternalState | None = None
    """Populated on user turns only. Excluded from the agent view by type."""
    policy: PolicyTrace | None = None


class CostSummary(Frozen):
    """Agent cost and harness cost are tracked apart on purpose.

    The cost penalty shapes ``agent_*`` only. Charging a policy for the tokens
    its *counterpart* spent teaches it to make the simulator terse, which is not
    a behaviour anyone wants to buy.
    """

    turns: int = 0
    agent_tokens: TokenUsage = TokenUsage()
    simulator_tokens: TokenUsage = TokenUsage()
    agent_usd: float = 0.0
    simulator_usd: float = 0.0
    price_table_id: str = "unpriced"

    @property
    def total_usd(self) -> float:
        return self.agent_usd + self.simulator_usd


class EpisodeSeeds(Frozen):
    """Every random draw in an episode descends from one integer.

    Derivation is by hash rather than by a shared generator so that adding a
    call site — one more noise op, one more world coin flip — cannot shift the
    stream consumed by every other component.
    """

    root: Seed
    world: Seed
    simulator: Seed
    noise: Seed

    @classmethod
    def derive(cls, run_seed: int, scenario_id: str, persona_id: str, episode_index: int) -> Self:
        root = _hash_seed(f"{run_seed}|{scenario_id}|{persona_id}|{episode_index}")
        return cls(
            root=root,
            world=_hash_seed(f"{root}|world"),
            simulator=_hash_seed(f"{root}|simulator"),
            noise=_hash_seed(f"{root}|noise"),
        )

    def for_turn(self, channel: str, turn_index: int) -> int:
        return _hash_seed(f"{self.root}|{channel}|{turn_index}")


def _hash_seed(material: str) -> int:
    return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "big") >> 1


class Trajectory(Mutable):
    """One episode, complete enough to replay offline and to feed a rollout
    buffer. Serialises to one JSONL line."""

    episode_id: str
    scenario_id: str
    persona_id: str
    seeds: EpisodeSeeds
    config_digest: str
    harness_version: str
    created_at: datetime
    turns: list[Turn] = Field(default_factory=list)
    initial_world: WorldSnapshot
    final_world: WorldSnapshot | None = None
    termination: TerminationReason | None = None
    cost: CostSummary = CostSummary()

    def agent_view(self) -> tuple[AgentTurnView, ...]:
        """The projection handed to the adapter. One choke point, tested."""
        return tuple(
            AgentTurnView.from_turn(t) for t in self.turns if t.speaker is not Speaker.SYSTEM
        )


# --------------------------------------------------------------------------- #
# Verification and reward
# --------------------------------------------------------------------------- #


class ClaimCheck(Frozen):
    """One factual assertion the agent made, and what the world says about it."""

    turn_index: int
    kind: ClaimKind
    surface: str
    """The span as the agent wrote it, kept for hand-inspection of the verifier."""
    parsed_value: JSONValue = None
    ground_truth: JSONValue = None
    verdict: ClaimVerdict
    grounded_in_call_id: str | None = None
    """The tool result that entitles the agent to say this, if any."""
    bound_field: str | None = None
    """The ground-truth field the typed grammar bound this claim to, when it was
    layer 1 that matched. Lets an evidence requirement ask for a correct
    statement of *this specific policy* rather than of any policy at all."""


class CheckResult(Frozen):
    check_id: str
    description: str
    passed: bool
    severity: Severity
    detail: JSONObject = Field(default_factory=dict)
    tag: FailureTag | None = None


class RewardComponent(Frozen):
    name: str
    raw: float
    weight: float
    detail: JSONObject = Field(default_factory=dict)

    @property
    def weighted(self) -> float:
        return self.raw * self.weight


class RewardConfig(Frozen):
    """Weights are data, not constants in the verifier, so a sweep can vary them
    and the digest can record which set produced a number."""

    w_task_success: float = 1.0
    w_field_accuracy: float = 0.3
    w_forbidden_mutation: float = -1.0
    w_claim_accuracy: float = 0.5
    w_termination: float = 0.2
    w_cost: float = -0.1
    cost_shaping_enabled: bool = False
    cost_reference_turns: Annotated[int, Field(ge=1)] = 8
    cost_reference_tokens: Annotated[int, Field(ge=1)] = 4000
    unparsed_policy: Literal["neutral", "penalise"] = "neutral"
    """What happens to a claim the grammar could not bind. ``neutral`` drops it
    from the claim-accuracy denominator and counts it against
    ``Scorecard.claim_coverage``; ``penalise`` scores it as incorrect. Approved
    default is ``neutral`` — see DESIGN_NOTE.md."""

    @property
    def digest(self) -> str:
        return digest_of(self)


class RewardBreakdown(Frozen):
    """Named components plus a scalar that is arithmetically bound to them.

    The validator is the point. A reward that can be reported as 0.8 while its
    parts sum to 0.3 is worse than no reward at all, and that divergence is
    exactly what creeps in during a refactor.
    """

    components: tuple[RewardComponent, ...]
    scalar: float
    cost_shaping_enabled: bool
    config_digest: str

    @model_validator(mode="after")
    def _scalar_matches_parts(self) -> Self:
        expected = sum(c.weighted for c in self.components)
        if abs(expected - self.scalar) > 1e-9:
            raise ValueError(f"scalar {self.scalar} != sum of weighted components {expected}")
        return self

    def component(self, name: str) -> RewardComponent | None:
        return next((c for c in self.components if c.name == name), None)

    @classmethod
    def from_components(cls, components: tuple[RewardComponent, ...], config: RewardConfig) -> Self:
        return cls(
            components=components,
            scalar=sum(c.weighted for c in components),
            cost_shaping_enabled=config.cost_shaping_enabled,
            config_digest=config.digest,
        )


class Scorecard(Frozen):
    """The eval product's unit of output; also the RL loop's diagnostic record.
    Both consumers read the same object, which is the whole point of the design.
    """

    episode_id: str
    scenario_id: str
    persona_id: str
    seeds: EpisodeSeeds
    passed: bool
    checks: tuple[CheckResult, ...]
    claim_checks: tuple[ClaimCheck, ...] = ()
    claim_coverage: Annotated[float, Field(ge=0.0, le=1.0)] = 1.0
    """Fraction of checkable-looking spans the grammar actually bound. Falling
    coverage is how a verifier goes quietly blind; the sweep reports it."""
    failures: tuple[FailureTag, ...] = ()
    termination: TerminationReason | None = None
    reward: RewardBreakdown
    cost: CostSummary
    verifier_version: str

    @property
    def critical_failures(self) -> tuple[CheckResult, ...]:
        return tuple(c for c in self.checks if not c.passed and c.severity is Severity.CRITICAL)


# --------------------------------------------------------------------------- #
# Run configuration
# --------------------------------------------------------------------------- #


class AdapterConfig(Frozen):
    kind: Literal["http", "local", "scripted"] = "http"
    endpoint: str | None = None
    model: str | None = None
    timeout_s: float = 30.0
    max_tool_iterations: Annotated[int, Field(ge=1)] = 6
    headers: dict[str, str] = Field(default_factory=dict)


class RunConfig(Frozen):
    """Everything that must be identical for two runs to be comparable. Its
    digest is stamped into every trajectory and every scorecard."""

    run_seed: Seed = 0
    episodes: Annotated[int, Field(ge=1)] = 1
    scenario_ids: tuple[str, ...] = ()
    persona_ids: tuple[str, ...] = ()
    noise: NoiseConfig = NoiseConfig()
    simulator: SimulatorConfig = SimulatorConfig()
    adapter: AdapterConfig = AdapterConfig()
    reward: RewardConfig = RewardConfig()
    output_dir: str = "results"

    @property
    def digest(self) -> str:
        return digest_of(self)


# --------------------------------------------------------------------------- #
# Red-team
# --------------------------------------------------------------------------- #


class ActiveTarget(Frozen):
    """One fact the red team is currently trying to verify."""

    field: str
    true_value: str
    suspicion_level: Literal["low", "medium", "high"] = "low"


class Casefile(Mutable):
    """Strategic state shared between the red-team Speaker and Analyst."""

    confirmed_facts: list[str] = Field(default_factory=list)
    discrepancies: list[str] = Field(default_factory=list)
    pending_clarification: str | None = None
    active_targets: list[ActiveTarget] = Field(default_factory=list)
    next_move: str | None = None
    cracked: bool = False


class ClientBeliefs(Frozen):
    """The simulated client's private (possibly false) beliefs, keyed by topic."""

    facts: dict[str, str] = Field(default_factory=dict)


class RedTeamEpisodeResult(Frozen):
    """Outcome of one red-team caller vs. one simulated client episode."""

    cracked: bool
    transcript: tuple[Turn, ...]
    casefile: Casefile


# Forward references resolved after every model in the module exists.
SimulatorOutput.model_rebuild()
SimulatorTurnView.model_rebuild()
SimulatorContext.model_rebuild()
AgentTurnView.model_rebuild()
