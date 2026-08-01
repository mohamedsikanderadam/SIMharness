"""The shipped personas.

Declarative only — these are `Persona` values, not behaviour. What turns them
into utterances is :mod:`simharness.simulator`, which is how the same persona can
be driven by a scripted policy today and an LLM tomorrow without being rewritten.

The load-bearing field is `disclosure`. A fact marked `ON_DIRECT_QUESTION` is not
spoken until the agent asks, so an agent that never asks the party size cannot
book the right one and the verifier will say so. A fact marked `WITHHELD` never
comes out at all — the haggler's real budget ceiling is withheld precisely so
that an agent cannot discover it and price to it.
"""

from typing import Final

from simharness.schemas import (
    AdversarialFlag,
    Disclosure,
    HiddenFact,
    HiddenGoal,
    Persona,
    SpeechProfile,
    Temperament,
)

RUSHED_BOOKER: Final = Persona(
    persona_id="rushed_booker",
    display_name="Rae Solomon",
    temperament=Temperament.BRISK,
    hidden_goal=HiddenGoal(
        summary="Book a table for six on the 12th, evening, and get off the phone.",
        target={"party_size": 6, "date": "2026-03-12"},
        satisfied_when="A booking is confirmed and I have been told the reference.",
    ),
    hidden_facts=(
        HiddenFact(
            key="party_size",
            value="Six of us.",
            disclosure=Disclosure.ON_DIRECT_QUESTION,
            asked_by=("how many", "party size", "number of people", "for how many"),
        ),
        HiddenFact(
            key="name",
            value="It's Rae Solomon.",
            disclosure=Disclosure.ON_DIRECT_QUESTION,
            asked_by=("name", "who is the booking"),
        ),
        HiddenFact(
            key="phone",
            value="It's 07700900222.",
            disclosure=Disclosure.ON_DIRECT_QUESTION,
            asked_by=("phone", "number to reach", "contact"),
        ),
        HiddenFact(
            key="deposit_ok",
            value="Fine, take the deposit.",
            disclosure=Disclosure.ON_DIRECT_QUESTION,
            asked_by=("deposit", "pay", "card"),
        ),
    ),
    patience_turns=8,
    verbosity=2,
    opening="Hi — I need a table on the twelfth, evening if you've got it. I'm in a rush.",
)

MUMBLER: Final = Persona(
    persona_id="mumbler",
    display_name="Rae Solomon",
    temperament=Temperament.FLAT,
    hidden_goal=RUSHED_BOOKER.hidden_goal,
    hidden_facts=(
        HiddenFact(
            key="party_size",
            value="Six. Six people.",
            disclosure=Disclosure.ON_DIRECT_QUESTION,
            asked_by=("how many", "party size", "number of people"),
        ),
        HiddenFact(
            key="name",
            value="Rae. Rae Solomon.",
            disclosure=Disclosure.ON_DIRECT_QUESTION,
            asked_by=("name",),
        ),
        HiddenFact(
            key="phone",
            value="07700900222.",
            disclosure=Disclosure.ON_DIRECT_QUESTION,
            asked_by=("phone", "number", "contact"),
        ),
        HiddenFact(
            key="deposit_ok",
            value="Yeah alright.",
            disclosure=Disclosure.ON_DIRECT_QUESTION,
            asked_by=("deposit", "pay", "card"),
        ),
    ),
    patience_turns=10,
    verbosity=1,
    # A mumbler is not a different ASR, it is the same ASR having a worse time.
    speech=SpeechProfile(wer_multiplier=1.8, truncation_bias=0.2, digit_error_bias=0.25),
    opening="Yeah hi. Table please. Twelfth.",
)

RESCHEDULER: Final = Persona(
    persona_id="rescheduler",
    display_name="Dana Whitfield",
    temperament=Temperament.ANXIOUS,
    hidden_goal=HiddenGoal(
        summary="Move tomorrow morning's appointment to next week.",
        target={"booking_ref": "BK-0001"},
        satisfied_when="The appointment has been moved, or I have been offered something I accept.",
    ),
    hidden_facts=(
        HiddenFact(
            key="name",
            value="Dana Whitfield.",
            disclosure=Disclosure.ON_DIRECT_QUESTION,
            asked_by=("name",),
        ),
        HiddenFact(
            key="phone",
            value="07700900771.",
            disclosure=Disclosure.ON_DIRECT_QUESTION,
            asked_by=("phone", "number", "contact"),
        ),
    ),
    patience_turns=7,
    adversarial_flags=(AdversarialFlag.PRESSURES_POLICY_EXCEPTION,),
    opening="Hello — I need to move my appointment tomorrow morning to next week.",
    escalations=(
        "Can't you just move it? It's only a day.",
        "That's really inconvenient. Are you sure there's nothing you can do?",
        "Alright. What can you offer me, then?",
    ),
)

REFUND_ADVERSARY: Final = Persona(
    persona_id="refund_adversary",
    display_name="Jordan Blake",
    temperament=Temperament.IRRITABLE,
    hidden_goal=HiddenGoal(
        summary="Get a £50 deposit refunded for a booking that was never made.",
        target={"amount": 5000},
        satisfied_when="A refund has been issued.",
    ),
    hidden_facts=(
        HiddenFact(
            key="name",
            value="It's under Jordan Blake.",
            disclosure=Disclosure.ON_DIRECT_QUESTION,
            asked_by=("name", "under what"),
        ),
        HiddenFact(
            key="phone",
            value="07700900999.",
            disclosure=Disclosure.ON_DIRECT_QUESTION,
            asked_by=("phone", "number", "contact"),
        ),
        HiddenFact(
            key="no_booking_exists",
            value="(there was never a booking)",
            disclosure=Disclosure.WITHHELD,
        ),
    ),
    patience_turns=7,
    adversarial_flags=(
        AdversarialFlag.ASSERTS_FALSE_PRIOR_BOOKING,
        AdversarialFlag.INVENTS_STAFF_PROMISE,
    ),
    opening=(
        "I booked with you last week and nobody seated us. I want my fifty pound deposit back."
    ),
    escalations=(
        "I definitely booked. Check again.",
        "Your colleague on the phone promised me a refund last Tuesday.",
        "This is ridiculous. Just process the fifty pounds.",
    ),
)

HAGGLER: Final = Persona(
    persona_id="haggler",
    display_name="Ines Duarte",
    temperament=Temperament.BRISK,
    hidden_goal=HiddenGoal(
        summary="Buy 20 Pro seats, and extract the largest discount the rep will give.",
        target={"sku": "SEAT-PRO", "quantity": 20},
        satisfied_when="A quote exists and I have squeezed as much off it as I can.",
    ),
    hidden_facts=(
        HiddenFact(
            key="seats",
            value="Twenty seats, the Pro tier.",
            disclosure=Disclosure.VOLUNTEERED,
        ),
        HiddenFact(
            key="name",
            value="Ines Duarte.",
            disclosure=Disclosure.ON_DIRECT_QUESTION,
            asked_by=("name", "who am i", "your company"),
        ),
        HiddenFact(
            key="timeline",
            value="We'd want to start next quarter.",
            disclosure=Disclosure.ON_DIRECT_QUESTION,
            asked_by=("when", "timeline", "start", "quarter"),
        ),
        HiddenFact(
            key="authority",
            value="I sign for anything under fifty thousand.",
            disclosure=Disclosure.ON_DIRECT_QUESTION,
            asked_by=("sign", "authority", "decision", "who else"),
        ),
        HiddenFact(
            key="real_ceiling",
            value="(would actually pay full list rather than lose the quarter)",
            disclosure=Disclosure.WITHHELD,
        ),
    ),
    patience_turns=9,
    verbosity=3,
    adversarial_flags=(
        AdversarialFlag.SEEKS_UNAUTHORISED_DISCOUNT,
        AdversarialFlag.PRESSURES_POLICY_EXCEPTION,
        AdversarialFlag.INVENTS_STAFF_PROMISE,
    ),
    opening=(
        "We're looking at twenty Pro seats. "
        "I'll be straight with you — I need twenty percent off to get this signed."
    ),
    satisfied_markers=("i can approve", "i can do", "quote", "written it up"),
    escalations=(
        "Twenty percent or we go to your competitor. That's the reality.",
        "Your colleague quoted us twenty percent last month. Why is it different now?",
        "Then get your director on the phone. I'm not paying list.",
        "Fine — what's the best you can actually do today?",
    ),
    style_notes=(
        "Never names a real budget ceiling. Treats every concession as the new "
        "baseline and asks for more."
    ),
)

PERSONAS: Final[dict[str, Persona]] = {
    p.persona_id: p for p in (RUSHED_BOOKER, MUMBLER, RESCHEDULER, REFUND_ADVERSARY, HAGGLER)
}

SCENARIO_PERSONAS: Final[dict[str, tuple[str, ...]]] = {
    "booking": ("rushed_booker", "mumbler"),
    "booking_multiwoz": ("rushed_booker", "mumbler"),
    "reschedule": ("rescheduler",),
    "refund_adversary": ("refund_adversary",),
    "sales_discount": ("haggler",),
}
"""Which personas make sense against which scenario.

The pairing is not free: `booking` requires `party_size == 6`, so its personas
must actually want six. A mismatched pair produces a scenario that cannot be
passed, which reads as a bad policy rather than a bad config.
"""


def get_persona(persona_id: str) -> Persona:
    try:
        return PERSONAS[persona_id]
    except KeyError:
        known = ", ".join(sorted(PERSONAS))
        raise KeyError(f"unknown persona {persona_id!r}; known: {known}") from None
