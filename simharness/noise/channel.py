"""The voice channel: ASR-style corruption of the customer's utterance.

Deterministic under a seed — same (text, config, seed, turn) gives byte-identical
output, which is what lets an episode be replayed exactly.

**Scope, honestly.** The confusion table below is hand-written, not derived. The
design calls for deriving homophone candidates by phoneme distance over CMUdict;
`homophones_for` is the seam where that goes, and it falls back to the table when
`cmudict` is not installed. What is here today is weaker than the real thing, and
the WER parameter is therefore *a knob on this model*, not a measured equivalence
to any real ASR system. Do not report a WER figure from a sweep as if a real
recogniser would produce the same errors at that rate.

The one thing it does get right is priority: numbers are corrupted preferentially,
because fifteen/fifty is one phoneme apart and an order of magnitude apart in
consequence, and because a corrupted number is the only error whose downstream
effect the verifier can actually see.

Filler rates are calibrated against real speech — see rl-gym's
`rl_gym/voice/DATA.md` for the measurement (SpokenWOZ, 83,296 turns: 20.9% of
turns carry a filler; um/uh/like dominate).
"""

from __future__ import annotations

import hashlib

from simharness.schemas import (
    NoiseConfig,
    NoiseOp,
    NoiseOpKind,
    NoiseTrace,
    SpeechProfile,
)

NUMBER_CONFUSIONS: dict[str, str] = {
    "fifteen": "fifty", "fifty": "fifteen",
    "sixteen": "sixty", "sixty": "sixteen",
    "seventeen": "seventy", "seventy": "seventeen",
    "eighteen": "eighty", "eighty": "eighteen",
    "nineteen": "ninety", "ninety": "nineteen",
    "fourteen": "forty", "forty": "fourteen",
    "thirteen": "thirty", "thirty": "thirteen",
    "two": "to", "four": "for", "eight": "ate", "one": "won",
    "six": "sixty", "seven": "seventy", "ten": "tan",
}  # fmt: skip

# Frequency-weighted from real calls; the repetition is the weighting.
FILLERS: tuple[str, ...] = ("um", "um", "um", "uh", "uh", "uh", "like", "like", "well", "mm")


def homophones_for(word: str) -> tuple[str, ...]:
    """Candidate mishearings for a word.

    Extension point for the CMUdict phoneme-distance derivation. Today it is a
    table lookup; when `cmudict` is a hard dependency this becomes "all words
    within edit distance 1 in phoneme space", and the table becomes the fallback
    for words CMUdict does not carry.
    """
    swap = NUMBER_CONFUSIONS.get(word.lower())
    return (swap,) if swap else ()


def _draw(seed: int, key: str) -> float:
    digest = hashlib.sha256(f"{seed}|{key}".encode()).digest()
    return int.from_bytes(digest[:6], "big") / float(1 << 48)


def corrupt(
    text: str,
    config: NoiseConfig,
    seed: int,
    turn: int,
    speech: SpeechProfile | None = None,
) -> tuple[str, NoiseTrace]:
    """Returns the delivered text and a trace explaining every change."""
    speech = speech or SpeechProfile()
    wer = min(1.0, config.target_wer * speech.wer_multiplier)
    tokens = text.split()
    if wer <= 0 or not tokens:
        return text, NoiseTrace(seed=seed, target_wer=config.target_wer, measured_wer=0.0)

    out: list[str] = []
    ops: list[NoiseOp] = []
    for position, token in enumerate(tokens):
        key = f"{turn}:{position}"
        roll = _draw(seed, key)
        bare = token.strip(".,!?").lower()
        tail = token[len(bare) :] if token.lower().startswith(bare) else ""

        candidates = homophones_for(bare)
        digit_rate = wer * (2.0 + speech.digit_error_bias * 2.0)
        if candidates and roll < digit_rate:
            swapped = candidates[int(_draw(seed, key + ":h") * len(candidates))]
            out.append(swapped + tail)
            ops.append(
                NoiseOp(
                    kind=NoiseOpKind.DIGIT if bare.isdigit() else NoiseOpKind.HOMOPHONE,
                    token_index=position,
                    before=token,
                    after=swapped + tail,
                )
            )
            continue

        if roll < wer * 0.4:
            ops.append(NoiseOp(kind=NoiseOpKind.DROP, token_index=position, before=token, after=""))
            continue

        if roll > 1 - wer * 0.3:
            filler = FILLERS[int(_draw(seed, key + ":f") * len(FILLERS))]
            out.append(filler)
            ops.append(
                NoiseOp(kind=NoiseOpKind.FILLER, token_index=position, before="", after=filler)
            )
        out.append(token)

    if speech.truncation_bias and _draw(seed, f"{turn}:trunc") < speech.truncation_bias:
        keep = max(1, int(len(out) * (1 - config.max_truncation_fraction)))
        if keep < len(out):
            ops.append(
                NoiseOp(
                    kind=NoiseOpKind.TRUNCATE,
                    token_index=keep,
                    before=" ".join(out[keep:]),
                    after="",
                )
            )
            out = out[:keep]

    delivered = " ".join(out) if out else text
    changed = sum(1 for op in ops if op.kind is not NoiseOpKind.FILLER)
    return delivered, NoiseTrace(
        seed=seed,
        target_wer=config.target_wer,
        measured_wer=changed / len(tokens),
        ops=tuple(ops),
    )
