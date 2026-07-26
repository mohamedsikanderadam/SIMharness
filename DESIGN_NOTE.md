# The one contestable decision

**What the verifier does with a factual claim its grammar could not parse.**

## Why this is the decision that matters

Your spec settles the big question already: claims are extracted by regex and a
constrained grammar, never by asking a model. I agree, and Phase 1 will build it
that way.

What the spec does not settle is the consequence. A grammar has recall below 1,
and always will. All of these are the same claim:

```
"the deposit is £15 per person"          → parses
"that'll be fifteen pounds a head"       → parses if number words are in scope
"we'd want about fifteen quid up front"  → maybe
"it's the standard one-five per cover"   → no
"there's a small holding charge, nothing dramatic"  → never
```

Every span the grammar misses is scored one of two ways: ignored, or punished.
There is no third option, and the choice propagates into everything downstream —
it is the difference between an RL agent that learns to be accurate and one that
learns to be vague.

## The options

**A — ignore unparsed claims.** Simple, no false positives, and exactly what
most harnesses do. It is also a live reward hack. Under GRPO the policy is
searching for anything that raises reward, and "phrase prices in a way the
grammar cannot bind" strictly dominates "state prices correctly": both avoid the
penalty, and the vague one cannot ever be caught being wrong. The failure is
silent and it points the wrong way — eval scores improve while the product gets
worse. This is the failure mode I would expect to actually happen, not a
theoretical one.

**B — penalise unparsed claims.** Kills the evasion incentive, but makes the
grammar normative. Every gap in our regexes becomes a training signal about
phrasing rather than truth, and the policy converges on speaking in the grammar's
canonical forms — a correct agent that sounds like a form letter. For the eval
product it is worse than that: we would be marking down a third party's voice
agent for failing to match a style we never published.

**C — ground by construction, ignore the residue, and gate on coverage.**
My recommendation.

## Recommendation: C

Two layers, because "claim" is not one kind of thing.

**Layer 1 — numeric grounding.** Numbers are a closed class, regex handles them
at near-perfect recall (digits, number words, currency, times, dates), and
numbers are where bookings actually break. So: every number-bearing span in an
agent utterance must resolve to a value the agent was entitled to say — one that
appears in a `ToolResult` it actually received this episode, or in the scenario's
public ground truth. Unmatched → `UNGROUNDED`, and that scores against the agent.

This is the recall-safe net. Paraphrase "fifteen quid" however you like; the
value `15` still has to be justified by something the agent was told. Evasion via
phrasing does not help, because the check keys on the number, not the sentence
around it. Evasion via *omitting the number* does not help either — a booking
confirmation with no numbers in it fails the required-records check on its own.

**Layer 2 — typed claim grammar** for the qualitative facts that have no number:
policy windows, opening-hours claims, "we don't take deposits". High precision,
lower recall. Parsed → `CORRECT` / `INCORRECT`. Unparsed → `UNPARSED`, scored
neutral.

**Then make the residue visible.** `Scorecard.claim_coverage` is bound spans over
candidate spans, reported per episode and aggregated per sweep cell. Falling
coverage is treated as a verifier regression, not a footnote. Concretely: if an
RL run's reward climbs while coverage drops, that is the evasion hack showing up
on an instrument instead of in production, and it is a stop-the-run condition.
This is the same shape as a std-collapse detector — a cheap metric whose only job
is to catch the reward going hollow.

## What I am accepting, stated plainly

A claim that is qualitative, unparsed, *and* false scores neutral. Example: "we
don't take deposits at all", phrased outside the grammar. Layer 1 does not catch
it because there is no number; layer 2 does not catch it because the grammar
missed it.

The mitigation is the judge in `diagnostics/`, pointed at the verifier rather
than at the agent: sample transcripts offline, have a model enumerate every
factual claim, diff that against what the grammar bound, and extend the grammar
where it missed. That measures our recall. It never touches the scalar and it
never scores an episode. I think this is the only defensible use of a judge in
this design, and it is why I want one in the tree at all.

## What I want from you

1. **Approve C**, or tell me you want A for v1 simplicity — it is a defensible
   call if the near-term priority is the eval product rather than training, since
   the hack only bites once a policy is optimising against the reward.
2. I would like to add `RewardConfig.unparsed_policy: "neutral" | "penalise"`,
   defaulting to `neutral`, so an RL run can measure the difference rather than
   argue about it. Cheap to build now, expensive to retrofit once trajectories
   exist with the old contract baked into their digests.

Not asking you to decide on anything else in Phase 0 — the rest of the design I
am confident in.
