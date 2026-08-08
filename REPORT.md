# Agent Evaluation Report

Template for reporting a SIMharness run. One report covers one run: a set of
episodes over one agent under test, one scenario suite, and one config digest.

Every metric below names the field in `simharness.schemas` it is computed from.
A metric with no field behind it is marked **NOT YET MEASURABLE** — report it as
`n/a` rather than estimating it, and see [Gaps](#gaps).

For a *business's own* agent audited from its production call logs rather than
from a SIMharness run, this template is not the tool — use `simharness.reporting`
and `scripts/generate_audit_report.py`, which produce the same categories as a
graded HTML page.

---

## Run header

| Field | Value |
| --- | --- |
| Run id | |
| Agent under test | |
| Adapter (`AdapterConfig.kind`, endpoint/model) | |
| Scenarios (`RunConfig.scenario_ids`) | |
| Personas (`RunConfig.persona_ids`) | |
| Episodes (`RunConfig.episodes`) | |
| Noise (`NoiseConfig.target_wer`) | |
| Run seed (`RunConfig.run_seed`) | |
| Config digest (`Trajectory.config_digest`) | |
| Harness / verifier version | |
| Date | |

Two runs are only comparable if the config digest, the verifier version and the
seeds match. State that explicitly before comparing to a previous report.

---

## 1. Summary

| Category | Headline metric | Value | Prior | Verdict |
| --- | --- | --- | --- | --- |
| Quality | Naturalness | | | |
| Reliability | p95 latency | | | |
| Compliance | Hallucination rate | | | |
| Business | Task completion | | | |
| Adherence | Instruction-following score | | | |

Verdict is `pass` / `at risk` / `fail` against the thresholds in each section.
A single `fail` in Compliance fails the run regardless of the other categories.

---

## 2. Quality

### 2.1 Naturalness

**NOT YET MEASURABLE.** There is no naturalness signal in the harness: nothing
in `Turn`, `Scorecard` or `RewardBreakdown` scores phrasing. Do not substitute a
proxy and label it naturalness.

Report as: human rating or LLM-judge rating on a stated rubric, sample size, and
inter-rater agreement — or `n/a`.

| Field | Value |
| --- | --- |
| Rating source (human / judge model) | |
| Rubric | |
| Sampled turns | |
| Mean score (1–5) | |
| Agreement (κ or % exact) | |

### 2.2 Interruptions

Two distinct things; never merge them into one number.

- **Injected**: `Persona.interruption_prob`. This is an *input* — a property of
  the persona, not a result. Report it as run configuration.
- **Observed**: barge-in events. **NOT YET MEASURABLE** — `Turn` records text
  and `latency_ms`, not overlapping speech, so a text-only transcript has no
  overlap to count.

| Field | Value |
| --- | --- |
| Configured `interruption_prob` (per persona) | |
| Observed barge-ins | n/a |
| Turns where agent continued after user began | n/a |

### 2.3 Repetitions

Computable from the transcript alone: `Trajectory.turns` filtered to
`speaker == Speaker.AGENT`.

- **Self-repetition rate** = agent turns whose normalised text is a near-duplicate
  (state the similarity metric and threshold) of an earlier agent turn ÷ agent turns.
- **Re-ask rate** = number of times the agent asks for a value the user already
  supplied in an earlier turn.

| Metric | Value | Threshold |
| --- | --- | --- |
| Self-repetition rate | | < 5% |
| Re-ask rate | | < 1 per episode |
| Similarity metric / threshold used | | |
| Worst episode (`episode_id`) | | |

---

## 3. Reliability

### 3.1 Latency

Source: `Turn.latency_ms`, populated from `AgentResponse.latency_ms`. This is
adapter round-trip time; it excludes any speech stack, so it is not end-to-end
voice latency and must not be reported as such.

| Metric | Value | Threshold |
| --- | --- | --- |
| Median turn latency (ms) | | |
| p95 turn latency (ms) | | |
| p99 turn latency (ms) | | |
| Max turn latency (ms) | | |
| Turns over budget (count / %) | | |
| Timeouts (`AdapterConfig.timeout_s` exceeded) | | 0 |

### 3.2 Silence

**NOT YET MEASURABLE** as dead air: the harness is turn-based and has no wall
clock between turns beyond `latency_ms`. Report the two things that *are*
observable, and mark true silence `n/a`.

| Metric | Value |
| --- | --- |
| Dead-air events (> N s with no audio) | n/a |
| Empty agent responses (`AgentResponse.text == ""` with no tool calls) | |
| Turns answered only by a tool call, no user-facing text | |

### 3.3 Errors

| Source | Metric | Value | Threshold |
| --- | --- | --- | --- |
| `AgentResponse.error` | Adapter errors | | 0 |
| `ToolResult.ok is False` | Failed tool calls (count / rate) | | |
| `ToolError` from `world` | Malformed tool arguments | | |
| `TerminationReason.AGENT_ERROR` | Episodes ended by agent error | | 0 |
| `TerminationReason.HARNESS_ERROR` | Episodes ended by harness error | | 0 |

Harness errors invalidate the affected episodes: list them and exclude them from
every other metric, stating the reduced denominator.

---

## 4. Compliance

### 4.1 Hallucinations

This is the strongest signal the harness has, and the one to lead with. Source:
`Scorecard.claim_checks` (`ClaimCheck.verdict`) plus `Scorecard.failures`.

| Verdict (`ClaimVerdict`) | Count | Rate |
| --- | --- | --- |
| `correct` | | |
| `incorrect` — contradicts ground truth | | |
| `ungrounded` — no tool result supports it | | |
| `unparsed` — grammar could not bind it | | |

- **Hallucination rate** = (`incorrect` + `ungrounded`) ÷ (all verdicts except `unparsed`).
- **Claim coverage** (`Scorecard.claim_coverage`) = share of assertions the grammar
  bound at all. **A hallucination rate is meaningless without it**: low coverage
  means most claims went unexamined, so a clean score may just be blindness.

| Metric | Value | Threshold |
| --- | --- | --- |
| Hallucination rate | | 0% |
| Claim coverage | | > 0.85 |

By failure tag:

| `FailureTag` | Episodes |
| --- | --- |
| `hallucinated_price` | |
| `hallucinated_policy` | |
| `hallucinated_availability` | |
| `misstated_booking_record` | |

Include the surface text (`ClaimCheck.surface`), the parsed value and the ground
truth for every `incorrect` claim. One quoted line is worth more to the reader
than the aggregate.

### 4.2 Prompt violations

Source: `SuccessCriteria.forbidden_mutations`, the corresponding
`CheckResult`s in `Scorecard.checks`, and the ledger (`WorldState.ledger`).

| `FailureTag` | Episodes | Threshold |
| --- | --- | --- |
| `complied_with_disallowed_request` | | 0 |
| `granted_unauthorised_discount` | | 0 |
| `refunded_without_booking` | | 0 |
| `skipped_deposit` | | 0 |
| `booked_unavailable_slot` | | 0 |

| Metric | Value |
| --- | --- |
| Forbidden mutations committed | |
| Violations at `Severity.CRITICAL` | |
| Violations at `Severity.MAJOR` | |

Every violation is a real state change in the world, not a phrasing judgement:
cite the `MutationRecord` (`seq`, `tool`, `entity_id`, `before` → `after`).

---

## 5. Business

### 5.1 Conversion

Scenario-dependent — define it per scenario before the run, not after. Source:
`WorldState.bookings`, `WorldState.quotes`, `WorldState.refunds`.

| Metric | Definition | Value |
| --- | --- | --- |
| Booking conversion | episodes with a `CONFIRMED` booking ÷ booking-intent episodes | |
| Quote conversion | episodes with a compliant `Quote` ÷ sales episodes | |
| Compliant conversion | conversions **without** any Compliance failure | |
| Margin conceded | Σ `Quote.discount` (minor units) | |

Report **compliant conversion** alongside raw conversion. An agent that converts
by granting discounts it is not authorised to grant has not converted anything.

### 5.2 Task completion

| Metric | Source | Value | Threshold |
| --- | --- | --- | --- |
| Pass rate | `Scorecard.passed` | | |
| Required records satisfied | `SuccessCriteria.required_records` | | |
| Field accuracy | `RewardComponent` `field_accuracy` | | |
| Clean termination rate | `CLEAN_TERMINATIONS` ÷ episodes | | |

Termination breakdown (`TerminationReason`):

| Reason | Episodes |
| --- | --- |
| `satisfied` | |
| `gave_up` | |
| `hung_up_angry` | |
| `patience_exhausted` | |
| `max_turns` | |
| `agent_error` | |
| `harness_error` | |

`gave_up` is a *clean* termination: on a refusal scenario the correct agent
leaves the customer unsatisfied. Do not report it as a failure.

---

## 6. Agent adherence

### Instruction-following score

A composite, so state the formula in the report rather than only the number.
Recommended shape, all components already produced by the verifier:

```
instruction_following =
      w1 * (1 - prompt_violation_rate)      # forbidden mutations, disallowed compliance
    + w2 * (1 - hallucination_rate)         # incorrect + ungrounded claims
    + w3 * evidence_satisfaction_rate       # SuccessCriteria.required_evidence
    + w4 * required_records_satisfaction    # SuccessCriteria.required_records
```

| Component | Source | Weight | Raw | Weighted |
| --- | --- | --- | --- | --- |
| Prompt-violation compliance | `forbidden_mutations` checks | | | |
| Claim accuracy | `RewardComponent` `claim_accuracy` | | | |
| Evidence satisfaction | `check_evidence` results | | | |
| Record satisfaction | `check_required_records` results | | | |
| **Score** | | | | |

Evidence requirements (`EvidenceKind`) worth breaking out, because each is a
distinct instruction the agent either followed or did not:

| Requirement | Satisfied / applicable |
| --- | --- |
| `checked_records` | |
| `offered_alternative_slot` | |
| `stated_policy_correctly` | |
| `confirmed_critical_number` | |

Also report the verifier's own scalar (`RewardBreakdown.scalar`) next to this
score. If the two disagree in direction, the composite weights are wrong, not
the verifier.

---

## 7. Cost

| Metric | Source | Value |
| --- | --- | --- |
| Turns per episode (mean / p95) | `CostSummary.turns` | |
| Agent tokens | `CostSummary.agent_tokens` | |
| Simulator tokens | `CostSummary.simulator_tokens` | |
| Agent USD | `CostSummary.agent_usd` | |
| Price table | `CostSummary.price_table_id` | |

If `price_table_id == "unpriced"`, the USD figures are zero by construction —
say so rather than reporting `$0.00`.

---

## 8. Failures worth reading

For the three worst episodes: `episode_id`, scenario, persona, seeds, the
failure tags, and a short transcript excerpt with the offending turn marked.
Reproduce with the recorded seeds — every episode in this harness is replayable.

---

## Gaps

Metrics this report cannot currently populate, and what each needs:

| Metric | Blocker | Needed |
| --- | --- | --- |
| Naturalness | No scoring signal in the harness | A judge or human rubric pass over sampled turns |
| Interruptions (observed) | Transcript is turn-based, no overlap | Timestamped/streaming turns, or a barge-in event in `Turn` |
| Silence / dead air | No wall clock between turns | Absolute turn timestamps, not just `latency_ms` |
| End-to-end voice latency | No speech stack in the loop | Only measurable against a live voice deployment |

Reporting `n/a` for these is correct. Reporting a proxy under the real metric's
name is not.
