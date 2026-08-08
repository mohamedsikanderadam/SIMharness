# What this is for

SIMharness audits conversations against what a business has actually published
and actually instructed. It reads call logs, checks every claim against a fact
sheet, checks every call against a script, and grades the result 0-100 per
category with an overall letter.

The tool does not care whether the voice on the other end of the call was
synthetic. That is the whole point of this document.

---

## 1. The obvious case: a business deploying an AI agent

A business puts a voice agent on its phone line. Within a week it is handling
more calls than the team ever did, and nobody knows what it is saying.

The current state of the art is a dashboard: call volume, average handle time,
"customer satisfaction" from a survey two percent of callers complete. None of
it answers the only question the owner actually has, which is *is it telling
people the truth?*

SIMharness answers that specific question:

| Question the owner asks | What the report shows |
| --- | --- |
| Is it quoting the right prices? | Every price the agent said, checked against the catalogue, with the call and turn it was said on |
| Is it inventing policies? | Cancellation windows, deposits, party sizes, opening hours, each compared to the published fact |
| Is it promising things we cannot do? | Free upgrades, waived fees, guarantees - matched against a do-not-say list |
| Is it confirming bookings that do not exist? | Confirmations with no corresponding tool call, flagged as ungrounded |
| Is it pretending to be human? | Flagged critical. So is reading a card number back |
| Is it slow, or does it repeat itself? | Latency percentiles and dead air where the log has timing; re-asks and repeats always |
| Is it converting? | Taken from the source system's outcome, never inferred from the agent sounding pleased with itself |

Each finding carries the exact quote. An owner who disagrees with a grade can
read the sentence that produced it. That is the difference between a score and
an argument you can settle.

---

## 2. The bigger case: humans in an outsourced contact centre

A transcript is a transcript. `Representative: You can cancel free of charge up
to 24 hours before arrival` is wrong in precisely the same way, and for
precisely the same reason, as an AI agent saying it. The auditor does not need
to know which one it was reading.

This matters because the outsourced-service industry has a measurement problem
that is *older and larger* than the AI one.

### What QA looks like today

A brand outsources its customer service to a BPO. The contract specifies a
script, an SOP, a fact sheet, and a set of compliance obligations. Then:

- The centre handles **millions of calls a month**.
- QA listens to **a handful per agent per month** - in most operations well
  under one percent of volume.
- Those few calls are scored by a human with a spreadsheet, subjectively, and
  usually **weeks after the call happened**.
- The agent being coached has no memory of the call. The customer is long gone.
- The brand sees a monthly deck built from that sample and is asked to believe
  it represents the other 99%.

Every serious problem in this arrangement comes from the same root: **the sample
is too small to find anything rare, and rare things are what hurt.** A rep who
misquotes the refund window on one call in fifty is invisible to a 1% sample and
extremely visible to a regulator.

### What changes when you audit all of it

Running every transcript through the same pipeline turns each of those into
something concrete:

**Script adherence, measured rather than sampled.** The centre's own script is
data - a JSON list of required behaviours. Did the rep use the approved
greeting? Give the call-recording notice? Complete the identity check *before*
discussing the account? Ask if anything else was needed before closing? Each one
is a rule, checked on every call, reported as a rate with the failures quoted.

**SOP and fact-sheet drift, caught early.** A brand updates its cancellation
policy from 24 to 48 hours. The email goes out. Two weeks later, how many reps
are still quoting the old number? Today: nobody knows until a customer
complains. With a full-volume audit: it is a count, per rep, from the day the
fact sheet changed. Fact drift is the single most common failure in outsourced
service and it is completely invisible to sampling.

**Compliance obligations, evidenced.** Recording notices, disclosure
requirements, "do not give advice you are not licensed to give", "never read a
card number back". These are the findings that carry legal weight, and a
critical one caps the grade regardless of how good the other numbers look. The
report is the evidence pack: quote, call ID, turn index, timestamp.

**Fair, comparable agent scoring.** Every rep is measured on the same rubric,
on all of their calls, not on the three a supervisor happened to pull. Coaching
stops being "I listened to a call and I think you rushed her" and becomes "you
quoted the wrong deposit on 6 of 340 calls, here they are."

**Vendor accountability that survives a contract dispute.** A brand auditing its
BPO, or a BPO proving its own performance to the brand, both get the same
artefact: a graded report with per-call drill-down, generated from the raw logs
by a rubric neither side can quietly adjust after the fact. The rubric is
versioned and its digest is printed on every report.

**Triage instead of trawling.** With millions of calls, the useful output is not
a score - it is *which fifty calls should a human listen to today*. The report
ranks by severity and quotes the offending line, so QA time goes to the calls
that actually contain a problem.

### How the same tool does both

Nothing had to be bolted on for this. The pieces that make it work were already
the design:

- **Ingest is speaker-label agnostic.** `Representative:`, `Advisor:`,
  `Operator:`, `CSR:`, `Agent:` all normalise to the same role. Plain text with
  `[hh:mm:ss]` stamps is a first-class input, because that is what call-recording
  platforms export.
- **The rule packs are data, not code.** `--policy-rules` and `--script` take
  the client's own do-not-say list and SOP as JSON. The built-in packs are a
  starting point, not the standard.
- **Unmeasurable stays unmeasured.** If the export has no timestamps, latency
  and dead air are reported as unavailable, not as zero. A QA report that
  invents a clean number is worse than one that admits the gap - and in a vendor
  dispute, it is worthless.
- **The LLM judge is optional and labelled.** Deterministic checks - facts,
  script, compliance, timing - run offline and reproducibly. Anything an LLM
  judged is marked as judged, weighted lower, and can never create a critical
  finding. A grade you cannot reproduce is a grade you cannot defend.

Try it:

```bash
python scripts/generate_audit_report.py \
    --logs examples/callcentre/calls \
    --business examples/marina_bay.json \
    --script examples/callcentre/script.json \
    --out audit/
```

Two human-agent transcripts, audited against a client script. One rep is clean;
the other skips the recording notice, discusses a booking before verifying the
caller, closes without asking if anything else was needed, and quotes a 24-hour
cancellation window against a published 48. All four are in the report, quoted,
with the call and turn they came from.

---

## 3. Where this goes

The same pipeline reaches further than either case above, because the input is
just "a conversation with a business on one side of it":

- **Mixed human/AI operations.** Most contact centres will run both for years:
  AI on tier one, humans on escalation. One audit, one rubric, one report - so
  the comparison is finally apples to apples, and the deflection decision is
  made on evidence rather than vendor slides.
- **Pre-deployment gating.** Point the same rubric at a red-team run instead of
  production logs and it becomes a release gate: an agent does not go live until
  it audits clean. `--fail-under` already exits non-zero for CI.
- **Regression detection.** The report is deterministic without the judge, so
  nightly runs are diffable. A prompt change that quietly breaks the deposit
  answer shows up as a category dropping overnight, not as a complaint next
  month.
- **Chat, email and messaging.** Nothing in the fact checking is voice-specific.
  Only the timing metrics need audio; the rest works on any transcript.
- **Sales and collections, not just support.** Any conversation with rules -
  what may be promised, what must be disclosed, what must never be said - is the
  same problem wearing a different job title.

---

## The short version

Every business that talks to its customers at scale has the same unanswered
question: *are the things being said actually true, and actually allowed?*

Sampling cannot answer it, because the failures are rare and the sample is
small. A dashboard cannot answer it, because handle time is not honesty. This
project answers it on every call, cites the sentence, and produces a grade that
survives someone disagreeing with it.

Whether the voice was a person or a model is, for this purpose, a detail.
