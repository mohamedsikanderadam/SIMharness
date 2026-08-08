# SIMharness

Red-team dialogue harness and production call-audit toolkit for voice agents.

SIMharness does two things:

1. **Red-team simulator** — a caller probes an agent in a controlled conversation
   and tries to expose false or ungrounded claims, using only the published
   business fact sheet and the transcript.
2. **Production audit** — it ingests real call logs or recordings, checks every
   agent claim against the fact sheet and a required script, and produces a
   graded report for business owners and a JSON artefact for the technical team.

The question it answers is: *is the agent on the phone telling the truth?*

---

## What it does

A business can drop a voice agent onto a phone line and suddenly have hundreds
of calls a day. The existing dashboards show volume, handle time, and survey
scores; none of them show whether the agent quoted the right price, invented a
cancellation policy, or confirmed a booking that was never made.

SIMharness fills that gap. The red-team side runs a scripted caller (Speaker +
Analyst) against a simulated client before go-live, so a broken agent does not
reach customers. The audit side runs the same rubric over live logs, so problems
already in production are found with call IDs and quoted evidence instead of
hunting through a one-percent sample.

---

## Architecture

```text
Business facts (JSON / Context.dev)  ->  FactSheet
                                       |
Call logs / recordings  ->  ingest  ->  CallLog
                                       |
                                       v
              Red-team caller (Speaker + Analyst)
                         |              |
                         v              v
              Simulated client adapter   |
                         |               |
                         v               v
              Verifier (ground-truth)    |
                         |               |
                         v               v
              analyse_calls()  ->  AuditReport
                                       |
                           render_html()
                                       |
                   report.html  +  report.json
```

- `simharness/simulator/redteam.py` — the two-agent red-team caller.
- `simharness/adapters/` — client adapters and the ElevenLabs voice wrapper.
- `simharness/reporting/` — log ingestion, fact sheets, grading, and rendering.
- `simharness/verifier/` — ground-truth claim scoring.
- `simharness/world/` — tool side effects and business state.

---

## Quick start on a clean machine

```bash
git clone git@github.com:mohamedsikanderadam/SIMharness.git
cd SIMharness
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

If you are using `uv` instead of `pip`:

```bash
uv sync --frozen
```

### API keys

Create a `secrets.env` file in the repo root (already ignored by `.gitignore`):

```env
ELEVENLABS_API_KEY=...
CONTEXT_DEV_API_KEY=...
ANTHROPIC_API_KEY=...   # only if you use the optional LLM judge
```

The scripts load `secrets.env` automatically. If you are not running voice or
Context.dev, the deterministic scripts do not need any keys.

### Run the live fact-check demo

```bash
PYTHONPATH=. .venv/bin/python scripts/demo_live_voice.py --voice
```

A caller phones a hotel knowing none of its policies. The agent states a
cancellation deadline, the caller fetches that policy from the hotel's live page
through Context.dev, quotes the site back, and records a false claim if the agent
repeats itself. Drop `--voice` for text only; add `--honest` for the control run,
where a truthful agent is confirmed rather than accused.

Needs `CONTEXT_DEV_API_KEY`, and `ELEVENLABS_API_KEY` for `--voice`.

### Run a red-team voice demo

```bash
PYTHONPATH=. .venv/bin/python scripts/run_voice_conversation.py            # writes MP3s
PYTHONPATH=. .venv/bin/python scripts/run_voice_conversation_realtime.py   # streamed
```

Both TTS the red-team side and STT the client side, ending when a discrepancy is
cracked or patience runs out. The first writes each turn to `audio_conversation/`
(ignored by git); the second streams PCM to the speakers and to Scribe over a
websocket, with no files in between.

### Run a production audit

```bash
PYTHONPATH=. .venv/bin/python scripts/generate_audit_report.py \
    --logs examples/callcentre/calls \
    --business examples/marina_bay.json \
    --script examples/callcentre/script.json \
    --out audit/

open audit/report.html
```

### Run tests

```bash
pip install -e ".[dev]"
PYTHONPATH=. .venv/bin/python -m pytest tests/
```

---

## The report

`scripts/generate_audit_report.py` produces the audit output in two forms:

- **`audit/report.html`** — a self-contained one-page graded report for business
  owners and QA managers. It shows an overall grade, a score per category
  (Quality, Reliability, Compliance, Business, Adherence), and a drill-down with
  the exact quote, call ID, turn, and timestamp for every failure.
- **`audit/report.json`** — the same `AuditReport` object as JSON for the
  technical team. It can feed dashboards, CI jobs, triage queues, or nightly
  regression checks.

The pipeline is deterministic end-to-end unless you pass `--judge` to add an
LLM judge for subjective checks. See `BENEFITS.md` and `REPORT.md` for the
business rationale and the report template.
