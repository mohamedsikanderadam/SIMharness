# SIMharness — Technical specification

## 01 Problem

**Who:** operators and engineering teams running voice agents for customer
service, booking, reservations, and support.

**Pain:** once a voice agent is on a phone line it can handle hundreds of calls
per day. Existing dashboards show call volume, latency, and a small sample of
human QA scores. They do not answer the question the business actually cares
about: *is the agent quoting the right prices and policies, or is it inventing
them?* A single hallucinated cancellation window or an ungrounded booking
confirmation is a regulatory, customer-service, and brand risk, and it is
invisible to sampling-based QA.

**Why voice:** voice is the channel where the problem is hardest. The customer
cannot scroll back to check what was said, the agent is not constrained by a
pre-rendered web page, and the transcript is the only durable record. SIMharness
treats the transcript as first-class evidence and checks every claim against a
published fact sheet.

## 02 Architecture

```text
                scrape (Context.dev)  or  hand-written JSON
                         |
                         v
BusinessConfig --------------------> FactSheet
    |                                      |
    |         Red-team simulator           |       Production audit
    |    (simulator/redteam.py)            |  (reporting/ + scripts/)
    |                                      |
    v                                      v
Simulated client adapter             CallLog  (ingest/transcribe)
    |                                      |
    v                                      v
Verifier (ground-truth)              analyse_calls()
    |                                      |
    v                                      v
RedTeamEpisodeResult                 AuditReport
                                          |
                                          v
                                  render_html()
                                          |
                             report.html + report.json
```

The red-team side is a deterministic two-agent caller: the **Analyst** reads the
transcript and ground-truth `FactSheet` to maintain a `Casefile` of active
targets, and the **Speaker** phrases the next question. If the client gives a
value that does not match the fact sheet, the caller asks a clarifying follow-up
before marking the target as cracked.

The audit side is a four-stage pipeline: **ingest** loads call logs or
transcribes recordings, **factsheet** builds the ground-truth fact sheet,
**analyse** grades every call, and **render** writes `report.html` and
`report.json`. The HTML page is for business owners and QA; the JSON is for the
technical team and CI.

## 03 Tool rationale

- **ElevenLabs.** Voice is the problem, so the voice stack has to be real. The
  red-team and client sides both use ElevenLabs TTS for synthesis and the
  `scribe_v2` STT model for transcription. This gives us a single API for both
  directions and high-enough transcription accuracy that a scripted agent can
  still be cracked through audio.
- **Context.dev.** A fact sheet is only useful if it matches what the business
  actually publishes. Context.dev turns a real URL (the hotel's website, the
  restaurant's booking page) into structured facts, so the audit is checking the
  agent against the same public information the customer sees.
- **Anthropic (optional).** Subjective or fuzzy checks (tone, script phrasing
  nuance) are routed through an `AnthropicJudge` only when explicitly requested.
  All core checks — facts, script adherence, compliance, timing — are
  deterministic and reproducible.
- **Devin.** The repository was built and iterated through Devin sessions. The
  value is not the choice of editor; it is that the tool chain (git, ssh,
  venv, tests) is the one a contributor would use directly, so the project is
  deployable outside the session.

## 04 Feasibility — scoping to six hours

The initial scope was a working red-team loop with a ground-truth checker and a
voice channel. To keep it inside a single day we made three scoping decisions:

1. **Deterministic callers first.** Speaker and Analyst start as scripted
   classes, not LLM agents. That makes the data flow testable and avoids prompt
   engineering loops. LLM-driven Speaker/Analyst are a v2 extension.
2. **Two fixed client adapters.** `SimulatedClientAgent` (for red-team) and
   `MockBookingClient` (for the Dubai hotel voice demo). More adapters are a
   matter of implementing the `Agent` protocol.
3. **Audit report is a static artefact.** `generate_audit_report.py` writes
   `report.html` and `report.json`, not a live service. A service wrapper is
   trivial to add because `AuditReport` is a Pydantic model and the renderer is
   pure Python.

The result is a repo that can run an end-to-end red-team voice call, an audit
report, and a full test suite from a clean machine in a few commands.

## 05 Extensibility — what v2 looks like

- **Real-time voice.** The current demo is turn-by-turn with saved files. v2
  streams audio chunks directly through `VoiceClient` for a live-sounding call.
- **LLM-driven red team.** Replace the scripted `Speaker` / `Analyst` with models
  that read the `Casefile` and the transcript. Keep the `Casefile` as the
  handoff boundary so the separation of concerns stays clean.
- **More adapters.** Any agent that exposes `respond(AgentRequest)` works, so
  real Twilio/phone adapters, custom in-house agents, and public API demos can
  all be dropped into `simharness/adapters/`.
- **CI gating.** The audit script already exits non-zero on critical findings.
  v2 adds `--fail-under` thresholds, nightly batching, and diff reports between
  model versions.
- **Agent owner dashboard.** `audit/report.json` is the API. A v2 web service
  would store these reports, show trend lines, and let QA click straight to the
  offending call recording.
