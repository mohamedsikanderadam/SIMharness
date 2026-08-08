# SIMharness

**Evaluation tool for voice agents.** Simulated customers call an agent over a
channel that models a noisy phone line, and a verifier scores what actually
happened — not what a judge thought of it.

One harness, two consumers, one difference between them:

| | Eval product | RL environment |
|---|---|---|
| Agent under test | third party, over HTTP | our policy, in process |
| Adapter | `HTTPAgent` | `AnthropicAgent` / `CallableAgent` |
| Primary output | `Scorecard` | `RewardBreakdown` + `Trajectory` |
| Everything else | personas, world, noise, verifier, runner | *identical code path* |

Text only. The voice channel is modelled by injecting ASR-style corruption
between the caller and the agent; there is no audio anywhere and no audio
dependency.

## Evaluate a deployed agent

Describe the business in JSON — published prices, hours, policies. That is the
ground truth every claim gets checked against, and it is the only onboarding
step:

```json
{
  "business_id": "example-clinic",
  "name": "Example Clinic",
  "currency": "AED",
  "catalogue": [{"sku": "CONSULT", "name": "Consultation", "unit_price": 25000}],
  "policies": {"cancellation_window_hours": 24, "max_party_size": 1},
  "opening_hours": {"open": "08:00", "close": "20:00", "closed_weekday": 4}
}
```

Money is **minor units** — `25000` is AED 250.00.

```bash
python scripts/eval_http_agent.py \
  --endpoint https://vendor.example/agent \
  --facts examples/dubai_clinic.json \
  --episodes 10
```

You get, per call: every price/hours/policy claim checked against the fact
sheet, whether the critical number was confirmed back on a noisy line, how the
call ended, p95 latency, and a tally of failure modes.

### What a black-box eval cannot tell you

The vendor's agent writes to the vendor's backend. Its tool calls never reach
our world, so **"did it actually complete the task" is not measured** — only
what the transcript can prove. `blackbox_scenario()` leaves the record checks
empty deliberately rather than letting them pass vacuously, and the scorecard
prints the caveat.

If the vendor will point a *test* instance's webhooks at a URL you control, the
world comes back and so does the whole verifier. That is the difference between
"did it say true things" and "did it do the right thing".

### A new vendor is ten lines

Wire formats differ; transport, retry, timing and failure handling are shared.
Supply a `build` and a `parse` — see `simharness/adapters/http.py`, and
`tests/test_http_agent.py` for how to test one against a fake transport with no
live endpoint.

## The verifier

The primary score is a pure function of (initial world, final world, scenario
spec, transcript). **No LLM judge in the primary path** — it returns a named
component breakdown whose scalar is arithmetically bound to its parts, so it is
usable as an RL reward unmodified.

`verifier/` may import `simharness.schemas` and nothing else from the package,
and a test enforces that against the import graph. That is what makes "pure
function, no provider coupling" a checked property rather than a claim.

Claims are checked in two layers: a typed grammar that binds a phrasing to a
specific ground-truth field (so it can say *incorrect*), and numeric grounding
that requires every number the agent says to trace to a tool result it received,
the fact sheet, or something the caller actually said. Anything neither layer
binds is scored neutral and counted against `claim_coverage` — a metric that
falls when the grammar goes blind, which is the failure mode worth stopping for.

## Scenarios

| Scenario | The interesting part |
|---|---|
| `booking` | A party size that must survive a noisy line into the database |
| `reschedule` | **Compliance is the failure** — the correct agent refuses and offers an alternative |
| `refund_adversary` | A caller insisting on a booking that does not exist |
| `sales_discount` | Holding a discount line under pressure from a haggler |
| `blackbox` | Any third-party agent, transcript evidence only |

## Development

```bash
uv venv --python 3.11 && uv pip install -e ".[dev]"
ruff check simharness/ && mypy simharness/ && pytest
```

The suite is hermetic — scripted counterpart, seeded channel, mocked providers.
No API key and no network. A key is needed only to put a real model in the loop
(`scripts/coverage_report.py`) or to drive a live vendor endpoint.

- [ARCHITECTURE.md](ARCHITECTURE.md) — module boundaries, episode lifecycle, and
  which objects cross which boundary
- [DESIGN_NOTE.md](DESIGN_NOTE.md) — what the verifier does with a claim it
  cannot parse, and why

## Data

`simharness/data/restaurant_db.json` is the MultiWOZ restaurant table
(Budzianowski et al., 2018), MIT-licensed — 110 real Cambridge restaurants. The
business identity is real; the prices are derived, because MultiWOZ records only
a price *band*.

## Licence

Apache-2.0. See [LICENSE](LICENSE).
