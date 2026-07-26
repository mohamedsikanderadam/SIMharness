# simharness — architecture

Status: **Phase 0, design only.** Nothing below `schemas.py` is implemented yet.

## 1. What this is

One harness, two consumers, one difference between them:

| | Eval product | RL environment |
|---|---|---|
| Agent under test | third party, over HTTP, weights not ours | our policy, in process |
| Adapter | `HTTPAgent` | `LocalPolicyAgent` |
| Primary output | `Scorecard` | `RewardBreakdown` + `Trajectory` |
| Everything else | personas, world, noise, verifier, runner | *identical objects, same code path* |

The design rule that follows from this: **the runner must not be able to tell which
consumer it is serving.** It holds an object satisfying the `Agent` protocol and
calls it. If a feature needs `isinstance(adapter, LocalPolicyAgent)` anywhere
outside `adapters/`, the feature is wrong.

This is text only. The voice channel is modelled as ASR-style corruption injected
between the customer and the agent. There is no audio anywhere, and no audio
dependency may be added.

## 2. Module boundaries

```
                            schemas.py
                    (pydantic only; imports nothing from the package)
                                 ▲
        ┌──────────┬─────────────┼─────────────┬────────────┬───────────┐
        │          │             │             │            │           │
     world/    personas/      noise/       simulator/   adapters/   verifier/
   ground     declarative   deterministic   LLM         HTTP /      pure
   truth +    persona       corruption      counterpart in-process  function
   mutable    specs         (cmudict)       policy      policy      scorer
   store                                                                │
        └──────────┴─────────────┴─────────────┴────────────┴───────────┘
                                 ▲
                              runner/           ← the only module that knows all of them
                                 ▲
                    ┌────────────┴────────────┐
                  api/                    __main__.py
              (FastAPI, Gym-shaped)          (CLI)

                            diagnostics/
              optional LLM judge — imported by nothing in the primary path
```

Dependency rules, each of which gets a test in `tests/test_boundaries.py` that
greps the import graph:

1. `schemas` imports nothing from `simharness`.
2. **`verifier` imports `simharness.schemas` and nothing else from the package.**
   This is what makes "the primary score is a pure function" a fact rather than
   an aspiration. It is possible because everything the verifier needs — ground
   truth, the mutation ledger, the transcript, the tool results — is already in
   the snapshot and the trajectory.
3. `world`, `personas`, `noise` import `schemas` only. None of them may import
   `simulator` or `adapters`.
4. No module except `simulator/providers/` may import `anthropic` or `openai`.
   No module except `adapters/http.py` may import `httpx`.
5. `runner` may import anything. `api` and the CLI may import only `runner`,
   `schemas`, and the registries.

### What each module owns

- **`world/`** — a seeded mock business. Frozen ground truth (`BusinessConfig`:
  catalogue, opening hours, `Policies`, availability calendar) plus a mutable
  store (bookings, CRM records, refunds) and an append-only `MutationRecord`
  ledger. Exposed as declarative `ToolSpec`s with JSON Schema parameters, so a
  scenario enables a subset by name and the world never reads the scenario.
  `snapshot()` / `restore()` for exact reset.
- **`verifier/`** — `verify(initial, final, scenario, trajectory, config) ->
  (Scorecard, RewardBreakdown)`. No LLM, no I/O, no clock, no randomness.
- **`personas/`** — declarative `Persona` specs plus the fidelity probe suite.
- **`simulator/`** — the counterpart policy behind one interface, with
  `anthropic`, `openai_compatible`, and `scripted` providers.
- **`noise/`** — a pure function `(text, config, seed) -> (text, NoiseTrace)`.
- **`adapters/`** — the `Agent` protocol and its implementations.
- **`runner/`** — the episode loop and JSONL trajectory writer.
- **`diagnostics/`** — the optional LLM judge. Never imported by `verifier`.
  Its job is to audit the verifier, not the agent; see §8.

## 3. Which objects cross which boundary

This is the part worth arguing about, so it is written down explicitly.

| Boundary | → direction | Object | Never crosses |
|---|---|---|---|
| runner → simulator | in | `SimulatorContext` (persona, its own `SimulatorInternalState`, `tuple[SimulatorTurnView, ...]`, turn index, seed) | `WorldState`, `Scenario`, `SuccessCriteria`, tool results, `RewardBreakdown` |
| simulator → runner | out | `SimulatorOutput` (`utterance`, `internal_state`, `terminate`, `termination`, `usage`) | — |
| runner → noise | in | `str` + `NoiseConfig` + `SpeechProfile` + turn seed | everything else |
| noise → runner | out | corrupted `str` + `NoiseTrace` | — |
| runner → adapter | in | `AgentRequest` (`tuple[AgentTurnView, ...]`, enabled `ToolSpec`s, brief, pending `ToolResult`s) | `Persona`, `HiddenGoal`, `SimulatorInternalState`, `WorldState`, `Scenario`, clean pre-noise text |
| adapter → runner | out | `AgentResponse` (`text`, `tool_calls`, `usage`, `latency_ms`, optional `PolicyTrace`) | — |
| runner → world | in | `ToolCall` | — |
| world → runner | out | `ToolResult`, appended `MutationRecord` | raw mutable handles — the world hands out copies |
| runner → verifier | in | `WorldSnapshot` ×2, `Scenario`, `Trajectory`, `RewardConfig` | the live `WorldState`, any provider client, any clock |
| verifier → runner | out | `Scorecard`, `RewardBreakdown` | — |

Three consequences worth stating separately:

**The agent's only view of the world is through tool results.** The runner
executes tool calls against the world and hands back `ToolResult`s; the adapter
never holds world state. This is why the same code can drive a third-party agent
we do not control.

**The no-leak guarantee is type-level, not prompt-level.** `AgentTurnView` has
two fields and `extra="forbid"`. There is no field on it through which a hidden
goal could travel, and `AgentTurnView.from_turn` is the single sanctioned
projection. The test in `tests/test_no_leak.py` asserts against that projection
rather than against a formatted prompt string, so it keeps holding when prompt
formatting changes.

**The noise model is one line of asymmetry.** `AgentTurnView.from_turn` reads
`turn.delivered_text`; `SimulatorTurnView.from_turn` reads `turn.text`. The agent
lives in the delivered world, the customer lives in the spoken world. A customer
who said "fifteen" does not remember saying "fifty", and will push back when the
agent reads back the wrong number — which is precisely the behaviour we want the
noise sweep to measure.

## 4. Episode lifecycle

```
reset
  1  seeds  = EpisodeSeeds.derive(run_seed, scenario_id, persona_id, episode_index)
  2  world  = world_builders[scenario.world_builder](scenario.world_seed)   # `now` pinned here
  3  initial= WorldSnapshot.of(world, turn_index=0)                          # deep copy + digest
  4  state  = SimulatorInternalState(patience_remaining=persona.patience_turns)
  5  tools  = [tool_specs[t] for t in scenario.enabled_tools]
  6  if scenario.opening_speaker is AGENT: run one agent turn (the greeting)

step  ── repeat until terminal ──────────────────────────────────────────────
  a  ctx  = SimulatorContext(persona, state, simulator_view(turns), i, seeds.for_turn("simulator", i))
  b  out  = simulator(ctx)                                   → SimulatorOutput
  c  if out.terminate: record the turn, stop with out.termination
  d  noised, trace = corrupt(out.utterance, noise_cfg ⊗ persona.speech,
                             seed=seeds.for_turn("noise", i))
  e  append Turn(USER, text=out.utterance, delivered_text=noised,
                 noise=trace, internal_state=out.internal_state)
  f  inner tool loop, at most adapter.max_tool_iterations:
       resp = agent(AgentRequest(history=trajectory.agent_view(), tools=tools,
                                 pending_tool_results=results))
       if resp.tool_calls: results = [world.execute(c) for c in resp.tool_calls]  # ledger grows
       else: break
  g  append Turn(AGENT, text=resp.text, delivered_text=resp.text,
                 tool_calls, tool_results, usage, policy)
  h  patience_remaining -= 1
     if patience_remaining <= 0 → PATIENCE_EXHAUSTED
     if user_turns >= scenario.max_turns → MAX_TURNS

verify
  final = WorldSnapshot.of(world, turn_index=len(turns))
  scorecard, reward = verify(initial, final, scenario, trajectory, reward_config)
  write trajectory + scorecard as one JSONL line each
```

Termination is owned by the simulator where possible (`SATISFIED`, `GAVE_UP`,
`HUNG_UP_ANGRY`); the patience and turn budgets are harness backstops that
indicate the agent failed to converge. `CLEAN_TERMINATIONS` deliberately includes
`GAVE_UP`, because on the reschedule scenario a correct agent leaves the customer
unsatisfied and must not be punished for it.

Tool calls happen *inside* a turn, not as turns of their own, so `max_turns`
means the same thing regardless of how tool-happy the agent is.

## 5. Determinism

Everything the harness controls is derived by hash from one integer:

```
root      = H(run_seed, scenario_id, persona_id, episode_index)
world     = H(root, "world")     simulator = H(root, "simulator")
noise     = H(root, "noise")     per turn  = H(root, channel, turn_index)
```

Per-turn seeds are hashed rather than drawn from a shared generator, so adding
one noise operator or one world coin flip cannot shift the stream every other
component consumes. Beyond that: `WorldState.now` is pinned at reset and never
reads the wall clock; entity ids come from the monotonic `seq` counter, not from
a random source; money is integer minor units; dict ordering is insertion order.

**The honest part.** The simulator is an LLM, and an LLM API is not reproducible —
not across runs at temperature 1, and not bitwise even at temperature 0. Gate 3
("same seed reproduces the identical transcript and reward breakdown") is
therefore satisfied in two ways, and I will report which one each claim rests on:

- `provider="scripted"` — fully deterministic, no network, no key. The
  determinism and no-leak tests use this, so CI is hermetic.
- `cassette_mode="record" | "replay"` — live runs record every provider
  response keyed by `H(model, temperature, system, messages, turn_seed)`.
  Replay reproduces byte-identical transcripts. **A cache miss in replay mode is
  a hard error, never a silent live call** — a replay that quietly re-samples is
  worse than no replay at all.

What is *not* claimed: that two independent live runs at the same seed match.
They will not, and any design that promised otherwise would be lying.

## 6. Reward and cost

`verify()` returns named components; `RewardBreakdown` recomputes the scalar from
them and raises if the two disagree, so a refactor cannot leave a reported 0.8
sitting on top of parts that sum to 0.3.

| Component | Raw | Default weight |
|---|---|---|
| `task_success` | all `required_records` matched → 1 | +1.0 |
| `field_accuracy` | fraction of individual `FieldMatch`es satisfied | +0.3 |
| `forbidden_mutation` | 1 if the ledger contains any forbidden op | −1.0 |
| `claim_accuracy` | correct / (correct + incorrect + ungrounded) | +0.5 |
| `termination` | 1 if in `CLEAN_TERMINATIONS` | +0.2 |
| `cost` | normalised agent turns + tokens | −0.1, **off by default** |

`field_accuracy` exists for RL rather than for eval. A pure 0/1 task reward makes
every rollout in a GRPO group identical when the task is hard, the group's
advantage std collapses to zero, and the update is noise. Partial credit on
individual field matches keeps a gradient. The sweep should report per-group
reward std alongside mean; a std near zero means the signal is dead regardless of
what the mean says.

**Cost accounting splits agent from harness.** `CostSummary` tracks
`agent_tokens` and `simulator_tokens` separately. The penalty shapes agent tokens
and turn count only — charging a policy for its counterpart's tokens teaches it
to make the *simulator* terse, which is not a behaviour anyone wants to buy. The
simulator's spend is still reported, because it is what the $15 sweep budget
actually gets spent on. USD comes from a configurable price table
(`price_table_id` is recorded on every run) rather than from constants baked into
the scorer.

Cost shaping is off by default and toggled by `RewardConfig.cost_shaping_enabled`,
so the eval product's numbers are never silently a function of token prices.

## 7. Extension points

- **A third adapter** requires one class implementing the `Agent` protocol and
  one registry entry. Nothing else changes — that is the load-bearing claim of
  §2 and it gets a test.
- **A scenario** is one file in `scenarios/` registering a `Scenario` and, if it
  needs a new starting world, a world builder. See the README's "how to add a
  scenario" (Phase 4).
- **Noise locales.** `NoiseConfig.locale` selects an operator set. Code-switching
  and Arabic transliteration plug in here as additional `NoiseOpKind`s plus a
  locale-specific candidate generator. Explicitly not implemented now.
- **Simulator providers** implement one `generate(ctx) -> SimulatorOutput`
  method. The simulator being later *trainable* is why `SimulatorContext` and
  `SimulatorOutput` are plain serialisable models: a trained counterpart is a
  provider swap, not a rewrite.

## 8. The LLM judge, and where it is allowed to live

`diagnostics/` may contain a judge. It is never imported by `verifier/`, never
summed into the scalar, and never appears in `RewardBreakdown`.

Its actual job is to audit the *verifier*, not the agent: sample transcripts,
ask a model to list every factual claim, and compare that list against what the
claim grammar bound. That measures the grammar's recall and tells us where to
extend it. See DESIGN_NOTE.md — this is the mitigation for the one design
decision I think is genuinely contestable.

## 9. Anti-goals, and what enforces each

| Anti-goal | Enforcement |
|---|---|
| No audio | no audio dependency in `pyproject.toml`; noise operates on text |
| No LLM judge in the primary reward | verifier may import only `schemas` (tested) |
| No framework lock-in | dependencies are pydantic, httpx, fastapi, provider SDKs, numpy, matplotlib, cmudict — no agent framework |
| No provider coupling in the verifier | same import test as above |
| Correctness before scale | the episode loop is synchronous; parallelism, if ever needed, goes at the sweep level where episodes are already independent |

## 10. Open, pending review

1. The unparsed-claim policy — see DESIGN_NOTE.md. This is the one I want a
   decision on before Phase 1, because it determines what the verifier's contract
   is and the verifier defines the reward.
2. Whether the eval product needs per-turn latency SLAs as a scored component.
   Currently `latency_ms` is recorded but not scored, which is right for RL and
   possibly wrong for eval.
