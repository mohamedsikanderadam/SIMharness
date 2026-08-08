# Red-Team Dialogue: First Principles

This document records the agreed design for the two-agent red-team caller that attempts to expose false or ungrounded claims from a simulated client agent.

## 1. Goal

Build a red-team simulator that can hold a multi-turn conversation with an agent under test, identify contradictions or hallucinations in that agent's statements, and do so in a way that can be scored by the existing SIMharness `verifier`.

## 2. Roles

- **Client agent** (`Agent` under test, `simharness/adapters/`). In training and testing this is our own simulated agent. It is loaded with a ground-truth fact sheet and, optionally, an overlay of false beliefs or hallucinations.
- **Speaker** (the first agent, inside `simulator/`). Only the Speaker sees the raw transcript and produces the next conversational turn. It does not see hidden client state or the full ground truth directly.
- **Analyst** (the second agent, also inside `simulator/`). The Analyst reads the transcript, the known ground-truth fact sheet, and `verifier` output. It maintains a strategic `Casefile` that tells the Speaker what to probe next.

## 3. What "crack" means

A "crack" is not a reward for *making* the client agent lie. It is a reward for *exposing* a claim that the `verifier` scores as `INCORRECT` or `UNGROUNDED` relative to the ground-truth world. The `verifier` is the sole arbiter of success.

## 4. The Casefile

The Analyst does not emit a raw list of questions. It writes and updates a small structured `Casefile`:

```json
{
  "confirmed_facts": [],
  "discrepancies": [],
  "active_targets": [
    {"field": "deposit", "true_value": "1500", "suspicion_level": "high"}
  ],
  "next_move": "Ask the agent to repeat and explain the deposit amount in detail."
}
```

The Speaker reads the `Casefile` and composes the actual next utterance. This keeps the Analyst strategic and the Speaker natural.

## 5. Knowledge separation

- The red team may know the **public ground-truth fact sheet** (the real business policy/prices/hours).
- The red team may use the `verifier` to score the resulting transcript.
- The red team **must not** read the client agent's internal state, tool calls, or the exact false beliefs that were injected. Overfitting to hidden state would make the red team useless against a real agent.

## 6. Data flow

1. Ingest ground truth: scraped or hand-written data is written as a `BusinessConfig` JSON (e.g. `examples/*.json`).
2. Optionally prepare a `client_beliefs` overlay for the simulated client that contains the false claims to be exposed.
3. Build the simulated client as an `Agent` adapter with the combined knowledge base.
4. Run episodes: Speaker and Analyst talk to the client; `world` records any tool side effects; `verifier` scores the transcript.
5. Reward the red team for finding verified discrepancies in as few turns as possible, while keeping `claim_coverage` high.

## 7. Code locations

- Simulated client: new `Agent` adapter under `simharness/adapters/`.
- Red-team caller: new `simulator` provider under `simharness/simulator/` that internally runs Speaker and Analyst.
- Ground truth and client beliefs: live in `simharness/world/`, `BusinessConfig` files, or a new lightweight knowledge overlay.
- Scoring: reuse `simharness/verifier/`. Add a new `RewardConfig` or scenario only if the existing reward shape does not capture red-team success.
