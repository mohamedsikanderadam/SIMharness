# simharness

Persona-driven conversation simulator with verifiable rewards.

Simulated customers talk to an agent over a noisy text channel that models a
voice line. The same customers serve two consumers: an eval product that scores a
third-party agent over HTTP, and an RL environment that trains our own policy
multi-turn. The only difference between them is the adapter.

**Status: Phase 0 — design only.** The implementation starts at Phase 1.

- [ARCHITECTURE.md](ARCHITECTURE.md) — module boundaries, episode lifecycle, and
  which objects cross which boundary
- [DESIGN_NOTE.md](DESIGN_NOTE.md) — the one decision that needs a call before
  the verifier gets built
- [simharness/schemas.py](simharness/schemas.py) — every object that crosses a
  boundary

The module-boundary diagram and the "how to add a scenario" guide land here in
Phase 4, once there is something to add a scenario to.

## Development

```bash
uv venv --python 3.11 && uv pip install -e ".[dev]"
ruff check simharness/ && mypy simharness/ && pytest
```
