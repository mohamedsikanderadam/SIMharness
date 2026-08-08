"""The dependency rules from ARCHITECTURE.md §2, enforced against the import graph.

The first of these is the load-bearing one: if `verifier` only ever imports
`schemas`, then "the primary score is a pure function with no provider coupling"
is a checked property rather than a claim in a document.
"""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent.parent / "simharness"


def _modules(subpackage: str) -> list[Path]:
    root = PACKAGE / subpackage if subpackage else PACKAGE
    return sorted(p for p in root.rglob("*.py"))


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module)
    return found


def _internal(imports: set[str]) -> set[str]:
    return {name for name in imports if name.startswith("simharness")}


def test_schemas_imports_nothing_from_the_package() -> None:
    assert _internal(_imports(PACKAGE / "schemas.py")) == set()


def test_verifier_imports_only_schemas_and_itself() -> None:
    for module in _modules("verifier"):
        for name in _internal(_imports(module)):
            assert name == "simharness.schemas" or name.startswith("simharness.verifier"), (
                f"{module.name} imports {name}; the verifier may only import schemas"
            )


def test_verifier_has_no_provider_or_network_dependency() -> None:
    banned = {"anthropic", "openai", "httpx", "requests", "random", "time"}
    # `time` and `httpx` are legitimate in adapters; in the verifier they would
    # mean a wall clock or a network call inside the reward, which is the whole
    # thing this rule exists to prevent.
    for module in _modules("verifier"):
        offending = _imports(module) & banned
        assert not offending, f"{module.name} imports {offending}"


def test_world_and_scenarios_do_not_depend_on_the_conversation_layers() -> None:
    forbidden = ("simharness.simulator", "simharness.adapters", "simharness.runner")
    for subpackage in ("world", "scenarios"):
        for module in _modules(subpackage):
            for name in _internal(_imports(module)):
                assert not name.startswith(forbidden), f"{module} imports {name}"


def test_provider_sdks_stay_confined() -> None:
    """An SDK import may appear only in a provider client or the LLM adapter."""
    allowed = {"providers", "llm.py"}
    for module in _modules(""):
        if not ({"anthropic", "openai"} & _imports(module)):
            continue
        located = allowed & (set(module.parts) | {module.name})
        assert located, f"{module} imports a provider SDK outside a provider module"


def test_no_audio_dependencies_anywhere() -> None:
    banned = {"whisper", "pyaudio", "sounddevice", "torchaudio", "librosa", "soundfile", "tts"}
    for module in _modules(""):
        offending = {name.split(".")[0] for name in _imports(module)} & banned
        assert not offending, f"{module} imports audio library {offending}"
