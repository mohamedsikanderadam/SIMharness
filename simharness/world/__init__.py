"""The mock business backend: ground truth, mutable store, declarative tools."""

from simharness.world.backend import ToolError, World
from simharness.world.tools import TOOL_SPECS, specs_for

__all__ = [
    "TOOL_SPECS",
    "ToolError",
    "World",
    "specs_for",
]
