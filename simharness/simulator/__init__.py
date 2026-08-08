"""The counterpart policy, behind one swappable interface."""

from simharness.simulator.base import ScriptedSimulator, Simulator
from simharness.simulator.redteam import RedTeamSimulator

__all__ = ["RedTeamSimulator", "ScriptedSimulator", "Simulator"]
