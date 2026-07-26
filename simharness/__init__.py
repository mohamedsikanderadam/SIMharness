"""simharness: persona-driven conversation simulator with verifiable rewards.

Two consumers share one codebase:

* the eval product, which points a :class:`~simharness.schemas.AgentRequest` at a
  third-party agent over HTTP and collects a :class:`~simharness.schemas.Scorecard`;
* the RL environment, which points the same request at an in-process policy and
  collects a :class:`~simharness.schemas.Trajectory` plus a
  :class:`~simharness.schemas.RewardBreakdown`.

The only thing that differs between them is the adapter. See ARCHITECTURE.md.
"""

__version__ = "0.1.0"
