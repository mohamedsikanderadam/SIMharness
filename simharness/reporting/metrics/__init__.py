"""Metric families, one module per report category.

Each module exposes a single ``*_metrics`` function taking the normalised call
logs and returning ``(metrics, findings, gaps)``. Keeping the shape identical is
what lets :mod:`simharness.reporting.analyse` stay a dozen lines and lets a new
category be added without touching the renderer.
"""

from simharness.reporting.metrics.adherence import (
    DEFAULT_REQUIRED_BEHAVIOURS,
    RequiredBehaviour,
    adherence_metrics,
)
from simharness.reporting.metrics.business import business_metrics
from simharness.reporting.metrics.compliance import (
    DEFAULT_POLICY_RULES,
    PolicyRule,
    audit_claims,
    compliance_metrics,
)
from simharness.reporting.metrics.quality import quality_metrics
from simharness.reporting.metrics.reliability import reliability_metrics

__all__ = [
    "DEFAULT_POLICY_RULES",
    "DEFAULT_REQUIRED_BEHAVIOURS",
    "PolicyRule",
    "RequiredBehaviour",
    "adherence_metrics",
    "audit_claims",
    "business_metrics",
    "compliance_metrics",
    "quality_metrics",
    "reliability_metrics",
]
