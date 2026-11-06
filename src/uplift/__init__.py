"""Experimentation and uplift modeling package."""

from uplift.experimentation.ab_test import ABTestAnalyzer, ABTestResult
from uplift.experimentation.power import PowerCalculator
from uplift.models.base import BaseUpliftLearner
from uplift.models.baselines import (
    OutcomeOnlyLearner,
    RandomPolicy,
    TreatAllPolicy,
    TreatNonePolicy,
)
from uplift.models.doubly_robust import DoublyRobustLearner
from uplift.models.s_learner import SLearner
from uplift.models.t_learner import TLearner

__version__ = "0.1.0"

__all__ = [
    "ABTestAnalyzer",
    "ABTestResult",
    "BaseUpliftLearner",
    "DoublyRobustLearner",
    "OutcomeOnlyLearner",
    "PowerCalculator",
    "RandomPolicy",
    "SLearner",
    "TLearner",
    "TreatAllPolicy",
    "TreatNonePolicy",
    "__version__",
]
