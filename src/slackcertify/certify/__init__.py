"""Slack-certifier: bounded and probabilistic certification.

Re-exports the certificate value object, its JSON Schema, and the pure
bounded / probabilistic certification helpers.
"""

from __future__ import annotations

from slackcertify.certify.bounded import is_delta_certified, unsafe_pairs_bounded
from slackcertify.certify.certificate import Certificate
from slackcertify.certify.probabilistic import (
    is_epsilon_certified,
    per_conflict_risk,
    plan_risk_upper_bound,
    unsafe_pairs_probabilistic,
)
from slackcertify.certify.schema import CERTIFICATE_SCHEMA

__all__ = [
    "CERTIFICATE_SCHEMA",
    "Certificate",
    "is_delta_certified",
    "is_epsilon_certified",
    "per_conflict_risk",
    "plan_risk_upper_bound",
    "unsafe_pairs_bounded",
    "unsafe_pairs_probabilistic",
]
