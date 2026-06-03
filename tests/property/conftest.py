"""Hypothesis profile registration for the property tests.

Two profiles are registered:

* ``ci`` — used in continuous integration. Fewer examples, no deadline,
  ``derandomize=True`` so the test suite is bit-for-bit reproducible.
* ``dev`` — used locally. More examples, no deadline, randomised.

The ``ci`` profile is the default; override with
``HYPOTHESIS_PROFILE=dev pytest tests/property``.
"""

from __future__ import annotations

import os

from hypothesis import HealthCheck, settings

settings.register_profile(
    "ci",
    max_examples=50,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
settings.register_profile(
    "dev",
    max_examples=200,
    deadline=None,
    derandomize=False,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "ci"))
