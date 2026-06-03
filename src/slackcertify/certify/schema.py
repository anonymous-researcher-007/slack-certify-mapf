"""JSON Schema for :class:`slackcertify.certify.Certificate`.

This module materialises the schema once at import time so that conformance
tests, downstream consumers, and the CLI ``slackcertify cert validate``
sub-command can use it without having to reach into Pydantic at runtime.

Examples
--------
>>> "mode" in CERTIFICATE_SCHEMA["properties"]
True
"""

from __future__ import annotations

from typing import Any

from slackcertify.certify.certificate import Certificate

__all__ = ["CERTIFICATE_SCHEMA"]


CERTIFICATE_SCHEMA: dict[str, Any] = Certificate.model_json_schema()
