"""Certificate IO with JSON Schema validation on load."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema

from slackcertify.certify.certificate import Certificate
from slackcertify.certify.schema import CERTIFICATE_SCHEMA

__all__ = ["load_certificate", "save_certificate"]


def save_certificate(cert: Certificate, path: str | Path) -> None:
    """Write ``cert`` to ``path`` as UTF-8 JSON.

    Examples
    --------
    >>> import tempfile, pathlib
    >>> from datetime import datetime, timezone
    >>> from slackcertify.certify.certificate import Certificate
    >>> tmp = pathlib.Path(tempfile.mkdtemp()) / "cert.json"
    >>> save_certificate(Certificate(
    ...     mode="bounded", delta=0, p_d=None, epsilon=None,
    ...     total_wait_inserted=0, per_conflict_waits={},
    ...     solver_used="x", plan_hash="abc",
    ...     created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    ...     proof_sketch="ok",
    ... ), tmp)
    >>> tmp.read_text()[:7]
    '{"mode"'
    """
    Path(path).write_text(cert.model_dump_json(indent=2), encoding="utf-8")


def load_certificate(path: str | Path) -> Certificate:
    """Load a :class:`Certificate` from JSON, validating against ``CERTIFICATE_SCHEMA``.

    Raises
    ------
    jsonschema.ValidationError
        If the on-disk JSON does not conform to the schema produced by
        :mod:`slackcertify.certify.schema`.

    Examples
    --------
    >>> import tempfile, pathlib
    >>> from datetime import datetime, timezone
    >>> from slackcertify.certify.certificate import Certificate
    >>> cert = Certificate(
    ...     mode="bounded", delta=0, p_d=None, epsilon=None,
    ...     total_wait_inserted=0, per_conflict_waits={},
    ...     solver_used="x", plan_hash="abc",
    ...     created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    ...     proof_sketch="ok",
    ... )
    >>> tmp = pathlib.Path(tempfile.mkdtemp()) / "cert.json"
    >>> save_certificate(cert, tmp)
    >>> load_certificate(tmp).mode
    'bounded'
    """
    raw = Path(path).read_text(encoding="utf-8")
    data = json.loads(raw)
    jsonschema.validate(instance=data, schema=CERTIFICATE_SCHEMA)
    return Certificate.model_validate(data)
