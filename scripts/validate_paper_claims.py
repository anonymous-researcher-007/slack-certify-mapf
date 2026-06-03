#!/usr/bin/env python3
"""Pre-submission claim validator for slack-certify-mapf.

Reads a YAML claims registry (default
``docs/PAPER_NUMERICAL_CLAIMS.yaml``), evaluates each claim
against the real Phase 7 CSVs under ``--results-dir``, and
produces a Markdown report grouped by verdict
(Confirmed / Stronger / Weaker / Refuted / Skipped). Also emits
a companion LaTeX-tables file with per-RQ booktabs fragments
that are ``\\input{}``-includable from the paper.

The validator never modifies the paper LaTeX. It only proposes
replacement sentences in the report; the human decides whether
to apply them.

Exit code is 0 iff zero Refuted and zero Weaker claims fire —
making this directly usable as a CI gate.

Usage::

    python scripts/validate_paper_claims.py \\
        --results-dir results/raw \\
        --claims docs/PAPER_NUMERICAL_CLAIMS.yaml \\
        --out reports/claim_validation.md \\
        --tables-out reports/claim_validation_tables.tex \\
        --section all
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import yaml

__all__ = ["main"]


# ----------------------------------------------------------- schema constants

_AGGREGATIONS = {
    "mean",
    "median",
    "p95",
    "min",
    "max",
    "fraction_true",
    "fraction_ge",
    "count",
}
_OPERATORS = {">=", "<=", "==", "!=", "in_range"}
_FILTER_OPS = {">=", "<=", "==", "!=", ">", "<"}

Verdict = Literal["Confirmed", "Stronger", "Weaker", "Refuted", "Skipped"]
_VERDICT_ORDER: list[Verdict] = [
    "Refuted",
    "Weaker",
    "Stronger",
    "Confirmed",
    "Skipped",
]


# --------------------------------------------------------------- error types


class ClaimSchemaError(ValueError):
    """Raised when a claims-YAML entry has a structural error."""


# ----------------------------------------------------------- data structures


@dataclass(frozen=True, slots=True)
class Threshold:
    """Numeric comparison: ``observed <operator> value``."""

    operator: str
    value: float | tuple[float, float]


@dataclass(frozen=True, slots=True)
class PairedSpec:
    """Paired-method comparison: ``method_a - method_b`` per (map, n, seed)."""

    filter_common: dict[str, Any]
    method_a: dict[str, Any]
    method_b: dict[str, Any]
    metric: str | None
    metric_from_diagnostics: str | None
    operation: str  # "subtract"
    aggregation: str


@dataclass(frozen=True, slots=True)
class Claim:
    """One row of the claims registry."""

    id: str
    section: str
    paper_sentence: str
    source_csv: str
    # For unpaired claims:
    filter: dict[str, Any] | None
    metric: str | None
    metric_from_diagnostics: str | None
    aggregation: str | None
    fraction_ge_sub_threshold: float | None
    # Threshold and tolerance (apply to both unpaired and paired):
    threshold: Threshold
    tolerance: float
    # For paired-comparison claims:
    paired: PairedSpec | None


@dataclass(slots=True)
class EvaluationResult:
    """Verdict + bookkeeping rendered into the report."""

    claim: Claim
    verdict: Verdict
    observed: float | None
    reason: str = ""
    # Provenance: row count after filtering (or after the paired-merge
    # for paired claims). Surfaced in the report so the reviewer can
    # see how much data backed the verdict.
    n_rows: int = 0
    extras: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------- YAML loading


def _require(node: dict[str, Any], key: str, where: str) -> Any:  # noqa: ANN401 - YAML values are genuinely polymorphic
    """Return ``node[key]`` or raise :class:`ClaimSchemaError`."""
    if key not in node:
        raise ClaimSchemaError(f"{where}: required field {key!r} missing")
    return node[key]


def _parse_threshold(raw: dict[str, Any], where: str) -> Threshold:
    """Parse ``{operator, value}`` into a :class:`Threshold`."""
    op = _require(raw, "operator", where)
    val = _require(raw, "value", where)
    if op not in _OPERATORS:
        raise ClaimSchemaError(
            f"{where}: threshold.operator {op!r} not in {sorted(_OPERATORS)}"
        )
    if op == "in_range":
        if (
            not isinstance(val, list)
            or len(val) != 2
            or not all(isinstance(x, (int, float)) for x in val)
        ):
            raise ClaimSchemaError(
                f"{where}: operator='in_range' requires value=[lo, hi]"
            )
        return Threshold(operator=op, value=(float(val[0]), float(val[1])))
    if not isinstance(val, (int, float)):
        raise ClaimSchemaError(f"{where}: threshold.value must be numeric")
    return Threshold(operator=op, value=float(val))


def _parse_paired(raw: dict[str, Any], where: str) -> PairedSpec:
    filter_common = _require(raw, "filter_common", f"{where}.paired_comparison")
    method_a = _require(raw, "method_a", f"{where}.paired_comparison")
    method_b = _require(raw, "method_b", f"{where}.paired_comparison")
    operation = _require(raw, "operation", f"{where}.paired_comparison")
    aggregation = _require(raw, "aggregation", f"{where}.paired_comparison")
    if operation != "subtract":
        raise ClaimSchemaError(
            f"{where}: paired_comparison.operation {operation!r} unsupported "
            f"(supported: subtract)"
        )
    if aggregation not in _AGGREGATIONS:
        raise ClaimSchemaError(
            f"{where}: paired_comparison.aggregation {aggregation!r} "
            f"not in {sorted(_AGGREGATIONS)}"
        )
    metric = raw.get("metric")
    metric_diag = raw.get("metric_from_diagnostics")
    if metric is None and metric_diag is None:
        raise ClaimSchemaError(
            f"{where}.paired_comparison: requires `metric` or "
            f"`metric_from_diagnostics`"
        )
    if metric is not None and metric_diag is not None:
        raise ClaimSchemaError(
            f"{where}.paired_comparison: `metric` and "
            f"`metric_from_diagnostics` are mutually exclusive"
        )
    return PairedSpec(
        filter_common=dict(filter_common),
        method_a=dict(method_a),
        method_b=dict(method_b),
        metric=metric,
        metric_from_diagnostics=metric_diag,
        operation=operation,
        aggregation=aggregation,
    )


def _parse_claim(raw: dict[str, Any], idx: int) -> Claim:
    where = f"claims[{idx}]"
    claim_id = _require(raw, "id", where)
    section = str(_require(raw, "section", where))
    paper_sentence = _require(raw, "paper_sentence", where).strip()
    evaluation = _require(raw, "evaluation", where)
    source_csv = _require(evaluation, "source_csv", f"{where}.evaluation")
    threshold = _parse_threshold(
        _require(evaluation, "threshold", f"{where}.evaluation"),
        f"{where}.evaluation.threshold",
    )
    tolerance_raw = evaluation.get("tolerance", 0.0)
    if not isinstance(tolerance_raw, (int, float)):
        raise ClaimSchemaError(f"{where}.evaluation.tolerance must be numeric")
    tolerance = float(tolerance_raw)

    paired = None
    if "paired_comparison" in evaluation:
        paired = _parse_paired(
            evaluation["paired_comparison"], f"{where}.evaluation"
        )
        return Claim(
            id=claim_id,
            section=section,
            paper_sentence=paper_sentence,
            source_csv=source_csv,
            filter=None,
            metric=None,
            metric_from_diagnostics=None,
            aggregation=None,
            fraction_ge_sub_threshold=None,
            threshold=threshold,
            tolerance=tolerance,
            paired=paired,
        )

    # Unpaired claim path.
    metric = evaluation.get("metric")
    metric_diag = evaluation.get("metric_from_diagnostics")
    if metric is None and metric_diag is None:
        raise ClaimSchemaError(
            f"{where}.evaluation: requires `metric`, "
            f"`metric_from_diagnostics`, or `paired_comparison`"
        )
    if metric is not None and metric_diag is not None:
        raise ClaimSchemaError(
            f"{where}.evaluation: `metric` and `metric_from_diagnostics` "
            f"are mutually exclusive"
        )
    aggregation = _require(evaluation, "aggregation", f"{where}.evaluation")
    if aggregation not in _AGGREGATIONS:
        raise ClaimSchemaError(
            f"{where}.evaluation.aggregation {aggregation!r} not in "
            f"{sorted(_AGGREGATIONS)}"
        )
    sub_threshold = None
    if aggregation == "fraction_ge":
        sub_threshold_raw = evaluation.get("fraction_ge_sub_threshold")
        if sub_threshold_raw is None:
            raise ClaimSchemaError(
                f"{where}.evaluation: aggregation='fraction_ge' requires "
                f"`fraction_ge_sub_threshold`"
            )
        if not isinstance(sub_threshold_raw, (int, float)):
            raise ClaimSchemaError(
                f"{where}.evaluation.fraction_ge_sub_threshold must be numeric"
            )
        sub_threshold = float(sub_threshold_raw)
    return Claim(
        id=claim_id,
        section=section,
        paper_sentence=paper_sentence,
        source_csv=source_csv,
        filter=dict(evaluation.get("filter") or {}),
        metric=metric,
        metric_from_diagnostics=metric_diag,
        aggregation=aggregation,
        fraction_ge_sub_threshold=sub_threshold,
        threshold=threshold,
        tolerance=tolerance,
        paired=None,
    )


def load_claims(path: Path) -> list[Claim]:
    """Load and schema-validate the claims YAML at ``path``."""
    text = path.read_text(encoding="utf-8")
    doc = yaml.safe_load(text)
    if not isinstance(doc, dict) or "claims" not in doc:
        raise ClaimSchemaError(f"{path}: top-level mapping must contain `claims`")
    raw_claims = doc["claims"]
    if not isinstance(raw_claims, list):
        raise ClaimSchemaError(f"{path}: `claims` must be a list")
    out: list[Claim] = []
    for i, c in enumerate(raw_claims):
        if not isinstance(c, dict):
            raise ClaimSchemaError(f"{path}: claims[{i}] must be a mapping")
        out.append(_parse_claim(c, i))
    return out


# ------------------------------------------------------------- DataFrame ops


def _apply_filter(df: pd.DataFrame, filt: dict[str, Any]) -> pd.DataFrame:
    """Apply a single claim's filter spec to ``df``.

    Supports equality (string or numeric) and the operator-form
    ``{operator: ..., value: ...}`` for numeric comparisons.
    """
    if df.empty:
        return df
    result = df
    for col, spec in filt.items():
        if col not in result.columns:
            # Column missing → empty filter result is the correct
            # signal (validator marks claim as no_rows_matched).
            return result.iloc[0:0]
        if isinstance(spec, dict) and "operator" in spec and "value" in spec:
            op = spec["operator"]
            value = spec["value"]
            if op not in _FILTER_OPS:
                raise ClaimSchemaError(
                    f"filter operator {op!r} not in {sorted(_FILTER_OPS)}"
                )
            series = pd.to_numeric(result[col], errors="coerce")
            if op == ">=":
                mask = series >= value
            elif op == "<=":
                mask = series <= value
            elif op == ">":
                mask = series > value
            elif op == "<":
                mask = series < value
            elif op == "==":
                mask = series == value
            else:  # "!="
                mask = series != value
            result = result[mask.fillna(False)]
        else:
            # Equality filter — handle numeric columns gracefully by
            # also trying numeric comparison after string equality
            # fails for every row.
            mask = result[col] == spec
            if not mask.any() and isinstance(spec, (int, float)):
                series = pd.to_numeric(result[col], errors="coerce")
                mask = series == spec
            result = result[mask]
    return result


def _parse_diagnostics(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    try:
        if isinstance(value, float) and math.isnan(value):
            return {}
    except (TypeError, ValueError):
        pass
    if not isinstance(value, (str, bytes, bytearray)):
        return {}
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _extract_metric(df: pd.DataFrame, claim: Claim) -> pd.Series:
    """Pull the metric series out of ``df`` per the claim's recipe."""
    if claim.metric is not None:
        if claim.metric not in df.columns:
            return pd.Series([], dtype=float)
        return pd.to_numeric(df[claim.metric], errors="coerce").dropna()
    if claim.metric_from_diagnostics is not None:
        if "diagnostics" not in df.columns:
            return pd.Series([], dtype=float)
        key = claim.metric_from_diagnostics
        values: list[Any] = []
        for raw in df["diagnostics"]:
            diag = _parse_diagnostics(raw)
            if key in diag and diag[key] is not None:
                values.append(diag[key])
        return pd.Series(values)
    return pd.Series([], dtype=float)


def _aggregate(
    series: pd.Series,
    aggregation: str,
    fraction_ge_sub: float | None,
) -> float | None:
    """Aggregate ``series`` per the claim's recipe."""
    if series.empty:
        return None
    if aggregation == "count":
        return float(len(series))
    if aggregation == "fraction_true":
        # Coerce to bool (Python ``bool(x)`` handles JSON booleans).
        bools = [bool(x) for x in series if x is not None]
        if not bools:
            return None
        return float(sum(1 for b in bools if b)) / float(len(bools))
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return None
    if aggregation == "mean":
        return float(numeric.mean())
    if aggregation == "median":
        return float(numeric.median())
    if aggregation == "p95":
        return float(numeric.quantile(0.95))
    if aggregation == "min":
        return float(numeric.min())
    if aggregation == "max":
        return float(numeric.max())
    if aggregation == "fraction_ge":
        assert fraction_ge_sub is not None
        return float((numeric >= fraction_ge_sub).mean())
    raise ClaimSchemaError(f"unsupported aggregation: {aggregation!r}")


# --------------------------------------------------------------- evaluation


def _classify(
    observed: float, threshold: Threshold, tolerance: float
) -> Verdict:
    """Return the verdict given observed vs threshold ± tolerance."""
    op = threshold.operator
    if op == "in_range":
        lo, hi = threshold.value  # type: ignore[misc]
        if (lo - tolerance) <= observed <= (hi + tolerance):
            return "Confirmed"
        return "Refuted"
    value = float(threshold.value)  # type: ignore[arg-type]
    diff = observed - value
    if op == ">=":
        if observed >= value - tolerance and observed <= value + tolerance:
            return "Confirmed"
        if observed > value + tolerance:
            return "Stronger"
        # observed < value - tolerance:
        if observed >= value * 0.0 - 1e18:  # i.e. always — direction check
            # Direction: paper says >= value. Observed below by more
            # than tolerance. Distinguish "in-direction but short"
            # (Weaker) from "contradicts direction" (Refuted).
            # We treat values >= 0 differently from the threshold:
            # if the observation has the *opposite sign* of the
            # threshold, that's Refuted. Otherwise Weaker.
            #
            # Concretely: ">= 0.95" with observed 0.40 — same sign,
            # in-direction-but-short → Weaker. ">= 0.20" (paired diff)
            # with observed -0.15 — opposite sign → Refuted.
            if value > 0 and observed < 0:
                return "Refuted"
            if value < 0 and observed > 0:
                return "Refuted"
            return "Weaker"
        return "Refuted"
    if op == "<=":
        if observed >= value - tolerance and observed <= value + tolerance:
            return "Confirmed"
        if observed < value - tolerance:
            return "Stronger"
        if value > 0 and observed < 0:
            return "Refuted"
        if value < 0 and observed > 0:
            return "Refuted"
        return "Weaker"
    if op == "==":
        if abs(diff) <= tolerance:
            return "Confirmed"
        return "Refuted"
    if op == "!=":
        if abs(diff) > tolerance:
            return "Confirmed"
        return "Refuted"
    raise ClaimSchemaError(f"unsupported operator: {op!r}")


def _load_csv(results_dir: Path, source_csv: str) -> pd.DataFrame | None:
    """Return the CSV as a DataFrame, or ``None`` if missing/empty."""
    path = (results_dir / source_csv).resolve()
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None
    if df.empty:
        return None
    return df


def _evaluate_unpaired(claim: Claim, df: pd.DataFrame) -> EvaluationResult:
    """Evaluate a non-paired claim."""
    filtered = _apply_filter(df, claim.filter or {})
    if filtered.empty:
        return EvaluationResult(
            claim=claim,
            verdict="Skipped",
            observed=None,
            reason="no_rows_matched",
        )
    series = _extract_metric(filtered, claim)
    assert claim.aggregation is not None
    observed = _aggregate(
        series, claim.aggregation, claim.fraction_ge_sub_threshold
    )
    if observed is None:
        return EvaluationResult(
            claim=claim,
            verdict="Skipped",
            observed=None,
            reason="empty_metric_series",
            n_rows=int(len(filtered)),
        )
    verdict = _classify(observed, claim.threshold, claim.tolerance)
    return EvaluationResult(
        claim=claim,
        verdict=verdict,
        observed=observed,
        n_rows=int(len(filtered)),
    )


def _evaluate_paired(claim: Claim, df: pd.DataFrame) -> EvaluationResult:
    """Evaluate a paired-method comparison."""
    paired = claim.paired
    assert paired is not None
    common = _apply_filter(df, paired.filter_common)
    if common.empty:
        return EvaluationResult(
            claim=claim,
            verdict="Skipped",
            observed=None,
            reason="no_rows_matched",
        )
    df_a = _apply_filter(common, paired.method_a)
    df_b = _apply_filter(common, paired.method_b)
    if df_a.empty or df_b.empty:
        return EvaluationResult(
            claim=claim,
            verdict="Skipped",
            observed=None,
            reason="no_rows_matched",
        )

    # The pairing key: every column shared by both sides that the
    # filter_common pinned + the canonical per-cell key
    # (map_name, n_agents, scen_seed). This lets a difference like
    # method_a - method_b be computed per-cell before aggregation.
    key_cols = [
        c for c in ("map_name", "n_agents", "scen_seed", "delta")
        if c in df_a.columns and c in df_b.columns
    ]
    # Build a per-cell metric for each side.
    fake_claim_a = Claim(
        id=claim.id, section=claim.section, paper_sentence="",
        source_csv=claim.source_csv, filter=None,
        metric=paired.metric, metric_from_diagnostics=paired.metric_from_diagnostics,
        aggregation=None, fraction_ge_sub_threshold=None,
        threshold=claim.threshold, tolerance=claim.tolerance, paired=None,
    )
    metric_a = _extract_metric(df_a, fake_claim_a)
    metric_b = _extract_metric(df_b, fake_claim_a)
    if metric_a.empty or metric_b.empty:
        return EvaluationResult(
            claim=claim,
            verdict="Skipped",
            observed=None,
            reason="empty_metric_series",
            n_rows=int(len(df_a) + len(df_b)),
        )
    a_indexed = df_a.assign(_metric=metric_a.to_numpy()).set_index(key_cols)
    b_indexed = df_b.assign(_metric=metric_b.to_numpy()).set_index(key_cols)
    joined = a_indexed[["_metric"]].join(
        b_indexed[["_metric"]], how="inner", lsuffix="_a", rsuffix="_b"
    )
    if joined.empty:
        return EvaluationResult(
            claim=claim,
            verdict="Skipped",
            observed=None,
            reason="no_paired_rows",
        )
    diffs = joined["_metric_a"] - joined["_metric_b"]
    observed = _aggregate(diffs, paired.aggregation, None)
    if observed is None:
        return EvaluationResult(
            claim=claim,
            verdict="Skipped",
            observed=None,
            reason="empty_metric_series",
            n_rows=int(len(joined)),
        )
    verdict = _classify(observed, claim.threshold, claim.tolerance)
    return EvaluationResult(
        claim=claim,
        verdict=verdict,
        observed=observed,
        n_rows=int(len(joined)),
    )


def evaluate(claim: Claim, results_dir: Path) -> EvaluationResult:
    """Evaluate one claim against a CSV in ``results_dir``."""
    df = _load_csv(results_dir, claim.source_csv)
    if df is None:
        return EvaluationResult(
            claim=claim,
            verdict="Skipped",
            observed=None,
            reason=f"missing_csv:{claim.source_csv}",
        )
    if claim.paired is not None:
        return _evaluate_paired(claim, df)
    return _evaluate_unpaired(claim, df)


# -------------------------------------------------------------- report (md)


def _format_observed(observed: float | None) -> str:
    if observed is None:
        return "—"
    if math.isnan(observed):
        return "NaN"
    return f"{observed:.4f}"


def _format_threshold(t: Threshold) -> str:
    if t.operator == "in_range":
        lo, hi = t.value  # type: ignore[misc]
        return f"in_range [{lo:.4f}, {hi:.4f}]"
    return f"{t.operator} {float(t.value):.4f}"  # type: ignore[arg-type]


def _suggest_replacement(result: EvaluationResult) -> str:
    """Render a one-sentence suggested replacement for the paper prose."""
    claim = result.claim
    obs = result.observed
    if obs is None:
        return "(no observed value — evaluation skipped)"
    obs_str = f"{obs:.4f}"
    if result.verdict == "Refuted":
        return (
            f"DATA-DRIVEN REPLACEMENT: the observed value of "
            f"{obs_str} contradicts the predicted direction "
            f"({_format_threshold(claim.threshold)}); rewrite the "
            f"sentence to either reverse the direction or remove the "
            f"claim before resubmission."
        )
    if result.verdict == "Weaker":
        return (
            f"SOFTENING SUGGESTION: \"The observed value is {obs_str}, "
            f"which is in the predicted direction but below the "
            f"previously stated bound ({_format_threshold(claim.threshold)}).\""
        )
    if result.verdict == "Stronger":
        return (
            f"TIGHTENING SUGGESTION: \"The observed value is {obs_str}, "
            f"exceeding the previously stated bound "
            f"({_format_threshold(claim.threshold)}); the prose can "
            f"safely tighten the magnitude to match.\""
        )
    return ""


def _section_sort_key(section: str) -> tuple[int, ...]:
    parts: list[int] = []
    for piece in section.replace(",", ".").split("."):
        piece = piece.strip()
        if piece.isdigit():
            parts.append(int(piece))
        else:
            parts.append(0)
    return tuple(parts)


def _git_commit() -> str:
    try:
        result = subprocess.run(  # noqa: S603 - argv is a literal
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5.0,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "(unknown — not a git checkout or git missing)"


def render_report(
    results: list[EvaluationResult],
    claims_path: Path,
    results_dir: Path,
) -> str:
    """Render the per-verdict Markdown report."""
    lines: list[str] = []
    timestamp = _dt.datetime.now(tz=_dt.timezone.utc).isoformat(timespec="seconds")
    lines.append("# Paper Claim Validation Report")
    lines.append(f"Generated: {timestamp}")
    lines.append(f"Commit: {_git_commit()}")
    lines.append(f"Claims file: {claims_path}")
    lines.append(f"Results directory: {results_dir}/")
    lines.append("")

    counts: dict[Verdict, int] = {v: 0 for v in _VERDICT_ORDER}
    for r in results:
        counts[r.verdict] = counts.get(r.verdict, 0) + 1

    lines.append("## Summary")
    lines.append("| Verdict | Count |")
    lines.append("|---|---|")
    for v in ("Confirmed", "Stronger", "Weaker", "Refuted", "Skipped"):
        lines.append(f"| {v} | {counts.get(v, 0)} |")
    lines.append("")
    total = len(results)
    blocking = counts.get("Refuted", 0) + counts.get("Weaker", 0)
    lines.append(f"Total claims evaluated: {total}")
    lines.append(f"Submission-blocking verdicts (Refuted + Weaker): {blocking}")
    lines.append("")

    # Per-verdict sections.
    by_verdict: dict[Verdict, list[EvaluationResult]] = {
        v: [] for v in _VERDICT_ORDER
    }
    for r in results:
        by_verdict[r.verdict].append(r)
    for v in by_verdict:
        by_verdict[v].sort(key=lambda r: _section_sort_key(r.claim.section))

    # Refuted, Weaker, Stronger: full detail.
    for v in ("Refuted", "Weaker", "Stronger"):
        lines.append(f"## {v}")
        if not by_verdict[v]:  # type: ignore[index]
            lines.append("_(none)_")
            lines.append("")
            continue
        for r in by_verdict[v]:  # type: ignore[index]
            lines.append(f"### {r.claim.id} (§{r.claim.section})")
            lines.append(
                f"**Paper sentence.** {r.claim.paper_sentence.strip()}"
            )
            agg = (
                r.claim.aggregation
                if r.claim.paired is None
                else r.claim.paired.aggregation
            )
            lines.append(
                f"**Observed.** {_format_observed(r.observed)} "
                f"(aggregation: {agg}, n_rows: {r.n_rows})"
            )
            lines.append(
                f"**Threshold.** {_format_threshold(r.claim.threshold)} "
                f"(tolerance: {r.claim.tolerance})"
            )
            lines.append(f"**Verdict.** {v}.")
            lines.append(
                f"**Suggested replacement sentence.** {_suggest_replacement(r)}"
            )
            lines.append("")

    lines.append("## Confirmed")
    if not by_verdict["Confirmed"]:
        lines.append("_(none)_")
    else:
        for r in by_verdict["Confirmed"]:
            lines.append(
                f"- ✓ `{r.claim.id}` (§{r.claim.section}) — "
                f"observed {_format_observed(r.observed)}"
            )
    lines.append("")

    lines.append("## Skipped")
    if not by_verdict["Skipped"]:
        lines.append("_(none)_")
    else:
        for r in by_verdict["Skipped"]:
            lines.append(
                f"- `{r.claim.id}` (§{r.claim.section}) — reason: {r.reason}"
            )
    lines.append("")

    return "\n".join(lines)


# ----------------------------------------------------- report (LaTeX tables)


def _section_to_rq(section: str) -> str:
    """Map a §V section number to its RQ tag (best-effort)."""
    rq_map = {
        "5.1": "RQ1",
        "5.2": "RQ2",
        "5.3": "RQ3",
        "5.4": "RQ4",
        "5.5": "RQ5",
        "5.6": "Ablations",
    }
    return rq_map.get(section.strip(), f"§{section}")


def _latex_escape(value: str) -> str:
    table = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "^": r"\^{}",
        "~": r"\~{}",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(table.get(ch, ch) for ch in value)


def render_tables(results: list[EvaluationResult]) -> str:
    """Render the LaTeX-tables-only output (one per RQ)."""
    by_rq: dict[str, list[EvaluationResult]] = {}
    for r in results:
        rq = _section_to_rq(r.claim.section)
        by_rq.setdefault(rq, []).append(r)

    out: list[str] = []
    out.append(
        "% Auto-generated by scripts/validate_paper_claims.py — do not edit."
    )
    out.append("% Requires booktabs + colortbl in the host document.")
    out.append("")
    for rq in sorted(by_rq):
        rows = sorted(by_rq[rq], key=lambda r: _section_sort_key(r.claim.section))
        out.append(f"% ----- {rq} -----")
        out.append("\\begin{tabular}{llrrl}")
        out.append("\\toprule")
        out.append(
            "\\textbf{Claim id} & \\textbf{§} & \\textbf{Observed} & "
            "\\textbf{Threshold} & \\textbf{Verdict} \\\\"
        )
        out.append("\\midrule")
        for r in rows:
            row_color = ""
            if r.verdict in ("Refuted", "Weaker"):
                row_color = "\\rowcolor{yellow!20} "
            obs_str = "--" if r.observed is None else f"{r.observed:.4f}"
            thr_str = _format_threshold(r.claim.threshold)
            out.append(
                f"{row_color}\\texttt{{{_latex_escape(r.claim.id)}}} & "
                f"{_latex_escape(r.claim.section)} & "
                f"{obs_str} & "
                f"{_latex_escape(thr_str)} & "
                f"{r.verdict} \\\\"
            )
        out.append("\\bottomrule")
        out.append("\\end{tabular}")
        out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------- driver


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns exit code per the spec."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Directory containing the Phase 7 CSVs (e.g. results/raw).",
    )
    parser.add_argument(
        "--claims",
        type=Path,
        required=True,
        help="Path to the claims YAML.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Where to write the Markdown report.",
    )
    parser.add_argument(
        "--tables-out",
        type=Path,
        default=None,
        help="Where to write the LaTeX-tables companion file (optional).",
    )
    parser.add_argument(
        "--section",
        type=str,
        default="all",
        help="Filter to a single §V section (e.g. '5.1'); 'all' for every section.",
    )
    args = parser.parse_args(argv)

    try:
        claims = load_claims(args.claims)
    except ClaimSchemaError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.section != "all":
        claims = [c for c in claims if c.section == args.section]

    results: list[EvaluationResult] = []
    for claim in claims:
        results.append(evaluate(claim, args.results_dir))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = render_report(results, args.claims, args.results_dir)
    args.out.write_text(report, encoding="utf-8")

    if args.tables_out is not None:
        args.tables_out.parent.mkdir(parents=True, exist_ok=True)
        tables = render_tables(results)
        args.tables_out.write_text(tables, encoding="utf-8")

    blocking = sum(
        1 for r in results if r.verdict in ("Refuted", "Weaker")
    )
    return 0 if blocking == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
