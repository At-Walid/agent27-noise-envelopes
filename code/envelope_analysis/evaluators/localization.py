"""Run outcome for metrics A, F and G: "localization success" (SPEC section 13).

A run succeeds iff the agent returned a valid unified diff (status "success": the adapter could
extract one) AND that diff modifies every file the issue's gold patch modifies
(``localization_match``). Every other scored run is a failure, including "no_patch". Runs that did
not complete normally (timeout, truncated, cost_exceeded, error, dry_run_skipped, ...) get no
outcome: they are excluded from SC/TS rather than counted as failures. "truncated" is a reply cut at
the token limit with no usable diff: a budget limit like cost_exceeded, not an answer; runs
recorded before that status existed (status "no_patch", stop_reason "max_tokens") are read as
truncated (``scoring_status``). So are runs on variants flagged
``quality_warning`` or ``noop`` (see the metrics engine; the runner records them as
"skipped_variant" without calling the agent).

Localization success is weaker than test-based resolution: touching the right files says nothing
about whether the change is correct. Test execution is deferred to the larger study.

Without a gold patch (custom issues, synthetic demo fixtures) localization cannot be checked; such
runs fall back to "valid diff extracted" and are reported as such by the metrics engine.

Everything that shows an outcome (metrics engine, dashboard, exports) goes through this module, so
the numbers and the GUI cannot disagree.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from envelope_analysis.metrics.patch_similarity import (
    extract_final_patch,
    file_overlap_jaccard,
    patch_files,
)

SUCCESS = "success"
FAILURE = "failure"
EXCLUDED = "excluded"

# Statuses that describe what the agent produced. Any other status means the run did not complete
# normally and is excluded from the outcome-based metrics.
SCORED_STATUSES = frozenset({"success", "failure", "no_patch"})
# A reply without a usable diff: a failure in every project (SPEC section 13). A sensitivity
# analysis may exclude it instead (``score_runs(..., exclude_no_patch=True)``).
NO_PATCH = "no_patch"
# A reply cut at the token limit without a usable diff: excluded like a timeout (SPEC section 13).
TRUNCATED = "truncated"


def scoring_status(status: str, metadata: str | dict | None) -> str:
    """The status a run is scored with: its own, except that a "no_patch" run whose reply was cut
    at the token limit (``stop_reason`` "max_tokens" in its metadata), recorded before the
    adapter had a "truncated" status, counts as truncated. Stored runs are never rewritten."""
    if status != "no_patch" or not metadata:
        return status
    if isinstance(metadata, str):
        import json

        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            return status
    if isinstance(metadata, dict) and metadata.get("stop_reason") == "max_tokens":
        return TRUNCATED
    return status


@dataclass(frozen=True)
class Localization:
    files_in_patch: list[str]
    files_in_gold: list[str]
    file_overlap_jaccard: float
    localization_match: bool  # every gold file is modified by the patch


def localize(patch: str | None, gold_patch: str) -> Localization:
    """Compare the files a run's patch modifies with the files the gold patch modifies."""
    # Runs recorded before the adapter extracted patches stored the whole reply; extraction is a
    # no-op on an already-clean diff.
    in_patch = patch_files(extract_final_patch(patch))
    in_gold = patch_files(gold_patch)
    return Localization(
        files_in_patch=in_patch,
        files_in_gold=in_gold,
        file_overlap_jaccard=file_overlap_jaccard(in_patch, in_gold),
        localization_match=bool(in_gold) and set(in_gold) <= set(in_patch),
    )


def run_outcome(status: str, localization_match: bool | None) -> str | None:
    """"success", "failure", or None when the run is excluded from the outcome metrics.

    ``localization_match`` is None when the issue has no gold patch: the outcome then falls back to
    "a valid diff was extracted".
    """
    if status not in SCORED_STATUSES:
        return None
    if status != "success":
        return FAILURE
    if localization_match is None:
        return SUCCESS
    return SUCCESS if localization_match else FAILURE


def outcome_value(row: dict) -> int | None:
    """1 / 0 for a run_data row, None when the run is excluded.

    Rows carrying an "outcome" key (from the metrics engine) use it; rows with only a "status"
    are scored as if no gold patch existed.
    """
    outcome = row["outcome"] if "outcome" in row else run_outcome(row["status"], None)
    if outcome not in (SUCCESS, FAILURE):
        return None
    return int(outcome == SUCCESS)


# ---------------------------------------------------------------------------
# Reading the engine's results back (dashboard and exports)
#
# Metric rows are given as (scope, scope_key, metric_name, metric_value) tuples so the same code
# serves ORM rows (exports) and DataFrame rows (dashboard).
# ---------------------------------------------------------------------------


def recorded_outcomes(metric_rows: Iterable[tuple]) -> dict[str, str]:
    """run id -> "success" | "failure", from the engine's run-scope ``outcome`` metrics."""
    return {
        scope_key: SUCCESS if (value or 0.0) >= 0.5 else FAILURE
        for scope, scope_key, name, value in metric_rows
        if scope == "run" and name == "outcome"
    }


def recorded_run_values(metric_rows: Iterable[tuple], metric_name: str) -> dict[str, float]:
    """run id -> value of a run-scope metric (e.g. ``file_overlap_jaccard``) the engine stored."""
    return {
        scope_key: value
        for scope, scope_key, name, value in metric_rows
        if scope == "run" and name == metric_name
    }


def displayed_outcome(run_id: str, status: str, recorded: dict[str, str]) -> str:
    """The outcome every view shows for a run: "success", "failure" or "excluded".

    It is the outcome the metrics engine recorded; a run it left out is "excluded". Experiments
    scored before run-level outcomes existed have no recorded outcomes at all and fall back to the
    status-only rule, which is what their stored metrics were computed with.
    """
    if recorded:
        return recorded.get(run_id, EXCLUDED)
    return run_outcome(status, None) or EXCLUDED


def exclusion_counts(metric_rows: Iterable[tuple]) -> dict:
    """Runs the engine excluded: totals by reason, and per transformation for the variant reasons
    (quality warning, no-op)."""
    counts: dict = {
        "status": 0,
        "quality_warning": 0,
        "quality_warning_by_transformation": {},
        "noop": 0,
        "noop_by_transformation": {},
    }
    per_cell_metric = {
        "excluded_quality_warning": counts["quality_warning_by_transformation"],
        "excluded_noop": counts["noop_by_transformation"],
    }
    totals = {
        "excluded_runs_status": "status",
        "excluded_runs_quality_warning": "quality_warning",
        "excluded_runs_noop": "noop",
    }
    for _scope, scope_key, name, value in metric_rows:
        if name in totals:
            counts[totals[name]] = int(value or 0)
        elif name in per_cell_metric:
            per_transformation = per_cell_metric[name]
            transformation = scope_key.split("|", 1)[-1]
            per_transformation[transformation] = (
                per_transformation.get(transformation, 0) + int(value or 0)
            )
    return counts
