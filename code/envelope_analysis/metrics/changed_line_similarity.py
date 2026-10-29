"""Changed-line similarity: does a variant's patch change the same lines as the original's?

For every scored non-original run, the similarity of its patch to the patch of the same issue's
original run (same agent), on the changed lines only: the ``+`` and ``-`` lines of the unified
diff without the ``+++`` / ``---`` file headers, joined with newlines and compared character by
character by ``difflib.SequenceMatcher`` (ratio 2M/T: M matching characters, T characters in both
texts). 1.0 = the same changed lines in the same order; 0.0 = nothing in common. Context lines,
hunk headers and line numbers do not count, so the same edit at a shifted position is identical.
Two runs without a diff are identical (1.0); a run without a diff against one with a diff is 0.0.

Control runs (re-runs of the unchanged original) are scored the same way: their similarity is the
noise floor, what two runs on the very same text reach. Per transformation the engine reports the
mean over issues and the mean paired difference to the control (by issue, variant minus control),
each with a 95% percentile bootstrap CI (5000 resamples, fixed seed), and the two-sided Wilcoxon
signed-rank p-value of the paired differences. With 20 issues and no correction for multiple
comparisons, the tests are exploratory.

Same exclusions as SC/TS: only runs with an outcome are compared, and an issue whose original run
has none contributes nothing. Patches go through ``extract_final_patch`` first, as for
localization (a no-op on a clean diff; older runs stored the whole reply).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from statistics import fmean

from envelope_analysis.evaluators.localization import outcome_value
from envelope_analysis.metrics.patch_similarity import extract_final_patch
from envelope_analysis.metrics.statistics import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    CI_LEVEL,
    WilcoxonResult,
    bootstrap_mean_ci,
    wilcoxon_signed_rank,
)

EXPLORATORY = "exploratory"


def changed_lines(patch: str | None) -> list[str]:
    """The ``+`` / ``-`` lines of a unified diff, prefix included, without the file headers."""
    return [
        line
        for line in (patch or "").replace("\r\n", "\n").split("\n")
        if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
    ]


def changed_line_similarity(patch_a: str | None, patch_b: str | None) -> float:
    """SequenceMatcher ratio of the two patches' changed lines, joined with newlines and compared
    as characters, in [0, 1] (1.0 when both have none).

    ``autojunk`` is off: with it, a character making up more than 1% of a 200+ character text
    (spaces, ``+``) would be ignored as junk.
    """
    text_a = "\n".join(changed_lines(extract_final_patch(patch_a)))
    text_b = "\n".join(changed_lines(extract_final_patch(patch_b)))
    return SequenceMatcher(None, text_a, text_b, autojunk=False).ratio()


@dataclass(frozen=True)
class SimilaritySummary:
    """The mean similarity over issues, its bootstrap CI and the per-issue values behind it."""

    mean: float
    ci_low: float
    ci_high: float
    per_issue: dict[str, float]

    @property
    def n(self) -> int:
        return len(self.per_issue)

    def metadata(self) -> dict:
        return {
            "n": self.n, "ci_low": self.ci_low, "ci_high": self.ci_high, **_bootstrap_settings(),
            "per_issue": self.per_issue,
        }


@dataclass(frozen=True)
class ControlComparison:
    """A transformation against the control, paired by issue: variant minus control."""

    mean_difference: float
    ci_low: float
    ci_high: float
    wilcoxon: WilcoxonResult
    per_issue: dict[str, float]

    @property
    def n_pairs(self) -> int:
        return len(self.per_issue)

    def metadata(self) -> dict:
        return {
            "n_pairs": self.n_pairs, "ci_low": self.ci_low, "ci_high": self.ci_high,
            **_bootstrap_settings(),
            "wilcoxon_p": self.wilcoxon.p_value,
            "wilcoxon_statistic": self.wilcoxon.statistic,
            "wilcoxon_n": self.wilcoxon.n,
            "wilcoxon_method": self.wilcoxon.method,
            "analysis": EXPLORATORY,
            "per_issue": self.per_issue,
        }


@dataclass
class SimilarityResults:
    # run id -> (similarity, the original run it was compared with); control runs included
    per_run: dict[str, tuple[float, str]] = field(default_factory=dict)
    # (agent, transformation) -> summary over issues
    per_transformation: dict[tuple[str, str], SimilaritySummary] = field(default_factory=dict)
    # agent -> summary of the control runs: the noise floor
    control: dict[str, SimilaritySummary] = field(default_factory=dict)
    # (agent, transformation) -> paired comparison with the control
    versus_control: dict[tuple[str, str], ControlComparison] = field(default_factory=dict)


class ChangedLineSimilarity:
    """Compute the changed-line similarity of every compared run and its summaries."""

    def compute(self, run_data: list[dict], control_data: list[dict]) -> SimilarityResults:
        """``run_data``: the scored non-control rows (originals included); ``control_data``: the
        scored control rows. Rows need run_id, agent_name, issue_id, transformation_name,
        is_original, patch and an outcome (or status)."""
        originals: dict[tuple[str, str], dict] = {}
        for row in run_data:
            key = (row["agent_name"], row["issue_id"])
            if row["is_original"] and outcome_value(row) is not None and key not in originals:
                originals[key] = row

        results = SimilarityResults()
        variant_values: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        control_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        compared = [(row, False) for row in run_data if not row["is_original"]]
        compared += [(row, True) for row in control_data]
        for row, is_control in compared:
            original = originals.get((row["agent_name"], row["issue_id"]))
            if original is None or outcome_value(row) is None:
                continue
            value = changed_line_similarity(original["patch"], row["patch"])
            results.per_run[row["run_id"]] = (value, original["run_id"])
            if is_control:
                control_values[row["agent_name"]][row["issue_id"]].append(value)
            else:
                cell = (row["agent_name"], row["transformation_name"])
                variant_values[cell][row["issue_id"]].append(value)

        for agent, by_issue in control_values.items():
            results.control[agent] = _summary(by_issue)
        for (agent, transformation), by_issue in variant_values.items():
            summary = _summary(by_issue)
            results.per_transformation[(agent, transformation)] = summary
            control = results.control.get(agent)
            if control is not None:
                comparison = _compare(summary.per_issue, control.per_issue)
                if comparison is not None:
                    results.versus_control[(agent, transformation)] = comparison
        return results


def _summary(by_issue: dict[str, list[float]]) -> SimilaritySummary:
    """One value per issue (the mean of its runs), their mean and its bootstrap CI."""
    per_issue = {issue_id: fmean(values) for issue_id, values in sorted(by_issue.items())}
    low, high = bootstrap_mean_ci(list(per_issue.values()))
    return SimilaritySummary(fmean(per_issue.values()), low, high, per_issue)


def _compare(variant: dict[str, float], control: dict[str, float]) -> ControlComparison | None:
    """Variant minus control on the issues both have; None when they share none."""
    differences = {
        issue_id: variant[issue_id] - control[issue_id]
        for issue_id in sorted(variant)
        if issue_id in control
    }
    if not differences:
        return None
    values = list(differences.values())
    low, high = bootstrap_mean_ci(values)
    return ControlComparison(fmean(values), low, high, wilcoxon_signed_rank(values), differences)


def _bootstrap_settings() -> dict:
    return {"ci_level": CI_LEVEL, "resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED}
