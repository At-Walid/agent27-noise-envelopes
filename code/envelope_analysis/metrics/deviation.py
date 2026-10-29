"""How far a variant's patch moves beyond the run-to-run noise of the original (noise envelope).

Definitions of the study design v1.0, section 8 (docs/analysis_plan.md).
With ``k`` runs of each issue's unchanged original per backend (the original run and its control
repeats: ``control_repeats=5`` gives k = 6), the issue's **noise envelope** is the set of the
k(k-1)/2 pairwise changed-line similarities between those runs (15 for k = 6). A variant run is
compared with every one of the k originals; its similarity ``s`` is the mean of those k values.

Deviation scores of a variant, per backend and issue:

- **z**: ``(s - envelope mean) / max(envelope sd, sd_floor)``. Negative: less similar to the
  originals than they are to each other. Deviating when ``z < z_threshold`` (-2).
- **rank** (robustness): ``s`` ranked among the k originals' own leave-one-out mean similarities
  (original j: the mean of its similarities to the k-1 others), ``(1 + #{loo_j <= s}) / (k + 1)``;
  deviating when ``s`` is below every one of them (rank = 1 / (k + 1)).

Outcome flips count only when the originals agree: when every scored original (at least two) has
the same outcome, a variant with the other outcome is a **gain** (originals fail, variant
succeeds) or a **loss**; otherwise the issue's flips are undetermined. An issue is **noisy** when
its scored originals disagree on the outcome (``noisy_similarity``, off by default, also marks an
issue whose envelope mean is below it). The **binary deviation** used for triage (RQ4) is
``z < -2`` or an outcome flip on a non-noisy issue.

Per transformation (paired by issue, one value per issue; 95% bootstrap CIs clustered by issue,
Wilcoxon signed-rank with zero differences dropped, matched-pairs rank-biserial):

- the **envelope** method: ``s`` minus the envelope mean;
- the **single-repeat** method (``single_repeat``, the earlier pilot's): the variant's similarity to the
  original run (repeat 0) minus the similarity of the first repeat to it;
- **conclusion stability**: the single-repeat method recomputed with each repeat r = 1..k-1 as
  the repeat; whether its conclusion (the CI excludes zero or not) changes with r.

Pure functions of ``RunRecord`` lists: ``paper_analysis.py`` builds them from data/runs.csv.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from itertools import combinations
from statistics import fmean, stdev
from typing import Optional

from envelope_analysis.metrics.changed_line_similarity import changed_line_similarity
from envelope_analysis.metrics.statistics import (
    bootstrap_mean_ci,
    rank_biserial,
    wilcoxon_signed_rank,
)

ORIGINAL = "original"
DEFAULT_Z_THRESHOLD = -2.0
DEFAULT_SD_FLOOR = 0.05
DEFAULT_NOISY_SIMILARITY: Optional[float] = None  # off: noisy means outcome disagreement only
GAIN, LOSS, UNDETERMINED = "gain", "loss", "undetermined"

Similarity = Callable[[Optional[str], Optional[str]], float]


@dataclass(frozen=True)
class RunRecord:
    """One scored run: its backend, issue, variant ("original" or a transformation) and repeat
    number (0: the original run or the variant run; 1..k-1: control repeats of the original)."""

    backend: str
    issue_id: str
    variant: str
    repeat: int
    patch: Optional[str]
    outcome: Optional[int]  # 1 success, 0 failure, None: excluded from the outcome metrics


@dataclass(frozen=True)
class NoiseEnvelope:
    backend: str
    issue_id: str
    n_originals: int
    pairwise: tuple[float, ...]
    original_outcomes: tuple[Optional[int], ...]
    # the original runs' repeat numbers, in the order of ``pairwise``'s combinations
    repeats: tuple[int, ...] = ()
    # the originals' patches are byte-identical, per pair (same order as ``pairwise``)
    identical: tuple[bool, ...] = ()

    @property
    def n_pairs(self) -> int:
        return len(self.pairwise)

    @property
    def mean(self) -> Optional[float]:
        return fmean(self.pairwise) if self.pairwise else None

    @property
    def sd(self) -> Optional[float]:
        return stdev(self.pairwise) if len(self.pairwise) > 1 else None

    @property
    def width(self) -> Optional[float]:
        """Max minus min of the pairwise similarities."""
        return max(self.pairwise) - min(self.pairwise) if self.pairwise else None

    @property
    def minimum(self) -> Optional[float]:
        return min(self.pairwise) if self.pairwise else None

    @property
    def leave_one_out(self) -> tuple[float, ...]:
        """Per original run, the mean of its similarities to the other originals."""
        k = self.n_originals
        if k < 2 or len(self.pairwise) != k * (k - 1) // 2:
            return ()
        sums = [0.0] * k
        for value, (i, j) in zip(self.pairwise, combinations(range(k), 2)):
            sums[i] += value
            sums[j] += value
        return tuple(total / (k - 1) for total in sums)

    @property
    def scored_outcomes(self) -> list[int]:
        return [o for o in self.original_outcomes if o is not None]

    @property
    def originals_agree(self) -> bool:
        scored = self.scored_outcomes
        return len(scored) >= 2 and len(set(scored)) == 1

    @property
    def originals_disagree(self) -> bool:
        return len(set(self.scored_outcomes)) > 1

    @property
    def agreed_outcome(self) -> Optional[int]:
        return self.scored_outcomes[0] if self.originals_agree else None


@dataclass(frozen=True)
class VariantDeviation:
    backend: str
    issue_id: str
    transformation: str
    similarity: float  # mean similarity to the k originals
    to_originals: tuple[float, ...]
    z: Optional[float]
    rank: Optional[float]
    deviates_z: bool
    deviates_rank: bool
    outcome: Optional[int]
    flip: Optional[str]  # GAIN | LOSS | None (no flip) | UNDETERMINED (originals do not agree)
    noisy_issue: bool
    # The single-repeat method, on the same runs: similarity to the original run (repeat 0, None
    # when it was excluded) and that run's outcome flip (no agreement required).
    first_run_similarity: Optional[float]
    first_run_flip: Optional[str]
    envelope_mean: Optional[float] = None

    @property
    def deviates(self) -> bool:
        """The binary deviation of the triage (RQ4): z < -2, or an outcome flip on a non-noisy
        issue (a flip is only counted when the originals agree)."""
        return self.deviates_z or (self.flip in (GAIN, LOSS) and not self.noisy_issue)


@dataclass(frozen=True)
class PairedComparison:
    """Per-issue paired differences of one transformation: mean, 95% bootstrap CI (issues
    resampled), Wilcoxon signed-rank p (zero differences dropped) and rank-biserial."""

    n: int
    mean_difference: Optional[float]
    ci_low: Optional[float]
    ci_high: Optional[float]
    wilcoxon_p: Optional[float]
    rank_biserial: Optional[float]

    @property
    def excludes_zero(self) -> Optional[bool]:
        if self.ci_low is None or self.ci_high is None:
            return None
        return self.ci_high < 0 or self.ci_low > 0

    @property
    def conclusion(self) -> Optional[str]:
        """"lower" / "higher" (the CI excludes zero on that side) or "no difference"."""
        if self.excludes_zero is None:
            return None
        if not self.excludes_zero:
            return "no difference"
        return "lower" if self.ci_high < 0 else "higher"


def paired_comparison(differences: list[float]) -> PairedComparison:
    if not differences:
        return PairedComparison(0, None, None, None, None, None)
    low, high = bootstrap_mean_ci(differences)
    return PairedComparison(
        len(differences), fmean(differences), low, high,
        wilcoxon_signed_rank(differences).p_value, rank_biserial(differences),
    )


@dataclass(frozen=True)
class TransformationSummary:
    backend: str
    transformation: str
    n: int
    deviating_z: int
    deviating_rank: int
    deviating: int  # the binary deviation of the triage
    gains: int
    losses: int
    undetermined: int  # variants of issues whose originals do not agree on the outcome
    noisy_issues: int
    mean_z: Optional[float]
    # the envelope method: s minus the envelope mean, paired by issue
    envelope: Optional[PairedComparison] = None

    @property
    def deviation_rate_z(self) -> Optional[float]:
        return self.deviating_z / self.n if self.n else None

    @property
    def deviation_rate_rank(self) -> Optional[float]:
        return self.deviating_rank / self.n if self.n else None


@dataclass(frozen=True)
class SingleRepeatSummary:
    """The single-repeat method (the earlier pilot's) on these runs: variant vs the original run,
    against the first repeat of the original."""

    backend: str
    transformation: str
    n: int
    mean_similarity: Optional[float]
    repeat_mean: Optional[float]
    mean_difference: Optional[float]
    ci_low: Optional[float]
    ci_high: Optional[float]
    wilcoxon_p: Optional[float]
    flips: int
    rank_biserial: Optional[float] = None
    comparison: Optional[PairedComparison] = None  # the paired difference, per issue


@dataclass(frozen=True)
class ConclusionStability:
    """The single-repeat comparison recomputed with each repeat as "the" repeat."""

    backend: str
    transformation: str
    by_repeat: dict[int, PairedComparison]

    @property
    def conclusions(self) -> dict[int, Optional[str]]:
        return {r: c.conclusion for r, c in self.by_repeat.items()}

    @property
    def excludes_zero(self) -> dict[int, Optional[bool]]:
        return {r: c.excludes_zero for r, c in self.by_repeat.items()}

    @property
    def changes(self) -> bool:
        """Whether the conclusion (the CI excludes zero or not) depends on the repeat used."""
        values = {v for v in self.excludes_zero.values() if v is not None}
        return len(values) > 1


@dataclass
class DeviationReport:
    envelopes: list[NoiseEnvelope] = field(default_factory=list)
    variants: list[VariantDeviation] = field(default_factory=list)
    summaries: list[TransformationSummary] = field(default_factory=list)
    single_repeat: list[SingleRepeatSummary] = field(default_factory=list)
    conclusion_stability: list[ConclusionStability] = field(default_factory=list)
    settings: dict = field(default_factory=dict)


def noise_envelopes(
    records: Iterable[RunRecord], similarity: Similarity = changed_line_similarity
) -> dict[tuple[str, str], NoiseEnvelope]:
    originals: dict[tuple[str, str], list[RunRecord]] = defaultdict(list)
    for record in records:
        if record.variant == ORIGINAL:
            originals[(record.backend, record.issue_id)].append(record)
    envelopes = {}
    for (backend, issue_id), runs in sorted(originals.items()):
        runs = sorted(runs, key=lambda r: r.repeat)
        pairs = list(combinations(runs, 2))
        envelopes[(backend, issue_id)] = NoiseEnvelope(
            backend, issue_id, len(runs),
            tuple(similarity(a.patch, b.patch) for a, b in pairs),
            tuple(r.outcome for r in runs),
            tuple(r.repeat for r in runs),
            tuple((a.patch or "") == (b.patch or "") for a, b in pairs),
        )
    return envelopes


def _flip(reference: Optional[int], outcome: Optional[int]) -> Optional[str]:
    if reference is None or outcome is None or reference == outcome:
        return None
    return GAIN if outcome == 1 else LOSS


def deviation_report(
    records: Iterable[RunRecord],
    *,
    z_threshold: float = DEFAULT_Z_THRESHOLD,
    sd_floor: float = DEFAULT_SD_FLOOR,
    noisy_similarity: Optional[float] = DEFAULT_NOISY_SIMILARITY,
    similarity: Similarity = changed_line_similarity,
) -> DeviationReport:
    records = list(records)
    envelopes = noise_envelopes(records, similarity)
    originals = {
        key: sorted(
            (r for r in records if r.variant == ORIGINAL and (r.backend, r.issue_id) == key),
            key=lambda r: r.repeat,
        )
        for key in envelopes
    }
    report = DeviationReport(
        envelopes=list(envelopes.values()),
        settings={"z_threshold": z_threshold, "deviating_when": "z < z_threshold",
                  "sd_floor": sd_floor, "noisy_similarity": noisy_similarity,
                  "rank": "leave-one-out means of the originals"},
    )
    for record in records:
        if record.variant == ORIGINAL:
            continue
        key = (record.backend, record.issue_id)
        envelope = envelopes.get(key)
        if envelope is None or not originals[key]:
            continue
        to_originals = tuple(similarity(record.patch, o.patch) for o in originals[key])
        s = fmean(to_originals)
        z = rank = None
        if envelope.pairwise:
            sd = max(envelope.sd or 0.0, sd_floor)
            z = (s - envelope.mean) / sd
        loo = envelope.leave_one_out
        if loo:
            rank = (1 + sum(1 for value in loo if value <= s)) / (len(loo) + 1)
        noisy = envelope.originals_disagree or (
            noisy_similarity is not None and envelope.mean is not None
            and envelope.mean < noisy_similarity
        )
        if envelope.originals_agree:
            flip = _flip(envelope.agreed_outcome, record.outcome)
        else:
            flip = UNDETERMINED if record.outcome is not None else None
        first = originals[key][0] if originals[key][0].repeat == 0 else None
        report.variants.append(VariantDeviation(
            backend=record.backend, issue_id=record.issue_id, transformation=record.variant,
            similarity=s, to_originals=to_originals, z=z, rank=rank,
            deviates_z=z is not None and z < z_threshold,
            deviates_rank=bool(loo) and s < min(loo),  # rank = 1 / (k + 1)
            outcome=record.outcome, flip=flip, noisy_issue=noisy,
            first_run_similarity=similarity(record.patch, first.patch) if first else None,
            first_run_flip=_flip(first.outcome, record.outcome) if first else None,
            envelope_mean=envelope.mean,
        ))
    report.summaries = _summaries(report.variants)
    report.single_repeat = _single_repeat_summaries(report.variants, originals, similarity)
    report.conclusion_stability = _conclusion_stability(report.variants, originals, similarity)
    return report


def _by_issue(values: Iterable[tuple[str, float]]) -> dict[str, float]:
    """One value per issue: the mean of its values (variant repeats)."""
    grouped: dict[str, list[float]] = defaultdict(list)
    for issue_id, value in values:
        grouped[issue_id].append(value)
    return {issue_id: fmean(v) for issue_id, v in sorted(grouped.items())}


def _summaries(variants: list[VariantDeviation]) -> list[TransformationSummary]:
    groups: dict[tuple[str, str], list[VariantDeviation]] = defaultdict(list)
    for v in variants:
        groups[(v.backend, v.transformation)].append(v)
    out = []
    for (backend, transformation), vs in sorted(groups.items()):
        zs = [v.z for v in vs if v.z is not None]
        envelope = _by_issue((v.issue_id, v.similarity - v.envelope_mean)
                             for v in vs if v.envelope_mean is not None)
        out.append(TransformationSummary(
            backend=backend, transformation=transformation, n=len(vs),
            deviating_z=sum(v.deviates_z for v in vs),
            deviating_rank=sum(v.deviates_rank for v in vs),
            deviating=sum(v.deviates for v in vs),
            gains=sum(v.flip == GAIN for v in vs),
            losses=sum(v.flip == LOSS for v in vs),
            undetermined=sum(v.flip == UNDETERMINED for v in vs),
            noisy_issues=sum(v.noisy_issue for v in vs),
            mean_z=fmean(zs) if zs else None,
            envelope=paired_comparison(list(envelope.values())),
        ))
    return out


def _repeat_similarity(
    originals: dict[tuple[str, str], list[RunRecord]], similarity: Similarity, repeat: int
) -> dict[tuple[str, str], float]:
    """Per (backend, issue): the similarity of repeat ``repeat`` to the original run (repeat 0);
    issues missing either run (excluded) are left out."""
    out = {}
    for key, runs in originals.items():
        by_repeat = {r.repeat: r for r in runs}
        if 0 in by_repeat and repeat in by_repeat:
            out[key] = similarity(by_repeat[0].patch, by_repeat[repeat].patch)
    return out


def _single_repeat_differences(
    vs: list[VariantDeviation], repeat_of: dict[tuple[str, str], float], backend: str
) -> dict[str, float]:
    return _by_issue(
        (v.issue_id, v.first_run_similarity - repeat_of[(backend, v.issue_id)])
        for v in vs
        if v.first_run_similarity is not None and (backend, v.issue_id) in repeat_of
    )


def _single_repeat_summaries(
    variants: list[VariantDeviation],
    originals: dict[tuple[str, str], list[RunRecord]],
    similarity: Similarity,
) -> list[SingleRepeatSummary]:
    repeat_of = _repeat_similarity(originals, similarity, 1)
    groups: dict[tuple[str, str], list[VariantDeviation]] = defaultdict(list)
    for v in variants:
        groups[(v.backend, v.transformation)].append(v)
    out = []
    for (backend, transformation), vs in sorted(groups.items()):
        diffs = _single_repeat_differences(vs, repeat_of, backend)
        comparison = paired_comparison(list(diffs.values()))
        repeats = [repeat_of[(backend, i)] for i in diffs]
        sims = [v.first_run_similarity for v in vs if v.first_run_similarity is not None]
        out.append(SingleRepeatSummary(
            backend=backend, transformation=transformation, n=len(vs),
            mean_similarity=fmean(sims) if sims else None,
            repeat_mean=fmean(repeats) if repeats else None,
            mean_difference=comparison.mean_difference,
            ci_low=comparison.ci_low, ci_high=comparison.ci_high,
            wilcoxon_p=comparison.wilcoxon_p,
            flips=sum(v.first_run_flip is not None for v in vs),
            rank_biserial=comparison.rank_biserial,
            comparison=comparison,
        ))
    return out


def _conclusion_stability(
    variants: list[VariantDeviation],
    originals: dict[tuple[str, str], list[RunRecord]],
    similarity: Similarity,
) -> list[ConclusionStability]:
    repeats = sorted({r.repeat for runs in originals.values() for r in runs if r.repeat > 0})
    repeat_of = {r: _repeat_similarity(originals, similarity, r) for r in repeats}
    groups: dict[tuple[str, str], list[VariantDeviation]] = defaultdict(list)
    for v in variants:
        groups[(v.backend, v.transformation)].append(v)
    return [
        ConclusionStability(backend, transformation, {
            r: paired_comparison(list(_single_repeat_differences(vs, repeat_of[r], backend).values()))
            for r in repeats
        })
        for (backend, transformation), vs in sorted(groups.items())
    ]

