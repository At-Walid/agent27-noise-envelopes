"""Small-sample statistics for the metrics engine: a percentile bootstrap CI of a mean and the
Wilcoxon signed-rank test, without SciPy.

Both follow the usual reference implementations so that numbers can be checked against them:

- ``bootstrap_mean_ci``: ``numpy.random.default_rng(seed)`` draws ``resamples`` index vectors of
  the sample's size at once (``rng.integers(0, n, size=(resamples, n))``), the CI is the
  ``numpy.percentile`` of the resampled means. A fresh generator per call, so a CI does not depend
  on which other CIs were computed before it.
- ``wilcoxon_signed_rank``: ``scipy.stats.wilcoxon(d)`` with its defaults (two-sided,
  ``zero_method="wilcox"``: zero differences are dropped; no continuity correction). The exact
  null distribution is used when no difference was zero, there are no ties among the absolute
  differences and at most 50 remain; otherwise the normal approximation with the tie correction.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

BOOTSTRAP_RESAMPLES = 5000
BOOTSTRAP_SEED = 42
CI_LEVEL = 0.95
_EXACT_MAX_N = 50

EXACT = "exact"
NORMAL = "normal"
UNDEFINED = "undefined"  # every difference is zero: there is nothing to rank


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    level: float = CI_LEVEL,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval of the mean of ``values`` (resampling values)."""
    sample = np.asarray(values, dtype=float)
    if sample.size == 0:
        raise ValueError("bootstrap_mean_ci needs at least one value")
    rng = np.random.default_rng(seed)
    means = sample[rng.integers(0, sample.size, size=(resamples, sample.size))].mean(axis=1)
    tail = 100 * (1 - level) / 2
    low, high = np.percentile(means, [tail, 100 - tail])
    return float(low), float(high)


@dataclass(frozen=True)
class WilcoxonResult:
    p_value: float | None  # two-sided; None when every difference is zero
    statistic: float | None  # min(W+, W-), the rank sum of the smaller side
    n: int  # non-zero differences ranked
    method: str  # EXACT | NORMAL | UNDEFINED


def wilcoxon_signed_rank(differences: Sequence[float], method: str = "auto") -> WilcoxonResult:
    """Two-sided Wilcoxon signed-rank test of paired differences against a zero median.

    ``method``: "auto" (see the module docstring), or force EXACT / NORMAL (exact is only valid
    without zeros and ties).
    """
    if method not in ("auto", EXACT, NORMAL):
        raise ValueError(f"unknown method {method!r}")
    all_differences = [float(d) for d in differences]
    nonzero = [d for d in all_differences if d != 0.0]
    n = len(nonzero)
    if n == 0:
        return WilcoxonResult(p_value=None, statistic=None, n=0, method=UNDEFINED)

    ranks = _average_ranks([abs(d) for d in nonzero])
    w_plus = sum(rank for rank, d in zip(ranks, nonzero) if d > 0)
    w_minus = sum(rank for rank, d in zip(ranks, nonzero) if d < 0)
    statistic = min(w_plus, w_minus)
    tie_sizes = _tie_sizes([abs(d) for d in nonzero])

    had_zeros = n < len(all_differences)
    exact = n <= _EXACT_MAX_N and not had_zeros and not tie_sizes
    if method == EXACT or (method == "auto" and exact):
        return WilcoxonResult(_exact_p(int(w_plus), n), statistic, n, EXACT)

    mean = n * (n + 1) / 4
    variance = n * (n + 1) * (2 * n + 1) / 24 - sum(t**3 - t for t in tie_sizes) / 48
    if variance <= 0:
        return WilcoxonResult(p_value=1.0, statistic=statistic, n=n, method=NORMAL)
    z = (w_plus - mean) / math.sqrt(variance)
    p_value = min(1.0, math.erfc(abs(z) / math.sqrt(2)))  # = 2 * (1 - Phi(|z|))
    return WilcoxonResult(p_value, statistic, n, NORMAL)


def _average_ranks(values: list[float]) -> list[float]:
    """1-based ranks, ties sharing the average of the ranks they span."""
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and values[order[end + 1]] == values[order[start]]:
            end += 1
        average = (start + end) / 2 + 1
        for position in range(start, end + 1):
            ranks[order[position]] = average
        start = end + 1
    return ranks


def _tie_sizes(values: list[float]) -> list[int]:
    """The size of every group of equal values that has more than one member."""
    counts: dict[float, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return [count for count in counts.values() if count > 1]


def _exact_p(w_plus: int, n: int) -> float:
    """Two-sided p-value of W+ under the exact null distribution (ranks 1..n, no ties)."""
    # counts[s] = number of subsets of {1..n} whose sum is s, i.e. sign patterns giving W+ = s.
    total = n * (n + 1) // 2
    counts = [1] + [0] * total
    for rank in range(1, n + 1):
        for s in range(total, rank - 1, -1):
            counts[s] += counts[s - rank]
    if 2 * w_plus == total:
        return 1.0
    patterns = 2**n
    p_less = sum(counts[: w_plus + 1]) / patterns
    p_greater = sum(counts[w_plus:]) / patterns
    return min(1.0, 2 * min(p_less, p_greater))


def rank_biserial(differences: Sequence[float]) -> float | None:
    """Matched-pairs rank-biserial correlation of paired differences, the effect size of the
    Wilcoxon signed-rank test: (W+ - W-) / (W+ + W-) over the non-zero differences (zero ones
    dropped, as in the test; ties share their average rank), in [-1, 1]. None when every
    difference is zero."""
    nonzero = [float(d) for d in differences if float(d) != 0.0]
    if not nonzero:
        return None
    ranks = _average_ranks([abs(d) for d in nonzero])
    w_plus = sum(rank for rank, d in zip(ranks, nonzero) if d > 0)
    w_minus = sum(rank for rank, d in zip(ranks, nonzero) if d < 0)
    return (w_plus - w_minus) / (w_plus + w_minus)


def benjamini_hochberg(p_values: Sequence[float | None]) -> list[float | None]:
    """Benjamini-Hochberg adjusted p-values (q-values), in the order given, over the p-values of
    one family; None stays None and is not counted in the family."""
    indexed = [(i, float(p)) for i, p in enumerate(p_values) if p is not None]
    m = len(indexed)
    adjusted: list[float | None] = [None] * len(p_values)
    running = 1.0
    for rank, (i, p) in reversed(list(enumerate(sorted(indexed, key=lambda item: item[1]), 1))):
        running = min(running, p * m / rank)
        adjusted[i] = min(1.0, running)
    return adjusted


def wilson_ci(successes: int, n: int, level: float = CI_LEVEL) -> tuple[float, float] | None:
    """Wilson score interval of a proportion (None when n is 0)."""
    if n == 0:
        return None
    z = _normal_quantile(1 - (1 - level) / 2)
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def _normal_quantile(q: float) -> float:
    """Standard normal quantile by bisection on erfc (no SciPy)."""
    low, high = -10.0, 10.0
    for _ in range(200):
        mid = (low + high) / 2
        if 0.5 * math.erfc(-mid / math.sqrt(2)) < q:
            low = mid
        else:
            high = mid
    return (low + high) / 2
