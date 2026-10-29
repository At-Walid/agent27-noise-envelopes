"""The paper: every table, figure and cited number, from the CSV data alone.

    python paper_analysis.py [--data DIR] [--out DIR]

No database, no API key, no network. The inputs are data/runs.csv and data/issues.csv; the
metrics are those of envelope_analysis/metrics/deviation.py and the modules it imports, next to
this script. The defaults are ../data/ and ../results/.

Writes each table as .csv and as a LaTeX table (booktabs; one IEEE column unless noted):

RQ1  rq1_noise (per backend: issues localized in 6/6 and 0/6 runs, noisy issues, repeat
     similarity median and IQR, byte-identical and same-changed-lines repeat pairs, control flips
     against the first run, envelope width), rq1_noisy_issues (the noisy issues and their 6
     outcomes), rq1_backends (per-issue repeat similarity and envelope width, Claude minus GPT-4o,
     paired by issue). rq1_per_issue.csv: the per-issue values behind Fig. 1 (CSV only).
RQ2  rq2_envelope (per backend and transformation: the envelope-method deviation, its CI,
     Wilcoxon p, BH q over the RQ2 family; the variants outside the envelope; gains and losses on
     non-noisy issues), rq2_conclusion_stability (the single-repeat estimate with each of the 5
     repeats; also rq2_conclusion_stability_ci.tex, two columns, with the CIs),
     rq2_posthoc_typo (POST-HOC, exploratory: each transformation minus typo injection, paired by
     issue; docs/deviations.md, section 2).
RQ3  rq3_filelist (Claude, the 25 RQ3 issues: file list vs issue only, paired by issue; BH q over
     the RQ3 family; exploratory), with descriptive rows (identical repeat pairs, localization
     counts, input tokens).
Figures (PDF): fig1_repeat_similarity (one column), fig2_conclusion_stability (two columns),
     fig3_rq3_deviation (one column).
Robustness: the whole pipeline again with no_patch runs excluded instead of counted as failures,
     in robustness_no_patch_excluded/; robustness_changes lists what differs.
NUMBERS.md: every number the paper cites, with its definition and source file.

Definitions are those of envelope_analysis/metrics/deviation.py and the study design
(docs/analysis_plan.md, sections 8, 9 and 11).
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
DATA_DIR, OUT_DIR = HERE.parent / "data", HERE.parent / "results"

from envelope_analysis.metrics.deviation import (  # noqa: E402
    DeviationReport,
    PairedComparison,
    RunRecord,
    deviation_report,
    paired_comparison,
)
from envelope_analysis.metrics.patch_similarity import extract_final_patch, patch_files  # noqa: E402
from envelope_analysis.metrics.statistics import (  # noqa: E402
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    CI_LEVEL,
    benjamini_hochberg,
    bootstrap_mean_ci,
)

CONDITIONS = ("issue only", "file list")
NO_PATCH = "no_patch"
CLAUDE, GPT = "Claude Sonnet 4.6", "GPT-4o"
BACKENDS = (CLAUDE, GPT)
SHORT_BACKEND = {CLAUDE: "Claude", GPT: "GPT-4o"}
REFERENCE = "typo_injection"
ORDER = ("typo_injection", "reordering", "paraphrase", "verbosity_reduction",
         "paragraph_to_structured", "neutralize_emotional")
LABELS = {"typo_injection": "Typo injection", "reordering": "Reordering",
          "paraphrase": "Paraphrase", "verbosity_reduction": "Verbosity reduction",
          "paragraph_to_structured": "Paragraph → structured",
          "neutralize_emotional": "Neutralize emotional"}
TEX_LABELS = {"typo_injection": "Typo injection", "reordering": "Reordering",
              "paraphrase": "Paraphrase", "verbosity_reduction": "Verbosity red.",
              "paragraph_to_structured": r"Para.\,$\to$\,struct.",
              "neutralize_emotional": "Neutralize emot."}
DEVIATION, BINARY = "deviation", "binary deviation"
MEASURES = (DEVIATION, BINARY)
ANALYSIS = "post-hoc, exploratory"
ROBUSTNESS_DIR = "robustness_no_patch_excluded"

# The paper style of every figure: Claude blue,
# filled circles, dashed; GPT-4o orange, hollow squares, dotted.
STYLE = {CLAUDE: {"color": "#2a78d6", "marker": "o", "filled": True, "ls": (0, (4, 2))},
         GPT: {"color": "#eb6834", "marker": "s", "filled": False, "ls": (0, (1, 1.5))}}
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e9e8e4"
PAPER_RC = {"font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "font.size": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 7,
            "ytick.labelsize": 7.5, "legend.fontsize": 7,
            "pdf.fonttype": 42, "ps.fonttype": 42}  # TrueType: PDF eXpress rejects Type 3
COLUMN_IN, TEXT_IN = 3.5, 7.0
COLUMN_PT, TEXT_PT = 252.0, 516.0  # \columnwidth and \textwidth of IEEEtran (conference)


# ── Data ─────────────────────────────────────────────────────────────────────

def load_runs(path: Path) -> pd.DataFrame:
    runs = pd.read_csv(path, dtype=str, keep_default_na=False)
    for column in ("run_order", "repeat", "control_repeat"):
        runs[column] = runs[column].astype(int)
    return runs


def load_issues(path: Path) -> pd.DataFrame:
    issues = pd.read_csv(path, dtype=str, keep_default_na=False)
    issues["rq3"] = issues["rq3"].astype(int)
    return issues


def records(runs: pd.DataFrame, condition: str, exclude_no_patch: bool = False) -> list[RunRecord]:
    """The scored runs of one condition: the runs the metrics engine scores (``excluded``
    empty), the non-control runs first, each in run order. ``exclude_no_patch``: the
    robustness analysis (no_patch runs left out)."""
    rows = runs[(runs.condition == condition) & (runs.excluded == "")]
    if exclude_no_patch:
        rows = rows[rows.scoring_status != NO_PATCH]
    rows = rows.sort_values(["control_repeat", "run_order"], kind="stable")
    return [RunRecord(r.backend, r.issue_id, r.variant, r.repeat, r.patch or None, int(r.outcome))
            for r in rows.itertuples(index=False)]


def repository(issue_id: str) -> str:
    owner, rest = issue_id.split("__", 1)
    return f"{owner}/{rest.rsplit('-', 1)[0]}"


def add_bh(df: pd.DataFrame, p_columns: list[str]) -> pd.DataFrame:
    """Benjamini-Hochberg q-values over every p-value of the columns ``p_columns`` (one family),
    each in a column ``<p column without p>q``."""
    cells = [(c, i, df.at[i, c]) for c in p_columns for i in df.index]
    q = benjamini_hochberg([None if pd.isna(p) else p for _c, _i, p in cells])
    for c in p_columns:
        df[c[:-1] + "q"] = None
    for (c, i, _p), value in zip(cells, q):
        df.at[i, c[:-1] + "q"] = value
    return df


def comparison_row(prefix: str, c: PairedComparison) -> dict:
    return {f"{prefix}n": c.n, f"{prefix}diff": c.mean_difference, f"{prefix}ci_low": c.ci_low,
            f"{prefix}ci_high": c.ci_high, f"{prefix}p": c.wilcoxon_p,
            f"{prefix}rank_biserial": c.rank_biserial, f"{prefix}verdict": c.conclusion}


def _missing(v) -> bool:
    return v is None or v is pd.NA or (isinstance(v, float) and np.isnan(v))


# ── Numbers cited in the paper (NUMBERS.md) ──────────────────────────────────

@dataclass
class Numbers:
    rows: list[tuple[str, str, str, str, str]] = field(default_factory=list)

    def add(self, section: str, name: str, value, definition: str, source: str) -> None:
        self.rows.append((section, name, str(value), definition, source))

    def markdown(self, preamble: str) -> str:
        lines = [preamble]
        for section in dict.fromkeys(s for s, *_ in self.rows):
            lines += ["", f"## {section}", "", "| Number | Value | Definition | Source |",
                      "|---|---|---|---|"]
            lines += [f"| {n} | {v} | {d} | {s} |".replace("\n", " ")
                      for sec, n, v, d, s in self.rows if sec == section]
        return "\n".join(lines) + "\n"


def f3(v, digits: int = 3) -> str:
    return "--" if _missing(v) else f"{v:.{digits}f}"


def fq(q) -> str:
    return "--" if _missing(q) else ("<0.001" if q < 0.001 else f"{q:.3f}")


def fci(low, high, digits: int = 3) -> str:
    return "--" if _missing(low) else f"[{low:.{digits}f}, {high:.{digits}f}]"


def eq(name: str, v) -> str:
    return f"{name} = --" if _missing(v) else (f"{name} < 0.001" if v < 0.001
                                              else f"{name} = {v:.3f}")


def share(k: int, n: int) -> str:
    return f"{k}/{n} ({100 * k / n:.1f}%)" if n else "--"


# ── LaTeX ────────────────────────────────────────────────────────────────────

def tnum(v, digits: int = 3) -> str:
    if _missing(v):
        return "--"
    text = f"{abs(v):.{digits}f}"
    return f"$-${text}" if v < 0 and float(text) != 0 else text


def tci(low, high, digits: int = 3) -> str:
    return "--" if _missing(low) else f"[{tnum(low, digits)}, {tnum(high, digits)}]"


def tq(q) -> str:
    return "--" if _missing(q) else (r"$<$0.001" if q < 0.001 else f"{q:.3f}")


def tshare(k: int, n: int) -> str:
    return f"{k}/{n} ({100 * k / n:.1f}\\%)" if n else "--"


def tex(text: str) -> str:
    return text.replace("_", r"\_").replace("%", r"\%")


def _width_pt(cell: str) -> float:
    """A rough width of a cell in \\footnotesize Times (to warn about tables wider than the
    column; no TeX run)."""
    cell = re.sub(r"\\multicolumn\{\d+\}\{[^}]*\}", "", cell)
    cell = cell.replace("$-$", "\x00").replace(r"\,", " ").replace(r"\quad", "    ")
    cell = re.sub(r"\\[a-zA-Z]+", "", cell)
    cell = re.sub(r"[{}$\\^]", "", cell)
    narrow = {" ": 2.0, ".": 2.0, ",": 2.0, ":": 2.2, "[": 2.7, "]": 2.7, "(": 2.7, ")": 2.7,
              "/": 2.2, "*": 4.0, "%": 6.7, "\x00": 6.2}  # \x00: a math minus
    return sum(narrow.get(c, 4.0 if c.isdigit() else 5.6 if c.isupper() else 3.7) for c in cell)


def latex(name: str, caption: str, header: list[str], body: list[list[str] | str], align: str,
          *, wide: bool = False) -> str:
    """A booktabs table as a float (``table``, or ``table*`` when ``wide``), at \\footnotesize,
    as wide as the column (or the page). ``body``: rows of cells, or a string for a group
    heading spanning every column."""
    env, width = ("table*", r"\textwidth") if wide else ("table", r"\columnwidth")
    rows = []
    for row in body:
        if isinstance(row, str):
            rows.append(rf"\multicolumn{{{len(header)}}}{{@{{}}l}}{{\emph{{{row}}}}} \\")
        else:
            rows.append(" & ".join(row) + r" \\")
    cells = [header] + [row for row in body if not isinstance(row, str)]
    columns = [max(_width_pt(r[i]) for r in cells) for i in range(len(header))]
    estimate = sum(columns) + 6.0 * (len(header) - 1)
    if estimate > (TEXT_PT if wide else COLUMN_PT):
        print(f"  warning: {name}.tex may be wider than {width} (about {estimate:.0f} pt; "
              f"columns {[round(c) for c in columns]})")
    return "\n".join([
        f"% {name}.tex: written by paper_analysis.py with {name}.csv; do not edit by hand.",
        rf"\begin{{{env}}}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{tab:{name}}}",
        r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
        rf"\begin{{tabular*}}{{{width}}}{{@{{\extracolsep{{\fill}}}}{align}@{{}}}}",
        r"\toprule", " & ".join(header) + r" \\", r"\midrule", *rows, r"\bottomrule",
        r"\end{tabular*}", rf"\end{{{env}}}", ""])


def write_table(df: pd.DataFrame, tex_text: str | None, out: Path, name: str) -> list[Path]:
    paths = [out / f"{name}.csv"]
    df.to_csv(paths[0], index=False, lineterminator="\n")
    if tex_text is not None:
        paths.append(out / f"{name}.tex")
        paths[1].write_text(tex_text, encoding="utf-8", newline="\n")
    return paths


# ── RQ1: noise ───────────────────────────────────────────────────────────────

def _outcomes_by_repeat(e) -> dict[int, int | None]:
    return dict(zip(e.repeats, e.original_outcomes))


def _control_flips(envelopes) -> tuple[int, int]:
    """(flips, compared): per issue and control repeat r = 1..5, whether its outcome differs from
    the first run's (repeat 0); pairs where either run has no outcome are left out (the metrics
    engine's control flip rate)."""
    flips = compared = 0
    for e in envelopes:
        outcomes = _outcomes_by_repeat(e)
        first = outcomes.get(0)
        for r, o in outcomes.items():
            if r > 0 and o is not None and first is not None:
                compared += 1
                flips += o != first
    return flips, compared


def noise_summary(envelopes) -> dict:
    """The RQ1 measures of a set of envelopes (one backend, or one RQ3 condition)."""
    sims = [e.mean for e in envelopes if e.mean is not None]
    pairs = [v for e in envelopes for v in e.pairwise]
    widths = [e.width for e in envelopes if e.width is not None]
    flips, compared = _control_flips(envelopes)
    complete = [e for e in envelopes if len(e.scored_outcomes) == e.n_originals]
    q1, med, q3 = np.percentile(sims, [25, 50, 75]) if sims else (None,) * 3
    p1, pmed, p3 = np.percentile(pairs, [25, 50, 75]) if pairs else (None,) * 3
    return {
        "issues": len(envelopes),
        "runs_per_issue": fmean(e.n_originals for e in envelopes) if envelopes else None,
        "localized_6of6": sum(all(o == 1 for o in e.scored_outcomes) for e in complete),
        "localized_0of6": sum(all(o == 0 for o in e.scored_outcomes) for e in complete),
        "noisy_issues": sum(e.originals_disagree for e in envelopes),
        "repeat_similarity_median": med, "repeat_similarity_q1": q1, "repeat_similarity_q3": q3,
        "repeat_similarity_mean": fmean(sims) if sims else None,
        "pair_similarity_median": pmed, "pair_similarity_q1": p1, "pair_similarity_q3": p3,
        "pairs": len(pairs),
        "identical_pairs_bytes": sum(sum(e.identical) for e in envelopes),
        "identical_pairs_lines": sum(v == 1.0 for v in pairs),
        "control_flips": flips, "control_pairs": compared,
        "control_flip_rate": flips / compared if compared else None,
        "envelope_width_mean": fmean(widths) if widths else None,
        "envelope_width_median": float(np.median(widths)) if widths else None,
    }


def rq1(report: DeviationReport, out: Path, numbers: Numbers) -> tuple[list[Path], pd.DataFrame]:
    by = defaultdict(list)
    for e in report.envelopes:
        by[e.backend].append(e)
    noise = pd.DataFrame([{"backend": b, **noise_summary(by[b])} for b in BACKENDS if b in by])
    per_issue = pd.DataFrame([{
        "backend": e.backend, "issue_id": e.issue_id, "repository": repository(e.issue_id),
        "repeat_similarity": e.mean, "envelope_sd": e.sd, "envelope_width": e.width,
        "identical_pairs_bytes": sum(e.identical),
        "identical_pairs_lines": sum(v == 1.0 for v in e.pairwise), "pairs": e.n_pairs,
        "outcomes": " ".join("-" if o is None else str(o) for o in e.original_outcomes),
        "localized": sum(o == 1 for o in e.scored_outcomes), "noisy": int(e.originals_disagree),
    } for b in BACKENDS for e in sorted(by.get(b, []), key=lambda e: e.issue_id)])
    noisy = per_issue[per_issue.noisy == 1][
        ["backend", "issue_id", "outcomes", "localized", "repeat_similarity"]]

    rows = []
    a, b = CLAUDE, GPT
    ea = {e.issue_id: e for e in by.get(a, [])}
    eb = {e.issue_id: e for e in by.get(b, [])}
    shared = sorted(set(ea) & set(eb))
    for measure, attribute in (("repeat similarity", "mean"), ("envelope width", "width")):
        diffs = [getattr(ea[i], attribute) - getattr(eb[i], attribute) for i in shared
                 if getattr(ea[i], attribute) is not None and getattr(eb[i], attribute) is not None]
        rows.append({"comparison": f"{a} - {b}", "measure": measure,
                     f"median_{SHORT_BACKEND[a]}": float(np.median([getattr(ea[i], attribute)
                                                                    for i in shared])),
                     f"median_{SHORT_BACKEND[b]}": float(np.median([getattr(eb[i], attribute)
                                                                    for i in shared])),
                     **comparison_row("", paired_comparison(diffs))})
    backends = add_bh(pd.DataFrame(rows), ["p"])

    # LaTeX
    cols = [r for _i, r in noise.iterrows()]
    body = [
        ["Issues"] + [str(r.issues) for r in cols],
        ["Localized in all 6 runs"] + [str(r.localized_6of6) for r in cols],
        ["Localized in none of the 6"] + [str(r.localized_0of6) for r in cols],
        ["Noisy (the 6 runs disagree)"] + [str(r.noisy_issues) for r in cols],
        ["Repeat similarity, median"] + [tnum(r.repeat_similarity_median) for r in cols],
        [r"\quad IQR"] + [tci(r.repeat_similarity_q1, r.repeat_similarity_q3) for r in cols],
        ["Byte-identical repeat pairs"] + [tshare(r.identical_pairs_bytes, r.pairs) for r in cols],
        [r"Same changed lines (sim.\ = 1)"] + [tshare(r.identical_pairs_lines, r.pairs)
                                               for r in cols],
        ["Repeats flipping vs.\\ first run"] +[tshare(r.control_flips, r.control_pairs)
                                            for r in cols],
        ["Envelope width, mean"] + [tnum(r.envelope_width_mean) for r in cols],
    ]
    noise_tex = latex(
        "rq1_noise",
        "RQ1: run-to-run noise of the 6 runs of each original issue (50 issues per backend). "
        "Repeat similarity: per issue, the mean changed-line similarity of the 15 pairs of runs; "
        "median and interquartile range over the issues.",
        ["", *[SHORT_BACKEND[r.backend] for r in cols]], body, "l" + "r" * len(cols))
    noisy_tex = latex(
        "rq1_noisy_issues",
        "RQ1: the noisy issues, whose 6 original runs disagree on localization (1: the patch "
        "modifies the gold file; runs in order).",
        ["Backend", "Issue", "Runs 1--6", "Repeat sim."],
        [[SHORT_BACKEND[r.backend], tex(r.issue_id), r.outcomes, tnum(r.repeat_similarity)]
         for r in noisy.itertuples()], "llcr")
    backends_tex = latex(
        "rq1_backends",
        "RQ1: Claude minus GPT-4o, paired by issue (50 issues): mean difference, 95\\% bootstrap "
        "CI (issues resampled), Wilcoxon signed-rank $p$, BH $q$, rank-biserial $r$.",
        ["Measure", r"$\Delta$", "95\\% CI", "$p$", "$q$", "$r$"],
        [[r.measure.capitalize(), tnum(r.diff), tci(r.ci_low, r.ci_high), tq(r.p), tq(r.q),
          tnum(r.rank_biserial, 2)] for r in backends.itertuples()], "lrrrrr")

    written = write_table(noise, noise_tex, out, "rq1_noise")
    written += write_table(noisy, noisy_tex, out, "rq1_noisy_issues")
    written += write_table(backends, backends_tex, out, "rq1_backends")
    written += write_table(per_issue, None, out, "rq1_per_issue")

    for r in cols:
        b, src = SHORT_BACKEND[r.backend], f"rq1_noise.csv, row {r.backend}"
        numbers.add("RQ1: noise", f"{b}: issues", r.issues, "issues with 6 scored original runs",
                    src)
        numbers.add("RQ1: noise", f"{b}: localized in 6/6 runs", r.localized_6of6,
                    "issues whose 6 original runs all modify every gold file", f"{src}, localized_6of6")
        numbers.add("RQ1: noise", f"{b}: localized in 0/6 runs", r.localized_0of6,
                    "issues whose 6 original runs all fail to localize", f"{src}, localized_0of6")
        numbers.add("RQ1: noise", f"{b}: noisy issues", r.noisy_issues,
                    "issues whose original runs disagree on the outcome (1-5 of 6 localize)",
                    f"{src}, noisy_issues")
        numbers.add("RQ1: noise", f"{b}: repeat similarity, median [IQR]",
                    f"{f3(r.repeat_similarity_median)} [{f3(r.repeat_similarity_q1)}, "
                    f"{f3(r.repeat_similarity_q3)}]",
                    "per issue, the mean changed-line similarity of its 15 pairs of original runs; "
                    "median and quartiles over the issues (numpy.percentile, linear)",
                    f"{src}, repeat_similarity_median/_q1/_q3")
        numbers.add("RQ1: noise", f"{b}: repeat similarity, mean over issues",
                    f3(r.repeat_similarity_mean), "mean of the per-issue repeat similarities",
                    f"{src}, repeat_similarity_mean")
        numbers.add("RQ1: noise", f"{b}: pair similarity, median [IQR]",
                    f"{f3(r.pair_similarity_median)} [{f3(r.pair_similarity_q1)}, "
                    f"{f3(r.pair_similarity_q3)}]",
                    f"over all {r.pairs} pairs of original runs (15 per issue)",
                    f"{src}, pair_similarity_*")
        numbers.add("RQ1: noise", f"{b}: byte-identical repeat pairs",
                    share(r.identical_pairs_bytes, r.pairs),
                    "pairs of original runs whose stored patches are byte-identical "
                    "(design section 9, RQ1)", f"{src}, identical_pairs_bytes")
        numbers.add("RQ1: noise", f"{b}: repeat pairs with the same changed lines",
                    share(r.identical_pairs_lines, r.pairs),
                    "pairs of original runs with changed-line similarity 1 (the same +/- lines; "
                    "context lines, hunk headers and text around the diff may differ)",
                    f"{src}, identical_pairs_lines")
        numbers.add("RQ1: noise", f"{b}: control flips against the first run",
                    share(r.control_flips, r.control_pairs),
                    "per issue and repeat r = 1..5, the repeat's outcome differs from the first "
                    "run's (the metrics engine's control flip rate)", f"{src}, control_flips")
        numbers.add("RQ1: noise", f"{b}: envelope width, mean", f3(r.envelope_width_mean),
                    "per issue, max minus min of the 15 pair similarities; mean over issues",
                    f"{src}, envelope_width_mean")
    for r in noisy.itertuples():
        numbers.add("RQ1: noise", f"{SHORT_BACKEND[r.backend]}: noisy issue {r.issue_id}",
                    f"outcomes {r.outcomes}; repeat similarity {f3(r.repeat_similarity)}",
                    "the 6 original runs' outcomes in run order (1: localized)",
                    "rq1_noisy_issues.csv")
    for r in backends.itertuples():
        numbers.add("RQ1: noise", f"Claude minus GPT-4o: {r.measure}",
                    f"{f3(r.diff)} {fci(r.ci_low, r.ci_high)}, Wilcoxon {eq('p', r.p)}, "
                    f"{eq('q', r.q)}, r = {f3(r.rank_biserial, 2)}, n = {r.n}",
                    "per issue, Claude's value minus GPT-4o's; mean, 95% bootstrap CI (issues "
                    "resampled), Wilcoxon signed-rank (zeros dropped), BH over the 2 rows, "
                    "matched-pairs rank-biserial",
                    f"rq1_backends.csv, row {r.measure}")
    return written, per_issue


# ── RQ2: what survives the noise floor ───────────────────────────────────────

def _cells(report: DeviationReport):
    return sorted(report.summaries,
                  key=lambda s: (BACKENDS.index(s.backend) if s.backend in BACKENDS else 9,
                                 ORDER.index(s.transformation) if s.transformation in ORDER else 9))


def rq2_envelope(report: DeviationReport) -> pd.DataFrame:
    """Per backend and transformation: the envelope method (and the single-repeat method with the
    first repeat, its partner in the RQ2 family) with the variant-level counts."""
    single = {(s.backend, s.transformation): s.comparison for s in report.single_repeat}
    rows = []
    for s in _cells(report):
        rows.append({
            "backend": s.backend, "transformation": s.transformation, "variants": s.n,
            **comparison_row("envelope_", s.envelope),
            **comparison_row("single_r1_", single.get((s.backend, s.transformation))
                             or paired_comparison([])),
            "outside_envelope": s.deviating_z, "outside_share": s.deviation_rate_z,
            "outside_rank": s.deviating_rank, "binary_deviation": s.deviating,
            "gains": s.gains, "losses": s.losses, "flips_undetermined": s.undetermined,
        })
    return add_bh(pd.DataFrame(rows), ["envelope_p", "single_r1_p"])


def rq2_stability(report: DeviationReport) -> pd.DataFrame:
    envelope = {(s.backend, s.transformation): s.envelope for s in report.summaries}
    order = {(s.backend, s.transformation): k for k, s in enumerate(_cells(report))}
    rows = []
    for c in sorted(report.conclusion_stability,
                    key=lambda c: order.get((c.backend, c.transformation), 99)):
        row = {"backend": c.backend, "transformation": c.transformation}
        for r, comp in sorted(c.by_repeat.items()):
            row.update({f"r{r}_n": comp.n, f"r{r}_diff": comp.mean_difference,
                        f"r{r}_ci_low": comp.ci_low, f"r{r}_ci_high": comp.ci_high,
                        f"r{r}_p": comp.wilcoxon_p, f"r{r}_verdict": comp.conclusion})
        e = envelope.get((c.backend, c.transformation))
        row.update({"repeats": len(c.by_repeat),
                    "excludes_zero_with": sum(1 for v in c.excludes_zero.values() if v),
                    "verdict_changes": "yes" if c.changes else "no",
                    "envelope_diff": e.mean_difference if e else None,
                    "envelope_verdict": e.conclusion if e else None})
        rows.append(row)
    return pd.DataFrame(rows)


def per_issue(report: DeviationReport, measure: str) -> dict[tuple[str, str], dict[str, float]]:
    """(backend, transformation) -> issue -> the measure (the mean of its variants' values)."""
    values: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for v in report.variants:
        if measure == DEVIATION:
            if v.envelope_mean is None:
                continue
            value = v.similarity - v.envelope_mean
        else:
            value = float(v.deviates)
        values[(v.backend, v.transformation)][v.issue_id].append(value)
    return {key: {i: fmean(x) for i, x in sorted(by_issue.items())}
            for key, by_issue in values.items()}


def repository_ci(
    differences: dict[str, float], *, resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED, level: float = CI_LEVEL,
) -> tuple[float | None, float | None]:
    """Percentile bootstrap CI of the mean per-issue difference, resampling repositories (each
    with all its issues); None when the issues come from fewer than 2 repositories."""
    clusters: dict[str, list[float]] = defaultdict(list)
    for issue_id, d in differences.items():
        clusters[repository(issue_id)].append(d)
    if len(clusters) < 2:
        return None, None
    sums = np.array([sum(v) for v in clusters.values()])
    sizes = np.array([len(v) for v in clusters.values()])
    draws = np.random.default_rng(seed).integers(0, len(sums), size=(resamples, len(sums)))
    means = sums[draws].sum(axis=1) / sizes[draws].sum(axis=1)
    tail = 100 * (1 - level) / 2
    low, high = np.percentile(means, [tail, 100 - tail])
    return float(low), float(high)


def reading(measure: str, low: float | None, high: float | None) -> str | None:
    """What the CI says, in words: does T deviate more or less than typo injection?"""
    if low is None or high is None:
        return None
    if low <= 0 <= high:
        return "no difference"
    more = high < 0 if measure == DEVIATION else low > 0
    return "deviates more than typo" if more else "deviates less than typo"


def typo_reference(report: DeviationReport) -> pd.DataFrame:
    """POST-HOC, exploratory (docs/deviations.md, section 2): per backend and transformation T
    other than typo injection, paired by issue (the issues where both variants were scored in
    that backend), T's measure minus typo injection's; BH over the table (a family of its own)."""
    rows = []
    for measure in MEASURES:
        values = per_issue(report, measure)
        for backend in sorted({b for b, _t in values}):
            reference = values.get((backend, REFERENCE), {})
            for t in sorted(t for b, t in values if b == backend and t != REFERENCE):
                own = values[(backend, t)]
                shared = sorted(set(own) & set(reference))
                differences = {i: own[i] - reference[i] for i in shared}
                c = paired_comparison(list(differences.values()))
                repo_low, repo_high = repository_ci(differences)
                rows.append({
                    "backend": backend, "transformation": t, "measure": measure, "n": c.n,
                    "transformation_mean": fmean(own[i] for i in shared) if shared else None,
                    "typo_mean": fmean(reference[i] for i in shared) if shared else None,
                    "diff": c.mean_difference, "ci_low": c.ci_low, "ci_high": c.ci_high,
                    "repo_ci_low": repo_low, "repo_ci_high": repo_high,
                    "p": c.wilcoxon_p, "rank_biserial": c.rank_biserial,
                    "reading": reading(measure, c.ci_low, c.ci_high),
                    "repo_reading": reading(measure, repo_low, repo_high),
                    "analysis": ANALYSIS,
                })
    df = pd.DataFrame(rows)
    return add_bh(df, ["p"]) if not df.empty else df


def _ordered(df: pd.DataFrame) -> pd.DataFrame:
    key = df.backend.map(lambda b: BACKENDS.index(b) if b in BACKENDS else 9) * 10 + \
        df.transformation.map(lambda t: ORDER.index(t) if t in ORDER else 9)
    return df.assign(_k=key).sort_values("_k", kind="stable").drop(columns="_k")


def rq2(report: DeviationReport, out: Path, numbers: Numbers) -> tuple[list[Path], dict]:
    env = rq2_envelope(report)
    stab = rq2_stability(report)
    post = typo_reference(report)
    post_paper = _ordered(post[post.measure == DEVIATION])

    def grouped(df, row):
        body = []
        for backend in BACKENDS:
            part = df[df.backend == backend]
            if len(part):
                body.append(backend)
                body += [row(r) for r in part.itertuples()]
        return body

    env_tex = latex(
        "rq2_envelope",
        "RQ2: deviation of each transformation from the noise envelope, paired by issue. "
        r"$\Delta$: mean over issues of the variant's mean similarity to the 6 original runs "
        r"minus the envelope mean (the mean of their 15 pair similarities); 95\% bootstrap CI "
        r"(issues resampled); BH $q$ over the RQ2 family (this method and the single-repeat "
        r"method with the first repeat). Outside: variants with $z<-2$. G/L: outcome flips on "
        r"non-noisy issues, gains/losses.",
        ["Transformation", "$n$", r"$\Delta$", "95\\% CI", "$q$", "Outside", "G/L"],
        grouped(env, lambda r: [TEX_LABELS[r.transformation], str(r.envelope_n),
                                tnum(r.envelope_diff), tci(r.envelope_ci_low, r.envelope_ci_high),
                                tq(r.envelope_q), tshare(r.outside_envelope, r.variants)
                                .split(" ")[0], f"{r.gains}/{r.losses}"]),
        "lrrrrrr")

    def mark(r, k):
        v = getattr(r, f"r{k}_diff")
        low, high = getattr(r, f"r{k}_ci_low"), getattr(r, f"r{k}_ci_high")
        star = "$^{*}$" if not _missing(low) and (high < 0 or low > 0) else r"\phantom{$^{*}$}"
        return tnum(v) + star

    def changes(r):
        return TEX_LABELS[r.transformation] + ("$^\\dagger$" if r.verdict_changes == "yes" else "")

    reps = sorted({int(c[1:-5]) for c in stab.columns if re.fullmatch(r"r\d+_diff", c)})
    stab_tex = latex(
        "rq2_conclusion_stability",
        "RQ2: conclusion stability. The single-repeat estimate (variant vs.\\ the first run, "
        "minus repeat $r$ vs.\\ the first run; mean over issues) with each of the 5 repeats "
        "$r=1..5$ of the first run. "
        r"$^{*}$: the 95\% CI excludes zero. $^\dagger$: the verdict changes with the repeat used.",
        ["Transformation", *[f"$r={k}$" for k in reps]],
        grouped(stab, lambda r: [changes(r), *[mark(r, k) for k in reps]]),
        "l" + "r" * len(reps))
    stab_ci_tex = latex(
        "rq2_conclusion_stability_ci",
        "RQ2: conclusion stability, the single-repeat estimate with each of the 5 repeats $r$ of "
        r"the first run and its 95\% CI (issues resampled). $^\dagger$: whether the CI excludes "
        "zero changes with the repeat used.",
        ["Transformation", *[f"$r={k}$" for k in reps]],
        grouped(stab, lambda r: [changes(r), *[
            f"{tnum(getattr(r, f'r{k}_diff'))} "
            f"{tci(getattr(r, f'r{k}_ci_low'), getattr(r, f'r{k}_ci_high'), 2)}" for k in reps]]),
        "l" + "r" * len(reps), wide=True)
    post_tex = latex(
        "rq2_posthoc_typo",
        "POST-HOC (exploratory, not pre-registered): each transformation's envelope-method "
        "deviation minus typo injection's, paired by issue (issues where both variants were "
        r"scored). Negative: further from the original runs than typo injection. 95\% bootstrap "
        r"CI (issues resampled); BH $q$ over the post-hoc family (20 tests, with the binary "
        "measure).",
        ["Post-hoc", "$n$", r"$\Delta$ vs.\ typo", "95\\% CI", "$q$"],
        grouped(post_paper, lambda r: [TEX_LABELS[r.transformation], str(r.n), tnum(r.diff),
                                       tci(r.ci_low, r.ci_high), tq(r.q)]),
        "lrrrr")

    written = write_table(env, env_tex, out, "rq2_envelope")
    written += write_table(stab, stab_tex, out, "rq2_conclusion_stability")
    (out / "rq2_conclusion_stability_ci.tex").write_text(stab_ci_tex, encoding="utf-8",
                                                              newline="\n")
    written.append(out / "rq2_conclusion_stability_ci.tex")
    written += write_table(post, post_tex, out, "rq2_posthoc_typo")

    sec = "RQ2: wording effects beyond the noise floor"
    for r in env.itertuples():
        b, t = SHORT_BACKEND[r.backend], LABELS[r.transformation]
        src = f"rq2_envelope.csv, row {r.backend} / {r.transformation}"
        numbers.add(sec, f"{b}, {t}: envelope deviation",
                    f"{f3(r.envelope_diff)} {fci(r.envelope_ci_low, r.envelope_ci_high)}, "
                    f"{eq('p', r.envelope_p)}, {eq('q', r.envelope_q)}, r = "
                    f"{f3(r.envelope_rank_biserial, 2)}, n = {r.envelope_n} issues",
                    "per issue, the variant's mean similarity to the 6 original runs minus the "
                    "envelope mean; mean over issues, 95% bootstrap CI (issues resampled), "
                    "Wilcoxon, BH over the RQ2 family (24 tests)", f"{src}, envelope_*")
        numbers.add(sec, f"{b}, {t}: outside the envelope (z < -2)",
                    share(r.outside_envelope, r.variants),
                    "variants whose z = (s - envelope mean) / max(sd, 0.05) is below -2",
                    f"{src}, outside_envelope")
        numbers.add(sec, f"{b}, {t}: gains / losses", f"{r.gains} / {r.losses}",
                    "outcome flips on non-noisy issues (all 6 originals agree): gain = originals "
                    "fail, variant localizes; loss = the reverse; flips on noisy issues "
                    f"({r.flips_undetermined} variants) are not counted", f"{src}, gains, losses")
        numbers.add(sec, f"{b}, {t}: single-repeat method (repeat 1)",
                    f"{f3(r.single_r1_diff)} {fci(r.single_r1_ci_low, r.single_r1_ci_high)}, "
                    f"{eq('q', r.single_r1_q)}",
                    "per issue, the variant's similarity to the first run minus repeat 1's; "
                    "as above", f"{src}, single_r1_*")
    changing = stab[stab.verdict_changes == "yes"]
    numbers.add(sec, "Cells whose single-repeat verdict changes with the repeat",
                f"{len(changing)}/{len(stab)}",
                "backend x transformation cells where the 95% CI of the single-repeat estimate "
                "excludes zero with some repeats and not with others",
                "rq2_conclusion_stability.csv, verdict_changes")
    for r in stab.itertuples():
        values = ", ".join(f"{f3(getattr(r, f'r{k}_diff'))} "
                           f"{fci(getattr(r, f'r{k}_ci_low'), getattr(r, f'r{k}_ci_high'))}"
                           for k in reps)
        numbers.add(sec, f"{SHORT_BACKEND[r.backend]}, {LABELS[r.transformation]}: single-repeat "
                    "estimates r = 1..5",
                    f"{values}; CI excludes zero with {r.excludes_zero_with}/{r.repeats}; "
                    f"changes: {r.verdict_changes}",
                    "the single-repeat estimate with each repeat r = 1..5 of the first run (the "
                    "variant vs the first run, minus repeat r vs the first run)",
                    f"rq2_conclusion_stability.csv, row {r.backend} / {r.transformation}")
    for r in _ordered(post).itertuples():
        numbers.add("RQ2 post-hoc (exploratory): each transformation minus typo injection",
                    f"{SHORT_BACKEND[r.backend]}, {LABELS[r.transformation]}: {r.measure}",
                    f"{f3(r.diff)} {fci(r.ci_low, r.ci_high)}, {eq('q', r.q)}, n = {r.n}; "
                    f"repository CI {fci(r.repo_ci_low, r.repo_ci_high)}",
                    "per issue, the transformation's value minus typo injection's (deviation: "
                    "s_T - s_typo; binary deviation: 1/0 difference); 95% bootstrap CI (issues "
                    "resampled; repository CI: repositories resampled); BH over the 20 rows",
                    f"rq2_posthoc_typo.csv, row {r.backend} / {r.transformation} / {r.measure}")
    return written, {"stability": stab, "envelope": env}


# ── RQ3: the file list ───────────────────────────────────────────────────────

def _per_issue_runs(recs: list[RunRecord], backend: str, gold: dict[str, set[str]]):
    """Per issue: (localization rate, share of patches touching a file outside the gold set)
    over every scored run of the backend (originals, repeats and variants)."""
    by_issue = defaultdict(list)
    for r in recs:
        if r.backend == backend:
            by_issue[r.issue_id].append(r)
    out = {}
    for issue_id, runs in by_issue.items():
        outcomes = [r.outcome for r in runs if r.outcome is not None]
        outside = [bool(set(patch_files(extract_final_patch(r.patch))) - gold.get(issue_id, set()))
                   for r in runs if r.patch]
        out[issue_id] = (fmean(outcomes) if outcomes else None,
                         fmean(outside) if outside else None)
    return out


def _measure_name(measure: str) -> str:
    t = measure.removeprefix("deviation: ")
    return f"deviation, {LABELS[t]}" if t != measure else measure


def _deviation(report: DeviationReport, t: str, ids: set[str], backend: str) -> dict[str, float]:
    return {v.issue_id: v.similarity - v.envelope_mean for v in report.variants
            if v.transformation == t and v.issue_id in ids and v.envelope_mean is not None
            and v.backend == backend}


def rq3(core: DeviationReport, filelist: DeviationReport, core_recs: list[RunRecord],
        fl_recs: list[RunRecord], issues: pd.DataFrame, runs: pd.DataFrame, out: Path,
        numbers: Numbers) -> tuple[list[Path], dict]:
    """Claude, the RQ3 issues: file list minus issue only, paired by issue (exploratory; the
    measures and BH family of the study design), and descriptive rows."""
    ids = set(issues.issue_id[issues.rq3 == 1])
    gold = {r.issue_id: set(filter(None, r.gold_files.split(";"))) for r in issues.itertuples()}
    env = {CONDITIONS[0]: {e.issue_id: e for e in core.envelopes
                           if e.backend == CLAUDE and e.issue_id in ids},
           CONDITIONS[1]: {e.issue_id: e for e in filelist.envelopes if e.issue_id in ids}}
    runs_core = _per_issue_runs(core_recs, CLAUDE, gold)
    runs_fl = _per_issue_runs(fl_recs, CLAUDE, gold)
    measures = {
        "repeat similarity": ({i: e.mean for i, e in env[CONDITIONS[0]].items()},
                              {i: e.mean for i, e in env[CONDITIONS[1]].items()}),
        "envelope width": ({i: e.width for i, e in env[CONDITIONS[0]].items()},
                           {i: e.width for i, e in env[CONDITIONS[1]].items()}),
        "localization rate": ({i: v[0] for i, v in runs_core.items()},
                              {i: v[0] for i, v in runs_fl.items()}),
        "patches outside the gold files": ({i: v[1] for i, v in runs_core.items()},
                                           {i: v[1] for i, v in runs_fl.items()}),
    }
    for t in ORDER:
        measures[f"deviation: {t}"] = (_deviation(core, t, ids, CLAUDE),
                                       _deviation(filelist, t, ids, CLAUDE))
    rows, values = [], {}
    for measure, (c, f) in measures.items():
        shared = sorted(i for i in ids if not _missing(c.get(i)) and not _missing(f.get(i)))
        values[measure] = ({i: c[i] for i in shared}, {i: f[i] for i in shared})
        comparison = paired_comparison([f[i] - c[i] for i in shared])
        rows.append({"kind": "paired", "measure": measure,
                     "issue_only": fmean(c[i] for i in shared) if shared else None,
                     "file_list": fmean(f[i] for i in shared) if shared else None,
                     **comparison_row("", comparison), "analysis": "exploratory"})
    table = add_bh(pd.DataFrame(rows), ["p"])

    summaries = {cond: noise_summary(list(env[cond].values())) for cond in CONDITIONS}
    called = runs[(runs.backend == CLAUDE) & runs.issue_id.isin(ids) & (runs.token_input != "")]
    tokens = {cond: called[called.condition == cond].token_input.astype(int) for cond in CONDITIONS}
    descriptive = [
        ("issues", "issues", None),
        ("byte-identical repeat pairs", "identical_pairs_bytes", "pairs"),
        ("repeat pairs with the same changed lines", "identical_pairs_lines", "pairs"),
        ("issues localized in 6/6 runs", "localized_6of6", None),
        ("issues localized in 0/6 runs", "localized_0of6", None),
        ("noisy issues", "noisy_issues", None),
        ("repeats flipping vs the first run", "control_flips", "control_pairs"),
    ]
    extra = []
    for measure, key, of in descriptive:
        extra.append({"kind": "descriptive", "measure": measure,
                      "issue_only": summaries[CONDITIONS[0]][key],
                      "file_list": summaries[CONDITIONS[1]][key],
                      "n": summaries[CONDITIONS[0]][of] if of else None})
    for measure, fn in (("input tokens per run, mean", "mean"), ("input tokens per run, max", "max")):
        extra.append({"kind": "descriptive", "measure": measure,
                      "issue_only": float(getattr(tokens[CONDITIONS[0]], fn)()),
                      "file_list": float(getattr(tokens[CONDITIONS[1]], fn)()),
                      "n": len(tokens[CONDITIONS[0]])})
    table = pd.concat([table, pd.DataFrame(extra)], ignore_index=True)
    table.loc[table.kind == "descriptive", "analysis"] = "descriptive"
    table["n"] = table["n"].astype("Int64")

    label = {"repeat similarity": "Repeat similarity", "envelope width": "Envelope width",
             "localization rate": "Localization rate",
             "patches outside the gold files": "Outside gold files"}
    body = []
    for r in table[table.kind == "paired"].itertuples():
        t = r.measure.removeprefix("deviation: ")
        name = rf"\enspace {TEX_LABELS[t]}" if r.measure.startswith("deviation: ") else \
            label[r.measure]
        if r.measure == f"deviation: {ORDER[0]}":
            body.append("Deviation from the envelope")
        body.append([name, tnum(r.issue_only), tnum(r.file_list), tnum(r.diff),
                     tci(r.ci_low, r.ci_high), tq(r.q)])
    d = {r.measure: r for r in table[table.kind == "descriptive"].itertuples()}

    def count(m):
        r = d[m]
        if _missing(r.n):
            return [str(int(r.issue_only)), str(int(r.file_list))]
        return [f"{int(r.issue_only)}/{int(r.n)}", f"{int(r.file_list)}/{int(r.n)}"]
    body.append("Descriptive")
    body.append(["Identical (bytes)", *count("byte-identical repeat pairs"), "", "", ""])
    body.append(["Identical (lines)", *count("repeat pairs with the same changed lines"),
                 "", "", ""])
    body.append(["Localized 6/6 / 0/6", f"{int(d['issues localized in 6/6 runs'].issue_only)} / "
                 f"{int(d['issues localized in 0/6 runs'].issue_only)}",
                 f"{int(d['issues localized in 6/6 runs'].file_list)} / "
                 f"{int(d['issues localized in 0/6 runs'].file_list)}", "", "", ""])
    body.append(["Noisy issues", *count("noisy issues"), "", "", ""])
    body.append(["Input tokens, mean", f"{d['input tokens per run, mean'].issue_only:,.0f}",
                 f"{d['input tokens per run, mean'].file_list:,.0f}", "", "", ""])
    body.append(["Input tokens, max", f"{d['input tokens per run, max'].issue_only:,.0f}",
                 f"{d['input tokens per run, max'].file_list:,.0f}", "", "", ""])
    rq3_tex = latex(
        "rq3_filelist",
        f"RQ3 (exploratory): Claude on the {len(ids)} RQ3 issues, the issue only (no list) vs.\\ "
        r"with the repository's file list in the prompt (list), paired by issue. $\Delta$: list "
        r"minus no list; 95\% bootstrap CI (issues resampled); BH $q$ over the RQ3 family. Localization "
        "rate and patches outside the gold files: per issue, over all its scored runs. "
        "Identical: pairs of the 6 original runs (15 per issue) with byte-identical patches, or "
        "with the same changed lines (similarity 1).",
        ["Measure", "No list", "List", r"$\Delta$", "95\\% CI", "$q$"],
        body,
        "lrrrrr")
    written = write_table(table, rq3_tex, out, "rq3_filelist")

    sec = "RQ3: the file list (Claude, exploratory)"
    for r in table.itertuples():
        src = f"rq3_filelist.csv, row {r.measure}"
        if r.kind == "paired":
            numbers.add(sec, _measure_name(r.measure),
                        f"issue only {f3(r.issue_only)}, file list {f3(r.file_list)}; difference "
                        f"{f3(r.diff)} {fci(r.ci_low, r.ci_high)}, {eq('p', r.p)}, "
                        f"{eq('q', r.q)}, n = {r.n} issues",
                        "paired by issue: file list minus issue only; mean, 95% bootstrap CI "
                        "(issues resampled), Wilcoxon, BH over the 10 paired rows", src)
        else:
            of = "" if _missing(r.n) or r.measure.startswith("input") else f" of {int(r.n)}"
            numbers.add(sec, r.measure, f"issue only {r.issue_only:,.0f}{of}, file list "
                        f"{r.file_list:,.0f}{of}" if not r.measure.startswith("input") else
                        f"issue only {r.issue_only:,.1f}, file list {r.file_list:,.1f}",
                        "Claude, the RQ3 issues; tokens over the runs that called the model "
                        f"({int(r.n)} per condition)" if r.measure.startswith("input") else
                        "Claude, the RQ3 issues, the 6 original runs per issue", src)
    return written, values


# ── Figures ──────────────────────────────────────────────────────────────────

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update(PAPER_RC)
    return plt


def _style_axes(ax, grid_axis: str = "x") -> None:
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=TEXT, length=2)


def _save(fig, path: Path) -> Path:
    fig.savefig(path, metadata={"CreationDate": None, "ModDate": None})  # same bytes every run
    _plt().close(fig)
    return path


def fig1_repeat_similarity(per_issue_df: pd.DataFrame, path: Path) -> Path:
    """Per backend, the issues sorted by repeat similarity (the mean of their 15 pairs); noisy
    issues ringed."""
    plt = _plt()
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(COLUMN_IN, 2.1))
    _style_axes(ax, "y")
    handles = []
    for backend in BACKENDS:
        st = STYLE[backend]
        rows = per_issue_df[per_issue_df.backend == backend].sort_values(
            ["repeat_similarity", "issue_id"])
        x = np.arange(1, len(rows) + 1)
        face = st["color"] if st["filled"] else "white"
        ax.plot(x, rows.repeat_similarity, color=st["color"], ls=st["ls"], lw=0.8,
                marker=st["marker"], ms=3.2, mfc=face, mec=st["color"], mew=0.8, zorder=3)
        noisy = rows.noisy.to_numpy() == 1
        ax.scatter(x[noisy], rows.repeat_similarity[noisy], s=46, facecolors="none",
                   edgecolors=TEXT, linewidths=0.8, zorder=4)
        handles.append(Line2D([], [], color=st["color"], ls=st["ls"], lw=0.8,
                              marker=st["marker"], ms=3.2, mfc=face, mec=st["color"],
                              label=backend))
    handles.append(Line2D([], [], ls="none", marker="o", ms=6.5, mfc="none", mec=TEXT, mew=0.8,
                          label="noisy issue (runs disagree on localization)"))
    ax.set_xlim(0, len(per_issue_df) / len(BACKENDS) + 1)
    ax.set_ylim(0, 1.04)
    ax.set_xlabel("Issues, sorted by repeat similarity (per backend)")
    ax.set_ylabel("Repeat similarity\n(mean of the 15 pairs)")
    ax.legend(handles=handles, frameon=False, loc="lower right", handlelength=2.2,
              borderaxespad=0.2)
    fig.tight_layout(pad=0.3)
    return _save(fig, path)


def _lighten(color: str, amount: float) -> tuple[float, float, float]:
    from matplotlib.colors import to_rgb
    r, g, b = to_rgb(color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


def fig2_conclusion_stability(stab: pd.DataFrame, env: pd.DataFrame, path: Path) -> Path:
    """Per backend (one panel each) and transformation: the 5 single-repeat estimates with their
    95% CIs (small marks; filled when the CI excludes zero) and the envelope estimate (large
    mark), against zero."""
    plt = _plt()
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(1, len(BACKENDS), figsize=(TEXT_IN, 2.35), sharex=True, sharey=True)
    reps = sorted({int(c[1:-5]) for c in stab.columns if re.fullmatch(r"r\d+_diff", c)})
    order = [t for t in ORDER if t in set(stab.transformation)]
    # shared x range, with room right of the data for the "changes (k/5)" notes in each panel
    xmin = min([stab[f"r{k}_ci_low"].min() for k in reps] + [env.envelope_ci_low.min()]) - 0.01
    xmax = max([stab[f"r{k}_ci_high"].max() for k in reps]
               + [env.envelope_ci_high.max(), 0.0]) + 0.09
    for ax, backend in zip(axes, BACKENDS):
        _style_axes(ax)
        st = STYLE[backend]
        light = _lighten(st["color"], 0.35)
        s_rows = stab[stab.backend == backend].set_index("transformation")
        e_rows = env[env.backend == backend].set_index("transformation")
        for y, t in enumerate(order):
            if t not in s_rows.index:
                continue
            r = s_rows.loc[t]
            yy = len(order) - 1 - y  # the first transformation on top
            for j, k in enumerate(reps):
                low, high, v = r[f"r{k}_ci_low"], r[f"r{k}_ci_high"], r[f"r{k}_diff"]
                if _missing(v):
                    continue
                dy = yy + 0.36 - 0.1 * j
                excludes = high < 0 or low > 0
                ax.plot([low, high], [dy, dy], color=light, lw=0.7, zorder=2)
                ax.plot([v], [dy], st["marker"], ms=2.8, color=light,
                        mfc=light if excludes else "white", mew=0.7, zorder=3)
            e = e_rows.loc[t]
            ax.plot([e.envelope_ci_low, e.envelope_ci_high], [yy - 0.22] * 2, color=st["color"],
                    lw=1.6, zorder=2)
            ax.plot([e.envelope_diff], [yy - 0.22], st["marker"], ms=5.5, color=st["color"],
                    mfc=st["color"], mec=TEXT, mew=0.5, zorder=3)
            if r.verdict_changes == "yes":  # inside the panel, right of zero, on the envelope line
                ax.text(xmax - 0.004, yy - 0.22,
                        f"changes ({r.excludes_zero_with}/{r.repeats})", ha="right",
                        va="center", fontsize=6, color=MUTED, style="italic", zorder=4)
        ax.axvline(0, color=TEXT, lw=0.6, ls="--", zorder=1)
        ax.set_title(backend, fontsize=7.5, loc="left", color=TEXT)
        ax.set_ylim(-0.6, len(order) - 0.45)
        ax.set_xlim(xmin, xmax)
        ax.set_xticks(np.arange(np.ceil(xmin / 0.05) * 0.05, 0.001, 0.05))  # none in the notes
    axes[0].set_yticks(range(len(order)), [LABELS[t] for t in reversed(order)])
    fig.supxlabel("Difference to the repeat baseline (changed-line similarity)", fontsize=7.5,
                  y=0.02)
    gray = _lighten(MUTED, 0.2)
    handles = [
        Line2D([], [], color=gray, lw=0.7, marker="o", ms=2.8, mfc=gray, mec=gray,
               label="single repeat r = 1–5, 95% CI excludes 0"),
        Line2D([], [], color=gray, lw=0.7, marker="o", ms=2.8, mfc="white", mec=gray,
               label="single repeat, CI includes 0"),
        Line2D([], [], color=TEXT, lw=1.6, marker="o", ms=5.5, mfc=MUTED, mec=TEXT, mew=0.5,
               label="envelope (all 6 runs)"),
    ]
    fig.legend(handles=handles, frameon=False, loc="upper center", ncol=3,
               bbox_to_anchor=(0.55, 1.0), handlelength=2.0)
    # fixed margins (tight_layout misjudges the → label): the row labels on the left
    fig.subplots_adjust(left=0.17, right=0.99, bottom=0.19, top=0.83, wspace=0.06)
    return _save(fig, path)


def fig3_rq3_deviation(values: dict, table: pd.DataFrame, path: Path) -> Path:
    """Per transformation, the mean envelope-method deviation with and without the file list
    (Claude, the RQ3 issues where both were scored) with 95% CIs, and the paired difference."""
    plt = _plt()
    from matplotlib.lines import Line2D
    st = STYLE[CLAUDE]
    order = [t for t in ORDER if f"deviation: {t}" in values]
    fig, ax = plt.subplots(figsize=(COLUMN_IN, 2.15))
    _style_axes(ax)
    paired = table[table.kind == "paired"].set_index("measure")
    for y, t in enumerate(order):
        yy = len(order) - 1 - y
        without, with_list = values[f"deviation: {t}"]
        means = []
        for dy, data, filled in ((0.14, without, False), (-0.14, with_list, True)):
            v = list(data.values())
            low, high = bootstrap_mean_ci(v)
            m = fmean(v)
            means.append((m, yy + dy))
            ax.plot([low, high], [yy + dy] * 2, color=st["color"], lw=0.9 if not filled else 1.4)
            ax.plot([m], [yy + dy], "o", ms=4, color=st["color"],
                    mfc=st["color"] if filled else "white", mew=0.9, zorder=3)
        ax.plot([means[0][0], means[1][0]], [means[0][1], means[1][1]], color=MUTED, lw=0.5,
                zorder=2)
        r = paired.loc[f"deviation: {t}"]
        ax.text(1.04, yy, f"{r['diff']:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}]"
                .replace("-", "−"), transform=ax.get_yaxis_transform(), ha="left",
                va="center", fontsize=6.3, color=TEXT)
    ax.text(1.04, len(order) - 0.4, "with − without [95% CI]", transform=ax.get_yaxis_transform(),
            ha="left", va="center", fontsize=6.3, color=MUTED, style="italic")
    ax.axvline(0, color=TEXT, lw=0.6, ls="--", zorder=1)
    ax.set_yticks(range(len(order)), [LABELS[t] for t in reversed(order)])
    ax.set_ylim(-0.6, len(order) - 0.25)
    ax.set_xlabel("Deviation (s − envelope mean)")
    handles = [Line2D([], [], color=st["color"], lw=0.9, marker="o", ms=4, mfc="white",
                      label="issue only"),
               Line2D([], [], color=st["color"], lw=1.4, marker="o", ms=4, mfc=st["color"],
                      label="with the file list")]
    fig.legend(handles=handles, frameon=False, loc="upper left", bbox_to_anchor=(0.30, 1.0),
               ncol=2, handlelength=1.8)
    # fixed margins (tight_layout misjudges the → label): labels left, differences right
    fig.subplots_adjust(left=0.315, right=0.655, bottom=0.19, top=0.87)
    return _save(fig, path)


# ── The pipeline ─────────────────────────────────────────────────────────────

def analyze(runs: pd.DataFrame, issues: pd.DataFrame, out: Path,
            exclude_no_patch: bool = False) -> tuple[list[Path], Numbers]:
    out.mkdir(parents=True, exist_ok=True)
    numbers = Numbers()
    core_recs = records(runs, CONDITIONS[0], exclude_no_patch)
    fl_recs = records(runs, CONDITIONS[1], exclude_no_patch)
    core = deviation_report(core_recs)
    filelist = deviation_report(fl_recs)
    written, per_issue_df = rq1(core, out, numbers)
    w, rq2_tables = rq2(core, out, numbers)
    written += w
    w, rq3_values = rq3(core, filelist, core_recs, fl_recs, issues, runs, out, numbers)
    written += w
    rq3_table = pd.read_csv(out / "rq3_filelist.csv")
    written.append(fig1_repeat_similarity(per_issue_df, out / "fig1_repeat_similarity.pdf"))
    written.append(fig2_conclusion_stability(rq2_tables["stability"], rq2_tables["envelope"],
                                             out / "fig2_conclusion_stability.pdf"))
    written.append(fig3_rq3_deviation(rq3_values, rq3_table, out / "fig3_rq3_deviation.pdf"))
    return written, numbers


def data_numbers(runs: pd.DataFrame, issues: pd.DataFrame, numbers: Numbers) -> None:
    sec = "Data"
    numbers.add(sec, "Issues (core), RQ3 issues, pilot issues among them",
                f"{len(issues)}, {int(issues.rq3.sum())}, {int(issues.pilot.astype(int).sum())}",
                "SWE-bench Lite issues of the selection (design section 4)", "data/issues.csv")
    for experiment, part in runs.groupby("experiment", sort=False):
        statuses = ", ".join(f"{s} {n}" for s, n in part.status.value_counts().items())
        cost = part.cost_usd.replace("", "0").astype(float).sum()
        numbers.add(sec, f"Runs, {experiment} ({part.experiment_id.iloc[0]})",
                    f"{len(part)}: {statuses}; ${cost:.2f}",
                    "every run recorded, by status, and the API cost of the agent runs",
                    "data/runs.csv")
        scored = part[part.excluded == ""]
        excluded = part[part.excluded != ""]
        numbers.add(sec, f"Scored runs, {experiment}", len(scored),
                    "runs the metrics engine scores (no excluded or no-op variant; status "
                    "success, failure or no_patch)", "data/runs.csv, excluded empty")
        if len(excluded):
            by = ", ".join(f"{b} / {v} {n}" for (b, v), n in
                           excluded.groupby(["backend", "variant"]).size().items())
            numbers.add(sec, f"Runs left out, {experiment}", f"{len(excluded)} ({by})",
                        "runs on excluded or no-op variants (skipped_variant) and runs without "
                        "a usable reply at the token limit (truncated)", "data/runs.csv, excluded")
        no_patch = int(((part.scoring_status == NO_PATCH) & (part.excluded == "")).sum())
        numbers.add(sec, f"no_patch runs, {experiment}", no_patch,
                    "scored runs without a usable diff: failures; left out by the robustness "
                    "analysis", "data/runs.csv, scoring_status")


def _changed(r) -> str:
    if _missing(r.cells):  # a figure
        return "none" if r.identical == "yes" else "PDF differs"
    return "--" if _missing(r.cells_changed) else str(int(r.cells_changed))


def robustness(out: Path, robust: Path, runs: pd.DataFrame, numbers: Numbers) -> list[Path]:
    """What changes when no_patch runs are excluded: per table, the cells that differ; per
    figure, whether its PDF differs."""
    rows = []
    for path in sorted(out.glob("*.pdf")):
        other = robust / path.name
        same = other.exists() and other.read_bytes() == path.read_bytes()
        rows.append({"table": path.stem, "rows": None, "cells": None, "cells_changed": None,
                     "identical": "yes" if same else "no"})
    for path in sorted(out.glob("*.csv")):
        if path.name == "robustness_changes.csv":
            continue
        other = robust / path.name
        a = pd.read_csv(path, dtype=str, keep_default_na=False)
        b = pd.read_csv(other, dtype=str, keep_default_na=False) if other.exists() else None
        if b is None or a.shape != b.shape or list(a.columns) != list(b.columns):
            changed = None
        else:
            changed = int((a.to_numpy() != b.to_numpy()).sum())
        rows.append({"table": path.stem, "rows": len(a), "cells": a.size,
                     "cells_changed": changed,
                     "identical": "yes" if changed == 0 else "no"})
    no_patch = {e: int(((p.scoring_status == NO_PATCH) & (p.excluded == "")).sum())
                for e, p in runs.groupby("experiment", sort=False)}
    df = pd.DataFrame(rows)
    note = ", ".join(f"{e} {n}" for e, n in no_patch.items())
    text = latex(
        "robustness_changes",
        "Robustness: the whole analysis again with no\\_patch runs excluded instead of counted "
        f"as failures (no\\_patch runs: {tex(note)}). Cells of each table that change; figures: "
        "whether the PDF changes.",
        ["Table or figure", "Rows", "Cells changed"],
        [[tex(r.table), "--" if _missing(r.rows) else str(int(r.rows)), _changed(r)]
         for r in df.itertuples()], "lrr")
    changed = df[df.identical != "yes"]
    numbers.add("Robustness: no_patch runs excluded", "Tables and figures that change",
                f"{len(changed)} of {len(df)}" + (f" ({', '.join(changed.table)})"
                                                 if len(changed) else ""),
                f"no_patch runs excluded instead of counted as failures (no_patch runs: {note}); "
                "every table recomputed and compared cell by cell, every figure byte by byte",
                "robustness_changes.csv")
    return write_table(df, text, out, "robustness_changes")


PREAMBLE = """\
# The numbers the paper cites

Written by `paper_analysis.py` from `data/runs.csv` and `data/issues.csv`; do not edit by hand.
Sources are files of this folder (the tables) and of the data folder.

Conventions, unless a row says otherwise:

- **Issue-level values.** Every comparison is paired by issue: one value per issue (the mean over
  its variants when it has several), then the mean over issues.
- **95% CI**: percentile bootstrap of the mean over issues, 5,000 resamples, seed 42, issues
  resampled ("clustered by issue").
- **p**: two-sided Wilcoxon signed-rank test, zero differences dropped (exact up to 50 non-zero
  differences without ties, else normal approximation with tie correction). **r**: matched-pairs
  rank-biserial. **q**: Benjamini-Hochberg within the family named in the row.
- **Verdict**: "lower" / "higher" when the 95% CI excludes zero on that side, else "no
  difference".
- **Similarity**: changed-line similarity of two patches (SequenceMatcher ratio of their +/- lines,
  1 when both have none). **Envelope** of an issue and backend: the 15 pair similarities of its 6
  original runs. **Repeat similarity**: their mean. **s** of a variant: its mean similarity to the
  6 original runs. **Envelope deviation**: s minus the envelope mean.
- **Localized** (the outcome): the run returned a valid diff that modifies every file the gold
  patch modifies. **Noisy issue**: its original runs disagree on the outcome.
- The 6 original runs of an issue are its first run and repeats 1-5, in run order. The
  single-repeat method (the earlier pilot's) compares a variant with the first run, against
  repeat r of the first run.
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, default=DATA_DIR, help=f"default: {DATA_DIR}")
    parser.add_argument("--out", type=Path, default=OUT_DIR, help=f"default: {OUT_DIR}")
    args = parser.parse_args(argv)
    runs = load_runs(args.data / "runs.csv")
    issues = load_issues(args.data / "issues.csv")
    written, numbers = analyze(runs, issues, args.out)
    robust, _ = analyze(runs, issues, args.out / ROBUSTNESS_DIR, exclude_no_patch=True)
    written += robust
    written += robustness(args.out, args.out / ROBUSTNESS_DIR, runs, numbers)
    data_numbers(runs, issues, numbers)
    (args.out / "NUMBERS.md").write_text(numbers.markdown(PREAMBLE), encoding="utf-8",
                                          newline="\n")
    written.append(args.out / "NUMBERS.md")
    print(f"Written {len(written)} files to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
