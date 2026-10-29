# Deviations from the pre-registration (`agent-prereg`)

The study design v1.0 (`analysis_plan.md`), the configs, the issue selection and the analysis and
rating code were frozen as the git tag `agent-prereg` on 23 October 2026 (commit ee42913), before
any paid run of this study. Every change made after it is listed here.

*Anonymized for double-blind review: the tool's name, its earlier pilot study and a planned
follow-up study are named neutrally ("the tool", "the earlier pilot", "the follow-up study"). File
paths refer to the tool's repository, which is not part of this package; the metric modules the
analysis uses are in `code/envelope_analysis/` here, and `paper_analysis.py` is in `code/`.*

## 1. LLM-backed variant generation: placeholders only when the input has some (28 October 2026)

**What happened.** The smoke test (`agent_smoke.json`, experiment
5b8292a2-d128-4aad-8689-cc9903613330, 28 October 2026) excluded three LLM-backed variants of
django__django-11019 after both attempts: paraphrase, verbosity_reduction and
paragraph_to_structured, each with `placeholder_lost: [[CODE_1]] unknown, [[CODE_2]] unknown`.
The issue's code (a form definition and a warning) is neither fenced nor indented after a blank
line, so code masking (the tool's specification, section 11.7) found nothing to mask: the model's
input had no placeholder. The prompt still ended with the generic placeholder rules ("placeholders
such as [[CODE_1]]") and its few-shot example used [[CODE_1]] and [[CODE_2]], and the model
replaced the form definition and the warning by placeholders of its own ("All code has been
replaced with placeholders as instructed"); the retry's feedback did not tell it that the input
had none. The double-backtick limitation of the retention check (specification section 11.5)
plays no part: the issue has no backtick.

**What changed** (the tool's specification v3.3, sections 11.4 and 11.7):

- New versions of the four LLM-backed transformations: paraphrase, verbosity_reduction and
  neutralize_emotional 5 (were 4), paragraph_to_structured 6 (was 5). The prompt mentions
  placeholders only when the masked input has some, and then names exactly those; the few-shot
  examples use none; neutralize_emotional's prompt no longer mentions placeholders.
- A placeholder the model invents is retried with the feedback "remove [[CODE_n]], it does not
  exist", followed by the input's placeholders or the statement that it has none. Still two
  attempts in all, then exclusion.
- The study's configs (`agent_smoke.json`, `agent_core.json`, `agent_filelist.json`) pin the new
  versions. Each variant records its version (`transform_version`), in the project database and
  the shared variant store.

**What it applies to.** Newly generated variants only: the LLM-backed variants of the 30 issues
that are not pilot issues (120 in the core grid; the file-list condition reuses them). No variant
of this study had been generated before the change: its database and the shared store held none.

**What did not change.**

- The 20 pilot issues' LLM-backed variants (80, 4 of them excluded) are still the earlier pilot's,
  reused as they are (design section 6): each new version reuses every variant of the version
  before it, whatever its outcome (`reuses_versions`). A test checks that this study finds the
  same 80 variants as the pilot's versions do.
- The earlier pilot's runs, replays and previews still use the versions the frozen pilot ran
  (paraphrase, verbosity_reduction and neutralize_emotional 4, paragraph_to_structured 5, typo
  injection 1, reordering 2), with their prompt unchanged.
- Typo injection v2 and reordering v3, the issues (50 core, 25 RQ3), the grid, the seeds, the
  deviation thresholds, the RQ1–RQ4 procedures, the H1 test and the rating kit.
- No analysis code changed: the tool's metric and analysis modules and the study's scripts are as
  they were at `agent-prereg` (a `git diff agent-prereg` of them shows only scripts added later,
  all new files: `posthoc_typo_reference.py`, section 2; `paper_analysis.py` and
  `export_release.py`, section 4).

**The smoke test's runs are not runs of this study.** Experiment 5b8292a2 was run by another
installation of the tool (version 1.0.0, an editable install of a separate folder, first on the
PATH), not by the repository's code: its config snapshot has no project, it went to the tool's
legacy database at the repository root instead of the study's database, and its variants are typo
injection v1 and reordering v2. Its generation failure is nonetheless the one described above:
that installation's prompt and masking code are the same as the repository's before the change.
The smoke test is run again with the repository's code before the core grid.

## 2. Post-hoc exploratory analysis: each transformation against typo injection (28 October 2026)

**What was added.** `scripts/posthoc_typo_reference.py`, which the design does not have. Per
backend and transformation other than typo injection, paired by issue (the issues where both
variants were scored in that backend), the transformation's deviation minus typo injection's,
typo injection v2 being the minimal-edit reference (a few plain prose words get a typo; code and
technical tokens are never touched). Two measures:

- the envelope-method deviation of RQ2 (s minus the envelope mean). Both variants are measured
  against the same envelope, so the difference is s_T − s_typo; negative: the transformation's
  patches are further from the originals than typo injection's;
- the binary deviation (z < −2, or an outcome flip on a non-noisy issue; 1 or 0). The mean
  difference is the difference of the deviating shares; positive: the transformation deviates
  more often.

Each has the mean difference, its 95% bootstrap CI clustered by issue (issues resampled, 5,000
resamples, seed 42, as in RQ2), the Wilcoxon signed-rank p (zero differences dropped), the
rank-biserial and Benjamini–Hochberg q over the table as a family of its own. As a check, there is
also a 95% bootstrap CI clustered by repository: repositories are resampled with all their
issues, and with 9 repositories it is only approximate. The paper pipeline (section 4) computes
the same table from the CSV data, `results/rq2_posthoc_typo.*` in this package, labelled
post-hoc: the comparison is one function, `paper_analysis.typo_reference`, which
`posthoc_typo_reference.py` imports.

**When and why.** Decided on 28 October 2026, after the core grid (experiment
ad6aa499-fae3-4c06-9996-b46185e1ec87) had run and its RQ1 and RQ2 tables were known. In RQ2,
typo injection v2, the minimal edit, itself has an envelope-method CI below zero with both
backends (Claude Sonnet 4.6 −0.105, 95% CI [−0.165, −0.048], q = 0.006; GPT-4o −0.074
[−0.122, −0.030], q = 0.070). Comparisons against the envelope alone therefore do not say which
transformations move the patches further than a minimal edit does; this comparison does.

**Status.** Post-hoc and exploratory: chosen after seeing the data, and reported as such (every
row of the table says "post-hoc, exploratory"). It is a multiplicity family of its own and is
never used for a confirmatory claim, for H1 or for the rating sample.

**What did not change.** The pre-registered analysis and its tables: the study's analysis and
rating-sample scripts and the tool's metric and analysis modules are as they were at
`agent-prereg`. The new script imports the analysis script and the deviation metrics and adds
nothing to them.

## 3. RQ4 removed from the paper: deferred to the follow-up study (29 October 2026)

**What changed.** RQ4 (design sections 9 and 10: H1, the blind rating of the sampled variants,
Cohen's kappa against the 0.8 target and the one-sided Cochran-Armitage trend test) takes its
ground truth from two independent raters. Only one rater is available for this paper, so RQ4
cannot be run as registered: one rater gives no agreement check and no adjudication, so the
labels would not be the registered ground truth. RQ4 and contribution 4 (the first test of H1) are
therefore removed from the paper and deferred to the follow-up study, where the design already
places the full triage study (design section 3). The paper answers RQ1-RQ3.

**When.** Decided on 29 October 2026, after the core grid (experiment
ad6aa499-fae3-4c06-9996-b46185e1ec87) and the file-list condition (experiment
72e7798d-b1b3-47c8-994e-c37aa4b853da) had run, and before any rating: the rating kit was
generated on 28 October 2026 and never sent to a rater. No rating exists, so the decision does not
depend on an RQ4 result.

**What is kept, for the follow-up study.** Unchanged, and not part of this package: the rating
kit (49 variants by the design's rule: all 9 of stratum "both", which has fewer than 20, and 20
each of "one" and "none"; seed 2027; per rater a PDF kit and a CSV answer sheet, in their own
order; the key and the sample log, never given to a rater), the rating guideline, and the scripts
that draw the sample, import the ratings and analyze them.

**What changes in the pipeline.** The paper pipeline (`paper_analysis.py`, section 4) writes no
RQ4 output. The binary deviation, defined for the RQ4 triage, is still computed and reported
descriptively (per transformation in `rq2_envelope`, and as the second measure of the post-hoc
table, section 2); no test of H1 uses it.

**What did not change.** RQ1-RQ3: their measures, tests, multiplicity families and thresholds.

## 4. The paper's tables and figures, computed from the CSV data (29 October 2026)

**What was added.** Two new scripts. `export_release.py` writes the data snapshot: every run of
both experiments, scored exactly as the tool's metrics engine scores it (`data/runs.csv`), the
issue texts and variants, the issues, the configs, the selection, this file and the analysis code.
`paper_analysis.py` computes every table, figure and cited number of the paper from those CSV
files alone (no database, no API key), with the tool's metric modules (`deviation.py` and the
modules it imports; this package ships copies of them in `code/envelope_analysis/`). Outputs:
`results/` here, with `NUMBERS.md` (each cited number, its definition and source file) and the
robustness analysis in `results/robustness_no_patch_excluded/`.

**Checks.** The export checks that `data/runs.csv` gives back the database's scored runs, run for
run. Every value the paper tables share with the database-backed analysis script and
`posthoc_typo_reference.py` is identical: RQ1 (noisy issues, byte-identical pairs, envelope
widths, the backend comparison), RQ2 (both methods with their CIs, p and q, the variants outside
the envelope, gains and losses, the conclusion stability with each repeat), the post-hoc table and
RQ3.

**Descriptive measures the design does not name** (no test, no multiplicity family): issues
localized in 6/6 and in 0/6 runs; control flips against the first run (the metrics engine's
control flip rate); the repeat similarity's median and IQR over issues and over all pairs; and,
next to the design's byte-identical repeat pairs, the pairs with the same changed lines
(changed-line similarity 1: the same +/- lines, while context lines, hunk headers or text around
the diff may differ). The two counts differ (Claude 274 vs 337 of 750 pairs, GPT-4o 90 vs 211);
the tables report both, labelled. For RQ3, the same counts and the mean and maximum input tokens
per run.

**What did not change.** The database-backed analysis and rating-sample scripts, the tool's metric
and analysis modules, and every definition and test of RQ1-RQ3.
