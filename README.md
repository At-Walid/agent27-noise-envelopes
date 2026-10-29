# Run-to-run noise and wording sensitivity in LLM issue repair: replication package

This is the data and analysis behind the paper *How Much Wording Sensitivity Survives Run-to-Run
Noise?*. The CSV files here are enough to reproduce every table, figure and cited number of the
paper. You don't need a database, an API key or network access (except to install the Python
packages).

The package is anonymized for double-blind review. The section [Anonymization](#anonymization)
lists what was changed.

## The study in brief

- **Issues:** 50 SWE-bench Lite issues (`data/issues.csv`; how they were drawn: `selection/`).
- **Systems:** a single-call LLM baseline on Claude Sonnet 4.6 and on GPT-4o. It runs at
  temperature 0 with at most 4,096 output tokens and no repository access, and a forced
  `submit_patch` tool call returns one unified diff.
- **Runs:** each original issue runs 6 times per backend (the first run and 5 repeats). Each of 6
  rewordings of it runs once: typo injection, reordering, paraphrase, verbosity reduction,
  paragraph → structured, and neutralize emotional.
- **Experiments:**
  - the core grid: experiment `ad6aa499-fae3-4c06-9996-b46185e1ec87`, 1,200 runs;
  - RQ3, the file-list condition: experiment `72e7798d-b1b3-47c8-994e-c37aa4b853da`, 300 runs.
    Claude ran on 25 of the issues with the repository's file list in the prompt.

  Both experiments ran on 28 October 2026.
- **Outcome:** localization. A run succeeds when it returns a valid diff that modifies every file
  that the issue's gold patch modifies.
- **Patch similarity:** changed-line similarity. This is the ratio of Python's
  `difflib.SequenceMatcher` over the `+`/`-` lines of the two patches.
- **Noise envelope** of an issue and backend: the 15 pairwise similarities of its 6 original
  runs. Each variant is compared against this envelope.

The study design was frozen as the git tag `agent-prereg` on 23 October 2026, before the paid runs.
That design is the analysis plan in `docs/analysis_plan.md`. Every change made after the freeze is
listed in `docs/deviations.md`, with its reason and date. Two of those changes matter most:

- The rating study planned as RQ4 is not part of the paper (section 3 there).
- The comparison with typo injection is post-hoc and exploratory (section 2 there).

## Requirements

- Python 3.11. The outputs in `results/` were made and checked with Python 3.11.9 on Windows 11.
- The packages in `requirements.txt`, pinned to the versions the paper used.
- Times New Roman, for byte-identical PDFs (see the note below).

```bash
python -m venv .venv
. .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Reproducing the tables and figures

To write every table and figure at once (one to two minutes):

```bash
python code/paper_analysis.py --out reproduced
```

This writes each table as CSV and as a LaTeX table (booktabs), and the three figures as PDF. It
also writes `NUMBERS.md`, which lists every number the paper cites with its definition and source
file. Finally, it repeats the whole analysis with no_patch runs excluded, in
`reproduced/robustness_no_patch_excluded/`.

To reproduce one table or figure and check it, use `code/reproduce.py`. Each command runs the whole
analysis (one to two minutes) and writes only that item's files to `reproduced/`. It then compares
each file byte for byte with its copy in `results/` and prints `identical` or `DIFFERS`.

| Paper | Command | Files (in `results/`) |
|---|---|---|
| RQ1: noise per backend | `python code/reproduce.py rq1-noise` | `rq1_noise.csv`, `.tex` |
| RQ1: the noisy issues | `python code/reproduce.py rq1-noisy-issues` | `rq1_noisy_issues.csv`, `.tex` |
| RQ1: Claude vs GPT-4o | `python code/reproduce.py rq1-backends` | `rq1_backends.csv`, `.tex` |
| Fig. 1: per-issue repeat similarity | `python code/reproduce.py fig1` | `fig1_repeat_similarity.pdf` (data: `rq1_per_issue.csv`) |
| RQ2: deviation from the envelope | `python code/reproduce.py rq2-envelope` | `rq2_envelope.csv`, `.tex` |
| RQ2: conclusion stability | `python code/reproduce.py rq2-stability` | `rq2_conclusion_stability.csv`, `.tex`, `_ci.tex` |
| Fig. 2: conclusion stability | `python code/reproduce.py fig2` | `fig2_conclusion_stability.pdf` |
| RQ2 post-hoc: each transformation minus typo injection | `python code/reproduce.py rq2-posthoc-typo` | `rq2_posthoc_typo.csv`, `.tex` |
| RQ3: the file list | `python code/reproduce.py rq3-filelist` | `rq3_filelist.csv`, `.tex` |
| Fig. 3: RQ3 deviation | `python code/reproduce.py fig3` | `fig3_rq3_deviation.pdf` |
| Robustness: no_patch runs excluded | `python code/reproduce.py robustness` | `robustness_changes.csv`, `.tex`, `robustness_no_patch_excluded/` |
| Every number cited in the text | `python code/reproduce.py numbers` | `NUMBERS.md` |
| All of the above | `python code/reproduce.py all` | all 41 files |

`results/` holds the outputs the paper used. On Windows 11 with Python 3.11.9 and the pinned
packages, every reproduced file is byte-identical to its copy in `results/`, PDFs included.

**Note on fonts.** The figures use Times New Roman. On a machine without it, matplotlib falls back
to another serif font, so the PDFs differ in their bytes and `reproduce.py` reports them as
`DIFFERS`. The CSV, LaTeX and Markdown files are not affected.

## Contents

| Path | What |
|---|---|
| `data/runs.csv` | every run of both experiments (one row per run) |
| `data/variants.csv` | every issue text the runs used: the originals and the variants |
| `data/issues.csv` | the 50 issues |
| `data/experiments.json` | both experiments: status, run date, cost and the configuration each ran with |
| `configs/` | the two experiment configurations |
| `selection/` | how the issues were drawn: the issue list, the selection log (rule, seeds, quotas), and the file-list sizes per issue (`file_list_stats.csv`) |
| `docs/analysis_plan.md` | the study design as frozen before the paid runs (`agent-prereg`) |
| `docs/deviations.md` | the changes made after the freeze |
| `code/paper_analysis.py` | the analysis: reads `data/runs.csv` and `data/issues.csv`, writes every table, figure and `NUMBERS.md` |
| `code/reproduce.py` | one table or figure at a time, checked against `results/` |
| `code/envelope_analysis/` | the metric modules the analysis uses: patch similarity, localization outcome, noise envelope and deviation, statistics |
| `requirements.txt` | the Python packages, at the versions the paper used |
| `results/` | the outputs of `code/paper_analysis.py`, as the paper used them |
| `LICENSE` | MIT, for the code |
| `SHA256SUMS` | a checksum of every file |

### `data/runs.csv`

| Column | Meaning |
|---|---|
| `experiment`, `experiment_id` | `core` or `filelist`, and the experiment's id |
| `condition` | `issue only` (core) or `file list` (RQ3) |
| `run_id`, `run_order` | the run's id, and its position in its experiment, by start time |
| `agent_name`, `model`, `backend` | the configured agent, the API model id, and the backend's name |
| `issue_id` | the SWE-bench instance id |
| `variant`, `variant_id`, `transform_version` | `original` or the transformation; the text the run used (`variants.csv`); the transformation's version |
| `control_repeat`, `repeat` | `control_repeat` is 1 for a repeat of the original. `repeat` is 0 for the first run of the original and for a variant's run, and 1–5 for the repeats, in run order |
| `status` | as recorded: `success`, `truncated` (the reply was cut at the token limit without a diff), or `skipped_variant` (the variant was excluded or a no-op, so the agent was not called) |
| `scoring_status` | the status the analysis scores (the same as `status` in these experiments) |
| `excluded` | empty when the run is scored. `quality_warning` or `noop`: a run on an excluded or no-op variant. `status`: the run did not complete normally |
| `outcome` | 1 if localized, 0 if not; empty for an excluded run |
| `files_in_patch` | the files the patch modifies (`;`-separated) |
| `patch` | the unified diff the agent submitted |
| `token_input`, `token_output`, `cost_usd`, `runtime_seconds`, `stop_reason` | as reported by the API |
| `files_listed` | file-list condition only: how many files the prompt listed |
| `error_message` | why a run was not made, if it was not |

The raw API replies are not included, because the analysis uses only the patch.

### `data/variants.csv`

| Column | Meaning |
|---|---|
| `variant_id`, `issue_id` | the variant's id and its issue |
| `variant`, `transform_version` | `original` or the transformation, and its version |
| `generator` | `original`, `deterministic`, or the model that wrote an LLM-backed variant |
| `generation_mode` | how the variant was generated (empty for the originals) |
| `noop` | 1 if the variant is identical to the original after whitespace normalization |
| `quality_warning`, `exclusion_reason` | 1 if the variant was excluded, and why |
| `generation_cost_usd` | what generating the variant cost |
| `params` | the transformation's parameters as JSON; `reused_from_pilot: true` marks a variant reused from the earlier pilot (design section 6) |
| `notes` | the generator's notes |
| `text` | the text the agent received as the issue |

The issue texts are SWE-bench Lite's, verbatim. In 15 issues, the quoted tracebacks contain local
paths from the machines of the people who reported them (Linux, macOS and Windows home folders).
These paths are part of the public issues. They are left unchanged because they are the text the
models received.

### `data/issues.csv`

| Column | Meaning |
|---|---|
| `issue_id`, `repository`, `base_commit` | the SWE-bench instance |
| `pilot` | 1 if the issue is one of the 20 issues of the earlier pilot |
| `rq3` | 1 if the issue is one of the 25 RQ3 issues |
| `gold_files` | the files the gold patch modifies |
| `file_list_files`, `file_list_tokens` | the size of the file list given in the file-list condition |
| `gold_patch` | the gold patch |

## Definitions

The full definitions are in `results/NUMBERS.md` (its conventions section, and one line per
number) and in the docstring of `code/envelope_analysis/metrics/deviation.py`. In short:

- Comparisons are paired by issue.
- **95% CIs** are percentile bootstraps: 5,000 resamples, seed 42, issues resampled.
- **p** is a two-sided Wilcoxon signed-rank test, with zero differences dropped.
- **q** is Benjamini–Hochberg within each research question's family.
- A **noisy issue** is one whose 6 original runs disagree on the outcome.
- A variant is **outside the envelope** when z = (s − envelope mean) / max(envelope sd, 0.05) is
  below −2. Here s is the variant's mean similarity to the 6 original runs.
- **Gains and losses** are outcome flips on non-noisy issues.

## Anonymization

This package is derived from the authors' data snapshot. The following changes were made for
double-blind review:

- **Names.** The tool's name and the name and venue of the earlier pilot study are replaced by
  neutral wording ("the tool", "the earlier pilot"). This applies to the documents, the
  configurations, the selection log, the text fields of the data, and the code's comments and
  identifiers.
- **Code.** `code/envelope_analysis/` holds the tool's metric modules under a neutral package
  name, with some identifiers renamed. An unused loader that read the tool's database was removed,
  along with its SQLAlchemy dependency. No computation was changed.
- **`data/runs.csv`.** The `created_at` timestamps are dropped. The analysis does not use them,
  and `run_order` keeps the order of the runs.
- **`data/variants.csv`.** In `params`, the pointers into the earlier pilot's database (record
  ids, its project name and its experiment ids) and an internal cache key are dropped. A
  `reused_from_pilot` flag replaces them.
- **`data/experiments.json`.** The experiments' timestamps are reduced to the run date.
- **Analysis plan.** `docs/analysis_plan.md` and `docs/deviations.md` say at the top what they
  neutralize.

The data values the analysis reads were not changed. Every table and figure in `results/` is
byte-identical to the one the paper used. The only differences in `NUMBERS.md` are in three of its
labels, where the old wording named the earlier pilot.

## Sources and license

- **Issues:** the issue texts and gold patches are from SWE-bench Lite (Jimenez et al., ICLR
  2024). The issues were written in the repositories named in `data/issues.csv`.
- **Generated for this study:** the variants, the agents' patches and the outcomes.
- **Code:** `code/` is under the MIT License (`LICENSE`).
