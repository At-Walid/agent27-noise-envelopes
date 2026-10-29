# Study design v1.0: the frozen analysis plan

*Frozen as the git tag `agent-prereg` on 23 October 2026, before any paid run. Changes made after
it are listed in `deviations.md`.*

*Anonymized for double-blind review. The name of the tool, the earlier pilot study and its venue,
the venues and deadline of this paper and of a planned follow-up study are replaced by neutral
wording ("the tool", "the earlier pilot", "the follow-up study"). Nothing about the issues, the
systems, the transformations, the measurements, the research questions, the tests, the
thresholds or the seeds was changed.*

## 1. Goal and framing

**Working title:** *How Much Wording Sensitivity Survives Run-to-Run Noise? A Per-Issue Noise Floor for LLM Issue Repair*

Evaluations of LLM-based issue repair, including the earlier pilot, usually compare a reworded issue with **one** run of the original. At temperature 0, two runs of the same issue already differ: in the earlier pilot, a repeat of the original reached only about 0.75 changed-line similarity to the first run. This study runs every original **six times**, so each issue gets its own noise envelope. It then asks how many apparent wording effects remain once that envelope is taken into account.

Framing sentence for the paper:
> We do not rank models. We measure how noisy single-call issue repair is at temperature 0, how many wording effects survive a per-issue noise floor, whether repository context changes either, and whether agreement between models can flag rewrites that changed the task.

## 2. Contributions

1. **A per-issue noise envelope** for single-call issue repair at T=0, for two backbones from different vendors, at outcome and patch level.
2. **A comparison of two evaluation methods** on the same data: a single-repeat comparison (as in the earlier pilot) against the per-issue envelope. The comparison shows how often single-repeat conclusions change when a different repeat is used.
3. **A first measurement of how repository context** (the file list) affects noise and wording sensitivity.
4. **A first test of differential triage (H1):** do variants that move both models outside their envelopes more often change the task or come from an ambiguous issue? Tested against blind ratings by two researchers.
5. **Data and scripts** released with the paper (an extension of the tool of the earlier pilot; the tool itself is described there and cited, not re-described).

## 3. Relation to the other papers

| Paper | What it contributes | Data |
|---|---|---|
| Earlier pilot (submitted) | The tool; a proof of concept | 20 issues, 2 backends, 1 repeat, aggregate noise floor |
| **This paper** | Per-issue noise envelope; what survives; context; first triage test | 50 issues, 2 backends, 5 repeats, new runs; file-list condition; 60 rated variants |
| Follow-up study (planned) | Scaffolded agents (OpenHands), test-based resolution, crossed scaffold × backbone design, full triage study | SWE-bench Verified, 50 + 10 issues, 4 systems |

Rules:
- This paper cites the earlier pilot for the tool and does not reuse its text, tables or figures.
- The 20 pilot issues are included, with all-new runs, and this is stated.
- The follow-up study will cite this paper as its preliminary study.

## 4. Issues

- **Pool:** the 62 SWE-bench Lite issues that meet the earlier pilot's localizability rule. The rule was fixed before any agent run: the issue text names the gold-patch file, the gold patch touches one file, the text has 100–400 words and at least 2 paragraphs.
- **Sample: 50 issues.** It includes the 20 pilot issues. The 50 are stratified by repository: each repository gets a quota proportional to its share of the 62 (largest-remainder apportionment), and at least its number of pilot issues; its places beyond its pilot issues are drawn at random from its other eligible issues (seed 2027). All nine repositories are represented (quotas: django 14, sympy 11, matplotlib 9, scikit-learn 5, astropy 3, seaborn 2, requests 2, pylint 2, pytest 2); 12 of the 62 are left out. *(Reworded on 23 October 2026, before any paid run, to describe the rule the selection script implements; the first wording said "drawn round-robin over repositories". The 50 issues are unchanged.)*
- **Why localizable issues:** a backend without repository access can only localize when the text points at the file. Without this filter, outcome metrics hit a floor, as observed in the earlier pilot's pre-screen.
- Selection script, seed and log: projects/agent2027/data/ (in this package: the seed and log in `selection/`).

## 5. Systems and settings

| ID | System | Backbone |
|---|---|---|
| B-C | LLM Baseline, single call | Claude Sonnet 4.6 (claude-sonnet-4-6) |
| B-G | LLM Baseline, single call | GPT-4o (gpt-4o) |

- Settings are identical to the earlier pilot: T=0, max 4,096 output tokens, 120 s timeout, forced `submit_patch` tool call returning one unified diff, no repository access (except in the file-list condition, Section 7).
- Model ids are pinned and all runs are done within a short window (a few days), to limit drift from model updates.

## 6. Transformations

The earlier pilot's six transformations, with two corrected versions:

| Transformation | Kind | Version |
|---|---|---|
| Paraphrase | LLM (claude-haiku-4-5-20251001) | as in the earlier pilot |
| Verbosity reduction | LLM | as in the earlier pilot |
| Paragraph → structured | LLM | as in the earlier pilot |
| Neutralize emotional | LLM | as in the earlier pilot |
| Typo injection | deterministic, seeded | **v2:** runs on masked text, never touches code, doctests, inline code, tracebacks or technical tokens; passes the quality gate |
| Reordering | deterministic, seeded | **v3:** boundary-preserving; a line ending in ":" stays with the block that follows it (fixes the psf/requests-2317 defect seen in the earlier pilot) |

- All variants pass the quality gate: mask-and-restore, no-op detection, technical-token retention, invented-detail check, placeholder integrity, one retry with feedback, then exclusion with a recorded reason.
- Variants are generated once and cached with provenance. Both backends receive identical variant texts.
- For the 20 pilot issues, the earlier pilot's cached variants are reused, except for typo injection and reordering, where the new versions replace them. This is reported.

## 7. Conditions and grid

**Core grid (RQ1, RQ2, RQ4):**
- 50 issues × 2 backends × (1 original + 5 repeats + 6 variants) = up to 1,200 runs.
- About 1,178 runs after no-op variants, which are recorded but not run.

**File-list condition (RQ3):**
- 25 of the 50 issues × Claude Sonnet 4.6 × (1 original + 5 repeats + 6 variants) = up to 300 runs.
- The prompt is the core prompt plus the list of the repository's `.py` files at the issue's base commit. Files under tests/, test/, docs/, doc/, examples/ and benchmarks/ are excluded.
- **No truncation.** If the filtered list exceeds a fixed budget of 15,000 tokens, the issue is excluded from RQ3 and logged. Files are never selected or ordered using the gold patch.
- The 25 issues are drawn with a fixed seed from the 50, stratified by repository, among issues whose list fits the budget.

## 8. Measurements

**Run statuses.**
- Excluded from outcome and similarity metrics, counted and reported: `truncated` (output hit the token limit), `timeout`, `cost_exceeded`, API/infrastructure `error`.
- `no_patch` (the model answered but produced no usable diff) counts as a **failure**, because that is the system's behavior. A robustness check repeats the outcome analyses with `no_patch` excluded.

**Outcome.** Localization success: a valid diff that touches every gold-patch file. The gold patch is never shown to a model.

**Patch.**
- Changed-line similarity: difflib ratio over the +/− lines, file headers excluded.
- File-set Jaccard.

**Per-issue noise envelope** (per issue × backend × condition):
- *Patch:* the 15 pairwise changed-line similarities among the 6 original runs.
- *Outcome:* agreement among the 6 original runs. An issue is **noisy** if the original runs disagree.

**Per-variant deviation:**
- **z-score:** (mean similarity of the variant to the 6 original runs − mean of the 15 pairwise similarities) / max(SD of the 15, ε).
- **Rank version** (robustness): the variant's mean similarity ranked among the 6 originals' own leave-one-out mean similarities.
- **Outcome flip:** counted only when all original runs agree. Split into gains (failure → success) and losses (success → failure).
- **Binary deviation** (used only for triage, RQ4): z < −2, or an outcome flip on a non-noisy issue. The thresholds are fixed before the runs (Section 11).

## 9. Research questions

**RQ1 — How noisy is single-call issue repair at T=0?**
- Per backend: the distribution of pairwise repeat similarity, the share of byte-identical repeats, the share of noisy issues (outcome disagreement), and per-issue envelope widths.
- Do noise levels differ between backbones (paired by issue, Wilcoxon) and across repositories (descriptive)?
- Expected value: concrete numbers for something evaluations usually assume away.

**RQ2 — How many wording effects survive a per-issue noise floor?**
- *(a) Transformation level:*
  - Paired difference between variant similarity and the repeat baseline, computed two ways:
    - **single-repeat** (the earlier pilot's method): the variant vs one original run, against one repeat;
    - **envelope** (this paper): the variant vs all 6 originals, against the 15 pairwise similarities.
  - For the single-repeat method, recompute the result for each of the 5 possible repeats. Report how often a transformation's conclusion (CI excludes zero or not) **changes depending on which repeat was used**. This is the conclusion-stability measure.
- *(b) Variant level:* the share of variants outside their issue's envelope per transformation and backend, with flips split into gains and losses.
- Confirmation targets from the earlier pilot: reordering, paraphrase and paragraph→structured beyond the noise floor for both backends; typo and neutralize within it. Reordering v3 and typo v2 may change this, which is itself informative.

**RQ3 — Does repository context (a file list) change noise and wording sensitivity?** (Claude, 25 issues)
- Paired by issue, with vs without the file list:
  - envelope width (RQ1 measures);
  - per-transformation deviation (RQ2 measures);
  - localization rate;
  - patches touching files outside the gold set.
- Interpretation: a first, cheap look at whether giving the model more of the repository, a step toward a scaffolded agent, stabilizes or destabilizes behavior.

**RQ4 — Do cross-model deviations flag task-changing rewrites? (H1, preliminary)**
- *H1:* the more backends a variant deviates in (none / one / both), the more often raters judge that it changed the task or that the original issue is ambiguous.
- Ground truth: the blind rating study (Section 10).
- Test: one-sided Cochran–Armitage trend test across the three strata. Report the proportions per stratum with 95% CIs.
- Competing explanation, discussed either way: both models may share biases and react alike to a valid rewording.

## 10. Rating study (RQ4)

- **Sample: 60 variants**, stratified by deviation breadth: 20 deviating in both backends, 20 in one, 20 in none. If a stratum has fewer than 20 variants, all of them are taken. The sample is spread across transformations, fixed seed.
- **Raters:** the two authors, independently.
- **Shown:** the original issue and the variant. Not shown: any model output, the stratum, model names. Order randomized; neutral item ids.
- **Labels:**
  - (1) *Does the variant preserve the task?* yes/no; if no, the reason (lost information, added information, changed emphasis)
  - (2) *Is the original issue itself ambiguous?* yes/no
- **Agreement:** Cohen's κ per label, target ≥ 0.8. If κ < 0.8: the guidelines are refined on a separate set of 10 items, then re-rated. Disagreements are resolved by discussion. Pre- and post-discussion labels are both reported. H1 uses the adjudicated labels.
- **A variant counts as explained** if label 1 = no OR label 2 = yes.
- **Effort:** about 60 × 3 min ≈ 3 h per rater, including the guideline read.

## 11. Statistics and pre-registration

- Wilcoxon signed-rank (zero differences dropped) for paired comparisons; matched-pairs rank-biserial as the effect size.
- 95% bootstrap CIs, **clustered by issue**, 5,000 resamples, fixed seed.
- Benjamini–Hochberg within each RQ family.
- Everything is reported as estimates with CIs; claims are framed by effect size, not only by p-values.
- **Rough power:** the earlier pilot's paired differences were about −0.18 to −0.21, with SD about 0.3 (d ≈ 0.6–0.7). With 50 issues, power is above 0.9 per transformation at α=0.05 and stays adequate after BH across 6 transformations. RQ3, with 25 issues, has lower power and is labelled exploratory.
- **Frozen before the paid runs:** this document, the deviation thresholds, the RQ2 procedure, the H1 test, and the selection seed. Freeze them as a local git tag `agent-prereg` with the date, mentioned in the paper.

## 12. Cost

Measured from the earlier pilot's runs: Claude $0.009603 per run, GPT-4o $0.004861 per run, variant generation $0.00231 per variant.

| Phase | Runs | Estimated cost |
|---|---|---|
| Smoke test (2 issues, both backends, full grid) | about 48 | about $0.40 |
| Core grid | about 1,178 | about $8.50 + about $0.30 variant generation |
| File-list condition (RQ3) | up to 300 | about $8 (re-estimate with the real list sizes) |
| **Total** | about 1,530 | **about $17–18** |

Top-ups: about $15 on Anthropic and $5 on OpenAI, with auto-reload off and per-experiment caps set in the configs.

## 13. Threats to validity

- **Construct:**
  - localization is weaker than test-based resolution;
  - changed-line similarity is a proxy, and a different patch can be equally valid;
  - a 6-run envelope is still a small sample of a model's run-to-run behavior.
- **Internal:**
  - the variant generator (Claude Haiku 4.5) comes from the same vendor as one backend;
  - provider-side model updates during the run window, mitigated by pinned ids and a short window;
  - the two raters are the authors (mitigated by blinding and κ).
- **External:**
  - single-call, repository-blind backends (except the file list), not scaffolded agents;
  - only SWE-bench Lite issues whose text names the file (62/300 eligible), Python only;
  - two backbones.
- **Conclusion:**
  - 50 issues and 25 for RQ3;
  - a preliminary H1 test with 60 rated variants;
  - all analyses beyond RQ1/RQ2 framed as exploratory.

## 14. Paper outline (IEEE, 8 pages incl. references)

1. **Introduction** (0.75 p): the single-run problem; the earlier pilot's observation (repeat similarity about 0.75); contributions.
2. **Background** (0.5 p): the tool (cite the earlier pilot), metamorphic relation, noise in LLM code generation, closest related work.
3. **Study design** (1.75 p): issues, systems, transformations (v2/v3 changes), conditions, envelope and deviation definitions, RQ1–RQ4, rating study, statistics, pre-registration.
4. **Results** (2.5 p): RQ1 (noise figure), RQ2 (single-repeat vs envelope table, conclusion-stability figure), RQ3 (paired comparison), RQ4 (strata proportions, κ).
5. **Discussion** (0.75 p): implications for agent evaluation (report repeats; use per-issue envelopes; single-repeat conclusions can flip), what changes for scaffolded agents (the follow-up study).
6. **Threats to validity** (0.5 p).
7. **Related work** (0.5 p), if not merged into Background.
8. **Conclusion and data availability** (0.25 p).
9. **References** (about 0.75 p).

## 15. Timeline

| Date | Task |
|---|---|
| Oct 22–23 | Implementation fixes (typo v2, file-list rule, no_patch, estimator); freeze `agent-prereg` |
| Oct 24 | Smoke test (about $0.40); check outputs |
| Oct 25–27 | Core grid; file-list condition |
| Oct 28–31 | Rating study (both raters; κ; adjudication) |
| Nov 1–5 | Analysis RQ1–RQ4; figures and tables |
| Nov 6–11 | Writing; internal review by the co-author |
| Nov 12 | Final checks (page limit, references, links) |
| Nov 13 | Submit |
