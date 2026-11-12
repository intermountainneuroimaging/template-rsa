# Lab RSA methods: Kim et al. (2020) and Bretton et al. (2024)

This document formalizes the representational similarity methods of two papers and maps each step onto this workflow:

- **[K]** Kim, Smolker, Smith, Banich & Lewis-Peacock (2020). *Changes to information in working memory depend on distinct removal operations.* Nature Communications 11:6239. doi:10.1038/s41467-020-20085-4
- **[B]** Bretton, Kim, Banich & Lewis-Peacock (2024). *Suppressing the maintenance of information in working memory alters long-term memory traces.* Journal of Cognitive Neuroscience 36(10):2117–2136.

Sources: both methods sections, plus the authors' released MATLAB code for [K] ([LewisPeacockLab/NCOMMS1930876B](https://github.com/LewisPeacockLab/NCOMMS1930876B): `analysis/rsa_decode_01.m`, `analysis/rsa_decode_02_item.m`). [B] says it reuses "a custom pipeline in Python from our laboratory (Kim, Smolker, et al., 2020)". Where a paper is silent, the code for [K] is taken as the reference and the step is marked *(from code)*.

Configs: [examples/config-kim2020-ncomms.example.json](../examples/config-kim2020-ncomms.example.json) and [examples/config-bretton2024-jocn.example.json](../examples/config-bretton2024-jocn.example.json).

Code is split in two:

- **Core pipeline (design-agnostic), two analysis stages:**
  - *template* — [workflows/template_rsa.py](../workflows/template_rsa.py) (subject: category masks, weights, templates saved to disk, template validation) and [workflows/group_template_rsa.py](../workflows/group_template_rsa.py) (validation across subjects)
  - *decoding* — [workflows/decode_templates.py](../workflows/decode_templates.py) (subject: target events scored against the saved templates) and [workflows/group_decoding.py](../workflows/group_decoding.py) (group raw table, descriptive summaries, comparison-set check)
- **Study-specific post-hoc analyses:** [_interactive_notebooks/kim2020_removal_operations.ipynb](../_interactive_notebooks/kim2020_removal_operations.ipynb) and [_interactive_notebooks/bretton2024_ltm_fidelity.ipynb](../_interactive_notebooks/bretton2024_ltm_fidelity.ipynb), on helpers in [rsa_posthoc_common.py](../_interactive_notebooks/rsa_posthoc_common.py).

---

## 0. What kind of RSA this is, and what changed in the workflow

> **Vocabulary.** This document follows the papers: an *item* is one specific picture (a face, a scene). The pipeline calls that an **exemplar**, the finest level of `events.hierarchy` (here `category > subcategory > stimulus`), and keeps *item* for a single event of interest. So the papers' "item-level templates / weights" are `feature_weighting.levels: ["exemplar"]` and `exemplar_contrast` in the config, and the task operation is in an `operation` column (not "condition", which the pipeline uses for a coarser hierarchy level).

The lab's "RSA" is **feature-weighted template matching**: item-level pattern similarity between a template phase and a target phase. It is not the classic Kriegeskorte-style comparison of a data RDM with model RDMs.

- Each item gets a *template*, an activity pattern from a phase where it was perceived (the localizer in [K], the pre-manipulation memory test in [B]).
- Later patterns are scored by their Pearson correlation with the templates. The patterns come from the WM task in [K] and from the post-manipulation test in [B].
- Voxels are first restricted to category-selective voxels and then weighted by item-selective (or category-selective) GLM t-values.

The RDM config (`events.hierarchy` → RDM → `model_rdms`) has no slot for templates, target phases, weights, comparison sets or timecourses. So the workflow was extended rather than bent to fit:

| Change | Where |
|---|---|
| New analysis families, selected with `pipeline.analyses: ["template", "decoding"]` (alongside or instead of `"rdm"`): *template* builds and validates the templates, *decoding* scores target events against them | `utils/rsa_common.py::default_config`, `workflows/run_pipeline.py` |
| New `template_rsa` config section: `templates`, `feature_weighting`, named `targets`, per-target `group` | same; validated by `utils/validate_rsa_config.py::validate_template` |
| Time windows use mvpa-decoding's syntax: `{"start"/"end": {"reference": "onset"/"offset_end", "offset_seconds"}}` | `rsa_common.py::validate_window`, `window_volumes` |
| Atlas-label ROIs: `{"atlas_pattern", "labels"}` (e.g. fMRIPrep `aparcaseg`), resampled to the BOLD grid | `rsa_common.py::resolve_roi_mask` (used by both analysis families) |
| Subject and group stages for each, and common report pages (workflow diagram, template validation with every presentation shown separately, descriptive decoding) | `template_rsa.py`, `decode_templates.py`, `group_template_rsa.py`, `group_decoding.py`, `generate_report.py::template_section`, `decoding_section` |
| **Core / post-hoc split.** The core stops at a group raw table (`group/decoding/raw/…_scores.csv`, every events.tsv column carried). Design-specific statistics run in notebooks, as in mvpa-decoding: reference conditions, memory flags, trial N+1 sequences, contrasts and mixed models. | `_interactive_notebooks/` |
| `extra_sets`: score a trial against the templates named in any events column, at exemplar, category or subcategory level (e.g. the replacement category) | `template_rsa.py::extra_sets` |

The config layout deliberately mirrors mvpa-decoding: `templates` ≈ `training`, `targets` ≈ `testing` / `timecourse_decoding`, and `window` uses the same syntax. A lab config can therefore hold both the decoding and the RSA settings for one study.

---

## 1. Formal definition

### 1.1 Data

Fix subject *s* and ROI $R$, with $V$ voxels. For run $r$, let $X_r \in \mathbb{R}^{T_r\times V}$ be the preprocessed BOLD restricted to $R$.

- $C_r \subset \{0,\dots,T_r-1\}$ is the set of censored volumes: scrubbed volumes, non-steady-state volumes, and dropped leading volumes.
- $\tilde X_r$ is $X_r$ after detrending, temporal filtering and optional confound regression, then z-scored over time per voxel and run. Censored volumes are excluded from the z-scoring and set to missing.

Rows $\tilde x_{r,t} \in \mathbb{R}^V$ are single-volume patterns.

**Workflow:** `template_rsa.timeseries` (`detrend`, `high_pass`, `low_pass`, `standardize`, `regress_confounds`), `pattern_estimation.confounds` / `.censoring` → `template_rsa.py::load_run` (nilearn `signal.clean` with `sample_mask`).

### 1.2 Event windows

An event $e$ has onset $o_e$ and duration $d_e$. A window $W$ is a start and an end, each relative to the onset or the offset, plus a lag $\ell$. The event's volumes are

$$\mathcal T_W(e) = \{\lfloor (o_e+\ell)/\mathrm{TR}\rfloor + k : k = 0,\dots,\max(1,\mathrm{round}(d_W/\mathrm{TR}))-1\} \setminus C_r$$

Its window pattern is $p_W(e) = \frac{1}{|\mathcal T_W(e)|}\sum_{t\in\mathcal T_W(e)} \tilde x_{r,t}$.

**Workflow:** `rsa_common.py::window_volumes` (same rounding as mvpa-decoding's `compute_volume_range`), `template_rsa.py::event_pattern`.

### 1.3 Category masks (first feature-selection step)

Over the template-phase runs, fit a GLM with one boxcar regressor per category $c \in \{1..C\}$, convolved with the HRF, plus nuisance regressors. In [K] the boxcars are 14.26 s mini-blocks of 3 same-subcategory trials, and the nuisance regressors are motion parameters. For each category, a t-contrast against the mean of the others:

$$\mathbf c_c = \mathbf e_c - \tfrac{1}{C-1}\textstyle\sum_{c'\neq c}\mathbf e_{c'},\qquad t_c(v)$$

$$M_c = \mathrm{clusters}_{\ge k}\big(\{v\in R: p(t_c(v)) < \alpha\}\big)$$

with $\alpha = .05$ uncorrected, one-sided, and $k = 10$ voxels in both papers.

- [K]: face vs. {fruit, scene}, and so on.
- [B]: scene > face, the only two categories.

Each item is analyzed only within its own category's mask $M_{c(i)}$.

**Workflow:** `feature_weighting.category_mask` (`p_threshold`, `cluster_extent`, `events`, `block_merge_gap_seconds` for mini-blocks) → `template_rsa.py::feature_selection`, `cluster_filter` (26-connectivity).

### 1.4 Item weights (second feature-selection step)

One least-squares-all (LSA) GLM over the template-phase runs, with one regressor per item $i \in \{1..N\}$. In [K], $N = 54$ and each item appears once in each of 5 runs. In [B], $N = 90$ and each item appears once in total. The weights are item-vs-others contrasts:

$$\mathbf c_i = \mathbf e_i - \tfrac{1}{N-1}\textstyle\sum_{j\neq i}\mathbf e_j,\qquad w_i(v) = t_{\mathbf c_i}(v)$$

The code for [K] also has a within-subcategory variant; the paper uses all other items.

For [B]'s **category-level templates**, $w_i(v) = t_{c(i)}(v)$, the category contrast from §1.3.

**Workflow:**

- `feature_weighting.exemplar_contrast` takes `all_others`, `within_category` or `within_subcategory`.
- `feature_weighting.levels` takes `exemplar` (the papers' item level), `category` and/or `none`, and every level listed is computed side by side.
- Both GLMs are fit as **one concatenated model** across runs. Item and category regressors are shared; drift, intercept, nuisance events and confounds are run-specific. This is how SPM does it, and it handles items seen in only one run [B]. Code: `template_rsa.py::concat_glm`.

### 1.5 Templates

For item $i$ with template events $E_i$ and template window $W_\text{tmpl}$:

- [K]: the 1.38 s image presentation, shifted 4.6 s = 10 TRs.
- [B]: the 2 s presentation, shifted 5 s.

$$T_i = \frac{1}{|E_i|}\sum_{e\in E_i} p_{W_\text{tmpl}}(e),\qquad \tilde T_i = (T_i \odot w_i)\big|_{M_{c(i)}}$$

$\odot$ is the element-wise product. In [B], $|E_i| = 1$: the template is the item's single pre-test exposure.

**Workflow:** `templates.events`, `templates.window` → `template_rsa.py::build_template_sets`. The templates are built from all template runs and saved (`…_templates.npz`) for the decoding stage.

### 1.5b Template validation

Let $u$ index presentation units: one per item per template run, $p_u$ = the mean window pattern of that item's presentations in that run (in [K] each item is shown once per run, so a unit is one presentation).

**`kim2020` (published; [K] code, `rsa_decode_02_item.m`, "targ vs. non-targ").** For every ordered pair of units $u \neq v$ with items $i_u, i_v$:

$$z_{uv} = \operatorname{artanh}\operatorname{corr}\!\big((p_u\odot w_{i_u})|_{M_{c(i_u)}},\ (p_v\odot w_{i_v})|_{M_{c(i_u)}}\big)$$

Each pattern is weighted by **its own** item's weights, and both are cut to the **row** item's category mask. Masks and weights come from all template runs. A unit's target score is the mean $z_{uv}$ over the other units of the same item; related, over units of other items in the same category; others, over units of other categories.

**`cv` (not in the papers; from mvpa-decoding's `kfold_cv`).** Folds over the template runs (`per_run`, `group_kfold`, `explicit_groups`). In fold $k$, the category masks, weights and templates $\tilde T^{(k)}_j$ are refit on the training runs only. Each presentation $e$ in the held-out run is scored against every fold template exactly as a target (§1.6): $r_j = \operatorname{corr}(\tilde T^{(k)}_j, (p(e)\odot w^{(k)}_j)|_{M^{(k)}_{c(j)}})$, then mean r over the set and Fisher z.

**Workflow:** `validation.methods`, `validation.cross_validation` → `template_rsa.py::validate_kim2020`, `validate_cv`, `resolve_folds`.

### 1.6 Scoring a target pattern

A target pattern $y\in\mathbb R^V$ is either a single volume (timecourse) or a window mean. It is weighted by the *same* item's weights before correlating *(from code: `ytarg_pat .* xbeta`)*:

$$r_i(y) = \operatorname{corr}\!\big(\tilde T_i,\ (y\odot w_i)|_{M_{c(i)}}\big)$$

For a set of items $A$, the score averages Pearson r first and then Fisher-transforms *(from code)*:

$$S_A(y) = \operatorname{artanh}\!\Big(\tfrac{1}{|A|}\textstyle\sum_{i\in A} r_i(y)\Big)$$

The sets, for a trial whose item is $i^\*$:

- **target** $= \{i^\*\}$
- **related** $= \{j\neq i^\*: c(j)=c(i^\*)\}$, other items of the same category
- **others** $= \{j: c(j)\neq c(i^\*)\}$, items of other categories
- Optional named sets from other event columns (`extra_sets`): $\{j: \ell(j) = e[\text{column}]\}$ for a level $\ell \in$ {exemplar, category, subcategory}. For example, on [K]'s replace trials, every template of the category switched to (`replacement`). The paper tracked the replacement with the category classifier; this is its item-RSA counterpart.

**Workflow:** `targets.<set>.related_level` (category or subcategory), `targets.<set>.extra_sets`, `feature_weighting.weight_target_patterns` → `template_rsa.py::comparison_sets`, `extra_sets`, `score_matrix` (vectorized over all volumes of a run), `TemplateSet.score`.

### 1.7 Group inference

Trials are averaged within subject before any t-test. Mixed models use every trial.

**Timecourse [K, Fig. 4].** For trial $n$ with onset $o_n$:

- Score the unshifted volumes $k=0,\dots,29$ from trial onset (30 TRs = 13.8 s): $S^{(n)}_A(k)$.
- Average per subject × operation × $k$.
- Baseline: subtract each curve's mean over $k\in[0,6)$, the first 2.76 s.
- Removal relative to maintenance: $\Delta_\text{op}(k) = S_\text{op}(k) - S_\text{maintain}(k)$.
- Test window $k = 6..20$ (TRs 7–21), split into five 3-TR blocks. In each block:
  - one-sample t-test vs 0 per operation (removal onset = first block below 0)
  - repeated-measures ANOVA across operations
  - pairwise paired t-tests, FDR-corrected

Replacement tracking is analysed the same way: on replace trials, the `replacement` set and the replaced item (`target`), both baseline-corrected, each tested against 0 per bin, plus their difference (the crossover).

**Workflow:** core target set `study_timecourse` (30 TRs from onset, `extra_sets.replacement`). Analysis in the Kim notebook, sections 2–3: `BASELINE_SECONDS = (0, 2.76)`, `REFERENCE = "maintain"`, `STAT_WINDOW_SECONDS = (2.76, 9.66)`, `BIN_SECONDS = 1.38`. All timing is in seconds, converted with the data's TR. Helpers: `baseline_correct`, `subtract_reference`, `bin_means`, `one_sample_by_bin`, `first_significant_bin`, `compare_groups_by_bin`.

**Encoding fidelity [K, Fig. 5].** Encoding fidelity $\mathrm{EF}_n = S_{\{i_n\}}(p_W(n))$, with $W$ = the 2.76 s image presentation shifted 4.6 s. Two analyses use it:

- *Sensitivity check:* target > related > others.
- *Proactive interference:* for each operation on trial $N$, compare $\mathrm{EF}_{N+1}$ when item $N+1$ is from the same category as item $N$ vs. a different one, giving a same − different score per operation.
  - Replace trials whose replacement category equals item $N+1$'s category are removed.
  - Tests: a one-sample t-test per operation, plus a repeated-measures ANOVA across operations.

**Workflow:** core target set `encoding_fidelity` (window mean, + 4.6 s). The analysis is in the Kim notebook, sections 4–5:

- `add_previous_trial_columns(..., ["category", "operation"], matches={"same_replacement": ("replace_category", "category")})` runs on the **full** trial sequence, before exclusions, so "previous trial" always means the trial actually shown.
- Then trials following excluded operations are dropped, along with the replace-overlap trials.
- Tests: `paired_contrast` (same vs. different, by `prev_condition`), and `difference_scores` → `rm_anova` + `pairwise_tests`.

**Representational fidelity [B, Fig. 6].** Fidelity $F_i = S_{\{i\}}(p_W(e^\text{post}_i))$ is the post-test pattern scored against the pre-test template, at both the item and category weighting levels. Each item carries:

- its operation from the manipulation phase
- its memory outcome on the post-test: remembered = "sure old" hit, forgotten = any other response

Three analyses:

1. *Validation:* in the manipulation phase, target > other items.
2. *Paired t-test* of subject means, remembered vs. forgotten, within each operation. Only subjects with both outcomes enter, which gave n = 18 in [B].
3. *Mixed model* per operation: $F \sim \text{memory} + (1+\text{memory}\mid\text{subject})$ (statsmodels `mixedlm`). The paper also reports bootstrap 95% CIs.

The same analysis is repeated in the VVS, PFC, parietal and hippocampus ROIs.

**Workflow:** the core target sets `manipulation` and `post_test` (window means, + 5 s), in all four `template_rsa.rois`. The rest is in the Bretton notebook:

- **Memory flag:** derived from the raw post-test `response` / `confidence` columns (`memory_flag`).
- **Operation:** joined from the study phase by item (`join_exemplar_columns`).
- **Template check:** `paired_contrast` of target vs. related.
- **Remembered vs. forgotten:** `paired_contrast` with a subject-bootstrap CI over 10,000 resamples.
- **Mixed models:** `mixed_model`, `z ~ C(memory, Treatment('remembered'))` with `re_formula` `~C(memory, Treatment('remembered'))`.

---

## 2. Step-by-step mapping

✅ = implemented as described. ≈ = implemented with a stated difference. ✗ = not implemented here. MVPA = belongs to mvpa-decoding.

### Kim et al. (2020)

| Step | Paper / code | Workflow (config → code) | Status |
|---|---|---|---|
| Preprocessing | FSL: motion correction, 128 s high-pass, detrend, FD > 0.9 mm volumes removed, first 10 TRs dropped, native space | fMRIPrep `space-T1w`; `timeseries.high_pass: 0.0078`, `detrend`; `confounds` FD 0.9 scrub; `censoring.drop_initial_volumes: 10` | ≈ fMRIPrep instead of FSL |
| Run-wise z-scoring | Princeton MVPA toolbox | `timeseries.standardize: zscore_sample` | ✅ |
| VVS ROI | Harvard-Oxford regions at 20%, in native space | `rois.vvs.mask_pattern` (build it from the HO atlas), or `atlas_pattern` + labels | ≈ ROI file is user-provided |
| Category masks | SPM GLM on localizer, mini-block boxcars + motion, cat > others, p < .05, k = 10 | `category_mask` with `block_merge_gap_seconds: 5`, `hrf_model: spm`, `drift_model: cosine`, AR(1) | ≈ nilearn instead of SPM; FD-censored volumes are also left out of the GLM |
| Item weights | SPM LSA, item vs. 53 others, t-values | `exemplar_contrast: all_others`, `levels: [exemplar]` | ✅ |
| Templates | z-scored, 3 presentation TRs + 10-TR shift, averaged over 5 runs, × weights | `templates.window` onset + 4.6 → offset + 4.6 | ✅ |
| Localizer validation (Fig. 3b) | Every presentation × every other; each pattern × its own item's t-map, inside the first item's category mask; z per pair; same > related > others | `validation.methods: ["kim2020"]` → `validate_kim2020` (§1.5b) | ✅ (an earlier leave-one-presentation-out version of this workflow was not the published method and has been removed) |
| Cross-validated localizer validation | — | `validation.methods: ["cv"]`: masks, weights and templates refit per fold | not in the paper (sensitivity check) |
| Target weighting; mean r → z | `ytarg .* xbeta`; mean r over set, then Fisher z | `weight_target_patterns: true`; `S_A` | ✅ |
| Item-level timecourse (Fig. 4b) | 30 unshifted TRs, baseline first 6 TRs, minus maintain, 5 × 3-TR blocks, one-sample t, rm-ANOVA, pairwise FDR | core `study_timecourse` → notebook §2 | ✅ |
| Replace-new tracking (Fig. 4a, light blue) | category-classifier evidence for the replacement category on replace trials | core `extra_sets.replacement` (category templates) → notebook §3 | ≈ item-RSA counterpart of a classifier measure |
| Encoding fidelity check (Fig. 5a) | target > related > others; rm-ANOVA, Tukey–Kramer | core `encoding_fidelity` → notebook §4 (rm-ANOVA + paired t, FDR) | ≈ paired t + FDR instead of Tukey–Kramer |
| N+1 proactive interference (Fig. 5b) | same − different EF by operation; replace-overlap trials removed; rm-ANOVA | notebook §5 (`add_previous_trial_columns` with `same_replacement`, `paired_contrast`, `rm_anova`) | ✅ |
| Trial-balanced bootstrap (15 trials, 1000 iterations) | supplementary | — | ✗ (subject bootstrap of the contrast instead) |
| Category / subcategory classifiers, operation classifiers, importance maps, between-subject and hyperaligned classification, sliding-window classifiers | MVPA | mvpa-decoding | MVPA |
| Univariate FEAT contrasts | FSL | — | ✗ (out of scope) |

### Kim et al. (2020): this workflow compared with the released code

The released MATLAB code was read line by line: `rsa_decode_01.m` (GLMs) and `rsa_decode_02_item.m` (templates, localizer validation, study-phase timecourse).

**Templates used for decoding: match.**

| Step | [K] code | This workflow | |
|---|---|---|---|
| Category masks | One SPM model; all localizer runs concatenated into one session; category > others, p < .05, k = 10 | Concatenated GLM over all template runs, same contrast and threshold | ≈ nilearn instead of SPM |
| Item weights | spmT of item vs. all others, same model | LSA exemplar-vs-all-others t-map, all runs | ✅ |
| Per-presentation pattern | Mean of z-scored, lag-shifted volumes; spike volumes excluded | `event_pattern`: mean of the window; censored volumes excluded | ✅ |
| Template | Mean over runs of (pattern × own item t), inside own category mask | Mean pattern × item t inside own category mask (the same, since the weighting is linear) | ✅ |

**Decoding (study-phase timecourse): match.**

| Step | [K] code | This workflow | |
|---|---|---|---|
| Score vs. template j | Study volume inside j's category mask × j's t-map (`ytarg_pat .* xbeta`), `corr2` with template j | `TemplateSet.score` / `score_matrix` with `weight_target_patterns: true` | ✅ |
| Sets | target = own item; related = other items of the category; others = other categories | `comparison_sets` with `related_level: category` | ✅ |
| Aggregation | Mean r over the set's items, then Fisher z, per TR; spike TRs skipped | Same; censored volumes give no score | ✅ |

The study runs are separate from the localizer runs, so this is already a held-out test.

**Localizer validation.** The earlier version of this workflow scored a presentation against its own template rebuilt without it, with both vectors weighted by template j's map. That was not the published method. It is replaced by `kim2020` (§1.5b). On one Clearvale subject (sub-090, VVS, mean Fisher z, target / related / others):

| Validation | target | related | others |
|---|---|---|---|
| `kim2020` (published) | 0.210 | 0.026 | 0.005 |
| Earlier leave-one-presentation-out (removed) | 0.258 | 0.114 | 0.027 |
| `cv` (feature selection refit per fold) | 0.162 | 0.114 | 0.032 |

In the published method, feature selection sees every presentation, including the one being validated. The `cv` method shows how much that matters: target drops by about 0.1 when the masks and weights are refit without the held-out run. Decoding of separate runs is unaffected.

### Bretton et al. (2024)

| Step | Paper | Workflow (config → code) | Status |
|---|---|---|---|
| Preprocessing | fMRIPrep 21.0.1, MNI152NLin2009cAsym, TR 1 s; fMRIPrep confounds "used to clean the data" (exact set not stated) | `bold_glob` in MNI; `confounds` motion (24) + WM/CSF + cosines; `timeseries.regress_confounds: true` | ≈ confound set assumed |
| ROIs | VVS, PFC, parietal, hippocampus from each participant's parcellation (Desikan-Killiany) in MNI, summed and binarized | `rois.*` = `atlas_pattern` (aparcaseg) + `labels` | ≈ VVS label list needs lab confirmation |
| Category mask + category weights | GLM on pre-test, scene vs. face t, p < .05, k = 10 | `category_mask.events` = all pre-test trials; `levels` includes `category` | ✅ |
| Item weights | LSA, each of 90 items vs. the 89 others | `exemplar_contrast: all_others`, concatenated GLM (items in one run each) | ✅ |
| Templates | pre-test pattern, single exposure, the 90 manipulated scenes | `templates.events` = pre-test scenes whose `trial_type` matches `new_.*` (the item's status is in the event name), `window` + 5 s; old scenes and faces are nuisance regressors in the item GLM | ✅ (validation skipped: single exposures) |
| Template check on manipulation phase | target > all other items | core `manipulation` → notebook §1 (target vs. related, i.e. other scenes) | ✅ |
| Memory flag | remembered = "sure old" hit on the post-test; forgotten otherwise | notebook `memory_flag` from `response` / `confidence` | ✅ |
| Operation per post-test item | from the manipulation phase | notebook `join_exemplar_columns` | ✅ |
| Fidelity, remembered vs. forgotten | per operation, paired t on subject means (n = 18 with both outcomes) | notebook §2 `paired_contrast` (subjects with both) | ✅ |
| Mixed model | (1 + memory \| participant), per operation | notebook §3 `mixed_model` (statsmodels) | ✅ |
| Bootstrap CIs | 10,000 bootstrap resamples of participants | notebook `paired_contrast` (`N_BOOTSTRAP = 10000`) | ≈ CI of paired difference, not of a pooled trial-level regression |
| ROI specificity (PFC, parietal, hippocampus) | same analysis per ROI | `template_rsa.rois` → notebook §4 | ✅ |
| Category-evidence classifier, operation classifier, engagement-variance and engagement → memory regressions | MVPA | mvpa-decoding | MVPA |

---

## 3. Validation status

- **Synthetic** ([tests/synthetic_lab.py](../tests/synthetic_lab.py)). A dataset built like both designs, using the lab's event schema:
  - a localizer
  - a study phase with maintain, replace-with-the-other-category and suppress trials
  - a post-test of old and new items with response and confidence columns

  Core checks ([tests/test_template_rsa.py](../tests/test_template_rsa.py)):
  - category masks split the ROI correctly
  - validation: `cv` ranks target > related > others; `kim2020` puts target well above both (own-item weights remove most shared category signal from the synthetic patterns); post-test target > related > others
  - the replacement set exists only on replace trials and overtakes the replaced item
  - maintain > suppress late in the trial
  - selecting a subset of template items through `templates.events`
  - single-exposure templates work
  - post-hoc keys in a core config are rejected

  Both notebooks are **executed** against it ([tests/test_notebooks.py](../tests/test_notebooks.py)). They recover:
  - suppress below maintain in the late bins
  - the replacement-minus-target crossover
  - target > related > others encoding fidelity
  - the template check
  - remembered > forgotten

  The helpers have unit tests ([tests/test_posthoc_common.py](../tests/test_posthoc_common.py)).
- **Haxby 2001** ([tutorial/config-haxby-template.example.json](../tutorial/config-haxby-template.example.json); categories stand in for items, i.e. the hierarchy is just `["category"]`):
  - item-weighted templates give held-out target − others = 0.54 Fisher z (unweighted: 0.30), n = 6, both p < .001
  - the timecourse peaks about 5–20 s after block onset
  - 3–4 of the 8 category masks per subject fell below 10 voxels with an 8-way contrast and fell back to the whole ROI (a guard that is not part of the published method; see §4.5)
- **Not yet validated:** reproducing published numbers from either paper. That needs the lab's data. The first run on real data should compare against [K] Fig. 3b / 5a (target ≈ 0.17, related ≈ 0.13, others ≈ 0.02 Fisher z) and [B]'s manipulation-phase check.

## 4. Assumptions and open questions for the lab

1. **Events schema.** The configs and notebooks assume these columns:
   - `stimulus`, `category`, `subcategory` (`events.hierarchy`: category > subcategory > stimulus; each stimulus must always have the same category and subcategory)
   - `operation`
   - `replace_category` ([K]: the category switched to on replace trials)
   - `response` and `confidence` on post-test rows ([B])
   - pre-test `trial_type` = `old_<item>` / `new_<item>` ([B]: the new scenes are the ones that will be manipulated, and the only template items)

   Rename them in the config queries and in the notebooks' configuration cells if yours differ.
2. **[B] VVS labels.** The paper lists Harvard-Oxford region names (intracalcarine, occipital pole, temporal occipital fusiform…) but says Desikan-Killiany. The config uses the nearest aparc labels. Confirm the label set, or use the HO atlas with `mask_pattern`.
3. **[B] confounds.** Which fMRIPrep regressors were used for the RSA data? The config assumes 24 motion parameters + WM/CSF + cosines.
4. **Feature-selection GLM details.** [K] used SPM's canonical HRF and default AR(1); here they are nilearn `spm` HRF + AR(1). Censored volumes are dropped from the GLM as well. Whether SPM runs included them is not stated.
5. **Empty category masks.** If no cluster of ≥ k voxels survives, the workflow falls back to the whole ROI and flags it (`fallback_to_roi` in `*_feature-selection.csv`). This is not in the papers, so check how the lab handled it, if it ever happened.
6. **[K] statistics.** Tukey–Kramer post hocs are approximated by paired t-tests + FDR. The trial-balanced bootstrap is not implemented.
7. **Weighting with negative t-values.** The category mask keeps only voxels with a positive category contrast, but an item's t-values inside it can still be negative. Templates and targets are multiplied by the raw signed t-values, as in the [K] code.

## 5. Running

```bash
python workflows/run_pipeline.py --config examples/config-kim2020-ncomms.example.json
```

The command runs the event table, validation, the template stage (templates + validation) and the decoding stage for every subject, the group stages, and the report. Then open the study's notebook in `_interactive_notebooks/`, set its configuration cell, and run all cells. The stages can also be run separately:

```bash
python workflows/template_rsa.py --config CONFIG --subject 01
python workflows/decode_templates.py --config CONFIG --subject 01
python workflows/group_template_rsa.py --config CONFIG
python workflows/group_decoding.py --config CONFIG
python workflows/generate_report.py --config CONFIG
```

Outputs:

- `<output_dir>/sub-<s>/template_rsa/`: saved templates, category masks, feature-selection QC, template validation (summary and every presentation pair), cv folds
- `<output_dir>/sub-<s>/decoding/`: trial-level scores per target set
- `<output_dir>/group/template_rsa/`: validation per subject and the presentation-level matrices averaged over subjects
- `<output_dir>/group/decoding/`: `raw/` (the notebooks' input), subject means, summaries, comparison-set tests, figures
- `<output_dir>/posthoc/<study>/`: the notebook's `stats_and_figures.pdf`, one CSV per table, `run_settings.json`
