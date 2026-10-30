# Lab RSA methods: Kim et al. (2020) and Bretton et al. (2024)

This document formalizes the representational similarity methods of two papers and maps each step onto this workflow:

- **[K]** Kim, Smolker, Smith, Banich & Lewis-Peacock (2020). *Changes to information in working memory depend on distinct removal operations.* Nature Communications 11:6239. doi:10.1038/s41467-020-20085-4
- **[B]** Bretton, Kim, Banich & Lewis-Peacock (2024). *Suppressing the maintenance of information in working memory alters long-term memory traces.* Journal of Cognitive Neuroscience 36(10):2117–2136.

Sources: both methods sections, plus the authors' released MATLAB code for [K] ([LewisPeacockLab/NCOMMS1930876B](https://github.com/LewisPeacockLab/NCOMMS1930876B): `analysis/rsa_decode_01.m`, `analysis/rsa_decode_02_item.m`). [B] says it reuses "a custom pipeline in Python from our laboratory (Kim, Smolker, et al., 2020)". Where a paper is silent, the code for [K] is taken as the reference and the step is marked *(from code)*.

Configs: [examples/config-kim2020-ncomms.example.json](../examples/config-kim2020-ncomms.example.json) and [examples/config-bretton2024-jocn.example.json](../examples/config-bretton2024-jocn.example.json).

Code: [workflows/template_rsa.py](../workflows/template_rsa.py) (subject level) and [workflows/group_template_rsa.py](../workflows/group_template_rsa.py) (group level).

---

## 0. What kind of RSA this is, and what changed in the workflow

The lab's "RSA" is **feature-weighted template matching**: item-level pattern similarity between a template phase and a target phase. It is not the classic Kriegeskorte-style comparison of a data RDM with model RDMs.

- Each item gets a *template*, an activity pattern from a phase where it was perceived (the localizer in [K], the pre-manipulation memory test in [B]).
- Later patterns are scored by their Pearson correlation with the templates. The patterns come from the WM task in [K] and from the post-manipulation test in [B].
- Voxels are first restricted to category-selective voxels and then weighted by item-selective (or category-selective) GLM t-values.

The existing config (`conditions.items` → RDM → `model_rdms`) has no slot for templates, target phases, weights, comparison sets or timecourses. So the workflow was extended rather than bent to fit:

| Change | Where |
|---|---|
| New analysis family, selected with `pipeline.analyses: ["template"]` (alongside or instead of `"rdm"`) | `utils/rsa_common.py::default_config`, `workflows/run_pipeline.py` |
| New `template_rsa` config section: `templates`, `feature_weighting`, named `targets`, per-target `group` | same; validated by `utils/validate_rsa_config.py::validate_template` |
| Time windows use mvpa-decoding's syntax: `{"start"/"end": {"reference": "onset"/"offset_end", "offset_seconds"}}` | `rsa_common.py::validate_window`, `window_volumes` |
| Atlas-label ROIs: `{"atlas_pattern", "labels"}` (e.g. fMRIPrep `aparcaseg`), resampled to the BOLD grid | `rsa_common.py::resolve_roi_mask` (used by both analysis families) |
| Subject stage, group stage and report pages for template RSA | `template_rsa.py`, `group_template_rsa.py`, `generate_report.py::template_pages` |

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

- `feature_weighting.item_contrast` takes `all_others`, `within_category` or `within_subcategory`.
- `feature_weighting.levels` takes `item`, `category` and/or `none`, and every level listed is computed side by side.
- Both GLMs are fit as **one concatenated model** across runs. Item and category regressors are shared; drift, intercept, nuisance events and confounds are run-specific. This is how SPM does it, and it handles items seen in only one run [B]. Code: `template_rsa.py::concat_glm`.

### 1.5 Templates

For item $i$ with template events $E_i$ and template window $W_\text{tmpl}$:

- [K]: the 1.38 s image presentation, shifted 4.6 s = 10 TRs.
- [B]: the 2 s presentation, shifted 5 s.

$$T_i = \frac{1}{|E_i|}\sum_{e\in E_i} p_{W_\text{tmpl}}(e),\qquad \tilde T_i = (T_i \odot w_i)\big|_{M_{c(i)}}$$

$\odot$ is the element-wise product. In [B], $|E_i| = 1$: the template is the item's single pre-test exposure.

**Workflow:** `templates.events`, `templates.window` → `template_rsa.py::build_template_sets`.

### 1.6 Scoring a target pattern

A target pattern $y\in\mathbb R^V$ is either a single volume (timecourse) or a window mean. It is weighted by the *same* item's weights before correlating *(from code: `ytarg_pat .* xbeta`)*:

$$r_i(y) = \operatorname{corr}\!\big(\tilde T_i,\ (y\odot w_i)|_{M_{c(i)}}\big)$$

For a set of items $A$, the score averages Pearson r first and then Fisher-transforms *(from code)*:

$$S_A(y) = \operatorname{artanh}\!\Big(\tfrac{1}{|A|}\textstyle\sum_{i\in A} r_i(y)\Big)$$

The sets, for a trial whose item is $i^\*$:

- **target** $= \{i^\*\}$
- **related** $= \{j\neq i^\*: c(j)=c(i^\*)\}$, other items of the same category
- **others** $= \{j: c(j)\neq c(i^\*)\}$, items of other categories
- Optional named sets from other event columns (`extra_item_columns`), e.g. a specific replacement item.

**Workflow:** `targets.<set>.related_level` (category or subcategory), `feature_weighting.weight_target_patterns` → `template_rsa.py::comparison_sets`, `score_matrix` (vectorized over all volumes of a run), `TemplateSet.score`.

### 1.7 Group inference

Trials are averaged within subject before any t-test. Mixed models use every trial.

**Timecourse [K, Fig. 4].** For trial $n$ with onset $o_n$:

- Score the unshifted volumes $k=0,\dots,29$ from trial onset (30 TRs = 13.8 s): $S^{(n)}_A(k)$.
- Average per subject × condition × $k$.
- Baseline: subtract each curve's mean over $k\in[0,6)$, the first 2.76 s.
- Removal relative to maintenance: $\Delta_\text{op}(k) = S_\text{op}(k) - S_\text{maintain}(k)$.
- Test window $k = 6..20$ (TRs 7–21), split into five 3-TR blocks. In each block:
  - one-sample t-test vs 0 per condition (removal onset = first block below 0)
  - repeated-measures ANOVA across conditions
  - pairwise paired t-tests, FDR-corrected

**Workflow:** `targets.study_timecourse.group` (`baseline_trs: [0, 6]`, `reference: maintain`, `window_trs: [6, 20]`, `block_trs: 3`) → `group_template_rsa.py::timecourse_stats`.

**Encoding fidelity [K, Fig. 5].** Encoding fidelity $\mathrm{EF}_n = S_{\{i_n\}}(p_W(n))$, with $W$ = the 2.76 s image presentation shifted 4.6 s. Two analyses use it:

- *Sensitivity check:* target > related > others.
- *Proactive interference:* for each operation on trial $N$, compare $\mathrm{EF}_{N+1}$ when item $N+1$ is from the same category as item $N$ vs. a different one, giving a same − different score per operation.
  - Replace trials whose replacement category equals item $N+1$'s category are removed.
  - Tests: a one-sample t-test per operation, plus a repeated-measures ANOVA across operations.

**Workflow:** `previous_trial_columns: ["category", "condition"]` creates `prev_*` and `same_*` columns. `previous_trial_matches.same_replacement` compares trial N's `replace_category` with trial N+1's `category`. The contrast `same_minus_different_category` uses `split_by: prev_condition`, `anova: true`, and a `filter` that drops the replace-overlap trials. Code: `group_template_rsa.py::window_stats`.

**Representational fidelity [B, Fig. 6].** Fidelity $F_i = S_{\{i\}}(p_W(e^\text{post}_i))$ is the post-test pattern scored against the pre-test template, at both the item and category weighting levels. Each item carries:

- its operation from the manipulation phase
- its memory outcome on the post-test: remembered = "sure old" hit, forgotten = any other response

Three analyses:

1. *Validation:* in the manipulation phase, target > other items.
2. *Paired t-test* of subject means, remembered vs. forgotten, within each operation. Only subjects with both outcomes enter, which gave n = 18 in [B].
3. *Mixed model* per operation: $F \sim \text{memory} + (1+\text{memory}\mid\text{subject})$ (statsmodels `mixedlm`). The paper also reports bootstrap 95% CIs.

The same analysis is repeated in the VVS, PFC, parietal and hippocampus ROIs.

**Workflow:**

- `targets.post_fidelity`, with `item_attributes` copying `condition` from the study phase by item.
- `contrasts.remembered_vs_forgotten` (`split_by: condition`, subject-bootstrap CI over 10,000 resamples).
- `mixed_models.fidelity_by_memory`.
- `template_rsa.rois` lists all four ROIs.

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
| Item weights | SPM LSA, item vs. 53 others, t-values | `item_contrast: all_others`, `levels: [item]` | ✅ |
| Templates | z-scored, 3 presentation TRs + 10-TR shift, averaged over 5 runs, × weights | `templates.window` onset + 4.6 → offset + 4.6 | ✅ |
| Localizer validation (Fig. 3b) | Correlations among the 5 repetitions × 6 exemplars per subcategory; same > related > others | `validation` (leave-one-event-out templates); RDM path's pattern-level RDM for the matrix view | ≈ cross-validated variant |
| Target weighting; mean r → z | `ytarg .* xbeta`; mean r over set, then Fisher z | `weight_target_patterns: true`; `S_A` | ✅ |
| Item-level timecourse (Fig. 4b) | 30 unshifted TRs, baseline first 6 TRs, minus maintain, 5 × 3-TR blocks, one-sample t, rm-ANOVA, pairwise FDR | `targets.study_timecourse` | ✅ |
| Encoding fidelity check (Fig. 5a) | target > related > others; rm-ANOVA, Tukey–Kramer | `encoding_fidelity` comparison contrasts (paired t, FDR) | ≈ paired t + FDR instead of Tukey–Kramer |
| N+1 proactive interference (Fig. 5b) | same − different EF by operation; replace-overlap trials removed; rm-ANOVA | `previous_trial_columns`, `previous_trial_matches`, contrast + `filter` + `anova` | ✅ |
| Trial-balanced bootstrap (15 trials, 1000 iterations) | supplementary | — | ✗ (subject bootstrap of the contrast instead) |
| Category / subcategory classifiers, operation classifiers, importance maps, between-subject and hyperaligned classification, sliding-window classifiers | MVPA | mvpa-decoding | MVPA |
| Univariate FEAT contrasts | FSL | — | ✗ (out of scope) |

### Bretton et al. (2024)

| Step | Paper | Workflow (config → code) | Status |
|---|---|---|---|
| Preprocessing | fMRIPrep 21.0.1, MNI152NLin2009cAsym, TR 1 s; fMRIPrep confounds "used to clean the data" (exact set not stated) | `bold_glob` in MNI; `confounds` motion (24) + WM/CSF + cosines; `timeseries.regress_confounds: true` | ≈ confound set assumed |
| ROIs | VVS, PFC, parietal, hippocampus from each participant's parcellation (Desikan-Killiany) in MNI, summed and binarized | `rois.*` = `atlas_pattern` (aparcaseg) + `labels` | ≈ VVS label list needs lab confirmation |
| Category mask + category weights | GLM on pre-test, scene vs. face t, p < .05, k = 10 | `category_mask.events` = all pre-test trials; `levels` includes `category` | ✅ |
| Item weights | LSA, each of 90 items vs. the 89 others | `item_contrast: all_others`, concatenated GLM (items in one run each) | ✅ |
| Templates | pre-test pattern, single exposure | `templates.events` = pre-test manipulated scenes, `window` + 5 s | ✅ (validation skipped: single exposures) |
| Template check on manipulation phase | target > all other items | `targets.manipulation_validation` (target vs. related, i.e. other scenes) | ✅ |
| Fidelity, remembered vs. forgotten | per operation, paired t on subject means (n = 18 with both outcomes) | `post_fidelity` contrast (subjects with both) | ✅ |
| Mixed model | (1 + memory \| participant), per operation | `mixed_models` (statsmodels) | ✅ |
| Bootstrap CIs | 10,000 bootstrap resamples of participants | subject bootstrap in contrasts (`n_bootstrap: 10000`) | ≈ CI of paired difference, not of a pooled trial-level regression |
| ROI specificity (PFC, parietal, hippocampus) | same analysis per ROI | `template_rsa.rois` | ✅ |
| Category-evidence classifier, operation classifier, engagement-variance and engagement → memory regressions | MVPA | mvpa-decoding | MVPA |

---

## 3. Validation status

- **Synthetic** ([tests/test_template_rsa.py](../tests/test_template_rsa.py)). A dataset built like both designs: localizer, maintain/suppress study phase, and post-test with remembered/forgotten items. Checks:
  - category masks split the ROI correctly
  - target > related > others (validation and post-test)
  - suppress minus maintain < 0 late in the trial
  - remembered > forgotten in both the contrast and the mixed model
  - previous-trial columns, the rm-ANOVA path, single-exposure templates, report pages, atlas-label ROIs
- **Haxby 2001** ([tutorial/config-haxby-template.example.json](../tutorial/config-haxby-template.example.json); categories stand in for items):
  - item-weighted templates give held-out target − others = 0.54 Fisher z (unweighted: 0.30), n = 6, both p < .001
  - the timecourse peaks about 5–20 s after block onset
  - 3–4 of the 8 category masks per subject fell below 10 voxels with an 8-way contrast and fell back to the whole ROI (a guard that is not part of the published method; see §4.5)
- **Not yet validated:** reproducing published numbers from either paper. That needs the lab's data. The first run on real data should compare against [K] Fig. 3b / 5a (target ≈ 0.17, related ≈ 0.13, others ≈ 0.02 Fisher z) and [B]'s manipulation-phase check.

## 4. Assumptions and open questions for the lab

1. **Events schema.** Both configs assume these columns, and it is easiest to add them to the events.tsv files:
   - `stimulus`, `category`, `subcategory`
   - `condition` (operation)
   - `replace_category` ([K])
   - `item_type` (manipulated scenes, [B])
   - `memory` ([B])

   Otherwise, edit the queries in the configs.
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

The command runs the event table, validation, template RSA for every subject, group stats and the report. The stages can also be run separately:

```bash
python workflows/template_rsa.py --config CONFIG --subject 01
python workflows/group_template_rsa.py --config CONFIG
python workflows/generate_report.py --config CONFIG
```

Outputs:

- `<output_dir>/sub-<s>/template_rsa/`: trial-level scores, category masks, feature-selection QC, template validation
- `<output_dir>/group/template_rsa/`: subject means, timecourses, block stats, contrasts, mixed models, figures
