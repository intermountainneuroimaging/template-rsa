#!/usr/bin/env python3
"""Rebuilds docs/code_walkthrough.md: the commentary below plus code extracted from the current
source of rsaworkflow and the sibling betaestimation project (checked out next to this repo).

A line '@@ path | function | [[start_marker, end_marker], ...]' becomes that function's source --
or the slices from the first line containing start_marker to the next line containing
end_marker -- with its real line numbers. Run after changing the code:

    python docs/build_code_walkthrough.py
"""
import ast
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECTS = os.path.dirname(REPO)
OUT = os.path.join(REPO, "docs", "code_walkthrough.md")


def func_lines(path, qualname):
    src = open(os.path.join(PROJECTS, path)).read().splitlines()
    tree = ast.parse("\n".join(src))
    parts = qualname.split(".")
    nodes = tree.body
    node = None
    for part in parts:
        node = next(n for n in nodes if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == part)
        nodes = node.body
    start = (node.decorator_list[0].lineno if getattr(node, "decorator_list", None) else node.lineno)
    return src, start, node.end_lineno


def snippet(path, qualname, ranges=None):
    src, start, end = func_lines(path, qualname)
    if not ranges:
        chunks = [(start, end)]
    else:
        chunks, pos = [], start
        for a, b in ranges:
            s = next(i for i in range(pos, end + 1) if a in src[i - 1])
            e = next(i for i in range(s, end + 1) if b in src[i - 1]) if b else end
            chunks.append((s, e))
            pos = e + 1
    out = []
    for k, (s, e) in enumerate(chunks):
        if k:
            indent = len(src[s - 1]) - len(src[s - 1].lstrip())
            out.append(" " * indent + "# ...")
        out.extend(src[s - 1:e])
    return {"path": path, "start": chunks[0][0], "end": chunks[-1][1], "code": "\n".join(out)}


DOC = r'''
# Code walkthrough: betaestimation and rsaworkflow

The methodological core of both projects, as code. Each block below is copied from the current source; the heading link opens the exact lines. The narrative goes in the order the data flows:

**fMRIPrep output → nuisance model → patterns or cleaned time series (betaestimation) → event table → RDM analysis or template RSA (rsaworkflow) → group table → post-hoc notebooks.**

Formal definitions of the lab's template method are in [lab_methods.md](lab_methods.md); this document shows where each piece lives in code and which choices it encodes. Points that a reviewer should know about are marked **▲ Note**.

---

# Part A — betaestimation (shared with mvpa-decoding)

## A1. The nuisance model for one run

Everything that is "noise" for a run is decided once, here, and then reused by every estimation method: which confound regressors enter, which volumes are censored, and whether censored volumes are removed or modeled.

### Where the confound list comes from

The confounds are chosen in the config, under `pattern_estimation.confounds`; `censoring` sits beside it. [`utils/preprocessing.py`](../utils/preprocessing.py) passes both dicts to `run_nuisance` unchanged. If `confounds` is omitted, no confound regressors are added. The `source` key picks between two ways of specifying them.

**`"source": "fmriprep"` (the default): name confound groups, not columns.** Every other key goes straight to nilearn's `load_confounds`, which picks the matching columns from the run's `desc-confounds_timeseries.tsv`. From [`examples/config-kim2020-ncomms.example.json`](../examples/config-kim2020-ncomms.example.json):

```json
"confounds": {
  "source": "fmriprep",
  "strategy": ["motion", "scrub"],
  "motion": "basic",
  "fd_threshold": 0.9,
  "std_dvars_threshold": 1000,
  "scrub": 0
},
"censoring": {"drop_initial_volumes": 10, "mode": "censor"}
```

- `strategy` lists the groups to include: `motion`, `wm_csf`, `global_signal`, `compcor`, `high_pass`, `scrub`, `non_steady_state`, `ica_aroma`.
- Each group has its own setting for how many columns it adds:
  - `motion` and `wm_csf` can be `basic`, `power2`, `derivatives` or `full`. For motion these give 6 to 24 columns.
  - `compcor` and `n_compcor` control the CompCor components (see below).
- `scrub`, `fd_threshold` and `std_dvars_threshold` add no regressors. They decide which volumes are flagged, and `censoring.mode` then decides what happens to those volumes.

**CompCor: `compcor` vs. `n_compcor`.** Both settings take effect only when `"compcor"` is in `strategy`.

`compcor` picks the kind of component, which is set by the noise mask it came from:

| `compcor` | fMRIPrep columns | mask |
|---|---|---|
| `"anat_combined"` (nilearn's default) | `a_comp_cor_*` | white matter + CSF together |
| `"anat_separated"` | `w_comp_cor_*`, `c_comp_cor_*` | white matter and CSF separately |
| `"temporal"` | `t_comp_cor_*` | highest-variance voxels |
| `"temporal_anat_combined"` / `"temporal_anat_separated"` | both sets | both |

`n_compcor` sets how many components of that kind to take:
- An integer takes the first N, which explain the most variance.
- `"all"` (the default) takes every component fMRIPrep wrote. That is however many explain 50% of the mask's variance, so the count varies across runs.
- With `"anat_separated"`, N applies to each mask.

A fixed integer keeps the number of nuisance regressors, and so the GLM's degrees of freedom, comparable across runs and subjects:

```json
"confounds": {
  "source": "fmriprep",
  "strategy": ["motion", "high_pass", "compcor"],
  "compcor": "anat_combined",
  "n_compcor": 5
}
```

fMRIPrep computes CompCor after high-pass filtering, so `high_pass` must be in the strategy with it; `validate_nuisance` makes that an error. CompCor and `wm_csf` both model white-matter/CSF signal, so they are partly redundant, and most pipelines use one or the other.

**`"source": "file"`: list exact columns, regexes or both, from any TSV.** `{subject}`, `{session}`, `{task}` and `{run}` in `file_pattern` are filled in per run. A named column that is missing raises an error instead of being skipped:

```json
"confounds": {
  "source": "file",
  "file_pattern": "/data/derivatives/fmriprep/sub-{subject}/func/sub-{subject}_task-{task}_run-{run}_desc-confounds_timeseries.tsv",
  "columns": ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z", "csf", "white_matter"],
  "regex": ["a_comp_cor_0[0-4]"]
}
```

**Where the regression happens.** `run_nuisance` only builds the confound matrix:
- **GLM path:** the confounds become extra design-matrix columns, fit at the same time as the task regressors (A5–A7).
- **Volume-average path and the z-scored template-RSA series:** `clean_run` regresses them out when `regress_confounds` is on (A3).

`validate_nuisance` also warns when `high_pass` is used together with the GLM's own `drift_model`, because the cosines would be modeled twice.

### The code

@@ betaestimation/betaestimation/nuisance.py | run_nuisance | [["censoring = censoring or {}", "return RunNuisance"]]

- `load_confounds` reads fMRIPrep's `desc-confounds_timeseries.tsv` for the strategy in the config and, when the strategy includes `scrub` / `non_steady_state`, also returns the **sample mask** (indices of volumes to keep): FD / std-DVARS outliers and fMRIPrep's non-steady-state volumes.
- `drop_initial_volumes` intersects that mask with "everything after the first N volumes" (e.g. Kim et al.'s 10 dropped TRs).
- **Censor vs. regress.** With `mode: "censor"` the mask travels on (volumes are removed from every fit). With `mode: "regress"` each flagged volume gets a one-hot spike column instead and the mask is dropped, so the series stays continuous (some prefer this with AR(1) prewhitening, whose noise model assumes regular sampling).
- Confounds are zero-filled (fMRIPrep's derivative columns start with NaN) and checked against the run length.

## A2. Volume math (identical to mvpa-decoding)

Every "which volumes does this event cover" question goes through one function:

@@ betaestimation/betaestimation/timeseries.py | compute_volume_range

The count of volumes comes from the **duration**, rounded to whole TRs, not from flooring the start and ceiling the end — that would add a volume to almost every event, because real onsets rarely fall on a TR boundary. `event_volumes` and `window_volumes` (mvpa-decoding's `{"start"/"end": {"reference", "offset_seconds"}}` windows) are thin wrappers that add the hemodynamic lag or window offsets first.

## A3. Cleaning a run without moving its volumes

TR-wise analyses (volume averaging, template RSA, TR-wise decoding) need a cleaned, z-scored series in which **volume k is still volume k** after censoring:

@@ betaestimation/betaestimation/timeseries.py | clean_run

`signal.clean` with `sample_mask` excludes censored volumes from the confound regression and the z-scoring; when a band-pass is requested, nilearn first interpolates across them so the filter (and detrend) see a continuous series, then drops them. The result is written back into a full-length array with **NaN rows at censored volumes**. Downstream code averages with `nanmean`, so a censored TR simply drops out of an event's window, and an event whose window is entirely censored is skipped.

GLMs get a different copy of the same data — kept volumes only, mean-scaled exactly as nilearn's `FirstLevelModel` does:

@@ betaestimation/betaestimation/timeseries.py | glm_ready

## A4. Masks

@@ betaestimation/betaestimation/masks.py | brain_mask

All runs of a subject share **one** mask (the intersection of fMRIPrep's per-run brain masks, if several match), so every pattern of that subject has the same voxels — a requirement for comparing patterns across runs.

@@ betaestimation/betaestimation/masks.py | roi_mask

Atlas ROIs are the union of integer labels (e.g. the Desikan-Killiany ids in fMRIPrep's `aparcaseg`), resampled with nearest-neighbour interpolation onto the BOLD grid, so a 1 mm anatomical parcellation can be applied to 2.4 mm functional data.

## A5. One pattern per condition, or per trial (LSA)

`estimate_run_patterns` takes an events table where `label` marks the events of interest (NaN = not of interest) and turns labels into regressor names:

@@ betaestimation/betaestimation/estimate.py | _regressor

@@ betaestimation/betaestimation/estimate.py | estimate_run_patterns | [["glm = dict(glm or {})", "yield PatternEstimate(first"]]

- **Unlabeled events are nuisance regressors** (`model_unlabeled_events`, default on): e.g. probes or cues get their own regressor by `trial_type` so their response does not end up in the implicit baseline, which would bias every condition's beta.
- `unit="condition"`: one regressor per label pooling all its events in the run; `unit="trial"` + `trial_model="lsa"`: one regressor per event in the same GLM (least-squares-all).
- Each pattern is a unit contrast on its column, returned as a t, z or beta map.

The fit itself is a plain nilearn first-level model with the run's confounds and sample mask:

@@ betaestimation/betaestimation/glm.py | fit_first_level

## A6. Single-trial estimates by least-squares-separate (LSS)

@@ betaestimation/betaestimation/glm.py | lss

@@ betaestimation/betaestimation/estimate.py | _lss

- One GLM **per target trial**: the target is the only regressor of interest; the others are modeled by condition (`lss_other: "condition"`, Mumford's LSS-N) or lumped into one regressor (`"single"`, the original LSS).
- The run is masked, smoothed and scaled **once**; each trial only builds a design matrix and runs `run_glm` on the already-masked data.
- **Censoring is applied after building the design** on the full timeline (`design.values[keep]`), so the HRF convolution of every regressor sees the true event timing — the same order of operations as `FirstLevelModel`.
- ▲ Note: when every condition occurs exactly once per run, LSS-N's design *is* the per-condition design (and the LSA design), and they give identical maps (verified on Haxby and in `tests/test_betaestimation.py`). LSS-N differs from LSA only when a condition repeats within a run — the case it exists for, since LSA's betas for closely spaced, collinear trials of the same condition get noisy. `lss_other: "single"` differs as soon as a run has three or more events.

## A7. Volume averaging (no GLM)

@@ betaestimation/betaestimation/estimate.py | _volume_average | [["data = clean_run(", "per_event.append("]]

The cleaned series (A3) is averaged over each event's volumes after shifting by `hemodynamic_lag` — mvpa-decoding's approach to "the pattern for an event". With `unit="condition"` the events of each label are then averaged within the run.

## A8. One GLM across several runs (SPM-style multi-session)

Template RSA needs item-vs-others contrasts where an item may appear in only one run (Bretton et al.: each scene is shown once in the pre-test). A per-run GLM cannot estimate that contrast, so the runs are stacked into one model:

@@ betaestimation/betaestimation/glm.py | fit_concatenated

- Columns whose name starts with `shared_prefix` (`item_…`, `cat_…`) are **shared across runs**; every other column — nuisance events, cosine drifts, the intercept, confounds — is renamed `run<k>_…` and therefore **run-specific** (block-diagonal), as SPM does for sessions.
- `fillna(0)` makes a run's design zero for items it never shows.
- Censored rows are dropped per run before stacking.
- ▲ Note: `run_glm` estimates the AR(1) noise model on the stacked series, so a few lag pairs straddle run boundaries; SPM estimates serial correlations per session. The effect on the t-maps is small (one boundary per run), and `noise_model: "ols"` avoids it entirely.

@@ betaestimation/betaestimation/glm.py | contrast

@@ betaestimation/betaestimation/glm.py | versus_others

`versus_others` builds the "item vs. the mean of the other items" weights used for item-selective t-maps.

## A9. Cluster-extent filtering

@@ betaestimation/betaestimation/masks.py | cluster_filter

Thresholded maps keep only 26-connected clusters of at least `min_size` voxels (the papers' "cluster correction, voxel extent = 10").

---

# Part B — rsaworkflow: from events.tsv to inputs

## B1. The event table

@@ rsaworkflow/workflows/generate_master_spreadsheet.py | process_events_file | [["excluded = events", "events[\"eventfile\"] = events_path"]]

One row per event, every events.tsv column kept. Fixation / block markers / post-RT rows are dropped **before** `trial_index` is assigned, so trial numbers count only real events. `run_id` (session_task_run) identifies runs across tasks; it is the cross-validation fold label later.

## B2. Which events are items

@@ rsaworkflow/utils/rsa_common.py | label_items

The `conditions.items` queries (the same and/or/not query language as mvpa-decoding) assign each event to at most one RDM item, first match wins; `include` removes events entirely (they become neither items nor nuisance). The validator warns about events that match several items.

**Items from a column.** When the item is already a column of events.tsv (an exemplar, stimulus or condition name), set `conditions.item_column` instead of writing `items`. Every distinct value of that column, among the events `include` keeps, becomes an item. Events with no value (`n/a`, empty) are not items. `conditions.attribute_columns` reads each item's attributes, for the categorical model RDMs, from other columns:

```json
"conditions": {
  "include": {"column": "trial_type", "match": "exact", "value": "stim"},
  "item_column": "exemplar",
  "attribute_columns": ["category", "subcategory"]
},
"model_rdms": {
  "category":    {"type": "categorical", "attribute": "category"},
  "subcategory": {"type": "categorical", "attribute": "subcategory"}
}
```

**Item column vs. attribute columns: a worked example.** Take four exemplars with a single attribute, `category`:

```
onset  trial_type  exemplar  category
10     stim        dog01     animal
14     stim        cat02     animal
18     stim        whisk02   tool
22     stim        hammer01  tool
...    (repeated across runs)
```

*The item column gives the data.* `item_column: "exemplar"` makes one beta map per exemplar. The RDM step compares every pair of those patterns and produces the **data RDM**, which says how differently the region responded to each pair (illustrative numbers):

|          | dog01 | cat02 | whisk02 | hammer01 |
|---|---|---|---|---|
| dog01    | 0   | 0.3 | 0.9 | 0.8 |
| cat02    | 0.3 | 0   | 0.8 | 0.9 |
| whisk02  | 0.9 | 0.8 | 0   | 0.4 |
| hammer01 | 0.8 | 0.9 | 0.4 | 0   |

This is only a measurement. On its own it says nothing about categories.

*The attribute columns give the hypotheses.* To ask whether the region represents animal vs. tool, you need a prediction of what the RDM would look like if it did. `attribute_columns: ["category"]` together with a `categorical` model RDM supplies that prediction: same category = 0 (similar), different category = 1 (dissimilar):

|          | dog01 | cat02 | whisk02 | hammer01 |
|---|---|---|---|---|
| dog01    | 0 | **0** | 1 | 1 |
| cat02    | **0** | 0 | 1 | 1 |
| whisk02  | 1 | 1 | 0 | **0** |
| hammer01 | 1 | 1 | **0** | 0 |

The comparison step (C3) correlates the two RDMs. Here they agree: dog–cat is low (0.3) where the model says 0, and dog–whisk is high (0.9) where it says 1. So the region's patterns are organized by category.

- **The item column** says *which patterns to estimate*: the rows of the data RDM.
- **An attribute column** says *which of those patterns belong together*: one model RDM per attribute.

The exemplar name alone carries no grouping. Nothing in `"dog01"` says it goes with `"cat02"` rather than `"whisk02"`, and the attribute column is where you supply that. Adding `subcategory` (mammal / bird / kitchen / workshop) gives a second, finer model, tested against the same data RDM. That lets you ask whether the region tracks only the coarse category or the subcategory too. Attributes never change which betas are estimated. To get one beta per category instead, make `category` the item column.

@@ rsaworkflow/utils/rsa_common.py | expand_conditions

- Items are ordered by the attribute columns, then by name, so each category forms a block in the RDM figures.
- An exemplar whose events disagree on an attribute (listed under two categories) is an error, not a silent first-value pick.
- The expansion runs on the whole event table, so every subject gets the same item list and RDM order.
- `items` and `item_column` are mutually exclusive. Hand-written `attributes` can still add or override values for generated items.

▲ **Note — name collisions.** Regressors and pattern files are named by `safe_name(item)`, which keeps only letters and digits, so `dog_1` and `dog1` would be fit as one regressor. The validator rejects such item sets, and betaestimation's `estimate_run_patterns` refuses colliding labels too.

## B3. Handing runs to betaestimation

@@ rsaworkflow/workflows/estimate_patterns.py | run_events

@@ rsaworkflow/workflows/estimate_patterns.py | estimate_subject | [["for boldfile, run_df in sub_df.groupby", "trial_tag = "]]

The item becomes betaestimation's `label` and `trial_index` its `event_id`; everything methodological from here to the pattern images is Part A.

---

# Part C — rsaworkflow: the RDM analysis

## C1. Pattern matrix per ROI

@@ rsaworkflow/workflows/rsa_workflow.py | roi_rsa | [["masker = NiftiMasker(", "rdm = compute_data_rdm("]]

@@ rsaworkflow/utils/rsa_common.py | clean_measurements

Voxels that are non-finite or constant across all patterns (outside the brain mask, or masked out by the GLM) are dropped, so they cannot produce NaN correlations or zero-variance distances.

## C2. Distances

@@ rsaworkflow/utils/rsa_common.py | compute_data_rdm

- `descriptor="item"`: rsatoolbox averages each item's patterns (all runs) before computing correlation or Euclidean distances.
- **Crossnobis** (`cv_descriptor="run"`): distances are computed between run-wise pattern differences from *different* runs, so noise that is independent across runs cancels in expectation — an unbiased distance whose expected value is 0 for identical patterns.
- Mahalanobis / crossnobis whiten with a noise precision estimated from the patterns' residuals around their item means (`prec_from_measurements`, shrinkage towards the diagonal by default). ▲ Note: the precision is estimated from the same patterns the distances use (standard in rsatoolbox examples); with few runs, a residual-based estimate from the first-level GLMs would be the stricter alternative.
- `calc_rdm` sorts items alphabetically; the RDM is reordered to the config's item order so it matches the model RDMs.

The pattern-level view (every pattern its own row) uses a non-cross-validated distance:

@@ rsaworkflow/utils/rsa_common.py | compute_pattern_rdm

## C3. Model RDMs and their comparison with the data

@@ rsaworkflow/utils/rsa_common.py | build_model_rdms | [["for name, spec in cfg[\"model_rdms\"].items():", "vectors.append(mat)"]]

@@ rsaworkflow/workflows/rsa_workflow.py | roi_rsa | [["present = list(rdm.pattern_descriptors", "sims = rsatoolbox.rdm.compare("]]

Categorical models are 0 within and 1 between attribute values — they predict **tied ranks**, which is why the main Haxby config compares with Kendall's τ-a (Nili et al., 2014) rather than Spearman's ρ (the crossnobis config uses cosine similarity, appropriate for ratio-scale distances).

## C4. Group inference

@@ rsaworkflow/workflows/group_rsa.py | noise_ceiling

Leave-one-subject-out noise ceiling: the **upper** bound correlates each subject with the mean RDM *including* that subject (over-fits), the **lower** bound with the mean of the *other* subjects (a true held-out prediction). A model reaching the lower bound explains as much as the between-subject consistency allows.

@@ rsaworkflow/workflows/group_rsa.py | random_effects_stats | [["for model in sims.columns:", "per_model[\"p_corrected\"] = correct("]]

Subjects are the random effect: one similarity per subject × model, a one-sample test against 0 (one-sided by default), FDR across models within the ROI.

@@ rsaworkflow/workflows/group_rsa.py | crossval_stats | [["if cv_cfg[\"scheme\"] == \"k_fold\":", "nc = np.asarray("]]

Weighted models have free weights, so they are only ever scored on **held-out subjects**: weights are fit on the training subjects' RDMs and the fitted model is compared with the left-out subject's RDM; fixed models are scored in the same folds so the comparison is fair.

---

# Part D — rsaworkflow: template RSA (Kim et al., 2020; Bretton et al., 2024)

## D1. Two copies of each run

@@ rsaworkflow/workflows/template_rsa.py | load_run

Every run in the template phase, the target phases and the category-mask events is loaded once per ROI, into a **z-scored TR series with NaN rows at censored volumes** (patterns and scoring) and a **mean-scaled, censored copy** (the feature-selection GLMs) — both from Part A.

## D2. Category masks

@@ rsaworkflow/workflows/template_rsa.py | feature_selection | [["if need_category_glm and len(categories) > 1:", "category_masks[c] = mask"]]

- One concatenated GLM (A8) over the runs of the category-mask events, with one shared regressor per category (optionally merged into mini-blocks, Kim et al.'s 14.26 s boxcars); all other events in those runs are nuisance regressors.
- For each category, category vs. the mean of the others → **t map** (kept as the category-level weights) and **z map**, thresholded one-sided at `p_threshold` and cluster-filtered (A9).
- ▲ Note: if fewer than `min_voxels` survive, the whole ROI is used and the fallback is recorded — not part of the published method (see lab_methods.md §4.5).

## D3. Item weights

@@ rsaworkflow/workflows/template_rsa.py | feature_selection | [["if \"item\" in fw[\"levels\"]:", "item_t[row[\"item\"]] = contrast_stat("]]

Least-squares-all over the template runs: one shared regressor per template item, and each item's weights are the t-values of **item vs. the mean of the other items** (or only those of its category / subcategory). Template-phase events that are *not* template events — e.g. Bretton's `old_<item>` scenes — fall into `glm_events`' nuisance branch, each with its own regressor because their `trial_type` is unique:

@@ rsaworkflow/workflows/template_rsa.py | glm_events

## D4. Templates

@@ rsaworkflow/workflows/template_rsa.py | build_template_sets | [["occurrences = {}", "templates[item] = np.mean("]]

A template is the item's z-scored pattern — averaged over the volumes of `templates.window` (lag-shifted) and over the item's template events — **restricted to its category's mask and multiplied voxel-wise by its weights**. The three `levels` differ only in `w`: the item's t map, its category's t map, or ones.

## D5. Scoring a target pattern

@@ rsaworkflow/workflows/template_rsa.py | TemplateSet.score

@@ rsaworkflow/workflows/template_rsa.py | score_matrix

- The target pattern is weighted by the **same item's** weights before correlating (from the authors' code: `ytarg_pat .* xbeta`), so template and target live in the same feature space.
- `score_matrix` computes the Pearson r of **every volume of a run** with every template in one vectorized pass; trial timecourses are then just indexing into it. Censored volumes (NaN rows) yield NaN scores.

@@ rsaworkflow/workflows/template_rsa.py | comparison_sets

@@ rsaworkflow/workflows/template_rsa.py | extra_sets

@@ rsaworkflow/workflows/template_rsa.py | score_targets | [["for comp, members in sets.items():", "\"n_items\": len(members), \"r\": r, \"z\": fisher_z(r)})"]]

- A set's score is the **mean Pearson r over its items, then Fisher z** (the authors' order of operations), for `target` (the trial's own item), `related` (other items of its category), `others` (items of other categories) and any `extra_sets` — e.g. every template of the category named in `replace_category`.
- Every events.tsv column is carried into each row (`meta`), which is what lets the post-hoc notebooks define memory outcomes, previous trials or conditions later without touching the core.

## D6. Template validation

@@ rsaworkflow/workflows/template_rsa.py | validate_templates | [["for k, (idx, pat) in enumerate(occ):", "r = np.nanmean(rs) if rs else np.nan"]]

Each template event is scored against its item's template rebuilt **without that event**. ▲ Note: the masks and weights (D2–D3) were still estimated from all template events, including the left-out one — the same mild circularity as the published analysis; a fully held-out check would re-fit the GLMs per fold.

## D7. The group raw table

@@ rsaworkflow/workflows/group_template_rsa.py | group_template_rsa | [["scores = load_scores(output_dir, roi, name, subjects)", "scores.to_csv(os.path.join(raw_dir"]]

@@ rsaworkflow/workflows/group_template_rsa.py | describe | [["scores = scores[scores[\"comparison\"].isin", "mean=(\"z\", \"mean\"), sem=(\"z\", stats.sem)"]]

The core stops here: every subject stacked, plus descriptive subject means. All study-specific inference starts from the raw table.

---

# Part E — post-hoc helpers (`_interactive_notebooks/rsa_posthoc_common.py`)

## E1. Subjects as the random effect, baselines and reference conditions

@@ rsaworkflow/_interactive_notebooks/rsa_posthoc_common.py | subject_means

@@ rsaworkflow/_interactive_notebooks/rsa_posthoc_common.py | baseline_correct | [["a, b = baseline_trs", "return out.drop"]]

@@ rsaworkflow/_interactive_notebooks/rsa_posthoc_common.py | subtract_reference | [["keys = [\"subject\", *by]", "return out.drop"]]

Trials are averaged within subject before any test; baselines are subtracted per subject × condition curve; "operation minus maintain" is matched on subject and TR.

## E2. Previous-trial columns

@@ rsaworkflow/_interactive_notebooks/rsa_posthoc_common.py | add_previous_trial_columns | [["ev = event_table(df).sort_values", "out[name] = np.where("]]

Computed on the event sequence within each run, **before** any exclusion in the notebook, so "the previous trial" is always the one the participant actually saw; `matches` compares a column of trial N with a different column of trial N+1 (Kim et al.: the replacement category of N vs. the category of N+1).

## E3. Tests

@@ rsaworkflow/_interactive_notebooks/rsa_posthoc_common.py | one_sample_by_bin | [["for keys, part in binned.groupby(", "out[\"p_fdr\"] = "]]

@@ rsaworkflow/_interactive_notebooks/rsa_posthoc_common.py | paired_contrast | [["for keys, part in groups:", "lo, hi, p_boot = bootstrap_ci("]]

@@ rsaworkflow/_interactive_notebooks/rsa_posthoc_common.py | mixed_model | [["try:", "ci = fit.conf_int()"]]

- Bin tests: one-sample t per condition × bin, FDR across the bins of each condition.
- Paired contrasts use only subjects with **both** levels (e.g. Bretton's n = 18 with remembered and forgotten items), with a subject-resampling bootstrap CI of the difference.
- Mixed models use every trial with subject as the grouping factor; `re_formula="~C(memory)"` gives random slopes, i.e. (1 + memory | subject).
'''


def render():
    out = []
    for line in DOC.strip("\n").splitlines():
        if not line.startswith("@@ "):
            out.append(line)
            continue
        parts = [p.strip() for p in line[3:].split("|")]
        path, func = parts[0], parts[1]
        ranges = json.loads(parts[2]) if len(parts) > 2 else None
        s = snippet(path, func, ranges)
        repo, rel = path.split("/", 1)
        link = f"../{rel}" if repo == "rsaworkflow" else f"../../{path}"
        out.append(f"[`{path}:{s['start']}`]({link}#L{s['start']}-L{s['end']}) — `{func}`")
        out.append("")
        out.append("```python")
        out.append(s["code"])
        out.append("```")
    text = "\n".join(out) + "\n"
    with open(OUT, "w") as f:
        f.write(text)
    print(f"wrote {OUT} ({len(text.splitlines())} lines)")



if __name__ == "__main__":
    render()
