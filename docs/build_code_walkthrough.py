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

**Vocabulary.** An *event* is one events.tsv row. An *item* is one event of interest (one viewing of one image). An *item* is all repeats of the same stimulus: the finest level of `events.hierarchy`. A *condition* is any coarser level (category, subcategory), pooling the items nested in it. The papers behind Part D say "item" for what is called an item here.

Formal definitions of the lab's template method are in [lab_methods.md](lab_methods.md); this document shows where each piece lives in code and which choices it encodes. Points that a reviewer should know about are marked **▲ Note**.

---

# Part A — betaestimation (shared with mvpa-decoding)

## A1. The nuisance model for one run

Everything that is "noise" for a run is decided once, here, and then reused by every estimation method: which confound regressors enter, which volumes are censored, and whether censored volumes are removed or modeled.

### Where the confound list comes from

The confounds are chosen in the config, under `pattern_estimation.confounds`; `censoring` sits beside it. [`utils/preprocessing.py`](../utils/preprocessing.py) passes both dicts to `run_nuisance` unchanged. If `confounds` is omitted, no confound regressors are added. The `source` key picks between two ways of specifying them.

**`"source": "fmriprep"` (the default): name confound groups, not columns.** Every other key goes to nilearn's `load_confounds` (`min_segment_volumes` is passed as nilearn's `scrub` argument; see below), which picks the matching columns from the run's `desc-confounds_timeseries.tsv`. From [`examples/config-kim2020-ncomms.example.json`](../examples/config-kim2020-ncomms.example.json):

```json
"confounds": {
  "source": "fmriprep",
  "strategy": ["motion", "scrub"],
  "motion": "basic",
  "fd_threshold": 0.9,
  "std_dvars_threshold": 1000,
  "min_segment_volumes": 0
},
"censoring": {"drop_initial_volumes": 10, "mode": "censor"}
```

- `strategy` lists the groups to include: `motion`, `wm_csf`, `global_signal`, `compcor`, `high_pass`, `scrub`, `non_steady_state`, `ica_aroma`.
- Each group has its own setting for how many columns it adds:
  - `motion` and `wm_csf` can be `basic`, `power2`, `derivatives` or `full`. For motion these give 6 to 24 columns.
  - `compcor` and `n_compcor` control the CompCor components (see below).
- `scrub`, `fd_threshold`, `std_dvars_threshold` and `min_segment_volumes` add no regressors. They decide which volumes are flagged, and `censoring.mode` then decides what happens to those volumes.

**Scrubbing: detection vs. what happens next.** The `confounds` and `censoring` sections split the work; neither overrides the other:

| config | role | without it |
|---|---|---|
| `strategy: [..., "scrub"]` + `fd_threshold` / `std_dvars_threshold` | find motion outliers; `load_confounds` returns them as a sample mask (`keep` below) | no motion censoring at all |
| `min_segment_volumes` | also flag short stretches of good volumes between outliers | nilearn's default of 5 applies |
| `censoring.drop_initial_volumes` | add the first N volumes (dummy scans), intersected with the scrub mask | only fMRIPrep's own non-steady-state flags, if `non_steady_state` is in the strategy |
| `censoring.mode` | remove the flagged volumes (`"censor"`) or give each a spike regressor (`"regress"`) | `"censor"` |

**`min_segment_volumes`.** Scrubbing runs in two steps. First, every volume over `fd_threshold` or `std_dvars_threshold` is flagged. Then any run of good volumes shorter than `min_segment_volumes`, stranded between outliers or at a run edge, is flagged too. Example: FD flags volumes 20 and 24:

```
volume:                  18 19 20 21 22 23 24 25 26
FD outlier:                     X           X
min_segment_volumes: 5 → ok ok  X  X  X  X  X ok ok    21–23 is only 3 good volumes (< 5), so it is dropped too
min_segment_volumes: 0 → ok ok  X ok ok ok  X ok ok    only the two outliers are dropped
```

The default of 5 comes from Power et al. (2014), written for resting-state connectivity. There, isolated volumes between motion spikes are likely contaminated, and fragments that short cannot be filtered or correlated reliably. The lab configs use `0`:

- It matches the papers. Kim et al. (2020) censor volumes with FD > 0.9 mm and nothing else; a segment rule would quietly remove more data than the methods report.
- A GLM does not need continuous segments. Each kept volume contributes through the design matrix, so three good volumes between two spikes still carry task signal.
- Template RSA scores events TR by TR. Each extra volume removed is one less TR in an event's window, and a short window can empty entirely.

nilearn calls this setting `scrub`, the same word as the strategy entry that switches detection on. betaestimation renames it `min_segment_volumes` and translates it back (`fmriprep_kwargs`). A config that still uses a `scrub` key is rejected with a message naming the new key. `min_segment_volumes` without `"scrub"` in the strategy has no effect and gives a warning.

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

The fMRIPrep spec reaches nilearn through one translation step, which is where `min_segment_volumes` becomes nilearn's `scrub`:

@@ betaestimation/betaestimation/nuisance.py | fmriprep_kwargs

- `load_confounds` reads fMRIPrep's `desc-confounds_timeseries.tsv` for the strategy in the config and, when the strategy includes `scrub` / `non_steady_state`, also returns the **sample mask** (indices of volumes to keep): FD / std-DVARS outliers and fMRIPrep's non-steady-state volumes.
- `drop_initial_volumes` intersects that mask with "everything after the first N volumes" (e.g. Kim et al.'s 10 dropped TRs).
- **Censor vs. regress.** With `mode: "censor"` the mask travels on (volumes are removed from every fit). With `mode: "regress"` each flagged volume gets a one-hot spike column instead and the mask is dropped, so the series stays continuous (some prefer this with AR(1) prewhitening, whose noise model assumes regular sampling).
- Confounds are zero-filled (fMRIPrep's derivative columns start with NaN) and checked against the run length.

## A2. Volume math (identical to mvpa-decoding)

Every "which volumes does this event cover" question goes through one function:

@@ betaestimation/betaestimation/timeseries.py | compute_volume_range

The count of volumes comes from the **duration**, rounded to whole TRs, not from flooring the start and ceiling the end — that would add a volume to almost every event, because real onsets rarely fall on a TR boundary. `event_volumes` and `window_volumes` (mvpa-decoding's `{"start"/"end": {"reference", "offset_seconds"}}` windows) are thin wrappers that add the hemodynamic lag or window offsets first.

## A3. Cleaning a run without moving its volumes

TR-wise analyses (volume averaging, template RSA, TR-wise fidelity) need a cleaned, z-scored series in which **volume k is still volume k** after censoring:

@@ betaestimation/betaestimation/timeseries.py | clean_run

`signal.clean` with `sample_mask` excludes censored volumes from the confound regression and the z-scoring; when a band-pass is requested, nilearn first interpolates across them so the filter (and detrend) see a continuous series, then drops them. The result is written back into a full-length array with **NaN rows at censored volumes**. Downstream code averages with `nanmean`, so a censored TR simply drops out of an event's window, and an event whose window is entirely censored is skipped.

GLMs get a different copy of the same data — kept volumes only, mean-scaled exactly as nilearn's `FirstLevelModel` does:

@@ betaestimation/betaestimation/timeseries.py | glm_ready

## A4. Masks

@@ betaestimation/betaestimation/masks.py | brain_mask

All runs of a subject share **one** mask (the intersection of fMRIPrep's per-run brain masks, if several match), so every pattern of that subject has the same voxels — a requirement for comparing patterns across runs.

@@ betaestimation/betaestimation/masks.py | roi_mask

Atlas ROIs are the union of integer labels (e.g. the Desikan-Killiany ids in fMRIPrep's `aparcaseg`), resampled with nearest-neighbour interpolation onto the BOLD grid, so a 1 mm anatomical parcellation can be applied to 2.4 mm functional data.

## A5. One pattern per label (pooled), or per item (LSA)

`estimate_run_patterns` takes an events table where `label` marks the items, i.e. the events of interest (NaN = not of interest), and turns labels into regressor names. A label is whatever the items are pooled by: an item, or a condition such as a category:

@@ betaestimation/betaestimation/estimate.py | _regressor

@@ betaestimation/betaestimation/estimate.py | estimate_run_patterns | [["glm = dict(glm or {})", "yield PatternEstimate(first"]]

- **Unlabeled events are nuisance regressors** (`model_unlabeled_events`, default on): e.g. probes or cues get their own regressor by `trial_type` so their response does not end up in the implicit baseline, which would bias every beta.
- `unit="pooled"`: one regressor per label pooling all its items in the run; `unit="item"` + `item_model="lsa"`: one regressor per item in the same GLM (least-squares-all).
- Each pattern is a unit contrast on its column, returned as a t, z or beta map.

The fit itself is a plain nilearn first-level model with the run's confounds and sample mask:

@@ betaestimation/betaestimation/glm.py | fit_first_level

## A6. Item estimates by least-squares-separate (LSS)

@@ betaestimation/betaestimation/glm.py | lss

@@ betaestimation/betaestimation/estimate.py | _lss

- One GLM **per item**: the target item is the only regressor of interest; the other events are modeled by their label or nuisance type (`lss_other: "by_label"`, Mumford's LSS-N) or lumped into one regressor (`"single"`, the original LSS).
- The run is masked, smoothed and scaled **once**; each item only builds a design matrix and runs `run_glm` on the already-masked data.
- **Censoring is applied after building the design** on the full timeline (`design.values[keep]`), so the HRF convolution of every regressor sees the true event timing — the same order of operations as `FirstLevelModel`.
- ▲ Note: when every label occurs exactly once per run, LSS-N's design *is* the pooled design (and the LSA design), and they give identical maps (verified on Haxby and in `tests/test_betaestimation.py`). LSS-N differs from LSA only when a label repeats within a run — the case it exists for, since LSA's betas for closely spaced, collinear items of the same label get noisy. `lss_other: "single"` differs as soon as a run has three or more events.

## A7. Volume averaging (no GLM)

@@ betaestimation/betaestimation/estimate.py | _volume_average | [["data = clean_run(", "per_event.append("]]

The cleaned series (A3) is averaged over each event's volumes after shifting by `hemodynamic_lag` — mvpa-decoding's approach to "the pattern for an event". With `unit="pooled"` the items of each label are then averaged within the run.

## A8. One GLM across several runs (SPM-style multi-session)

Template RSA needs item-vs-others contrasts where an item may appear in only one run (Bretton et al.: each scene is shown once in the pre-test). A per-run GLM cannot estimate that contrast, so the runs are stacked into one model:

@@ betaestimation/betaestimation/glm.py | fit_concatenated

- Columns whose name starts with `shared_prefix` (`exm_…`, `cat_…`) are **shared across runs**; every other column — nuisance events, cosine drifts, the intercept, confounds — is renamed `run<k>_…` and therefore **run-specific** (block-diagonal), as SPM does for sessions.
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

One row per event, every events.tsv column kept. Fixation / block markers / post-RT rows are dropped **before** `event_index` is assigned, so event numbers count only real events. `run_id` (session_task_run) identifies runs across tasks; it is the cross-validation fold label later.

## B2. Each event's role, from the events table

The events.tsv files carry the stimulus hierarchy as columns, coarse to fine, and the config only names them:

```
onset  duration  trial_type  category  subcategory  item
10.0   2.0       stim        animal    mammal       dog01      <- item instance
14.0   1.0       probe       n/a       n/a          n/a        <- nuisance
16.0   8.0       rest        n/a       n/a          n/a        <- baseline
22.0   2.0       stim        animal    mammal       dog01      <- item instance (a repeat of item dog01)
```

```json
"events": {
  "hierarchy": ["category", "subcategory", "item"],
  "item_types": ["stim"],
  "baseline_types": ["rest"]
},
"pattern_estimation": {"level": "item"},
"model_rdms": {
  "category":    {"type": "categorical", "level": "category"},
  "subcategory": {"type": "categorical", "level": "subcategory"}
}
```

@@ rsaworkflow/utils/rsa_common.py | label_events

- **baseline:** `trial_type` in `baseline_types`. Dropped, so it stays in the implicit baseline every beta is measured against.
- **item instance:** a value at every hierarchy level (plus a `trial_type` in `item_types` and passing `item_filter`, when those are set).
- **nuisance:** everything else, e.g. probes, cues, or incorrect trials that fail `item_filter`. Each `trial_type` gets its own regressor (A5).

`pattern_estimation.level` picks what one pattern is: `"item_instance"` (one per event of interest) or a hierarchy level, whose item instances are pooled per run (`"item"`: an item's repeats; `"category"`: all its items).

@@ rsaworkflow/utils/rsa_common.py | check_hierarchy | [["for coarse, fine in zip(levels, levels[1:]):", "errors.append(f\"{lvl} values"]]

- **Nesting** is what makes the hierarchy linear: every value has exactly one parent at the level above (dog01 is always animal / mammal). The validator rejects a file where it doesn't hold, instead of silently picking one parent.
- ▲ **Note — name collisions.** Regressors and pattern files are named by `safe_name(value)`, which keeps only letters and digits, so `dog_1` and `dog1` would be fit as one regressor. The validator rejects such values at every level, and betaestimation's `estimate_run_patterns` refuses colliding labels too.

@@ rsaworkflow/utils/rsa_common.py | resolve_rows | [["items = label_events(df, cfg)", "rows = sorted(parents"]]

The RDM rows are the values of the RDM level (the pattern level, or the items when patterns are item instances), ordered by their parent levels, so each category forms a block in the figures. The rows are resolved on the whole event table, so every subject gets the same row list and order. Each row's parents are what the categorical models read.

**Hierarchy levels vs. the RDM: a worked example.** Take four items under one coarser level, `category`:

```
onset  trial_type  category  item
10     stim        animal    dog01
14     stim        animal    cat02
18     stim        tool      whisk02
22     stim        tool      hammer01
...    (repeated across runs)
```

*The pattern level gives the data.* `level: "item"` makes one beta map per item per run. The RDM step compares every pair of those patterns and produces the **data RDM**, which says how differently the region responded to each pair (illustrative numbers):

|          | dog01 | cat02 | whisk02 | hammer01 |
|---|---|---|---|---|
| dog01    | 0   | 0.3 | 0.9 | 0.8 |
| cat02    | 0.3 | 0   | 0.8 | 0.9 |
| whisk02  | 0.9 | 0.8 | 0   | 0.4 |
| hammer01 | 0.8 | 0.9 | 0.4 | 0   |

This is only a measurement. On its own it says nothing about categories.

*Coarser levels give the hypotheses.* To ask whether the region represents animal vs. tool, you need a prediction of what the RDM would look like if it did. A categorical model at `level: "category"` supplies it: same category = 0 (similar), different category = 1 (dissimilar):

|          | dog01 | cat02 | whisk02 | hammer01 |
|---|---|---|---|---|
| dog01    | 0 | **0** | 1 | 1 |
| cat02    | **0** | 0 | 1 | 1 |
| whisk02  | 1 | 1 | 0 | **0** |
| hammer01 | 1 | 1 | **0** | 0 |

The comparison step (C3) correlates the two RDMs. Here they agree: dog–cat is low (0.3) where the model says 0, and dog–whisk is high (0.9) where it says 1. So the region's patterns are organized by category.

- **The pattern level** says *which patterns to estimate*: the rows of the data RDM.
- **A coarser level** says *which of those rows belong together*: one model RDM per level.

The item name alone carries no grouping. Nothing in `"dog01"` says it goes with `"cat02"` rather than `"whisk02"`; the category column is where that is written down. A subcategory level (mammal / bird / kitchen / workshop) gives a second, finer model tested against the same data RDM, so you can ask whether the region tracks only the coarse category or the subcategory too. Coarser levels never change which betas are estimated; to get one beta per category instead, set `level: "category"`.

**Groupings that cross the hierarchy** (stimulus identity across task phases, a cue status) can't be levels, because they don't nest. They stay as their own events.tsv columns, for `item_filter`, template-RSA queries and post-hoc notebooks, or become an explicit model: `{"type": "categorical", "groups": {row: group}}`. A crossed design whose cells you want as RDM rows (phase × stimulus) gets a column naming the combination (`encoding_0`), nested under the phase.

## B3. Handing runs to betaestimation

@@ rsaworkflow/workflows/estimate_patterns.py | run_events

@@ rsaworkflow/workflows/estimate_patterns.py | estimate_subject | [["for boldfile, run_df in sub_df.groupby", "event_tag = "]]

Each item's value at the RDM level becomes betaestimation's `label` and its `event_index` the `event_id`; `level` decides `unit="pooled"` or `unit="item"`. Everything methodological from here to the pattern images is Part A. The pattern index records the hierarchy columns down to the RDM level for every pattern.

---

# Part C — rsaworkflow: the RDM analysis

## C1. Pattern matrix per ROI

@@ rsaworkflow/workflows/rsa_workflow.py | roi_rsa | [["masker = NiftiMasker(", "rdm = compute_data_rdm("]]

@@ rsaworkflow/utils/rsa_common.py | clean_measurements

Voxels that are non-finite or constant across all patterns (outside the brain mask, or masked out by the GLM) are dropped, so they cannot produce NaN correlations or zero-variance distances.

## C2. Distances

@@ rsaworkflow/utils/rsa_common.py | compute_data_rdm

- `descriptor=level` (e.g. `"item"`): rsatoolbox averages each row's patterns (all runs, or all item instances) before computing correlation or Euclidean distances.
- **Crossnobis** (`cv_descriptor="run"`): distances are computed between run-wise pattern differences from *different* runs, so noise that is independent across runs cancels in expectation — an unbiased distance whose expected value is 0 for identical patterns.
- Mahalanobis / crossnobis whiten with a noise precision estimated from the patterns' residuals around their row means (`prec_from_measurements`, shrinkage towards the diagonal by default). ▲ Note: the precision is estimated from the same patterns the distances use (standard in rsatoolbox examples); with few runs, a residual-based estimate from the first-level GLMs would be the stricter alternative.
- `calc_rdm` sorts rows alphabetically; the RDM is reordered to the hierarchy order (B2) so it matches the model RDMs.

The pattern-level view (every pattern its own row) uses a non-cross-validated distance:

@@ rsaworkflow/utils/rsa_common.py | compute_pattern_rdm

## C3. Model RDMs and their comparison with the data

@@ rsaworkflow/utils/rsa_common.py | build_model_rdms | [["for name, spec in cfg[\"model_rdms\"].items():", "vectors.append(mat)"]]

@@ rsaworkflow/workflows/rsa_workflow.py | roi_rsa | [["present = list(rdm.pattern_descriptors", "sims = rsatoolbox.rdm.compare("]]

Categorical models are 0 within and 1 between values of a coarser level (or of a `groups` mapping) — they predict **tied ranks**, which is why the main Haxby config compares with Kendall's τ-a (Nili et al., 2014) rather than Spearman's ρ (the crossnobis config uses cosine similarity, appropriate for ratio-scale distances).

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

Every run in the template phase and the category-mask events (template stage), or in the target phases (fidelity stage), is loaded once per ROI, into a **z-scored TR series with NaN rows at censored volumes** (patterns and scoring) and a **mean-scaled, censored copy** (the feature-selection GLMs) — both from Part A.

## D2. Category masks

@@ rsaworkflow/workflows/template_rsa.py | feature_selection | [["if need_category_glm and len(categories) > 1:", "category_masks[c] = mask"]]

- One concatenated GLM (A8) over the runs of the category-mask events, with one shared regressor per category (optionally merged into mini-blocks, Kim et al.'s 14.26 s boxcars); all other events in those runs are nuisance regressors.
- For each category, category vs. the mean of the others → **t map** (kept as the category-level weights) and **z map**, thresholded one-sided at `p_threshold` and cluster-filtered (A9).
- ▲ Note: if fewer than `min_voxels` survive, the whole ROI is used and the fallback is recorded — not part of the published method (see lab_methods.md §4.5).

## D3. Item weights (the papers' item weights)

@@ rsaworkflow/workflows/template_rsa.py | feature_selection | [["if \"item\" in fw[\"levels\"]:", "item_t[row[\"item\"]] = contrast_stat("]]

Least-squares-all over the template runs: one shared regressor per template item (the finest `events.hierarchy` level), and each item's weights are the t-values of **item vs. the mean of the other items** (or only those of its category / subcategory, the hierarchy levels named by `category_level` / `subcategory_level`). Template-phase events that are *not* template events — e.g. Bretton's `old_<item>` scenes — fall into `glm_events`' nuisance branch, each with its own regressor because their `trial_type` is unique:

@@ rsaworkflow/workflows/template_rsa.py | glm_events

## D4. Templates

@@ rsaworkflow/workflows/template_rsa.py | build_template_sets | [["occurrences = {}", "templates[item] = np.mean("]]

A template is the item's z-scored pattern — averaged over the volumes of `templates.window` (lag-shifted) and over the item's template events — **restricted to its category's mask and multiplied voxel-wise by its weights**. The three `levels` differ only in `w`: the item's t map, its category's t map, or ones. The templates are built from all template runs and saved, with their masks and weights, for the fidelity stage:

@@ rsaworkflow/workflows/template_rsa.py | save_template_set

## D5. Scoring a target pattern (fidelity stage, `template_fidelity.py`)

@@ rsaworkflow/workflows/template_rsa.py | TemplateSet.score

@@ rsaworkflow/workflows/template_rsa.py | score_matrix

- The target pattern is weighted by the **same item's** weights before correlating (from the authors' code: `ytarg_pat .* xbeta`), so template and target live in the same feature space.
- `score_matrix` computes the Pearson r of **every volume of a run** with every template in one vectorized pass; trial timecourses are then just indexing into it. Censored volumes (NaN rows) yield NaN scores.

@@ rsaworkflow/workflows/template_rsa.py | comparison_sets

@@ rsaworkflow/workflows/template_fidelity.py | extra_sets

@@ rsaworkflow/workflows/template_fidelity.py | score_targets | [["for comp, members in sets.items():", "\"n_items\": len(members), \"r\": r, \"z\": fisher_z(r)})"]]

- A set's score is the **mean Pearson r over its items, then Fisher z** (the authors' order of operations), for `target` (the trial's own item), `related` (other items of its category), `others` (items of other categories) and any `extra_sets` — e.g. every template of the category named in `replace_category`.
- Every events.tsv column is carried into each row (`meta`), which is what lets the post-hoc notebooks define memory outcomes, previous trials or operations later without touching the core.

## D6. Template validation

@@ rsaworkflow/workflows/template_rsa.py | validate_kim2020 | [["PW = np.array(", "pairs.append((\"kim2020\""]]

`kim2020` is the published validation (Kim et al. 2020, `rsa_decode_02_item.m`): every item instance (item × run) against every other, each pattern weighted by **its own** item's weights and both cut to the row item's category mask, Fisher z per pair. ▲ Note: as published, the masks and weights (D2–D3) come from all template runs, including the item instance being validated.

@@ rsaworkflow/workflows/template_rsa.py | resolve_folds

@@ rsaworkflow/workflows/template_rsa.py | validate_cv | [["for fold, train, test in folds:", "rs = {j: ts.score(pat, j) for j in ts.templates}"]]

`cv` (not in the papers; mvpa-decoding's `kfold_cv`) re-fits D2–D4 on the training runs of each fold and scores each held-out item instance against every fold template exactly as a target is scored (D5) — the fully held-out check.

## D7. The group raw table (fidelity group stage)

@@ rsaworkflow/workflows/group_fidelity.py | group_fidelity | [["scores = load_scores(output_dir, roi, name, subjects)", "scores.to_csv(os.path.join(raw_dir"]]

@@ rsaworkflow/workflows/group_fidelity.py | describe | [["scores = scores[scores[\"comparison\"].isin", "mean=(\"z\", \"mean\"), sem=(\"z\", stats.sem)"]]

The core stops here: every subject stacked, plus descriptive subject means. All study-specific inference starts from the raw table.

---

# Part E — post-hoc helpers (`_interactive_notebooks/rsa_posthoc_common.py`)

## E1. Subjects as the random effect, baselines and reference operations

@@ rsaworkflow/_interactive_notebooks/rsa_posthoc_common.py | subject_means

@@ rsaworkflow/_interactive_notebooks/rsa_posthoc_common.py | baseline_correct | [["a, b = baseline_trs", "return out.drop"]]

@@ rsaworkflow/_interactive_notebooks/rsa_posthoc_common.py | subtract_reference | [["keys = [\"subject\", *by]", "return out.drop"]]

Trials are averaged within subject before any test; baselines are subtracted per subject × operation curve; "operation minus maintain" is matched on subject and TR.

## E2. Previous-trial columns

@@ rsaworkflow/_interactive_notebooks/rsa_posthoc_common.py | add_previous_trial_columns | [["ev = event_table(df).sort_values", "out[name] = np.where("]]

Computed on the event sequence within each run, **before** any exclusion in the notebook, so "the previous trial" is always the one the participant actually saw; `matches` compares a column of trial N with a different column of trial N+1 (Kim et al.: the replacement category of N vs. the category of N+1).

## E3. Tests

@@ rsaworkflow/_interactive_notebooks/rsa_posthoc_common.py | one_sample_by_bin | [["for keys, part in binned.groupby(", "out[\"p_fdr\"] = "]]

@@ rsaworkflow/_interactive_notebooks/rsa_posthoc_common.py | paired_contrast | [["for keys, part in groups:", "lo, hi, p_boot = bootstrap_ci("]]

@@ rsaworkflow/_interactive_notebooks/rsa_posthoc_common.py | mixed_model | [["try:", "ci = fit.conf_int()"]]

- Bin tests: one-sample t per operation × bin, FDR across the bins of each operation.
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
