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

[`betaestimation/betaestimation/nuisance.py:84`](../../betaestimation/betaestimation/nuisance.py#L84-L121) — `run_nuisance`

```python
    censoring = censoring or {}
    conf, keep = None, None
    if confounds:
        if confounds.get("source", "fmriprep") == "fmriprep":
            from nilearn.interfaces.fmriprep import load_confounds
            kwargs = {k: (tuple(v) if k == "strategy" else v) for k, v in confounds.items() if k != "source"}
            conf, keep = load_confounds(boldfile, **kwargs)
        else:
            conf = load_file_confounds(confounds, entities)

    drop = int(censoring.get("drop_initial_volumes") or 0)
    if drop:
        keep = np.arange(drop, n_frames) if keep is None else keep[keep >= drop]
    if keep is not None:
        keep = np.asarray(keep, dtype=int)
        if len(keep) == n_frames:
            keep = None

    flagged = np.setdiff1d(np.arange(n_frames), keep) if keep is not None else np.array([], dtype=int)
    if keep is not None and censoring.get("mode", "censor") == "regress":
        spikes = pd.DataFrame(np.eye(n_frames)[:, flagged], columns=[f"spike_{v:04d}" for v in flagged])
        conf = spikes if conf is None else pd.concat([conf.reset_index(drop=True), spikes], axis=1)
        keep = None
    if conf is not None:
        conf = conf.reset_index(drop=True).fillna(0.0)
        if len(conf) != n_frames:
            raise ValueError(f"{boldfile}: {len(conf)} confound rows but {n_frames} volumes")

    qc = {"n_volumes": n_frames, "n_flagged": len(flagged), "pct_flagged": round(100 * len(flagged) / n_frames, 1),
          "censoring": censoring.get("mode", "censor") if len(flagged) else "",
          "n_confounds": 0 if conf is None else conf.shape[1]}
    tsv = fmriprep_confounds_path(boldfile)
    if os.path.exists(tsv):
        fd = pd.read_csv(tsv, sep="\t", usecols=lambda c: c == "framewise_displacement")
        if "framewise_displacement" in fd:
            qc["mean_fd"] = round(float(fd["framewise_displacement"].mean()), 3)
            qc["max_fd"] = round(float(fd["framewise_displacement"].max()), 3)
    return RunNuisance(conf, keep, qc)
```

- `load_confounds` reads fMRIPrep's `desc-confounds_timeseries.tsv` for the strategy in the config and, when the strategy includes `scrub` / `non_steady_state`, also returns the **sample mask** (indices of volumes to keep): FD / std-DVARS outliers and fMRIPrep's non-steady-state volumes.
- `drop_initial_volumes` intersects that mask with "everything after the first N volumes" (e.g. Kim et al.'s 10 dropped TRs).
- **Censor vs. regress.** With `mode: "censor"` the mask travels on (volumes are removed from every fit). With `mode: "regress"` each flagged volume gets a one-hot spike column instead and the mask is dropped, so the series stays continuous (some prefer this with AR(1) prewhitening, whose noise model assumes regular sampling).
- Confounds are zero-filled (fMRIPrep's derivative columns start with NaN) and checked against the run length.

## A2. Volume math (identical to mvpa-decoding)

Every "which volumes does this event cover" question goes through one function:

[`betaestimation/betaestimation/timeseries.py:24`](../../betaestimation/betaestimation/timeseries.py#L24-L32) — `compute_volume_range`

```python
def compute_volume_range(start_time: float, stop_time: float, tr: float, n_frames: int):
    """Return [start_vol, stop_vol) covering start_time for a span of (stop_time - start_time)
    seconds, clipped to n_frames. The volume count comes from the duration rounded to whole TRs
    (not from flooring the start and ceiling the stop, which inflates every window by a volume),
    with at least one volume kept -- the mvpa-decoding convention."""
    start_vol = int(math.floor(start_time / tr))
    n_volumes = max(1, round((stop_time - start_time) / tr))
    stop_vol = min(start_vol + n_volumes, n_frames)
    return start_vol, max(stop_vol, start_vol)
```

The count of volumes comes from the **duration**, rounded to whole TRs, not from flooring the start and ceiling the end — that would add a volume to almost every event, because real onsets rarely fall on a TR boundary. `event_volumes` and `window_volumes` (mvpa-decoding's `{"start"/"end": {"reference", "offset_seconds"}}` windows) are thin wrappers that add the hemodynamic lag or window offsets first.

## A3. Cleaning a run without moving its volumes

TR-wise analyses (volume averaging, template RSA, TR-wise decoding) need a cleaned, z-scored series in which **volume k is still volume k** after censoring:

[`betaestimation/betaestimation/timeseries.py:81`](../../betaestimation/betaestimation/timeseries.py#L81-L95) — `clean_run`

```python
def clean_run(raw: np.ndarray, tr: float, nuisance=None, detrend=True, standardize="zscore_sample",
              high_pass=None, low_pass=None, regress_confounds=True) -> np.ndarray:
    """nilearn signal.clean of one run (confound regression, detrend, band-pass, z-scoring) with
    censored volumes excluded from the fit and returned as NaN rows, so volume indices keep their
    timing. `nuisance` is a RunNuisance (or None)."""
    n_frames = raw.shape[0]
    keep = np.arange(n_frames) if nuisance is None else nuisance.kept(n_frames)
    conf = None
    if regress_confounds and nuisance is not None and nuisance.confounds is not None:
        conf = nuisance.confounds.values
    cleaned = signal.clean(raw, detrend=detrend, standardize=standardize, high_pass=high_pass, low_pass=low_pass,
                           t_r=tr, confounds=conf, sample_mask=None if len(keep) == n_frames else keep)
    out = np.full(raw.shape, np.nan)
    out[keep] = cleaned
    return out
```

`signal.clean` with `sample_mask` excludes censored volumes from the confound regression and the z-scoring; when a band-pass is requested, nilearn first interpolates across them so the filter (and detrend) see a continuous series, then drops them. The result is written back into a full-length array with **NaN rows at censored volumes**. Downstream code averages with `nanmean`, so a censored TR simply drops out of an event's window, and an event whose window is entirely censored is skipped.

GLMs get a different copy of the same data — kept volumes only, mean-scaled exactly as nilearn's `FirstLevelModel` does:

[`betaestimation/betaestimation/timeseries.py:98`](../../betaestimation/betaestimation/timeseries.py#L98-L105) — `glm_ready`

```python
def glm_ready(raw: np.ndarray, nuisance=None, signal_scaling=0) -> np.ndarray:
    """Kept volumes only, mean-scaled as nilearn's FirstLevelModel does (signal_scaling 0 =
    percent signal change per voxel, 1 = per volume, False = none)."""
    keep = np.arange(raw.shape[0]) if nuisance is None else nuisance.kept(raw.shape[0])
    data = raw[keep]
    if signal_scaling is not False:
        data, _ = mean_scaling(data, axis=signal_scaling)
    return data
```

## A4. Masks

[`betaestimation/betaestimation/masks.py:13`](../../betaestimation/betaestimation/masks.py#L13-L25) — `brain_mask`

```python
def brain_mask(mask_files=None, reference_bold=None):
    """One brain mask for all of a subject's runs, so every pattern has the same voxels:
    the single file given, the intersection of several (e.g. fMRIPrep's per-run desc-brain_mask),
    or -- with no files -- an EPI mask computed from the mean of reference_bold.
    -> (mask image, how it was made: "file" / "intersection" / "epi")."""
    files = list(mask_files or [])
    if len(files) == 1:
        return nib.load(files[0]), "file"
    if files:
        return intersect_masks(files, threshold=1, connected=False), "intersection"
    if reference_bold is None:
        raise ValueError("brain_mask: give mask files or a reference BOLD file")
    return compute_epi_mask(mean_img(reference_bold, copy_header=True)), "epi"
```

All runs of a subject share **one** mask (the intersection of fMRIPrep's per-run brain masks, if several match), so every pattern of that subject has the same voxels — a requirement for comparing patterns across runs.

[`betaestimation/betaestimation/masks.py:28`](../../betaestimation/betaestimation/masks.py#L28-L40) — `roi_mask`

```python
def roi_mask(path: str, labels=None, reference_img=None):
    """Binary ROI from a mask image (non-zero voxels) or a label image (the union of `labels`,
    e.g. Desikan-Killiany ids from fMRIPrep's aparcaseg). With reference_img, resampled
    (nearest neighbour) onto its grid -- e.g. a 1 mm atlas onto the 2.4 mm BOLD grid."""
    img = nib.load(path)
    data = np.asarray(img.dataobj)
    data = np.isin(np.round(data).astype(int), list(labels)) if labels else data > 0
    mask = nib.Nifti1Image(data.astype(np.uint8), img.affine)
    if reference_img is not None:
        ref = nib.load(reference_img) if isinstance(reference_img, str) else reference_img
        if ref.shape[:3] != mask.shape or not np.allclose(ref.affine, mask.affine):
            mask = resample_to_img(mask, ref, interpolation="nearest", force_resample=True, copy_header=True)
    return mask
```

Atlas ROIs are the union of integer labels (e.g. the Desikan-Killiany ids in fMRIPrep's `aparcaseg`), resampled with nearest-neighbour interpolation onto the BOLD grid, so a 1 mm anatomical parcellation can be applied to 2.4 mm functional data.

## A5. One pattern per condition, or per trial (LSA)

`estimate_run_patterns` takes an events table where `label` marks the events of interest (NaN = not of interest) and turns labels into regressor names:

[`betaestimation/betaestimation/estimate.py:46`](../../betaestimation/betaestimation/estimate.py#L46-L51) — `_regressor`

```python
def _regressor(row, unit: str) -> str:
    if pd.isna(row["label"]):
        return f"nuisance_{safe_name(row['trial_type'])}"
    if unit == "trial":
        return f"lbl_{safe_name(row['label'])}_ev{safe_name(row['event_id'])}"
    return f"lbl_{safe_name(row['label'])}"
```

[`betaestimation/betaestimation/estimate.py:77`](../../betaestimation/betaestimation/estimate.py#L77-L91) — `estimate_run_patterns`

```python
    glm = dict(glm or {})
    output_type = glm.get("output_type", "t")
    model_df = events if model_unlabeled_events else events.dropna(subset=["label"]).reset_index(drop=True)
    if unit == "trial" and glm.get("trial_model", "lss") == "lss":
        yield from _lss(boldfile, model_df, tr, n_frames, mask_img, nuisance, glm, output_type)
        return

    design_events = model_df.assign(trial_type=[_regressor(r, unit) for _, r in model_df.iterrows()])
    model = fit_first_level(boldfile, design_events, tr, mask_img, nuisance, glm)
    labeled = model_df.dropna(subset=["label"])
    groups = labeled.groupby("event_id", sort=False) if unit == "trial" else labeled.groupby("label", sort=False)
    for _, grp in groups:
        first = grp.iloc[0]
        img = contrast_map(model, _regressor(first, unit), output_type)
        yield PatternEstimate(first["label"], first["event_id"] if unit == "trial" else None, len(grp), img)
```

- **Unlabeled events are nuisance regressors** (`model_unlabeled_events`, default on): e.g. probes or cues get their own regressor by `trial_type` so their response does not end up in the implicit baseline, which would bias every condition's beta.
- `unit="condition"`: one regressor per label pooling all its events in the run; `unit="trial"` + `trial_model="lsa"`: one regressor per event in the same GLM (least-squares-all).
- Each pattern is a unit contrast on its column, returned as a t, z or beta map.

The fit itself is a plain nilearn first-level model with the run's confounds and sample mask:

[`betaestimation/betaestimation/glm.py:60`](../../betaestimation/betaestimation/glm.py#L60-L67) — `fit_first_level`

```python
def fit_first_level(boldfile, events: pd.DataFrame, tr: float, mask_img, nuisance=None, glm: dict = None):
    """nilearn FirstLevelModel on one run, with the run's confounds and censored volumes."""
    model = FirstLevelModel(t_r=tr, mask_img=mask_img, **first_level_kwargs(glm))
    confounds = None if nuisance is None else nuisance.confounds
    sample_masks = None if nuisance is None or nuisance.sample_mask is None else [nuisance.sample_mask]
    model.fit(boldfile, events=events[["onset", "duration", "trial_type"]], confounds=confounds,
              sample_masks=sample_masks)
    return model
```

## A6. Single-trial estimates by least-squares-separate (LSS)

[`betaestimation/betaestimation/glm.py:84`](../../betaestimation/betaestimation/glm.py#L84-L106) — `lss`

```python
def lss(data: np.ndarray, keep: np.ndarray, times: np.ndarray, events: pd.DataFrame, targets,
        confounds: pd.DataFrame = None, glm: dict = None, lss_other: str = "condition"):
    """Least-squares-separate (Mumford et al., 2012): one GLM per target event, with that event
    as the only regressor of interest. The other events are modeled by their trial_type
    ("condition", Mumford's LSS-N) or as one regressor ("single", Mumford's original LSS).

    data    kept volumes x voxels, already masked / smoothed / scaled (see timeseries.glm_ready)
    keep    indices of the kept volumes; designs are built on the full timeline (`times`) and
            censored rows are then dropped, as FirstLevelModel does
    events  onset / duration / trial_type; `targets` are row positions in it
    Yields (row position, nilearn Contrast of the target regressor)."""
    events = events.reset_index(drop=True)
    base = ["lss_other"] * len(events) if lss_other == "single" else list(events["trial_type"])
    kwargs = design_kwargs(glm)
    noise_model = (glm or {}).get("noise_model", "ar1")
    for k in targets:
        names = list(base)
        names[k] = "lss_target"
        ev = pd.DataFrame({"onset": events["onset"].astype(float), "duration": events["duration"].astype(float),
                           "trial_type": names})
        design = make_first_level_design_matrix(times, ev, add_regs=confounds, **kwargs)
        labels, results = run_glm(data, design.values[keep], noise_model=noise_model)
        yield k, compute_contrast(labels, results, (design.columns == "lss_target").astype(float), stat_type="t")
```

[`betaestimation/betaestimation/estimate.py:94`](../../betaestimation/betaestimation/estimate.py#L94-L104) — `_lss`

```python
def _lss(boldfile, model_df, tr, n_frames, mask_img, nuisance, glm, output_type):
    masker, raw = load_masked(boldfile, mask_img, tr, glm.get("smoothing_fwhm"))
    keep = np.arange(n_frames) if nuisance is None else nuisance.kept(n_frames)
    data = glm_ready(raw, nuisance, glm.get("signal_scaling", 0))
    times = frame_times(raw.shape[0], tr, glm.get("slice_time_ref", 0.0))
    events = model_df.assign(trial_type=[_regressor(r, "condition") for _, r in model_df.iterrows()])
    targets = list(np.flatnonzero(model_df["label"].notna().values))
    confounds = None if nuisance is None else nuisance.confounds
    for k, con in lss(data, keep, times, events, targets, confounds, glm, glm.get("lss_other", "condition")):
        row = model_df.iloc[k]
        yield PatternEstimate(row["label"], row["event_id"], 1, masker.inverse_transform(contrast_values(con, output_type)))
```

- One GLM **per target trial**: the target is the only regressor of interest; the others are modeled by condition (`lss_other: "condition"`, Mumford's LSS-N) or lumped into one regressor (`"single"`, the original LSS).
- The run is masked, smoothed and scaled **once**; each trial only builds a design matrix and runs `run_glm` on the already-masked data.
- **Censoring is applied after building the design** on the full timeline (`design.values[keep]`), so the HRF convolution of every regressor sees the true event timing — the same order of operations as `FirstLevelModel`.
- ▲ Note: when every condition occurs exactly once per run, LSS-N's design *is* the per-condition design (and the LSA design), and they give identical maps (verified on Haxby and in `tests/test_betaestimation.py`). LSS-N differs from LSA only when a condition repeats within a run — the case it exists for, since LSA's betas for closely spaced, collinear trials of the same condition get noisy. `lss_other: "single"` differs as soon as a run has three or more events.

## A7. Volume averaging (no GLM)

[`betaestimation/betaestimation/estimate.py:109`](../../betaestimation/betaestimation/estimate.py#L109-L121) — `_volume_average`

```python
    data = clean_run(raw, tr, nuisance, detrend=va["detrend"], standardize=va["standardize"],
                     high_pass=va["high_pass"], low_pass=va.get("low_pass"))
    per_event = []
    for _, ev in events.dropna(subset=["label"]).iterrows():
        vols = event_volumes(float(ev["onset"]), float(ev["duration"]), tr, n_frames, va["hemodynamic_lag"])
        if len(vols) == 0:
            print(f"  (!) event {ev['trial_type']} at {ev['onset']}s falls past the end of the run; skipped")
            continue
        window = data[vols]
        if np.isnan(window[:, 0]).all():
            print(f"  (!) every volume of event {ev['trial_type']} at {ev['onset']}s is censored; skipped")
            continue
        per_event.append((ev["label"], ev["event_id"], np.nanmean(window, axis=0)))
```

The cleaned series (A3) is averaged over each event's volumes after shifting by `hemodynamic_lag` — mvpa-decoding's approach to "the pattern for an event". With `unit="condition"` the events of each label are then averaged within the run.

## A8. One GLM across several runs (SPM-style multi-session)

Template RSA needs item-vs-others contrasts where an item may appear in only one run (Bretton et al.: each scene is shown once in the pre-test). A per-run GLM cannot estimate that contrast, so the runs are stacked into one model:

[`betaestimation/betaestimation/glm.py:109`](../../betaestimation/betaestimation/glm.py#L109-L128) — `fit_concatenated`

```python
def fit_concatenated(runs: list, shared_prefix: str, glm: dict = None):
    """One GLM over several runs, as SPM fits a multi-session model: regressors whose name starts
    with `shared_prefix` are shared across runs; every other column (nuisance events, drift,
    intercept, confounds) is made run-specific. Handles conditions that occur in only some runs.

    runs  [{"data": kept volumes x voxels (timeseries.glm_ready), "keep": kept indices,
            "n_frames": int, "tr": float, "events": onset/duration/trial_type, "confounds": df|None}]
    -> (labels, results, design columns) for contrast()."""
    glm = glm or {}
    kwargs = design_kwargs(glm)
    designs, data = [], []
    for k, run in enumerate(runs):
        times = frame_times(run["n_frames"], run["tr"], glm.get("slice_time_ref", 0.0))
        design = make_first_level_design_matrix(times, run["events"], add_regs=run.get("confounds"), **kwargs)
        design = design.rename(columns={c: f"run{k}_{c}" for c in design.columns if not str(c).startswith(shared_prefix)})
        designs.append(design.iloc[run["keep"]].reset_index(drop=True))
        data.append(run["data"])
    design = pd.concat(designs, ignore_index=True).fillna(0.0)
    labels, results = run_glm(np.vstack(data), design.values, noise_model=glm.get("noise_model", "ar1"))
    return labels, results, list(design.columns)
```

- Columns whose name starts with `shared_prefix` (`item_…`, `cat_…`) are **shared across runs**; every other column — nuisance events, cosine drifts, the intercept, confounds — is renamed `run<k>_…` and therefore **run-specific** (block-diagonal), as SPM does for sessions.
- `fillna(0)` makes a run's design zero for items it never shows.
- Censored rows are dropped per run before stacking.
- ▲ Note: `run_glm` estimates the AR(1) noise model on the stacked series, so a few lag pairs straddle run boundaries; SPM estimates serial correlations per session. The effect on the t-maps is small (one boundary per run), and `noise_model: "ols"` avoids it entirely.

[`betaestimation/betaestimation/glm.py:131`](../../betaestimation/betaestimation/glm.py#L131-L137) — `contrast`

```python
def contrast(labels, results, columns: list, weights: dict, stat: str = "t") -> np.ndarray:
    """{column: weight} contrast of a run_glm / fit_concatenated fit -> t or z values."""
    vec = np.zeros(len(columns))
    for col, w in weights.items():
        vec[columns.index(col)] = w
    con = compute_contrast(labels, results, vec, stat_type="t")
    return np.ravel(con.stat() if stat == "t" else con.z_score())
```

[`betaestimation/betaestimation/glm.py:140`](../../betaestimation/betaestimation/glm.py#L140-L143) — `versus_others`

```python
def versus_others(target: str, others) -> dict:
    """Contrast weights: +1 target, -1/len(others) each other (target vs. the mean of the others)."""
    others = [o for o in others if o != target]
    return {target: 1.0, **{o: -1.0 / len(others) for o in others}}
```

`versus_others` builds the "item vs. the mean of the other items" weights used for item-selective t-maps.

## A9. Cluster-extent filtering

[`betaestimation/betaestimation/masks.py:43`](../../betaestimation/betaestimation/masks.py#L43-L52) — `cluster_filter`

```python
def cluster_filter(significant: np.ndarray, masker, min_size: int) -> np.ndarray:
    """Keep only voxels of `significant` (a boolean vector in masker space) that belong to
    26-connected clusters of at least min_size voxels."""
    if min_size <= 1 or not significant.any():
        return significant
    vol = np.asarray(masker.inverse_transform(significant.astype(np.float32)).dataobj) > 0
    lab, n = ndimage.label(vol, structure=np.ones((3, 3, 3)))
    sizes = ndimage.sum(vol, lab, index=np.arange(1, n + 1))
    keep = np.isin(lab, np.flatnonzero(sizes >= min_size) + 1)
    return masker.transform(nib.Nifti1Image(keep.astype(np.float32), masker.mask_img_.affine)).ravel() > 0.5
```

Thresholded maps keep only 26-connected clusters of at least `min_size` voxels (the papers' "cluster correction, voxel extent = 10").

---

# Part B — rsaworkflow: from events.tsv to inputs

## B1. The event table

[`rsaworkflow/workflows/generate_master_spreadsheet.py:78`](../workflows/generate_master_spreadsheet.py#L78-L99) — `process_events_file`

```python
    excluded = events["trial_type"].apply(is_excluded_trial_type)
    invalid = events["onset"].isna() | events["duration"].isna() | ~np.isfinite(events["duration"])
    if excluded.sum():
        print(f"  (i) excluded {int(excluded.sum())} administrative row(s) (fixation/block/postRT) from {events_path}")
    if (invalid & ~excluded).sum():
        print(f"  (!) skipping {int((invalid & ~excluded).sum())} row(s) with non-finite onset/duration in {events_path}")
    events = events[~excluded & ~invalid].reset_index(drop=True)

    session = entities.get("ses", "")
    events.insert(0, "subject", entities["sub"])
    events.insert(1, "session", session)
    events.insert(2, "task", entities["task"])
    events.insert(3, "run", str(int(entities["run"])) if entities["run"].isdigit() else entities["run"])
    events.insert(4, "run_label", entities["run"])
    events.insert(5, "run_id", "_".join(p for p in (session, entities["task"], entities["run"]) if p))
    events.insert(6, "trial_index", np.arange(1, len(events) + 1))
    for k, v in extra_entities.items():
        events[k] = v
    events["tr"] = tr
    events["n_frames"] = n_frames
    events["boldfile"] = bold_path
    events["eventfile"] = events_path
```

One row per event, every events.tsv column kept. Fixation / block markers / post-RT rows are dropped **before** `trial_index` is assigned, so trial numbers count only real events. `run_id` (session_task_run) identifies runs across tasks; it is the cross-validation fold label later.

## B2. Which events are items

[`rsaworkflow/utils/rsa_common.py:405`](../utils/rsa_common.py#L405-L419) — `label_items`

```python
def label_items(df: pd.DataFrame, conditions_cfg: dict) -> pd.DataFrame:
    """Add an `item` column naming which RDM item (conditions.items key) each event
    belongs to, or NaN if none. Rows outside conditions.include are dropped entirely
    (they are neither items nor nuisance). An event matching several items is
    assigned to the first, in config order -- validate_rsa_config.py flags these."""
    df = df.copy()
    include = conditions_cfg.get("include")
    if include:
        df = df[evaluate_query_node(include, df)].copy()
    df["item"] = np.nan
    df["item"] = df["item"].astype(object)
    for name, query in conditions_cfg["items"].items():
        hit = evaluate_query_node(query, df) & df["item"].isna()
        df.loc[hit, "item"] = name
    return df
```

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

[`rsaworkflow/utils/rsa_common.py:425`](../utils/rsa_common.py#L425-L470) — `expand_conditions`

```python
def expand_conditions(cfg: dict, df: pd.DataFrame) -> list:
    """With conditions.item_column set, every distinct value of that column (among the events
    kept by conditions.include, across all subjects) becomes an item, and each item's
    attributes are read from conditions.attribute_columns. Writes conditions.items /
    .attributes in place and returns a list of problems (also kept in
    conditions._expand_errors for the validator).

    Item order -- the RDM's row order -- sorts by the attribute columns, then the item, so
    items of one category sit together. An item whose events disagree on an attribute
    (e.g. one exemplar listed under two categories) is an error. Hand-written attributes for
    a generated item are kept and override the columns."""
    cond = cfg["conditions"]
    col, attr_cols = cond.get("item_column"), list(cond.get("attribute_columns") or [])
    if not col:
        return []
    if cond.get("items") and not cond.get("_items_generated"):
        errors = ["conditions: set either item_column (items read from the event table) or items "
                  "(written by hand), not both"]
        cond["_expand_errors"] = errors
        return errors
    missing = [c for c in [col, *attr_cols] if c not in df.columns]
    errors = [f"conditions: column {c!r} is not in the event table" for c in missing]
    if not errors:
        sub = df[evaluate_query_node(cond["include"], df)] if cond.get("include") else df
        sub = sub[~sub[col].isna()]
        sub = sub.assign(**{c: sub[c].astype(str) for c in [col, *attr_cols]})
        sub = sub[~sub[col].isin(MISSING_VALUES)]
        if sub.empty:
            errors.append(f"conditions.item_column={col!r}: no events with a value in that column")
        attributes = {}
        for item, grp in sub.groupby(col, sort=False):
            attributes[item] = {}
            for a in attr_cols:
                values = sorted(set(grp[a]) - MISSING_VALUES)
                if len(values) > 1:
                    errors.append(f"item {item!r}: its events disagree on {a!r} ({', '.join(values)})")
                attributes[item][a] = values[0] if values else None
        order = sorted(attributes, key=lambda i: ([str(attributes[i][a]) for a in attr_cols], i))
        manual = cond.setdefault("_manual_attributes", copy.deepcopy(cond.get("attributes") or {}))
        cond["items"] = {i: {"column": col, "match": "exact", "value": i} for i in order}
        cond["attributes"] = {i: {**attributes[i], **manual.get(i, {})} for i in order}
        cond["_items_generated"] = True
    cond["_expand_errors"] = errors
    if errors:
        cond["items"] = {}
    return errors
```

- Items are ordered by the attribute columns, then by name, so each category forms a block in the RDM figures.
- An exemplar whose events disagree on an attribute (listed under two categories) is an error, not a silent first-value pick.
- The expansion runs on the whole event table, so every subject gets the same item list and RDM order.
- `items` and `item_column` are mutually exclusive. Hand-written `attributes` can still add or override values for generated items.

▲ **Note — name collisions.** Regressors and pattern files are named by `safe_name(item)`, which keeps only letters and digits, so `dog_1` and `dog1` would be fit as one regressor. The validator rejects such item sets, and betaestimation's `estimate_run_patterns` refuses colliding labels too.

## B3. Handing runs to betaestimation

[`rsaworkflow/workflows/estimate_patterns.py:84`](../workflows/estimate_patterns.py#L84-L88) — `run_events`

```python
def run_events(run_df: pd.DataFrame) -> pd.DataFrame:
    """The run's events in betaestimation's form: label = RDM item (NaN = not of interest)."""
    return pd.DataFrame({"onset": run_df["onset"].astype(float).values, "duration": run_df["duration"].astype(float).values,
                         "trial_type": run_df["trial_type"].values, "label": run_df["item"].values,
                         "event_id": run_df["trial_index"].astype(int).values})
```

[`rsaworkflow/workflows/estimate_patterns.py:104`](../workflows/estimate_patterns.py#L104-L121) — `estimate_subject`

```python
    for boldfile, run_df in sub_df.groupby("boldfile", sort=True):
        if run_df["item"].notna().sum() == 0:
            continue
        first = run_df.iloc[0]
        nuisance = run_nuisance(pe, run_df)
        qc_rows.append(nuisance.qc)
        flagged = f", {nuisance.qc['n_flagged']} vol flagged ({nuisance.qc['censoring']})" if nuisance.qc["n_flagged"] else ""
        print(f"  run {first['run_id']}: {run_df['item'].notna().sum()} item event(s) [{pe['method']}], "
              f"{nuisance.qc['n_confounds']} confound(s){flagged}")
        stem = os.path.basename(boldfile).split("_desc-")[0].split("_bold")[0]
        patterns = estimate_run_patterns(
            boldfile, run_events(run_df), float(first["tr"]), int(first["n_frames"]), mask_img, nuisance,
            method=pe["method"], unit=UNITS[pe["unit"]], glm=pe["glm"], volume_average=pe["volume_average"],
            model_unlabeled_events=pe["model_unlabeled_events"])
        for est in patterns:
            item, n_events, img = est.label, est.n_events, est.img
            trial = None if est.event_id is None else int(est.event_id)
            trial_tag = f"_trial-{trial:04d}" if trial is not None else ""
```

The item becomes betaestimation's `label` and `trial_index` its `event_id`; everything methodological from here to the pattern images is Part A.

---

# Part C — rsaworkflow: the RDM analysis

## C1. Pattern matrix per ROI

[`rsaworkflow/workflows/rsa_workflow.py:62`](../workflows/rsa_workflow.py#L62-L72) — `roi_rsa`

```python
    masker = NiftiMasker(mask_img=resolve_roi_mask(roi_cfg, subject, index["pattern_file"].iloc[0])).fit()
    raw = masker.transform(index["pattern_file"].tolist())
    data, kept = clean_measurements(raw)
    n_dropped = int((~kept).sum())
    if n_dropped:
        print(f"    (i) dropped {n_dropped}/{raw.shape[1]} non-finite or constant voxel(s)")
    if data.shape[1] < 2:
        raise SystemExit(f"ROI {roi_name} has <2 usable voxels for subject {subject}")

    dataset = build_dataset(data, index, {"subject": subject, "roi": roi_name})
    rdm = compute_data_rdm(dataset, cfg["rdm"], items)
```

[`rsaworkflow/utils/rsa_common.py:641`](../utils/rsa_common.py#L641-L647) — `clean_measurements`

```python
def clean_measurements(measurements: np.ndarray):
    """Drop voxels that are non-finite or constant across all patterns (outside-brain or
    masked-out voxels); return (cleaned, kept_mask)."""
    finite = np.all(np.isfinite(measurements), axis=0)
    varying = np.zeros_like(finite)
    varying[finite] = np.std(measurements[:, finite], axis=0) > 0
    return measurements[:, varying], varying
```

Voxels that are non-finite or constant across all patterns (outside the brain mask, or masked out by the GLM) are dropped, so they cannot produce NaN correlations or zero-variance distances.

## C2. Distances

[`rsaworkflow/utils/rsa_common.py:703`](../utils/rsa_common.py#L703-L725) — `compute_data_rdm`

```python
def compute_data_rdm(dataset, rdm_cfg: dict, item_order: list):
    """rdm_cfg -> rsatoolbox calc_rdm. Patterns sharing an item are averaged (or, for
    cross-validated methods, contrasted across `cv_descriptor` folds). Output is
    reordered to item_order (calc_rdm otherwise sorts items alphabetically)."""
    import rsatoolbox

    method = rdm_cfg["method"]
    kwargs = {"method": method, "descriptor": "item"}
    dataset = _remove_voxel_mean(dataset, rdm_cfg)
    if method in RDM_METHODS_NEEDING_NOISE:
        noise_method = (rdm_cfg.get("noise") or {}).get("method", "shrinkage_diag")
        kwargs["noise"] = rsatoolbox.data.noise.prec_from_measurements(
            dataset, obs_desc="item", method=noise_method)
    if method in RDM_METHODS_NEEDING_CV:
        kwargs["cv_descriptor"] = rdm_cfg.get("cv_descriptor", "run")
    rdm = rsatoolbox.rdm.calc_rdm(dataset, **kwargs)
    # calc_rdm stores the voxel x voxel noise precision as a descriptor; it is huge and differs
    # in size across subjects, which breaks saving and rsatoolbox.rdm.concat at the group level
    rdm.descriptors.pop("noise", None)
    present = [i for i in item_order if i in set(rdm.pattern_descriptors["item"])]
    current = list(rdm.pattern_descriptors["item"])
    rdm.reorder([current.index(i) for i in present])
    return rdm
```

- `descriptor="item"`: rsatoolbox averages each item's patterns (all runs) before computing correlation or Euclidean distances.
- **Crossnobis** (`cv_descriptor="run"`): distances are computed between run-wise pattern differences from *different* runs, so noise that is independent across runs cancels in expectation — an unbiased distance whose expected value is 0 for identical patterns.
- Mahalanobis / crossnobis whiten with a noise precision estimated from the patterns' residuals around their item means (`prec_from_measurements`, shrinkage towards the diagonal by default). ▲ Note: the precision is estimated from the same patterns the distances use (standard in rsatoolbox examples); with few runs, a residual-based estimate from the first-level GLMs would be the stricter alternative.
- `calc_rdm` sorts items alphabetically; the RDM is reordered to the config's item order so it matches the model RDMs.

The pattern-level view (every pattern its own row) uses a non-cross-validated distance:

[`rsaworkflow/utils/rsa_common.py:688`](../utils/rsa_common.py#L688-L700) — `compute_pattern_rdm`

```python
def compute_pattern_rdm(dataset, rdm_cfg: dict) -> np.ndarray:
    """Pattern-level RDM: every pattern (each item x run, or each trial) is its own row, in
    the dataset's row order -- nothing is averaged. A QC / visualization view of the data
    behind the item RDM: item structure shows as blocks, run or drift effects as stripes."""
    import rsatoolbox

    dataset = _remove_voxel_mean(dataset, rdm_cfg)
    method = pattern_rdm_method(rdm_cfg)
    kwargs = {"method": method}
    if method in RDM_METHODS_NEEDING_NOISE:
        noise_method = (rdm_cfg.get("noise") or {}).get("method", "shrinkage_diag")
        kwargs["noise"] = rsatoolbox.data.noise.prec_from_measurements(dataset, obs_desc="item", method=noise_method)
    return rsatoolbox.rdm.calc_rdm(dataset, **kwargs).get_matrices()[0]
```

## C3. Model RDMs and their comparison with the data

[`rsaworkflow/utils/rsa_common.py:604`](../utils/rsa_common.py#L604-L617) — `build_model_rdms`

```python
    for name, spec in cfg["model_rdms"].items():
        mtype = spec["type"]
        if mtype == "categorical":
            labels = [attributes[i][spec["attribute"]] for i in items]
            mat = np.array([[0.0 if a == b else 1.0 for b in labels] for a in labels])
        elif mtype == "feature":
            feats = np.array([np.atleast_1d(spec["features"][i]) for i in items], dtype=float)
            ds = rsatoolbox.data.Dataset(feats, obs_descriptors={"item": items})
            mat = rsatoolbox.rdm.calc_rdm(ds, method=spec.get("metric", "euclidean")).get_matrices()[0]
        else:
            mat = load_matrix_file(spec["path"]).astype(float)
            mat = (mat + mat.T) / 2.0
            np.fill_diagonal(mat, 0.0)
        vectors.append(mat)
```

[`rsaworkflow/workflows/rsa_workflow.py:94`](../workflows/rsa_workflow.py#L94-L97) — `roi_rsa`

```python
        present = list(rdm.pattern_descriptors["item"])
        models = model_rdms.subset_pattern("item", present)
        models.reorder([list(models.pattern_descriptors["item"]).index(i) for i in present])
        sims = rsatoolbox.rdm.compare(models, rdm, method=cfg["comparison"]["method"]).ravel()
```

Categorical models are 0 within and 1 between attribute values — they predict **tied ranks**, which is why the main Haxby config compares with Kendall's τ-a (Nili et al., 2014) rather than Spearman's ρ (the crossnobis config uses cosine similarity, appropriate for ratio-scale distances).

## C4. Group inference

[`rsaworkflow/workflows/group_rsa.py:91`](../workflows/group_rsa.py#L91-L105) — `noise_ceiling`

```python
def noise_ceiling(data_rdms, method: str):
    import rsatoolbox

    n = data_rdms.n_rdm
    if n < 2:
        return np.nan, np.nan
    vecs = data_rdms.get_vectors()
    upper, lower = [], []
    for s in range(n):
        subj = data_rdms[s]
        mean_all = rsatoolbox.rdm.RDMs(vecs.mean(axis=0, keepdims=True))
        mean_others = rsatoolbox.rdm.RDMs(np.delete(vecs, s, axis=0).mean(axis=0, keepdims=True))
        upper.append(rsatoolbox.rdm.compare(subj, mean_all, method=method).item())
        lower.append(rsatoolbox.rdm.compare(subj, mean_others, method=method).item())
    return float(np.mean(lower)), float(np.mean(upper))
```

Leave-one-subject-out noise ceiling: the **upper** bound correlates each subject with the mean RDM *including* that subject (over-fits), the **lower** bound with the mean of the *other* subjects (a true held-out prediction). A model reaching the lower bound explains as much as the between-subject consistency allows.

[`rsaworkflow/workflows/group_rsa.py:125`](../workflows/group_rsa.py#L125-L136) — `random_effects_stats`

```python
    for model in sims.columns:
        x = sims[model].dropna().values
        if len(x) < 2:
            stat, p = np.nan, np.nan
        elif test == "wilcoxon":
            stat, p = stats.wilcoxon(x, alternative=alt)
        else:
            stat, p = stats.ttest_1samp(x, 0.0, alternative=alt)
        rows.append({"model": model, "n_subjects": len(x), "mean": x.mean() if len(x) else np.nan,
                     "sem": stats.sem(x) if len(x) > 1 else np.nan, "statistic": stat, "p": p})
    per_model = pd.DataFrame(rows)
    per_model["p_corrected"] = correct(per_model["p"], group_cfg["multiple_comparisons"])
```

Subjects are the random effect: one similarity per subject × model, a one-sample test against 0 (one-sided by default), FDR across models within the ROI.

[`rsaworkflow/workflows/group_rsa.py:201`](../workflows/group_rsa.py#L201-L208) — `crossval_stats`

```python
    if cv_cfg["scheme"] == "k_fold":
        train, test, ceil = sets_k_fold_rdm(data_rdms, k_rdm=cv_cfg["k"], random=True, rdm_descriptor="subject")
    else:
        train, test, ceil = sets_leave_one_out_rdm(data_rdms, rdm_descriptor="subject")
    result = crossval(models, data_rdms, train, test, ceil_set=ceil, method=method)
    scores = np.asarray(result.evaluations)[0]  # models x folds
    fold_labels = ["+".join(t[0].rdm_descriptors["subject"]) for t in test]
    nc = np.asarray(result.noise_ceiling, dtype=float).reshape(2, -1).mean(axis=1)
```

Weighted models have free weights, so they are only ever scored on **held-out subjects**: weights are fit on the training subjects' RDMs and the fitted model is compared with the left-out subject's RDM; fixed models are scored in the same folds so the comparison is fair.

---

# Part D — rsaworkflow: template RSA (Kim et al., 2020; Bretton et al., 2024)

## D1. Two copies of each run

[`rsaworkflow/workflows/template_rsa.py:106`](../workflows/template_rsa.py#L106-L115) — `load_run`

```python
def load_run(cfg: dict, run_df: pd.DataFrame, masker: NiftiMasker) -> RunData:
    first = run_df.iloc[0]
    tr, n_frames = float(first["tr"]), int(first["n_frames"])
    nuisance = run_nuisance(cfg["pattern_estimation"], run_df)
    raw = masker.transform(first["boldfile"])
    ts = cfg["template_rsa"]["timeseries"]
    clean = clean_run(raw, tr, nuisance, detrend=ts["detrend"], standardize=ts["standardize"],
                      high_pass=ts["high_pass"], low_pass=ts["low_pass"], regress_confounds=ts["regress_confounds"])
    glm_data = glm_ready(raw, nuisance, cfg["pattern_estimation"]["glm"].get("signal_scaling", 0))
    return RunData(first["boldfile"], tr, n_frames, nuisance.kept(n_frames), clean, glm_data, nuisance.confounds)
```

Every run in the template phase, the target phases and the category-mask events is loaded once per ROI, into a **z-scored TR series with NaN rows at censored volumes** (patterns and scoring) and a **mean-scaled, censored copy** (the feature-selection GLMs) — both from Part A.

## D2. Category masks

[`rsaworkflow/workflows/template_rsa.py:155`](../workflows/template_rsa.py#L155-L181) — `feature_selection`

```python
    if need_category_glm and len(categories) > 1:
        cat_events = select(sub_df, cm.get("events") or tr_cfg["templates"]["events"])
        events_by_run = {}
        for bold, run_df in sub_df[sub_df["boldfile"].isin(cat_events["boldfile"])].groupby("boldfile"):
            interest = run_df.index.isin(cat_events.index)
            names = "cat_" + run_df[cat_col].astype(str).map(safe_name)
            events_by_run[bold] = glm_events(run_df, interest, names, cm.get("block_merge_gap_seconds"))
        labels, results, columns = concat_glm(cfg, runs, events_by_run, "cat_")
        present = [c for c in dict.fromkeys(cat_events[cat_col].astype(str)) if f"cat_{safe_name(c)}" in columns]
        z_crit = norm.isf(cm["p_threshold"])
        for c in categories:
            if c not in present:
                continue
            others = [o for o in present if o != c]
            weights = {f"cat_{safe_name(c)}": 1.0, **{f"cat_{safe_name(o)}": -1.0 / len(others) for o in others}}
            category_t[c] = contrast_stat(labels, results, columns, weights, "t")
            fallback = False
            if cm["enabled"]:
                z = contrast_stat(labels, results, columns, weights, "z")
                mask = cluster_filter(z > z_crit, masker, int(cm["cluster_extent"]))
                if mask.sum() < int(cm["min_voxels"]):
                    # not part of the published method: an empty mask would silently drop every item
                    # of this category, so fall back to the whole ROI and flag it
                    print(f"    (!) category {c!r}: {int(mask.sum())} voxels survive p<{cm['p_threshold']}, "
                          f"k>={cm['cluster_extent']} (< min_voxels={cm['min_voxels']}); using the whole ROI")
                    mask, fallback = np.ones(n_vox, bool), True
                category_masks[c] = mask
```

- One concatenated GLM (A8) over the runs of the category-mask events, with one shared regressor per category (optionally merged into mini-blocks, Kim et al.'s 14.26 s boxcars); all other events in those runs are nuisance regressors.
- For each category, category vs. the mean of the others → **t map** (kept as the category-level weights) and **z map**, thresholded one-sided at `p_threshold` and cluster-filtered (A9).
- ▲ Note: if fewer than `min_voxels` survive, the whole ROI is used and the fallback is recorded — not part of the published method (see lab_methods.md §4.5).

## D3. Item weights

[`rsaworkflow/workflows/template_rsa.py:188`](../workflows/template_rsa.py#L188-L209) — `feature_selection`

```python
    if "item" in fw["levels"]:
        tmpl_events = select(sub_df, tr_cfg["templates"]["events"])
        events_by_run = {}
        item_col = tr_cfg["item_column"]
        for bold, run_df in sub_df[sub_df["boldfile"].isin(tmpl_events["boldfile"])].groupby("boldfile"):
            interest = run_df.index.isin(tmpl_events.index)
            names = "item_" + run_df[item_col].astype(str).map(safe_name)
            events_by_run[bold] = glm_events(run_df, interest, names)
        labels, results, columns = concat_glm(cfg, runs, events_by_run, "item_")
        for _, row in items.iterrows():
            pool = items
            if fw["item_contrast"] == "within_category":
                pool = items[items["category"] == row["category"]]
            elif fw["item_contrast"] == "within_subcategory":
                pool = items[items["subcategory"] == row["subcategory"]]
            others = [o for o in pool["item"] if o != row["item"] and f"item_{safe_name(o)}" in columns]
            if not others:
                raise SystemExit(f"item contrast for {row['item']!r} has no comparison items "
                                 f"(item_contrast={fw['item_contrast']!r})")
            weights = {f"item_{safe_name(row['item'])}": 1.0,
                       **{f"item_{safe_name(o)}": -1.0 / len(others) for o in others}}
            item_t[row["item"]] = contrast_stat(labels, results, columns, weights, "t")
```

Least-squares-all over the template runs: one shared regressor per template item, and each item's weights are the t-values of **item vs. the mean of the other items** (or only those of its category / subcategory). Template-phase events that are *not* template events — e.g. Bretton's `old_<item>` scenes — fall into `glm_events`' nuisance branch, each with its own regressor because their `trial_type` is unique:

[`rsaworkflow/workflows/template_rsa.py:130`](../workflows/template_rsa.py#L130-L136) — `glm_events`

```python
def glm_events(run_df: pd.DataFrame, interest: pd.Series, names: pd.Series, merge_gap=None) -> pd.DataFrame:
    """Events for one run: rows flagged in `interest` get regressor `names`; other events in the
    run become nuisance regressors by trial_type (so they don't leak into the baseline)."""
    trial_type = np.where(interest, names, "nuis_" + run_df["trial_type"].map(safe_name))
    events = pd.DataFrame({"onset": run_df["onset"].astype(float), "duration": run_df["duration"].astype(float),
                           "trial_type": trial_type})
    return merge_blocks(events, merge_gap) if merge_gap is not None else events
```

## D4. Templates

[`rsaworkflow/workflows/template_rsa.py:285`](../workflows/template_rsa.py#L285-L307) — `build_template_sets`

```python
    occurrences = {}  # item -> list of (event index, raw pattern)
    for idx, ev in tmpl_events.iterrows():
        pat = event_pattern(runs[ev["boldfile"]], window, ev)
        if pat is not None:
            occurrences.setdefault(str(ev[item_col]), []).append((idx, pat))

    sets = {}
    cat_of = items.set_index("item")["category"]
    for level in fw["levels"]:
        masks, weights, templates = {}, {}, {}
        for item in items["item"]:
            if item not in occurrences:
                continue
            c = cat_of[item]
            m = category_masks[c]
            if level == "item":
                w = item_t[item]
            elif level == "category":
                w = category_t.get(c, np.ones_like(m, dtype=float))
            else:
                w = np.ones(m.shape)
            masks[item], weights[item] = m, w
            templates[item] = np.mean([p for _, p in occurrences[item]], axis=0)[m] * w[m]
```

A template is the item's z-scored pattern — averaged over the volumes of `templates.window` (lag-shifted) and over the item's template events — **restricted to its category's mask and multiplied voxel-wise by its weights**. The three `levels` differ only in `w`: the item's t map, its category's t map, or ones.

## D5. Scoring a target pattern

[`rsaworkflow/workflows/template_rsa.py:232`](../workflows/template_rsa.py#L232-L239) — `TemplateSet.score`

```python
    def score(self, y: np.ndarray, item: str, template=None) -> float:
        """Pearson r between item's (weighted, masked) template and y weighted by the same item's weights."""
        m = self.masks[item]
        t = self.templates[item] if template is None else template
        x = y[m] * self.weights[item][m] if self.weight_target else y[m]
        if np.isnan(x).any() or len(x) < 3:
            return np.nan
        return pearson(t, x)
```

[`rsaworkflow/workflows/template_rsa.py:249`](../workflows/template_rsa.py#L249-L261) — `score_matrix`

```python
def score_matrix(ts: TemplateSet, data: np.ndarray, items: list) -> np.ndarray:
    """Pearson r of every row of `data` (volumes x voxels) with each item's template:
    -> volumes x items (NaN for censored volumes). Vectorized over volumes."""
    out = np.full((data.shape[0], len(items)), np.nan)
    for j, item in enumerate(items):
        m = ts.masks[item]
        x = data[:, m] * (ts.weights[item][m] if ts.weight_target else 1.0)
        t = ts.templates[item] - ts.templates[item].mean()
        xc = x - x.mean(axis=1, keepdims=True)
        denom = np.sqrt((xc ** 2).sum(axis=1) * (t @ t))
        with np.errstate(invalid="ignore", divide="ignore"):
            out[:, j] = (xc @ t) / denom
    return out
```

- The target pattern is weighted by the **same item's** weights before correlating (from the authors' code: `ytarg_pat .* xbeta`), so template and target live in the same feature space.
- `score_matrix` computes the Pearson r of **every volume of a run** with every template in one vectorized pass; trial timecourses are then just indexing into it. Censored volumes (NaN rows) yield NaN scores.

[`rsaworkflow/workflows/template_rsa.py:268`](../workflows/template_rsa.py#L268-L276) — `comparison_sets`

```python
def comparison_sets(ts: TemplateSet, item: str, related_level: str) -> dict:
    row = ts.items.set_index("item").loc[item]
    same = ts.items[ts.items["category"] == row["category"]]
    if related_level == "subcategory" and "subcategory" in ts.items and pd.notna(row.get("subcategory")):
        related = same[(same["subcategory"] == row["subcategory"]) & (same["item"] != item)]["item"].tolist()
    else:
        related = same[same["item"] != item]["item"].tolist()
    others = ts.items[ts.items["category"] != row["category"]]["item"].tolist()
    return {"target": [item], "related": related, "others": others}
```

[`rsaworkflow/workflows/template_rsa.py:313`](../workflows/template_rsa.py#L313-L327) — `extra_sets`

```python
def extra_sets(ts: TemplateSet, ev, spec: dict) -> dict:
    """extra_sets {name: {"column", "level"}}: the templates whose item / category / subcategory equals
    this event's value in `column` -- e.g. on replace trials, every template of the replacement
    category ({"column": "replace_category", "level": "category"}). Events with no value (or no
    matching template) get no score for that set."""
    sets = {}
    for name, xs in spec.items():
        value = ev.get(xs["column"])
        if value is None or pd.isna(value):
            continue
        level = xs.get("level", "item")
        members = ts.items[ts.items[level].astype(str) == str(value)]["item"].tolist()
        if members:
            sets[name] = members
    return sets
```

[`rsaworkflow/workflows/template_rsa.py:357`](../workflows/template_rsa.py#L357-L361) — `score_targets`

```python
                        for comp, members in sets.items():
                            r = np.nanmean(S[vol, [col_of[m] for m in members]]) if members else np.nan
                            rows.append({**meta, "target_set": name, "level": level, "comparison": comp,
                                         "tr_offset": k, "time_s": k * rd.tr, "volume": int(vol),
                                         "n_items": len(members), "r": r, "z": fisher_z(r)})
```

- A set's score is the **mean Pearson r over its items, then Fisher z** (the authors' order of operations), for `target` (the trial's own item), `related` (other items of its category), `others` (items of other categories) and any `extra_sets` — e.g. every template of the category named in `replace_category`.
- Every events.tsv column is carried into each row (`meta`), which is what lets the post-hoc notebooks define memory outcomes, previous trials or conditions later without touching the core.

## D6. Template validation

[`rsaworkflow/workflows/template_rsa.py:385`](../workflows/template_rsa.py#L385-L390) — `validate_templates`

```python
            for k, (idx, pat) in enumerate(occ):
                loo = np.mean([p for j, (_, p) in enumerate(occ) if j != k], axis=0)[m] * w[m]
                sets = comparison_sets(ts, item, "category")
                for comp, members in sets.items():
                    rs = [ts.score(pat, item, template=loo)] if comp == "target" else [ts.score(pat, mm) for mm in members]
                    r = np.nanmean(rs) if rs else np.nan
```

Each template event is scored against its item's template rebuilt **without that event**. ▲ Note: the masks and weights (D2–D3) were still estimated from all template events, including the left-out one — the same mild circularity as the published analysis; a fully held-out check would re-fit the GLMs per fold.

## D7. The group raw table

[`rsaworkflow/workflows/group_template_rsa.py:103`](../workflows/group_template_rsa.py#L103-L108) — `group_template_rsa`

```python
            scores = load_scores(output_dir, roi, name, subjects)
            if scores is None:
                print(f"  (!) no scores for ROI {roi}, target set {name}; skipped")
                continue
            tag = f"roi-{roi}_target-{safe_name(name)}"
            scores.to_csv(os.path.join(raw_dir, f"{tag}_scores.csv"), index=False)
```

[`rsaworkflow/workflows/group_template_rsa.py:62`](../workflows/group_template_rsa.py#L62-L68) — `describe`

```python
    scores = scores[scores["comparison"].isin(g["comparisons"])].copy()
    scores["group"] = group_label(scores, list(g["group_by"] or []))
    by = [*KEY, "group"] + (["tr_offset"] if timecourse else [])
    subj = scores.groupby(["subject", *by], as_index=False).agg(
        z=("z", "mean"), n_trials=("z", "size"), **({"time_s": ("time_s", "first")} if timecourse else {}))
    summary = subj.groupby(by, as_index=False).agg(
        mean=("z", "mean"), sem=("z", stats.sem), n=("z", "count"), **({"time_s": ("time_s", "first")} if timecourse else {}))
```

The core stops here: every subject stacked, plus descriptive subject means. All study-specific inference starts from the raw table.

---

# Part E — post-hoc helpers (`_interactive_notebooks/rsa_posthoc_common.py`)

## E1. Subjects as the random effect, baselines and reference conditions

[`rsaworkflow/_interactive_notebooks/rsa_posthoc_common.py:143`](../_interactive_notebooks/rsa_posthoc_common.py#L143-L146) — `subject_means`

```python
def subject_means(df: pd.DataFrame, by, value: str = "z") -> pd.DataFrame:
    by = list(by)
    return df.groupby(["subject", *by], as_index=False, dropna=False).agg(
        **{value: (value, "mean"), "n_trials": (value, "size")})
```

[`rsaworkflow/_interactive_notebooks/rsa_posthoc_common.py:151`](../_interactive_notebooks/rsa_posthoc_common.py#L151-L156) — `baseline_correct`

```python
    a, b = baseline_trs
    keys = ["subject", *by]
    base = sub[(sub[x] >= a) & (sub[x] < b)].groupby(keys, dropna=False)[value].mean().rename("_base")
    out = sub.join(base, on=keys)
    out[value] = out[value] - out["_base"]
    return out.drop(columns="_base")
```

[`rsaworkflow/_interactive_notebooks/rsa_posthoc_common.py:162`](../_interactive_notebooks/rsa_posthoc_common.py#L162-L166) — `subtract_reference`

```python
    keys = ["subject", *by]
    ref = sub[sub[column] == reference].set_index(keys)[value].rename("_ref")
    out = sub[sub[column] != reference].join(ref, on=keys)
    out[value] = out[value] - out["_ref"]
    return out.drop(columns="_ref").dropna(subset=[value])
```

Trials are averaged within subject before any test; baselines are subtracted per subject × condition curve; "operation minus maintain" is matched on subject and TR.

## E2. Previous-trial columns

[`rsaworkflow/_interactive_notebooks/rsa_posthoc_common.py:109`](../_interactive_notebooks/rsa_posthoc_common.py#L109-L118) — `add_previous_trial_columns`

```python
    ev = event_table(df).sort_values(["subject", "run_id", "onset"])
    out = ev[EVENT_KEY].copy()
    needed = list(dict.fromkeys(columns + [a for a, _ in matches.values()]))
    for col in needed:
        out[f"prev_{col}"] = ev.groupby(["subject", "run_id"])[col].shift(1).values
    pairs = {f"same_{c}": (c, c) for c in columns} | matches
    for name, (a, b) in pairs.items():
        prev = out[f"prev_{a}"]
        same = ev[b].astype(str).values == prev.astype(str).values
        out[name] = np.where(prev.isna(), None, np.where(same, "same", "different"))
```

Computed on the event sequence within each run, **before** any exclusion in the notebook, so "the previous trial" is always the one the participant actually saw; `matches` compares a column of trial N with a different column of trial N+1 (Kim et al.: the replacement category of N vs. the category of N+1).

## E3. Tests

[`rsaworkflow/_interactive_notebooks/rsa_posthoc_common.py:206`](../_interactive_notebooks/rsa_posthoc_common.py#L206-L212) — `one_sample_by_bin`

```python
    for keys, part in binned.groupby([*by, "bin_start", "bin_stop", "bin"], dropna=False):
        x = part[value].dropna()
        t, p = stats.ttest_1samp(x, 0.0, alternative=alternative) if len(x) > 1 else (np.nan, np.nan)
        rows.append({**dict(zip([*by, "bin_start", "bin_stop", "bin"], keys)), "n": len(x), "mean": x.mean(),
                     "sem": stats.sem(x) if len(x) > 1 else np.nan, "t": t, "p": p})
    out = pd.DataFrame(rows).sort_values([*by, "bin_start"])
    out["p_fdr"] = out.groupby(list(by), dropna=False)["p"].transform(lambda p: fdr(p)) if by else fdr(out["p"])
```

[`rsaworkflow/_interactive_notebooks/rsa_posthoc_common.py:276`](../_interactive_notebooks/rsa_posthoc_common.py#L276-L284) — `paired_contrast`

```python
    for keys, part in groups:
        keys = keys if isinstance(keys, tuple) else (keys,)
        wide = part.pivot_table(index="subject", columns=column, values=value, aggfunc="mean")
        if a not in wide or b not in wide:
            continue
        both = wide[[a, b]].dropna()
        diff = both[a] - both[b]
        t, p = stats.ttest_rel(both[a], both[b]) if len(both) > 1 else (np.nan, np.nan)
        lo, hi, p_boot = bootstrap_ci(diff.values)
```

[`rsaworkflow/_interactive_notebooks/rsa_posthoc_common.py:312`](../_interactive_notebooks/rsa_posthoc_common.py#L312-L314) — `mixed_model`

```python
        try:
            fit = smf.mixedlm(formula, part, groups=part["subject"], re_formula=re_formula).fit(reml=True)
            ci = fit.conf_int()
```

- Bin tests: one-sample t per condition × bin, FDR across the bins of each condition.
- Paired contrasts use only subjects with **both** levels (e.g. Bretton's n = 18 with remembered and forgotten items), with a subject-resampling bootstrap CI of the difference.
- Mixed models use every trial with subject as the grouping factor; `re_formula="~C(memory)"` gives random slopes, i.e. (1 + memory | subject).
