# rsaworkflow

A config-driven representational similarity analysis (RSA) workflow for fMRI, in Python.
Structured like [mvpa-decoding](https://github.com/intermountainneuroimaging/mvpa-decoding): one JSON config file describes the whole analysis, and every stage reads its own section of it.

It is built on public tools:

| Job | Tool |
|---|---|
| First-level GLMs, masking, EPI masks | [nilearn](https://nilearn.github.io) |
| RDMs (correlation, Euclidean, Mahalanobis, crossnobis, Poisson), RDM comparison, noise precision, searchlight, bootstrap inference | [rsatoolbox](https://rsatoolbox.readthedocs.io) |
| Group statistics, multiple-comparison correction | scipy, statsmodels |

## Layout

```
rsaworkflow/
├── utils/
│   ├── rsa_common.py              # config defaults/loading, query DSL, volume math, model + data RDM helpers
│   ├── rsa_plots.py               # RDM / MDS / model-fit / weight plots (shared by stages 3-5)
│   ├── preprocessing.py           # confounds (fMRIPrep / TSV), censoring, per-run QC
│   └── validate_rsa_config.py     # stage 1b: check the config against the event table
├── workflows/
│   ├── generate_master_spreadsheet.py   # stage 1: BIDS events.tsv -> one row per event
│   ├── estimate_patterns.py             # stage 2: one pattern image per item x run, or per trial (LSA / LSS)
│   ├── rsa_workflow.py                  # stage 3: subject RDMs per ROI / searchlight + model comparison
│   ├── group_rsa.py                     # stage 4: mean RDM, MDS, noise ceiling, group inference, weighted models
│   ├── generate_report.py               # stage 5: multi-page PDF report (group, or one subject)
│   └── run_pipeline.py                  # all stages locally, from one config
├── slurm/                         # same stages as chained SLURM jobs (0_submit_rsa_pipeline.sh)
├── tutorial/                      # Haxby 2001 validation dataset: fetch script, configs, results
├── examples/                      # fMRIPrep study template; working-memory skeleton config
├── docs/                          # design notes (working_memory_rsa.md)
└── tests/                         # unit tests + synthetic end-to-end test
```

## Install

Requires Python 3.10 or newer.

```bash
python3 -m venv ~/.venvs/rsaworkflow
```

```bash
~/.venvs/rsaworkflow/bin/pip install -r requirements-dev.txt
```

```bash
~/.venvs/rsaworkflow/bin/python -m pytest
```

On Apple-silicon Macs, use a native (arm64) Python. An x86_64 conda under Rosetta installs an arm64 rsatoolbox wheel that won't import.

## Running

```bash
python workflows/run_pipeline.py --config tutorial/config-haxby-rsa.example.json
```

You can also run each stage by hand, the same way mvpa-decoding does:

```bash
python workflows/generate_master_spreadsheet.py --config CONFIG
python utils/validate_rsa_config.py --config CONFIG
python workflows/estimate_patterns.py --config CONFIG --subject 1
python workflows/rsa_workflow.py --config CONFIG --subject 1
python workflows/group_rsa.py --config CONFIG
python workflows/generate_report.py --config CONFIG                 # group PDF
python workflows/generate_report.py --config CONFIG --subject 1     # single-subject PDF
```

On a cluster, set `pipeline.subjects` in the config, then:

```bash
PYTHON=~/.venvs/rsaworkflow/bin/python SBATCH_EXTRA="--account=X --partition=Y" bash slurm/0_submit_rsa_pipeline.sh CONFIG
```

Relative paths in the config are resolved against the directory you run from (the repo root).

## Pipeline stages

1. **Event table** (`event_extraction`). Every `events.tsv` under `bids_root` is matched to its preprocessed BOLD file (`derivatives_root` + `bold_glob`, with `{subject}`, `{session}`, `{task}` and `{run}` filled from the events filename). The output has one row per event and keeps every events.tsv column, plus `subject`, `session`, `task`, `run`, `run_id`, `trial_index`, `tr`, `n_frames` and `boldfile`. Fixation, block-marker and post-RT rows are dropped, as in mvpa-decoding.
2. **Validation.** Checks query syntax and that each item matches events for every subject. It also checks model attributes, features and files against the items, that mask files exist, that method names are valid, and that cross-validated methods have at least 2 runs. Overlapping items produce a warning.
3. **Patterns** (`pattern_estimation`). Produces one 3-D image per item per run (`condition_per_run`) or per event (`trial`). Every run of a subject uses the same brain mask. If `mask_pattern` matches several files (e.g. fMRIPrep's per-run `desc-brain_mask`), their intersection is used. Nuisance handling is described [below](#preprocessing-fmriprep-inputs).
   - `glm`: a nilearn GLM per run. Events that match no item are modeled as nuisance regressors. Saves t, z or beta maps. For `unit: "trial"`, `glm.trial_model` picks how single trials are estimated:
     - `lss` (default): least-squares-separate (Mumford et al., 2012). One GLM per trial, with that trial as the only regressor of interest. With `lss_other: "condition"`, the other trials are grouped by item (LSS-N); with `"single"`, they share one regressor. The run is loaded once, and each trial costs one design matrix and one fit.
     - `lsa`: least-squares-all. One GLM per run, with every trial as its own regressor. This is fine for slow designs, but the estimates get noisy when trials overlap in time.
   - `volume_average`: cleans the run (detrend, z-score, optional high-pass), then averages the volumes each event spans, shifted by `hemodynamic_lag`.
4. **Subject RSA** (`rois`, `searchlight`, `rdm`, `model_rdms`, `comparison`). For each ROI:
   - Masks the patterns and drops dead voxels.
   - Computes the RDM with `rdm.method`: patterns of the same item are averaged, or cross-validated over `cv_descriptor` folds for crossnobis and poisson_cv.
   - Compares the RDM with each model RDM.

   The optional searchlight writes one native-space model-fit map per model.
5. **Group** (`group`). Computes the mean RDM and an MDS plot, plus a leave-one-subject-out noise ceiling (Nili et al., 2014). Models are tested against 0 and against each other with one of three methods:
   - `ttest`: subject random effects.
   - `wilcoxon`
   - `bootstrap`: rsatoolbox `eval_bootstrap_rdm` over subjects.

   p-values are FDR-corrected within each ROI.

   **Weighted models** (`weighted_models`) combine existing model RDMs with fitted, non-negative weights (rsatoolbox `ModelWeighted`). Extra free parameters would inflate an in-sample fit, so these models are evaluated by cross-validation over subjects (`group.crossval`, rsatoolbox `crossval`): weights are fit on the training subjects and scored on the held-out subject. By default the fixed models are scored in the same folds, so all models are directly comparable. The stage tests fold scores against 0 and compares each weighted model with each of its components. It also reports the cross-validated noise ceiling and the weights fit on all subjects.
6. **Report.** A PDF with these pages:
   - Analysis settings.
   - Pattern inventory: patterns per subject and item.
   - The model RDMs.
   - For each ROI: the mean RDM, MDS, fixed and cross-validated model fits with noise ceiling, weights, statistics tables, and every subject's RDM.

   The `--subject` version shows one subject's RDMs, MDS and model fits.

## Preprocessing (fMRIPrep inputs)

The pipeline assumes minimally preprocessed data: fMRIPrep's `desc-preproc_bold` in `space-T1w` or MNI space. The GLM-stage steps that fMRIPrep leaves to the user are optional, are set in `pattern_estimation`, and work the same for every pattern method:

| Step | Config | What happens |
|---|---|---|
| **Nuisance regression** | `confounds: {"source": "fmriprep", "strategy": [...], ...}` | nilearn [`load_confounds`](https://nilearn.github.io/stable/modules/generated/nilearn.interfaces.fmriprep.load_confounds.html) reads the run's `desc-confounds_timeseries.tsv`. Strategies: `motion` (`basic` 6, `derivatives` / `power2` 12, `full` 24), `wm_csf`, `global_signal`, `compcor` (+ `n_compcor`), `high_pass` (fMRIPrep cosines), `ica_aroma`, `scrub`, `non_steady_state`. For `glm`, the confounds go into the design matrix. For `volume_average`, they are regressed out in the cleaning step. |
| | `confounds: {"source": "file", "file_pattern", "columns", "regex"}` | Explicit columns from any TSV |
| **Censoring / scrubbing** | `confounds.strategy` with `scrub` (`fd_threshold`, `std_dvars_threshold`, `scrub` min segment) and/or `non_steady_state`; `censoring.drop_initial_volumes` | Flagged volumes are dropped from the fit (`censoring.mode: "censor"`), or modeled with one spike regressor each (`"regress"`, which keeps the time series continuous). For LSS, design matrices are built on the full timeline before censored rows are dropped, as `FirstLevelModel` does. For `volume_average`, censored volumes are excluded from the event averages. |
| **Temporal filtering** | `glm.drift_model` + `glm.high_pass`; `volume_average.high_pass` / `low_pass` / `detrend` | GLM: cosine drift regressors (the usual approach). Set `drift_model: null` when you use fMRIPrep's `high_pass` cosines instead; the validator warns if you use both. `volume_average`: Butterworth band-pass in nilearn `signal.clean`. |
| **Prewhitening** | `glm.noise_model` | `"ar1"` (default), `"arN"` (e.g. `"ar2"`), or `"ols"` (none). This applies to the per-run GLM and to every LSS fit. |
| **Scaling** | `glm.signal_scaling` | `0` scales each voxel to percent signal change (default); `False` does no scaling. |
| **Smoothing** | `glm.smoothing_fwhm` / `volume_average.smoothing_fwhm` | Off by default; most RSA leaves the data unsmoothed. |

Each run's QC goes to `sub-<s>/patterns/sub-<s>_run-qc.csv`: volumes flagged, % flagged, number of confound regressors, and mean and max FD. The report's "Data inventory" page shows it.

The validator checks for:

- missing confound files
- the incompatible CompCor-without-`high_pass` combination
- a double high-pass filter
- global signal regression (flagged as a reminder that it alters pattern similarity)

[examples/config-fmriprep-template.example.json](examples/config-fmriprep-template.example.json) sets a common default: 24 motion parameters, WM/CSF, fMRIPrep cosines, scrubbing at FD > 0.5 mm or std DVARS > 1.5, AR(1), and no smoothing.

## Config reference

Top-level sections (defaults live in `utils/rsa_common.py::default_config`):

| Section | Keys |
|---|---|
| `event_extraction` | `bids_root`, `derivatives_root`, `bold_glob`, `events_glob`, `output_file` |
| `conditions` | `items`: `{name: query}`; the RDM row order is the key order. `attributes`: `{name: {attr: value}}`, used by categorical models. `include`: optional query that drops events before anything else. |
| `pattern_estimation` | `method` (`glm` or `volume_average`), `unit` (`condition_per_run` or `trial`), `mask_pattern` (a glob; multiple matches are intersected), `model_unlabeled_events`. `confounds`: `source` (`fmriprep` or `file`) plus `load_confounds` kwargs, or `file_pattern` / `columns` / `regex`. `censoring`: `drop_initial_volumes`, `mode` (`censor` or `regress`). `glm`: `output_type` (`t`, `z` or `beta`), `trial_model` (`lss` or `lsa`), `lss_other` (`condition` or `single`), and any `FirstLevelModel` kwarg (`noise_model`, `drift_model`, `high_pass`, `signal_scaling`, `smoothing_fwhm`, `hrf_model`, …). `volume_average`: `hemodynamic_lag`, `detrend`, `standardize`, `high_pass`, `low_pass`, `smoothing_fwhm`. |
| `rois` | `{name: {mask_pattern}}` |
| `searchlight` | `enabled`, `mask_pattern`, `radius_voxels`, `threshold`, `n_jobs` |
| `rdm` | `method` (`correlation`, `euclidean`, `mahalanobis`, `crossnobis`, `poisson` or `poisson_cv`), `cv_descriptor`, `noise.method` (`shrinkage_diag`, `shrinkage_eye`, `diag` or `full`), `remove_voxel_mean` |
| `model_rdms` | `{name: spec}`, where spec is one of:<br>`{"type": "categorical", "attribute": a}`: 0 if two items share the attribute, else 1<br>`{"type": "feature", "features": {item: vector}, "metric": ...}`<br>`{"type": "file", "path": "n_items x n_items .csv/.tsv/.npy"}` |
| `weighted_models` | `{name: {"components": [model_rdms names], "fitter": f}}`, where `f` is one of:<br>`optimize_positive` (default; non-negative weights, works with any comparison method)<br>`optimize`<br>`regress_nn` or `regress` (only with `corr`, `cosine`, `corr_cov` or `cosine_cov`) |
| `comparison` | `method`: any `rsatoolbox.rdm.compare` method (`corr`, `spearman`, `tau-a`, `rho-a`, `cosine`, `corr_cov`, `cosine_cov`, …) |
| `group` | `test` (`ttest`, `wilcoxon` or `bootstrap`), `alternative`, `noise_ceiling`, `multiple_comparisons` (a statsmodels method or `none`), `n_bootstrap`, `random_state`. `crossval`: `scheme` (`leave_one_subject_out` or `k_fold`), `k`, `include_fixed` (used by `weighted_models`; needs ≥3 subjects). |
| `pipeline` | `output_dir`, `subjects` |

Queries use the same grammar as mvpa-decoding:

- Leaves: `{"column", "match": "exact" | "in" | "regex", "value" | "values"}`.
- Nodes: `{"and": [...]}`, `{"or": [...]}`, `{"not": {...}}`.

Any events.tsv column can be queried.

Some method notes:

- For categorical models, which predict tied ranks, use `tau-a` (Nili et al., 2014).
- Use `cosine` or `corr` with crossnobis.
- Correlation distance on t-maps is the classic Kriegeskorte et al. (2008) setup.
- Crossnobis on betas is unbiased and is generally preferred when you have several runs.

## Outputs

```
<output_dir>/
├── master_spreadsheet*.csv
├── sub-<s>/patterns/   *_pattern.nii.gz, sub-<s>_patterns.csv
├── sub-<s>/rsa/        roi-<roi>/sub-<s>_roi-<roi>_rdm.{csv,hdf5,png}, sub-<s>_model-comparison.csv, searchlight/
└── group/              roi-<roi>/group_roi-<roi>_{mean-rdm.csv,mean-rdm.png,mds.png,models.png,pairwise.csv}
                        roi-<roi>/group_roi-<roi>_{crossval.csv,crossval-folds.csv,crossval-pairwise.csv,
                                                   crossval.png,weights.csv,weights.png}   (weighted_models)
                        group_model-comparison.csv, rsa_report.pdf
```

`sub-<s>/rsa/sub-<s>_rsa_report.pdf` is written by `generate_report.py --subject`.

## Validation dataset

See [tutorial/README.md](tutorial/README.md). It uses Haxby et al. (2001): 6 subjects, 8 object categories, 12 runs (OpenNeuro ds000105). The tutorial includes a re-run of the paper's split-half correlation analysis.

## Working-memory analyses

[docs/working_memory_rsa.md](docs/working_memory_rsa.md) is a design note covering:

- the questions RSA can answer about working memory
- the fMRI traps specific to WM: overlapping phases, probe confounds, within-run similarity, load effects
- which analyses run today (delay-period item RDMs, phase × item cross-phase RDMs, priority-state models)
- what needs new code (block-restricted model masks, timecourse RSA, trial-level reinstatement)

[examples/config-wm-delay.example.json](examples/config-wm-delay.example.json) is the matching skeleton config.

## Not yet implemented

- **Searchlight, deferred because of its compute cost.** Per-subject searchlight maps already work, but these are still missing:
  - Group statistics. The maps are in native space, so they need normalization (or an MNI-space `bold_glob`) and a second-level model.
  - Parallelizing over SLURM.
  - Searchlight pages in the PDF report.
- **Unweighted model families** (rsatoolbox `ModelSelect` or `ModelFamily`) and interpolated models.
