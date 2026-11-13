# Alignment with the lab's SPM12 Clearvale pipeline

The lab has a second Clearvale implementation of the Kim et al. (2020) item-level RSA:
`cval_mni_blanca_pipeline_spm12` (v4.4.5, `/pl/active/banich/studies/Clearvale/scripts/jake_scripts/rsa/`).
Both pipelines use the same 68 subjects, the same bids-hcp MNI152NLin6Asym BOLD data, TR 0.46 s, and no
smoothing. Both also compute target similarity the same way:
- **Template:** the mean of the localizer view TRs shifted 4.6 s, weighted by the image's own
  item-vs-all-other-items t map inside its category mask (p < .05, k ≥ 10).
- **WM pattern:** weighted by the same map, inside the same mask.
- **Score:** Pearson r, Fisher z.

They differ in the choices below. Each one this pipeline supports is an option. The
**SPM12-aligned** setup turns them all on together:

- core config [configs/config-cv-kim2020-spm12-aligned.json](configs/config-cv-kim2020-spm12-aligned.json)
- post-hoc preset `PRESET = "spm12_aligned"` (or `RSA_POSTHOC_PRESET=spm12_aligned`) in
  [notebooks/05_wm_operations_posthoc.ipynb](notebooks/05_wm_operations_posthoc.ipynb)

The defaults are unchanged.

| # | Step | SPM12 pipeline | This pipeline (default) | Aligned setting |
|---|---|---|---|---|
| 1 | Categories in the templates and category GLM | Face and place only. Neutral objects aren't modelled, so each mask is face > place or place > face. | Face, place and object. Each mask is category > mean of the other two. | `template_rsa.templates.events`: `(positive\|negative\|neutral)_(face\|place)` |
| 2 | Category GLM events | One boxcar per image | Images merged into mini-block boxcars | `category_mask.block_merge_gap_seconds: null` |
| 3 | Confounds and censoring | motion6 and run indicators; no scrubbing; no dropped volumes | motion6 plus FD > 0.9 mm spike regressors; first 10 volumes censored | `outlier_regex` removed; `censoring.drop_initial_volumes: 0` |
| 4 | Pattern time series | Detrend, then per-run z-score (MVPA pattern cache); no high-pass | Detrend, z-score and 128 s high-pass | `timeseries.high_pass: null` |
| 5 | Minimum category-mask size | 3 voxels | 10 voxels | `category_mask.min_voxels: 3` |
| 6 | Encoding fidelity (and N+1) | Mean of the 6 single-TR correlations (Kim's method). Correlating the mean pattern is their sensitivity analysis. | Correlate the window-mean pattern once | `targets.encoding_fidelity.aggregate: "volumes"` (new option) |
| 7 | Timecourse baseline | **Raw** is their primary result. A Kim version subtracts one group mean per operation × valence (all participants, first 6 TRs). | Each subject's own view-period mean, per operation | Notebook `BASELINE_MODE`: `"none"` (their primary) or `"group"` (Kim's MATLAB code) |
| 8 | Trials in the timecourse | Only trials with all 30 TRs | Every trial; truncated ones contribute the TRs they have | Notebook `COMPLETE_TRIALS_ONLY = True` |
| 9 | Valence contrast | positive − negative | negative − positive | Notebook `VALENCE_CONTRAST` |
| 10 | N+1 post hocs | Tukey-Kramer for the planned pairs (suppress vs clear, maintain and switch); 15-trial equal-count bootstrap | FDR-corrected paired t-tests | Added to the notebook for both presets: `rm_tukey`, `equal_trial_bootstrap` |

Two standardization choices look different but aren't. Population vs sample SD rescales every voxel by the
same factor, which doesn't change any correlation. Their "+10 TR" shift is our 4.6 s window offset.

## Still different

- **GLM engine:** SPM12 canonical HRF (microtime 16/8) here is nilearn's `spm` HRF with a cosine drift
  model at 128 s. Their t maps are SPM's. Masks and weights will differ slightly even with every setting
  matched.
- **VVS mask file:** they use `mask_vvps_temporal.nii.gz` resampled to the BOLD grid. The HPC copy of the
  aligned config points at that file; the repo copy keeps the local aparc+aseg ROI.
- **Their GLM trims each run's volumes before its first event.** Here, every volume enters the GLM.
- **Their object handling isn't documented.** Neutral objects are probably absent from their item GLM as
  well, but that isn't certain from the code. The aligned config leaves them out of both GLMs.

## Effect on the current HPC results (68 subjects)

These are from the post-hoc notebook run on the existing group tables, so only settings 7–9 change; 1–6
need stages 2–3 rerun with the aligned config. For the removal timecourses (operation minus maintain,
2.76–9.66 s):

| | Default | `spm12_aligned` preset |
|---|---|---|
| Clear below maintain | 4 of 5 windows, p_fdr < .05 (first: 2.76–4.14 s) | No window survives FDR |
| Operation main effect (whole window) | F = 10.5, p < .001 | F = 1.35, p = .26 |
| Operation × valence | F = 4.36, p = .015 | F = 3.41, p = .036 |
| Clear, valence difference | ns | positive − negative = +0.013, p = .011 |

So the clear-removal result depends on the baseline. Before the cue, clear trials happen to start higher
than maintain trials. Subtracting each subject's view-period mean removes that offset; the raw comparison
keeps it.

For sub-090, the aligned core config changes:
- the category masks: face 670 voxels (was 951), place 1,099 (was 1,479)
- encoding fidelity: target 0.112 (was 0.153), because the mean of single-TR correlations is lower than
  the correlation of the averaged pattern
