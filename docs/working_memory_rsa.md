# RSA for a working-memory task: design notes

This is a working draft for planning. It covers which questions RSA can answer about working memory (WM), the fMRI design and analysis traps specific to WM, and how each analysis maps onto this pipeline: what runs today, and what needs new code. The specific methods will come from the publications that will drive this analysis.

[examples/config-wm-delay.example.json](../examples/config-wm-delay.example.json) is a skeleton config for the analyses marked **ready**. Its task has 4 orientations, encoding and delay phases, fMRIPrep confounds, crossnobis distances, and V1, IPS and dlPFC ROIs.

Vocabulary: "the remembered item" below is the WM-literature sense (the stimulus held in mind). In the pipeline's terms, each stimulus is an **item** (the finest level of `events.hierarchy`), a single encoding or delay event of it is an **item**, and coarser hierarchy levels (phase, category) are **conditions**. Factors that cross the hierarchy (stimulus identity across phases, cue status) are separate columns, used for filtering or as explicit `groups` models.

## 1. Questions RSA can answer in WM

| Question | RSA form |
|---|---|
| Is the remembered item represented during the delay, and where? | Delay-period item RDM vs. a stimulus model (identity, feature distance, category) |
| In what format: sensory, categorical or abstract? | Compare competing model RDMs, e.g. a perceptual-similarity RDM from a localizer vs. a category RDM vs. a semantic RDM. Weighted models show their relative contribution. |
| Is the delay code the same as the encoding (perceptual) code? | Cross-phase RSA: the encoding × delay block of an RDM over phase × stimulus items. A transformed code shows as a delay RDM matching the model without matching encoding. |
| How do attended and unattended items differ (retro-cue / priority)? | Items = stimulus × priority state. Typical findings: the unattended item weakens in the RDM, or appears in a transformed code. |
| How does representation change over the trial? | Timecourse RSA: RDM per TR after trial onset, with the model fit traced over time. Temporal generalization uses cross-time RDMs. |
| Does representational strength predict behavior? | Trial-level pattern similarity (item-specific vs. other items) related to accuracy, precision or RT |
| Does load change the representation? | Load as an attribute; RDMs or model fits within each load level |

## 2. WM-specific design and analysis traps

1. **Phases overlap in time.** Encoding, delay and probe responses overlap: with a 6–12 s delay, the encoding response bleeds into the delay estimate.
   - Model each phase as its own regressor. Use LSS for trial-wise estimates.
   - Prefer jittered delays and a few catch trials (no delay, or no probe), which make phase estimates separable.
   - Check the correlation between the encoding and delay regressors in the design matrix (to be added to run QC; see §4).
2. **The probe is confounded with the memory item.** The probe or response often resembles the remembered item, so model the probe phase separately. Here that means probe events are nuisance regressors: they are not in `events.item_types`, so they get their own regressor (`model_unlabeled_events`). Counterbalance probe identity.
3. **Within-run similarity is inflated by shared noise and temporal proximity.** Compare patterns only across runs. Use cross-validated distances (crossnobis, `cv_descriptor: run`). This needs every item in ≥2 runs (`validate_rsa_config.py` checks this).
4. **Univariate load and difficulty effects.** Overall activation differences such as load or effort inflate pattern distances between conditions.
   - Correlation distance or `rdm.remove_voxel_mean` removes them.
   - Alternatively, include a univariate "activation magnitude" model RDM as a competing model.
5. **Few trials per item.** Item-level RDMs need items repeated across runs. With unique stimuli, bin features (e.g. orientation bins) or use trial-level analyses (§3E).
6. **Motion around responses.** Response-locked motion is common in WM tasks. Use fMRIPrep confounds plus censoring (see the main README) and inspect the run QC page.
7. **Space and ROIs.** Native space (fMRIPrep `space-T1w`) with subject-specific ROIs (retinotopic V1–V3, IPS0–5, dlPFC/IFS) is standard for item-level WM codes. MNI space is fine for group searchlight later.

## 3. Analysis families and pipeline support

### A. Delay-period item RDM: ready
Give the delay events the hierarchy `trial_type > phase_stimulus` and set `events.item_types: ["delay"]` (encoding events then become nuisance), or keep both phases as items and look at the delay block (§B). Compare the delay RDM with:

- **Feature models:** orientation as `[cos 2θ, sin 2θ]` features, color angle, etc.
- **Categorical models:** a coarser hierarchy level (`level`), or a grouping that crosses it (`groups`, e.g. stimulus identity across phases).
- **File models:** a perceptual RDM from a localizer or a behavioral confusion RDM.

Use crossnobis on betas, cross-validated over runs.

### B. Encoding vs. delay (cross-phase): partly ready
Items = phase × stimulus: the skeleton config's events carry a `phase_stimulus` column (`encoding_0` … `delay_135`) under `trial_type`, so the hierarchy `trial_type > phase_stimulus` nests. One crossnobis RDM then contains three blocks: within-encoding, within-delay, and encoding × delay.

- **Ready:** model RDMs over the whole 2n × 2n matrix. Examples:
  - `stimulus_identity`: same stimulus counts as similar regardless of phase (a `groups` model, since identity crosses the phases).
  - `phase`: encoding vs. delay (`{"type": "categorical", "level": "trial_type"}`).
  - A weighted model of the two.
- **New:**
  - *Block-restricted comparison.* Fit a model only within the encoding × delay block, or only within-delay. rsatoolbox requires matching NaN positions, so the plan is a model `mask` option that NaNs the same cells in the data RDM before comparing. This is small and contained.
  - *Cross-phase generalization index.* Same-item minus different-item distance in the cross block, a cheap, interpretable summary, reported per ROI and subject.

### C. Priority / retro-cue: ready (with suitable events)
Items = stimulus × status, e.g. `cued_0`, `uncued_0`, …, as an events column under a `status` level. Models:

- the cued item represented
- the uncued item represented
- a transformed ("rotated") code: a file RDM, or a feature model with remapped features

Weighted models test whether both items contribute.

### D. Timecourse RSA: new
This is the RSA analogue of mvpa-decoding's `timecourse_decoding`, with the same window syntax (`onset` / `offset_end` ± seconds).

- **Patterns:** one pattern per item per TR offset within the window. Two options:
  - `volume_average` with a single-TR window per lag; no HRF assumption.
  - A FIR GLM (nilearn `hrf_model: "fir"`, `fir_delays`); the cleaner option for overlapping phases.
- **Per lag:** an RDM and model fits, giving a model-fit time course per subject.
- **Group:** a t-test at each lag, with cluster-based permutation over time; a plot with phase boundaries marked.
- **Option:** temporal generalization, a cross-lag RDM (lag i vs. lag j patterns, cross-validated over runs) testing whether the code is stable or dynamic.

This is the biggest addition. It needs a `timepoint` pattern level, a lag dimension in the RSA and group stages, and a report page.

### E. Trial-level similarity (reinstatement / ERS): new
This uses LSS trial patterns (ready) but doesn't build an RDM. For each trial, it compares:

- the trial's encoding pattern with its own delay pattern (same trial, same item), versus
- delay patterns of other trials with the same item, versus
- other trials with different items.

All comparisons are made across runs. The output is a per-trial table of similarities, joined to behavior from events.tsv (accuracy, recall error, RT). These feed a mixed model, either in Python (statsmodels `mixedlm`) or exported for R. This is a separate stage that reads the existing pattern index.

### F. Linking to decoding (mvpa-decoding): small
- Crossnobis distance is a cross-validated estimate of linear discriminability, so RSA and decoding results should agree in sign.
- Classifier confusion matrices from mvpa-decoding can enter as data RDMs. An `import_rdm` path (CSV → rsatoolbox RDMs) would let the group stage, noise ceiling and report run on decoding outputs too.

## 4. Proposed backlog (to reprioritize once the publications are in)

| # | Feature | Size | Unlocks |
|---|---|---|---|
| 1 | Model-RDM masks (block-restricted comparisons) | S | B: cross-phase fits; any within/between-block test |
| 2 | Design-matrix QC: encoding–delay regressor correlation, VIF, per run | S | Trap 1; feeds the run QC report page |
| 3 | Timecourse RSA (FIR / per-lag patterns, lag-wise group stats, plot) | L | D |
| 4 | Trial-level similarity stage + behavioral join | M | E |
| 5 | Cross-phase generalization index | S | B |
| 6 | Import external RDMs (decoding confusion, behavior) | S | F |
| 7 | Searchlight group stats (deferred) | L | Whole-brain localization of A–C |

Things to pin down from the publications:

- Phase timing and delay length.
- Whether patterns are betas or TR-averages.
- Distance measure and comparison method.
- ROI definitions: retinotopic, atlas, or functional localizer.
- Whether they report timecourses.
- Which behavioral measure links to representational strength.
