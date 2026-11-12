# Post-hoc notebooks

The core pipeline is design-agnostic. It builds templates, scores target events against them, and writes one **group raw table** per ROI and target set:

`<output_dir>/group/fidelity/raw/roi-<roi>_target-<set>_scores.csv`

Each table has one row per subject × event × [volume] × weighting level × comparison set. Every events.tsv column rides along. Anything specific to a study design starts from that table, in a notebook here, as in mvpa-decoding:

- which operations or groups to compare
- what counts as remembered
- which previous trial matters
- baseline and reference conventions
- statistical tests

| Notebook | Paper | Analyses |
|---|---|---|
| [kim2020_removal_operations.ipynb](kim2020_removal_operations.ipynb) | Kim et al. (2020) *Nat Commun* | timecourses by operation; removal = operation − maintain in 3-TR bins with onsets; replacement-category tracking on replace trials; encoding fidelity (target / related / others); trial N+1 proactive interference |
| [bretton2024_ltm_fidelity.ipynb](bretton2024_ltm_fidelity.ipynb) | Bretton et al. (2024) *J Cogn Neurosci* | memory flag from the post-test responses; operation joined from the WM phase; template check; fidelity remembered vs. forgotten per operation × ROI × level (paired t, bootstrap CI); mixed models; ROI specificity |

[rsa_posthoc_common.py](rsa_posthoc_common.py) holds what the notebooks share. It knows nothing about any study:

- **loading:** the group raw table and the event table
- **table helpers:** previous-trial columns, item columns joined across phases
- **aggregation:** subject means, baseline correction, reference subtraction
- **statistics:** bins, one-sample tests with FDR, rm-ANOVA, pairwise tests, paired contrasts with bootstrap CIs, mixed models
- **plots, outputs, toy data**

## Using a notebook

1. Run the core pipeline for the study:

   ```bash
   python workflows/run_pipeline.py --config examples/config-kim2020-ncomms.example.json
   ```

2. Open the notebook from the repository root or from this folder. Edit its **Configuration** cell:
   - the config path
   - the ROI and weighting level
   - the event column names
   - the timing in seconds
   - alpha, FDR and the number of bootstraps

3. Run all cells. The figures go to `stats_and_figures.pdf`, alongside one CSV per table and `run_settings.json`. They are written in `<output_dir>/posthoc/<study>/…`, or in `$RSA_POSTHOC_OUT`.

`RSA_CONFIG` and `RSA_POSTHOC_OUT` override the config path and output folder without editing the notebook. `tests/test_notebooks.py` uses them to execute both notebooks against a synthetic lab-style dataset (`tests/synthetic_lab.py`).

## Adding a notebook for a new design

Copy the closest notebook. Point its configuration cell at your config and target-set names, then replace the analysis sections. Put new design-specific logic in the notebook. Move a function into `rsa_posthoc_common.py` only once a second notebook needs it, and keep it free of study names and column names.
