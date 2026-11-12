# Clearvale (CV) template RSA

Everything specific to the Clearvale study: the analysis configs, the scripts that turn the raw study
files into BIDS inputs, the Flywheel notebooks, and the record of data deviations. The analysis
itself is the core pipeline (`workflows/`), driven by the configs here.

```
examples/clearvale/
├── configs/
│   ├── config-cv-kim2020-denoised.json   # main analysis: motion confounds + censoring (first 10 vol, FD > 0.9 mm)
│   ├── config-cv-kim2020-raw.json        # identical, without confounds / censoring
│   └── config-cv-traditional-rdm.json    # sanity check: classic RDM of the localizer (no masks / weights)
├── scripts/
│   ├── fetch_bold.py                # Flywheel -> only the BOLD runs, motion files and aparc+aseg a subject needs
│   ├── make_events.py               # PsychoPy stim.csv -> BIDS events.tsv (all sessions with --all)
│   ├── export_events.py             # events.tsv -> Flywheel-style names, zip for the HPC, Flywheel upload
│   ├── audit_stim.py                # audit of every session's PsychoPy logs (feeds analysis_deviations.md)
│   ├── make_confounds.py            # motion parameters + framewise displacement for the GLMs / censoring
│   ├── make_brain_mask.py           # aparc+aseg brain mask on the BOLD grid (traditional RDM config only)
│   ├── fig_template_matching.py     # methods figure: how one presentation is matched to every template
│   └── compare_traditional_rdm.py   # template RSA vs. traditional RDM on one subject
├── notebooks/
│   ├── 01_download_from_flywheel.ipynb     # one subject's stim files + preprocessed outputs
│   ├── 02_methods_figures.ipynb            # methods figures for the report (sub-090)
│   ├── 03_download_all_stim.ipynb          # every session's PsychoPy files
│   └── 04_run_lengths_from_flywheel.ipynb  # volumes per run from the DICOM metadata -> sourcedata/run_lengths.csv
└── analysis_deviations.md           # data deviations and how they are handled (for the report)
```

## Where the data lives

The data are not in git (`data/` is ignored). Everything here expects the Clearvale BIDS folder at
`<repo>/data/cv-bids`, so on the HPC make that a symlink:

```bash
mkdir -p data && ln -s /path/to/Clearvale/cv-bids data/cv-bids
```

The scripts and notebooks also honour `CV_BIDS=/path/to/cv-bids` instead. The configs use paths relative
to the repo root (`data/cv-bids/...`), so run the pipeline from the repo root. The folder holds:

```
data/cv-bids/
├── sub-<s>/ses-<S>/func/*_events.tsv                       # from make_events.py
├── sourcedata/stim/...  sourcedata/run_lengths.csv          # notebooks 03 and 04
├── derivatives/bids-hcp/sub-<s>/ses-<S>/func/*_bold.nii.gz   # fetch_bold.py (bids-hcp outputs)
├── derivatives/confounds/...                                # make_confounds.py
└── out/kim2020_denoised/ ...                                # pipeline outputs
```

### Events on Flywheel and the HPC

The events from `make_events.py` are also on Flywheel, in each BOLD acquisition, under Flywheel-style
names: `func-bold_task-loc_dir-pa_run-01_desc-rsainputs_events.tsv` (next to the original
`desc-stimulus` / `desc-blocks` files). For the HPC the same files are zipped under BIDS names, in
their folders (`sub-090/ses-A1/func/sub-090_ses-A1_task-loc_dir-pa_run-01_desc-rsainputs_events.tsv`).
`export_events.py` makes both (a dry run on Flywheel unless `--upload`); unzip into the cv-bids folder:

```bash
unzip cv-events-rsainputs.zip -d data/cv-bids
```

These match the configs' `events_glob` as they are. Keep only one copy of each run's events under
cv-bids (these or the plain `make_events.py` output), or the run is read twice. (Flywheel-style names
without `sub-` / `ses-` also work: the pipeline then takes subject and session from the folders.)

## Run order

Flywheel steps use the saved CLI login (`fw login` once) or, in notebooks 01 and 03, a hidden prompt; no key is ever stored in a file.

1. **Run lengths and stimulus files (once, all subjects):** notebooks 04 and 03.
2. **Events (once, all sessions):** `python examples/clearvale/scripts/make_events.py --all`, then
   `python examples/clearvale/scripts/audit_stim.py` to refresh the audit, and
   `python examples/clearvale/scripts/export_events.py --upload` to put them on Flywheel and zip them.
3. **Per subject:**
   ```bash
   python examples/clearvale/scripts/fetch_bold.py --subject 090      # needs the Flywheel SDK
   python examples/clearvale/scripts/make_confounds.py --subject 090 --session A1
   ```
4. **Analysis:** list the subjects in the config's `pipeline.subjects` (or set it to `null` to run every
   subject under `derivatives/bids-hcp`), then
   ```bash
   python workflows/run_pipeline.py --config examples/clearvale/configs/config-cv-kim2020-denoised.json
   ```
   or, on the cluster,
   ```bash
   bash slurm/0_submit_rsa_pipeline.sh examples/clearvale/configs/config-cv-kim2020-denoised.json
   ```
   Repeat with `config-cv-kim2020-raw.json` for the no-denoising comparison.

The pipeline runs the template stage (templates + `kim2020` and `cv` validation) and the fidelity stage
(WM timecourse and encoding fidelity against each image's own template), the group stages, and the
common report (`out/kim2020_denoised/group/rsa_report.pdf`; per subject with
`workflows/generate_report.py --subject 090`).

## Open decisions

See `analysis_deviations.md`: whether to model the 710- vs. 770-volume run-length cohorts, and the subject
inclusion list (sub-104 pilot, sub-023 with 5 of 8 WM runs).
