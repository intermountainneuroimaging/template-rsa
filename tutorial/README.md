# Tutorial / validation: Haxby et al. (2001)

**Dataset.** Haxby, Gobbini, Furey, Ishai, Schouten & Pietrini (2001), *Distributed and overlapping representations of faces and objects in ventral temporal cortex*, Science 293:2425 ([doi](https://doi.org/10.1126/science.1063736)). Raw data: [OpenNeuro ds000105](https://openneuro.org/datasets/ds000105).

The data have 6 subjects and 12 runs each (TR 2.5 s). Each run has one 24 s block of each of 8 categories: face, cat, house, chair, shoe, bottle, scissors and scrambled pictures. The release also includes a ventral-temporal (VT) mask plus face- and house-selective VT masks.

This is the same dataset as the mvpa-decoding tutorial. It is also the dataset that RSA-style analysis of fMRI grew out of: the paper's split-half correlation matrix is a cross-validated similarity matrix.

We use the motion-corrected release that nilearn and PyMVPA distribute. `fetch_haxby.py` re-lays it out as a BIDS-style tree:

```bash
python tutorial/fetch_haxby.py --subjects 1 2 3 4 5 6 --out data/haxby-bids --cache data/nilearn
```

Each block becomes one event, and `fetch_haxby.py` writes the stimulus hierarchy into the events files as columns, `animacy > superordinate > category` (e.g. `animate > animate > face`, `inanimate > object > chair`, `none > scrambled > scrambledpix`). The configs set `events.hierarchy` to those columns, so the RDM rows are the 8 categories (each run's block of a category pooled per run) and the animacy and superordinate models are just coarser levels. The face-vs-other and intact-vs-scrambled models cross the hierarchy, so they list explicit `groups`. To add the columns to a copy downloaded before this change, run `python tutorial/fetch_haxby.py --out data/haxby-bids --update-events`.

This needs about 3.4 GB in total: 1.7 GB for the nilearn cache and 1.7 GB for the BIDS copy. Once the BIDS copy exists, you can delete `data/nilearn`.

In this checkout, `data/` is a symlink to `~/data/rsaworkflow`, so the images stay out of OneDrive sync.

Sub-5 has 88 blocks instead of 96. One of its runs is missing labels in the release, and the pipeline handles this automatically.

## Run

```bash
python workflows/run_pipeline.py --config tutorial/config-haxby-rsa.example.json
python tutorial/haxby2001_identification.py --config tutorial/config-haxby-rsa.example.json
python workflows/run_pipeline.py --config tutorial/config-haxby-rsa-crossnobis.example.json --stages patterns rsa group report
python workflows/run_pipeline.py --config tutorial/config-haxby-rsa-lss.example.json --stages patterns rsa group report
```

Running times on a laptop for all 6 subjects:

- Main config: about 1.5 minutes.
- LSS config: about 5 minutes, since it fits one GLM per block (about 45 s per subject).

Each config writes a PDF report to `out/<config>/group/rsa_report.pdf`.

## Check 1: the original Haxby (2001) split-half analysis

`haxby2001_identification.py` re-runs the paper's analysis on our GLM t-maps:

1. Average each category's patterns over the even runs and over the odd runs.
2. Within each half, subtract the mean across categories.
3. Correlate the two halves.
4. Score category A against category B as correct if r(A, A) > r(A, B).

Mean accuracy over all category pairs and subjects (pipeline run of 2026-10-29):

| ROI | accuracy (mean ± SD over subjects) |
|---|---|
| VT | **94.5% ± 3.2** |
| house-selective VT | 90.3% ± 4.3 |
| face-selective VT | 78.9% ± 10.5 |

Per category in VT: face 100, house 100, scrambled 100, cat 97.6, scissors 96.4, chair 94.0, shoe 90.5, bottle 77.4.

The paper reports identification in VT of roughly 96% of pairwise comparisons. It reports faces and houses at ceiling and lower accuracy for some small man-made objects. **Check these numbers against the paper's text and Fig. 3 before citing them.** I could not confirm the exact percentages from an accessible copy.

Our pattern matches the paper. Differences in preprocessing and modeling are expected: we use a GLM with t-maps and the shipped VT mask, while the original used its own GLM betas and masks.

## Check 2: RSA on the representational geometry

The main config computes correlation-distance RDMs on run-wise GLM t-maps, with patterns averaged over runs. It compares them with categorical model RDMs using Kendall's τ-a, then runs a one-sided t-test across the 6 subjects with FDR correction across models.

Group mean RDM in VT, shown as an MDS plot in `out/haxby/group/roi-vt/group_roi-vt_mds.png`:

- Face and cat are close together (the animate cluster).
- Bottle, shoe and scissors form a tight cluster, with chair nearby.
- House and scrambled pictures sit far from everything else.

This is the animate/inanimate and face/place organization of VT reported for these categories, including:

- Haxby et al. (2001).
- Kriegeskorte et al. (2008), Neuron, for a related stimulus set.
- Connolly et al. (2012), J Neurosci.

Model fits (mean τ-a; noise ceiling [lower, upper]):

| ROI | animacy | superordinate (animate / place / object / scrambled) | face vs. other | intact vs. scrambled | noise ceiling |
|---|---|---|---|---|---|
| VT | 0.20* | **0.36*** | 0.04 | 0.16* | [0.76, 0.80] |
| face-selective | **0.37*** | 0.28* | 0.24* | 0.14* | [0.46, 0.56] |
| house-selective | 0.16* | **0.30*** | 0.01 | 0.14* | [0.71, 0.76] |

\* FDR-corrected p < .05.

What to expect from these results:

- **The superordinate model wins in VT.** It is far below the noise ceiling, because VT also separates the small objects from each other (the basis of the 94% identification above).
- **"face vs. other" fails in VT and house cortex.** That model treats all non-face categories as identical, and VT clearly doesn't.
- **Animacy fits best in face-selective cortex**, as expected.
- **The high noise ceiling** (τ-a ≈ 0.8 between subjects) means the geometry is very consistent across people. That is what you expect from a block design with 12 runs.

The crossnobis variant (`config-haxby-rsa-crossnobis.example.json`) uses beta maps, crossnobis distances cross-validated over runs, and cosine similarity. It gives the same model ordering in every ROI. Cosine values are on a different scale, because crossnobis distances are ratio-scale with a true zero, so compare models within a method, not across methods.

## Check 3: weighted models (cross-validated over subjects)

Both configs define two weighted models:

- `animacy+intact`: animacy and intact vs. scrambled.
- `all_categorical`: all four categorical models.

Weights are non-negative and fit with `optimize_positive`. They are fit on 5 subjects and scored on the held-out 6th, so the weighted models get no advantage from their extra parameters.

Mean τ-a on the held-out subject:

| ROI | best fixed | animacy+intact | all_categorical | cv noise ceiling |
|---|---|---|---|---|
| VT | 0.36 (superordinate) | 0.27* | 0.37* | [0.73, 0.79] |
| face-selective | 0.37 (animacy) | 0.36* | **0.52*** | [0.46, 0.58] |
| house-selective | 0.30 (superordinate) | 0.21* | 0.30* | [0.70, 0.76] |

In face-selective cortex, the combination of all four models reaches the lower bound of the noise ceiling, so it explains about as much as is explainable there. In VT and house cortex, the combination does no better than superordinate alone (paired over folds, p = .91 in VT). Most of the VT geometry those models leave unexplained is the separation among the individual small objects.

For VT, weights fit on all subjects are: animacy 0.86, superordinate 0.45, intact 0.19, face 0.13. With a scale-free comparison like τ-a, only the ratios between weights matter.

## Check 4: item-level LSS

`config-haxby-rsa-lss.example.json` (`pattern_estimation.level: "item"`) estimates one pattern per block (each block is one item) with least-squares-separate GLMs and then averages a category's blocks in the RDM. Haxby has exactly one block per category per run, so with `lss_other: "by_label"` each LSS design contains the same regressors as the per-run GLM. The two must therefore give identical patterns.

They do: the RDMs from the two configs match to floating-point precision for every subject and ROI. This confirms that the LSS path (masking, scaling, design matrices, contrasts) matches nilearn's `FirstLevelModel`. A dataset with an exemplar repeated within a run, or `lss_other: "single"`, would give different estimates.

## Check 5: the lab's template RSA (docs/lab_methods.md)

[config-haxby-template.example.json](config-haxby-template.example.json) runs the feature-weighted template-matching method of Kim et al. (2020) and Bretton et al. (2024). Haxby has no repeated exemplars, so its 8 categories play the role of exemplars (the config's hierarchy is just `["category"]`).

- **Templates:** odd runs, with a 4 s lag.
- **Weighting:** exemplar-vs-others LSA t-maps within each exemplar's category mask (p < .05, k ≥ 10) inside VT.
- **Targets:** even-run blocks, scored as window means and as a 30 s timecourse.

```bash
python workflows/run_pipeline.py --config tutorial/config-haxby-template.example.json
```

Results:

- **Window means:** held-out target − others is 0.54 Fisher z with exemplar-weighted templates and 0.30 unweighted (n = 6, both p < .001). Feature weighting nearly doubles exemplar discriminability, which is the papers' rationale for it.
- **Template validation:** leave-one-block-out target z is 0.70 vs. 0.11 for others.
- **Timecourse:** rises about 5 s after block onset, plateaus over the block, and falls below zero after it.
- **Category masks:** with an 8-way category contrast, 3–4 of the 8 masks per subject are too small and fall back to the whole ROI. The run log and `*_feature-selection.csv` flag this.

## What this validates (and what it doesn't)

- **Validated:**
  - Event extraction and BIDS matching.
  - Run-wise GLM pattern estimation.
  - ROI masking.
  - RDMs: correlation, plus crossnobis with noise precision.
  - RDM row ordering by the events hierarchy.
  - Model RDM construction and comparison.
  - Group statistics and noise ceiling.
  - Cross-validated weighted models.
  - LSS item-level estimation, which matches the per-run GLM exactly where theory says it must.
  - The PDF report.
  - The published Haxby (2001) result, reproduced to within a couple of percentage points.
- **Also covered by `tests/test_pipeline_synthetic.py`:** recovery of a known geometry with:
  - `volume_average`
  - `glm`
  - crossnobis
  - LSS and LSA item-level estimation
  - weighted-model recovery (the correct component gets the larger weight)
  - generation of both reports
- **Not validated against published values:**
  - The searchlight path. It runs, but there is no published Haxby searchlight-RSA map to compare it with.
  - LSS versus LSA on a rapid event-related design, where they should differ.
  - Bootstrap group inference. Haxby has only 6 subjects.
