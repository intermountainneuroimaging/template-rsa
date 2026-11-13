"""Unit tests for the post-hoc notebook helpers (_interactive_notebooks/rsa_posthoc_common.py)."""

import numpy as np
import pandas as pd

from _interactive_notebooks.rsa_posthoc_common import (
    add_previous_trial_columns, baseline_correct, bin_means, complete_trials, difference_scores, equal_trial_bootstrap,
    first_significant_bin, join_item_columns, rm_tukey, make_bins, make_toy_scores, mixed_model, one_sample_by_bin, paired_contrast,
    subject_means, subtract_reference)


def scores_table():
    """Two runs of one subject, 3 trials each, two comparison rows per trial."""
    ev = pd.DataFrame({"subject": "1", "run_id": ["r1"] * 3 + ["r2"] * 3, "event_index": [1, 2, 3, 1, 2, 3],
                       "onset": [0, 10, 20, 0, 10, 20], "stimulus": list("abcdef"),
                       "category": ["face", "face", "scene", "scene", "face", "face"],
                       "operation": ["maintain", "replace", "suppress", "replace", "maintain", "suppress"],
                       "replace_category": [None, "scene", None, "face", None, None]})
    return pd.concat([ev.assign(comparison="target", z=0.5), ev.assign(comparison="others", z=0.1)], ignore_index=True)


def test_previous_trial_columns_are_within_run_and_merged_onto_every_row():
    out = add_previous_trial_columns(scores_table(), ["category", "operation"],
                                     matches={"same_replacement": ("replace_category", "category")})
    ev = out[out["comparison"] == "target"].set_index(["run_id", "event_index"])
    assert pd.isna(ev.loc[("r2", 1), "prev_category"])                 # first trial of a run has no N-1
    assert ev.loc[("r1", 2), "same_category"] == "same"                # face -> face
    assert ev.loc[("r1", 3), "same_category"] == "different"
    assert ev.loc[("r1", 3), "same_replacement"] == "same"             # trial 2 replaced with scene; trial 3 is a scene
    assert ev.loc[("r2", 2), "same_replacement"] == "same"             # trial 1 replaced with face; trial 2 is a face
    assert pd.isna(ev.loc[("r2", 3), "same_replacement"])              # trial N had no replacement: not applicable
    assert len(out) == len(scores_table())


def test_join_item_columns_fills_missing_by_subject_and_item():
    master = pd.DataFrame({"subject": ["1", "1", "2"], "run_id": ["s1", "s1", "s1"], "onset": [0, 5, 0],
                           "task": "study", "stimulus": ["a", "b", "a"], "operation": ["suppress", "maintain", "replace"]})
    post = pd.DataFrame({"subject": ["1", "1", "2", "2"], "stimulus": ["a", "b", "a", "z"], "operation": [None, None, None, "x"]})
    out = join_item_columns(post, master, {"column": "task", "match": "exact", "value": "study"}, "stimulus", ["operation"])
    assert out["operation"].tolist() == ["suppress", "maintain", "replace", "x"]


def test_reference_subtraction_and_bins_recover_the_toy_effect():
    sub = subject_means(make_toy_scores(), ["operation", "tr_offset", "time_s"])
    removal = subtract_reference(sub, "operation", "maintain", ["tr_offset", "time_s"])
    assert set(removal["operation"]) == {"suppress"}
    tests = one_sample_by_bin(bin_means(removal, make_bins(0, 11, 3), ["operation"]), ["operation"])
    assert tests["bin"].tolist() == ["0-2", "3-5", "6-8", "9-11"]
    assert tests.iloc[-1]["mean"] < -0.08 and tests.iloc[-1]["p_fdr"] < 0.05
    assert abs(tests.iloc[0]["mean"]) < 0.05
    onset = first_significant_bin(tests, ["operation"], direction="below")
    assert onset["bin_start"].iloc[0] >= 6


def test_baseline_correct_zeroes_the_baseline():
    sub = subject_means(make_toy_scores(), ["operation", "tr_offset", "time_s"])
    bc = baseline_correct(sub, (0, 3), ["operation"])
    base = bc[bc["tr_offset"] < 3].groupby(["subject", "operation"])["z"].mean()
    assert np.allclose(base, 0)


def test_paired_contrast_uses_subjects_with_both_levels():
    rng = np.random.default_rng(0)
    rows = [{"subject": s, "op": "suppress", "memory": m, "z": (0.5 if m == "remembered" else 0.3) + rng.normal(0, 0.02)}
            for s in range(8) for m in ("remembered", "forgotten")]
    rows.append({"subject": 99, "op": "suppress", "memory": "remembered", "z": 5.0})   # only one outcome: excluded
    res = paired_contrast(pd.DataFrame(rows), "memory", "remembered", "forgotten", by=["op"])
    assert res["n"].iloc[0] == 8 and 0.15 < res["mean_diff"].iloc[0] < 0.25
    assert res["boot_ci_low"].iloc[0] > 0 and res["p"].iloc[0] < 0.001
    diff = difference_scores(pd.DataFrame(rows), "memory", "remembered", "forgotten", by=["op"])
    assert len(diff) == 8


def test_mixed_model_reports_terms_or_failures():
    rng = np.random.default_rng(1)
    df = pd.DataFrame([{"subject": s, "memory": m, "z": 0.3 + (0.2 if m == "remembered" else 0) + rng.normal(0, 0.1)}
                       for s in range(10) for m in rng.choice(["remembered", "forgotten"], 20)])
    res = mixed_model(df, "z ~ C(memory, Treatment('remembered'))")
    term = res[res["term"].str.contains("forgotten", na=False)]
    assert len(term) == 1 and term["beta"].iloc[0] < -0.1


def test_baseline_modes_group_none_and_subject():
    sub = subject_means(make_toy_scores(), ["operation", "tr_offset", "time_s"])
    sub["z"] += sub["subject"].astype(int) * 0.01           # subject-specific offsets
    raw = baseline_correct(sub, (0, 3), ["operation"], mode="none")
    assert raw["z"].equals(sub["z"])
    grp = baseline_correct(sub, (0, 3), ["operation"], mode="group")
    # one constant per operation: the group mean over subjects and baseline TRs is zero ...
    assert np.allclose(grp[grp["tr_offset"] < 3].groupby("operation")["z"].mean(), 0)
    # ... but subjects keep their own offsets (unlike the per-subject mode)
    per_subject = grp[grp["tr_offset"] < 3].groupby(["subject", "operation"])["z"].mean()
    assert per_subject.std() > 0.005
    shift = (sub["z"] - grp["z"]).round(10).groupby(sub["operation"]).nunique()
    assert (shift == 1).all()


def test_complete_trials_drops_trials_missing_a_tr():
    t = make_toy_scores()
    n = int(t["tr_offset"].max()) + 1
    t["run_id"] = "r1"
    t["event_index"] = t.groupby(["subject", "operation", "event_index"]).ngroup()  # one id per trial
    first = t.drop_duplicates(["subject", "run_id", "event_index"]).iloc[0]
    cut = t.drop(t[(t["subject"] == first["subject"]) & (t["event_index"] == first["event_index"]) & (t["tr_offset"] == n - 1)].index)
    kept = complete_trials(cut, n)
    gone = kept[(kept["subject"] == first["subject"]) & (kept["event_index"] == first["event_index"])]
    assert gone.empty and len(kept) == len(cut) - (n - 1)


def test_rm_tukey_matches_a_hand_computation():
    rng = np.random.default_rng(0)
    long = pd.DataFrame([{"subject": s, "op": o, "v": rng.normal(i * 0.3, 1)} for s in range(20)
                         for i, o in enumerate(["a", "b", "c"])])
    out = rm_tukey(long, "v", "op", ["a", "b", "c"], pairs=[("a", "c")])
    w = long.pivot(index="subject", columns="op", values="v")[["a", "b", "c"]].to_numpy()
    resid = w - w.mean(1, keepdims=True) - w.mean(0) + w.mean()
    mse = (resid ** 2).sum() / (19 * 2)
    q = abs(w[:, 0].mean() - w[:, 2].mean()) / np.sqrt(mse / 20)
    assert len(out) == 1 and np.isclose(out["q"].iloc[0], q) and out["df_error"].iloc[0] == 38
    from scipy import stats
    assert np.isclose(out["p_tukey"].iloc[0], stats.studentized_range.sf(q, 3, 38))


def test_equal_trial_bootstrap_recovers_a_same_minus_different_effect():
    rng = np.random.default_rng(1)
    rows = [{"subject": s, "op": o, "same": sp, "z": rng.normal(0.2 if (o == "x" and sp == "same") else 0.0, 0.1)}
            for s in range(12) for o in ("x", "y") for sp in ("same", "different") for _ in range(rng.integers(5, 30))]
    summary, pairs = equal_trial_bootstrap(pd.DataFrame(rows), "op", "same", n_trials=15, n_boot=300, seed=0)
    eff = summary[summary["component"] == "same - different"].set_index("op")
    assert eff.loc["x", "mean"] > 0.15 and eff.loc["x", "ci5"] > 0 and abs(eff.loc["y", "mean"]) < 0.05
    assert pairs.iloc[0]["p_boot"] < 0.05
