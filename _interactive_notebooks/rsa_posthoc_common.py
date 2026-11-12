#!/usr/bin/env python3
"""
Everything the post-hoc notebooks share, so each notebook holds only its own study's analysis.

The core pipeline (workflows/run_pipeline.py) ends with one group-level raw table per ROI and
target set -- <output_dir>/group/fidelity/raw/roi-<roi>_target-<set>_scores.csv, one row per
subject x event [x volume] x weighting level x comparison set, carrying every events.tsv column.
The notebooks start from that table; nothing here knows about any particular study.

  settings        configure(**settings) / setting(name): ALPHA, FDR_METHOD, N_BOOTSTRAP, SEED
  loading         load_config, load_scores (the group raw table), load_master (the event table),
                  load_validation
  table helpers   add_previous_trial_columns (trial N-1 columns, e.g. for N+1 analyses),
                  join_exemplar_columns (copy a column from another phase by exemplar, e.g. the
                  operation onto post-test exemplars), event_table (one row per event)
  aggregation     subject_means (trials -> subject means), baseline_correct, subtract_reference,
                  summarize (mean / SEM / n over subjects)
  statistics      make_bins, bin_means, one_sample_by_bin, compare_groups_by_bin (rm ANOVA + pairwise),
                  paired_contrast (paired t + subject bootstrap CI), rm_anova, pairwise_tests,
                  mixed_model, first_significant_bin, fdr
  plots           plot_timecourse (mean +- SEM, significance markers per bin), plot_points
  outputs         save_outputs (PDF of the figures, one CSV per table, the settings as JSON)
  teaching        make_toy_scores (a small synthetic table for the worked example)
"""

import itertools
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
from utils.rsa_common import evaluate_query_node, safe_name  # noqa: E402
from utils.rsa_common import load_config as _load_config  # noqa: E402

EVENT_KEY = ["subject", "run_id", "event_index"]

# =====================================================
# Settings
# =====================================================

SETTING_NAMES = ("ALPHA", "FDR_METHOD", "N_BOOTSTRAP", "SEED")
settings = {"ALPHA": 0.05, "FDR_METHOD": "fdr_bh", "N_BOOTSTRAP": 10000, "SEED": 0}


def configure(namespace=None, **kwargs):
    """configure(globals()) takes every SETTING_NAMES entry the notebook defines; kwargs override."""
    if namespace is not None:
        settings.update({n: namespace[n] for n in SETTING_NAMES if n in namespace})
    settings.update(kwargs)


def setting(name, default=None):
    return settings.get(name, default)


# =====================================================
# Loading
# =====================================================

def load_config(path: str) -> dict:
    return _load_config(path)


def raw_scores_path(cfg: dict, roi: str, target_set: str) -> str:
    return os.path.join(cfg["pipeline"]["output_dir"], "group", "fidelity", "raw",
                        f"roi-{roi}_target-{safe_name(target_set)}_scores.csv")


def load_scores(cfg: dict, roi: str, target_set: str, level: str = None) -> pd.DataFrame:
    """The group raw table for one ROI and target set (optionally one weighting level)."""
    path = raw_scores_path(cfg, roi, target_set)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} not found -- run the core pipeline (template, fidelity and fidelity_group stages) first")
    df = pd.read_csv(path, dtype={"subject": str, "run": str, "run_id": str}, low_memory=False)
    return df[df["level"] == level].reset_index(drop=True) if level else df


def load_master(cfg: dict) -> pd.DataFrame:
    from utils.rsa_common import read_master_spreadsheet
    return read_master_spreadsheet(cfg["event_extraction"]["output_file"])


def load_validation(cfg: dict, roi: str) -> pd.DataFrame:
    path = os.path.join(cfg["pipeline"]["output_dir"], "group", "template_rsa", f"roi-{roi}_template-validation.csv")
    return pd.read_csv(path, dtype={"subject": str}) if os.path.isfile(path) else None


# =====================================================
# Table helpers
# =====================================================

def event_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per event (drops the comparison / volume / level repetition)."""
    return df.drop_duplicates(EVENT_KEY).reset_index(drop=True)


def add_previous_trial_columns(df: pd.DataFrame, columns, matches: dict = None) -> pd.DataFrame:
    """Add prev_<col> (the preceding event's value in the same run, by onset) and same_<col>
    ("same" / "different"; empty for a run's first event). matches {name: (prev_col, cur_col)} adds
    <name> = "same" where trial N's prev_col equals trial N+1's cur_col (e.g. trial N's replacement
    category vs trial N+1's category). Computed on events, then merged back onto every row."""
    columns = list(columns)
    matches = dict(matches or {})
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
    drop = [c for c in out.columns if c in df.columns and c not in EVENT_KEY]
    return df.drop(columns=drop).merge(out, on=EVENT_KEY, how="left")


def join_exemplar_columns(df: pd.DataFrame, master: pd.DataFrame, source_query, item_col: str, columns,
                          keep_existing: bool = True) -> pd.DataFrame:
    """Copy `columns` from the events matching source_query (e.g. the manipulation phase) onto the
    rows of df by subject and exemplar (`item_col`: the exemplar column, e.g. "stimulus"). An
    exemplar seen several times in the source takes its first occurrence (by run, onset). With
    keep_existing, values df already has are not overwritten."""
    src = master[evaluate_query_node(source_query, master)] if source_query else master
    src = src.sort_values(["subject", "run_id", "onset"]).assign(**{item_col: lambda d: d[item_col].astype(str)})
    src = src.drop_duplicates(["subject", item_col]).set_index(["subject", item_col])
    out = df.copy()
    key = pd.MultiIndex.from_arrays([out["subject"].astype(str), out[item_col].astype(str)])
    for col in columns:
        joined = pd.Series(src[col].reindex(key).values, index=out.index)
        out[col] = out[col].where(out[col].notna(), joined) if (keep_existing and col in out) else joined
    return out


# =====================================================
# Aggregation (subjects are the random effect: trials are averaged within subject first)
# =====================================================

def subject_means(df: pd.DataFrame, by, value: str = "z") -> pd.DataFrame:
    by = list(by)
    return df.groupby(["subject", *by], as_index=False, dropna=False).agg(
        **{value: (value, "mean"), "n_trials": (value, "size")})


def baseline_correct(sub: pd.DataFrame, baseline_trs, by, x: str = "tr_offset", value: str = "z") -> pd.DataFrame:
    """Subtract each subject x `by` curve's mean over x in [baseline_trs[0], baseline_trs[1])."""
    a, b = baseline_trs
    keys = ["subject", *by]
    base = sub[(sub[x] >= a) & (sub[x] < b)].groupby(keys, dropna=False)[value].mean().rename("_base")
    out = sub.join(base, on=keys)
    out[value] = out[value] - out["_base"]
    return out.drop(columns="_base")


def subtract_reference(sub: pd.DataFrame, column: str, reference, by, value: str = "z") -> pd.DataFrame:
    """Each level of `column` minus the reference level, matched on subject and `by` (e.g. each
    operation minus maintain at every TR). The reference rows are dropped."""
    keys = ["subject", *by]
    ref = sub[sub[column] == reference].set_index(keys)[value].rename("_ref")
    out = sub[sub[column] != reference].join(ref, on=keys)
    out[value] = out[value] - out["_ref"]
    return out.drop(columns="_ref").dropna(subset=[value])


def summarize(sub: pd.DataFrame, by, value: str = "z") -> pd.DataFrame:
    return sub.groupby(list(by), as_index=False, dropna=False).agg(
        mean=(value, "mean"), sem=(value, stats.sem), n=(value, "count"))


# =====================================================
# Statistics
# =====================================================

def fdr(pvals, method=None):
    from statsmodels.stats.multitest import multipletests
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if ok.any():
        out[ok] = multipletests(p[ok], method=method or setting("FDR_METHOD"))[1]
    return out


def make_bins(first: int, last: int, size: int) -> list:
    """Consecutive bins [(start, stop), ...] covering first..last inclusive (last bin may be shorter)."""
    return [(s, min(s + size - 1, last)) for s in range(first, last + 1, size)]


def bin_means(sub: pd.DataFrame, bins, by, x: str = "tr_offset", value: str = "z") -> pd.DataFrame:
    """Subject x `by` x bin means of a TR-wise subject table."""
    parts = []
    for start, stop in bins:
        part = sub[(sub[x] >= start) & (sub[x] <= stop)]
        part = part.groupby(["subject", *by], as_index=False, dropna=False)[value].mean()
        parts.append(part.assign(bin=f"{start}-{stop}", bin_start=start, bin_stop=stop))
    return pd.concat(parts, ignore_index=True)


def one_sample_by_bin(binned: pd.DataFrame, by, value: str = "z", alternative: str = "two-sided") -> pd.DataFrame:
    """One-sample t-test vs 0 per `by` x bin; p_fdr is corrected across the bins of each `by` group."""
    rows = []
    for keys, part in binned.groupby([*by, "bin_start", "bin_stop", "bin"], dropna=False):
        x = part[value].dropna()
        t, p = stats.ttest_1samp(x, 0.0, alternative=alternative) if len(x) > 1 else (np.nan, np.nan)
        rows.append({**dict(zip([*by, "bin_start", "bin_stop", "bin"], keys)), "n": len(x), "mean": x.mean(),
                     "sem": stats.sem(x) if len(x) > 1 else np.nan, "t": t, "p": p})
    out = pd.DataFrame(rows).sort_values([*by, "bin_start"])
    out["p_fdr"] = out.groupby(list(by), dropna=False)["p"].transform(lambda p: fdr(p)) if by else fdr(out["p"])
    return out.reset_index(drop=True)


def rm_anova(long: pd.DataFrame, dv: str, within: str):
    """Repeated-measures ANOVA on subjects with every level -> (F, p, n_subjects)."""
    from statsmodels.stats.anova import AnovaRM
    levels = long[within].nunique()
    counts = long.dropna(subset=[dv]).groupby("subject")[within].nunique()
    complete = counts[counts == levels].index
    data = long[long["subject"].isin(complete)].dropna(subset=[dv])
    if levels < 2 or len(complete) < 2:
        return np.nan, np.nan, len(complete)
    row = AnovaRM(data, dv, "subject", within=[within], aggregate_func="mean").fit().anova_table.iloc[0]
    return float(row["F Value"]), float(row["Pr > F"]), len(complete)


def pairwise_tests(long: pd.DataFrame, dv: str, within: str, levels=None) -> pd.DataFrame:
    """Paired t-tests between every pair of `within` levels (subjects with both), FDR across pairs."""
    wide = long.pivot_table(index="subject", columns=within, values=dv, aggfunc="mean")
    rows = []
    for a, b in itertools.combinations(levels or sorted(wide.columns), 2):
        both = wide[[a, b]].dropna()
        t, p = stats.ttest_rel(both[a], both[b]) if len(both) > 1 else (np.nan, np.nan)
        rows.append({"a": a, "b": b, "n": len(both), "mean_diff": (both[a] - both[b]).mean(), "t": t, "p": p})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_fdr"] = fdr(out["p"])
    return out


def compare_groups_by_bin(binned: pd.DataFrame, group_col: str, value: str = "z") -> pd.DataFrame:
    """Per bin: a repeated-measures ANOVA across the levels of group_col, and pairwise paired t-tests
    (FDR across the pairs of that bin)."""
    rows = []
    for (start, label), part in binned.groupby(["bin_start", "bin"]):
        F, p, n = rm_anova(part, value, group_col)
        rows.append({"bin": label, "bin_start": start, "test": "rm_anova", "comparison": "all", "n": n,
                     "statistic": F, "p": p, "p_fdr": np.nan})
        for _, r in pairwise_tests(part, value, group_col).iterrows():
            rows.append({"bin": label, "bin_start": start, "test": "paired_t", "comparison": f"{r['a']} - {r['b']}",
                         "n": r["n"], "mean_diff": r["mean_diff"], "statistic": r["t"], "p": r["p"], "p_fdr": r["p_fdr"]})
    return pd.DataFrame(rows)


def bootstrap_ci(diffs, n_boot=None, seed=None):
    """Resample subjects' difference scores -> (2.5th, 97.5th percentile of the mean, two-sided p)."""
    d = np.asarray(diffs, dtype=float)
    d = d[np.isfinite(d)]
    n_boot = int(n_boot or setting("N_BOOTSTRAP"))
    if len(d) < 2 or n_boot <= 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(setting("SEED") if seed is None else seed)
    boots = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(axis=1)
    p = min(1.0, 2 * min((boots <= 0).mean(), (boots >= 0).mean()))
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)), float(p)


def paired_contrast(sub: pd.DataFrame, column: str, a, b, by=(), value: str = "z") -> pd.DataFrame:
    """Subject-level paired t-test of column == a vs column == b within each `by` group (subjects
    with both levels only), with a subject-bootstrap CI of the mean difference."""
    by = list(by)
    rows = []
    groups = sub.groupby(by, dropna=False) if by else [((), sub)]
    for keys, part in groups:
        keys = keys if isinstance(keys, tuple) else (keys,)
        wide = part.pivot_table(index="subject", columns=column, values=value, aggfunc="mean")
        if a not in wide or b not in wide:
            continue
        both = wide[[a, b]].dropna()
        diff = both[a] - both[b]
        t, p = stats.ttest_rel(both[a], both[b]) if len(both) > 1 else (np.nan, np.nan)
        lo, hi, p_boot = bootstrap_ci(diff.values)
        rows.append({**dict(zip(by, keys)), "a": a, "b": b, "n": len(both), "mean_a": both[a].mean(),
                     "mean_b": both[b].mean(), "mean_diff": diff.mean(), "t": t, "p": p,
                     "boot_ci_low": lo, "boot_ci_high": hi, "p_boot": p_boot})
    return pd.DataFrame(rows)


def difference_scores(sub: pd.DataFrame, column: str, a, b, by=(), value: str = "z") -> pd.DataFrame:
    """Per subject x `by`: value at column == a minus value at column == b (long table, column 'diff')."""
    by = list(by)
    wide = sub.pivot_table(index=["subject", *by], columns=column, values=value, aggfunc="mean")
    if a not in wide or b not in wide:
        return pd.DataFrame(columns=["subject", *by, "diff"])
    return (wide[a] - wide[b]).rename("diff").dropna().reset_index()


def mixed_model(df: pd.DataFrame, formula: str, re_formula: str = None, by=(), value: str = "z") -> pd.DataFrame:
    """statsmodels mixedlm on trial-level rows (subject = grouping factor) within each `by` group;
    re_formula "~C(memory)" gives (1 + memory | subject). Failed fits are reported, not raised."""
    import statsmodels.formula.api as smf
    by = list(by)
    rows = []
    groups = df.groupby(by, dropna=False) if by else [((), df)]
    for keys, part in groups:
        keys = keys if isinstance(keys, tuple) else (keys,)
        part = part.dropna(subset=[value])
        base = {**dict(zip(by, keys)), "formula": formula, "re_formula": re_formula or "~1",
                "n_trials": len(part), "n_subjects": part["subject"].nunique()}
        try:
            fit = smf.mixedlm(formula, part, groups=part["subject"], re_formula=re_formula).fit(reml=True)
            ci = fit.conf_int()
            for term in fit.fe_params.index:
                rows.append({**base, "term": term, "beta": fit.fe_params[term], "se": fit.bse_fe[term],
                             "z": fit.tvalues[term], "p": fit.pvalues[term], "ci_low": ci.loc[term, 0],
                             "ci_high": ci.loc[term, 1], "converged": fit.converged})
        except Exception as err:  # singular fits, missing levels
            rows.append({**base, "term": "(fit failed)", "error": str(err)[:200]})
    return pd.DataFrame(rows)


def first_significant_bin(tests: pd.DataFrame, by, direction: str = "below", p_col: str = "p") -> pd.DataFrame:
    """Per `by` group, the first bin whose mean is below (or above) 0 at p < ALPHA -- e.g. the
    removal onset of an operation-minus-maintain curve."""
    alpha = setting("ALPHA")
    sign = (tests["mean"] < 0) if direction == "below" else (tests["mean"] > 0)
    hits = tests[sign & (tests[p_col] < alpha)].sort_values("bin_start")
    return hits.groupby(list(by), dropna=False).head(1).reset_index(drop=True)


# =====================================================
# Plots
# =====================================================

def plot_timecourse(ax, sub: pd.DataFrame, hue: str, title: str, x: str = "time_s", value: str = "z",
                    tests: pd.DataFrame = None, tr_seconds: float = None, p_col: str = "p_fdr", colors: dict = None,
                    shade=None):
    """Mean +- SEM over subjects per level of `hue`; `tests` (one_sample_by_bin output with a hue
    column) adds a marker under each bin with p_col < ALPHA; `shade` = (x0, x1) highlights a span."""
    summary = summarize(sub, [hue, x], value)
    levels = list(dict.fromkeys(summary[hue]))
    colors = colors or {lvl: f"C{i}" for i, lvl in enumerate(levels)}
    if shade is not None:
        ax.axvspan(*shade, color="0.92", zorder=0)
    for lvl in levels:
        part = summary[summary[hue] == lvl].sort_values(x)
        ax.plot(part[x], part["mean"], color=colors[lvl], lw=1.6, label=str(lvl))
        ax.fill_between(part[x], part["mean"] - part["sem"], part["mean"] + part["sem"], color=colors[lvl], alpha=0.22)
    ax.axhline(0, color="k", lw=0.8)
    if tests is not None and len(tests) and tr_seconds:
        lo = ax.get_ylim()[0]
        span = np.diff(ax.get_ylim())[0]
        for i, lvl in enumerate(levels):
            sig = tests[(tests[hue] == lvl) & (tests[p_col] < setting("ALPHA"))]
            for _, r in sig.iterrows():
                ax.plot([r["bin_start"] * tr_seconds, (r["bin_stop"] + 1) * tr_seconds], [lo + 0.03 * span * (i + 1)] * 2,
                        color=colors[lvl], lw=3, solid_capstyle="butt")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("time (s)" if x == "time_s" else x, fontsize=9)
    ax.set_ylabel("similarity (Fisher z)", fontsize=9)
    ax.legend(fontsize=8, frameon=False)


def plot_points(ax, sub: pd.DataFrame, x: str, hue: str, title: str, value: str = "z", order=None, hue_order=None):
    """Bars = mean +- SEM over subjects, dots = subjects, for x x hue."""
    order = order or list(dict.fromkeys(sub[x]))
    hue_order = hue_order or list(dict.fromkeys(sub[hue]))
    width = 0.8 / max(len(hue_order), 1)
    for j, h in enumerate(hue_order):
        for i, lvl in enumerate(order):
            vals = sub[(sub[x] == lvl) & (sub[hue] == h)][value].dropna()
            if vals.empty:
                continue
            pos = i + (j - (len(hue_order) - 1) / 2) * width
            ax.bar(pos, vals.mean(), width * 0.9, yerr=vals.sem() if len(vals) > 1 else None, capsize=2,
                   color=f"C{j}", label=str(h) if i == 0 else None)
            ax.scatter(np.full(len(vals), pos), vals, s=7, color="k", zorder=3)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(order)), order, fontsize=8)
    ax.set_ylabel("similarity (Fisher z)", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, frameon=False)


# =====================================================
# Outputs
# =====================================================

def save_outputs(out_dir: str, figures: dict, tables: dict, run_settings: dict, pdf_name: str = "stats_and_figures.pdf"):
    """figures {name: Figure} -> one PDF (in order); tables {name: DataFrame} -> <name>.csv;
    run_settings -> run_settings.json. Returns the list of files written."""
    from matplotlib.backends.backend_pdf import PdfPages
    os.makedirs(out_dir, exist_ok=True)
    written = []
    pdf_path = os.path.join(out_dir, pdf_name)
    with PdfPages(pdf_path) as pdf:
        for fig in figures.values():
            pdf.savefig(fig)
    written.append(pdf_path)
    for name, table in tables.items():
        path = os.path.join(out_dir, f"{name}.csv")
        table.to_csv(path, index=False)
        written.append(path)
    path = os.path.join(out_dir, "run_settings.json")
    with open(path, "w") as f:
        json.dump(run_settings, f, indent=2, default=str)
    written.append(path)
    return written


# =====================================================
# Teaching
# =====================================================

def make_toy_scores(seed=0, n_subjects=12, n_trials=10, n_tr=12, effect=(0.0, 0.0, 0.05, 0.12)):
    """Toy timecourse table: two operations ("maintain", "suppress"), target similarity that drops for
    suppress by `effect` (one value per 3-TR bin) -- for the worked example at the end of a notebook."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_subjects):
        subj_offset = rng.normal(0, 0.05)
        for cond in ("maintain", "suppress"):
            for trial in range(n_trials):
                for k in range(n_tr):
                    drop = effect[min(k // 3, len(effect) - 1)] if cond == "suppress" else 0.0
                    rows.append({"subject": f"{s + 1:02d}", "operation": cond, "event_index": trial, "tr_offset": k,
                                 "time_s": k * 1.0, "z": 0.2 + subj_offset - drop + rng.normal(0, 0.15)})
    return pd.DataFrame(rows)
