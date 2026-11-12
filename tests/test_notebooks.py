"""Run the core pipeline on the synthetic lab-style dataset, then execute both post-hoc notebooks
against it (RSA_CONFIG / RSA_POSTHOC_OUT point them at the synthetic run) and check their outputs."""

import os

import pandas as pd
import pytest

nbformat = pytest.importorskip("nbformat")
nbclient = pytest.importorskip("nbclient")

from tests.synthetic_lab import make_config, make_dataset, write_config  # noqa: E402
from workflows.template_fidelity import fidelity_subject  # noqa: E402
from workflows.generate_master_spreadsheet import build_master_spreadsheet  # noqa: E402
from workflows.group_fidelity import group_fidelity  # noqa: E402
from workflows.group_template_rsa import group_template_rsa  # noqa: E402
from workflows.template_rsa import template_rsa_subject  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOKS = os.path.join(REPO, "_interactive_notebooks")


@pytest.fixture(scope="module")
def core_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("lab")
    make_dataset(root, ("1", "2", "3", "4"))
    cfg, user = make_config(root)
    df = build_master_spreadsheet(cfg)
    os.makedirs(os.path.dirname(cfg["event_extraction"]["output_file"]), exist_ok=True)
    df.to_csv(cfg["event_extraction"]["output_file"], index=False)
    for sub in ("1", "2", "3", "4"):
        template_rsa_subject(cfg, df, sub, cfg["pipeline"]["output_dir"])
        fidelity_subject(cfg, df, sub, cfg["pipeline"]["output_dir"])
    group_template_rsa(cfg, cfg["pipeline"]["output_dir"])
    group_fidelity(cfg, cfg["pipeline"]["output_dir"])
    return root, write_config(root, user)


def execute(name, config_path, out_dir, monkeypatch):
    monkeypatch.setenv("RSA_CONFIG", config_path)
    monkeypatch.setenv("RSA_POSTHOC_OUT", str(out_dir))
    book = nbformat.read(os.path.join(NOTEBOOKS, name), as_version=4)
    nbclient.NotebookClient(book, timeout=600, kernel_name="python3",
                            resources={"metadata": {"path": NOTEBOOKS}}).execute()
    return book


def test_kim_notebook(core_run, tmp_path, monkeypatch):
    _, config_path = core_run
    execute("kim2020_removal_operations.ipynb", config_path, tmp_path, monkeypatch)
    assert (tmp_path / "stats_and_figures.pdf").exists() and (tmp_path / "run_settings.json").exists()
    removal = pd.read_csv(tmp_path / "removal_vs_maintain_by_bin.csv")
    assert set(removal["operation"]) == {"replace_category", "suppress"}
    # synthetic suppress drops the item after the stimulus: late bins below maintain
    late = removal[(removal["operation"] == "suppress") & (removal["bin_start"] >= 6)]
    assert (late["mean"] < 0).all()
    repl = pd.read_csv(tmp_path / "replacement_minus_target.csv")
    assert repl["mean_diff"].iloc[-1] > 0
    pi = pd.read_csv(tmp_path / "proactive_interference.csv")
    assert {"prev_operation", "mean_diff", "boot_ci_low"} <= set(pi.columns)
    fid = pd.read_csv(tmp_path / "encoding_fidelity_pairwise.csv")
    assert (fid["mean_diff"] > 0).all()  # target > related > others


def test_bretton_notebook(core_run, tmp_path, monkeypatch):
    _, config_path = core_run
    execute("bretton2024_ltm_fidelity.ipynb", config_path, tmp_path, monkeypatch)
    assert (tmp_path / "stats_and_figures.pdf").exists()
    check = pd.read_csv(tmp_path / "template_check_manipulation.csv")
    assert (check["mean_diff"] > 0).all()
    memory = pd.read_csv(tmp_path / "fidelity_remembered_vs_forgotten.csv")
    item = memory[memory["level"] == "item"]
    assert len(item) and (item["mean_diff"] > 0).mean() >= 2 / 3  # remembered items came back stronger
    mixed = pd.read_csv(tmp_path / "fidelity_mixed_models.csv")
    assert len(mixed)
