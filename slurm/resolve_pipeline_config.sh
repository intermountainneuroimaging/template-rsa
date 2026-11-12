#!/bin/bash
#
# Resolves what the RSA SLURM pipeline (slurm/0_-3_) needs from the config as plain shell
# variables. Nothing to hand-edit here: the only input to the whole pipeline is the config path each
# stage script takes as its first argument (`sbatch slurm/N_....sh <config.json>`).
#
# Sourced (not run) by each stage script after its bootstrap block has set and exported
#   CONFIG_FILE   the config path (validated)
#   SCRIPTS_DIR   the repo root: pipeline.scripts_dir from the config, else the directory sbatch was
#                 invoked from ($SLURM_SUBMIT_DIR) when it is the repo root
# (sbatch copies a batch script to its own spool file before running it, so a stage script cannot
# find the repo from its own location -- hence the config / submit directory.)
#
# Exports (relative config paths are resolved against SCRIPTS_DIR, where the workflows run):
#   OUTPUT_DIR          pipeline.output_dir
#   MASTER_SPREADSHEET  event_extraction.output_file
#   DERIVATIVES_ROOT    event_extraction.derivatives_root
#   ANALYSES            pipeline.analyses, space-separated (default: rdm)
#   SUBJECTS            pipeline.subjects, space-separated; when unset, every sub-* folder under
#                       derivatives_root (the array size and task -> subject order of stage 2)
#   N_SUBJECTS
#
# "pipeline": {
#   "scripts_dir": "/path/to/rsaworkflow",        (optional, see above)
#   "output_dir": "data/cv-bids/out/kim2020_denoised",
#   "subjects": ["090", "091"],                  (optional; null = every subject under derivatives_root)
#   "analyses": ["template", "fidelity"]
# }

if [ -z "$CONFIG_FILE" ] || [ -z "$SCRIPTS_DIR" ]; then
    echo "resolve_pipeline_config.sh: CONFIG_FILE and SCRIPTS_DIR must be set by the calling stage script" >&2
    exit 1
fi

_pipeline_vars=$(python3 - "$CONFIG_FILE" "$SCRIPTS_DIR" <<'PYEOF'
import glob
import json
import os
import shlex
import sys

config_path, scripts_dir = sys.argv[1], sys.argv[2]
with open(config_path) as f:
    cfg = json.load(f)
pipeline = cfg.get("pipeline", {})
extraction = cfg.get("event_extraction", {})


def resolve(path):
    if not path:
        return ""
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(scripts_dir, path))


derivatives_root = resolve(extraction.get("derivatives_root") or extraction.get("bids_root"))
subjects = pipeline.get("subjects")
if subjects:
    subjects = [str(s) for s in subjects]
else:
    subjects = sorted(os.path.basename(p)[4:] for p in glob.glob(os.path.join(derivatives_root, "sub-*"))
                      if os.path.isdir(p))
if not subjects:
    print(f"resolve_pipeline_config.sh: no subjects -- pipeline.subjects is not set in {config_path} and "
          f"there are no sub-* folders under {derivatives_root or '(event_extraction.derivatives_root unset)'}",
          file=sys.stderr)
    sys.exit(1)

out = {
    "OUTPUT_DIR": resolve(pipeline.get("output_dir", "out")),
    "MASTER_SPREADSHEET": resolve(extraction.get("output_file")),
    "DERIVATIVES_ROOT": derivatives_root,
    "ANALYSES": " ".join(pipeline.get("analyses") or ["rdm"]),
    "SUBJECTS": " ".join(subjects),
    "N_SUBJECTS": str(len(subjects)),
}
for key, value in out.items():
    print(f"export {key}={shlex.quote(value)}")
PYEOF
) || exit 1
eval "$_pipeline_vars"
unset _pipeline_vars

# true if pipeline.analyses includes $1
has_analysis() { [[ " $ANALYSES " == *" $1 "* ]]; }

# betaestimation (the shared GLM / nuisance package): use the clone next to this repo
# (<parent>/betaestimation, or $BETAESTIMATION_DIR) when there is one, so it need not be pip-installed
# into the conda environment. It is put first on PYTHONPATH, so it wins over an installed copy.
BETAESTIMATION_DIR="${BETAESTIMATION_DIR:-$(dirname "$SCRIPTS_DIR")/betaestimation}"
if [ -f "$BETAESTIMATION_DIR/betaestimation/__init__.py" ]; then
    export PYTHONPATH="$BETAESTIMATION_DIR${PYTHONPATH:+:$PYTHONPATH}"
fi

# fail early, with the import error, if the active python cannot load the pipeline's dependencies
check_python_env() {
    if ! ( cd "$SCRIPTS_DIR" && python -c "import utils.rsa_common" ); then
        echo "$(basename "$0"): python ($(command -v python)) cannot import the pipeline (see the error above)." >&2
        echo "  betaestimation: clone it next to this repo ($BETAESTIMATION_DIR) or set BETAESTIMATION_DIR;" >&2
        echo "  other packages: pip install -r $SCRIPTS_DIR/requirements.txt in the conda env (\$RSA_CONDA_ENV)." >&2
        exit 1
    fi
}
