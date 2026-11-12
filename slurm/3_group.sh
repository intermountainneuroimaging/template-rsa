#!/bin/bash
#SBATCH --job-name=rsa_group
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
set -euo pipefail
cd "$REPO"
ANALYSES=$($PYTHON -c "import json,sys; print(' '.join(json.load(open(sys.argv[1])).get('pipeline',{}).get('analyses',['rdm'])))" "$CONFIG")
[[ " $ANALYSES " == *" rdm "* ]] && $PYTHON workflows/group_rsa.py --config "$CONFIG"
[[ " $ANALYSES " == *" template "* ]] && $PYTHON workflows/group_template_rsa.py --config "$CONFIG"
[[ " $ANALYSES " == *" fidelity "* ]] && $PYTHON workflows/group_fidelity.py --config "$CONFIG"
$PYTHON workflows/generate_report.py --config "$CONFIG"
