#!/bin/bash
#SBATCH --job-name=rsa_spreadsheet
#SBATCH --time=00:30:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
set -euo pipefail
cd "$REPO"
$PYTHON workflows/generate_master_spreadsheet.py --config "$CONFIG"
$PYTHON utils/validate_rsa_config.py --config "$CONFIG"
