#!/bin/bash
#SBATCH --job-name=rsa_group
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
set -euo pipefail
cd "$REPO"
$PYTHON workflows/group_rsa.py --config "$CONFIG"
$PYTHON workflows/generate_report.py --config "$CONFIG"
