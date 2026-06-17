#!/bin/bash
#SBATCH --job-name=heb-trainj
#SBATCH --partition=work
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=train_joint_%j.out
# Usage: PROFILE=base sbatch experiments/train_joint.slurm   (or PROFILE=tiny)
cd "$SLURM_SUBMIT_DIR"
export PARSAN_RUNS=$PWD/runs
PY=${PY:-python}
PROFILE=${PROFILE:-base}
D=$PWD/data/ud
$PY scripts/train_joint.py --profile "$PROFILE" \
    --train $D/he_iahltwiki-ud-train.conllu,$D/he_iahltknesset-ud-train.conllu \
    --dev   $D/he_iahltwiki-ud-dev.conllu,$D/he_iahltknesset-ud-dev.conllu \
    --test  $D/he_iahltwiki-ud-test.conllu,$D/he_iahltknesset-ud-test.conllu \
    --epochs 40
echo "=== EXIT $? ==="
