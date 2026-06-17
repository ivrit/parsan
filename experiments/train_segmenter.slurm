#!/bin/bash
#SBATCH --job-name=heb-seg
#SBATCH --partition=work
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=2:00:00
#SBATCH --output=train_seg_%j.out
cd "$SLURM_SUBMIT_DIR"
export PARSAN_RUNS=$PWD/runs
PY=${PY:-python}
D=$PWD/data/seg
$PY scripts/train_segmenter.py \
    --train $D/iahlt_train_ctx.tab --dev $D/iahlt_dev_ctx.tab --test $D/iahlt_test_ctx.tab \
    --model dicta-il/dictabert-char --out runs/seg_char_ctx --epochs 15 --batch 32
echo "=== EXIT $? ==="
