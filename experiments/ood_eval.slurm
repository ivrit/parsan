#!/bin/bash
#SBATCH --job-name=heb-ood
#SBATCH --partition=work
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=2:00:00
#SBATCH --output=ood_eval_%j.out
# Usage: [PROFILE=base] [SEGMENTER=rftok] [NORMALIZE=1] sbatch experiments/ood_eval.slurm
cd "$SLURM_SUBMIT_DIR"
export PARSAN_RUNS=$PWD/runs PARSAN_LEMMA_BANK=$PWD/data/lemma_bank_amir.json
PY=${PY:-python}
PROFILE=${PROFILE:-base}; SEGMENTER=${SEGMENTER:-rftok}
NORMFLAG=""; [ "${NORMALIZE:-1}" = "0" ] && NORMFLAG="--no-normalize"
D=$PWD/ood/data
GENRES="bagatz geektime davar israelhayom allrights"

for g in $GENRES; do
  $PY scripts/predict.py --text $D/ood_$g.txt --sent newline \
      --profile "$PROFILE" --segmenter "$SEGMENTER" $NORMFLAG --out $D/preds_$g.conllu
done
PAIRS=""
for g in $GENRES; do PAIRS="$PAIRS --pair $g $D/ood_$g.conllu $D/preds_$g.conllu"; done
echo "===== OOD ($PROFILE / $SEGMENTER / normalize=${NORMALIZE:-1}) ====="
$PY scripts/evaluate.py $PAIRS
echo "=== EXIT $? ==="
