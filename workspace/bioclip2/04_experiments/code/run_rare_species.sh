#!/usr/bin/env bash
# Run BioCLIP2 hierarchical prompt experiments on imageomics/rare-species.
#
# Usage (override options with environment variables):
#   RS_ROOT=/content/drive/MyDrive/bioclip2_data/rare_species bash run_rare_species.sh
#   MODEL=bioclip2 BATCH_SIZE=64 AMP=1 DEVICE=cuda bash run_rare_species.sh
#   MAX_PER_CLASS=5 bash run_rare_species.sh
#
# Steps:
#   STEP 1: export HuggingFace Rare Species to images/ + taxonomy.csv.
#   STEP 2: run RQ1 + RQ2 + RQ3 through run_experiment.py.
#   STEP 3: list JSON outputs in OUT_DIR.

set -e

cd "$(dirname "$0")"

# ----------------------------------------------------------------------------
# Options (environment variables may override these)
# ----------------------------------------------------------------------------
RS_ROOT="${RS_ROOT:-../data/rare_species}"
MODEL="${MODEL:-openclip-vitl14}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-4}"
N_SEEDS="${N_SEEDS:-5}"
SEED="${SEED:-42}"
AMP="${AMP:-1}"
MAX_PER_CLASS="${MAX_PER_CLASS:-}"
DEVICE="${DEVICE:-cuda}"

OUT_DIR="${OUT_DIR:-../results/rare_species_${MODEL}}"
TAX_CSV="${RS_ROOT}/taxonomy.csv"
IMAGE_ROOT="${RS_ROOT}/images"
CACHE_EMB="${CACHE_EMB:-${OUT_DIR}/img_emb_${MODEL//\//_}.npz}"

# ----------------------------------------------------------------------------
# STEP 1 - prepare Rare Species export
# ----------------------------------------------------------------------------
echo "[STEP 1] Preparing Rare Species at ${RS_ROOT} (complete export -> skip)"
python prepare_rare_species.py --out_dir "$RS_ROOT"

# ----------------------------------------------------------------------------
# STEP 2 - run experiments
# ----------------------------------------------------------------------------
mkdir -p "$OUT_DIR" "$(dirname "$CACHE_EMB")"

ARGS=(
    "run_experiment.py"
    "--csv"         "$TAX_CSV"
    "--image_root"  "$IMAGE_ROOT"
    "--model"       "$MODEL"
    "--seed"        "$SEED"
    "--n_seeds"     "$N_SEEDS"
    "--batch_size"  "$BATCH_SIZE"
    "--num_workers" "$NUM_WORKERS"
    "--out"         "$OUT_DIR"
    "--cache_emb"   "$CACHE_EMB"
)
[[ "$AMP" == "1" ]]         && ARGS+=("--amp")
[[ -n "$DEVICE" ]]          && ARGS+=("--device" "$DEVICE")
[[ -n "$MAX_PER_CLASS" ]]   && ARGS+=("--max_per_class" "$MAX_PER_CLASS")

echo "[STEP 2] python ${ARGS[*]}"
python "${ARGS[@]}"

# ----------------------------------------------------------------------------
# STEP 3 - summary
# ----------------------------------------------------------------------------
echo ""
echo "[STEP 3] Done. Outputs:"
ls -la "$OUT_DIR"
