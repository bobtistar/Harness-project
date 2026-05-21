#!/usr/bin/env bash
# CUB-200-2011 데이터셋으로 BioCLIP2 hierarchical prompt 실험을 실행 (Linux/macOS).
#
# 사용법 (환경 변수로 옵션 제어):
#   CUB_ROOT=/workspace/CUB_200_2011 bash run_cub.sh
#   MODEL=bioclip2 BATCH_SIZE=128 AMP=1 bash run_cub.sh
#   MAX_PER_CLASS=5 bash run_cub.sh                # 빠른 테스트 (1000장)
#
# 단계:
#   STEP 1: pygbif로 200종 taxonomy 자동 생성 (약 3분, 캐시 후 즉시).
#   STEP 2: 이미지 임베딩 + 6 조건 ablation 실험.
#   STEP 3: 결과는 ../results/cub200_<model>/ 에 JSON으로 저장.

set -e

# ----------------------------------------------------------------------------
# 옵션 (환경변수로 오버라이드 가능)
# ----------------------------------------------------------------------------
CUB_ROOT="${CUB_ROOT:-/Users/feynman1227/프로젝트/Harness-project/workspace/bioclip2/CUB_200_2011}"
MODEL="${MODEL:-openclip-vitb32}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-4}"
N_SEEDS="${N_SEEDS:-5}"
SEED="${SEED:-42}"
AMP="${AMP:-0}"                       # 1이면 float16 사용
MAX_PER_CLASS="${MAX_PER_CLASS:-}"    # 빈값이면 전체 사용
DEVICE="${DEVICE:-}"                  # 빈값이면 자동 감지

OUT_DIR="${OUT_DIR:-../results/cub200_${MODEL}}"
TAX_CSV="${CUB_ROOT}/cub_taxonomy.csv"
CACHE_JSON="${CUB_ROOT}/cub_gbif_cache.json"
CACHE_EMB="${OUT_DIR}/img_emb_${MODEL//\//_}.npz"

cd "$(dirname "$0")"

# ----------------------------------------------------------------------------
# STEP 1 — taxonomy CSV 생성
# ----------------------------------------------------------------------------
if [[ -f "$TAX_CSV" ]]; then
    echo "[STEP 1] $TAX_CSV exists → skip"
else
    echo "[STEP 1] Building taxonomy via GBIF (~3 min) ..."
    python cub200_build_taxonomy.py \
        --cub_root "$CUB_ROOT" \
        --out_csv  "$TAX_CSV" \
        --cache    "$CACHE_JSON"
fi

# ----------------------------------------------------------------------------
# STEP 2 — 실험 실행
# ----------------------------------------------------------------------------
mkdir -p "$OUT_DIR"

ARGS=(
    "run_experiment.py"
    "--csv"         "$TAX_CSV"
    "--image_root"  "$CUB_ROOT/images"
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
# STEP 3 — 요약
# ----------------------------------------------------------------------------
echo ""
echo "[STEP 3] Done. Outputs:"
ls -la "$OUT_DIR"
