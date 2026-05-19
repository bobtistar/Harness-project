#!/usr/bin/env bash
# Full sweep: 6 datasets x 1 seed (extend --seeds 0 1 2 for headline runs).
set -euo pipefail

DATASETS="flowers102 food101 fgvc_aircraft oxford_pets caltech101 eurosat"

python -m experiments.exp1_diagnosis --datasets $DATASETS --backbone ViT-B-16 --pretrained laion2b_s34b_b88k --seeds 0 1 2 --out outputs/exp1_results.json
python -m experiments.exp2_geometry  --datasets $DATASETS --backbone ViT-B-16 --pretrained laion2b_s34b_b88k --seed 0           --out outputs/exp2_results.json
python -m experiments.exp3_calibrators --datasets $DATASETS --backbone ViT-B-16 --pretrained laion2b_s34b_b88k --seeds 0 1 2 --out outputs/exp3_results.json
python -m experiments.exp4_downstream  --datasets $DATASETS --backbone ViT-B-16 --pretrained laion2b_s34b_b88k --seeds 0 1 2 --out outputs/exp4_results.json
