#!/usr/bin/env bash
# Full reproduction script (Linux / Mac).
# Estimated time on a single A100 40GB:
#   - toy run (this script as-is):             <1 min
#   - openclip ViT-L/14 + 1000 species:        ~30 min
#   - bioclip2 ViT-L/14 + 1000 species:        ~45 min (incl. HF download ~6 GB)
#   - all 5 domains (RQ4):                     ~6 hours

set -e
cd "$(dirname "$0")"
python -m pip install -r requirements.txt

# 1) Toy sanity check (always runs, no network needed except OpenCLIP weights)
python run_experiment.py --model openclip-vitb32 --toy --seed 42 --out ../results/toy_openclip

# 2) Fully offline mock (in case OpenCLIP weights cannot be downloaded)
python run_experiment.py --mock --toy --seed 42 --out ../results/mock

# 3) Full-scale runs (uncomment when ready and on a GPU machine)
# python run_experiment.py --model openclip-vitl14 --csv data/treeoflife_eval.csv --image_root data/images --device cuda --out ../results/openclip_vitl14
# python run_experiment.py --model bioclip        --csv data/treeoflife_eval.csv --image_root data/images --device cuda --out ../results/bioclip
# python run_experiment.py --model bioclip2       --csv data/treeoflife_eval.csv --image_root data/images --device cuda --out ../results/bioclip2

# 4) Cross-domain (RQ4) — re-run with --csv pointing at each domain split
# for dom in Aves Insecta Plantae Fungi Actinopterygii; do
#   python run_experiment.py --model bioclip2 --csv data/${dom}.csv --image_root data/images --device cuda --out ../results/bioclip2_${dom}
# done
