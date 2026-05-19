# Full reproduction script (Windows PowerShell).
# Estimated time on a single A100 40GB:
#   - toy run (this script as-is):             <1 min
#   - openclip ViT-L/14 + 1000 species:        ~30 min
#   - bioclip2 ViT-L/14 + 1000 species:        ~45 min (incl. HF download ~6 GB)
#   - all 5 domains (RQ4):                     ~6 hours

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot
python -m pip install -r requirements.txt

# 1) Toy sanity check (downloads OpenCLIP ViT-B/32 weights ~600 MB on first run)
python run_experiment.py --model openclip-vitb32 --toy --seed 42 --out ../results/toy_openclip

# 2) Fully offline mock
python run_experiment.py --mock --toy --seed 42 --out ../results/mock

# 3) Full-scale runs (uncomment on a GPU machine with HF cache)
# python run_experiment.py --model openclip-vitl14 --csv data/treeoflife_eval.csv --image_root data/images --device cuda --out ../results/openclip_vitl14
# python run_experiment.py --model bioclip        --csv data/treeoflife_eval.csv --image_root data/images --device cuda --out ../results/bioclip
# python run_experiment.py --model bioclip2       --csv data/treeoflife_eval.csv --image_root data/images --device cuda --out ../results/bioclip2

# 4) Cross-domain (RQ4)
# foreach ($dom in @("Aves", "Insecta", "Plantae", "Fungi", "Actinopterygii")) {
#   python run_experiment.py --model bioclip2 --csv ("data/{0}.csv" -f $dom) --image_root data/images --device cuda --out ("../results/bioclip2_{0}" -f $dom)
# }
