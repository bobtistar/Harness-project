# Code: BioCLIP2 Hierarchical Prompt — Geometric Diagnostic & Counterfactual Ablation

This directory holds the runnable experiment skeleton for:

- **RQ1** — `experiment_1` in `run_experiment.py`
- **RQ2** — `experiment_2` (per-rank + latent taxonomy probe)
- **RQ3** — `experiment_3` (6-condition counterfactual C0..C5)
- **RQ4** — re-run RQ1/RQ3 with `--csv data/<domain>.csv`

## File map

```
code/
├── data_loader.py        # toy dataset + real CSV loader
├── prompt_variants.py    # C0..C5 prompt generators
├── metrics.py            # geometry & taxonomic metrics, bootstrap, permutation
├── extract_embeddings.py # OpenCLIP / BioCLIP / BioCLIP2 loaders + mock fallback
├── models/
│   ├── baseline_flat.py     # thin alias for C0
│   └── proposed_textfree.py # C5: linear adapter + hierarchical InfoNCE
├── run_experiment.py     # main entry (argparse)
├── run_all.sh / .ps1     # full reproduction scripts
├── requirements.txt
└── README.md (this file)
```

## Quickstart

```bash
# 1) Install dependencies (assumes Python 3.10+; CUDA optional)
pip install -r requirements.txt

# 2) Fully offline mock run (no model weights, no internet)
python run_experiment.py --mock --toy --out ../results/mock

# 3) Real OpenCLIP toy run (downloads ViT-B/32 weights ~600 MB on first call)
python run_experiment.py --model openclip-vitb32 --toy --out ../results/toy_openclip

# 4) BioCLIP2 on a real CSV
python run_experiment.py --model bioclip2 \
    --csv data/treeoflife_eval.csv \
    --image_root data/images \
    --device cuda --out ../results/bioclip2

# Optional: use a HuggingFace token for authenticated downloads
python run_experiment.py --model bioclip2 \
    --csv data/treeoflife_eval.csv \
    --image_root data/images \
    --hf_token_file ~/.cache/huggingface/token \
    --out ../results/bioclip2
```

## HuggingFace token via `.env`

You can also put the token in a local `.env` file at the repository root or in
this `code/` directory:

```bash
HF_TOKEN=hf_your_token_here
# or:
HUGGING_FACE_HUB_TOKEN=hf_your_token_here
```

`run_experiment.py` automatically searches upward for `.env`. The repository
`.gitignore` excludes `.env` and `.env.*`, so the token should stay local.

```bash
python run_experiment.py --model bioclip2 \
    --csv ~/CUB_200_2011/cub_taxonomy.csv \
    --image_root ~/CUB_200_2011/images \
    --out ../results/bioclip2
```

To use a non-default env file:

```bash
python run_experiment.py --env_file /path/to/.env --model bioclip2 ...
```

## Expected outputs (in `--out`)

```
exp1_geometry.json          # RQ1 metrics + paired permutation p-values
exp2_rank_levels.json       # RQ2 per-rank silhouette + latent taxonomy probe
exp3_counterfactuals.json   # RQ3 preservation ratios for C2..C5
run_log.txt
```

## Expected runtime

| Configuration | Runtime |
|---|---|
| `--mock --toy` (CPU)            | <5 s |
| `--model openclip-vitb32 --toy` | ~30 s (CPU), <5 s (GPU) |
| OpenCLIP ViT-L/14 + 1000 species| ~30 min on A100 |
| BioCLIP2 ViT-L/14 + 1000 species| ~45 min on A100 (plus ~6 GB HF download) |
| Full RQ1+RQ2+RQ3 + 5 domains    | ~6 hours on A100 |

## CSV format for real runs

```csv
file,kingdom,phylum,class,order,family,genus,species
img_0001.jpg,Animalia,Chordata,Aves,Passeriformes,Passeridae,Passer,Passer domesticus
img_0002.jpg,Animalia,Chordata,Aves,Passeriformes,Fringillidae,Carduelis,Carduelis carduelis
...
```

`file` is relative to `--image_root`. The `class` column is the Linnaean rank
(also Python keyword — handled internally).

## Reproducibility checklist

- All random sampling routes through `numpy.random.default_rng(seed)`.
- `torch.manual_seed`, `torch.backends.cudnn.deterministic=True` set in `set_global_seeds`.
- Model checkpoints are versioned by `pretrained` tag in `MODEL_REGISTRY`.
- `requirements.txt` pins major versions; for byte-exact reproduction freeze
  output of `pip freeze` after first successful run.

## Known limitations / caveats

1. **C5 adapter is a *light* baseline.** For a real comparison against the BioCLIP2
   paper, replace `LinearAdapter` with LoRA on the image encoder's attention
   projections. The current skeleton trains only the adapter on top of frozen
   embeddings to keep CPU feasibility.
2. **Mock text embeddings** hash prompts -> deterministic vectors. They produce
   geometrically plausible numbers but the *absolute values must not be
   interpreted as evidence about real CLIP encoders*. The mock is used only to
   validate the analysis pipeline end-to-end.
3. The toy dataset is synthetic and primarily exercises the analysis code.
   It will *not* reproduce paper-quality effect sizes.
