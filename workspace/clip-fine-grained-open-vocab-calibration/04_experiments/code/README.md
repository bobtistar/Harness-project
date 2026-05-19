# Code Skeleton — fine-grained × open-vocabulary CLIP calibration

## Layout

```
code/
├── data.py                       # Dataset loaders (torchvision + ImageFolder)
├── models.py                     # open_clip wrapper, prompt ensemble
├── calibrators.py                # C0–C6: Raw, TS, VS, Dirichlet, DAC, CAC, HB
├── metrics.py                    # ECE family, AURC, conformal helpers
├── utils.py                      # seed, env snapshot, run log
├── run_all.py                    # Driver
├── run_all.sh                    # Full-sweep convenience script
├── requirements.txt
└── experiments/
    ├── exp1_diagnosis.py         # RQ1: fine-grained vs coarse ΔECE
    ├── exp2_geometry.py          # RQ2: τ_txt regression + whitening
    ├── exp3_calibrators.py       # RQ3: head-to-head Pareto
    └── exp4_downstream.py        # RQ4: ECE → AURC / conformal
```

## Install

```bash
pip install -r requirements.txt
```

## Sanity-check (CIFAR-10 toy slice)

```bash
python -m experiments.exp1_diagnosis --datasets cifar10 --backbone ViT-B-32 \
    --pretrained openai --toy
```

The toy path loads 128 CIFAR-10 images and finishes in seconds — useful to
confirm the code path works.  See `outputs/exp1_results.json`.

## Full sweep

```bash
bash run_all.sh
```

(adjust `--datasets`, `--backbone`, `--pretrained`, `--seeds` as needed)

## Outputs

- `outputs/exp{1..4}_results.json` — per-experiment records.
- `outputs/logs/exp{1..4}.json` — env snapshot, git hash, CLI args.

## Notes / Caveats

- For RQ3 calibrators whose parameter count depends on K (VS, Dirichlet, HB),
  the union/novel evaluation falls back to a temperature-scaling proxy. This
  is documented in `expected_results.md §Limitations` and matches the DAC
  paper convention.
- CAC requires a "zero-shot view" of the logits. Without a fine-tuned CLIP
  checkpoint, we substitute a scaled (×0.5) version of the same logits to
  keep the code paths exercised. **In the headline experiments this MUST be
  replaced with logits from the un-fine-tuned CLIP.**
- DAC's "distance to base" uses the calibration-set text embeddings; for
  novel classes it queries the same base-text matrix, exactly as in
  Wang et al. 2024.
