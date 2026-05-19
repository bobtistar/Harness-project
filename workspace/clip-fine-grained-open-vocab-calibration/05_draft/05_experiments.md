# 4. Experimental Protocol

This section specifies the datasets, models, baselines, metrics, hyper-
parameters, and statistical tests against which RQ1--RQ4 are evaluated.
Every experiment shares the seed set $\{0, 1, 2\}$, the OpenCLIP
backbones, and the CoOp base-to-novel split.

## 4.1 Datasets

**Fine-grained group (7).** CUB-200-2011 (200 classes), Stanford Cars
(196), FGVC-Aircraft (100), Flowers-102 (102), Oxford-IIIT Pets (37),
Food-101 (101), iNaturalist-2021 subset (200 species). These benchmarks
are characterised by small inter-class image differences and nearly
isomorphic class-name strings.

**Coarse group (4, RQ1 control).** ImageNet-1k val (1000), Caltech-101
(101), SUN397 (397), EuroSAT (10).

**Base-to-novel split.** Each class set is partitioned alphabetically:
the first half becomes $\mathcal{C}_{\text{base}}$, the second half
$\mathcal{C}_{\text{novel}}$. Three random splits (CoOp's seed1/2/3) are
averaged. Per-class test sub-sampling is capped at 50 images for the
class-balanced sensitivity analysis reported in the appendix.

**Confounder controls.** (i) Class-count matching: a 200-class
sub-sample of ImageNet/SUN397 is reported alongside the full-class
result. (ii) Sample-count matching: per-class cap to $\min(50,
\text{available})$. (iii) Pre-training exposure proxy: LAION-400M
metadata class-name token frequency reported as a covariate where
available. (iv) Prompt template: results reported for both
`"a photo of a {C}."` and the 7-template OpenAI ensemble.

## 4.2 Backbones and fine-tuning recipes

**Backbones.** OpenCLIP ViT-B/16 (`laion2b_s34b_b88k`) and ViT-L/14
(`laion2b_s32b_b82k`)~\cite{cherti2023openclip}.

**Fine-tuning recipes** (we re-use published checkpoints where
available; zero-shot is the fallback): CoOp~\cite{zhou2022coop} with
$n_{\text{ctx}}=16$, 16-shot, 50 epochs; CoCoOp~\cite{zhou2022cocoop}
with $n_{\text{ctx}}=4$ and image-conditioned context; MaPLe~\cite{khattak2023maple}
with prompt depth 9; LoRA-adapter (rank 8 on image-encoder QKV) as a
secondary recipe.

## 4.3 Baselines

We compare seven calibrator function classes (C0--C6) as listed in §3.3.
The recipe-level baselines are: B0 zero-shot CLIP; B1 zero-shot + global
TS~\cite{levine2023zsts}; B2 CoOp/CoCoOp/MaPLe + TS; B3 DAC~\cite{wang2024dac};
B4 CAC~\cite{lv2025cac}; B5 Dirichlet~\cite{kull2019dirichlet}; B6
prompt-ensemble best recipe of~\cite{tu2024empirical}. For RQ4, the
conformal baseline is Conf-OT~\cite{silvarodriguez2025confot}; the
selective-prediction baseline is the MSP threshold of~\cite{geifman2017selective},
evaluated under the AURC protocol of~\cite{traub2024selective}.

## 4.4 Metrics

**Calibration.** ECE (15-bin equal-mass), Adaptive ECE~\cite{nixon2019measuring},
classwise ECE, MCE, Brier score, NLL. Reported on
$\mathcal{C}_{\text{base}}$, $\mathcal{C}_{\text{novel}}$, and their union.

**Accuracy.** Top-1 accuracy and the difference $\Delta\mathrm{acc} =
\mathrm{acc}_{\text{cal}} - \mathrm{acc}_{\text{raw}}$, required to
satisfy $|\Delta\mathrm{acc}| \leq 0.5\,\mathrm{pp}$ for any calibrator
to be considered valid.

**Selective prediction.** AURC (Geifman-El-Yaniv; finite-sample
estimator of~\cite{andradeloarca2024aurc}), selective accuracy at
coverage $\{0.7, 0.8, 0.9\}$, and AUROC of correct-vs-incorrect
prediction.

**Conformal prediction.** Split conformal with non-conformity score
$s(x, y) = 1 - p_y(x)$, at miscoverage $\alpha = 0.1$. We report
empirical coverage on base test and on novel test, and average set
size. A coverage gap $|\hat{\mathrm{cov}} - (1 - \alpha)|$ above
$1\,\mathrm{pp}$ flags the calibrator.

## 4.5 Implementation details

**Optimisation.** SGD with $\mathrm{lr}=2 \times 10^{-3}$, momentum
$0.9$, weight decay $5 \times 10^{-4}$, cosine schedule; batch size 32
(ViT-B/16) and 16 (ViT-L/14). Calibration parameters are fit by L-BFGS,
`max_iter=200`, `tolerance_grad=1e-7`, in fp32; the forward pass uses
fp16.

**Software.** Python 3.12, PyTorch 2.7.1+cu118, open_clip_torch 3.2.0,
torchvision 0.22.1+cu118, scikit-learn 1.5.1, scipy 1.13.1, numpy
1.26.4. The pinned `requirements.txt` is shipped with the code.

**Hardware.** NVIDIA GPU with $\geq 16\,\text{GB}$ VRAM (required for
ViT-L/14 inference at batch 16). The sanity-check environment is
Windows 11, CUDA 11.8.

**Seed control.** `torch.manual_seed`, `numpy.random.seed`, and
`random.seed` are all set per run, and
`torch.backends.cudnn.deterministic = True`.

**Reproducibility logging.** Every run writes git hash, env hash, CLI
arguments, seed, dataset checksum, and output path to
`logs/<run_id>.json`.

## 4.6 Statistical tests

| Hypothesis | Test | Multiple-comparison correction |
|------------|------|-------------------------------|
| H1.RQ1 (fine-grained $\Delta\mathrm{ECE}$ larger) | paired Wilcoxon signed-rank; group Mann–Whitney U (one-sided) | Holm–Bonferroni across (backbone $\times$ prompt) |
| H1.RQ2a (correlation) | Spearman $\rho$ with 10k permutations; partial Spearman conditioning on accuracy and #classes | Holm across (dataset-level, class-level) |
| H1.RQ2b (intervention) | paired bootstrap CI (10k); one-sided sign test on per-dataset $\Delta\mathrm{ECE}$ | Holm per dataset |
| H1.RQ3 (relative reduction $\geq 30\%$) | paired bootstrap CI on $(\mathrm{ECE}_{\text{TS}} - \mathrm{ECE}_{\text{C}_k}) / \mathrm{ECE}_{\text{TS}}$; Friedman + Nemenyi for rank aggregation | Nemenyi at $\alpha = 0.05$ |
| H1.RQ4 (monotonicity) | Spearman $\rho$ within each dataset; mixed-effects regression $y \sim \mathrm{ECE} + (1 \mid \mathrm{dataset})$ with $y \in \{\mathrm{AURC}, \bar{|\mathcal{S}|}\}$ | Holm across $\{\mathrm{AURC}, \bar{|\mathcal{S}|}\}$ |

Effect sizes are reported as paired Cohen's $d$ or rank-biserial
correlation, alongside 95% bootstrap CIs.

## 4.7 Execution status

The code path of all four experiments (`exp1_diagnosis.py`,
`exp2_geometry.py`, `exp3_calibrators.py`, `exp4_downstream.py`) has been
verified end-to-end on a toy CIFAR-10 slice (128 images, ViT-B/32
OpenAI, single seed). All four scripts complete successfully and write
valid JSON to `code/outputs/*.json`. The toy numbers verify the code
path; they do not support scientific conclusions. The full sweep
(11 datasets $\times$ 2 backbones $\times$ 3 seeds $\times$ 7
calibrators) was *not* executed in the current environment because
CUB-200, iNat subset, ImageNet-1k val, and SUN397 require manual
download outside torchvision's automatic fetcher. The full-sweep
infrastructure (`run_all.sh`) is in place; only the dataset staging and
the swap from zero-shot to fine-tuned checkpoints remain.

## 4.8 Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Missing fine-tuning checkpoint | Zero-shot CLIP fallback; explicit annotation in results |
| Dataset download restrictions | torchvision-fetchable datasets first; `ImageFolder` stubs for the rest |
| ViT-L/14 OOM | Gradient checkpointing; batch reduction; fp16 |
| Multiple-comparison false positives | Holm-Bonferroni correction applied throughout |
| Calibration-set hyperparameter overfitting | Base-validation NLL only; nested CV |
| Conformal coverage violation | Zero-shot baseline coverage reported as sanity check |
