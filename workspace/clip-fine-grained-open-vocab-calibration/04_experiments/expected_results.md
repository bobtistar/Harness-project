# Expected Results & Sanity-Check Status

> **EXECUTION STATUS — 2026-05-20**
>
> - **Sanity-check (toy CIFAR-10, ViT-B/32 OpenAI, single seed)**: PASSED.
>   All four scripts (`exp1_diagnosis`, `exp2_geometry`, `exp3_calibrators`,
>   `exp4_downstream`) complete successfully and write valid JSON to
>   `code/outputs/*.json`.  These toy numbers are useful only to verify the
>   code path, not to draw scientific conclusions.
> - **Full sweep (11 datasets × 2 backbones × 3 seeds × 7 calibrators)**:
>   **NOT EXECUTED** in this environment.  Reason: most fine-grained
>   datasets (CUB-200, iNat subset, ImageNet-1k validation, SUN397) require
>   manual download outside torchvision's automatic fetcher.  The headline
>   numbers below are therefore **expected / hypothesised**, not measured.

The expected directions below are anchored on (i) DAC paper Tables 1–3
(Wang et al. ICML 2024, arXiv:2402.04655), (ii) CAC paper Table 2
(Lv et al. arXiv:2501.19060), and (iii) DOR paper Table 1 (Wang et al.
arXiv:2410.02681).  Numerical values are **placeholders** until the full
sweep runs.

---

## RQ1 — fine-grained vs coarse ΔECE

**Expected direction.** ΔECE (=ECE_novel − ECE_base) on fine-grained
benchmarks ≈ +5–9 pp on average vs +2–3 pp on coarse benchmarks (≥1.5×
ratio).  Paired Wilcoxon between matched (CoOp+TS) configurations across
dataset groups should yield p<0.05.

| Group | Backbone | n datasets | Expected mean ΔECE (raw) | Expected mean ΔECE (+TS) |
|---|---|---|---|---|
| Fine-grained | ViT-B/16 | 7 | ~+6.5 pp | ~+4.0 pp |
| Coarse       | ViT-B/16 | 4 | ~+2.0 pp | ~+1.0 pp |
| Fine-grained | ViT-L/14 | 7 | ~+5.0 pp | ~+3.0 pp |
| Coarse       | ViT-L/14 | 4 | ~+1.5 pp | ~+0.7 pp |

**Statistical test.** Mann–Whitney U (one-sided, alternative=greater) on
per-dataset ΔECE between groups.  Holm correction across {backbone × prompt}
pairs (4 combinations).

**Decision rule.** H1 is supported iff (mean ratio ≥1.5) AND (p<0.05 after
correction) AND (the relationship survives a class-count-matched
sub-sampling robustness check).

**Risks / threats to validity.**
- Confounding by class count — controlled via 200-class-cap sub-sampling.
- Confounding by image resolution — all inputs go through the same CLIP
  preprocess transforms.
- Confounding by pre-training corpus overlap — reported alongside as
  LAION-400M token-frequency covariate.

---

## RQ2 — τ_txt as a causal driver

**Expected direction.**
1. *Correlation.* Spearman ρ(τ_txt, ECE_union) across 11 datasets ≥ +0.55
   (positive) with p<0.05; partial correlation conditioning on accuracy
   ≥ +0.45.
2. *Intervention.* PCA-whitening of the K×D text-embedding matrix
   decreases ECE by ≥1 pp on at least 8/11 datasets; sign test p<0.01.
3. *Dose-response.* When we synthetically interpolate from
   non-whitened to fully-whitened embeddings, ECE decreases monotonically.

**Sanity-check note.** On the toy CIFAR-10 slice (128 images,
ViT-B/32) PCA-whitening reduced ECE from 0.105 → 0.095 (+small
improvement); orthogonalisation actually *increased* ECE to 0.341
(over-rotation destroyed class semantics).  This matches the warning in
the DAC paper that *destructive* embedding manipulations can hurt
accuracy.  Headline experiments must monitor accuracy drop ≤ 0.5 pp.

**Statistical test.** Spearman correlation with permutation p-value
(10k perms).  Paired bootstrap CI on the per-dataset ΔECE.  Sign test
for the intervention's direction.

**Risks.** Orthogonalisation may distort semantics; we will compare
sentence-BERT similarity before/after intervention and exclude cases
with > 0.2 cosine drop.

---

## RQ3 — Calibrator head-to-head

**Expected direction (per Wang24 / Lv25 paper trends).**

| Calibrator | base ECE | novel ECE | union ECE | ΔECE (novel-base) | accuracy drop |
|---|---|---|---|---|---|
| C0 Raw | ~0.04 | ~0.13 | ~0.09 | +0.09 | 0 |
| C1 TS (Guo17) | ~0.03 | ~0.10 | ~0.07 | +0.07 | 0 |
| C2 Vector Scaling | ~0.025 | ~0.11 | ~0.07 | +0.085 | ≤0.3 |
| C3 Dirichlet (Kull19) | ~0.025 | ~0.10 | ~0.07 | +0.075 | ≤0.3 |
| C4 **DAC** (Wang24) | ~0.028 | ~0.06 | ~0.045 | +0.032 | 0 |
| C5 **CAC** (Lv25) | ~0.030 | ~0.055 | ~0.043 | +0.025 | 0 |
| C6 Histogram Binning | ~0.02 | ~0.13 | ~0.08 | +0.11 | ≤0.5 |

**Expected Pareto frontier.** DAC and CAC dominate; CAC achieves the
lowest novel ECE.  TS provides the strongest unconditional baseline.

**Statistical tests.**
- Paired bootstrap CI on (ECE_C1 − ECE_Ck)/ECE_C1, target ≥30%.
- Friedman test across 11 datasets per backbone; Nemenyi post-hoc at
  α=0.05.
- Effect size: rank-biserial correlation between paired calibrator
  rankings.

**Pass criteria (per RQ).**
1. Best-of-set reduces union ECE by ≥30% vs TS  *(expected to pass for CAC/DAC)*.
2. Best-of-set novel/base ECE ratio ≤ 1.5  *(borderline; sub-classes vary)*.
3. Accuracy drop ≤ 0.5 pp  *(expected to pass for TS/VS/DAC/CAC, borderline for Dirichlet/HB on small calibration sets)*.

**Limitations / caveats noted in the skeleton.**
- VS / Dirichlet / HB parameter dimensionality depends on K.  When
  predicting novel-class logits whose K differs from training K, our
  code falls back to a temperature-scaling proxy.  In a full run we
  recommend either (a) re-fitting these calibrators on the union class
  set (still using only base labels for the loss) or (b) following the
  DAC paper convention and reporting them only on the base split.
- CAC requires a *zero-shot view* of the same image-text pair.  In a
  proper experiment the zero-shot view comes from an un-fine-tuned
  CLIP; in our skeleton we use a `×0.5` scaled proxy purely so the code
  path exercises end-to-end.  **Headline runs must replace this.**

---

## RQ4 — ECE → AURC / conformal monotonic transfer

**Expected direction.**
- Spearman ρ(ECE_union, AURC) ≤ −0.55 within each fine-grained dataset
  (i.e. lower ECE → lower AURC).
- Average conformal set size at α=0.1 decreases monotonically with ECE
  improvement; per-dataset slope sign negative for ≥9/11 datasets.
- |empirical coverage − 0.9| ≤ 1 pp for all calibrators on base-test
  and within ±2 pp on novel-test (acceptable since novel is OOD relative
  to calibration set).

**Statistical test.** Mixed-effects regression
`y ~ ece + (1 | dataset)` with `y ∈ {AURC, mean_set_size}`.  Report
fixed-effect coefficient, 95% CI, and likelihood-ratio test vs null.

**Coverage diagnostic.** A separate table reports
(empirical_coverage_base, empirical_coverage_novel) for each calibrator
— any calibrator with coverage gap > 1 pp on either is flagged as
unsuitable for a deployment claim, even if it improves ECE.

**Risks.** AURC and ECE are not mathematically guaranteed to move
together (calibration changes confidence ranking only when the
calibrator is non-monotone, e.g. Histogram Binning).  We expect
monotone calibrators (TS, VS) to leave AURC unchanged, and to see
movement primarily for HB / Dirichlet / DAC / CAC.  This is consistent
with Andrade-Loarca 2024's analysis.

---

## Limitations of the Current Skeleton

1. **No fine-tuned CLIP checkpoints**: all experiments here use the
   *zero-shot* OpenCLIP weights.  Headline numbers in the paper should
   re-run with CoOp/CoCoOp/MaPLe checkpoints.  The infrastructure is in
   place — only the model-loading line needs to be replaced.
2. **Dataset coverage**: torchvision automatically fetches Flowers102,
   Food101, FGVCAircraft, OxfordIIITPet, StanfordCars, Caltech101,
   EuroSAT, and CIFAR-10.  CUB-200, SUN397, ImageNet-1k val, and the
   iNat subset must be staged into `$CLIP_DATA_ROOT` manually
   (ImageFolder layout).  Loader stubs already exist in `data.py`.
3. **Statistical tests on toy data are not informative**: the small
   number of CIFAR-10 toy samples (128) and the fact that CIFAR-10 is a
   coarse dataset mean the per-RQ effect sizes cannot be evaluated
   here.  Use the toy run only to verify code paths.
4. **DAC base-text policy**: we use the calibration-time base text
   matrix as DAC's anchor.  When the novel test runs, DAC computes
   distance against the same anchor — matching the original paper.
