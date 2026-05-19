# Experimental Protocol

본 프로토콜은 `02_rqs.md`에 정의된 4개 RQ를 검정하기 위한 데이터셋, 모델, 베이스라인, 메트릭, 하이퍼파라미터, 통계 검정을 정밀하게 명세한다. 모든 실험은 동일한 시드 집합 `{0, 1, 2}`, 동일한 backbone 체크포인트, 동일한 base-to-novel split을 공유한다.

---

## 1. Datasets

### 1.1 Fine-grained Group (7 datasets)
| Dataset | #classes | #test | License | Source |
|---|---|---|---|---|
| CUB-200-2011 | 200 | 5,794 | Research-only | Wah et al. 2011 |
| Stanford Cars | 196 | 8,041 | Research-only | Krause et al. 2013 |
| FGVC-Aircraft | 100 | 3,333 | Research-only | Maji et al. 2013 |
| Flowers-102 | 102 | 6,149 | Non-commercial | Nilsback & Zisserman 2008 |
| Oxford-IIIT Pets | 37 | 3,669 | CC-BY-SA 4.0 | Parkhi et al. 2012 |
| Food-101 | 101 | 25,250 | Research-only | Bossard et al. 2014 |
| iNaturalist-2021 subset (200 species) | 200 | ~10k | CC-BY-NC | iNat 2021 |

### 1.2 Coarse Group (4 datasets, RQ1 대조군)
| Dataset | #classes | #test | License |
|---|---|---|---|
| ImageNet-1k (val) | 1000 | 50,000 | ImageNet T&Cs |
| Caltech-101 | 101 | 2,465 | CC-BY |
| SUN397 | 397 | ~19,850 | Research-only |
| EuroSAT | 10 | 5,400 | MIT |

### 1.3 Splits (CoOp base-to-novel convention)
- 각 데이터셋에서 클래스를 알파벳 순으로 정렬 후 처음 절반을 **base**, 나머지를 **novel** 로 지정.
- Train: base classes의 16-shot (per-class) — fine-tuning baselines (CoOp/CoCoOp/MaPLe) 학습에만 사용.
- Calibration (val): base classes의 held-out validation (per-class N=512 cap; 부족 시 전체).
- Test: base classes test set + novel classes test set 분리.
- Seeds: `{0, 1, 2}` — 세 random base/novel split (CoOp의 seed1/2/3에 해당).

### 1.4 Confounder 통제
- **클래스 수 매칭**: ImageNet/SUN397과 fine-grained 비교 시 200/100 class sub-sampling 변종도 보고.
- **샘플 수 매칭**: per-class test 샘플 수를 min(50, available) 로 캡한 결과를 부록에 보고.
- **Pre-training 노출도 proxy**: LAION-400M 메타데이터에서 class-name token frequency를 측정해 covariate로 보고(가능 시).
- **프롬프트 템플릿**: `"a photo of a {C}."` 고정 결과 + 7-template OpenAI ensemble 결과를 양쪽 보고.

---

## 2. Models / Backbones

- **Primary backbones**: OpenCLIP ViT-B/16 (`laion2b_s34b_b88k`) 및 ViT-L/14 (`laion2b_s32b_b82k`).
- **Fine-tuning recipes** (PromptLearning 계열):
  - CoOp: `n_ctx=16`, class-token position end, 16-shot, 50 epochs.
  - CoCoOp: image-conditioned, `n_ctx=4`.
  - MaPLe: layers J=9, prompt depth 9.
  - LoRA-adapter (보조): rank=8 on image encoder QKV.
- **체크포인트 공급원**: CoOp/MaPLe 공식 repo의 base-only 학습 체크포인트. 재현이 어려운 경우 zero-shot CLIP을 fallback 기준선으로 사용하고 명시.

---

## 3. Calibrators (RQ3 함수 클래스 7종)

| ID | Name | Function class | Trainable params | Calibration data |
|---|---|---|---|---|
| C0 | Raw (uncalibrated) | identity | 0 | — |
| C1 | Temperature Scaling (TS) | `softmax(z/T)`, scalar `T>0` | 1 | base val |
| C2 | Vector Scaling (VS) | `softmax(diag(w)·z + b)` | 2K | base val |
| C3 | Matrix / Dirichlet | `softmax(W·log_softmax(z) + b)` | K(K+1) | base val (with L2 reg) |
| C4 | DAC | `T_i = T·g(d_txt(y_i, base))` | 1 + analytic | base val |
| C5 | CAC | `softmax( z_ft − α·(z_zs − z_ft) )` weighting | 1 | base val (+ZS logits) |
| C6 | Histogram Binning (HB) | per-class bin map | 15·K | base val |

- 검색 그리드: `T ∈ [0.5, 5.0]` (40 log-spaced), `α ∈ {0.1, 0.3, 0.5, 1.0}`. L-BFGS로 NLL 최소화.
- **Hyperparameter selection**: base validation NLL 기준 (novel 누설 금지). early-stop patience 3.

---

## 4. Metrics

### 4.1 Calibration
- **ECE** (15-bin equal-mass binning, Guo et al. 2017):
  $$ \mathrm{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{n} \, |\mathrm{acc}(B_m) - \mathrm{conf}(B_m)| $$
- **Adaptive ECE** (Nixon 2019): equal-count bins.
- **Classwise ECE**: per-class one-vs-rest ECE 평균.
- **MCE**: $\max_m |\mathrm{acc}(B_m) - \mathrm{conf}(B_m)|$.
- **Brier score**: $\frac{1}{n}\sum_i \|p_i - \mathbf{1}_{y_i}\|_2^2$.
- **NLL**: $-\frac{1}{n}\sum_i \log p_i[y_i]$.
- **ΔECE** := ECE_novel − ECE_base.

### 4.2 Accuracy
- top-1 accuracy, accuracy difference between calibrated and raw (must be ≤ 0.5 pp).

### 4.3 Selective Prediction
- **AURC** (area under risk-coverage), as defined by Geifman & El-Yaniv 2017; finite-sample estimator from Andrade-Loarca 2024.
- **Selective accuracy** at coverage ∈ {0.7, 0.8, 0.9}.
- **AUROC for misclassification detection** (binary: correct vs incorrect).

### 4.4 Conformal Prediction
- Split conformal with non-conformity score `s(x,y) = 1 − p_y(x)`.
- Empirical coverage at α=0.1; |coverage − (1−α)| ≤ 1 pp 목표.
- Average prediction set size (efficiency).

---

## 5. Baselines (per RQ)

| RQ | Baselines actually compared |
|---|---|
| RQ1 | Zero-shot CLIP, CoOp+TS, MaPLe+TS — fine-grained vs coarse 그룹 분리하여 ΔECE 측정. |
| RQ2 | Same models as RQ1; intervention = PCA-whitening of text embeddings (and orthogonalization). |
| RQ3 | C0–C6 (7 함수 클래스), 11 datasets × 2 backbones × 3 seeds. |
| RQ4 | RQ3의 ECE 스펙트럼을 입력으로 받아 selective/conformal 지표 측정. Conf-OT를 비교군. |

---

## 6. Hyperparameters & Training

- Optimizer: SGD, lr=0.002, momentum=0.9, weight_decay=5e-4 (CoOp recipe).
- Scheduler: cosine.
- Batch size: 32 (ViT-B/16), 16 (ViT-L/14).
- Calibration optim: L-BFGS, max_iter=200, tolerance_grad=1e-7.
- Precision: fp16 forward, fp32 calibration parameter updates.
- Seed control: `torch.manual_seed(seed)`, `numpy.random.seed(seed)`, `random.seed(seed)`, `torch.backends.cudnn.deterministic = True`.

---

## 7. Statistical Tests

| Hypothesis | Test | Multiple-comparison correction |
|---|---|---|
| H1.RQ1: fine-grained ΔECE > coarse ΔECE | dataset-level paired Wilcoxon signed-rank + group Mann–Whitney U | Holm–Bonferroni across backbones |
| H1.RQ2 (correlation): Spearman ρ between τ_txt and ECE | Spearman ρ with permutation p-value (10k perms), partial Spearman conditioning on accuracy & #classes | Holm across (dataset-level, class-level) |
| H1.RQ2 (intervention): paired ΔECE pre/post whitening | paired bootstrap CI (10k resamples), one-sided sign test | per-dataset, Holm correction |
| H1.RQ3: union ECE relative reduction ≥30% | paired bootstrap CI on (ECE_C1 − ECE_Ck) / ECE_C1; Friedman + Nemenyi for rank aggregation | Nemenyi at α=0.05 |
| H1.RQ4: Spearman ECE↔AURC ≤ −0.5 | Spearman ρ on calibrator-level points within each dataset; mixed-effects regression with dataset random intercept | Holm across (AURC, set size) |

Effect size 보고: paired Cohen's d 또는 rank-biserial correlation.

---

## 8. Execution Environment

- **Hardware**: NVIDIA GPU (≥16GB VRAM 권장, ViT-L/14 추론). 본 sanity-check 환경: Windows 11, CUDA 11.8.
- **Software** (pinned in `code/requirements.txt`):
  - Python 3.12
  - PyTorch 2.7.1+cu118
  - open_clip_torch 3.2.0
  - torchvision 0.22.1+cu118
  - scikit-learn 1.5.1, scipy 1.13.1, numpy 1.26.4
  - tqdm, pyyaml, pandas
- **Reproducibility logging**: 매 실행 시 git hash, env hash, CLI args, seed, dataset checksum, output path를 `logs/<run_id>.json`으로 기록.

---

## 9. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Fine-tuning checkpoint 부재 | zero-shot CLIP fallback, 결과에 명시 |
| Test 데이터 다운로드 제약 | torchvision 내장 dataset (CIFAR-10/100, Flowers, Food, FGVC) 우선 사용 |
| GPU 메모리 부족 (ViT-L/14) | gradient checkpointing, batch_size 줄임, fp16 |
| 다중 비교에 의한 false positive | Holm–Bonferroni 보정 명시 |
| Calibration set의 hyperparameter overfit | base val NLL 기준 선택 + nested CV |
| Conformal coverage 위반 | 베이스라인 검증: zero-shot CLIP에서 empirical coverage 보고 |

---

## 10. Execution Order

1. `data.py` 자체 테스트 (toy 데이터 한 배치 로딩).
2. `models.py` zero-shot CLIP 한 데이터셋 평가 (sanity acc).
3. `calibrators.py` 단위 테스트 (synthetic logits).
4. `metrics.py` ECE/AURC 단위 테스트.
5. `experiments/exp1_diagnosis.py` (RQ1) — fine-grained 1개 + coarse 1개로 dry-run.
6. `experiments/exp2_geometry.py` (RQ2) — τ_txt 회귀 + whitening.
7. `experiments/exp3_calibrators.py` (RQ3) — 7 calibrators.
8. `experiments/exp4_downstream.py` (RQ4) — RQ3 결과 로드 후 AURC/conformal.
9. `run_all.py` — 모두 실행.
