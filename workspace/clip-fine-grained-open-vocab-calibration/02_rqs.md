# Research Questions

## Overview
본 연구는 **fine-grained × open-vocabulary** 라는 결합 조건 하에서 CLIP 계열 모델의 confidence calibration을 정량적으로 진단·완화하는 것을 목표로 한다. 이를 위해 RQ는 (i) 문제의 존재·규모 진단(RQ1: base→novel 캘리브레이션 격차의 fine-grained 특이성), (ii) 원인 분리(RQ2: inter-class similarity 및 텍스트 임베딩 거리 분포가 mis-calibration에 미치는 인과적 기여), (iii) 해법 비교(RQ3: 함수 클래스별 calibrator의 base+novel 합산 ECE 최소화 성능), (iv) downstream 전이(RQ4: selective prediction·conformal prediction 지표로의 단조 전이성)의 네 축으로 구성된다. RQ1–2는 기술적·설명적, RQ3는 비교적, RQ4는 인과·예측적 성격이며, 누적적으로 문제 정의의 Gap을 커버한다. 모든 기준선과 데이터셋은 §Baselines·§Datasets에 명시된 동일한 평가 프로토콜(CoOp base-to-novel split, seed 3개 평균)을 공유한다.

## RQ1: fine-grained 도메인에서 fine-tuned CLIP의 base→novel 캘리브레이션 격차(ΔECE)는 coarse 도메인 대비 통계적으로 더 큰가?
- **Type**: Comparative / Descriptive
- **Hypothesis (H1)**: fine-grained 벤치마크(CUB-200, Stanford Cars, FGVC-Aircraft, Flowers102)에서 fine-tuned CLIP의 ΔECE := ECE_novel − ECE_base 는 coarse 벤치마크(ImageNet, Caltech101, SUN397) 대비 평균적으로 더 크다(상대적으로 ≥1.5×, paired test p<0.05).
- **Null hypothesis (H0)**: 두 도메인 간 ΔECE 분포는 통계적으로 구분되지 않는다.
- **Variables**
  - Independent: 도메인 유형(fine-grained vs. coarse), backbone(ViT-B/16, ViT-L/14), fine-tuning 방법(CoOp, CoCoOp, MaPLe, LoRA-adapter).
  - Dependent: ΔECE, Adaptive ECE, MCE, Brier score, NLL — base/novel/union 각각.
  - Confounders to control: 클래스 수(클래스 수 매칭 sub-sampling으로 통제), per-class 표본 수(class-balanced 평가셋 sampling), pre-training corpus 노출도(LAION/WIT 메타데이터 기반 클래스명 frequency proxy 보고), prompt 템플릿(고정 "a photo of a {C}" 및 7-template ensemble 양쪽 보고).
- **Measurement**: ECE(15-bin equal-mass + Adaptive ECE), MCE, Brier, NLL을 base / novel / union 분할에서 산출. ΔECE의 dataset-level paired Wilcoxon signed-rank test. bootstrap 95% CI 보고.
- **Proposed Method (candidate)**: 동일 backbone·동일 fine-tuning recipe로 11개 데이터셋 sweep, fine-grained vs. coarse 그룹 평균 비교.
- **Success threshold**: fine-grained 그룹의 ΔECE 평균이 coarse 그룹의 1.5배 이상이고 paired test p<0.05.
- **Falsification condition**: ΔECE 차이가 통계적으로 유의하지 않거나, fine-grained ΔECE < coarse ΔECE인 경우.

## RQ2: fine-grained inter-class textual similarity가 mis-calibration의 인과적 동인인가?
- **Type**: Explanatory / Causal (controlled intervention)
- **Hypothesis (H1)**: 클래스 텍스트 임베딩 간 평균 cosine similarity(τ_txt)가 증가할수록 union ECE가 단조 증가한다(Spearman ρ ≥ 0.5, p<0.05). 또한 텍스트 임베딩을 PCA whitening 등으로 spread 시키는 개입은 ECE를 유의하게 감소시킨다(절대 ΔECE ≥ 1.0 pp).
- **Null hypothesis (H0)**: τ_txt와 ECE 간 상관이 0과 구분되지 않으며, whitening 개입은 ECE에 유의한 변화를 주지 않는다.
- **Variables**
  - Independent: τ_txt(데이터셋 단위, 클래스 단위), 텍스트 임베딩 spread 개입(없음 / PCA whitening / orthogonalization).
  - Dependent: ECE_union, Brier, predictive entropy 분포(평균, KL to uniform).
  - Confounders to control: 정확도(accuracy를 covariate로 회귀에 포함; partial correlation 보고), 클래스 수, image encoder 동결 여부, LLM-paraphrased description 사용 시 의미 변화는 sentence-BERT similarity로 모니터.
- **Measurement**: (1) 데이터셋·클래스 단위 τ_txt와 ECE의 Spearman/Pearson 상관. (2) 개입 전후 paired ECE 차이의 부호 검정. (3) 보조: logit margin·entropy 분포 변화의 Wasserstein 거리.
- **Proposed Method (candidate)**: τ_txt 측정 후 회귀 분석; 동일 모델에 대해 텍스트 임베딩만 whitening/orthogonalization을 적용한 ablation; 합성적으로 클래스명 perturbation을 통해 τ_txt를 단계적으로 변화.
- **Success threshold**: |Spearman ρ| ≥ 0.5 with p<0.05, 그리고 whitening 개입의 absolute ECE 감소가 ≥1.0 pp(95% CI 하한 > 0).
- **Falsification condition**: 상관이 약하거나(|ρ|<0.3) 부호가 일치하지 않고, 개입이 ECE를 유의하게 감소시키지 못함.

## RQ3: base와 novel을 동시 최적화하는 calibrator 함수 클래스는 무엇인가?
- **Type**: Comparative
- **Hypothesis (H1)**: distance-aware 또는 contrast-aware reweighting을 포함한 conditional/vector scaling 계열 calibrator가 단일 스칼라 temperature scaling 대비 union ECE를 상대적으로 ≥30% 감소시키며, 동시에 novel ECE / base ECE 비를 1.5 이하로 유지한다.
- **Null hypothesis (H0)**: 모든 calibrator 함수 클래스 간 union ECE 차이가 통계적으로 유의하지 않거나, 개선이 30% 미만이다.
- **Variables**
  - Independent: calibrator 함수 클래스 — {(a) raw zero-shot, (b) Temperature Scaling, (c) Vector/Matrix Scaling, (d) Dirichlet calibration, (e) Distance-Aware Calibration / DAC (arXiv:2402.04655), (f) Contrast-Aware Calibration / CAC (arXiv:2501.19060), (g) 본 연구가 제안할 candidate}; calibration set 구성(base only / base+small novel proxy).
  - Dependent: ECE_base, ECE_novel, ECE_union, Brier, NLL, accuracy(보존 확인).
  - Confounders to control: calibration set 크기(고정 N=512/class), 동일 backbone·동일 fine-tuning checkpoint 사용, hyperparameter는 base validation에서만 선택(novel 누설 금지), 평가 seed 3회 평균.
- **Measurement**: 각 calibrator에 대해 base/novel/union ECE 및 Brier·NLL 산출. 베이스라인(b) 대비 union ECE 상대 감소율. novel/base ECE 비율. paired bootstrap test로 calibrator 간 비교.
- **Proposed Method (candidate)**: 동일 평가 프로토콜로 7개 calibrator를 11개 데이터셋·2개 backbone에 적용; Pareto frontier(base ECE × novel ECE) 시각화; rank-aggregation으로 최종 우열 결정.
- **Success threshold**: 최선의 calibrator가 단일 temperature scaling 대비 union ECE 상대 ≥30% 감소(p<0.05, paired) **AND** novel/base ECE 비율 ≤1.5 **AND** accuracy 감소 ≤0.5 pp.
- **Falsification condition**: 어떤 calibrator도 위 임계 3개를 동시에 만족하지 못함.

## RQ4: calibration 개선은 selective prediction과 conformal prediction의 downstream 지표로 단조 전이되는가?
- **Type**: Predictive / Causal
- **Hypothesis (H1)**: RQ3에서 union ECE가 낮아질수록 (i) selective prediction AURC가 단조 감소하고(Spearman ρ ≤ −0.5), (ii) split conformal prediction의 average set size가 동일 coverage(1−α=0.9)에서 단조 감소하며, coverage 위반율은 |empirical − nominal| ≤ 1.0 pp 이내로 유지된다.
- **Null hypothesis (H0)**: ECE 감소와 AURC·conformal set size 간 단조 관계가 없거나(|ρ|<0.3), coverage 보장이 체계적으로 위반된다.
- **Variables**
  - Independent: calibrator(RQ3의 7종), confidence score 정의(max softmax probability, entropy, MSP after calibration).
  - Dependent: AURC, selective accuracy at coverage∈{0.7,0.8,0.9}, conformal prediction set size(APS / RAPS / Thr), empirical coverage gap.
  - Confounders to control: 동일 accuracy 수준(calibrator가 accuracy를 변화시킬 경우 stratified 비교), score function fix, conformal calibration split은 base만 사용(novel은 test only), α는 고정.
- **Measurement**: AURC 계산, coverage-risk 곡선, conformal set 평균 크기 및 coverage. ECE와의 cross-method Spearman 상관.
- **Proposed Method (candidate)**: RQ3 결과의 calibrator별 ECE를 x축, downstream 지표를 y축으로 배치한 후 mixed-effects regression(데이터셋=random intercept)으로 단조성 검정.
- **Success threshold**: ECE↔AURC Spearman ρ ≤ −0.5(p<0.05), conformal set size 단조 감소 검증되며 coverage gap ≤1 pp.
- **Falsification condition**: ECE 개선에도 불구하고 AURC 또는 set size가 개선되지 않거나 악화, 혹은 coverage 위반이 1 pp를 초과.

## Baselines (공통)
- **B0**: Zero-shot CLIP (ViT-B/16, ViT-L/14) without calibration.
- **B1**: Zero-shot CLIP + global Temperature Scaling (LeVine et al., arXiv:2303.12748).
- **B2**: CoOp / CoCoOp / MaPLe fine-tuned CLIP + Temperature Scaling.
- **B3**: Distance-Aware Calibration (DAC), Wang et al., ICML 2024 (arXiv:2402.04655).
- **B4**: Contrast-Aware Calibration (CAC), arXiv:2501.19060.
- **B5**: Dirichlet calibration / Vector scaling (parametric upper-bound 비교).
- **B6 (보조)**: Prompt ensemble + temperature, "Empirical Study" (arXiv:2402.07417) 의 best recipe.

## Datasets (공통)
- **Fine-grained**: CUB-200, Stanford Cars, FGVC-Aircraft, Flowers102, Oxford Pets, Food-101, iNaturalist subset.
- **Coarse (대조군, RQ1)**: ImageNet-1k, Caltech101, SUN397, EuroSAT.
- Base-to-novel split은 CoOp 관행에 따름(클래스 절반 base, 절반 novel; 3 seeds).

## RQ Dependencies
- **RQ1 → RQ2**: RQ1이 fine-grained 특이적 격차의 존재를 입증해야 RQ2의 원인 탐색이 의미를 가진다(존재 부정 시 RQ2는 null 결과 보고로 축소).
- **RQ1, RQ2 → RQ3**: 원인(τ_txt, 거리 분포)을 알아야 RQ3의 conditional calibrator 설계 동기가 정당화되지만, RQ3는 RQ2의 결과와 독립적으로도 비교 실험으로 실행 가능.
- **RQ3 → RQ4**: RQ4는 RQ3가 산출한 calibrator 집합의 ECE 스펙트럼을 입력으로 사용. RQ3 없이는 비교 단조성 검정 불가.
- RQ2와 RQ4는 서로 독립적으로 답할 수 있다.

## Out of Scope
- **다중 라벨 fine-grained calibration**: 라벨 결합 공간이 RQ 구조를 변경 — 별도 연구로 분리.
- **생성형 VLM(BLIP-2, LLaVA)의 token-level uncertainty**: contrast-based score가 부재하므로 calibrator 함수 클래스 비교의 동등 조건이 성립하지 않음.
- **Adversarial / corruption robustness 하의 calibration**: distribution shift의 또 다른 축으로 RQ1의 base→novel 축과 교란됨; 후속 연구.
- **Active learning 루프 안에서의 calibration 동학(動學)**: 본 연구는 정적 평가에 한정하며, 루프 효과는 향후 과제.
- **Conformal prediction 알고리즘 자체의 비교(APS vs. RAPS vs. SAPS)**: RQ4에서는 한 알고리즘을 고정해 단조성만 본다.
