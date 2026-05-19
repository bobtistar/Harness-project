# Literature Review

## Search Strategy
- **검색 쿼리들** (실제 실행):
  1. `CLIP "Learning Transferable Visual Models From Natural Language Supervision" Radford 2021 arxiv`
  2. `ALIGN "Scaling Up Visual and Vision-Language Representation Learning" Jia 2021`
  3. `BioCLIP "vision foundation model for the tree of life" Stevens 2024 CVPR`
  4. `FG-CLIP fine-grained visual textual alignment 2025 arxiv 2505.05071`
  5. `CoOp / CoCoOp / CLIP-Adapter / MaPLe — base-to-novel prompt learning`
  6. `ProGrad KgCoOp PromptSRC base-to-novel CLIP prompt learning generalization gap`
  7. `Open-Vocabulary Calibration for Fine-tuned CLIP Wang ICML 2024 arxiv 2402.04655 DAC`
  8. `Contrast-Aware Calibration fine-tuned CLIP 2501.19060`
  9. `Enabling Calibration In The Zero-Shot Inference LeVine 2303.12748`
  10. `An Empirical Study Into What Matters for Calibrating Vision-Language Models 2402.07417`
  11. `Understanding and Mitigating Miscalibration in Prompt Tuning CLIP 2410.02681`
  12. `Towards Calibrated Robust Fine-Tuning of Vision-Language Models NeurIPS 2024`
  13. `Guo 2017 On Calibration of Modern Neural Networks temperature scaling`
  14. `Calibrating Deep Neural Networks using Focal Loss Mukhoti NeurIPS 2020`
  15. `Dirichlet calibration Kull NeurIPS 2019`
  16. `Label smoothing Mueller Hinton 2019`
  17. `Revisiting the Calibration of Modern Neural Networks Minderer NeurIPS 2021`
  18. `Measuring Calibration in Deep Learning Nixon Adaptive ECE`
  19. `Conformal Prediction for Zero-Shot Models 2505.24693`
  20. `Conformal Prediction Under Covariate Shift Tibshirani 2019 weighted`
  21. `Selective Classification for Deep Neural Networks Geifman El-Yaniv NeurIPS 2017 AURC`
  22. `Prompt ensemble CLIP zero-shot calibration multiple templates`
  23. `FineCLIP self-distilled region-based NeurIPS 2024`
  24. `Visual classification by description Menon Vondrick ICLR 2023`
  25. `GLIP Grounded Language-Image Pre-training CVPR 2022`
- **검색 도구**: WebSearch (25+ 쿼리), 핵심 후보는 별도 확인.
- **포함 기준**: 2017–2026, 영어, 주류 venue(CVPR, ICCV, ECCV, NeurIPS, ICML, ICLR, IJCV) 또는 arXiv preprint(인용·코드 공개 우선), CLIP·VLM·calibration·conformal·selective prediction 도메인.
- **제외 기준**: 도메인 외 calibration(주로 NLP-only, tabular), 코드/평가 재현 불가한 단일 abstract-only 자료, generative VLM(BLIP-2, LLaVA) token-level uncertainty(scope out).
- **검색 일자**: 2026-05-20

## Categorized Findings

### Category A: CLIP / Vision-Language Pretraining (기초)
| Paper | Year | Approach | Key Result | Relevance |
|-------|------|----------|------------|-----------|
| Radford et al. — Learning Transferable Visual Models From Natural Language Supervision (CLIP) [arXiv:2103.00020] | 2021 | 400M image-text pair contrastive pretraining, dual encoder | Strong zero-shot transfer across 30+ benchmarks; introduces "a photo of a {class}" 프로토콜 | 모든 RQ의 backbone |
| Jia et al. — ALIGN [arXiv:2102.05918] | 2021 | 1.8B noisy alt-text 쌍, dual encoder, scale-over-noise | SoTA retrieval; CLIP과 함께 dual-encoder VLM 패러다임 정착 | 비교 backbone (보조) |
| Ilharco, Wightman, et al. — OpenCLIP / Reproducible scaling laws [arXiv:2212.07143] | 2022–2023 | Public LAION-400M/2B 기반 재현 | ViT-B/32 62.96% top-1 ImageNet zero-shot; ViT-G/14 80%+ | RQ1·RQ3 backbone 선택지 |
| Stevens et al. — BioCLIP [arXiv:2311.18803, CVPR 2024 Oral, Best Student Paper] | 2024 | TreeOfLife-10M로 fine-grained 종 분류 도메인 적응 | 기존 CLIP 대비 fine-grained biology 분류 +16–17% absolute | fine-grained 도메인 baseline (CUB/iNat) |

### Category B: Fine-grained Recognition with VLMs
| Paper | Year | Approach | Key Result | Relevance |
|-------|------|----------|------------|-----------|
| Menon & Vondrick — Visual Classification via Description from LLMs [ICLR 2023] | 2023 | GPT-3로 클래스별 attribute description 생성 → CLIP score 평균 | Hand-crafted prompt 대비 fine-grained 일관 개선, 해석성 향상 | RQ2 — prompt diversity 조작용 baseline |
| Xie et al. — FG-CLIP [arXiv:2505.05071, ICML 2025] | 2025 | 1.6B long-caption 쌍 + 12M region-bbox 정렬 + 10M hard negative | Fine-grained 분류/grounding에서 CLIP·SigLIP 대비 일관 향상 | RQ1 baseline (fine-grained 강화 backbone) |
| Tian et al. — FineCLIP [NeurIPS 2024] | 2024 | 실시간 self-distillation + region-text contrastive | Dense prediction·fine-grained 인식 개선 | RQ1 보조 baseline |
| Li et al. — GLIP [arXiv:2112.03857, CVPR 2022 Best Paper Finalist] | 2022 | Detection + phrase grounding 통합 사전학습 | Zero-shot LVIS 26.9 AP, COCO 49.8 AP | adjacency — grounding 기반 fine-grained, scope out (classification 위주) |

### Category C: Open-Vocabulary Classification & Prompt Engineering
| Paper | Year | Approach | Key Result | Relevance |
|-------|------|----------|------------|-----------|
| Zhou et al. — CoOp [arXiv:2109.01134, IJCV 2022] | 2022 | Learnable context tokens(continuous prompt) | Hand-crafted prompt 대비 16-shot 평균 +15% | RQ1·RQ3 baseline (fine-tuning recipe) |
| Zhou et al. — CoCoOp [arXiv:2203.05557, CVPR 2022] | 2022 | Image-conditioned dynamic prompt | CoOp의 base→novel 일반화 결손 완화, novel 정확도 회복 | base-to-novel split 프로토콜의 원조 |
| Gao et al. — CLIP-Adapter [arXiv:2110.04544] | 2021 | Residual feature adapter | Frozen encoder + 경량 adapter로 few-shot 적응 | RQ3 adapter baseline |
| Khattak et al. — MaPLe [arXiv:2210.03117, CVPR 2023] | 2023 | Vision+language branch 동시 multi-modal prompt | CoCoOp 대비 novel +3.45%, HM +2.72% | RQ1 baseline (multi-modal prompt) |
| Khattak et al. — PromptSRC [arXiv:2307.06948, ICCV 2023] | 2023 | "a photo of" 앵커에 self-regularization | Novel HM +2.46% 평균 | RQ3 보조 baseline |
| Zhu et al. — ProGrad / Yao et al. — KgCoOp | 2023 | Gradient projection / hand-crafted prompt regularization | Forgetting 완화, novel 일반화 향상 | RQ1·RQ3 baseline 후보 |
| Radford et al. (CLIP §3) + OpenAI prompts.md | 2021 | 7~80 template prompt ensemble (logit/feature 평균) | ImageNet zero-shot +3.5–5% | RQ2 — prompt ensemble 위치 ablation |

### Category D: Calibration of Deep Networks (General)
| Paper | Year | Approach | Key Result | Relevance |
|-------|------|----------|------------|-----------|
| Guo, Pleiss, Sun, Weinberger — On Calibration of Modern Neural Networks [arXiv:1706.04599, ICML 2017] | 2017 | Temperature scaling(단일 스칼라 T) | Modern DNN miscalibration 보고; T로 ECE 대폭 감소, accuracy 불변 | RQ3 B1 baseline (단일 T) |
| Kull et al. — Beyond Temperature Scaling: Dirichlet Calibration [arXiv:1910.12656, NeurIPS 2019] | 2019 | Dirichlet 분포 기반 multi-class calibration = log-prob에 linear layer | confidence-ECE·class-wise ECE·Brier·log-loss 일관 개선 | RQ3 B5 (parametric upper bound) |
| Müller, Kornblith, Hinton — When Does Label Smoothing Help? [arXiv:1906.02629, NeurIPS 2019] | 2019 | Soft target | Calibration 개선, but distillation에 해로움 | RQ3 학습-기반 비교군 |
| Mukhoti et al. — Calibrating DNNs using Focal Loss [NeurIPS 2020] | 2020 | Focal loss → 자연스러운 well-calibrated 모델 | + temperature scaling 결합 시 SoTA | RQ3 학습-기반 비교군 |
| Nixon et al. — Measuring Calibration in Deep Learning [arXiv:1904.01685, CVPR 2019 Workshop] | 2019 | Adaptive ECE / ACE / TACE | ECE binning artefact 폭로, adaptive binning 권고 | RQ1·RQ3·RQ4 metric 정합성 |
| Minderer et al. — Revisiting the Calibration of Modern Neural Networks [arXiv:2106.07998, NeurIPS 2021] | 2021 | ViT, MLP-Mixer 포함 재평가 | Modern ViT는 잘 calibrated; distribution shift 하 size↑→calibration↑ | RQ1 — ViT-B/L backbone 차이 해석 |
| Geifman & El-Yaniv — Selective Classification for Deep Neural Networks [arXiv:1705.08500, NeurIPS 2017] | 2017 | Softmax response threshold + risk-coverage 곡선 | 99.9% 확률로 risk ≤ ε guarantee | RQ4 selective prediction baseline |

### Category E: Calibration of VLMs / CLIP
| Paper | Year | Approach | Key Result | Relevance |
|-------|------|----------|------------|-----------|
| LeVine et al. — Enabling Calibration In The Zero-Shot Inference of Large VLMs [arXiv:2303.12748] | 2023 | Modified zero-shot용 temperature scaling | 단일 학습 T가 inference dataset·prompt에 걸쳐 일반화됨을 입증; zero-shot CLIP miscalibration 체계 보고 | RQ3 B1 baseline (zero-shot+T) |
| Wang, Wang, Wang, Zhang, Zhou, Wei — Open-Vocabulary Calibration for Fine-tuned CLIP (DAC) [arXiv:2402.04655, ICML 2024] | 2024 | Predicted label과 base class 간 textual distance를 통해 temperature를 sample-wise 스케일 | 7 prompt learning 방법 × 11 dataset에서 일관 ECE 감소; novel에서의 over-confidence 완화 | RQ3 **B3 핵심 baseline** |
| Tu, Deng, Campbell, Gould, Gedeon — An Empirical Study Into What Matters for Calibrating VLMs [arXiv:2402.07417, ICML 2024] | 2024 | 아키텍처·데이터·학습전략 sweep | VLM은 본질적으로 uncalibrated이나 temperature scaling이 distribution/label shift에도 일관 효과; 매우 적은 샘플로 calibration 가능 | RQ3 B6 best recipe |
| Lv, Chen, Zhou, Li, Guo — Contrast-Aware Calibration (CAC) [arXiv:2501.19060] | 2025 | Original CLIP과 fine-tuned CLIP의 contrastive 차이로 calibration weight 산출 | 11 dataset × 5 fine-tuning에서 train 클래스와 unseen 클래스 동시 calibration | RQ3 **B4 핵심 baseline** — 우리 (C) gap의 가장 가까운 비교군 |
| Wang, Li, Wei — Understanding and Mitigating Miscalibration in Prompt Tuning (DOR) [arXiv:2410.02681] | 2024 | Dynamic Outlier Regularization: 대규모 vocabulary에서 sampled novel label의 feature deviation 최소화 | CoOp = novel over-confidence, KgCoOp = base under-confidence 양상 진단 후 양쪽 동시 보정 | RQ2·RQ3 핵심 — **base/novel trade-off의 명시적 진단** |
| Oh et al. — Towards Calibrated Robust Fine-Tuning of VLMs [arXiv:2311.01723, NeurIPS 2024] | 2024 | Constrained multimodal contrastive loss + EMA self-distillation; ID covariance smallest singular value 제약 | OOD accuracy와 calibration 동시 개선 | RQ1 — fine-tuning recipe로서 비교군 |
| Wang, Zhou, Wei — Confidence Calibration in Contrastive VLMs (Springer chapter, 2025) | 2025 | DAC 계열 종합 정리 | 본 분야 survey | 본 연구의 context 정리용 |

### Category F: Selective Prediction & Conformal Prediction under (Distribution) Shift
| Paper | Year | Approach | Key Result | Relevance |
|-------|------|----------|------------|-----------|
| Tibshirani, Barber, Candès, Ramdas — Conformal Prediction Under Covariate Shift [arXiv:1904.06019, NeurIPS 2019] | 2019 | Weighted conformal: likelihood ratio reweighting | Covariate shift에서도 marginal coverage 유지 | RQ4 — base→novel shift 처리의 이론적 토대 |
| Silva-Rodríguez, Ben Ayed, Dolz — Conformal Prediction for Zero-Shot Models (Conf-OT) [arXiv:2505.24693, CVPR 2025] | 2025 | Split conformal + transductive transfer between calibration and query | 15 dataset × 3 non-conformity score에서 coverage 유지, set size 효율 | RQ4 **핵심 비교군** |
| Geifman & El-Yaniv — Selective Classification for DNNs [arXiv:1705.08500] | 2017 | MSP threshold | risk-coverage 곡선의 기초 | RQ4 baseline |
| Traub et al. — Overcoming Common Flaws in the Evaluation of Selective Classification [NeurIPS 2024] | 2024 | AURC·E-AURC 평가 표준화 | 실험 protocol 권고 | RQ4 평가 방법론 |
| Andrade-Loarca et al. — A Novel Characterization of Population AURC [arXiv:2410.15361] | 2024 | AURC의 통계적 성질, finite-sample estimator | 평가 metric 통계학적 기반 | RQ4 metric 정합성 |

## Per-RQ Coverage

### RQ1 — fine-grained vs. coarse에서 fine-tuned CLIP의 ΔECE
- **가장 가까운 선행연구**: [Wang24 / arXiv:2402.04655] "Open-Vocabulary Calibration for Fine-tuned CLIP"은 11 dataset(혼합 coarse + fine-grained) × 7 prompt learning에서 base/novel ECE를 측정하고 novel에서의 systematic over-confidence를 보고했다. [Wang24b / arXiv:2410.02681]은 CoOp vs. KgCoOp 사이의 base/novel calibration trade-off의 **부호**가 다르다는 것을 진단했다.
- **gap**: 두 연구 모두 **fine-grained 도메인 특이성**(coarse 대비 ΔECE 크기 차이)을 통계적으로 검증한 별도 가설 테스트를 제시하지 않는다. dataset-level paired test, fine-grained vs. coarse 두 그룹 분리, class-count·sample-count·prompt-template 통제, ViT-B vs. ViT-L 분리가 모두 결손.
- **우리 기여 Z**: 11 dataset을 fine-grained(7) vs. coarse(4) 그룹으로 명시 분리, Wilcoxon paired + bootstrap CI로 ΔECE 차이의 통계적 유의성을 정량화. confounder(클래스 수 매칭, 노출도 proxy) 통제 보고.

### RQ2 — fine-grained inter-class textual similarity의 인과적 기여
- **가장 가까운 선행연구**: [Lv25 / arXiv:2501.19060] CAC는 "poor intra-class and inter-class discriminative ability on unseen classes is the root cause"라고 **경험적으로 주장**하고, 텍스트 정렬 기반 보정을 제안. [Wang24 / DAC]도 base 클래스로부터의 textual distance를 핵심 신호로 사용.
- **gap**: 두 연구 모두 (i) τ_txt(클래스 임베딩 평균 코사인 유사도) 같은 명시적 dataset-level scalar과 ECE의 단조 관계를 데이터셋 횡단으로 회귀하지 않음, (ii) 정확도·클래스 수를 partial-out하지 않음, (iii) **개입적 실험**(PCA whitening, orthogonalization)으로 인과 화살표를 검정하지 않음. [Menon23]은 description 기반 prompt diversification을 제안했지만 calibration 측면 효과는 보고하지 않음.
- **우리 기여 Z**: 회귀 + 개입 ablation의 2-축 검증. 텍스트 공간 spread 개입 → ECE 단조 감소를 paired 비교로 입증(목표: ≥1 pp 절대 감소). 합성 perturbation으로 τ_txt를 단계 변화시켜 dose-response 확인.

### RQ3 — base+novel 동시 최적화 calibrator 함수 클래스
- **직접 비교 가능한 선행연구**: [Guo17 / TS], [Kull19 / Dirichlet], [Wang24 / DAC], [Lv25 / CAC], [Wang24b / DOR], [LeVine23 / zero-shot TS], [Tu24 / empirical best recipe]. 함수 클래스 spectrum이 거의 모두 존재한다.
- **gap**: (i) 동일 backbone·동일 fine-tuning checkpoint·동일 evaluation set으로 **이들 7종을 한 자리에서 head-to-head로 비교**한 연구는 부재. CAC[Lv25]는 DAC[Wang24]를 비교하지만 Dirichlet·DOR·empirical-study best recipe는 누락. (ii) base+novel 합산 ECE 관점의 Pareto frontier 시각화 + rank-aggregation 결정 절차가 부재. (iii) calibration set이 base only일 때 hyperparameter overfitting risk를 명시적으로 통제한 비교 부재.
- **우리 기여 Z**: 7 calibrator × 11 dataset × 2 backbone × 3 seeds 통합 비교. Pareto frontier(base ECE × novel ECE) + paired bootstrap 검정. accuracy 보존(≤0.5 pp) + novel/base 비(≤1.5) + union ECE 상대 30% 감소의 3-조건 동시 검증.

### RQ4 — calibration → selective prediction / conformal prediction 단조 전이
- **직접 비교 가능한 선행연구**: [Silva-Rodríguez25 / arXiv:2505.24693] Conf-OT는 split conformal + transductive 전이를 통해 zero-shot CLIP의 conformal coverage·set size를 보고한다. [Geifman17] selective classification, [Traub24]·[Andrade-Loarca24]가 AURC 평가 표준 정리.
- **gap**: ECE를 x축으로 두고 (a) AURC와 (b) conformal set size를 y축으로 하는 **단조성 검정**(mixed-effects regression)을 제공한 연구는 부재. 특히 RQ3에서 산출된 calibrator 스펙트럼이 downstream으로 어떻게 전이되는지의 정량적 관계, sharpness 손실로 인한 trade-off, base-only calibration이 novel-only test에서 coverage 위반을 일으키는지 여부가 미규명.
- **우리 기여 Z**: ECE↔AURC, ECE↔conformal set size의 Spearman 단조성을 dataset random-intercept 모델로 검정. coverage gap을 1 pp 이내로 유지하는 calibrator를 식별.

## Identified Gap (재확인)
문헌 조사 후 gap은 (a) 여전히 유효하며, (b) 처음 정의보다 **더 정확하게 좁혀진다**.

- **유효성**: [Wang24, Lv25, Wang24b] 세 편이 본 문제 영역을 직접 공략 중이지만, 그들의 평가 셋은 **coarse(ImageNet/SUN397/Caltech101) + fine-grained 혼합**이며 **fine-grained 특이성**의 정량 분해는 결손. [Menon23, FG-CLIP25, FineCLIP24]은 fine-grained 정확도를 끌어올리지만 calibration 측정 자체를 보고하지 않음. [Silva-Rodríguez25 / Conf-OT]는 conformal coverage를 다루지만 base-to-novel split의 ECE↔conformal 단조성은 검정 외부에 둠.
- **좁혀진 정의**: gap은 더 이상 "fine-grained × open-vocab calibration 문제가 존재하는가"가 아니라, **(i) fine-grained 도메인 특이성의 통계적 정량화(RQ1), (ii) 인과 동인의 개입적 검증(RQ2), (iii) 7 함수 클래스의 통제된 head-to-head 및 Pareto frontier(RQ3), (iv) downstream selective/conformal 지표로의 단조 전이성(RQ4)** 의 네 정밀화 축으로 명확화.
- **재정의 권고 불필요**: gap이 사실상 해결되었다고 판단할 근거는 없음. 다만 RQ3의 우리의 candidate 방법은 기존 DAC/CAC/DOR 대비 **새로운 함수 클래스**보다는 **함수 클래스 비교 + 결합/Pareto 권고**가 핵심 기여여야 함. 새로운 함수 클래스 제안이 필요하다면 추가 RQ로 분리 권고.

## Recommended Baselines for Experiments
- **B0 Zero-shot CLIP (ViT-B/16, ViT-L/14)** — OpenCLIP/OpenAI checkpoint 공개, 즉시 실행 가능. (`mlfoundations/open_clip`)
- **B1 Zero-shot CLIP + global Temperature Scaling** ([LeVine23 / arXiv:2303.12748]) — 단일 T 학습, 구현 단순. 코드 공개(논문 부록).
- **B2 CoOp / CoCoOp / MaPLe + Temperature Scaling** ([Zhou22, Khattak23]) — 코드 공개(`KaiyangZhou/CoOp`, `muzairkhattak/multimodal-prompt-learning`), 표준 base-to-novel split 포함.
- **B3 Distance-Aware Calibration (DAC)** ([Wang24 / arXiv:2402.04655, ICML 2024]) — 코드 공개(`ml-stat-Sustech/CLIP_Calibration`), plug-and-play post-hoc, hyperparameter tuning 불필요로 명시.
- **B4 Contrast-Aware Calibration (CAC)** ([Lv25 / arXiv:2501.19060]) — 11 dataset × 5 fine-tuning 보고, original CLIP과 fine-tuned CLIP 동시 필요(이중 inference).
- **B5 Dirichlet Calibration / Vector Scaling** ([Kull19 / arXiv:1910.12656]) — `dirichletcal.github.io` 코드 공개, parametric upper-bound 비교.
- **B6 Empirical-Study Best Recipe** ([Tu24 / arXiv:2402.07417]) — prompt ensemble + temperature 최적 조합.
- **B7 (보조) DOR** ([Wang24b / arXiv:2410.02681]) — fine-tuning loss 단에서 base/novel 동시 calibration, RQ3에서 학습-기반 비교군.
- **B8 (간단한 ablation) Label Smoothing** ([Müller19 / arXiv:1906.02629]) — 학습-기반 sanity check.
- **B9 (downstream) Conf-OT for Conformal** ([Silva-Rodríguez25 / arXiv:2505.24693, CVPR 2025]) — 코드 공개(`jusiro/CLIP-Conformal`), RQ4 conformal baseline.
- **Selective prediction**: MSP threshold ([Geifman17 / arXiv:1705.08500]) — 기본 baseline. AURC 평가는 [Traub24] protocol 권고.

## Search Coverage Notes
- **검색되지 않은 영역**:
  - fine-grained 도메인(CUB, FGVC-Aircraft 등) 특화 calibration의 **체계적 비교**: 표준 base-to-novel split 위에서 fine-grained-only Pareto frontier를 보고한 연구는 발견되지 않음 → 우리의 RQ1·RQ3 차별성 확보.
  - ECE × AURC × conformal set size의 **3축 동시 단조성 검정**: 미발견 → RQ4 차별성 확보.
  - τ_txt(클래스 텍스트 임베딩 유사도) 같은 단일 scalar과 calibration의 dataset-level 회귀: 미발견(DAC[Wang24]가 sample-level distance만 사용) → RQ2 차별성 확보.
- **접근 불가능한 자료**: Springer chapter "Confidence Calibration in Contrastive Vision-Language Models" [Wang25] 전문은 비공개(metadata만 확인). DAC/CAC 후속 워크숍 변형 일부는 OpenReview 비공개 상태.
- **검색 후 신규 발견**: [Wang24b / DOR, arXiv:2410.02681]는 base/novel calibration trade-off의 **부호 비대칭**(CoOp = novel over-confidence vs. KgCoOp = base under-confidence)을 명시 진단 — 우리 RQ2·RQ3에 강한 동기.

## References
[1] Radford, A., Kim, J. W., Hallacy, C., et al. (2021). Learning Transferable Visual Models From Natural Language Supervision. *ICML 2021* (PMLR 139:8748–8763). arXiv:2103.00020. https://arxiv.org/abs/2103.00020
[2] Jia, C., Yang, Y., Xia, Y., et al. (2021). Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision (ALIGN). *ICML 2021* (PMLR 139:4904–4916). arXiv:2102.05918. https://arxiv.org/abs/2102.05918
[3] Cherti, M., Beaumont, R., Wightman, R., Wortsman, M., Ilharco, G., Gordon, C., Schuhmann, C., Schmidt, L., Jitsev, J. (2022/2023). Reproducible scaling laws for contrastive language-image learning (OpenCLIP). arXiv:2212.07143. https://arxiv.org/abs/2212.07143
[4] Stevens, S., Wu, J., Thompson, M. J., et al. (2024). BioCLIP: A Vision Foundation Model for the Tree of Life. *CVPR 2024 (Oral, Best Student Paper)*. arXiv:2311.18803. https://arxiv.org/abs/2311.18803
[5] Menon, S., Vondrick, C. (2023). Visual Classification via Description from Large Language Models. *ICLR 2023*. https://openreview.net/forum?id=jlAjNL8z5cs
[6] Xie, C., et al. (2025). FG-CLIP: Fine-Grained Visual and Textual Alignment. *ICML 2025*. arXiv:2505.05071. https://arxiv.org/abs/2505.05071
[7] Tian, D., et al. (2024). FineCLIP: Self-distilled Region-based CLIP for Better Fine-grained Understanding. *NeurIPS 2024*. https://proceedings.neurips.cc/paper_files/paper/2024/hash/3122aaa22b2fe83f9cead1a696f65ceb-Abstract-Conference.html
[8] Li, L. H., Zhang, P., Zhang, H., et al. (2022). Grounded Language-Image Pre-training (GLIP). *CVPR 2022 (Best Paper Finalist)*. arXiv:2112.03857. https://arxiv.org/abs/2112.03857
[9] Zhou, K., Yang, J., Loy, C. C., Liu, Z. (2022). Learning to Prompt for Vision-Language Models (CoOp). *IJCV 2022*. arXiv:2109.01134. https://arxiv.org/abs/2109.01134
[10] Zhou, K., Yang, J., Loy, C. C., Liu, Z. (2022). Conditional Prompt Learning for Vision-Language Models (CoCoOp). *CVPR 2022*. arXiv:2203.05557. https://arxiv.org/abs/2203.05557
[11] Gao, P., Geng, S., Zhang, R., et al. (2021/2024). CLIP-Adapter: Better Vision-Language Models with Feature Adapters. *IJCV 2024*. arXiv:2110.04544. https://arxiv.org/abs/2110.04544
[12] Khattak, M. U., Rasheed, H., Maaz, M., Khan, S., Khan, F. S. (2023). MaPLe: Multi-modal Prompt Learning. *CVPR 2023*. arXiv:2210.03117. https://arxiv.org/abs/2210.03117
[13] Khattak, M. U., et al. (2023). Self-regulating Prompts: Foundational Model Adaptation without Forgetting (PromptSRC). *ICCV 2023*. arXiv:2307.06948. https://arxiv.org/abs/2307.06948
[14] Guo, C., Pleiss, G., Sun, Y., Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. *ICML 2017*. arXiv:1706.04599. https://arxiv.org/abs/1706.04599
[15] Kull, M., Perello-Nieto, M., Kängsepp, M., Silva Filho, T., Song, H., Flach, P. (2019). Beyond temperature scaling: Obtaining well-calibrated multiclass probabilities with Dirichlet calibration. *NeurIPS 2019*. arXiv:1910.12656. https://arxiv.org/abs/1910.12656
[16] Müller, R., Kornblith, S., Hinton, G. (2019). When Does Label Smoothing Help? *NeurIPS 2019*. arXiv:1906.02629. https://arxiv.org/abs/1906.02629
[17] Mukhoti, J., Kulharia, V., Sanyal, A., Golodetz, S., Torr, P., Dokania, P. (2020). Calibrating Deep Neural Networks using Focal Loss. *NeurIPS 2020*. http://torrvision.com/focal_calibration/
[18] Nixon, J., Dusenberry, M. W., Zhang, L., Jerfel, G., Tran, D. (2019). Measuring Calibration in Deep Learning. *CVPR Workshops 2019*. arXiv:1904.01685. https://arxiv.org/abs/1904.01685
[19] Minderer, M., Djolonga, J., et al. (2021). Revisiting the Calibration of Modern Neural Networks. *NeurIPS 2021*. arXiv:2106.07998. https://arxiv.org/abs/2106.07998
[20] Geifman, Y., El-Yaniv, R. (2017). Selective Classification for Deep Neural Networks. *NeurIPS 2017*. arXiv:1705.08500. https://arxiv.org/abs/1705.08500
[21] LeVine, W., Pikus, B., Raja, P., Amat Gil, F. (2023). Enabling Calibration In The Zero-Shot Inference of Large Vision-Language Models. arXiv:2303.12748. https://arxiv.org/abs/2303.12748
[22] Wang, S., Wang, J., Wang, G., Zhang, B., Zhou, K., Wei, H. (2024). Open-Vocabulary Calibration for Fine-tuned CLIP (DAC). *ICML 2024* (PMLR 235:51734–51754). arXiv:2402.04655. https://arxiv.org/abs/2402.04655 — Code: https://github.com/ml-stat-Sustech/CLIP_Calibration
[23] Tu, W., Deng, W., Campbell, D., Gould, S., Gedeon, T. (2024). An Empirical Study Into What Matters for Calibrating Vision-Language Models. *ICML 2024*. arXiv:2402.07417. https://arxiv.org/abs/2402.07417
[24] Lv, S.-L., Chen, Y.-Y., Zhou, Z., Li, Y.-F., Guo, L.-Z. (2025). Contrast-Aware Calibration for Fine-Tuned CLIP: Leveraging Image-Text Alignment. arXiv:2501.19060. https://arxiv.org/abs/2501.19060
[25] Wang, S., Li, Y., Wei, H. (2024/2025). Understanding and Mitigating Miscalibration in Prompt Tuning for Vision-Language Models (DOR). arXiv:2410.02681. https://arxiv.org/abs/2410.02681
[26] Oh, C., et al. (2024). Towards Calibrated Robust Fine-Tuning of Vision-Language Models. *NeurIPS 2024*. arXiv:2311.01723. https://arxiv.org/abs/2311.01723
[27] Tibshirani, R. J., Barber, R. F., Candès, E. J., Ramdas, A. (2019). Conformal Prediction Under Covariate Shift. *NeurIPS 2019*. arXiv:1904.06019. https://arxiv.org/abs/1904.06019
[28] Silva-Rodríguez, J., Ben Ayed, I., Dolz, J. (2025). Conformal Prediction for Zero-Shot Models (Conf-OT). *CVPR 2025*. arXiv:2505.24693. https://arxiv.org/abs/2505.24693 — Code: https://github.com/jusiro/CLIP-Conformal
[29] Traub, J., et al. (2024). Overcoming Common Flaws in the Evaluation of Selective Classification Systems. *NeurIPS 2024*. arXiv:2407.01032. https://arxiv.org/abs/2407.01032
[30] Andrade-Loarca, H., et al. (2024). A Novel Characterization of the Population Area Under the Risk Coverage Curve (AURC) and Rates of Finite Sample Estimators. arXiv:2410.15361. https://arxiv.org/abs/2410.15361
