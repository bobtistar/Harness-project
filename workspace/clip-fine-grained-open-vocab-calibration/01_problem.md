# Problem Statement

## 키워드
CLIP, fine-grained, open-vocabulary, calibration

## 키워드 해석
네 키워드는 독립 주제가 아니라 하나의 교차 문제를 형성한다. 가능한 해석을 정리하면:
- (A) CLIP을 **fine-grained 분류**에 적용했을 때의 calibration 문제.
- (B) CLIP의 **open-vocabulary(zero-shot 및 base→novel 일반화)** 설정에서의 calibration 문제.
- (C) (A)와 (B)의 **결합** — fine-grained 클래스 공간에서 base/novel을 모두 다루는 open-vocabulary 인식의 confidence calibration.

채택: **(C)**. 호출자 맥락이 명시적으로 결합 문제를 지시하며, (A)·(B) 각각은 이미 별도 연구 라인이 존재하지만 둘이 만나는 지점은 under-studied이다.

## 도메인 컨텍스트
대조 학습 기반 vision-language model(VLM), 특히 CLIP(Radford et al., 2021) 계열은 자연어 클래스 설명을 텍스트 인코더로 임베딩해 이미지 임베딩과의 코사인 유사도로 분류하는 패러다임을 일반화시켰다. 이 구조는 학습 시 보지 못한 클래스를 텍스트 프롬프트만으로 분류하는 **open-vocabulary recognition**을 가능하게 했고, 이후 CoOp, CoCoOp, MaPLe 등 prompt learning 계열과 LoRA·adapter 계열의 parameter-efficient fine-tuning 방법이 base 클래스 정확도를 끌어올리는 방향으로 확장되어 왔다.

한편 **fine-grained recognition**(예: CUB-200 조류 종, Stanford Cars 차종, FGVC-Aircraft 기체, Food-101, Oxford Pets·Flowers102, iNaturalist 종 단위)은 클래스 간 시각적 차이가 작고 텍스트 설명이 거의 동형(同型, "a photo of a {species}")이라는 특수성을 가진다. 이 도메인에서 CLIP은 coarse 분류(ImageNet 1k급)보다 뚜렷이 낮은 zero-shot 정확도를 보이며, fine-tuning을 적용하면 base 정확도는 회복되지만 novel 클래스 일반화와 confidence calibration이 동시에 악화된다는 보고가 누적되고 있다.

세 번째 축인 **calibration**은 모델의 예측 확률이 실제 정답률과 일치하는 정도를 의미한다. 분류기에서 ECE(Expected Calibration Error)·MCE·Brier score·NLL이 표준 지표로 쓰이며, post-hoc temperature scaling이 일반적 해법이다. 그러나 최근 연구(LeVine et al., 2023; Wang et al., ICML 2024 "Open-Vocabulary Calibration for Fine-tuned CLIP"; "Contrast-Aware Calibration", 2025)는 (i) fine-tuned CLIP이 base 클래스에 과적합하며 pre-training이 가졌던 calibration을 잃는다는 점, (ii) 단일 temperature는 novel 클래스에서 오히려 과신을 악화시킬 수 있다는 점을 일관되게 지적한다. 즉 calibration 연구는 활발하지만, fine-grained × open-vocabulary가 동시에 부과되는 조건은 체계적으로 다뤄지지 않았다.

## 핵심 문제 (Gap)
**fine-grained 클래스 공간에서 open-vocabulary 설정(base와 unseen novel 클래스가 추론 시 공존)으로 운용되는 CLIP 계열 모델은, 단일 스칼라 temperature·standard prompt ensemble·기존 distance-aware 보정으로는 base와 novel 양쪽에서 동시에 낮은 ECE를 달성하지 못한다.** 구체적으로 (1) fine-grained 도메인의 높은 inter-class similarity는 logit 분포를 좁고 평탄하게 만들어 softmax가 거의 균일에 가까워지거나 반대로 sharp한 winner-take-all이 되는 양극화를 유발하며, (2) novel 클래스의 텍스트 임베딩은 base 클래스 임베딩과의 거리 분포가 base 내부 거리 분포와 다르기 때문에 base에서 학습·튜닝된 calibration 함수가 novel에 외삽되지 않는다. 결과적으로 base에서 calibration된 모델이 novel에서는 systematic over-confidence를, 또는 그 역의 패턴을 보이며, 이 trade-off의 정량적 특성·원인·완화 방법이 아직 정립되어 있지 않다.

## 왜 중요한가
- **Selective prediction / abstention**: confidence를 임계값으로 사용해 거부(abstain)하는 운용 시, mis-calibration은 risk-coverage 곡선의 AUROC를 직접 떨어뜨려 자동 검토→인간 검토 라우팅을 망가뜨린다. fine-grained 의료(피부 병변 세부 유형), 생물 다양성 모니터링(종 동정), 산업 품질검사(결함 sub-type) 등 도메인에서 치명적이다.
- **Active learning · pseudo-labeling**: 라벨링 비용이 큰 fine-grained 도메인에서 unlabeled 풀에 대한 신뢰도 기반 샘플링·자기학습은 calibration에 직접 의존한다. 과신은 잘못된 pseudo-label을 강화하며 dataset bias를 누적시킨다.
- **Open-vocabulary 배포의 신뢰성**: 실서비스의 클래스 셋은 시간에 따라 확장된다(신규 상품 SKU, 신종 발견, 신규 결함 모드). novel 클래스에 대한 confidence가 신뢰 불가능하다면 zero-shot 확장이라는 CLIP의 핵심 가치 명제가 무너진다.
- **Downstream conformal prediction · uncertainty quantification**: conformal prediction set의 크기와 coverage 보장은 underlying score의 분포 가정에 민감하다. mis-calibrated CLIP score는 예측 집합을 비효율적으로 부풀리거나 coverage를 위반하게 한다.

## 누가 영향받는가
- VLM 기반 인식 시스템을 배포하는 **응용 ML 엔지니어**(자율주행 신호·표지판 sub-type, 의료 영상 sub-class, e-commerce 상품 attribute 분류).
- fine-grained 인식 벤치마크에서 CLIP 변형을 평가·비교하는 **연구자** — accuracy만으로는 누락되는 신뢰도 축을 측정해야 한다.
- safety-critical 도메인 **규제·감사 담당자** — 모델 confidence가 의사결정 근거로 사용될 때 calibration 보고가 요구된다.
- novel 클래스가 빈번한 **long-tail / open-set 운용 환경**의 최종 사용자(예: 시민 과학 앱 사용자가 보는 "확률 80%로 X 종" 표시).

## 측정 가능한 성공 기준 (Candidates)
- **ECE / Adaptive ECE**: base 클래스, novel 클래스, 합산(union) 각각에 대해 측정. 합리적 임계값으로 합산 ECE를 기존 fine-tuning 베이스라인 대비 30% 이상 감소시키되, novel에서의 ECE가 base 대비 1.5배를 넘지 않을 것.
- **MCE(Maximum Calibration Error)**: 최악 신뢰도 구간 추적. safety 보고용.
- **Brier score / NLL**: scalar proper scoring rule로 sharpness·calibration 동시 평가.
- **Selective prediction AUROC / AURC**: confidence 기반 거부 시 risk-coverage 곡선 면적. base+novel 혼합 평가 셋에서 측정.
- **Distribution-shift calibration gap**: base→novel 이동 시 ECE 증가량 ΔECE. 이 값이 베이스라인 대비 통계적으로 유의하게 작아야 함.
- **OOD 분리도(보조)**: novel을 OOD로 간주한 AUROC — 캘리브레이션이 인식 경계를 흐리지 않음을 확인.

## 범위 (Scope)
- 포함: CLIP 및 OpenCLIP 계열(ViT-B/16, ViT-L/14) backbone; fine-grained 표준 벤치마크(CUB-200, Stanford Cars, FGVC-Aircraft, Oxford Pets·Flowers, Food-101, EuroSAT, iNaturalist subset); base-to-novel split 프로토콜(CoOp/CoCoOp 관행); post-hoc + lightweight 학습 기반 calibration; prompt ensemble과의 상호작용 분석.
- 제외: 생성 모델(diffusion) calibration; VLA(vision-language-action) 행동 정책 calibration; 비-CLIP 계열(BLIP-2, LLaVA 등 generative VLM)에서의 token-level uncertainty — 단, 비교 가능하면 부록에서 다룸; 멀티라벨 fine-grained.

## 가정 (Assumptions)
- 평가용 검증 셋의 base 클래스 라벨은 충분하며, novel 클래스의 ground truth는 평가 시점에만 사용 가능하다(학습 누설 없음).
- 텍스트 프롬프트 템플릿은 클래스 간 동일하거나 LLM 생성 description set으로 통제된다.
- CLIP의 image·text encoder는 동결 또는 prompt/adapter 수준에서만 변경되어 large-scale pre-training의 사전 지식이 보존된다.
- ECE류 지표의 binning artefact를 보완하기 위해 Adaptive ECE 또는 ECE-KDE를 병행 보고한다.

## 열린 질문 (RQ 단계에서 해소)
- fine-grained inter-class similarity가 만드는 logit 분포 특성(엔트로피·margin)과 mis-calibration의 인과 관계를 어떤 측정량으로 분리할 것인가?
- base 클래스와 novel 클래스의 텍스트 임베딩 거리 분포 차이를 단일 distance 통계가 아닌 더 풍부한 representation으로 보정에 활용할 수 있는가?
- temperature scaling, vector scaling, Dirichlet calibration, contrast-aware reweighting 중 base+novel 합산 ECE를 최소화하는 함수 클래스는 무엇인가?
- prompt ensemble의 평균 위치(feature space vs. logit space vs. probability space)가 calibration에 미치는 영향은?
- calibration 개선이 selective prediction·conformal prediction 같은 downstream 의사결정 지표로 monotone하게 전이되는가, 아니면 sharpness 손실로 인해 trade-off가 존재하는가?

## 참고 자료
- LeVine et al., "Enabling Calibration In The Zero-Shot Inference of Large Vision-Language Models", arXiv:2303.12748.
- Wang et al., "Open-Vocabulary Calibration for Fine-tuned CLIP" (Distance-Aware Calibration, DAC), ICML 2024 / arXiv:2402.04655.
- "Contrast-Aware Calibration for Fine-Tuned CLIP: Leveraging Image-Text Alignment", arXiv:2501.19060.
- "An Empirical Study Into What Matters for Calibrating Vision-Language Models", arXiv:2402.07417.
- "FG-CLIP: Fine-Grained Visual and Textual Alignment", arXiv:2505.05071.
- "Conformal Prediction for Zero-Shot Models", arXiv:2505.24693.
- Radford et al., "Learning Transferable Visual Models From Natural Language Supervision" (CLIP), ICML 2021.
- Zhou et al., CoOp / CoCoOp — base-to-novel 평가 프로토콜 출처.
