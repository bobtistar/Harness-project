# Research Questions

## Overview
본 연구는 BioCLIP2의 hierarchical prompt가 만들어내는 성능 향상이 (i) 단순한 텍스트 정보량 증가 때문인지, (ii) embedding 공간의 기하학적 재조직(semantic organization) 때문인지를 분리(disentangle)하는 진단적 분석이다. 4개의 RQ로 구성하며, 답변 순서는 다음과 같다. **RQ1**은 hierarchical prompt가 실제로 분류학적 compactness(intra-class 응집·inter-class 분리)를 향상시키는지를 검증(claim 1). **RQ2**는 그 효과가 분류 깊이(rank)에 따라 어떻게 다른지와, 모델이 *사전학습 단계에서 이미 잠재 분류 구조를 갖고 있었는지*를 검증(claim 2 — latent taxonomic structure). **RQ3**은 counterfactual prompt ablation(무작위 토큰·셔플 계층·텍스트 없는 계층 contrastive)을 통해 "정보 채널 가설 vs 기하 조직 가설"을 결정적으로 가른다(claim 3 — semantic organizer). **RQ4**는 효과가 여러 분류군 도메인(조류·곤충·식물 등)에서 일관되게 재현되는지(외부 타당도)를 확인한다. RQ1→RQ2→RQ3는 점진적 종속이며, RQ4는 RQ1-3 결과의 일반화 검증이다.

## RQ1: Hierarchical taxonomic prompt를 사용한 BioCLIP2는 flat species-only prompt baseline 대비 embedding 공간에서 통계적으로 유의한 semantic compactness 향상을 보이는가?
- **Type**: Comparative (geometric diagnostic)
- **Hypothesis (H1)**: Hierarchical prompt로 인코딩된 이미지/텍스트 embedding은 flat prompt 대비 intra-class variance가 통계적으로 유의하게 감소하고 inter-class margin이 유의하게 증가한다.
- **Null hypothesis (H0)**: Intra-class variance와 inter-class margin이 두 prompt 조건 간 차이가 없다(또는 hierarchical이 더 나쁘다).
- **Variables**
  - Independent: Prompt 유형 {flat species-only, hierarchical 7-rank}.
  - Dependent: (a) intra-class mean pairwise cosine distance, (b) inter-class margin = nearest-other-centroid distance / intra-class std, (c) silhouette score at species level.
  - Confounders to control: 동일 체크포인트(BioCLIP2 공개 ViT-L/14), 동일 평가 데이터셋 split, 동일 클래스 빈도 분포, 텍스트 토큰 길이(필요 시 padding 또는 token-budget 정렬), batch별 normalization 동일.
- **Measurement**: TreeOfLife evaluation subset(최소 1,000종 × 종당 ≥ 20장)에서 이미지 embedding 추출 → 종별 centroid 계산 → intra-class variance, inter-class margin, silhouette을 계산. Bootstrap(B=1,000) 또는 paired permutation test로 두 prompt 조건의 차이에 대한 p-value 산출.
- **Proposed Method (candidate)**: Frozen BioCLIP2 인코더 + zero-shot prompt 변경 실험. (학습 X, 평가 시 prompt만 교체). 추가로 image-only embedding(텍스트 anchor 없이도)에서도 동일 측정을 수행해 효과가 visual representation에 내재하는지 확인.
- **Success threshold**: Intra-class variance ≥ 10% 감소(상대) AND inter-class margin ratio ≥ 1.2배 증가 AND silhouette ≥ +0.05 absolute, 모두 p < 0.01(paired permutation, Bonferroni 보정 후).
- **Falsification condition**: 위 세 지표 중 둘 이상이 임계값을 충족하지 못하거나, hierarchical prompt가 어느 한 지표에서라도 baseline보다 *유의하게 악화*되면 H1 기각.

## RQ2: Hierarchical prompt에 의한 기하 조직 효과는 분류 계층(species → kingdom)의 어느 rank에서 가장 두드러지며, 사전학습된(즉, hierarchical supervision을 받지 않은) baseline CLIP/OpenCLIP에서도 *부분적으로* 동일 구조가 잠재되어 있는가?
- **Type**: Explanatory (claim 2: latent taxonomic structure)
- **Hypothesis (H1)**: (a) Silhouette score와 kNN purity는 species → kingdom 모든 rank에서 hierarchical prompt 조건이 우세하되, 상위 rank(family 이상)에서 effect size가 더 크다. (b) Hierarchical supervision을 받지 않은 일반 CLIP(OpenCLIP ViT-L/14)도 species-only prompt만으로 분류학적 silhouette > random hierarchy baseline을 보인다(잠재 구조 존재).
- **Null hypothesis (H0)**: (a) Rank별 effect size가 통계적으로 동일하다. (b) 일반 CLIP의 분류학적 silhouette이 무작위 분류 라벨을 부여한 control과 차이가 없다.
- **Variables**
  - Independent: (a) 분류 rank ∈ {species, genus, family, order, class, phylum, kingdom}. (b) 모델 ∈ {OpenCLIP, BioCLIP, BioCLIP2}.
  - Dependent: Rank별 silhouette score, rank별 kNN purity(@k=10), mutual information I(embedding cluster ; taxonomic label).
  - Confounders to control: 평가 데이터셋·split 동일, 각 rank에서 클래스 수와 클래스당 샘플 수 균형화(stratified subsampling), embedding 정규화 동일.
- **Measurement**: 각 (모델, rank) 쌍에 대해 silhouette, kNN purity, MI 계산. Rank 간 effect size를 paired bootstrap으로 비교. 잠재 구조 검증을 위해 random taxonomy(클래스 라벨을 무작위 셔플) control을 두고 일반 CLIP에서 유의한 차이를 검정.
- **Proposed Method (candidate)**: 세 모델 모두 frozen, 동일 prompt 템플릿 적용(가능한 경우). 각 rank에서 cluster assignment를 ground-truth taxonomy로 정의, silhouette/MI를 계산. Random-taxonomy permutation test(B=1,000)로 잠재 구조의 유의성 검정.
- **Success threshold**: (a) Family 이상 rank에서 BioCLIP2의 silhouette 향상폭이 species rank 향상폭의 ≥ 1.5배(effect-size 비교, 95% CI 비중첩). (b) OpenCLIP의 분류학적 silhouette이 random-taxonomy 대비 ≥ +0.05 (p < 0.01)이고, 그 값이 BioCLIP2 silhouette의 ≥ 30%에 해당. 두 조건 모두 충족 시 H1 채택.
- **Falsification condition**: (a) Rank 간 effect size 차이가 유의하지 않다(모든 rank 95% CI 중첩). (b) OpenCLIP이 random-taxonomy control과 통계적으로 구분되지 않는다 → 잠재 구조 가설 기각.

## RQ3: Hierarchical prompt의 효과는 "추가 텍스트 정보"(information-channel)에서 오는가, 아니면 "embedding geometry의 조직"(semantic-organizer)에서 오는가?
- **Type**: Causal (counterfactual ablation — claim 3 핵심)
- **Hypothesis (H1)**: 계층 구조 자체를 보존하되 텍스트 *내용*을 의미 없는 토큰으로 치환한 ablation(예: "kingdom_A phylum_B … species_Z" 같은 placeholder)에서도 hierarchical prompt의 geometric benefit의 *상당 부분*(≥ 50%)이 유지된다. 반대로 계층 구조를 *셔플*한(분류학적으로 일관되지 않은) 조건은 그 benefit이 ≤ 20%로 붕괴한다. 즉, 효과의 주된 원천은 텍스트 의미가 아니라 *구조적 supervision*이다.
- **Null hypothesis (H0)**: Random-token hierarchy 조건이 flat baseline과 통계적으로 구분되지 않거나(즉, 텍스트 의미가 사라지면 효과도 사라진다 → 정보 채널 가설 지지), 또는 shuffled-hierarchy 조건이 정상 hierarchy와 구분되지 않는다(구조 무관 → 두 가설 모두 약화).
- **Variables**
  - Independent: Prompt 조건 ∈ {C0: flat species-only, C1: normal hierarchical, C2: random-token hierarchical (구조 보존, 내용 무의미), C3: shuffled hierarchical (계층 라벨을 같은 종 내에서 다른 종 것으로 교체, 구조 파괴), C4: hierarchical word-bag (계층 라벨을 무순서 bag-of-words로 제공, 순서 정보 제거), C5: text-free hierarchical contrastive (텍스트 인코더 우회, 이미지-이미지 hierarchical InfoNCE로 fine-tune)}.
  - Dependent: RQ1의 세 지표(intra-class variance, inter-class margin, silhouette) + taxonomic retrieval LCA depth + hierarchy consistency error.
  - Confounders to control: 토큰 수(C2는 정상 hierarchy와 동일 토큰 수), 평가 데이터, seed(C3 셔플에 대해 5개 seed 평균), C5는 동일 데이터·동일 에폭으로 fine-tune.
- **Measurement**: 각 조건에서 RQ1 지표 측정. 효과 보존율 = (C_x - C0) / (C1 - C0) 로 정의. C2, C5의 보존율과 C3, C4의 보존율을 비교. 95% bootstrap CI로 보존율 간 차이 검정.
- **Proposed Method (candidate)**: BioCLIP2 frozen 체크포인트를 기본으로, C1-C4는 zero-shot prompt swap, C5는 텍스트 인코더 없이 이미지-이미지 hierarchical contrastive head를 light fine-tuning(LoRA 또는 linear adapter). 모든 조건에서 동일 평가 프로토콜.
- **Success threshold**: **Semantic-organizer 가설 채택 조건** — (i) C2 보존율 ≥ 50% (95% CI 하한 ≥ 30%), AND (ii) C3 보존율 ≤ 20% (95% CI 상한 ≤ 40%), AND (iii) C5 보존율 ≥ 50%. 세 조건 모두 충족하면 H1 강한 증거. 둘 충족 시 약한 증거로 보고.
- **Falsification condition**: C2 보존율 < 30% → 정보 채널 가설 지지(텍스트 내용이 필수). 또는 C3 보존율 > 50% → 계층 구조 자체가 무관(다른 메커니즘). 또는 C5 보존율 < 20% → 텍스트 anchor가 필수 매개로, "텍스트는 단지 조직자"라는 주장 약화. 위 중 하나라도 발생하면 claim 3 (semantic organizer) 기각 또는 약화.

## RQ4: RQ1-3에서 관찰된 geometric reorganization 효과는 분류군 도메인(조류·곤충·식물·균류·어류)에 걸쳐 일관되게 나타나는가, 아니면 특정 도메인에 국한되는가?
- **Type**: Descriptive / Generalization
- **Hypothesis (H1)**: RQ1의 핵심 지표(silhouette 향상, intra-class variance 감소)와 RQ3의 C2 보존율은 5개 분류군 도메인 모두에서 동일 방향이며, 도메인 간 effect size의 변동계수(CV) ≤ 0.5.
- **Null hypothesis (H0)**: 도메인 간 effect size의 CV > 0.5이거나, 일부 도메인에서 효과 방향이 반대(intra-class variance 증가).
- **Variables**
  - Independent: 분류군 도메인 ∈ {Aves, Insecta, Plantae, Fungi, Actinopterygii}.
  - Dependent: 도메인별 RQ1 3개 지표 + RQ3 C2 보존율.
  - Confounders to control: 도메인별 클래스 수·샘플 수 균형화(각 도메인에서 동일한 N 종, 종당 동일 K 샘플로 stratified subsample), 도메인별 평균 visual similarity(가능하면 measured).
- **Measurement**: 5개 도메인 각각에서 RQ1, RQ3 분석을 반복. 도메인 간 effect size CV 계산. Random-effects meta-analysis로 pooled effect 및 heterogeneity (I²) 산출.
- **Proposed Method (candidate)**: TreeOfLife 평가 subset을 도메인별로 분할 → 동일 분석 파이프라인 반복 → meta-analysis로 통합.
- **Success threshold**: 5개 도메인 모두에서 RQ1 success threshold를 충족하고, effect size CV ≤ 0.5, I² ≤ 50%(moderate heterogeneity 이하).
- **Falsification condition**: 2개 이상 도메인에서 RQ1 임계값 미충족, 또는 CV > 0.5, 또는 한 도메인이라도 효과 방향이 반대. 이 경우 효과가 도메인 의존적이며 일반 명제로의 확장 보류.

## RQ Dependencies
- **RQ1 → RQ2**: RQ1에서 hierarchical prompt의 기본 geometric benefit이 확인되지 않으면 RQ2의 rank-level/latent-structure 분석은 무의미. RQ1은 진입 게이트.
- **RQ2 → RQ3**: RQ2에서 잠재 구조 존재가 부분이라도 확인되어야 RQ3의 "조직자(organizer)" 해석이 의미를 가진다. 잠재 구조가 전무하면 RQ3의 C2 보존율 결과는 "조직"이 아닌 "신규 정보 학습"으로 재해석될 위험.
- **RQ3 ↔ RQ1**: RQ3는 RQ1 지표를 재사용하므로 분석 파이프라인 공유 가능.
- **RQ4 ⊥ RQ1-3 (논리적으로 독립이나 결과에 종속)**: RQ4는 RQ1과 RQ3의 결과를 도메인별로 반복하므로, RQ1-3의 분석이 끝난 뒤 병렬 실행 가능.

## Out of Scope
- **새 대규모 사전학습 from scratch**: 예산·시간 외(01_problem.md scope에서 제외).
- **Hyperbolic / tree-metric embedding 신규 아키텍처 제안**: 후속 연구로 분리(claim 검증이 우선).
- **비-생물 도메인(ImageNet, COCO)으로의 외삽 검증**: 흥미롭지만 본 연구의 internal validity 확보 후 future work.
- **의료 영상·위성 영상 등 다른 fine-grained 도메인 일반화**: scope 제외.
- **종 분류 정확도(top-1/top-5) 자체의 개선**: 본 연구는 진단적(diagnostic)이며 SOTA 경쟁이 목표가 아님. 정확도는 부수 지표로만 보고.
- **텍스트 인코더 내부 attention 분석(mechanistic interpretability)**: embedding 기하학에 집중하고, 토큰-수준 attribution은 별도 연구로 분리.
