# Experiment Plan

## RQ → Experiment 매핑

| RQ  | Experiment | 답하는 방법 |
|-----|------------|------------|
| RQ1 | Exp1 (Geometry diagnostic — flat vs hierarchical) | 동일 BioCLIP2 frozen 모델에서 두 prompt 조건으로 image/text embedding을 추출하고 intra-class variance, inter-class margin, silhouette을 paired permutation test로 비교 |
| RQ2 | Exp2 (Rank-level + latent taxonomy probing) | (OpenCLIP, BioCLIP, BioCLIP2) × (species…kingdom 7 rank)에서 silhouette/MI/kNN purity를 계산하고 OpenCLIP은 random-taxonomy permutation으로 잠재 구조 유의성 검정 |
| RQ3 | Exp3 (6-condition counterfactual ablation) | C0(flat)–C5(text-free hierarchical InfoNCE)를 동일 평가 데이터에서 비교하고 "효과 보존율"을 bootstrap CI로 측정해 정보 채널 가설 vs 기하 조직 가설 분리 |
| RQ4 | Exp4 (Cross-domain meta-analysis) | Exp1·Exp3를 5개 분류군 도메인(Aves/Insecta/Plantae/Fungi/Actinopterygii)에서 반복하고 random-effects meta-analysis (I², CV)로 일관성 평가 |

## Exp1: Hierarchical vs Flat Prompt의 Geometric Compactness

- **Goal**: RQ1의 H1을 검증. Hierarchical prompt가 intra-class variance를 감소시키고 inter-class margin·silhouette을 증가시키는지 통계 검정.
- **Independent variables**: Prompt 유형 ∈ {flat species-only `"a photo of <species>"`, hierarchical 7-rank `"a photo of <kingdom> <phylum> <class> <order> <family> <genus> <species>"`}.
- **Dependent variables**:
  - `intra_var` = mean over classes of mean pairwise cosine distance within class
  - `inter_margin` = (nearest-other-centroid distance) / (intra-class std)
  - `silhouette` = sklearn silhouette at species level
  - 보조: `alignment` (Wang & Isola), `uniformity`, `RankMe` effective rank
- **Conditions / Treatments**: C0 (flat) vs C1 (normal hierarchical). 동일 frozen 체크포인트.
- **Controls**: 동일 평가 split, 동일 종 집합, 토큰 길이 truncation 일정(77 token), L2 normalize.
- **Sample size**: TreeOfLife eval subset 최소 1,000종 × 종당 20장. 본 코드 골격에서는 toy run으로 ≥ 8 클래스 × ≥ 6 샘플의 mock taxonomy 데이터 (OpenCLIP-ready) 사용.
- **Randomization / 시드**: 시드 5개 (42, 1337, 2024, 7, 314). Bootstrap B=1000.
- **Success criteria**: intra_var ≥ 10% 감소 AND inter_margin ≥ 1.2× 증가 AND silhouette ≥ +0.05, 모두 p < 0.01 (Bonferroni 보정).
- **Risks & mitigations**:
  - Risk: Modality gap이 image–text 비교를 오염 → Mitigation: image-only embedding에서도 동일 측정 (텍스트 anchor를 prompt 차이의 인디케이터로만 사용).
  - Risk: 토큰 길이 차이가 confounder → Mitigation: flat 조건에 동일 길이의 padding 토큰 추가 변형 보고.

## Exp2: Rank-level Effect + Latent Taxonomy Probing

- **Goal**: RQ2의 (a) rank별 effect size 분해 + (b) 일반 OpenCLIP의 잠재 구조 유의성 검정.
- **Independent variables**: (a) rank ∈ {species, genus, family, order, class, phylum, kingdom}, (b) model ∈ {OpenCLIP ViT-L/14, BioCLIP ViT-B/16, BioCLIP2 ViT-L/14}.
- **Dependent variables**: rank별 silhouette, kNN purity@10, mutual information I(predicted-cluster ; gt-rank-label).
- **Controls**: 동일 데이터, 동일 normalize, rank마다 stratified subsample.
- **Sample size**: 도메인 1개 (Aves) 우선; 도메인 일반화는 Exp4에서. 본 골격은 toy taxonomy.
- **Randomization / 시드**: random-taxonomy permutation B=1000, 시드 5개.
- **Success criteria**:
  - (a) Family 이상에서 BioCLIP2 silhouette 향상이 species 향상의 ≥ 1.5× (95% CI 비중첩).
  - (b) OpenCLIP silhouette > random-taxonomy control, p < 0.01, 값이 BioCLIP2 silhouette의 ≥ 30%.
- **Risks & mitigations**:
  - Risk: 모델 간 embedding 차원·norm 차이 → Mitigation: L2 normalize 후 silhouette은 cosine distance.
  - Risk: BioCLIP/BioCLIP2 가중치 다운로드 (HuggingFace) 실패 → Mitigation: OpenCLIP만 실제 실행, BioCLIP/BioCLIP2은 mock placeholder + 실행법 명시.

## Exp3: Counterfactual Prompt Ablation (RQ3 핵심)

- **Goal**: 6개 조건 C0–C5로 정보 채널 vs 기하 조직 가설 분리.
- **Independent variables**: prompt 조건 ∈
  - C0: flat species-only
  - C1: normal hierarchical 7-rank
  - C2: random-token hierarchical (placeholder 토큰, 구조 보존)
  - C3: shuffled hierarchical (다른 종의 상위 rank 라벨을 임의 교체, 구조 파괴)
  - C4: hierarchical word-bag (7개 라벨 무순서 셔플 — 순서 정보 제거, 어휘 보존)
  - C5: text-free hierarchical contrastive (텍스트 인코더 우회, image-image hierarchical InfoNCE를 linear adapter로 light fine-tune)
- **Dependent variables**: RQ1의 3개 지표 + LCA depth + hierarchy consistency error.
- **Controls**: 토큰 수 동일(C2는 C1과 동일 토큰 수), C3 셔플은 시드 5개 평균, C5는 동일 데이터·동일 epoch.
- **Sample size**: Exp1과 동일 eval set.
- **Randomization / 시드**: 5 seed, bootstrap B=1000.
- **Success criteria (Semantic-Organizer)**:
  - (i) C2 보존율 = (M(C2)-M(C0))/(M(C1)-M(C0)) ≥ 50% (95% CI 하한 ≥ 30%)
  - (ii) C3 보존율 ≤ 20% (CI 상한 ≤ 40%)
  - (iii) C5 보존율 ≥ 50%
- **Risks & mitigations**:
  - Risk: C5 fine-tune이 hyperparameter 민감 → Mitigation: 동일 LR/epoch grid를 모든 조건에 적용; 5 seed 평균; LoRA rank 고정.
  - Risk: 무작위 토큰이 사실은 의미적 토큰일 위험(BPE 충돌) → Mitigation: vocabulary 외 placeholder string (e.g., `tax0`, `tax1`, …) 사용.

## Exp4: Cross-Domain Generalization (RQ4)

- **Goal**: 5개 분류군 도메인에서 Exp1, Exp3를 반복하고 random-effects meta-analysis.
- **Independent variables**: 도메인 ∈ {Aves, Insecta, Plantae, Fungi, Actinopterygii}.
- **Dependent variables**: 도메인별 RQ1 3개 지표, RQ3 C2 보존율.
- **Controls**: 도메인별 동일 N 종 × 동일 K 샘플 stratified subsample.
- **Sample size**: 도메인당 ≥ 200 종 × ≥ 10 샘플 (toy run에서는 도메인당 4 종 × 5 샘플 placeholder).
- **Randomization / 시드**: 5 seed.
- **Success criteria**: 5 도메인 모두 RQ1 threshold 충족, CV ≤ 0.5, I² ≤ 50%.
- **Risks & mitigations**:
  - Risk: 도메인 간 평균 visual similarity 차이가 confound → Mitigation: 도메인별 baseline (C0) 대비 *상대* 향상을 사용.
  - Risk: 일부 도메인 (Fungi) eval 데이터 부족 → Mitigation: Treepy/iNat21 사용 또는 도메인 제외 보고.

## Execution Status Note

본 골격은 환경 제약상 BioCLIP2/BioCLIP HuggingFace 가중치 대용량 다운로드 및 TreeOfLife 평가 셋 다운로드를 수행하지 **않는다**. 대신:
1. OpenCLIP ViT-B/32 (소형, laion-2b 또는 default)을 toy taxonomy(8 placeholder 종, 4 도메인)로 실제 실행하여 파이프라인 sanity check.
2. BioCLIP/BioCLIP2 결과는 명시적으로 **MOCK**으로 표기하고, 실제 실행 명령어를 README에 제공.
