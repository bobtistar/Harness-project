# Experiment Plan

## RQ → Experiment 매핑

| RQ  | Experiment | 답하는 방법 |
|-----|------------|------------|
| RQ1 | Exp1 (Geometry diagnostic — flat vs hierarchical) | 동일 BioCLIP2 frozen 모델에서 두 prompt 조건으로 image/text embedding을 추출하고 intra-class variance, inter-class margin, silhouette을 paired permutation test로 비교 |
| RQ2 | Exp2 (Rank-level effect + latent taxonomy probing) | 각 Linnaean rank(species…order)에서 silhouette/kNN purity를 계산하고, 무작위-taxonomy permutation 대조로 image embedding 잠재 구조의 유의성을 검정 |
| RQ3 | Exp3 (6-condition counterfactual ablation) | C0(flat)…C5(text-free hierarchical InfoNCE)를 동일 평가 데이터에서 비교해 *정보 채널 가설* vs *기하 조직 가설*을 분리; 효과 보존율을 bootstrap CI로 측정 |
| RQ4 | Exp4 (Cross-domain meta-analysis) | Exp1·Exp3를 다중 분류군 도메인(Aves/Insecta/Plantae/Fungi/Actinopterygii)에서 반복하고 random-effects meta-analysis (I², CV)로 일관성 평가 — **본 실행에서는 미수행** |

## Exp1: Hierarchical vs Flat Prompt의 Geometric Compactness

- **Goal**: RQ1의 H1을 검증. Hierarchical prompt가 intra-class variance를 감소시키고 inter-class margin·silhouette을 증가시키는지 통계 검정.
- **Independent variables**: Prompt 유형 ∈ {flat species-only `"a photo of <species>"`, hierarchical 7-rank `"a photo of <kingdom> <phylum> <class> <order> <family> <genus> <species>"`}.
- **Dependent variables**:
  - `intra_var` = mean over classes of mean pairwise cosine distance within class
  - `inter_margin` = (nearest-other-centroid distance) / (intra-class std)
  - `silhouette` = sklearn silhouette at species level (cosine)
  - 보조: `RankMe` effective rank, `uniformity`, `knn_purity@10`
- **Conditions / Treatments**: C0 (flat) vs C1 (normal hierarchical). 동일 frozen 체크포인트.
- **Controls**: 동일 평가 split, 동일 종 집합, 토큰 길이 truncation 77 token, L2 normalize.
- **Sample size**: CUB-200-2011 전체 200 species × 평균 ~59 images = **11,788 images**.
- **Randomization / 시드**: 시드 5개 (100+s_idx for Exp3, 42+s_idx for Exp1; image embedding에 σ=1e-3 가우시안 noise를 더해 stochastic sampling 시뮬레이션).
- **Statistical tests**: Paired permutation (B=1000), 6 metric에 대해 Bonferroni 보정 시 α = 0.01/6.
- **Success criteria**: intra_var ≥ 10% 감소 AND inter_margin ratio ≥ 1.2× AND silhouette gain ≥ +0.05 (모두 동시 충족).
- **Risks & mitigations**:
  - Risk: CUB-200은 단일 class(Aves)에 한정 → 상위 rank(kingdom/phylum/class) 토큰이 종간 변별에 무의미 → C1이 오히려 noise 추가할 수 있음 → Mitigation: 이 한계를 explicit confound로 보고하고, 실험에서 rank별 분해(Exp2)로 확인.
  - Risk: Seed 간 noise(σ=1e-3)가 작아 std가 비현실적으로 낮음 → Cohen's d inflate → Mitigation: effect size를 raw difference + bootstrap CI로 함께 보고.

## Exp2: Rank-level Effect + Latent Taxonomy Probing

- **Goal**: RQ2의 (a) rank별 effect size 분해 + (b) image embedding 잠재 구조의 유의성 검정.
- **Independent variables**: rank ∈ {species, genus, family, order, class, phylum, kingdom}.
- **Dependent variables**: rank별 silhouette, kNN purity@10, intra/inter margin.
- **Controls**: 동일 image embedding (Exp1과 동일 BioCLIP2 캐시), rank label만 변경. 단일 클래스(< 2 distinct labels) rank는 skip.
- **Sample size**: CUB-200 전체 11,788 images. CUB-200 taxonomy 특성상 모든 종이 Aves → class/phylum/kingdom은 자동 skip.
- **Latent taxonomy probe**: 종 라벨을 무작위 permute한 50개 control에 대해 silhouette 분포 계산 → 실제 silhouette과 z-score 비교.
- **Success criteria**:
  - (a) family·order에서 C1 silhouette gain ≥ species 수준 gain의 +ε (방향 일관성).
  - (b) 실제 silhouette > random-taxonomy control, |z| > 10.
- **Risks & mitigations**:
  - Risk: CUB-200은 상위 rank (Aves 이상)에서 degenerate → Mitigation: skipped rank를 출력에 명시.
  - Risk: 50 permutation은 적음 → Mitigation: z-score가 큰 영역에서만 신뢰; 필요시 B=1000으로 확장.

## Exp3: Counterfactual Prompt Ablation (RQ3 핵심)

- **Goal**: 6개 조건 C0–C5로 정보 채널 가설(텍스트가 단순 정보 채널) vs 기하 조직 가설(구조가 본질) 분리.
- **Independent variables**: prompt 조건 ∈
  - C0: flat species-only
  - C1: normal hierarchical 7-rank
  - C2: random-token hierarchical (placeholder 토큰 `tax0…tax6`, 구조 보존)
  - C3: shuffled hierarchical (상위 rank 라벨을 다른 종에서 임의 교체, 구조 파괴)
  - C4: hierarchical word-bag (7개 라벨 매 호출마다 random permute — 순서 정보 제거, 어휘 보존)
  - C5: text-free hierarchical InfoNCE (텍스트 인코더 우회, linear adapter + LCA-weighted InfoNCE, 5 epoch)
- **Dependent variables**: intra_var, inter_margin, silhouette (3개 핵심 지표).
- **Effect-preservation ratio**: ρ_x = (M(C_x) − M(C0)) / (M(C1) − M(C0)). 각 seed별 paired ratio 계산 후 bootstrap CI.
- **Controls**: 토큰 수 동일 (C2는 C1과 동일), C3 셔플은 시드별 다른 permutation, C5는 동일 epoch/lr.
- **Sample size**: Exp1과 동일 11,788 images.
- **Randomization / 시드**: 5 seed (100…104), bootstrap B=1000.
- **Success criteria (Semantic-Organizer)**:
  - (i) C2 silhouette 보존율 ≥ 0.5, CI 하한 ≥ 0.3
  - (ii) C3 silhouette 보존율 ≤ 0.2, CI 상한 ≤ 0.4
  - (iii) C5 silhouette 보존율 ≥ 0.5
- **Risks & mitigations**:
  - Risk: CUB-200에서 (C1 − C0) < 0이면 preservation ratio 부호가 뒤집혀 해석 무의미 → Mitigation: raw silhouette 값과 절대 차이도 함께 보고; 분모 부호를 명시.
  - Risk: C5 5-epoch light adapter는 frozen BioCLIP2 representation을 능가하기 어려움 → Mitigation: 단일 도메인에서 lower-bound로만 해석하고, 더 긴 학습 grid를 future work로 명시.
  - Risk: random-token이 BPE 분해 후 의미 토큰과 충돌 → Mitigation: `tax0…tax6` 같은 vocabulary-foreign 문자열 사용.

## Exp4: Cross-Domain Generalization (RQ4) — **미수행**

- **Goal**: 5개 분류군 도메인에서 Exp1·Exp3를 반복하고 random-effects meta-analysis.
- **Independent variables**: 도메인 ∈ {Aves(CUB-200/NABirds), Insecta(BIOSCAN-1M), Plantae(PlantNet300K), Fungi(iNat21), Actinopterygii(FishNet)}.
- **본 실행에서의 상태**: **미수행**. Aves(CUB-200)만 완료. 나머지 4 도메인은 데이터 확보 + 컴퓨트 예산 부족.
- **Future work success criteria**: 5 도메인 모두 RQ1 threshold 충족, between-domain CV ≤ 0.5, I² ≤ 50%.

## Execution Status

| 구분 | 상태 |
|---|---|
| 코드 골격 작성 | 완료 |
| Exp1 (CUB-200, BioCLIP2 ViT-L/14 cuda) | **완료** (5 seeds, 11,788 images) |
| Exp2 (rank-level + latent probe) | **완료** (kingdom/phylum/class은 degenerate → skip) |
| Exp3 (C0…C5 counterfactual ablation) | **완료** (5 seeds) |
| Exp4 (5 도메인 cross-domain) | **미수행** — Aves만 |
| **OpenCLIP ViT-L/14 (추가 baseline 비교)** | **완료** — CUB-200 동일 데이터, Exp1·Exp2·Exp3 모두 실행 (≈ 68분). RQ4(다중 도메인 meta-analysis)와는 별개의 cross-model 비교; 동일 Aves 단일 도메인에서 일반 도메인 사전학습 모델(LAION-2B)과 생물 도메인 사전학습 모델(BioCLIP2)의 정성적 결론 일관성 확인 목적. |

실제 실행 결과는 [results.md](results.md) 참조.
