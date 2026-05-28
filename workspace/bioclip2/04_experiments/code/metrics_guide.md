# 평가 지표 가이드

---

## 1. 기하학적 지표 (RQ1)

Hierarchical prompt 조건(C0–C4)이 embedding 공간의 기하 구조를 어떻게 바꾸는지 진단한다.  
측정 흐름: **시각화 → 종내 응집 → 종간 분리 → 공간 품질**

---

### UMAP / t-SNE 시각화 (`plot_embeddings`, `plot_comparison_grid`)

> **논문 역할**: C0–C4 조건별 embedding 분포를 2D로 시각화해 기하 변화를 직관적으로 확인. 수치 지표 해석 전 sanity check 용도.

각 조건의 임베딩을 UMAP 또는 t-SNE로 축약해 scatter plot 생성. `plot_comparison_grid`는 C0–C4를 한 화면에 서브플롯으로 나란히 배치해 조건 간 비교를 쉽게 한다.

`metrics.py:154, 205` · `run_experiment.py:588, 609, 627`

---

### 종내 응집 / 종간 분리

#### `intra_class_variance`

> **논문 역할**: RQ1 1차 판정 지표. Hierarchical prompt가 같은 종 이미지를 embedding 공간에서 얼마나 뭉치게 하는지 측정. **낮을수록 좋음.**  
> RQ1 통과 기준: C0 대비 ≥ 10% 감소 (p < 0.01)

같은 종에 속하는 샘플들 간 cosine 거리를 모두 계산한 뒤 종별로 평균 → 전체 종에 대해 평균. L2 정규화된 벡터에서 cosine 거리 = 1 − cosine 유사도.

`metrics.py:43` · `run_experiment.py:88`

---

#### `inter_class_margin`

> **논문 역할**: RQ1 1차 판정 지표. 종간 분리 정도를 측정. 가장 가까운 타종과의 거리가 종내 퍼짐에 비해 얼마나 큰지 비율로 표현. **높을수록 좋음.**  
> RQ1 통과 기준: C0 대비 ≥ 1.2배 증가 (p < 0.01)

종별 centroid 간 cosine 거리 행렬 계산 → 각 종에 대해 (가장 가까운 타종 centroid까지의 거리) / (종내 샘플의 centroid까지의 거리 표준편차).

`metrics.py:64` · `run_experiment.py:89`

---

### 클러스터링 품질

#### `silhouette_cosine`

> **논문 역할**: RQ1 1차 판정 지표 + RQ3 보존율(ρ)의 headline 지표. 종내 응집과 종간 분리를 하나의 스칼라로 통합. **높을수록 좋음.**  
> RQ1 통과 기준: C0 대비 ≥ +0.05 (p < 0.01). RQ3 판정: ρ(C2) ≥ 0.5, ρ(C3) ≤ 0.2

sklearn silhouette_score를 cosine 거리 기준으로 계산. 각 샘플에 대해 (가장 가까운 타 클러스터 평균 거리 − 동일 클러스터 평균 거리) / max(둘 중 큰 값) → 전체 평균.

`metrics.py:90` · `run_experiment.py:90`

---

### Embedding 공간 품질

#### `uniformity`

> **논문 역할**: RQ1 보조 지표. Embedding이 구면에 고르게 분포하는지 확인. 너무 한 곳에 몰리면(차원 붕괴) 다른 지표 해석이 왜곡될 수 있어 보조적으로 함께 보고.

Wang & Isola 2020. 모든 쌍에 대해 `log E[exp(−t‖f(x)−f(y)‖²)]` 계산. L2 정규화 후 pairwise 거리 기반. 값이 클수록(덜 음수) 균등 분포.

`metrics.py:109` · `run_experiment.py:92`

---

#### `rankme`

> **논문 역할**: RQ1 보조 지표. Embedding 행렬의 effective rank로 차원 붕괴 여부 점검. C2(random-token) 조건에서 C1(정상 hierarchical)과 spectral footprint가 유지되는지 확인하는 데 특히 유효.

Garrido et al. 2023. SVD로 singular value 추출 → 정규화해 확률 분포 생성 → Shannon entropy의 지수값(exp(H)). 높을수록 다양한 차원 활용.

`metrics.py:123` · `run_experiment.py:91`

---

## 2. 분류학적 지표 (RQ2)

Embedding이 분류학적 계층 구조를 실제로 반영하는지 측정한다.

---

### `knn_purity_at_k`

> **논문 역할**: RQ2 핵심 지표. k-NN 이웃 중 같은 종 비율로 계층별 embedding 정렬 정도를 측정. rank(species→kingdom)별로 반복 계산해 "어느 계층에서 효과가 가장 크냐"를 추적.

L2 정규화 후 cosine 거리 기준 k-NN(기본 k=10) 탐색. 각 샘플의 이웃 중 동일 라벨 비율 → 전체 평균.

`metrics.py:138` · `run_experiment.py:93`

---

## 3. 통계 헬퍼

지표 값의 유의성과 효과 크기를 검증한다.

---

### `paired_permutation_test`

> **논문 역할**: RQ1 지표 차이의 통계적 유의성 검증. p < 0.01 기준 (Bonferroni 보정 후) 충족 여부가 RQ1 통과 조건.

두 조건 A, B의 쌍별 차이(A−B)를 관측. 부호를 무작위 반전시킨 1,000개 permutation 분포와 비교해 p-value 산출. 95% CI는 bootstrap으로 병행 계산.

`metrics.py:300` · `run_experiment.py:175`

---

### `bootstrap_ci`

> **논문 역할**: 보존율(ρ)과 지표 차이의 95% 신뢰구간 제공. RQ3 판정 조건(ρ CI 하한/상한)에 직접 사용.

복원 추출(B=1,000)로 통계량 분포 추정 → 2.5·97.5 백분위로 CI 산출.

`metrics.py:333` · `run_experiment.py:176, 290`

---

### `cohens_d_paired`

> **논문 역할**: 통계적 유의성(p-value)과 별개로 효과의 실질적 크기를 보고. p가 작아도 d가 작으면 의미 없는 차이임을 판별.

paired 차이의 평균 / 표준편차(ddof=1).

`metrics.py:366` · `run_experiment.py:179`

---

### 보존율 ρ (RQ3 핵심 판정)

> **논문 역할**: **RQ3의 중심 공식.** C2·C3·C4가 C1의 효과를 얼마나 보존하는지를 0–1 스칼라로 표현.  
> **판정 기준**: ρ(C2) ≥ 0.5 AND ρ(C3) ≤ 0.2 → H_geom(기하 조직자 가설) 채택

```
ρ = (M(Cx) − M(C0)) / (M(C1) − M(C0))
```

`run_experiment.py:282–293`에 인라인 구현. `metrics.py:354`의 `effect_preservation_ratio` 함수와 동일한 로직.
