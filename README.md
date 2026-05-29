# BioCLIP2 계층 프롬프트와 임베딩 기하학 분석

BioCLIP2의 hierarchical taxonomic prompt가 가져오는 성능 향상이 **추가 텍스트 정보** 때문인지, **임베딩 공간의 기하학적 재조직** 때문인지를 분리(disentangle)하는 진단적 연구입니다.

---

## 연구 배경

**BioCLIP2** (arXiv:2505.23883, NeurIPS'25 Spotlight)는 2억 1,400만 장 규모의 생물 이미지에 계층적 대조 학습을 적용해 BioCLIP 대비 종 분류 정확도 +18.1%p를 달성했습니다. 그러나 논문에서 주장하는 "emergent properties"(종간 임베딩이 부리 크기·서식지 등 생태적 특성과 자연스럽게 정렬됨)는 정성적 시각화에 머물러 있으며, 다음 세 가지가 부재합니다.

- flat-prompt 대조 실험 (계층 프롬프트 vs 종명만 사용)
- 기하학적 지표의 정량화
- 반사실적(counterfactual) 프롬프트 ablation

본 연구는 정확히 이 갭을 채우는 진단적 분석을 수행합니다.

---

## 핵심 주장과 검증 구조

검증하는 세 가지 주장:

> **Claim 1.** 계층 프롬프트는 임베딩 공간에서 의미적 집약도(클래스 내 분산 ↓, 클래스 간 마진 ↑)를 통계적으로 유의하게 향상시킨다.

> **Claim 2.** 이 향상은 사전학습된 비전-언어 모델이 이미 잠재적 분류학 구조를 내재하고 있음을 시사한다. 즉, 계층 supervision은 새 구조를 *학습시키는* 것이 아니라 이미 있던 구조를 *꺼내 정렬*시킨다.

> **Claim 3.** 계층 supervision은 단순히 텍스트 신호를 풍부하게 하는 것이 아니라, 임베딩 기하학의 **의미적 조직자(semantic organizer)** 로 기능한다.

---

## 연구 질문 (RQ)

| RQ | 내용 | 유형 |
|---|---|---|
| RQ1 | 계층 프롬프트는 flat baseline 대비 embedding semantic compactness를 통계적으로 유의하게 향상시키는가? | 비교 실험 |
| RQ2 | 기하 조직 효과는 분류 rank에 따라 어떻게 다르며, 계층 supervision 없이 학습된 일반 CLIP에서도 잠재 분류 구조가 존재하는가? | 설명적 분석 |
| RQ3 | 효과의 원천이 "정보 채널"인가 "기하 조직"인가? (counterfactual ablation) | 인과적 분석 |
| RQ4 | RQ1-3의 효과는 조류·곤충·식물·균류·어류 5개 도메인에서 일관되게 재현되는가? | 외부 타당도 검증 |

---

## 실험 설계: Counterfactual Prompt Ablation

핵심 실험은 텍스트 *내용*과 계층적 *구조*를 분리하는 5가지 프롬프트 조건입니다.

| 조건 | 설명 | 목적 |
|---|---|---|
| C0 — flat | 종명만 사용 (`Passer domesticus`) | 베이스라인 |
| C1 — hierarchical | 7-rank 린네 분류 전체 (`Animalia Chordata … Passer domesticus`) | 전체 신호 |
| C2 — random-token | 계층 구조 보존 + 토큰 의미 제거 (`taxA taxB … taxG`) | 기하 조직 가설 검증 |
| C3 — shuffled | 상위 분류 라벨을 다른 종 것으로 교체, 구조 파괴 | 정보 채널 가설 검증 |
| C4 — bag-of-words | 유효 토큰이지만 순서 제거 | 시퀀스 순서 효과 분리 |

**결정 규칙**

$$\rho_M(C_x) = \frac{\text{metric}(C_x) - \text{metric}(C_0)}{\text{metric}(C_1) - \text{metric}(C_0)}$$

- $\rho(C_2) \geq 0.5$ **AND** $\rho(C_3) \leq 0.2$ → **기하 조직 가설 채택**
- $\rho(C_2) < 0.3$ → 정보 채널 가설 지지
- 그 외 → 불확정(inconclusive)

---

## 평가 지표

- **Intra-class variance** — 종별 평균 쌍별 코사인 거리 (클래스 내 응집도)
- **Inter-class margin** — 최근접 타종 centroid 거리 / 클래스 내 표준편차
- **Silhouette score** — 각 분류 rank(species → kingdom)에서 측정
- **Taxonomic retrieval LCA depth** — top-k 검색 결과의 평균 최저 공통 조상 깊이
- **Hierarchy consistency error** — "better mistakes" 지표

5개 생물 도메인 전반 평가: **조류(Aves), 곤충(Insecta), 식물(Plantae), 균류(Fungi), 조기어류(Actinopterygii)**

---

## 관련 선행 연구

| 논문 | 연도 | 핵심 기여 | 본 연구와의 관계 |
|---|---|---|---|
| Stevens et al. *BioCLIP* (CVPR'24 Best Student Paper) | 2024 | TreeOfLife-10M, 7-rank 계층 프롬프트 | 분석 대상 베이스라인 모델 |
| Gu et al. *BioCLIP2* (NeurIPS'25 Spotlight) | 2025 | 214M 이미지, +18.1%p 종 분류 | 본 연구의 분석 대상 모델; 정량 진단 부재가 갭 |
| Novack et al. *CHiLS* (ICML'23) | 2023 | 계층 label set으로 CLIP zero-shot 향상 | 프롬프트 측 baseline |
| Liang & Davis *HAPrompts* (arXiv:2503.02248) | 2024 | LLM 생성 계층 프롬프트, "better mistakes" 목적 | RQ2 rank별 효과 비교 대상 |
| Zhang et al. *HiMulCon* (arXiv:2204.13207) | 2022 | label-tree 거리 기반 계층 대조 손실 | BioCLIP2 학습 설계의 원형 |
| Geng et al. *HiCLIP* (ICLR'23) | 2023 | supervision 없이 attention으로 계층 구조 발견 | RQ2 잠재 구조 가설의 직접 지지 근거 |

---

## 산출물

```
workspace/bioclip2/
  01_problem.md     # 갭 분석, 범위, 성공 기준
  02_rqs.md         # RQ1-4 가설 및 반증 조건
  03_litreview.md   # 선행연구 24편 범주화 정리
  06_reviews.md     # 가상 동료 리뷰 3인 + 메타 리뷰
  07_rebuttal.md    # 리뷰 코멘트별 반박문
```

---

## 연구 방법론: Harness 파이프라인

본 연구는 **Harness** — 멀티에이전트 Claude 오케스트레이션 기반 논문 자동화 파이프라인 — 을 통해 수행되었습니다.

키워드를 입력하면 7개의 전문화된 LLM 에이전트가 순차적으로 실행되며, 각 에이전트는 이전 단계의 산출물을 파일로 직접 읽어 처리합니다. 오케스트레이터가 산출물 전문을 프롬프트에 붙이지 않아 토큰 비용을 최소화합니다.

```
키워드 →  문제 정의  →  RQ 설계  →  문헌 조사  →  실험  →  논문 작성  →  리뷰  →  리버탈
           (Sonnet)     (Sonnet)    (Sonnet)    (Sonnet)   (Sonnet)   (Opus)   (Sonnet)
```

> 본 BioCLIP2 연구는 위 파이프라인을 통해 문제 정의에서 리버탈 초안까지 전체 7단계를 자동 완료한 실제 실행 사례입니다.
