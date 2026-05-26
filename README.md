# Harness Project — AI 기반 논문 자동화 파이프라인

**키워드 한 줄 → 투고 가능한 논문 드래프트**, 멀티에이전트 Claude 오케스트레이션으로 완전 자동화.

---

## 개요

이 저장소는 LLM 에이전트로 구동되는 엔드투엔드 학술 연구 파이프라인입니다. 연구 키워드 또는 주장 집합을 입력하면, 문제 정의부터 리버탈 초안 작성까지 논문 작성의 모든 단계를 자율적으로 수행합니다.

파이프라인은 **생물 분류학을 위한 비전-언어 모델** 연구 주제를 통해 구축 및 검증되었습니다. 구체적으로는 BioCLIP2의 계층적 프롬프팅 메커니즘을 분석하는 연구를 전체 파이프라인 실행 예시로 사용했습니다.

---

## 파이프라인 단계

```
키워드 입력
  │
  ├─ 01 문제 정의 (Problem Definer)     → 갭 분석, 범위, 성공 기준 도출
  ├─ 02 연구 질문 (RQ Formulator)       → 반증 조건을 포함한 검증 가능한 가설 설계
  ├─ 03 문헌 조사 (Lit Reviewer)        → 범주화된 선행연구 정리, 베이스라인 식별
  ├─ 04 실험 설계 (Experiment Runner)   → 실험 설계, 코드 작성, 결과 테이블 생성
  ├─ 05 논문 작성 (Paper Writer)        → 초록~결론 전체 드래프트 작성
  ├─ 06 리뷰 (Reviewer)                 → 3인 가상 동료 리뷰 + 메타 리뷰 시뮬레이션
  └─ 07 리버탈 (Rebuttal Drafter)       → 코멘트별 반박문 및 수정 약속 초안
```

각 단계는 독립적인 Claude 서브에이전트로 구성되며, 이전 단계의 산출물을 파일로 직접 읽어 처리합니다. 오케스트레이터가 이전 결과를 프롬프트에 통째로 붙이지 않으므로 토큰 비용이 낮습니다.

---

## 적용 연구 사례: BioCLIP2 계층 프롬프트와 임베딩 기하학

### 연구 질문

> BioCLIP2의 계층적 분류군 프롬프트가 가져오는 성능 향상은 *추가적인 텍스트 정보* 때문인가, 아니면 *임베딩 공간의 기하학적 재조직* 때문인가?

세 가지 핵심 주장을 검증합니다.

1. 계층 프롬프트는 임베딩 공간에서 **의미적 집약도**(클래스 내 분산 ↓, 클래스 간 마진 ↑)를 향상시킨다.
2. 이 향상은 사전학습된 비전-언어 모델이 이미 **잠재적 분류학 구조**를 내재하고 있음을 시사한다.
3. 계층 supervision은 단순히 텍스트 신호를 풍부하게 하는 것이 아니라, **임베딩 기하학의 의미적 조직자**로 기능한다.

### 연구 배경 및 갭

**BioCLIP2** (arXiv:2505.23883, NeurIPS'25 Spotlight)는 2억 1,400만 장 규모의 생물 이미지에 계층적 대조 학습을 적용해 BioCLIP 대비 종 분류 정확도를 18.1%p 향상시켰습니다. 그러나 "emergent properties" 주장(종간 임베딩이 부리 크기·서식지 등 생태적 특성과 정렬된다)은 정성적 시각화에 머물며, 다음 세 가지가 부재합니다.

- flat-prompt 대조 실험
- 기하학적 지표의 정량화
- 반사실적(counterfactual) 프롬프트 ablation

본 연구는 정확히 이 갭을 채웁니다.

### 실험 설계

| 프롬프트 조건 | 설명 | 검증 대상 |
|---|---|---|
| C0 — flat | 종명만 사용 | 베이스라인 |
| C1 — hierarchical | 7-rank 린네 분류 전체 시퀀스 | 전체 신호 |
| C2 — random-token hierarchical | 구조 보존, 토큰 의미 제거 | 기하 조직 가설 |
| C3 — shuffled hierarchical | 상위 분류 라벨을 다른 종 것으로 교체, 구조 파괴 | 정보 채널 가설 |
| C4 — bag-of-words hierarchical | 유효 토큰이지만 순서 제거 | 시퀀스 구조 효과 |

**결정 규칙**: C2 효과 보존율 ≥ 50% **AND** C3 효과 보존율 ≤ 20% → 기하 조직 가설 채택.

### 평가 지표

- 클래스 내 분산 (종별 평균 쌍별 코사인 거리)
- 클래스 간 마진 (최근접 타 종 centroid 거리 / 클래스 내 표준편차)
- 각 분류 rank에서의 실루엣 점수
- 분류학적 검색 LCA 깊이 (top-k 결과의 평균 최저 공통 조상)
- 계층 일관성 오류 ("better mistakes" 지표)

5개 생물 도메인 전반에 걸쳐 평가: 조류(Aves), 곤충(Insecta), 식물(Plantae), 균류(Fungi), 조기어류(Actinopterygii).

### 파이프라인 산출물 (전체 7단계 완료)

| 단계 | 산출 파일 |
|---|---|
| 문제 정의 | `workspace/bioclip2/01_problem.md` |
| 연구 질문 | `workspace/bioclip2/02_rqs.md` |
| 문헌 조사 (24편) | `workspace/bioclip2/03_litreview.md` |
| 가상 동료 리뷰 (3인 + 메타) | `workspace/bioclip2/06_reviews.md` |
| 리버탈 | `workspace/bioclip2/07_rebuttal.md` |

---

## 프로젝트 구조

```
.claude/
  _index.md          # 단계별 에이전트 라우팅 테이블
  _common.md         # 전 에이전트 공통 규약
  agents/            # 7개 단계별 에이전트 정의
  commands/paper.md  # 파이프라인 진입점 (/paper 명령)
workspace/<slug>/    # 실행 결과 산출물 (gitignore 대상)
```

**오케스트레이터 모델**: Claude Sonnet (저비용 라우팅)  
**비판적 추론 에이전트** (Reviewer): Claude Opus  
**나머지 에이전트**: Claude Sonnet

---

## 사용법

```bash
# 전체 파이프라인 자동 실행
/paper BioCLIP2 hierarchical prompts auto

# 특정 단계만 재실행 (예: 드래프트 수정 후 리뷰 재실행)
/paper --stage review <slug>
```

---

## 참고 문헌

- Stevens et al., *BioCLIP: A Vision Foundation Model for the Tree of Life*, CVPR 2024 (Best Student Paper)
- Gu, Stevens et al., *BioCLIP 2: Emergent Properties from Scaling Hierarchical Contrastive Learning*, arXiv:2505.23883, NeurIPS 2025 Spotlight
- Novack et al., *CHiLS: Zero-Shot Image Classification with Hierarchical Label Sets*, ICML 2023
- Liang & Davis, *HAPrompts: Making Better Mistakes in CLIP-Based Zero-Shot Classification*, arXiv:2503.02248
- Zhang et al., *Use All The Labels: A Hierarchical Multi-Label Contrastive Learning Framework*, arXiv:2204.13207
