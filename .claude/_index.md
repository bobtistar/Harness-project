---
name: agent-index
purpose: 라우터가 전체 agents/*.md 본문을 읽지 않고도 1~2개를 선택할 수 있게 하는 카탈로그.
loaded_by: paper.md 및 사용자가 명시적으로 Read하는 경우에만 (자동 로드 X)
---

# Agent Router Index

라우팅 원칙:
- **1단계 1 agent** 호출. 절대 2개 이상 동시 호출 금지 (병렬 비용 두 배).
- 호출 전 이 파일과 해당 agent 본문만 검토. 다른 agent 본문은 읽지 말 것.
- 어떤 단계인지 모호하면 사용자에게 한 줄 질문 후 결정. 추측 호출 금지.

## Stage → Agent Map

| Stage | Agent | Model | Input | Output |
|-------|-------|-------|-------|--------|
| 1 | `problem-definer`   | sonnet | 키워드 | `01_problem.md` |
| 2 | `rq-formulator`     | sonnet | `01_problem.md` | `02_rqs.md` |
| 3 | `lit-reviewer`      | sonnet | `01_problem.md` + `02_rqs.md` | `03_litreview.md` |
| 4 | `experiment-runner` | sonnet | `02_rqs.md` + `03_litreview.md` | `04_experiments/` |
| 5 | `paper-writer`      | sonnet | `01`~`04` 요약 | `05_draft/` |
| 6 | `reviewer`          | **opus** (최종 판단) | `05_draft/paper.md` | `06_reviews.md` |
| 7 | `rebuttal-drafter`  | sonnet | `05_draft/` + `06_reviews.md` | `07_rebuttal.md` |

## 라우팅 결정 규칙

1. `state.json`의 `stage` 필드로 다음 단계 결정. 없으면 stages_completed의 마지막 + 1.
2. 사용자가 "리뷰만 다시", "drafting만" 같은 부분 요청을 하면 그 단계 agent만 단독 호출.
3. 모델 escalation: **기본 sonnet**. opus 사용은 (a) `reviewer` 호출 시, (b) 사용자가 명시적으로 "opus로" 요청 시, (c) sonnet이 같은 입력에 두 번 연속 실패할 때만.

## 호출 시 prompt 컨벤션 (비용 절감)

- **이전 단계 산출물을 통째로 인용 금지**. 항상 파일 경로만 전달 → agent가 필요한 부분만 Read.
- 예: ❌ `01_problem.md 내용은 다음과 같습니다: <전문 붙여넣기>`
- 예: ✅ `workspace/<slug>/01_problem.md 를 읽고 RQ를 도출하세요. slug: <slug>`
- **slug**, **출력 경로**, **언어 선호도(있다면)** 만 명시. 나머지는 agent 본문에 이미 있음.

## 공통 규약
모든 agent는 작업 시작 시 `.claude/_common.md`를 Read하여 공통 출력 규약 / 환각 방지 / 보고 톤을 따른다.
