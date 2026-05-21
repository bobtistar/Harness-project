---
description: 키워드로부터 논문 작성 파이프라인을 자동 실행 (문제정의→RQ→문헌→실험→작성→리뷰→리부탈)
argument-hint: <키워드 또는 주제> [auto]
---

# Paper Writing Pipeline

입력: `$ARGUMENTS`

마지막 토큰이 `auto`면 모든 단계를 사용자 확인 없이 연속 실행. 아니면 각 단계 완료 시 한 줄로 "다음 진행?"을 묻고 응답을 기다림.

---

## 라우팅 정책 (비용 최적화)

**시작 전 1회만** `.claude/_index.md`를 Read. 이 파일이 단계→agent→모델 매핑의 단일 source of truth. 본 명령은 다음을 보장:

1. 매 단계에서 agent **1개만** 호출 (병렬 호출 금지).
2. 이전 단계 산출물을 prompt에 **통째로 인용하지 않음**. 파일 경로만 전달.
3. Agent 호출 시 `model` 필드를 명시하지 않음 — agent frontmatter의 기본 모델을 사용 (`_index.md` 표 참조).
4. 사용자가 명시적으로 "opus로" 요청하지 않는 한 모델 escalation 금지.

---

## 실행 절차

### Step 0 — 워크스페이스 준비

1. 입력에서 `auto` 플래그를 분리해 키워드만 추출.
2. 키워드 → 슬러그 변환: 소문자, 영숫자와 하이픈만, 공백→하이픈, 한글이면 영어로 음차/번역 (예: "확산모델 시계열" → `diffusion-timeseries`).
3. `workspace/<slug>/` 디렉토리 생성. 이미 존재하면 사용자에게 "기존 작업 이어서 진행할까요, 새로 시작할까요?" 한 줄로 질문.
4. `workspace/<slug>/state.json` 작성:
   ```json
   {
     "keyword": "<원본 키워드>",
     "slug": "<slug>",
     "stage": "0_init",
     "stages_completed": [],
     "auto_mode": <true|false>,
     "created_at": "<ISO 8601 현재 시각>"
   }
   ```
5. `.claude/_index.md`를 1회 Read하여 라우팅 표를 확보.
6. 사용자에게 한 줄 보고: `워크스페이스 준비됨: workspace/<slug>/ (mode: auto|interactive)`

---

### 공통 호출 패턴

각 단계는 다음을 따른다:

1. Agent 도구 호출:
   - `subagent_type`: `_index.md`의 해당 stage agent
   - `prompt`: **파일 경로 + slug + (필요 시) 언어**만. 산출물 본문 인용 금지.
2. 완료 후 `state.json` 업데이트: `stage`, `stages_completed`.
3. 한 줄 보고: `[N/7] <단계명> 완료 → <산출물 경로>`.
4. interactive 모드면 다음 단계 진행 여부 한 줄 질문.

---

### Step 1 — 문제 정의

- agent: `problem-definer`
- prompt: `키워드 '<원본>' 에 대해 문제 정의 작성. slug: <slug>. 출력: workspace/<slug>/01_problem.md`

### Step 2 — RQ 수립

- agent: `rq-formulator`
- prompt: `workspace/<slug>/01_problem.md 를 읽고 RQ 도출. 출력: workspace/<slug>/02_rqs.md`

### Step 3 — 문헌 조사

- agent: `lit-reviewer`
- prompt: `workspace/<slug>/01_problem.md 와 02_rqs.md 를 읽고 문헌 조사. WebSearch 적극 활용. 출력: workspace/<slug>/03_litreview.md`

### Step 4 — 실험

- agent: `experiment-runner`
- prompt: `workspace/<slug>/02_rqs.md 와 03_litreview.md 기반으로 실험 설계·코드. 가능하면 실행. 출력: workspace/<slug>/04_experiments/`

### Step 5 — 작성

- agent: `paper-writer`
- prompt: `workspace/<slug>/ 의 01~04 산출물을 종합해 논문 드래프트 작성. 출력: workspace/<slug>/05_draft/`
- 주의: paper-writer는 영어 default. 한국어 원하면 prompt에 "언어: 한국어" 추가.

### Step 6 — 리뷰 (Opus)

- agent: `reviewer` (opus 모델 — `_index.md` 참조)
- prompt: `workspace/<slug>/05_draft/paper.md 와 04_experiments/results.md 를 읽고 3 리뷰어 + 메타리뷰. 출력: workspace/<slug>/06_reviews.md`

### Step 7 — 리부탈

- agent: `rebuttal-drafter`
- prompt: `workspace/<slug>/05_draft/, 06_reviews.md, 04_experiments/ 를 읽고 리부탈. 출력: workspace/<slug>/07_rebuttal.md`

완료 시 `state.json`: `stage = "7_done"`. 최종 보고:
```
[7/7] 파이프라인 완료
산출물:
  workspace/<slug>/01_problem.md
  workspace/<slug>/02_rqs.md
  workspace/<slug>/03_litreview.md
  workspace/<slug>/04_experiments/
  workspace/<slug>/05_draft/
  workspace/<slug>/06_reviews.md
  workspace/<slug>/07_rebuttal.md
```

---

## 오류 처리

- 어떤 단계 agent가 실패하면 `state.json`에 `last_error` 기록, 즉시 사용자 보고. 자동 재시도 금지.
- 산출물이 50줄 미만이면 한 줄 경고 후 진행 여부 확인.
- 이전 단계 산출물이 없으면 그 단계부터 다시 실행.
- sonnet agent가 두 번 실패하면 사용자에게 "opus 재시도?" 한 줄 질문.

## 진행 보고 톤
한 단계 = 한 줄. 길게 설명하지 말 것. 사용자는 산출물 파일을 직접 열어볼 수 있음.
