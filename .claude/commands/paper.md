---
description: 키워드로부터 논문 작성 파이프라인을 자동 실행 (문제정의→RQ→문헌→실험→작성→리뷰→리부탈)
argument-hint: <키워드 또는 주제> [auto]
---

# Paper Writing Pipeline

입력: `$ARGUMENTS`

마지막 토큰이 `auto`면 모든 단계를 사용자 확인 없이 연속 실행. 아니면 각 단계 완료 시 한 줄로 "다음 진행?"을 묻고 응답을 기다림.

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
5. 사용자에게 한 줄 보고: `워크스페이스 준비됨: workspace/<slug>/ (mode: auto|interactive)`

---

### Step 1 — 문제 정의

Agent 도구 호출:
- `subagent_type`: `problem-definer`
- `prompt`: "키워드 '<원본>'에 대해 문제 정의를 작성하세요. 결과를 `workspace/<slug>/01_problem.md`에 저장하세요. 슬러그: `<slug>`"

완료 후:
- `state.json` 업데이트: `stage = "1_problem"`, `stages_completed += ["1_problem"]`
- 한 줄 보고: `[1/7] 문제 정의 완료 → workspace/<slug>/01_problem.md`
- interactive 모드면 "다음(RQ 수립) 진행?" 질문.

---

### Step 2 — RQ 수립

Agent: `rq-formulator`
- prompt: "`workspace/<slug>/01_problem.md`를 읽고 RQ들을 도출, `workspace/<slug>/02_rqs.md`에 저장하세요."

state 업데이트, `[2/7] RQ 수립 완료` 보고, 게이트.

---

### Step 3 — 문헌 조사

Agent: `lit-reviewer`
- prompt: "`workspace/<slug>/01_problem.md`와 `02_rqs.md`를 읽고 문헌 조사를 수행, `03_litreview.md`에 저장. WebSearch를 적극 활용하세요."

state 업데이트, `[3/7] 문헌 조사 완료` 보고, 게이트.

---

### Step 4 — 실험

Agent: `experiment-runner`
- prompt: "`workspace/<slug>/02_rqs.md`와 `03_litreview.md`를 기반으로 실험을 설계하고 코드 골격을 작성, `workspace/<slug>/04_experiments/`에 저장. 환경이 허락하면 실행하세요."

state 업데이트, `[4/7] 실험 완료` 보고, 게이트.

---

### Step 5 — 작성

Agent: `paper-writer`
- prompt: "`workspace/<slug>/`의 01~04 산출물을 종합하여 논문 드래프트를 `workspace/<slug>/05_draft/`에 작성하세요."

state 업데이트, `[5/7] 드래프트 완료` 보고, 게이트.

---

### Step 6 — 리뷰

Agent: `reviewer`
- prompt: "`workspace/<slug>/05_draft/paper.md`(또는 섹션들)에 대해 3명의 가상 리뷰어 + 메타리뷰를 작성, `06_reviews.md`에 저장하세요."

state 업데이트, `[6/7] 리뷰 완료` 보고, 게이트.

---

### Step 7 — 리부탈

Agent: `rebuttal-drafter`
- prompt: "`workspace/<slug>/05_draft/`와 `06_reviews.md`를 읽고 리부탈을 작성, `07_rebuttal.md`에 저장하세요."

state 업데이트: `stage = "7_done"`. 최종 보고:
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

- 어떤 단계 에이전트가 실패하면 `state.json`에 `last_error`를 기록하고 즉시 사용자에게 보고. 자동 재시도 금지.
- 산출물이 비었거나 빈약하면 (예: 50줄 미만) 사용자에게 한 줄 경고 후 진행 여부 확인.
- 이전 단계 산출물이 없으면 그 단계부터 다시 실행.

## 진행 보고 톤
한 단계 = 한 줄. 길게 설명하지 말 것. 사용자는 산출물 파일을 직접 열어볼 수 있음.
