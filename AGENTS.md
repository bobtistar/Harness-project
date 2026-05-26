# Harness Project — Orchestrator Guide

이 저장소는 **키워드 → 논문 드래프트** 자동화 파이프라인(`/paper` 명령)을 호스팅합니다. 본 문서는 메인 세션(오케스트레이터)이 따라야 할 규칙만 다룹니다. agent별 상세는 `.Codex/agents/*.md`에 있고, 라우팅 카탈로그는 `.Codex/_index.md`, 공통 규약은 `.Codex/_common.md`에 있습니다.

## 1. 모델 정책

- **오케스트레이터(메인 세션)는 `sonnet` 기본**. 사용자가 `/model opus`로 명시적 전환하지 않는 한 sonnet 유지.
- **Agent 모델은 각 agent의 frontmatter `model:` 필드**가 결정. 호출자가 override하지 말 것.
- 현재 배정 — Opus: `reviewer` 1개 (비판적 다관점 판단). 나머지 6개: Sonnet.
- 모델 escalation은 동일 입력에 sonnet이 두 번 실패한 경우에만 사용자에게 한 줄 질문 후 진행.

## 2. 라우팅 규칙

- 어떤 작업이 들어와도 **agents/*.md 본문을 전수 Read하지 말 것**. 라우팅은 `.Codex/_index.md` 표 1회 Read로 충분.
- 한 turn에 agent **1개만** 호출. 두 단계가 필요하면 첫 단계 산출물을 확인한 뒤 다음 단계 호출.
- agent에 전달하는 prompt는 **파일 경로 + slug + (필요 시) 언어**만. 이전 산출물 본문을 prompt에 인용하지 말 것 — agent가 Read로 가져간다.

## 3. 명령

- `/paper <키워드> [auto]` — 전체 파이프라인. `.Codex/commands/paper.md` 참조.
- 부분 재실행: 사용자가 "리뷰만 다시" 등 명시하면 `_index.md`에서 해당 stage agent만 단독 호출.

## 4. 디렉토리

```
.Codex/
  _index.md          # 라우터 카탈로그 (자동 로드 X, paper.md가 1회 Read)
  _common.md         # 모든 agent의 공통 규약 (각 agent가 시작 시 1회 Read)
  agents/            # 7개 stage agent
  commands/paper.md  # 파이프라인 진입점
workspace/<slug>/    # 파이프라인 산출물 (커밋 대상 아님, .gitignore)
```

## 5. 비용 절감 체크리스트

요청이 들어왔을 때 자문:

- [ ] 메인 세션 모델이 sonnet인가? (Opus면 의도된 전환인지 확인)
- [ ] 라우팅에 `_index.md` 1회 Read만 사용했는가?
- [ ] agent prompt에 이전 산출물을 통째로 붙이지 않았는가?
- [ ] 같은 단계 agent를 한 turn에 2회 이상 호출하지 않았는가?
- [ ] reviewer 외에 opus 호출이 발생하지 않았는가?

## 6. 작업 톤

- 한국어 응답이 default (사용자 선호도). 코드 식별자/논문 인용은 원문.
- 보고는 짧게. 한 단계 = 한 줄. 메타 발언("분석을 진행하겠습니다") 금지.
- 사용자가 산출물 파일을 직접 열 수 있으므로 본문 요약 길게 늘어놓지 말 것.
