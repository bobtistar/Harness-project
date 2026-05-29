# experiments.md — Experiments 섹션 작성 템플릿 (재사용)

> 용도: AI/ML 논문 Experiments 섹션(보통 Section V) 작성 시 에이전트에 주입하는 범용 가이드.
> **최우선 규칙: 실제 수치를 모르면 절대 지어내지 말 것 → placeholder.**

## PLACEHOLDER 규약 (필수)
- 정량 결과: `[[RESULT: <metric>, model=<...>, dataset=<...>, setting=<...>]]`
- 미정 설정값: `[[FILL: <무엇>]]` (예: backbone, dataset 통계, hardware)
- 그림: `[[FIGURE: <무엇을 보일지>]]`
→ 표/그림은 골격만 만들고 셀은 placeholder. 저자가 실제 값으로 채운다.

## 필수 구조
1. **Setup**:
   - Datasets (이름 + 핵심 통계)
   - Models / Baselines (제안법 + 비교군; backbone·출처)
   - Metrics (정의 + 무엇을 측정하는지)
   - Implementation (hardware, 주요 hyperparam, seed 수, 학습/zero-shot 여부)
2. **Main Results**: 핵심 질문에 답하는 표 1개 + 해석. 각 결과 → 어떤 claim/RQ에 대응하는지 연결.
3. **Ablation**: 구성요소/설계 선택의 기여 분리. (B형 논문은 조건별 비교가 main에 해당할 수 있음)
4. **Analysis / Qualitative**: 시각화·케이스·추가 진단. 정성 주장은 신중하게.
5. **(선택) Limitations**: abstract의 boundary conditions와 모순 없게.

## HARD RULES
- **수치 날조 금지**: 모르면 `[[RESULT]]`. 그럴듯한 숫자 생성 절대 금지.
- **claim 매핑**: 각 결과 문장은 검증하려는 주장/가설에 연결. "결과 나열"만 하지 말 것.
- **통계 정직성**: 평균±표준편차/seed 수 보고. CI·유의성 검정은 **실제 수행한 경우만** 언급.
- **조건부 서술**: 수치가 placeholder인 동안 해석은 "If A > B by margin, this indicates ..." 식 조건부로.
- **표/그림 채번**: 논문 기준으로 일관되게(앞 섹션 번호 이어서).

## CHECK
- [ ] 모든 정량 셀이 placeholder (지어낸 숫자 0개)
- [ ] Setup 4요소(data/model/metric/impl) 모두 기술
- [ ] 각 결과가 claim/RQ에 매핑됨
- [ ] 통계 보고가 실제 수행 범위와 일치
- [ ] Limitations가 abstract와 모순 없음
