# Workshop Paper TODO

> **Scope**: 4-6 페이지 워크샵 페이퍼 (NeurIPS AI4Science / CVPR FGVC / ICCV Animal AI 등)
> **Target**: 약 1.5-2주 작업
> **Framing**: *"A diagnostic case study of hierarchical prompts in BioCLIP2 on CUB-200 (Aves)"* — single-domain case study로 claim 범위 좁힘
> **현재 상태**: BioCLIP2 ViT-L/14 + CUB-200 11,788장 실험 완료 (results.md 참조)

---

## Week 1 — 실험 보강 (3-5일)

### P0 — 반드시 (이거 안 하면 워크샵도 어려움)

#### TODO 1. OpenCLIP ViT-L/14 (laion-2b) baseline 추가
- [ ] `MODEL=openclip-vitl14 bash run_cub.sh` 실행
- [ ] 산출물 `results/cub200_openclip_vitl14/{exp1,exp2,exp3}.json` 확인
- [ ] latent probe z-score를 BioCLIP2(121.7)와 비교
- **이유**: 현재 "latent taxonomy z=121.7"이 BioCLIP2 *특이적*인지 일반 CLIP에도 있는지 알 수 없음. 비교 없이는 contribution 성립 불가.
- **검증 기대**: OpenCLIP의 z-score가 BioCLIP2보다 *작아야* "BioCLIP2가 잠재 구조를 *증폭*" 주장 가능. 비슷하면 "잠재 구조는 CLIP의 일반 성질"로 narrative 수정.
- **컴퓨트**: ~30분 GPU
- **의존**: 없음. 즉시 시작 가능.

#### TODO 2. Seed noise σ=1e-3 제거 → 실제 data bootstrap으로 교체
- [ ] [run_experiment.py:144](code/run_experiment.py#L144) 의 `noise = 1e-3 * ...` 제거
- [ ] 각 seed에서 11,788장 중 80% subsample (with replacement 또는 stratified by class) 적용
- [ ] Exp1, Exp3의 per-seed loop를 subsample 기반으로 재작성
- [ ] BioCLIP2 재실행, exp1/exp3 결과 갱신
- [ ] CI가 자연스럽게 넓어지는지(폭 1e-7 → 합리적 수준) 확인
- **이유**: 현재 σ=1e-3 가우시안만 더해서 seed std가 1e-7~1e-8 → Cohen's d ×10⁵ inflate. reviewer가 stats 무결성으로 즉시 fail.
- **컴퓨트**: 코드 수정 0.5일 + 재실행 ~1시간
- **의존**: TODO 1과 병렬 가능, 단 baseline들도 모두 같은 수정 적용 후 재실행 필요

#### TODO 3. Preservation ratio 폐기, raw drop으로 통일
- [ ] [protocol.md](protocol.md) 의 ρ 정의 섹션 제거 → "raw drop |M(C_x) − M(C0)| 비교"로 대체
- [ ] [plan.md](plan.md) Exp3 success criteria 재작성: "C3 raw drop ≥ 3× C2 raw drop, 95% CI 비중첩"
- [ ] [results.md](results.md) 의 ρ 표 제거, raw drop 표만 유지
- [ ] code/run_experiment.py의 preservation_ratio 계산 부분도 정리 (계산은 유지하되 default report에서 제외)
- **이유**: (C1−C0)<0 데이터셋 특이성 때문에 ratio가 무효. honest reporting 위해 metric 자체를 빼는 게 낫다.
- **컴퓨트**: 없음 (문서 정리)
- **의존**: TODO 1, 2 완료 후 numbers refresh

### P1 — 강력 권장 (있으면 페이퍼 강도 ↑)

#### TODO 4. BioCLIP (v1, ViT-B/16) baseline 추가
- [ ] `MODEL=bioclip bash run_cub.sh` 실행
- [ ] BioCLIP1 vs BioCLIP2의 latent z-score / silhouette 비교
- **이유**: BioCLIP1 (10M 학습) vs BioCLIP2 (214M)로 *scale 효과* 분리. "효과가 데이터 규모와 함께 강해지나" 추가 spot 결과.
- **컴퓨트**: ~20분 GPU

#### TODO 5. C5 처리 결정
- [ ] **옵션 A (워크샵 추천)**: C5를 main result에서 제거, appendix로 강등. "preliminary text-free probe — 본격 검증은 future work" 1줄 처리.
- [ ] **옵션 B**: LoRA r=8 + 20 epoch으로 강화. [models/proposed_textfree.py](code/models/proposed_textfree.py) 의 LinearAdapter를 LoRA로 교체.
- **컴퓨트**: A는 0.5일 문서 작업 / B는 코드 1일 + GPU 4-6시간
- **추천**: A. 워크샵 페이지 분량에서 B는 부담 큼.

---

## Week 2 — 페이퍼 작성 (5-7일)

`workspace/bioclip2/05_draft/` 에 섹션 골격 존재. 워크샵 분량으로 리라이트.

### P0 — 작성

#### TODO 6. Title / Abstract 재정의
- [ ] [00_abstract.md](../05_draft/00_abstract.md) 리라이트
- [ ] Title 후보: *"What Does Hierarchical Prompting Actually Do? A Diagnostic Case Study of BioCLIP2 on CUB-200"*
- [ ] Abstract에서 **명시적으로 single-domain (Aves)** 한정
- [ ] 3 contributions 정리:
  1. CUB-200 (Aves)에서 hierarchical prompt가 *species가 아닌 genus/family/order rank에서* 우세함
  2. 텍스트 없이도 BioCLIP2 image embedding에 분류학이 z=121.7로 잠재 (OpenCLIP 대비 X배 강함)
  3. 6-condition ablation으로 *구조 > 순서 > 어휘* 위계를 raw silhouette drop으로 정량화
- **소요**: 0.5일

#### TODO 7. Introduction
- [ ] [01_introduction.md](../05_draft/01_introduction.md) 리라이트
- [ ] 정보 채널 vs 조직자 가설 framing (01_problem.md 참조)
- [ ] single-domain limitation을 abstract와 intro에 1문장씩 명시
- [ ] 핵심 figure (Figure 1 또는 latent probe histogram) intro에 미리 노출
- **소요**: 1일

#### TODO 8. Method
- [ ] [03_method.md](../05_draft/03_method.md) 리라이트
- [ ] 6 conditions 정의 (실제 prompt 예시 포함, *Cardinalis cardinalis* 같은 1 종으로)
- [ ] silhouette, paired permutation, bootstrap CI 정의 (수식)
- [ ] 분량 1페이지 이내로 압축
- **소요**: 1일

#### TODO 9. Results + Figures
- [ ] [05_results.md](../05_draft/05_results.md) 리라이트
- [ ] **Figure 1**: rank-level silhouette bar chart (species/genus/family/order × C0/C1) — *"역전" 패턴* 시각화
- [ ] **Figure 2**: latent taxonomy probe histogram — 실제 silhouette vs 50 random permutation 분포. BioCLIP2 / OpenCLIP / (BioCLIP1) 3-panel
- [ ] **Figure 3**: 6-condition raw silhouette drop bar chart — C3 절벽 추락 강조
- [ ] **Table 1**: 모델 비교 (OpenCLIP / BioCLIP1 / BioCLIP2) × silhouette·latent z-score
- [ ] **Table 2**: 6 conditions raw means (intra_var, inter_margin, silhouette)
- [ ] 각 figure에 1문장 caption + 본문에서 1문단 해석
- **소요**: 2일 (figure 1일 + 본문 1일)

#### TODO 10. Discussion + Limitations + Future Work
- [ ] [06_discussion.md](../05_draft/06_discussion.md) 리라이트
- [ ] Limitation 섹션 (reviewer 선제 대응):
  - "Single domain (Aves); higher Linnaean ranks degenerate" → cross-domain future work
  - "Preservation ratio originally planned but discarded due to negative denominator; raw drops reported instead"
  - "C5 text-free baseline is a 5-epoch linear adapter — lower bound only"
  - "Latent probe z-score may include train-set memorization; truly-unseen-taxonomy validation as future work"
- [ ] Future work: 5-domain meta-analysis (RQ4 원래 plan), C5 LoRA 강화, BioCLIP 변종 비교
- **소요**: 1일

### P1 — 폴리시

#### TODO 11. Related Work
- [ ] [02_related_work.md](../05_draft/02_related_work.md) 리라이트
- [ ] [03_litreview.md](../03_litreview.md) 압축 → 1페이지
- [ ] 3 줄기: BioCLIP/BioCLIP2 + HAPrompts/CHiLS + hierarchical contrastive
- **소요**: 0.5일

#### TODO 12. Reproducibility appendix
- [ ] `code/run_cub.sh` 사용법 + seed 고정 + CUB-200 다운로드 + GBIF taxonomy 캐시 안내
- [ ] HuggingFace repo 링크 (BioCLIP/BioCLIP2)
- [ ] 컴퓨트 요구사항 (1 GPU, ~40분 × 모델 수)
- **소요**: 0.5일

---

## 의존성 그래프

```
TODO 1 (OpenCLIP)    ┐
TODO 2 (bootstrap)   ├─→ TODO 9 (Figure 1,2,3 데이터)
TODO 3 (ratio 폐기)  ┘
TODO 4 (BioCLIP1)    ─→ TODO 9 (Table 1 3-model 비교)
TODO 5 (C5 결정)     ─→ TODO 10 (limitation 문구)

TODO 6 (Title/Abstract) → TODO 7 (Intro) → TODO 8,9,10
                                            ↓
                                          TODO 11, 12
```

병렬 가능 작업:
- TODO 1·4 (다른 모델 평가) 동시 실행
- TODO 2·3 (코드 수정 + 문서 정리) 병렬
- TODO 7·8·11 (섹션별 작성) 병렬

---

## 컴퓨트 총합

| 항목 | 시간 |
|---|---|
| OpenCLIP ViT-L/14 (TODO 1) | ~30분 |
| BioCLIP1 ViT-B/16 (TODO 4) | ~20분 |
| BioCLIP2 재실행 (TODO 2 bootstrap 반영) | ~40분 |
| **합계** | **~1.5시간 GPU** |

---

## 진척 추적

### Week 1 (실험 보강)
- [ ] TODO 1 — OpenCLIP baseline
- [ ] TODO 2 — bootstrap 교체
- [ ] TODO 3 — ratio 폐기, 문서 정리
- [ ] TODO 4 — BioCLIP1 baseline
- [ ] TODO 5 — C5 처리 결정 (A 또는 B)

### Week 2 (작성)
- [ ] TODO 6 — Title/Abstract
- [ ] TODO 7 — Introduction
- [ ] TODO 8 — Method
- [ ] TODO 9 — Results + Figures
- [ ] TODO 10 — Discussion + Limitations
- [ ] TODO 11 — Related Work
- [ ] TODO 12 — Reproducibility appendix

---

## 결정해야 할 것 (작업 시작 전 확정)

1. **TODO 5 — C5는 옵션 A (appendix 강등) vs 옵션 B (LoRA 강화)**
   - 워크샵이면 A 추천. "text-free로도 가능"을 main claim에 넣으려면 B.

2. **Target venue 확정**
   - NeurIPS Workshops on AI4Science (보통 9-10월 deadline)
   - CVPR FGVC Workshop (보통 3-4월 deadline)
   - ICCV Workshop on Animal AI (보통 7-8월 deadline)
   - deadline 보고 우선순위 / page budget 조정 필요

3. **Latent probe leakage caveat 깊이**
   - BioCLIP2가 CUB-200 학습 셋에 포함되었을 가능성 (TreeOfLife에 CUB 포함) → z=121.7이 일부 memorization. 어떤 톤으로 limitation에 넣을지 (단순 future work 언급 vs 별도 robustness 분석).

---

## 결과물 체크리스트 (워크샵 제출 직전)

- [ ] `paper.md` 또는 LaTeX 컴파일된 PDF (4-6 페이지)
- [ ] 3 figures + 2 tables
- [ ] `code/` 깃 저장소 공개 (또는 anonymous repo)
- [ ] `results/cub200_{bioclip2, openclip_vitl14, bioclip_v1}/` JSON 산출물 첨부
- [ ] Reproducibility appendix
- [ ] Limitation 섹션이 reviewer 공격 surface를 선제 대응
