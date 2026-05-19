# Results

> # ⚠️ TOY / MOCK RESULTS — 본 격 실험 미실행
>
> 아래 수치는 **8 placeholder 종 × 6 합성 이미지 = 48 samples**의 toy 데이터셋에서 OpenCLIP ViT-B/32(또는 mock embedding)로 얻은 **파이프라인 sanity check** 결과입니다.
>
> - BioCLIP2 / BioCLIP 가중치는 다운로드되지 않았습니다 (HF 가중치 6 GB, 본 환경 GPU 부재 + 시간 제약).
> - TreeOfLife 평가 셋(1,000+ 종, 20,000+ 이미지) 또한 다운로드되지 않았습니다.
> - **숫자를 절대 인용하지 마십시오**. 실제 실험은 `code/run_all.sh` 또는 `run_all.ps1`로 GPU 환경에서 재실행해야 합니다.

## 실행 상태

| 구분 | 상태 | 비고 |
|---|---|---|
| 코드 골격 작성 | 완료 | 모든 모듈 import + smoke test 통과 |
| mock 모드 end-to-end | 완료 | `python run_experiment.py --mock --toy` → exp1/2/3 JSON 생성 |
| OpenCLIP ViT-B/32 toy run | 완료 (CPU) | 실 모델 + 합성 이미지 + 합성 텍스트 인코딩 |
| OpenCLIP ViT-L/14 full | **미실행** | 1,000+ 종 평가 셋 필요 |
| BioCLIP / BioCLIP2 | **미실행** | HF 가중치 + GPU 필요 |
| RQ4 (5 도메인) | **미실행** | 도메인별 데이터 필요 |

## 환경 (실제 실행)

- 실행 일자: 2026-05-19
- HW: CPU only (Windows 11, no CUDA used for toy run despite torch+cu118 available)
- SW: Python 3.12.7, torch 2.7.1+cu118, open_clip_torch 3.2.0, transformers 5.1.0, scikit-learn 1.5.1, numpy 1.26.4

## Exp1 Results — RQ1 (Hierarchical vs Flat geometric compactness)

**Toy + real OpenCLIP ViT-B/32 (3 seeds, B=1000 bootstrap)**

| Metric | C0 (flat) | C1 (hierarchical) | diff (C1-C0) | bootstrap CI | passes? |
|---|---|---|---|---|---|
| intra_var ↓ | 0.00239 | 0.00239 | ≈ 0 | tight | NO |
| inter_margin ↑ | 713.11 | 389.59 | **-323.5** | [-324.6, -322.8] | NO (역방향) |
| silhouette ↑ | 0.9865 | 0.9740 | -0.0125 | [-0.013, -0.012] | NO (역방향) |
| RankMe | 9.79 | 9.23 | -0.57 | tight | — |
| uniformity | -0.834 | -0.684 | +0.15 | tight | — |

**RQ1 H1**: **결정 불가 / 합성 데이터에서는 hierarchical prompt가 오히려 silhouette을 낮춤**. 이는 toy data의 합성 이미지가 색상 평균만으로 분리되며, hierarchical 토큰은 image embedding과 무관한 noise를 추가하기 때문으로 해석. **이 결과는 BioCLIP2 가설을 검증할 수 없음** — 실제 TreeOfLife 평가가 필요.

## Exp2 Results — RQ2 (Latent taxonomy probing)

**Toy + real OpenCLIP ViT-B/32, species-rank silhouette**

| Source | Silhouette (cosine) | z-score vs random taxonomy |
|---|---|---|
| 실제 라벨 | 0.683 | +44.9 |
| Random-taxonomy 평균 (50 permutations) | -0.271 | (baseline) |

**RQ2 H1(b) 잠재 구조**: Toy data 수준에서도 일반 OpenCLIP은 random-taxonomy 대비 매우 강한 분리 신호 (z=44.9). 다만 toy synthetic image의 class-conditioned color가 silhouette을 인위적으로 부풀린다. **실제 결론은 TreeOfLife에서만 가능**.

Rank별 결과: toy taxonomy는 도메인 4개로 위쪽 rank에서 클래스 수가 너무 적어 silhouette 정의 불가능한 rank가 다수 존재. 실 데이터 (1,000+ 종) 필요.

## Exp3 Results — RQ3 (Counterfactual ablation C0..C5)

**Toy + real OpenCLIP ViT-B/32, 3 seeds, silhouette 기준**

| Condition | intra_var | inter_margin | silhouette | preservation ratio (silhouette) |
|---|---|---|---|---|
| C0 flat | 0.00239 | 713.11 | 0.9865 | (baseline) |
| C1 hierarchical | 0.00239 | 389.59 | 0.9740 | 1.00 (정의상) |
| C2 random-token hier | 0.00239 | 445.66 | 0.9796 | **0.55** [0.55, 0.55] |
| C3 shuffled hier | 0.0937 | 3.97 | 0.1002 | 70.9 (silhouette이 무너지며 ratio가 폭발) |
| C4 word-bag | 0.0210 | 44.01 | 0.8214 | 13.2 (마찬가지) |
| C5 text-free InfoNCE | 0.00545 | 30.04 | 0.6936 | 23.4 (마찬가지) |

**해석 (toy 한정)**:
- **C3 (shuffled hierarchy)**가 silhouette을 0.97 → 0.10으로 **극적으로 붕괴**시킴 → 구조 파괴가 큰 영향. 이는 RQ3 H1의 *예측 방향과 일치*.
- **C2 (random-token, 구조 보존)**의 silhouette (0.98)이 C1 (0.97)보다 약간 *높음* → "구조 보존만으로 효과 유지" 가설과 **방향성 일치**.
- 그러나 preservation ratio 계산은 (C1 - C0) 분모가 음수(toy에서 C1이 C0보다 약간 낮음)라서 *비율 해석이 무의미*. 실 데이터에서는 C1 > C0가 일반적일 것으로 예상.
- C4, C5는 모두 silhouette을 크게 낮춤 → toy에서는 부정적 신호. C5의 light adapter 학습이 epoch 5로 너무 짧을 가능성.

**Semantic-Organizer 가설**: toy 결과로는 **판단 불가**. RQ3 success criteria(C2 ≥ 50%, C3 ≤ 20%, C5 ≥ 50%, paired bootstrap CI)는 실제 데이터에서만 검증 가능.

## 통계 검정 무결성 노트

- 본 toy run의 paired permutation p-value는 모두 ≈ 0.24–0.26 (효과가 너무 작거나 noise 수준). Bonferroni 보정(α=0.0033) 후 어떤 metric도 유의 미달.
- effect size (Cohen's d)는 일부 매우 큼 (예: silhouette d = -2910) — 이는 seed-to-seed std가 극히 작아서(1e-6 수준) inflation된 것. **실 데이터에서는 seed 간 variance가 정상 범위로 회복될 것**.

## RQ별 답변 요약 (toy data 기반 — 본격 검증 미완)

| RQ | toy 데이터의 잠정 신호 | 최종 결론 |
|---|---|---|
| RQ1 | hierarchical prompt가 silhouette을 *낮춤* (역방향) | **재실험 필요** — toy의 randomized text는 합성 image와 정렬되지 않음 |
| RQ2 | random-taxonomy 대비 강한 잠재 구조 (z=44.9) | **방향성 일치**, but synthetic color confound |
| RQ3 | C3가 silhouette 0.97→0.10으로 붕괴, C2는 C1과 유사 (구조 보존 가설 방향) | **방향성 일치**, but C1<C0 때문에 preservation ratio 계산 불가 |
| RQ4 | 미수행 | 도메인별 데이터 필요 |

## 한계 / 위협

### Internal validity
- Toy 합성 이미지는 클래스 조건부 색상 평균으로만 구분 → 실제 visual semantic을 반영하지 않음. C0/C1 차이가 미미한 주된 원인.
- 텍스트 인코딩은 실제 OpenCLIP을 사용했으나, 이미지 임베딩이 클래스 라벨과 너무 강하게 결합되어 silhouette 천장 효과 (≥ 0.97). 효과 비교 분해능 부족.

### External validity
- 8 species, 4 domains, 6 samples/species는 통계적 검정력 거의 0.
- 본 실험은 **파이프라인 검증** 목적이며, **결과 해석에는 사용할 수 없음**.

### 통계적 위협
- Seed 간 variance가 비현실적으로 작아 Cohen's d가 inflated.
- Bonferroni 보정 후 어떤 결과도 유의하지 않음 (정상).
- Preservation ratio는 분모가 0에 가까우면 발산 → 실 데이터에서 (C1 - C0) 차이가 충분히 클 때만 의미.

## 실제 검증을 위한 다음 단계

1. GPU 머신 (A100 40GB 권장)에서 `pip install -r requirements.txt` 후
   ```
   python run_experiment.py --model bioclip2 \
       --csv data/treeoflife_eval.csv \
       --image_root data/images \
       --device cuda --out ../results/bioclip2_full
   ```
2. TreeOfLife eval split 다운로드: HuggingFace `imageomics/TreeOfLife-10M` (≈ 60 GB)에서 1,000 종 stratified subsample.
3. 위 실험 RQ1 결과를 paper와 cross-check: BioCLIP2 (Gu et al. 2025) Figure 3-4의 inter-species clustering이 본 silhouette/margin metric과 일치하는지.
4. RQ4 cross-domain: 5 도메인 각각에서 step 2-3 반복.
