# Experiment Protocol

## 1. Datasets

### Primary (실제 실행): CUB-200-2011

- **Source**: Wah et al. 2011, "The Caltech-UCSD Birds-200-2011 Dataset" (Caltech).
- **License**: CUB-200-2011 학술용 라이선스 (research only).
- **Scale**: 200 species (모두 Aves), 총 **11,788 images**, 평균 ~59 images/species.
- **Taxonomy 부착**: 본 연구에서 `cub200_build_taxonomy.py`로 GBIF API를 통해 200종에 7-rank Linnaean taxonomy 자동 매핑 (kingdom/phylum/class/order/family/genus/species). CUB-200은 모두 Animalia/Chordata/Aves에 속하므로 상위 3개 rank는 단일값 (degenerate).
- **Preprocessing**:
  - 이미지 resize 224×224, center crop, OpenCLIP normalize.
  - 7-rank label 부착 (GBIF 캐시 `cub_gbif_cache.json`, 1회 호출 후 영구 캐시).
- **Split**: BioCLIP2는 **frozen evaluation**이므로 train/val 분리 없이 전체 11,788 images를 평가에 사용. C5 adapter도 동일 데이터에서 5 epoch fine-tune (lower-bound 추정용).

### Future work: 더 큰 범위
- TreeOfLife-10M / TreeOfLife-200M (BioCLIP/BioCLIP2 학습 셋), iNat21 cross-domain — 본 실행에서는 데이터·컴퓨트 부족으로 미사용. plan.md Exp4 참조.

## 2. Models

| Model | Source | Frozen? | 실제 사용? |
|-------|--------|---------|-----------|
| **BioCLIP2 ViT-L/14** | `hf:imageomics/bioclip-2` (embedding_dim=768) | Yes (frozen) | **Yes** |
| BioCLIP ViT-B/16 | `imageomics/bioclip` HF | Yes | No (본 실행은 BioCLIP2만) |
| OpenCLIP ViT-L/14 (laion2b_s32b_b82k) | `open_clip_torch` | Yes | No |
| C5 adapter | 자체 구현 (`models/proposed_textfree.py`): linear D→D + L2 norm + LCA-weighted InfoNCE | Light fine-tuned (5 epoch, lr=1e-4 AdamW) | **Yes** |

## 3. Metrics (정확한 정의)

### Geometric metrics
- **Intra-class variance** (lower is better):
  $$\text{intra\_var} = \frac{1}{C} \sum_c \frac{1}{n_c(n_c-1)} \sum_{i \neq j \in c} (1 - \cos(z_i, z_j))$$
- **Inter-class margin** (higher is better):
  $$\text{inter\_margin} = \frac{1}{C} \sum_c \frac{\min_{c' \neq c} (1 - \cos(\mu_c, \mu_{c'}))}{\sigma_c}$$
- **Silhouette** (per sample, mean, cosine):
  $$s_i = \frac{b_i - a_i}{\max(a_i, b_i)}$$
- **Uniformity** (Wang & Isola, lower is better):
  $$\text{unif} = \log \mathbb{E}_{x,y} \exp(-2 \|f(x) - f(y)\|_2^2)$$
- **RankMe** (effective rank): entropy-exp of normalized singular values of Z.

### Taxonomic metrics
- **kNN purity@10**: $\frac{1}{N}\sum_i |\{j \in \text{kNN}(i) : y_j = y_i\}| / 10$.

### Effect-preservation ratio (Exp3 핵심)
$$\rho_x = \frac{M(C_x) - M(C_0)}{M(C_1) - M(C_0)}$$
seed별 paired ratio → bootstrap 95% CI (B=1000). **분모 (C1−C0) 부호가 음수일 때는 비율 해석이 뒤집힘 — 본 실행 결과에서 이 상황이 발생함 (results.md 참조).**

### C5 Loss: LCA-weighted hierarchical InfoNCE
$$L_i = -\sum_j \frac{w_{ij}}{\sum_k w_{ik}} \log \frac{\exp(s_{ij}/\tau)}{\sum_k \exp(s_{ik}/\tau)}, \quad w_{ij} = \mathrm{LCA\_depth}(y_i, y_j) / 7$$
($\tau = 0.07$, 자기 자신은 weight=0)

## 4. Baselines & Conditions

| Condition | 정의 |
|----------|------|
| **C0** flat | `"a photo of {species}"` |
| **C1** hierarchical | `"a photo of {kingdom} {phylum} {class} {order} {family} {genus} {species}"` (BioCLIP2 template) |
| **C2** random-token hierarchy | `"a photo of tax0 tax1 tax2 tax3 tax4 tax5 {species_id_token}"` (7 tokens, 동일 길이, 구조 보존) |
| **C3** shuffled hierarchy | C1의 상위 6 rank 라벨을 다른 종에서 임의 추출하여 교체 (구조 파괴) |
| **C4** word-bag hierarchy | C1의 7개 라벨을 매 호출마다 random permute (순서 제거, 어휘 보존) |
| **C5** text-free InfoNCE | linear adapter + LCA-weighted hierarchical InfoNCE, 5 epoch, lr=1e-4 AdamW |

## 5. Ablations

| Ablation | 목적 | 본 실행 |
|----------|------|--------|
| Prompt structure (C2 vs C3) | 구조 보존 vs 파괴 효과 분리 | **완료** |
| Order vs vocabulary (C1 vs C4) | 어휘만 보존, 순서 제거 효과 | **완료** |
| Text vs no-text (C1 vs C5) | 텍스트 encoder의 역할 격리 | **완료** |
| Per-rank evaluation | species…order rank별 효과 분해 | **완료** |
| Random-taxonomy permutation | 잠재 구조 유의성 검정 (50 permutations) | **완료** |

## 6. Statistical Tests

- **Paired permutation test** (B=1000): C0 vs C1 비교, 동일 image embedding의 두 prompt 결과를 paired로 본다.
- **Bootstrap 95% CI** (B=1000): preservation ratio ρ.
- **Bonferroni 보정**: Exp1의 6개 metric 동시 검정 시 α = 0.01 / 6 ≈ 0.00167.
- **Effect size**: Cohen's d (paired). **본 실행에서 seed-noise σ=1e-3이 작아 std가 비현실적으로 작고 Cohen's d가 inflate되는 현상이 발생 — results.md에서 raw difference 위주로 해석.**
- **Latent taxonomy probe**: B=50 random label permutations → z-score.

## 7. Execution Environment (실제 실행)

- **실행 일자**: 2026-05-19, 14:52:14 → 15:30:09 (총 ≈ 38분)
- **HW**: NVIDIA CUDA GPU 1장 (device=cuda)
- **SW**:
  - Python 3.12.7
  - torch 2.7.1 + cu118
  - open_clip_torch 3.2.0
  - transformers 5.1.0
  - scipy 1.13.1, numpy 1.26.4, scikit-learn 1.5.1, matplotlib 3.10.3
  - 자세한 버전은 `code/requirements.txt`
- **모델 가중치**: HuggingFace `imageomics/bioclip-2` (ViT-L/14, embedding_dim=768)
- **배치 설정**: batch_size=128, num_workers=4, AMP(float16) 활성화
- **재현성**: `numpy.random.default_rng(seed)`, `torch.manual_seed(seed)`, CuDNN deterministic 활성화. 5 seeds × 6 conditions.
- **임베딩 캐시**: `results/cub200_bioclip2/img_emb.npz` — 11,788×768 image embedding 1회 인코딩 후 재사용.

### 단계별 소요 시간 (run_log.txt 기반)
| Step | 시각 | 누적 |
|---|---|---|
| 모델 로드 (HF download + cuda 전송) | 14:52:15 → 14:52:26 | 11초 |
| 11,788 images 인코딩 | 14:52:26 → 14:53:09 | 43초 |
| Exp1 (RQ1, 5 seeds × 2 cond) | 14:53:17 → 14:58:26 | 5분 9초 |
| Exp2 (rank-level + permutation probe) | 14:58:26 → 15:04:09 | 5분 43초 |
| Exp3 (5 seeds × 6 cond, C5 fine-tune 포함) | 15:04:09 → 15:30:09 | **26분** |

## 8. Data Ethics & Licensing

- CUB-200-2011: 학술 연구용 라이선스, 비상업적 사용.
- GBIF taxonomy lookup: CC0 (퍼블릭 도메인).
- BioCLIP2: Imageomics 라이선스 (학술 비상업).
- PII 없음 (이미지에 사용자/촬영자 메타데이터 미포함).
