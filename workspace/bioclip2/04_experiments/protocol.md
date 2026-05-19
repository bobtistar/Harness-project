# Experiment Protocol

## 1. Datasets

### Primary: TreeOfLife evaluation subset (BioCLIP2 평가 분할)
- **Source**: `imageomics/TreeOfLife-10M` (BioCLIP), `imageomics/TreeOfLife-200M` (BioCLIP2) HuggingFace datasets.
- **License**: CC-BY-NC for the curated subset (iNaturalist+EOL+BHL aggregated); 개별 이미지의 원 라이선스 준수 필요.
- **Preprocessing**:
  - 종 ≥ 20 images filter
  - 7-rank Linnaean taxonomy 부착(이미 metadata에 존재)
  - resize 224×224, center crop, OpenCLIP normalize ([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
- **Split**: train / val / test = 70 / 15 / 15 (stratified by species). 본 연구는 **frozen** 평가이므로 test split만 사용. C5 fine-tune은 train split에서만.
- **통계**: 최소 1,000 species × 20+ images = ≥ 20,000 test images. Toy run은 8 species × 6 images = 48 images.

### Cross-domain (RQ4)
- Aves (Birds): NABirds + iNat21 birds subset
- Insecta: BIOSCAN-1M
- Plantae: PlantNet300K + iNat21 plants
- Fungi: iNat21 fungi
- Actinopterygii: FishNet

### Toy placeholder (코드 골격용)
- 8 placeholder species × 6 synthetic images (random RGB tensors with class-conditioned mean), 7-rank synthetic taxonomy.
- 목적: end-to-end 파이프라인 sanity check, 실제 가중치 없이 코드 검증.

## 2. Models

| Model | Source | Frozen? |
|-------|--------|---------|
| BioCLIP2 ViT-L/14 | `imageomics/bioclip-2` HF | Yes (frozen) |
| BioCLIP ViT-B/16 | `imageomics/bioclip` HF | Yes |
| OpenCLIP ViT-L/14 (laion2b_s32b_b82k) | `open_clip_torch` | Yes |
| OpenCLIP ViT-B/32 (laion2b_s34b_b79k) | `open_clip_torch` | Yes — toy run 용 |
| C5 adapter | 자체 구현 (LoRA r=8 on image encoder + linear head + hierarchical InfoNCE) | Light fine-tuned (5 epoch) |

## 3. Metrics (정확한 정의)

### Geometric metrics
- **Intra-class variance** (lower is better):
  $$\text{intra\_var} = \frac{1}{C} \sum_c \frac{1}{n_c(n_c-1)} \sum_{i \neq j \in c} (1 - \cos(z_i, z_j))$$
- **Inter-class margin** (higher is better):
  $$\text{inter\_margin} = \frac{1}{C} \sum_c \frac{\min_{c' \neq c} (1 - \cos(\mu_c, \mu_{c'}))}{\sigma_c}$$
  where $\mu_c$ = class centroid, $\sigma_c$ = within-class std.
- **Silhouette** (per sample, mean):
  $$s_i = \frac{b_i - a_i}{\max(a_i, b_i)}$$
  $a_i$ = mean cosine distance to same-class samples, $b_i$ = min over other classes of mean cosine distance.
- **Alignment** (Wang & Isola, lower is better):
  $$\text{align} = \mathbb{E}_{(x, x^+) \sim p_{\text{pos}}} \| f(x) - f(x^+) \|_2^2$$
- **Uniformity** (lower is better):
  $$\text{unif} = \log \mathbb{E}_{x, y \sim p_{\text{data}}} \exp(-2 \|f(x) - f(y)\|_2^2)$$
- **RankMe** (higher is better, effective rank):
  $$\text{RankMe}(Z) = \exp\left( -\sum_i p_i \log p_i \right), \quad p_i = \sigma_i / \sum_j \sigma_j + \epsilon$$
  where $\sigma_i$ = singular values of $Z$.

### Taxonomic metrics
- **kNN purity@k**: $\frac{1}{N}\sum_i \frac{|\{j \in \text{kNN}(i) : y_j = y_i\}|}{k}$
- **LCA depth**: For (predicted, gt) species, depth (in Linnaean tree from root) of their lowest common ancestor. Higher = better hierarchical agreement.
- **Hierarchy consistency error** (Bertinetto 2020): mean shortest path distance between predicted and true class in taxonomy tree.
- **Mutual information** I(cluster ; rank-label): rank-level taxonomic agreement, computed via `sklearn.metrics.mutual_info_score`.

### Effect-preservation ratio (Exp3 핵심)
$$\rho_x = \frac{M(C_x) - M(C_0)}{M(C_1) - M(C_0)}$$
for each metric M, condition x ∈ {C2, C3, C4, C5}. 95% CI via bootstrap.

## 4. Baselines

| Baseline | Hyperparameters | Source |
|----------|-----------------|--------|
| Flat prompt (C0) | `"a photo of {species}"` | CLIP standard |
| Hierarchical prompt (C1) | `"a photo of {kingdom} {phylum} {class} {order} {family} {genus} {species}"` (BioCLIP2 template) | Stevens et al. 2024 |
| Random-token hierarchy (C2) | `"a photo of tax0 tax1 tax2 tax3 tax4 tax5 {species_id_token}"` (7 tokens, 동일 길이) | 본 연구 자체 정의 |
| Shuffled hierarchy (C3) | C1의 상위 6 rank 라벨을 다른 종에서 임의 추출하여 교체. species만 정상 유지 | 본 연구 |
| Word-bag hierarchy (C4) | C1의 7개 라벨을 매 호출시 random permute | WaffleCLIP에서 영감 |
| Text-free InfoNCE (C5) | LoRA r=8, image-image hierarchical InfoNCE, 5 epoch, AdamW lr=1e-4 | Kokilepersaud 2024 손실 |
| WaffleCLIP descriptor | 공식 코드 hyperparameter | Roth et al. 2023 |
| CHiLS | 공식 코드 hyperparameter | Novack et al. 2023 |

## 5. Ablations

| Ablation | 목적 |
|----------|------|
| Prompt length matching | C0에 padding token 추가하여 C1과 토큰 수 동일 |
| Random taxonomy permutation control | 모든 클래스 라벨을 무작위 셔플 후 silhouette 재계산 |
| Image-only vs image+text | Modality gap의 영향 격리 |
| Per-rank evaluation | species → kingdom 7 rank 별도 평가 |
| Cross-model | OpenCLIP vs BioCLIP vs BioCLIP2 |

## 6. Statistical Tests

- **Paired permutation test** (B=1000): C0 vs C1 등 두 조건 비교. 동일 이미지의 두 prompt 결과를 paired로 본다.
- **Bootstrap 95% CI** (B=1000): 모든 metric, 효과 보존율 ρ.
- **Bonferroni 보정**: RQ1의 3개 지표 동시 검정 시 α = 0.01 / 3 = 0.0033.
- **Effect size**: Cohen's d (paired) for continuous metrics, Cliff's δ for non-parametric robustness check.
- **Meta-analysis (RQ4)**: random-effects pooled effect size, I² heterogeneity, between-domain CV.

## 7. Execution Environment

- **HW**: 본 코드 골격은 CPU에서도 toy run 가능. 전체 실험은 NVIDIA A100 40GB 1장 권장 (BioCLIP2 ViT-L/14 inference, batch 128 기준 약 8-12 시간 / 도메인).
- **SW**:
  - Python 3.10+ (확인: 3.12.7)
  - torch 2.7.1 + cu118 (확인)
  - open_clip_torch 3.2.0 (확인)
  - transformers 5.1.0 (확인)
  - scipy 1.13.1, numpy 1.26.4, scikit-learn (필요 시 설치), matplotlib 3.10.3 (확인)
  - 자세한 버전은 `code/requirements.txt`.
- **Reproducibility**: 모든 randomness는 `numpy.random.default_rng(seed)` 및 `torch.manual_seed(seed)`로 고정. CuDNN deterministic 설정 활성화.

## 8. Data Ethics & Licensing

- iNaturalist 이미지는 사용자 업로드, 개별 라이선스 (CC BY, CC BY-NC 등) 다양. 본 연구는 비상업·연구 목적이며 BioCLIP/BioCLIP2 라이선스(Imageomics) 준수.
- PII: 사용자 ID 등은 metadata에서 제거하여 사용.
- BIOSCAN, FishNet 등은 각 dataset 라이선스 (대개 CC BY) 준수.
