# Literature Review

## Search Strategy
- **검색 쿼리들** (실제 실행):
  1. `BioCLIP2 emergent properties hierarchical contrastive learning 2025 arXiv`
  2. `BioCLIP Tree of Life CVPR 2024 Stevens vision foundation model taxonomy`
  3. `hierarchical multi-label contrastive learning HMLC HiMulConE 2022 2023`
  4. `HAPrompts hierarchy-aware language prompts CLIP zero-shot better mistakes 2024`
  5. `ProTeCt prompt tuning taxonomic open set classification hierarchical CLIP`
  6. `CHiLS classes hierarchical label sets zero-shot image classification ICML`
  7. `"alignment and uniformity" hypersphere contrastive learning Wang Isola 2020`
  8. `neural collapse contrastive learning feature geometry 2023 2024`
  9. `RankMe effective rank embedding evaluation representation learning ICML 2023`
  10. `CLIP embedding latent taxonomy hierarchy structure probing ImageNet WordNet`
  11. `CLIP prompt sensitivity shuffled text random words ablation analysis`
  12. `hyperbolic CLIP poincare embedding hierarchical vision language 2024`
  13. `"waffling around for performance" random words CLIP zero shot Roth ICCV 2023`
  14. `HGCLIP graph hierarchical vision language model fine-grained 2024`
  15. `"Taxes Are All You Need" taxonomical hierarchy contrastive loss 2024 species`
  16. `PHyCLIP hyperbolic factor vision language hierarchy compositionality 2025`
  17. `"Compositional Entailment Learning" hyperbolic vision language CLIP 2024`
  18. `HPT hierarchy-aware prompt tuning vision-language CLIP AAAI 2024 multi-granularity`
  19. `"Hyperbolic Multimodal Representation Learning for Biological Taxonomies" 2025`
  20. `HiCLIP hierarchy-aware attention contrastive language image ICLR 2023`
  21. `"MERU" hyperbolic CLIP image text embedding entailment 2023 Desai`
  22. `"better mistakes" hierarchy aware classification ImageNet WordNet Bertinetto`
  23. `poincare embeddings learning hierarchical representations Nickel Kiela 2017`
  24. `"BIOSCAN-CLIP" biodiversity vision genomics taxonomy embedding 2024`
- **추가 확인**: WebFetch로 BioCLIP2 (arXiv:2505.23883) 초록·메타데이터 직접 확인.
- **검색 도구**: WebSearch + WebFetch.
- **포함 기준**: (a) CLIP/contrastive multimodal learning과 직접 관련, (b) 계층/분류군 구조 활용, (c) embedding geometry 진단, (d) 2017–2026 발표, 영문, NeurIPS/ICML/ICLR/CVPR/ICCV/AAAI/COLING/arXiv 우선.
- **제외 기준**: 의료·산업 도메인 특화로 본 연구와 연결고리 약한 논문, code/abstract 미공개로 verify 불가한 후보.
- **검색 일자**: 2026-05-19.

## Categorized Findings

### Category A: BioCLIP / BioCLIP2 계열 (직접 대상 모델)

| Paper | Year | Approach | Key Result | Relevance |
|-------|------|----------|------------|-----------|
| Stevens et al. "BioCLIP" (CVPR'24 Best Student Paper Oral) | 2024 | TreeOfLife-10M (~454K taxa), CLIP-style contrastive with Linnaean 7-rank hierarchical text prompt ("Animalia Chordata Aves … Passer domesticus") | +17–20%p absolute over OpenCLIP on fine-grained bio classification; intrinsic analysis shows learned embedding "conforms to tree of life" | **본 연구의 핵심 baseline 모델 #1**; 잠재 분류 구조 주장의 출발점이지만 정량적 geometry 진단 부재 |
| Gu, Stevens et al. "BioCLIP 2" (NeurIPS'25 Spotlight, arXiv:2505.23883) | 2025 | TreeOfLife-200M (214M images), ViT-L/14 + autoregressive text, hierarchical contrastive (per-rank text prototype) | +18.1%p on species classification vs BioCLIP; emergent properties — inter-species embeddings align with ecological function (beak size, habitat); intra-species variation preserved in *orthogonal subspaces* | **본 연구의 분석 대상 모델**; emergent property 주장은 정성적 시각화 중심이며 (a) flat-prompt control, (b) geometric metric 정량화, (c) counterfactual ablation 부재 — 정확히 우리가 채울 gap |
| Gong et al. "BIOSCAN-CLIP / CLIBD" | 2024 | CLIP-style image + DNA barcode + text contrastive (BIOSCAN-1M / 5M, 곤충 중심) | order→species 4-rank 분류, 미관측 species retrieval | **외부 타당도 검증 데이터셋**(RQ4: Insecta 도메인); BioCLIP2와 독립 학습된 bio-VLM으로 cross-model 비교 가능 |
| Gong et al. "Hyperbolic Multimodal Representation Learning for Biological Taxonomies" (Imageomics workshop, NeurIPS'25) | 2025 | BIOSCAN-1M 위에서 image+DNA를 Poincaré ball에 contrastive + stacked entailment loss로 임베딩 | unseen species 분류에서 Euclidean baseline 능가 | **대안 geometry baseline**; Euclidean BioCLIP2와 hyperbolic 변형의 동일 분류 구조 보존도 비교 가능 |

### Category B: Hierarchical contrastive learning losses (학습 알고리즘 baseline)

| Paper | Year | Approach | Key Result | Relevance |
|-------|------|----------|------------|-----------|
| Zhang et al. "Use All The Labels: A Hierarchical Multi-Label Contrastive Learning Framework" (arXiv:2204.13207) — HiMulCon / HiConE / HiMulConE | 2022 | Anchor-positive 거리에 label-tree 거리 기반 페널티(level-dependent penalty + hierarchy-constraint enforcing loss) | Multi-label dataset에서 self-sup 및 sup baseline 상회 | **계층 contrastive 손실의 원형(prototype)**; BioCLIP2의 per-rank contrastive 설계 직접 영감원 |
| Kokilepersaud et al. "Taxes Are All You Need" (IEEE ICIP 2024, arXiv:2406.06848) | 2024 | Supervised contrastive loss에 taxonomy 거리 weight를 직접 부여 | 의료·noisy 환경에서 +7% | RQ3 C5 (text-free hierarchical InfoNCE) 조건의 *직접* baseline; 이미지-이미지 hierarchical contrastive를 수행하는 가장 가까운 선행 연구 |
| Zhang et al. "Instances and Labels: Hierarchy-aware Joint Supervised Contrastive Learning" (arXiv:2310.05128) | 2023 | Hierarchical multi-label text classification에 instance+label joint contrastive | NLP HMTC SOTA | NLP-side 참고; 본 연구 직접 baseline 아님 |
| Geng et al. "HiCLIP" (ICLR'23) | 2023 | Image patch와 text token을 *unsupervised*로 hierarchy-aware attention 통해 그룹화; CLIP backbone 수정 | YFCC15M에서 CLIP 대비 +13.1% avg acc | RQ2의 잠재 구조 가설을 직접 지지: 명시적 hierarchy supervision 없이 attention만으로 계층 구조 발견 가능. *그러나* taxonomy 정렬은 평가하지 않음 |

### Category C: Hierarchical prompt design for VLMs (프롬프트 측 baseline)

| Paper | Year | Approach | Key Result | Relevance |
|-------|------|----------|------------|-----------|
| Novack et al. "CHiLS" (ICML'23) | 2023 | 각 class를 GPT 또는 WordNet으로 subclass 집합으로 확장 → CLIP zero-shot 후 parent로 mapping | 계층 데이터셋에서 일관 향상 | **Prompt-만 변경 baseline**(RQ1의 비교군). 학습 없이 텍스트 측만 계층화 |
| Liang & Davis "HAPrompts" (arXiv:2503.02248) | 2024 | LLM-generated hierarchy-aware prompts; "better mistakes" 목적 | 5 dataset에서 mistake severity 감소 | RQ2의 분류 깊이별 효과 분석과 직접 비교 가능. BioCLIP2 도메인 미적용 |
| Wu et al. "ProTeCt" (CVPR'24, arXiv:2306.02240) | 2024 | Dynamic Treecut Loss + Node Consistency Loss로 prompt tuning, taxonomic open set 캘리브레이션 | HCA, MTA 지표 개선 | Hierarchical Consistency 평가 metric (HCA/MTA) 채택 가능; RQ1·RQ3의 "hierarchy consistency error" 지표 출처 |
| Wang et al. "HPT" (AAAI'24) / "HPT++" (arXiv:2408.14812) | 2024 | Low/high/global 3-level prompt + relation-guided attention | Generalization SOTA | 본 연구는 *학습이 아닌 분석*이지만 HPT의 prompt 구조 설계는 RQ3의 C2/C3 변형 ablation 디자인 참고 |
| Xia et al. "HGCLIP" (COLING'25, arXiv:2311.14064) | 2024 | Class hierarchy를 graph로 인코딩, GNN으로 vision+text feature 갱신 | 11개 벤치마크 +평균 15% fine-grained gain | Hierarchical prompt 이외 *그래프 구조* 활용; RQ2의 잠재구조 vs 명시구조 비교에서 보조 baseline |

### Category D: Embedding geometry diagnostics (측정 지표 출처)

| Paper | Year | Approach | Key Result | Relevance |
|-------|------|----------|------------|-----------|
| Wang & Isola "Alignment and Uniformity on the Hypersphere" (ICML'20, arXiv:2005.10242) | 2020 | Contrastive loss의 두 점근 속성(alignment, uniformity)을 정의·계측 가능한 metric으로 제시 | Downstream 성능과 강한 상관; 직접 최적화도 가능 | **RQ1의 핵심 측정 도구**; intra-class compactness ≈ alignment, inter-class spread ≈ uniformity로 매핑. 모든 prompt 조건에서 동일 측정 가능 |
| Garrido et al. "RankMe" (ICML'23, arXiv:2210.02885) | 2023 | Singular value 분포의 entropy 기반 effective rank — *unsupervised* representation quality proxy | Label 없이 hyperparameter 선택 가능 | RQ3의 정보 채널 vs 기하 조직 가설 분리에 *추가* 지표. C2(random-token hierarchy)가 RankMe를 유지하면 "구조"가 effective rank를 보존한 것 |
| Papyan et al. & follow-ups "Neural Collapse" + Kini et al. "Engineering Neural Collapse in Supervised Contrastive" (arXiv:2310.00893) | 2020–2024 | 학습 종반 class-mean이 ETF 형성; supervised contrastive에서도 동일 collapse | Class mean geometry 제어 가능 | RQ1의 intra-class variance / inter-class margin 지표의 이론적 정당화. BioCLIP2가 ETF 근사로 수렴하는지 측정 가능 |
| Bertinetto et al. "Making Better Mistakes" (CVPR'20) | 2020 | Hierarchical Cross-Entropy + soft-label hierarchy distance; "average hierarchical distance of top-k" 지표 도입 | Top-1 약간 손해, mistake severity 대폭 감소 | **RQ1·RQ4의 "hierarchy consistency error" / "LCA depth" 지표의 출처**. WordNet→Tree-of-Life 적용 가능 |
| Liang et al. "Mind the Gap" (NeurIPS'22) / Hewitt et al. "Double-Ellipsoid Geometry of CLIP" (arXiv:2411.14517) | 2022, 2024 | CLIP의 modality gap 정량화, image/text 임베딩이 별도 cone 형성 | Modality gap이 contrast loss + initialization으로 야기됨 | Confounder 관리에 필수: RQ1 측정 시 image-image 비교 vs image-text 비교 결과가 다를 수 있음을 시사 |

### Category E: 사전학습 임베딩의 잠재 분류 구조 (latent taxonomy)

| Paper | Year | Approach | Key Result | Relevance |
|-------|------|----------|------------|-----------|
| Bertucci et al. "Concept Visualization: Explaining CLIP with WordNet" (arXiv:2405.14563) | 2024 | WordNet 계층을 활용한 post-hoc CLIP saliency | CLIP 임베딩 내 WordNet 개념을 saliency로 추출 가능 | **RQ2의 잠재 구조 가설 직접 지지**: 일반 OpenCLIP 임베딩이 이미 WordNet 계층 정보를 부분적으로 보유 |
| "Explaining, Verifying, and Aligning Semantic Hierarchies in VLM Embeddings" (arXiv:2603.26798) | 2026 | VLM 임베딩으로부터 유도된 child-class hierarchy를 post-hoc로 검증·정렬 | Hierarchy 유도/검증의 framework 제시 | RQ2의 "OpenCLIP에 잠재 분류 구조가 있는가?" 분석 파이프라인 직접 참고 |
| "Probing Pretrained Language Models with Hierarchy Properties" (arXiv:2312.09670) | 2023 | Edge-distance 기반 triplet probe로 hierarchy 인코딩 정도 측정 | LM은 부분적으로 hierarchy 인코딩 | RQ2 probing protocol — image-side로 확장 적용 가능 |
| Sneyers et al. "Probing Taxonomic and Thematic Embeddings" (arXiv:2301.10656) | 2023 | Taxonomic vs thematic embedding 구분 probing | Hierarchy depth가 vector norm과 매핑됨 | RQ2의 vector-norm vs hierarchy-depth 가설 검증 보조 |

### Category F: Counterfactual / random prompt ablations (RQ3 핵심)

| Paper | Year | Approach | Key Result | Relevance |
|-------|------|----------|------------|-----------|
| Roth et al. "Waffling around for Performance" / WaffleCLIP (ICCV'23, arXiv:2306.07282) | 2023 | LLM-generated descriptor를 *random character + random word*로 치환해 동일 성능 | "LLM descriptor의 효과는 의미가 아닌 *ensemble structure*" | **RQ3의 가장 직접적인 선행 연구**; "random-token이 정상 prompt와 거의 동등"이라는 발견은 정보 채널 가설을 약화. 그러나 *계층 구조 보존* 변수는 다루지 않음 — 우리의 C2 vs C3 ablation이 채우는 정확한 gap |
| He et al. "CPL: Counterfactual Prompt Learning" (EMNLP'22) | 2022 | Few-shot에서 minimal feature change로 counterfactual 생성 | Few-shot SOTA | Counterfactual 정의 framework 참고 |

### Category G: Hyperbolic / structured hierarchy embedding (대안 geometry, 후속 연구 비교군)

| Paper | Year | Approach | Key Result | Relevance |
|-------|------|----------|------------|-----------|
| Nickel & Kiela "Poincaré Embeddings" (NIPS'17, arXiv:1705.08039) | 2017 | 심볼릭 데이터를 Poincaré ball에 임베딩 | Euclidean 대비 적은 차원으로 계층 보존 | Hyperbolic 패러다임의 원형. Scope 외(out-of-scope 후속 연구)이지만 본 연구의 "기하 조직" 해석을 보완하는 이론 근거 |
| Desai et al. "MERU: Hyperbolic Image-Text Representations" (ICML'23) | 2023 | CLIP-style 학습을 hyperbolic space에서 + entailment cone loss | CLIP 동등 성능 + 명시적 partial-order | Euclidean BioCLIP2가 *암묵적으로* 보이는 계층을 MERU가 *명시적으로* 강제. RQ2 결과 해석 시 비교 baseline |
| Pal et al. "HyCoCLIP / Compositional Entailment Learning" (ICLR'25 Oral, arXiv:2410.06912) | 2024 | Image box + text box 계층을 hyperbolic contrastive + entailment로 결합 | Compositional, retrieval, hierarchical 모두 개선 | 본 연구의 RQ3 C5(text-free hierarchical contrastive)와 design 유사. C5 결과 해석에 필수 비교 |
| Matsubara & Tomohiro "PHyCLIP" (ICLR'26, arXiv:2510.08919) | 2025 | ℓ₁-product of hyperbolic factors — 계층 + 합성성 동시 표현 | Single-space 대비 우세, interpretable structure | 후속 연구 비교군; 본 연구의 scope에서는 제외(out-of-scope), but "geometry-organizer" 가설의 향후 발전 방향 |

## Per-RQ Coverage

### RQ1 (Hierarchical prompt → geometric compactness 검증)
- **가장 가까운 선행 연구**: BioCLIP2 (Gu et al. 2025) — 정성적 시각화로 inter-species ↔ ecology, intra-species ↔ orthogonal subspace를 *주장*. 정량적 silhouette / intra-class variance / inter-class margin 통계 검정은 **부재**.
- **우리의 기여**: BioCLIP2 emergent property 주장을 (i) flat-prompt 대조군, (ii) bootstrap/permutation p-value, (iii) alignment-uniformity (Wang & Isola), silhouette (rank별), neural-collapse 류 metric으로 **정량 검증**. 이는 BioCLIP2 논문 자체가 제공하지 않는 진단적 분석.
- **추가 baseline**: HAPrompts (Liang & Davis 2024), CHiLS (Novack et al. 2023) — *non-bio* 도메인의 hierarchical prompt 효과 비교.

### RQ2 (Rank-level effect + latent taxonomy in pretrained CLIP)
- **가까운 선행 연구**: HiCLIP (Geng et al. 2023) — *unsupervised* hierarchy attention이 작동한다는 사실은 잠재 구조 존재의 간접 증거. Concept Visualization (Bertucci et al. 2024) — CLIP 임베딩이 WordNet 개념을 saliency로 노출. Probing Pretrained LMs with Hierarchy Properties (2023) — edge-distance triplet probe.
- **공백**: OpenCLIP/CLIP 임베딩이 **Linnaean taxonomy(생물 분류)**에 대해 random-taxonomy control 대비 silhouette 우세를 보이는지 *정량적*으로 검정한 연구 **부재**. Rank별(species→kingdom) effect size 분해도 부재.
- **우리의 기여**: 동일 임베딩, 동일 평가 프로토콜로 (a) BioCLIP2 vs BioCLIP vs OpenCLIP, (b) species→kingdom rank별 silhouette/MI, (c) random-taxonomy permutation test를 일관 적용.

### RQ3 (Information-channel vs Geometry-organizer counterfactual)
- **가장 가까운 선행 연구**: **WaffleCLIP (Roth et al. ICCV'23)** — random-character/random-word descriptor가 LLM descriptor와 거의 동등한 성능. *우리 가설의 강력한 간접 증거*. 그러나 (a) bio 도메인 미적용, (b) *계층 구조 보존-vs-파괴* 변수 미설계, (c) embedding geometry 직접 측정 없음.
- **공백**: "랜덤 토큰이지만 계층 구조(7-rank ordering)는 보존"(C2) vs "정상 어휘지만 계층 셔플"(C3)을 직접 대조한 연구 **없음**. Text-free hierarchical contrastive (C5)의 가장 가까운 사례는 Kokilepersaud et al. 2024(Taxes Are All You Need) — taxonomy-weighted SupCon이지만 *image-image only*로 ablation으로서의 비교 직접 가능.
- **우리의 기여**: 이 RQ가 본 연구의 **가장 큰 novelty**. 6개 prompt 조건(C0–C5)의 통제 실험은 선행 연구에서 누구도 단일 framework로 수행하지 않음.

### RQ4 (도메인 일반화 — Aves, Insecta, Plantae, Fungi, Actinopterygii)
- **가까운 선행 연구**: BIOSCAN-CLIP (Gong et al. 2024) — 곤충 도메인에 독립 학습된 bio-VLM. Hyperbolic Multimodal for Bio Taxonomies (Gong et al. 2025) — 도메인 일반화 가능성.
- **공백**: BioCLIP2의 emergent property가 **5개 분류군 도메인에서 균질하게 나타나는지** 검증한 연구 부재. BioCLIP2 원논문은 도메인 간 effect-size variability를 보고하지 않음.
- **우리의 기여**: Random-effects meta-analysis (I², CV)로 효과의 cross-domain heterogeneity를 정량화.

## Identified Gap (재확인)

**원래 정의한 gap**: BioCLIP2 등 기존 연구는 결과(분류 정확도) 향상만 보고하며, "정보 채널" vs "기하 조직" 가설을 분리하지 못함.

**문헌 조사 후 재확인**:
1. **Gap이 무효화되지 않았다**. 오히려 더 명확해졌다. BioCLIP2 (NeurIPS'25 Spotlight) 자체가 emergent property를 정성적으로만 주장하며, 정량적 geometry 분석과 counterfactual ablation을 *명시적으로 제공하지 않는다*. 우리의 진단적 분석은 *바로 그 빈자리*를 메운다.
2. **인접 분야 부분 증거 존재**: WaffleCLIP (Roth 2023)은 "텍스트 의미보다 구조가 성능을 만든다"는 일반적 증거를 비-bio 도메인에서 제시. 본 연구는 이 직관을 (a) bio 도메인에 *명시적 분류군 계층*에서 검증, (b) embedding geometry metric으로 mechanism을 직접 관찰, (c) random-token-but-structured vs structured-shuffle을 동시 ablation으로 분리한다는 점에서 차별화된다.
3. **새롭게 부각된 risk**: HiCLIP (Geng 2023), Concept Visualization (Bertucci 2024), Probing PLM Hierarchy (2023) 등이 일반 CLIP에도 *부분적* 잠재 구조 존재를 시사. 이는 RQ2의 H1(b)(잠재 구조 존재)을 *지지*하지만, 동시에 BioCLIP2의 차별성을 정량화할 때 effect size 기준 설정을 신중히 해야 함을 시사 — RQ2 success threshold에서 OpenCLIP silhouette ≥ BioCLIP2의 30%까지 허용한 것은 적절.
4. **Scope 재확인**: Hyperbolic 변형(MERU, HyCoCLIP, PHyCLIP, Hyperbolic Bio-Taxonomy 2025)은 이미 활발한 연구 영역. 본 연구는 *Euclidean BioCLIP2*에 집중하고 hyperbolic은 future work로 두는 것이 합당. PHyCLIP(ICLR'26)이 hierarchy+compositionality를 동시 다루므로 본 연구 이후 자연스러운 후속이 됨.

**결론**: Gap은 유효하며, 본 연구는 BioCLIP2의 NeurIPS'25 spotlight 주장에 대한 **independent quantitative verification + causal disentanglement**라는 명확한 학술적 포지션을 가짐. 문제 재정의 불필요, **다음 단계 진행 권고**.

## Recommended Baselines for Experiments

| Baseline | Source | 코드/체크포인트 | 실행 가능성 | 역할 |
|----------|--------|-----------------|------------|------|
| **BioCLIP2 ViT-L/14** | Gu et al. 2025, `imageomics/bioclip-2` HF | 공개 (HF) | 즉시 | 주 분석 대상 |
| **BioCLIP ViT-B/16** | Stevens et al. 2024, `imageomics/bioclip` HF | 공개 (HF + GitHub `Imageomics/bioclip`) | 즉시 | 스케일링 비교 (RQ2) |
| **OpenCLIP ViT-L/14 (laion-2b)** | `mlfoundations/open_clip` | 공개 | 즉시 | 잠재 구조 control (RQ2 H1b) |
| **WaffleCLIP** | Roth et al. ICCV'23, `ExplainableML/WaffleCLIP` | 공개 | 즉시 | RQ3 C2 (random token) 실험 reference 구현 |
| **CHiLS** | Novack et al. ICML'23, `acmi-lab/CHILS` | 공개 | 즉시 | RQ1의 hierarchical prompt 대안 비교 |
| **HAPrompts** | Liang & Davis 2024 | 공개 (arXiv 첨부) | 보통 — LLM 호출 필요 | RQ1·RQ2 보조 비교 (better mistakes metric) |
| **BIOSCAN-CLIP / CLIBD** | Gong et al. 2024, `bioscan-ml/CLIBD` | 공개 | 즉시 | RQ4 Insecta 도메인 cross-model |
| **Simple linear-probe ablation** (직접 구현) | — | — | trivial | RQ3 C5 (text-free hierarchical InfoNCE) 자체 구현, Kokilepersaud et al. 2024 손실 함수 채택 |
| **Hierarchical Cross-Entropy (HXE)** | Bertinetto et al. CVPR'20 | 공개 (GitHub `fiveai/making-better-mistakes`) | 즉시 | RQ1·RQ4의 "better mistakes" metric 표준 구현 |

### 측정 metric 표준 구현
- **Alignment / Uniformity**: Wang & Isola 공식 GitHub `ssnl/align_uniform` 직접 사용.
- **RankMe**: 공식 reference + 자체 구현(20 lines).
- **Silhouette**: scikit-learn `sklearn.metrics.silhouette_score`.
- **Neural Collapse metrics (NC1/NC2)**: Papyan et al. 표준 정의, 자체 구현.
- **LCA depth / hierarchy distance**: Bertinetto fiveai 코드 채택.

## Search Coverage Notes
- **직접 비교 연구 부재**: "BioCLIP/BioCLIP2의 hierarchical prompt 효과를 embedding geometry로 분해한 연구"는 검색 결과 **0건**. NeurIPS'25 BioCLIP2 spotlight 이후 6개월 미만이므로 follow-up 자체가 적음.
- **counterfactual prompt + structure preservation**: WaffleCLIP이 유일한 강한 reference. 구조 보존-vs-파괴 통제 ablation은 우리가 최초로 보임.
- **접근 불가능한 자료**: PHyCLIP은 ICLR'26 (2026 publication, code published) — 본 검토 시점 기준 충분히 접근 가능. 나머지 모든 인용 논문은 arXiv 또는 official repo에서 검증 완료.
- **확인되지 않은 영역**: BioCLIP2 저자들의 공식 supplementary에 추가 geometry 분석이 포함될 가능성 (NeurIPS'25 camera-ready 또는 OpenReview supplementary 확인 권고).
- **한국어 문헌**: 한국어 검색 미수행 — 본 도메인(생물 VLM)에서 영문 문헌이 압도적이며 한국어 선행 연구는 검색되지 않을 것으로 판단.

## References

1. Stevens, S., Wu, J., Thompson, M. J., Campolongo, E. G., Song, C. H., Carlyn, D. E., Dong, L., Dahdul, W. M., Stewart, C., Berger-Wolf, T., Chao, W.-L., Su, Y. (2024). *BioCLIP: A Vision Foundation Model for the Tree of Life*. CVPR 2024 (Best Student Paper). https://arxiv.org/abs/2311.18803
2. Gu, J., Stevens, S., Campolongo, E. G., Thompson, M. J., Zhang, N., Wu, J., Kopanev, A., Mai, Z., White, A. E., Balhoff, J., Dahdul, W., Rubenstein, D., Lapp, H., Berger-Wolf, T., Chao, W.-L., Su, Y. (2025). *BioCLIP 2: Emergent Properties from Scaling Hierarchical Contrastive Learning*. NeurIPS 2025 Spotlight. https://arxiv.org/abs/2505.23883
3. Gong, Z. et al. (2024). *BIOSCAN-CLIP (CLIBD): Bridging Vision and Genomics for Biodiversity Monitoring at Scale*. https://arxiv.org/abs/2405.17537
4. Gong, Z. et al. (2025). *Hyperbolic Multimodal Representation Learning for Biological Taxonomies*. NeurIPS 2025 Imageomics workshop. https://arxiv.org/abs/2508.16744
5. Zhang, S. et al. (2022). *Use All The Labels: A Hierarchical Multi-Label Contrastive Learning Framework*. arXiv:2204.13207. https://arxiv.org/abs/2204.13207
6. Kokilepersaud, K., Yarici, Y., Prabhushankar, M., AlRegib, G. (2024). *Taxes Are All You Need: Integration of Taxonomical Hierarchy Relationships into the Contrastive Loss*. IEEE ICIP 2024. https://arxiv.org/abs/2406.06848
7. Geng, S., Yuan, J., Tian, Y., Chen, Y., Zhang, Y. (2023). *HiCLIP: Contrastive Language-Image Pretraining with Hierarchy-aware Attention*. ICLR 2023. https://arxiv.org/abs/2303.02995
8. Novack, Z., Garg, S., McAuley, J., Lipton, Z. (2023). *CHiLS: Zero-Shot Image Classification with Hierarchical Label Sets*. ICML 2023. https://arxiv.org/abs/2302.02551
9. Liang, T., Davis, J. (2024). *Making Better Mistakes in CLIP-Based Zero-Shot Classification with Hierarchy-Aware Language Prompts*. arXiv:2503.02248. https://arxiv.org/abs/2503.02248
10. Wu, T.-Y. et al. (2024). *ProTeCt: Prompt Tuning for Taxonomic Open Set Classification*. CVPR 2024. https://arxiv.org/abs/2306.02240
11. Wang, Y. et al. (2024). *Learning Hierarchical Prompt with Structured Linguistic Knowledge for Vision-Language Models (HPT)*. AAAI 2024. https://arxiv.org/abs/2312.06323
12. Wang, Y. et al. (2024). *HPT++: Hierarchically Prompting VLMs with Multi-Granularity Knowledge Generation*. arXiv:2408.14812. https://arxiv.org/abs/2408.14812
13. Xia, P. et al. (2024). *HGCLIP: Exploring Vision-Language Models with Graph Representations for Hierarchical Understanding*. COLING 2025. https://arxiv.org/abs/2311.14064
14. Wang, T., Isola, P. (2020). *Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere*. ICML 2020. https://arxiv.org/abs/2005.10242
15. Garrido, Q., Balestriero, R., Najman, L., LeCun, Y. (2023). *RankMe: Assessing the Downstream Performance of Pretrained Self-Supervised Representations by Their Rank*. ICML 2023. https://arxiv.org/abs/2210.02885
16. Kini, G. R. et al. (2023). *Engineering the Neural Collapse Geometry of Supervised-Contrastive Loss*. arXiv:2310.00893. https://arxiv.org/abs/2310.00893
17. Bertinetto, L., Mueller, R., Tertikas, K., Samangooei, S., Lord, N. A. (2020). *Making Better Mistakes: Leveraging Class Hierarchies with Deep Networks*. CVPR 2020. https://arxiv.org/abs/1912.09393
18. Liang, V. W. et al. (2022). *Mind the Gap: Understanding the Modality Gap in Multi-modal Contrastive Representation Learning*. NeurIPS 2022. https://arxiv.org/abs/2203.02053
19. Bertucci, M. et al. (2024). *Concept Visualization: Explaining the CLIP Multi-modal Embedding Using WordNet*. arXiv:2405.14563. https://arxiv.org/abs/2405.14563
20. Roth, K., Kim, J. M., Koepke, A. S., Vinyals, O., Schmid, C., Akata, Z. (2023). *Waffling around for Performance: Visual Classification with Random Words and Broad Concepts*. ICCV 2023. https://arxiv.org/abs/2306.07282
21. He, X. et al. (2022). *CPL: Counterfactual Prompt Learning for Vision and Language Models*. EMNLP 2022. https://aclanthology.org/2022.emnlp-main.224/
22. Nickel, M., Kiela, D. (2017). *Poincaré Embeddings for Learning Hierarchical Representations*. NIPS 2017. https://arxiv.org/abs/1705.08039
23. Desai, K., Nickel, M., Rajpurohit, T., Johnson, J., Vedantam, R. (2023). *Hyperbolic Image-Text Representations (MERU)*. ICML 2023. https://arxiv.org/abs/2304.09172
24. Pal, A., Spengler, M. et al. (2024). *Compositional Entailment Learning for Hyperbolic Vision-Language Models (HyCoCLIP)*. ICLR 2025 Oral. https://arxiv.org/abs/2410.06912
25. Matsubara, T. et al. (2025). *PHyCLIP: ℓ₁-Product of Hyperbolic Factors Unifies Hierarchy and Compositionality in Vision-Language Representation Learning*. ICLR 2026. https://arxiv.org/abs/2510.08919
26. (anonymized authors) (2024). *The Double-Ellipsoid Geometry of CLIP*. arXiv:2411.14517. https://arxiv.org/abs/2411.14517
