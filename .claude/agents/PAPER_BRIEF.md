# 00. PAPER BRIEF — 에이전트가 가장 먼저 읽을 문서

> 이 문서는 논문 **"Disentangling Structure and Content in Taxonomic Prompts for Biological Vision-Language Models"** 의 Section IV / V 를 작성하기 위한 컨텍스트와 규칙이다.
> Section I~III 는 이미 완성되어 있다. 너의 작업은 **Section IV (Method)** 와 **Section V (Experiments)** 다.
> 작성 언어: **영어** (IEEE Access two-column 양식). 본문은 영어로 쓰되, 이 브리핑의 한국어 지시는 이해만 하면 된다.

---

## ⚠️ 0. 가장 먼저 — 현재 초안의 치명적 오류

현재 PDF 초안의 **Section IV 본문은 이 논문과 무관한 다른 논문 내용**이다.
"conventional timestep schedules", "StepOptim algorithm", "Accelerating Diffusion Sampling with Optimized Time Steps", "log-SNR", "reverse ODE", "NFE", "data prediction network" 등은 **diffusion 모델 sampling 가속** 논문에서 잘못 붙여넣어진 텍스트다.

- 이 내용은 **전부 무시하고 삭제**한다. 절대 이어서 쓰거나 참고하지 않는다.
- 본문 안의 "Figure 2", "Equation (4)" (= the integral formula `x_ε = ...`) 도 diffusion 논문 잔재이므로 사용하지 않는다.
- 이 논문의 "method" 는 새 모델/알고리즘이 아니라 **controlled perturbation + embedding geometry 분석 프로토콜**이다.

---

## 1. 이 논문이 무엇인가 (한 문단 요약)

계층적(taxonomic) 프롬프트가 biological VLM의 표현 품질을 높인다는 선행연구(BioCLIP, BioCLIP2)가 있다. 그러나 계층 프롬프트는 **(a) 의미 토큰을 더 많이 넣는 것(semantic enrichment)** 과 **(b) 계층 구조 자체(hierarchical organization)** 를 동시에 바꾸기 때문에, 개선의 원인을 분리할 수 없다. 이 논문은 **의미 토큰 집합은 동일하게 유지한 채 구조만 교란하는 controlled perturbation 프레임워크**로 이 둘을 분리하고, embedding geometry 분석을 통해 계층 프롬프트가 단순한 **information channel** 인지 아니면 **geometric organizer** 인지 검증한다.

## 2. 핵심 연구 질문 (논문 전체의 축)

> **계층 프롬프트의 이득은 (A) 풍부한 의미 정보(information channel) 때문인가, 아니면 (B) 계층 구조 자체(geometric organizer) 때문인가?**

Section IV / V 의 모든 서술은 이 질문에 답하기 위한 것임을 잊지 말 것.

## 3. Section I~III 에서 이미 확립된 것 (중복 금지)

에이전트는 아래 내용을 **다시 설명하지 말고**, 필요하면 짧게 참조(refer back)만 한다.

- **Sec I (Introduction):** 동기, 문제 정의(두 요인이 동시에 변한다는 ambiguity), contributions 3개.
- **Sec II (Related Work):** A) Biological VLM (BioCLIP[1], BioCLIP2[2], BIOSCAN-CLIP/CLIBD[3]) / B) Hierarchical supervision & hierarchy-aware prompting [4–7] / C) Embedding geometry diagnostics — alignment·uniformity[8], RankMe[9], hierarchical error[10], CLIP-WordNet[11], WaffleCLIP[12].
- **Sec III (Preliminaries) — 여기 이미 정의된 것:**
  - VLM prompt embedding: `z_x = f_img(x)`, `z_p = f_text(p)` (Eq. 1), cosine sim `s(x,p)` (Eq. 2).
  - Hierarchical prompt 예: `Animal → Mammal → Canine → Husky` (Eq. 3).
  - 프롬프트의 두 구성요소: **semantic token content** vs **hierarchical organization**.
  - **5가지 perturbation 타입** (이미 정의됨, IV에서 재정의 금지):
    1. Structure-preserving: `Animal → Mammal → Canine → Husky`
    2. Structure-removed (flat): `Animal Mammal Canine Husky`
    3. Shuffled: `Husky Mammal Animal Canine`
    4. Reversed: `Husky → Canine → Mammal → Animal`
    5. Semantically inconsistent: `Vehicle → Flower → Husky`
  - **Geometry metric 4종족** (이미 언급됨): intra-class similarity, inter-class separability, clustering consistency(silhouette, kNN purity), embedding visualization(UMAP, t-SNE).

→ **따라서 IV는 III를 넘어서는 내용**(정식 프로토콜화, hypothesis 정의, rank-resolved probing, image-only vs image-text 비교, 통계 절차)을 다뤄야 한다. 자세한 것은 `01_SECTION4_METHOD.md` 참조.

## 4. 논문이 주장하는 것 / 절대 과장하지 말 것

**주장(claim)해도 되는 것:**
- 일관된 계층을 보존하면, 단순히 토큰을 늘리는 것보다 **higher-rank embedding 구조**가 더 안정적으로 좋아진다.
- 토큰 집합이 동일해도 구조를 교란하면 clustering consistency와 taxonomic 조직이 저하된다.

**반드시 hedge(완화)해야 하는 것 — 기존 본문의 톤을 그대로 유지:**
- 계층 구조"만"이 표현 품질을 보편적으로 결정한다고 주장하지 **않는다**.
- 메커니즘이 완전히 규명되었다고 주장하지 **않는다**.
- species-level compactness는 일관되게 개선되지 **않을 수 있음**을 인정한다.
- 결과는 "controlled perturbation 실험에 기반한 **analysis-oriented** 해석"임을 명시.

## 5. 작성 규칙 (HARD RULES)

1. **숫자/결과를 절대 지어내지 말 것.** 아직 실험 수치가 확정 입력되지 않았다. 모든 정량 결과는 아래 placeholder 형식으로 비워둔다:
   ```
   [[RESULT: <metric>, condition=<...>, dataset=<...>, model=<...>]]
   ```
   예: `[[RESULT: silhouette, condition=preserving, dataset=CUB-200, model=BioCLIP2]]`
   표/그림도 구조만 만들고 셀은 placeholder로 둔다. 저자(사용자)가 실제 수치로 채운다.
2. **표기(notation) 일관성:** Eq. 1–3의 기호(`z_x, z_p, f_img, f_text, s(x,p)`)를 그대로 쓴다. 새 기호는 처음 등장 시 정의.
3. **인용 번호:** 기존 [1]–[12] 체계를 유지. 새 참고문헌이 필요하면 [13]부터 이어가고, 어떤 문헌인지 주석으로 표시(`% NEW REF: ...`).
4. **톤:** 신중하고 분석 지향적이며 잘 hedge된 학술 영어. 단정·홍보성 표현 금지.
5. **양식 확인 필요(저자에게):** 현재 템플릿은 IEEE Access(장문) 양식이다. 실제 투고처가 KCC 같은 단문 학회라면 분량/깊이를 줄여야 하므로 저자 확인 전까지는 IEEE Access 기준으로 작성.
6. **채워야 할 빈칸 목록(저자 입력 대기):** ① multi-phylum rare-species 데이터셋 정식 명칭, ② OpenCLIP/BioCLIP2 backbone(예: ViT-B/16) 및 pretrain 출처, ③ 실제 정량 수치 전부, ④ Figure 실물.
