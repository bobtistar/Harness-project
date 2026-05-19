# Problem Statement

## 키워드
BioCLIP2 hierarchical prompts and embedding geometry

## 키워드 해석
이 키워드는 두 가지 축으로 해석될 수 있다.
- (A) **계층 프롬프트의 성능 향상 메커니즘 분석**: BioCLIP2에서 분류군 계층(kingdom → species)을 텍스트로 명시하는 hierarchical prompt가 *왜* 작동하는지를, embedding 공간의 기하학적 변화 관점에서 규명.
- (B) **계층 프롬프트를 활용한 새 학습 알고리즘 제안**: 계층 정보를 손실 함수/정규화로 통합하여 더 나은 vision-language 모델을 만드는 방향.

채택 해석: **(A)**. 근거: 사용자가 제시한 세 가지 claim이 모두 "관찰된 현상의 원인 규명"에 초점이 맞춰져 있으며, 특히 *"계층 supervision이 텍스트 정보 제공이 아니라 embedding geometry의 재조직자(semantic organizer)로 기능한다"*는 가설이 핵심이다. 따라서 본 연구는 분석적·진단적(diagnostic) 성격이 강하며, 새로운 학습 알고리즘 제안은 후속 작업(future work)으로 둔다.

## 도메인 컨텍스트
생물 분류(taxonomic classification)는 본래 fine-grained recognition 중에서도 가장 어려운 영역에 속한다. 종(species) 수가 수십만에 달하고 클래스 간 시각적 차이가 미세하며(예: 두 근연종 사이의 무늬·부리 형태 차이), 데이터 분포는 long-tail이다. 일반 ImageNet 사전학습 모델은 이러한 도메인에서 큰 성능 저하를 보였고, 이에 대응해 **BioCLIP** (Stevens et al., CVPR 2024, Best Student Paper)이 TreeOfLife-10M 데이터셋(약 45만 분류군)을 활용한 CLIP-style contrastive 학습으로 등장했다. BioCLIP은 텍스트 측에 단순 종명이 아니라 *"Animalia Chordata Aves … Passer domesticus"* 형태의 **계층적 분류군 시퀀스(hierarchical taxonomic prompt)**를 사용했으며, 결과적으로 학습된 embedding이 생명의 나무(tree of life)에 부합하는 계층 구조를 보였다.

이후 **BioCLIP2** (arXiv:2505.23883, 2025)는 약 2억 1,400만 장 규모로 데이터를 확장하고 ViT-L/14 + 자기회귀 텍스트 인코더로 *hierarchical contrastive learning*을 수행, BioCLIP 대비 종 분류에서 18.1%p 향상을 보고했다. 더 흥미로운 점은 **emergent properties**다: inter-species 임베딩 분포가 부리 크기·서식지 같은 기능적·생태학적 의미와 정렬되고, intra-species 변이는 보존된다는 보고다.

한편 일반 vision-language 영역에서도 hierarchy-aware prompts (HAPrompts, CHiLS, HPT 등)가 zero-shot fine-grained classification을 개선한다는 연구가 누적되고 있다. 그러나 이들 연구의 공통적 해석은 *"계층 정보가 추가 텍스트 신호로 작용해 더 풍부한 텍스트 임베딩을 만든다"*는 정도에 머문다. 계층 supervision이 임베딩 공간의 **기하학(geometry)** 자체를 재조직한다는 가설은 명시적으로 검증되지 않았다.

## 핵심 문제 (Gap)
BioCLIP2를 포함한 기존 hierarchical prompt 연구들은 *"계층 프롬프트를 쓰면 분류 정확도와 retrieval이 좋아진다"*는 결과 수준의 관찰을 제공하지만, **그 향상이 어디서 오는지를 분리(disentangle)하지 못한다**. 구체적으로 두 가지 가능한 메커니즘이 혼재되어 있다:

1. **정보 채널 가설(Information-channel hypothesis)**: 계층 프롬프트가 종명 외에 *부가 텍스트 토큰*(상위 분류군명)을 더해 텍스트 임베딩의 표현력을 단순히 증가시킨다.
2. **기하 조직 가설(Geometry-organizer hypothesis)**: 계층 supervision이 학습 신호로서 임베딩 공간을 *재구조화*하여, 분류학적으로 가까운 종이 공간상으로도 가깝게 묶이는 **semantic compactness**를 유도한다 — 즉, 사전학습된 vision-language 모델 내부에 이미 잠재(latent)되어 있던 분류학적 구조를 *"꺼내 정렬시키는"* 역할을 한다.

기존 평가 지표(top-1 accuracy, mAP)는 두 가설을 구분하지 못한다. 따라서 정량적·기하학적 진단(예: intra-class variance, inter-class margin, silhouette score, taxonomic alignment retrieval, neural collapse 류 지표)을 통해 두 메커니즘의 기여도를 분리해야 한다.

## 왜 중요한가
- **이론적 함의**: 만약 기하 조직 가설이 검증되면, 텍스트 supervision의 역할에 대한 일반적 이해(*"텍스트는 추가 의미 정보를 공급한다"*)를 *"텍스트는 잠재 구조를 정렬시키는 prior로 작용한다"*로 재정식화할 수 있다. 이는 CLIP-계열 모델 전반에 대한 해석 프레임을 바꾼다.
- **방법론적 함의**: 계층이 *조직자*라면, 굳이 계층 텍스트를 모든 학습 샘플에 부착하지 않아도 계층 기반 정규화(예: hyperbolic embedding, tree-metric regularizer, hierarchical InfoNCE)만으로 동등한 효과를 낼 가능성이 있다. 이는 라벨링 비용 절감으로 직결된다.
- **응용적 함의**: 생물다양성 모니터링, 멸종위기종 인식, 생태학적 trait prediction 등 BioCLIP2의 다운스트림 응용에서 *어떤 embedding 속성이 성능을 만드는가*를 알면, 도메인 특화 fine-tuning 전략을 더 정교화할 수 있다.

## 누가 영향받는가
- **Vision-language 연구자**: 텍스트 supervision의 메커니즘에 대한 재해석을 얻는다.
- **생물·생태 ML 실무자**: BioCLIP/BioCLIP2를 다운스트림에 적용할 때, *언제* 계층 프롬프트가 유의미한지에 대한 가이드를 얻는다(예: 학습 데이터가 적을수록, long-tail이 심할수록 기하 조직 효과가 클 것으로 예상).
- **분류학·생물다양성 도메인 사이언티스트**: 모델 임베딩이 분류학적 구조를 얼마나 충실히 반영하는지에 대한 정량 지표를 확보한다.

## 측정 가능한 성공 기준 (Candidates)
- **Intra-class variance (종 내 분산)**: 같은 종의 이미지 embedding의 평균 ℓ2 거리. 계층 프롬프트 사용 시 *flat* 프롬프트 baseline 대비 통계적으로 유의한 감소(예: ≥ 10% 감소, p<0.01).
- **Inter-class margin (종 간 마진)**: 가장 가까운 다른 종 centroid까지의 평균 거리 / intra-class variance 비율. 베이스라인 대비 ≥ 1.2배 증가.
- **Silhouette score (taxonomic levels에서)**: genus·family·order 각 수준에서의 silhouette 계수. 계층 프롬프트로 학습한 모델이 모든 수준에서 baseline을 상회.
- **Taxonomic retrieval alignment**: query 이미지에 대한 top-k 검색 결과의 *평균 LCA(lowest common ancestor) 깊이*. 깊을수록 분류학적으로 정렬된 검색. 베이스라인 대비 ≥ 0.5 분류 단계(rank) 개선.
- **Hierarchy consistency error**: top-1 종 예측의 상위 분류군이 ground-truth 상위 분류군과 일치하는 비율 ("better mistakes" metric 참조).
- **Counterfactual ablation**: (a) 계층 텍스트는 같지만 분류군명을 *무작위 토큰*으로 치환한 프롬프트, (b) 계층 정보 없이 종명만 사용, 두 baseline 대비 성능 격차. 정보 채널 가설이면 (a)가 크게 망가지고, 기하 조직 가설이면 (a)도 어느 정도 동작해야 함.

## 범위 (Scope)
- **포함**: BioCLIP2(또는 공개된 BioCLIP) 사전학습 체크포인트를 분석 대상으로 사용. TreeOfLife-10M 또는 그 부분집합을 평가 데이터로 사용. Embedding geometry 진단 지표 정의 및 측정. flat-prompt vs hierarchical-prompt 대조 실험. counterfactual prompt ablation.
- **제외**: 새로운 대규모 사전학습 처음부터 수행(예산 외). 비-생물 도메인(일반 ImageNet, COCO)으로의 일반화 검증. 의료영상 등 다른 fine-grained 도메인. Hyperbolic embedding 등 새 아키텍처 제안(후속 연구로 분리).

## 가정 (Assumptions)
- BioCLIP2의 공개 체크포인트 및 학습 코드/프롬프트 템플릿이 재현 가능한 수준으로 접근 가능하다.
- TreeOfLife 분류군 메타데이터(7-rank taxonomy)가 평가 데이터에 부착되어 있다.
- "기하 조직" 효과는 단일 데이터셋이 아닌 여러 분류군 도메인(곤충·조류·식물 등)에서 일관되게 관찰될 때만 유의미한 것으로 본다.
- CLIP-style contrastive 손실의 통상적 해석(image-text alignment in shared embedding space)을 그대로 사용한다.

## 열린 질문 (RQ 단계에서 해소)
- 기하 조직 효과는 **모든 분류 깊이(species → kingdom)**에서 동일하게 나타나는가, 아니면 특정 rank에서 두드러지는가?
- 효과의 크기는 **학습 데이터 규모**와 어떻게 스케일링되는가? (BioCLIP vs BioCLIP2 비교)
- **시각 인코더만의 표현**(이미지 embedding)에서도 기하 효과가 드러나는가, 아니면 텍스트 anchor를 거쳐야만 보이는가?
- counterfactual ablation에서 어느 정도의 잔존 성능을 "기하 조직" 가설의 증거로 인정할 것인가? (effect size 임계값 설정 문제)
- 본 분석이 일반 CLIP/SigLIP 등 비-bio 모델에도 외삽 가능한가? (Scope에서 제외했으나 토론거리로 남김)

## 참고 자료
- BioCLIP 2: Emergent Properties from Scaling Hierarchical Contrastive Learning, arXiv:2505.23883 (2025). https://arxiv.org/abs/2505.23883
- BioCLIP project page (imageomics). https://imageomics.github.io/bioclip-2/
- Stevens et al., "BioCLIP: A Vision Foundation Model for the Tree of Life", CVPR 2024 (Best Student Paper). https://openaccess.thecvf.com/content/CVPR2024/papers/Stevens_BioCLIP_A_Vision_Foundation_Model_for_the_Tree_of_Life_CVPR_2024_paper.pdf
- HAPrompts: "Making Better Mistakes in CLIP-Based Zero-Shot Classification with Hierarchy-Aware Language Prompts", arXiv:2503.02248.
- "Use All The Labels: A Hierarchical Multi-Label Contrastive Learning Framework", arXiv:2204.13207.
- "Hyperbolic Contrastive Learning", arXiv:2302.01409 (계층 구조의 기하학적 표현 비교 reference).
- ProTeCt: Prompt Tuning for Taxonomic Open Set Classification, arXiv:2306.02240.
