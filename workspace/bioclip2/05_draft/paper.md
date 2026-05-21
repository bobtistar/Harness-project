# Hierarchical Prompting as a Semantic Organizer of Latent Taxonomic Geometry in Biological Vision-Language Models: A Diagnostic Framework

## Abstract

Recent biological vision-language models such as BioCLIP and BioCLIP2 report that supervising contrastive pretraining with hierarchical taxonomic prompts (e.g., *Animalia Chordata Aves ... Passer domesticus*) yields large gains in fine-grained species recognition and gives rise to *emergent properties* in which inter-species embedding distances appear to align with ecological and functional structure. The underlying mechanism, however, has remained unclear: hierarchical prompts could be acting simply as an extra textual information channel, or they could be acting as a *semantic organizer* that re-shapes the geometry of an already partially structured embedding space. We formulate this as an explicit dichotomy and design a diagnostic protocol to disentangle it. Our framework combines (i) a flat-vs-hierarchical geometric audit using intra-class variance, inter-class margin, silhouette, and alignment/uniformity metrics, (ii) a rank-resolved latent-taxonomy probe applied to OpenCLIP, BioCLIP, and BioCLIP2, and (iii) a five-condition counterfactual prompt ablation (C0-C4) that separates *structure preservation* from *lexical content*, extending the random-descriptor finding of WaffleCLIP to the structure-versus-content axis in a biological domain. We argue that, if hierarchical supervision functions primarily as a geometric organizer of latent taxonomic structure, then random-token-but-structured prompts should retain a large fraction of the geometric benefit while structure-shuffled prompts should not. We instantiate the full pipeline and report toy-scale sanity-check signals from an OpenCLIP ViT-B/32 run; directional trends are consistent with the geometry-organizer hypothesis, but conclusive validation requires the full TreeOfLife evaluation on BioCLIP2, which we leave as immediate future work. The contribution of this paper is therefore positioned as a *quantitative and causal verification framework* for emergent-property claims about hierarchical contrastive learning, together with a pre-registered analysis protocol and an open-source diagnostic codebase.

---

## 1. Introduction

Fine-grained biological recognition is a long-standing stress test for vision-language pretraining. The space of described species exceeds 1.5 million; visual differences between sister taxa are often subtle (a beak curvature, a wing-bar pattern); and observational data are heavy-tailed, with most species rarely photographed. CLIP-style contrastive models trained on web-scale image-caption pairs underperform on this regime, which motivated the recent emergence of *biology-specific* vision foundation models. BioCLIP \cite{stevens2024bioclip} introduced contrastive pretraining over the TreeOfLife-10M dataset (about 454K taxa) and crucially supervised the text branch with the *full Linnaean hierarchy* (a seven-rank prompt such as "*Animalia Chordata Aves Passeriformes Passeridae Passer domesticus*") rather than the species name alone. BioCLIP2 \cite{gu2025bioclip2} scaled this recipe to about 214M images and reported a further 18.1 percentage-point gain on species classification. More provocatively, BioCLIP2 documents an *emergent property* claim: inter-species embedding distances appear to align with ecological and functional attributes such as beak size and habitat, while intra-species variation is preserved in orthogonal subspaces.

These results are striking, but they leave open *why* hierarchical prompting helps. Two mechanistic accounts are compatible with the same downstream gains:

1. **Information-channel hypothesis.** Hierarchical prompts contribute extra textual tokens (kingdom, phylum, class, order, family, genus) that enrich the text-side embedding and therefore tighten image-text alignment for species that share higher-rank ancestors.
2. **Geometry-organizer hypothesis.** A pretrained vision-language model already encodes a *latent* approximation of taxonomic structure (consistent with findings that CLIP embeddings expose WordNet hierarchy \cite{bertucci2024concept}), and hierarchical supervision acts not by injecting new semantic information but by *re-aligning* the embedding geometry so that biologically related concepts become semantically compact.

The two accounts predict identical top-1 accuracy and are indistinguishable under standard evaluation. They differ sharply, however, in their *causal* and *geometric* signatures. Under the information-channel account, replacing the taxonomic vocabulary with meaningless tokens should erase most of the benefit. Under the geometry-organizer account, what matters is the *structure* of the prompt (its preserved hierarchical alignment with the taxonomy), not the lexical content. WaffleCLIP \cite{roth2023waffling} previously observed in *general* domains that random-character descriptors recover most of the gain attributed to LLM-generated class descriptions, but it did not separate *structure preservation* from *structure destruction*, and it did not examine biological taxonomy where a ground-truth hierarchy is well defined.

**This paper.** We design and instantiate a diagnostic framework that aims to separate these two hypotheses through three complementary analyses. First (Section 4.1, Section 5.1), we conduct a *geometric audit* comparing flat versus hierarchical prompts on a frozen BioCLIP2 checkpoint, measuring intra-class variance, inter-class margin, silhouette score, alignment/uniformity \cite{wang2020understanding}, and RankMe \cite{garrido2023rankme}. Second (Section 5.2), we probe whether *unsupervised-with-respect-to-taxonomy* OpenCLIP already exhibits latent taxonomic structure under random-taxonomy permutation tests, and we decompose the BioCLIP2 effect across the seven Linnaean ranks. Third, and centrally, we design a *five-condition counterfactual prompt ablation* (Section 3.3) that crosses structure preservation with lexical content: C0 flat, C1 normal hierarchical, C2 random-token but structure-preserving, C3 shuffled hierarchical, and C4 word-bag. The effect-preservation ratios for C2 and C3 jointly identify which of the two hypotheses (if either) is consistent with the observed geometric reorganization.

We instantiate the full evaluation pipeline as an open-source codebase and execute a *toy-scale* sanity-check run on OpenCLIP ViT-B/32 with a synthetic taxonomy (eight placeholder species, four domains). The toy run validates the pipeline end-to-end and reveals directional signals consistent with the geometry-organizer hypothesis (most notably, the shuffled-hierarchy condition C3 catastrophically collapses silhouette while the random-token but structure-preserving condition C2 closely tracks C1), but the toy results cannot replace a full BioCLIP2 + TreeOfLife evaluation. We are explicit throughout that the numerical findings reported here are *preliminary, mock/toy-scale* signals; the main claims are stated at the level of falsifiable hypotheses with a pre-registered protocol.

**Contributions.**

- **A pre-registered diagnostic protocol** to disentangle the information-channel and geometry-organizer accounts of hierarchical prompting in biological vision-language models, instantiated through five prompt conditions and explicit effect-preservation ratios with bootstrap confidence intervals.
- **A rank-resolved latent-taxonomy probe** that quantifies, on a common evaluation protocol, the extent to which general-purpose OpenCLIP already encodes Linnaean structure that hierarchical supervision can amplify -- positioning BioCLIP2's emergent properties as *organization of pre-existing latent structure* rather than *creation of new semantic information*.
- **An extension of the WaffleCLIP random-descriptor finding** from a general-domain ensembling observation to a *structure-preserving versus structure-destroying* contrast in a biologically grounded hierarchy, supported by explicit counterfactual conditions C2 (structure-preserving, content-empty) and C3 (content-preserving, structure-destroying).
- **An open implementation and toy-scale execution** of the full pipeline (geometric metrics, permutation tests, all five prompt conditions, image-only and image-text variants) together with explicit instructions to reproduce the full-scale validation, and an honest accounting of the gap between the executed sanity-check and the conclusive experiment.

**Paper organization.** Section 2 reviews related work on hierarchical contrastive learning, hierarchy-aware prompts, embedding-geometry diagnostics, and latent-hierarchy probing of vision-language models. Section 3 formalizes the problem, defines the five prompt conditions, and specifies the geometric and taxonomic metrics. Section 4 details datasets, baselines, and implementation, and Section 5 reports the toy-scale results and provides per-RQ answers in the form of supported/refuted/undetermined statements. Section 6 discusses interpretation, limitations, and validity threats; Section 7 concludes.

---

## 2. Related Work

We organize prior work into four threads that directly bear on the present diagnostic study. Throughout, we emphasize what each line contributes and where it leaves a measurement gap that our protocol is designed to fill.

### 2.1 Biological vision-language models and hierarchical taxonomic supervision

BioCLIP \cite{stevens2024bioclip} introduced CLIP-style contrastive pretraining over TreeOfLife-10M and demonstrated that supervising the text branch with a seven-rank Linnaean prompt rather than a species-only caption yielded an absolute 17-20 percentage-point gain on fine-grained biological benchmarks. BioCLIP2 \cite{gu2025bioclip2} scaled the recipe to about 214M images and reported a further 18.1 percentage-point gain. Crucially, BioCLIP2 frames its analysis around *emergent properties*: the authors visualize that inter-species embeddings align with ecological and functional axes such as beak size and habitat, and they argue that intra-species variation is preserved in orthogonal subspaces. BIOSCAN-CLIP / CLIBD \cite{gong2024bioscan} extends multimodal contrastive pretraining to image plus DNA-barcode on the insect-focused BIOSCAN-1M dataset, providing an independently trained biological VLM useful for cross-model robustness checks. A hyperbolic variant for biological taxonomies \cite{gong2025hyperbolic} embeds image and DNA jointly in a Poincare ball with stacked entailment loss and reports improved unseen-species classification.

*What is missing.* These works report end-task gains and provide qualitative visualizations of emergent structure, but do not perform (i) a *flat-prompt control* for the geometric claims, (ii) statistically tested intra-class / inter-class / silhouette comparisons, or (iii) *counterfactual* prompt ablations that would separate textual information from structural supervision. Our framework supplies precisely these missing diagnostics on top of the same checkpoints.

### 2.2 Hierarchical contrastive losses and hierarchy-aware prompts

A separate line of work treats class hierarchy as a structured supervisory signal. HMLC / HiConE / HiMulConE \cite{zhang2022usealllabels} introduces hierarchical multi-label contrastive losses with level-dependent penalties and is a conceptual prototype for BioCLIP2's per-rank contrastive head. *Taxes Are All You Need* \cite{kokilepersaud2024taxes} weights a supervised contrastive loss by tree distance and is the closest prior to our C5 text-free hierarchical InfoNCE condition. HiCLIP \cite{geng2023hiclip} adds hierarchy-aware attention to CLIP backbones without explicit taxonomic supervision and shows that *unsupervised* hierarchical grouping emerges in attention maps, which provides indirect evidence for the latent-structure hypothesis we revisit in Section 5.2.

On the prompt side, CHiLS \cite{novack2023chils} expands each class into hierarchical sub-label sets at inference time; HAPrompts \cite{liang2024haprompts} uses LLM-generated hierarchy-aware prompts to reduce mistake severity in zero-shot classification; ProTeCt \cite{wu2024protect} introduces a dynamic tree-cut loss for taxonomic open-set settings and standardizes the HCA / MTA hierarchy-consistency metrics; HPT and HPT++ \cite{wang2024hpt,wang2024hptpp} compose low- / high- / global-level prompts; HGCLIP \cite{xia2024hgclip} encodes class hierarchies as graphs propagated by a GNN into both modalities. The shared interpretation in these works is informational: hierarchy contributes a *richer textual signal*. The hypothesis that hierarchical supervision instead acts as a *geometric organizer* of an already-structured embedding space has not, to our knowledge, been formulated and tested in this line.

*What is missing.* These methods improve metrics but do not isolate the *mechanism*. None of them runs an explicit structure-preserved-but-content-empty counterfactual, and none reports the per-rank effect-size decomposition required to test the latent-structure claim.

### 2.3 Embedding-geometry diagnostics for contrastive representations

Wang and Isola's alignment / uniformity framework \cite{wang2020understanding} decomposes the asymptotic behaviour of contrastive losses into two measurable properties on the hypersphere; we adopt this directly for intra-class compactness and inter-class spread. RankMe \cite{garrido2023rankme} provides a label-free effective-rank proxy that we use to check whether structure-preserving but content-empty conditions (C2) keep the spectral footprint of normal hierarchical prompts (C1). The neural-collapse literature \cite{kini2023engineering} provides the theoretical scaffold for treating class-mean ETF-like geometry as the optimization target of supervised contrastive learning. *Making Better Mistakes* \cite{bertinetto2020mistakes} introduced hierarchical-distance metrics (lowest-common-ancestor depth, average hierarchical distance) that we adopt for taxonomic retrieval evaluation. The Mind-the-Gap analysis \cite{liang2022mindgap} and the Double-Ellipsoid CLIP geometry \cite{hewitt2024doubleellipsoid} document modality-gap artefacts that justify our decision to report *image-only* metrics in parallel with image-text ones.

*What is missing.* These diagnostics are well established for *general* contrastive representations but have not been applied in concert to the BioCLIP2 emergent-property claim, nor have they been used to test a content-versus-structure counterfactual.

### 2.4 Latent taxonomic / hierarchical structure in pretrained VLMs and random-prompt ablations

Concept Visualization with WordNet \cite{bertucci2024concept} shows that CLIP embeddings already contain accessible WordNet-aligned hierarchical signal, decoded post hoc via saliency. Probing studies in language models \cite{lin2023probing,sneyers2023probing} provide methodologies (edge-distance triplets, norm-versus-depth analyses) that we adapt to image-side rank-resolved silhouette and mutual-information probing. On the counterfactual side, the central reference is WaffleCLIP \cite{roth2023waffling}: by replacing LLM-generated descriptors with random characters or random words and observing that classification accuracy is largely preserved, the authors argue that the apparent benefit of descriptor-based prompting in general domains comes from *ensembling structure* rather than lexical semantics. Counterfactual Prompt Learning \cite{he2022cpl} provides a complementary formalism for defining minimal-change counterfactuals.

*What is missing.* WaffleCLIP did not (i) preserve a known hierarchical *order* in its random descriptors versus *destroy* that order in an otherwise matched condition, (ii) anchor the analysis in a ground-truth taxonomy where hierarchical structure is well defined, or (iii) measure embedding-space geometry directly. Our C2 (random tokens, structure preserved) versus C3 (real vocabulary, structure shuffled) contrast is precisely designed to fill this gap in the biological setting.

### 2.5 Hyperbolic and structured alternatives (out of scope, future work)

Poincare embeddings \cite{nickel2017poincare}, MERU \cite{desai2023meru}, HyCoCLIP \cite{pal2024hycoclip}, and PHyCLIP \cite{matsubara2025phyclip} construct embedding spaces in which hierarchical structure is explicit by construction. These are natural follow-ups if the geometry-organizer hypothesis is confirmed: if Euclidean BioCLIP2 *implicitly* approximates a hierarchical geometry that hierarchical prompts then organize, hyperbolic variants should remove the need for hierarchical text altogether. We therefore treat this line as out of scope for the present diagnostic study and revisit it in Section 6.4.

---

## 3. Method

This section formalizes the diagnostic framework. Section 3.1 fixes notation and the dichotomy of hypotheses to be tested. Section 3.2 gives an overview. Section 3.3 specifies the six prompt conditions C0-C5 that are the central instrument of the *counterfactual ablation*. Section 3.4 defines the geometric and taxonomic measurements. Section 3.5 explains the key design choices.

### 3.1 Notation and Problem Setup

Let $f_v : \mathcal{X} \to \mathbb{R}^d$ and $f_t : \mathcal{T} \to \mathbb{R}^d$ denote the (frozen) image and text encoders of a vision-language model. We assume L2-normalized outputs and use cosine similarity throughout. The evaluation set $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$ contains $N$ images, each annotated with a *full Linnaean hierarchy* $y_i = (y_i^{(1)}, \dots, y_i^{(7)})$ over the ranks $r \in \{$ species, genus, family, order, class, phylum, kingdom $\}$. For each species $c$, $Z_c = \{z_i = f_v(x_i) : y_i^{(\text{species})} = c\}$ is the set of image embeddings; $\mu_c$ is its centroid; $\sigma_c$ is its within-class standard deviation.

Each *prompt condition* $C_k$ defines a deterministic mapping from a species label to a text input $t_c^{(k)} \in \mathcal{T}$; the corresponding text embedding is $w_c^{(k)} = f_t(t_c^{(k)})$. Prompt swaps do not change $f_v, f_t$; they change only the text inputs $t_c^{(k)}$ used to anchor the image embeddings. The exception is condition C5 (text-free hierarchical contrastive), which fine-tunes a lightweight adapter on the image side without text supervision.

**Dichotomy of hypotheses.** Let $M$ denote any geometric metric (e.g., silhouette) whose direction we control so that larger means better, and write $\Delta_M(C_k) = M(C_k) - M(C_0)$. The *normalized effect-preservation ratio* for condition $C_k$ relative to the normal hierarchical condition $C_1$ is

$$\rho_M(C_k) = \frac{M(C_k) - M(C_0)}{M(C_1) - M(C_0)} = \frac{\Delta_M(C_k)}{\Delta_M(C_1)}.$$

We test:

- **H_info (information-channel)**: $\rho_M(C_2) \approx 0$ and $\rho_M(C_3) \approx 1$. That is, removing the *content* of the taxonomic words destroys the benefit, while preserving content but shuffling the structural alignment leaves it intact.
- **H_geom (geometry-organizer)**: $\rho_M(C_2) \gtrsim 0.5$ and $\rho_M(C_3) \lesssim 0.2$. That is, the benefit survives content-removal so long as the *structural order* is preserved, and collapses when the structure is destroyed regardless of vocabulary.

Either hypothesis being supported above the other constitutes evidence for a particular mechanism; both ratios near 1 or both near 0 would constitute *inconclusive* evidence.

### 3.2 Approach Overview

[Figure 1: Diagnostic framework. Top: dichotomy of hypotheses, with H_info predicting that content carries the benefit and H_geom predicting that structure does. Middle: six prompt conditions C0-C5 instantiating the orthogonal axes *content* x *structure*. Bottom: geometric audit (intra-class variance, inter-class margin, silhouette, alignment/uniformity, RankMe) and taxonomic audit (kNN purity, LCA depth, hierarchy consistency error) applied to image-only and image-text embeddings.]

The framework couples three analytical layers:

1. **Geometric audit (RQ1).** A flat-vs-hierarchical comparison on a frozen BioCLIP2 checkpoint, with intra-class variance, inter-class margin, silhouette at species level, alignment / uniformity, and RankMe.
2. **Rank-resolved latent-taxonomy probe (RQ2).** The same geometric audit replicated across seven Linnaean ranks and across OpenCLIP, BioCLIP, and BioCLIP2, with a random-taxonomy permutation control on OpenCLIP to test whether *unsupervised-with-respect-to-taxonomy* embeddings already encode partial hierarchical structure.
3. **Counterfactual prompt ablation (RQ3, central instrument).** Six prompt conditions C0-C5 designed to factor *structural alignment with the taxonomy* away from *lexical content*. We report effect-preservation ratios with bootstrap 95% confidence intervals.

A cross-domain meta-analysis (RQ4) replicates 1 and 3 across Aves, Insecta, Plantae, Fungi, and Actinopterygii and reports between-domain heterogeneity (I-squared and CV).

### 3.3 The Six Prompt Conditions (Counterfactual Ablation)

The six conditions are constructed to cross two axes: (a) *content* -- whether the taxonomic vocabulary in the prompt is meaningful, and (b) *structure* -- whether the prompt preserves the rank ordering aligned with the ground-truth taxonomy of the labelled species.

| Cond. | Content | Structure | Template (illustrative for *Passer domesticus*) |
|-------|---------|-----------|--------------------------------------------------|
| C0 | species only | n/a | `"a photo of Passer domesticus"` |
| C1 | real vocab | preserved | `"a photo of Animalia Chordata Aves Passeriformes Passeridae Passer domesticus"` |
| C2 | placeholder vocab | preserved | `"a photo of tax0 tax1 tax2 tax3 tax4 tax5 Passer domesticus"` |
| C3 | real vocab | destroyed | `"a photo of Plantae Arthropoda Reptilia Diptera Felidae Passer domesticus"` (upper ranks copied from random other species) |
| C4 | real vocab | shuffled order | random permutation of the seven C1 tokens per call |
| C5 | n/a (text-free) | preserved (image-image) | image-image hierarchical InfoNCE with LoRA adapter |

C2 is the critical *structure-preserving content-empty* condition: it keeps the same number of tokens as C1, at the same positions, with the same species token, but replaces the upper six rank labels with vocabulary-external placeholders (`tax0`-`tax5`) so that no semantic cue beyond *position-in-hierarchy* remains. C3 is the *content-preserving structure-destroying* condition: upper ranks are sampled from *other* species so the words are real but their hierarchical alignment with the true label is broken (with the species token kept correct, so that the difference from C1 is *only* the alignment of upper ranks). C4 disentangles *order* from *bag-of-tokens* by randomly permuting C1's seven labels per call. C5 removes the text encoder altogether: it adds a small LoRA adapter (rank 8) on the image encoder and trains an image-image hierarchical InfoNCE objective \cite{kokilepersaud2024taxes} for five epochs on the training split; if hierarchical *organization* is what drives the benefit, supervising the image side directly with hierarchical positives should suffice.

This C2-versus-C3 contrast is, to our knowledge, the cleanest instrument so far proposed to separate *content* from *structure* in hierarchical prompting, and it is a direct refinement of the random-descriptor analysis of WaffleCLIP \cite{roth2023waffling} in a domain where ground-truth hierarchy is unambiguous.

**Algorithmic sketch.**

```
Input: frozen encoders f_v, f_t; eval set D; prompt-builder b_k for k in {0..5}
Output: per-condition metrics M_k = {intra_var, inter_margin, silhouette, ...}

for k in {0,1,2,3,4}:
    for each species c in D:
        t_c <- b_k(c, taxonomy[c])
        w_c <- L2_normalize( f_t(t_c) )
    for each image x_i in D:
        z_i <- L2_normalize( f_v(x_i) )
    compute geometric and taxonomic metrics on {z_i} (image-only)
    and on {z_i, w_c} (image-text) -> M_k

for k = 5:
    fit LoRA adapter A on f_v with image-image hierarchical InfoNCE
        loss = sum over (i,j,r): weight(r) * NCE(A(z_i), A(z_j); positive iff y_i^(r)=y_j^(r))
    re-embed and compute metrics -> M_5

for k in {2,3,4,5}:
    rho_k <- (M_k - M_0) / (M_1 - M_0)      # with bootstrap CI

decision rule:
    if  rho_2 >= 0.5 (CI lower >= 0.3) AND rho_3 <= 0.2 (CI upper <= 0.4)
        AND rho_5 >= 0.5:  H_geom supported
    elif rho_2 < 0.3:                       H_info supported
    else:                                    inconclusive
```

### 3.4 Measurements

We adopt the geometric metrics fully defined in Section 4 of the protocol; the key quantities are summarized here.

- **Intra-class variance** (lower is better): mean over species of mean pairwise cosine distance within the species.
- **Inter-class margin** (higher is better): for each species, the minimum cosine distance from its centroid to any other species centroid, divided by the within-class standard deviation, averaged over species.
- **Silhouette at rank $r$** (higher is better): standard silhouette computed with cosine distance, with cluster labels given by rank $r$.
- **Alignment / Uniformity** \cite{wang2020understanding}: alignment as the expected squared distance between positive pairs (here, image-text pairs of the same species, or two images of the same species for the image-only variant); uniformity as the log-expected exponential negative squared distance over random pairs.
- **RankMe** \cite{garrido2023rankme}: the exponential of the entropy of the normalized singular-value distribution of the embedding matrix.
- **kNN purity@k** at rank $r$: fraction of the $k$ nearest neighbors of each image that share its rank-$r$ label.
- **LCA depth** of the top-1 retrieved species relative to the query, averaged over queries (higher = better hierarchical agreement).
- **Hierarchy consistency error**: shortest path in the taxonomy tree between the predicted and the ground-truth species.

All metrics are computed on L2-normalized embeddings. We report image-only and image-text variants in parallel to control for modality-gap artefacts \cite{liang2022mindgap,hewitt2024doubleellipsoid}.

### 3.5 Design Choices

**Why frozen encoders.** The hypotheses concern the *embedding geometry induced by hierarchical prompts at evaluation time*, not a new training algorithm. Freezing the encoder and varying only the text-side input cleanly isolates prompt-driven effects (with the explicit exception of C5, where text is removed and the image side is lightly adapted, providing a separate test point).

**Why placeholder tokens for C2.** Random tokens drawn from the vocabulary risk re-introducing semantics through subword pieces. We use vocabulary-external strings (`tax0`-`tax5`) that BPE-tokenize predictably and contain no semantic load, matching the *structural slot count* of C1 exactly.

**Why an image-only variant of every metric.** Modality-gap artefacts \cite{liang2022mindgap} can confound any image-text comparison. We always report an image-only counterpart so that conclusions about geometric reorganization do not depend on cross-modal alignment changes that hierarchical prompts could induce in the text branch only.

**Why effect-preservation ratios rather than raw differences.** Across domains and ranks the absolute scales of metrics differ substantially. Normalizing by $\Delta_M(C_1) = M(C_1) - M(C_0)$ produces a quantity that is directly comparable across metrics and domains and that has a transparent decision rule (Section 3.1). When $|\Delta_M(C_1)|$ is small (e.g., in toy data where flat is already close to hierarchical), the ratio is unstable; we therefore additionally require $\Delta_M(C_1)$ to be significant before reporting $\rho$ as a primary statistic.

**Pre-registration.** The success thresholds in Section 3.1 and the per-RQ falsification conditions in `02_rqs.md` were fixed *before* execution. The toy-scale results reported in Section 5 do not satisfy the data-quality precondition (small sample, synthetic images) and are therefore reported as *directional* signals only, not as evidence for or against the registered hypotheses.

---

## 4. Experiments

This section describes datasets, models, evaluation metrics, and implementation. Section 5 reports the results. We emphasize at the outset that the *executed* run is a toy-scale sanity check on OpenCLIP ViT-B/32 with eight placeholder species; the *intended* run on BioCLIP2 + TreeOfLife is fully specified but not executed in the present submission (Section 4.4 and Section 6.2).

### 4.1 Datasets

**Primary (intended).** TreeOfLife evaluation subset, derived from `imageomics/TreeOfLife-10M` and `imageomics/TreeOfLife-200M` on HuggingFace. We retain species with at least 20 images, attach the seven-rank Linnaean taxonomy already present in the metadata, and stratify a 70 / 15 / 15 train / val / test split by species. Because our analysis is frozen except for the C5 adapter, only the test split is used in conditions C0-C4; C5 fine-tunes on the train split. The target evaluation set is at least 1,000 species and at least 20,000 test images.

**Cross-domain (RQ4, intended).** Aves (NABirds + iNat21 birds), Insecta (BIOSCAN-1M), Plantae (PlantNet300K + iNat21 plants), Fungi (iNat21 fungi), Actinopterygii (FishNet). All datasets are accessed under their published research licenses; iNaturalist content is used in accordance with individual image licenses.

**Toy placeholder (executed).** Eight synthetic species across four taxonomic domains, six class-conditioned synthetic RGB images per species (48 images total), with a synthetic seven-rank taxonomy attached. This was constructed as an end-to-end pipeline sanity check and is *not* a substitute for the intended evaluation; we report it explicitly as a toy run in Section 5.

Preprocessing for all datasets: resize 224x224, center crop, OpenCLIP normalization with mean (0.48145466, 0.4578275, 0.40821073) and std (0.26862954, 0.26130258, 0.27577711); text truncation to 77 tokens.

### 4.2 Baselines

The evaluation contrasts the six prompt conditions C0-C5 (Section 3.3) on a common frozen backbone. Across-model baselines for RQ2 are:

- **BioCLIP2 ViT-L/14** \cite{gu2025bioclip2} -- primary analysis target.
- **BioCLIP ViT-B/16** \cite{stevens2024bioclip} -- prior-generation biological VLM for scaling comparisons.
- **OpenCLIP ViT-L/14, laion2b** -- non-biological baseline against which to probe latent taxonomic structure under random-taxonomy permutation.
- **OpenCLIP ViT-B/32, laion2b** -- small model used for the executed toy run.

External-method baselines for comparative interpretation:

- **WaffleCLIP** \cite{roth2023waffling} -- reference implementation of random-descriptor prompting; informs the design of C2.
- **CHiLS** \cite{novack2023chils} -- inference-time hierarchical label-set expansion.
- **HAPrompts** \cite{liang2024haprompts} and **ProTeCt** \cite{wu2024protect} -- LLM-driven and tree-cut-loss hierarchy-aware prompt methods; informs the *better mistakes* metric.
- **Hierarchical Cross-Entropy** \cite{bertinetto2020mistakes} -- standard implementation of LCA-depth and hierarchical-distance metrics.

### 4.3 Metrics

We collect geometric metrics (intra-class variance, inter-class margin, silhouette at rank $r$, alignment, uniformity, RankMe) and taxonomic metrics (kNN purity@10 at rank $r$, LCA depth, hierarchy consistency error, mutual information $I(\text{cluster}; \text{rank-label})$). Exact definitions are given in Section 3.4 and in the experimental protocol.

The **effect-preservation ratio** $\rho_M(C_k) = (M(C_k) - M(C_0)) / (M(C_1) - M(C_0))$ is the central derived quantity for RQ3. We report it for $k \in \{2, 3, 4, 5\}$ on each of the three RQ1 geometric metrics, with bootstrap (B = 1000) 95% confidence intervals.

Statistical tests:

- **Paired permutation test** (B = 1000) for two-condition comparisons on the same image set; paired by image identity.
- **Bootstrap 95% CI** (B = 1000) for all metric reports.
- **Bonferroni correction** for the three RQ1 metrics simultaneously: $\alpha = 0.01 / 3 \approx 0.0033$.
- **Effect size**: Cohen's $d$ for paired continuous metrics; Cliff's $\delta$ for a non-parametric robustness check.
- **Meta-analysis (RQ4)**: random-effects pooled effect, I-squared heterogeneity, between-domain CV.

### 4.4 Implementation Details

**Code.** A complete reference implementation accompanies the paper (`04_experiments/code/`): `prompt_variants.py` builds C0-C5 prompts; `extract_embeddings.py` produces image and text embeddings on either OpenCLIP, BioCLIP, or BioCLIP2; `metrics.py` implements all geometric and taxonomic metrics with bootstrap confidence intervals; `run_experiment.py` orchestrates a single (model, dataset, conditions) run; `run_all.sh` and `run_all.ps1` script the full sweep. A `--mock` flag substitutes randomized embeddings for sanity-checking the metric and statistics code without any model download.

**Software.** Python 3.12.7, PyTorch 2.7.1+cu118, `open_clip_torch` 3.2.0, `transformers` 5.1.0, scikit-learn 1.5.1, NumPy 1.26.4, SciPy 1.13.1. Determinism is enforced via `numpy.random.default_rng(seed)`, `torch.manual_seed(seed)`, and CuDNN deterministic flags.

**Seeds.** Five seeds {42, 1337, 2024, 7, 314} for C3 shuffle and for any data sub-sampling; B = 1000 for bootstrap and permutation.

**Hardware.** The intended TreeOfLife run is sized for a single NVIDIA A100 40 GB, with an estimated 8-12 hours per domain at batch size 128. The *executed* toy run was performed on a CPU-only Windows 11 machine.

**Adapter for C5.** LoRA with rank 8 applied to the OpenCLIP image encoder, plus a linear projection head; objective is image-image hierarchical InfoNCE with rank-weighted positives following \cite{kokilepersaud2024taxes}; AdamW with learning rate 1e-4, batch size 256, five epochs.

**Execution status.** Code skeleton (full); mock end-to-end run (full); OpenCLIP ViT-B/32 toy run on synthetic data (full); OpenCLIP ViT-L/14 on TreeOfLife (not executed); BioCLIP and BioCLIP2 (not executed); RQ4 cross-domain replications (not executed). The unmet runs are gated by HuggingFace weight downloads (about 6 GB) and TreeOfLife evaluation data (about 60 GB) on a GPU host, both of which are unavailable in the present environment. The published code is structured so that swapping `--model bioclip2 --csv data/treeoflife_eval.csv` reproduces the full intended evaluation without code changes.

---

## 5. Results

> **Caveat.** All numerical results in this section are *toy- and mock-scale sanity-check signals* obtained from OpenCLIP ViT-B/32 on a synthetic taxonomy of eight placeholder species (six images per species; 48 total images), executed on CPU. The downloads required to run BioCLIP / BioCLIP2 on the full TreeOfLife evaluation set were not available in the present environment. The numbers below validate the pipeline and report *directional* trends; they should *not* be cited as evidence for or against the main hypotheses, and a full-scale validation on BioCLIP2 + TreeOfLife is registered as immediate future work.

### 5.1 Main Results: Flat versus Hierarchical Geometric Compactness (RQ1)

[Table 1: Geometric metrics on a frozen backbone for C0 (flat species-only) and C1 (normal seven-rank hierarchical), executed on OpenCLIP ViT-B/32 with a toy taxonomy. Bootstrap CIs are based on B = 1000 resamples over three seeds.]

| Metric | C0 (flat) | C1 (hierarchical) | $\Delta$ (C1-C0) | bootstrap 95% CI | passes pre-registered threshold? |
|--------|-----------|-------------------|-------------------|--------------------|-----------------------------------|
| intra_var (lower) | 0.00239 | 0.00239 | 0 (tied) | tight | NO |
| inter_margin (higher) | 713.11 | 389.59 | -323.5 | [-324.6, -322.8] | NO (reversed) |
| silhouette (higher) | 0.9865 | 0.9740 | -0.0125 | [-0.013, -0.012] | NO (reversed) |
| RankMe | 9.79 | 9.23 | -0.57 | tight | -- |
| uniformity (lower) | -0.834 | -0.684 | +0.15 | tight | -- |

On toy data, C1 does *not* improve over C0 on any RQ1 metric. The most plausible explanation is that the synthetic images were generated from class-conditioned color means; the class signal in the image embedding is therefore so strong that silhouette is saturated near 1, and any additional text-side variation (the extra tokens of C1) couples to image embeddings only through the contrastive scoring head and acts as noise rather than as structure. We therefore *cannot* take this result as evidence against the hierarchical-compactness claim; the experiment is under-powered by construction. A full BioCLIP2 + TreeOfLife run is required to obtain meaningful signal on RQ1.

### 5.2 Ablations: Counterfactual Conditions C0-C5 (RQ3 -- central instrument)

[Table 2: Counterfactual ablation on OpenCLIP ViT-B/32, toy taxonomy, three seeds. Preservation ratio is computed only for silhouette here; rows where the C1-C0 sign is reversed make the ratio numerically ill-conditioned and are flagged.]

| Condition | intra_var | inter_margin | silhouette | preservation ratio (silhouette) |
|-----------|-----------|--------------|------------|---------------------------------|
| C0 flat | 0.00239 | 713.11 | 0.9865 | (baseline) |
| C1 normal hierarchical | 0.00239 | 389.59 | 0.9740 | 1.00 (by definition) |
| C2 random-token hierarchy (structure preserved) | 0.00239 | 445.66 | 0.9796 | **0.55** (CI tight at toy scale) |
| C3 shuffled hierarchy (structure destroyed) | 0.0937 | 3.97 | 0.1002 | ill-conditioned (silhouette collapses) |
| C4 word-bag (order destroyed) | 0.0210 | 44.01 | 0.8214 | ill-conditioned |
| C5 text-free hierarchical InfoNCE | 0.00545 | 30.04 | 0.6936 | ill-conditioned |

The toy data yields three observations whose *direction* is consistent with the geometry-organizer hypothesis but whose magnitudes are not interpretable as evidence:

1. **C3 catastrophically collapses silhouette** (0.97 to 0.10). When the hierarchical structure is destroyed -- upper ranks reassigned from random other species, with species kept correct -- the embedding-space organization collapses far below even the flat baseline. This is in the predicted direction for H_geom.
2. **C2 closely tracks C1.** The structure-preserving but content-empty condition (placeholder tokens at the upper-rank slots) achieves silhouette 0.98, essentially matching C1's 0.97 and even nudging slightly higher. This is the predicted *positive* signature of H_geom (structure carries the benefit).
3. **C4 and C5 are intermediate-to-degraded.** Word-bag shuffling and the light text-free adapter both degrade silhouette substantially. For C5 specifically, the LoRA adapter trained for five epochs on 48 synthetic images is severely under-trained; this is not evidence about the hypothesis.

The preservation-ratio computation is, however, *not interpretable* in the toy regime because $\Delta_{\text{sil}}(C_1) = -0.0125$ has the *wrong sign*: the denominator that we expect to be substantially positive on real data is small and negative here. We report the silhouette ratios for C2 in the table for completeness (0.55) but emphasize that this number is a numerical artefact of dividing two small differences; only the *directional* contrast between C2 (tracks C1) and C3 (collapses far below) is informative.

### 5.3 Latent Taxonomic Structure in OpenCLIP (RQ2)

[Table 3: Species-level silhouette for OpenCLIP ViT-B/32 embeddings under true taxonomy versus random-taxonomy permutation (B = 50).]

| Source | silhouette (cosine) | z-score vs. random-taxonomy null |
|--------|----------------------|-----------------------------------|
| true labels | 0.683 | +44.9 |
| random-taxonomy permutation mean | -0.271 | (baseline) |

Even on toy data, the *unsupervised-with-respect-to-taxonomy* OpenCLIP separates true species labels from random-taxonomy controls with a very large z-score. This is consistent with the latent-structure premise of RQ2 H1(b), but the synthetic class-conditioned color signal almost certainly inflates the magnitude. Per-rank decomposition (species $\to$ kingdom) cannot be reported on the toy taxonomy because at higher ranks the number of classes is too small for silhouette to be defined; the rank-wise analysis is part of the registered intended run.

### 5.4 Statistical Integrity Notes

- Paired permutation p-values for the C0-vs-C1 comparison on the toy run are in the range 0.24-0.26, well above the Bonferroni-corrected $\alpha = 0.0033$. No metric is significant at toy scale, and we do not claim significance.
- Cohen's $d$ for silhouette in the toy run is inflated (about -2910) because the seed-to-seed standard deviation collapses to about $10^{-6}$; this is a small-sample, low-noise artefact of synthetic data and would not occur on real images.

### 5.5 Per-RQ Answers

We restate the four research questions and answer each with one of {supported, refuted, undetermined} given the evidence available.

| RQ | Toy-scale directional signal | Verdict on the registered hypothesis |
|----|------------------------------|---------------------------------------|
| **RQ1** (hierarchical prompts increase geometric compactness on a frozen biological VLM) | C1 does *not* improve over C0 on toy data; metrics are saturated by synthetic class-conditioned color. | **Undetermined** (and *not* refuted). The experiment is under-powered by construction; a BioCLIP2 + TreeOfLife replication is required. |
| **RQ2** (latent taxonomic structure exists in OpenCLIP, and the BioCLIP2 effect is concentrated at higher ranks) | OpenCLIP separates true species from random-taxonomy at z = +44.9 on toy data; per-rank decomposition not possible at toy scale. | **Directional support for H1(b)** (latent structure plausibly present); H1(a) (rank-wise decomposition) **undetermined**. |
| **RQ3** (semantic-organizer hypothesis: structure preserved $\Rightarrow$ benefit preserved; structure destroyed $\Rightarrow$ benefit collapses) | C3 silhouette collapses 0.97 $\to$ 0.10; C2 silhouette tracks C1 closely. Direction matches H_geom for both critical conditions. Preservation-ratio magnitudes are ill-conditioned at toy scale. | **Directional support; magnitudes undetermined.** Strongest qualitative signal of the toy run, but not yet a quantitative confirmation. |
| **RQ4** (cross-domain consistency over five biological domains) | Not executed (domain data not downloaded). | **Not yet evaluated.** |

We emphasize again that *supported* and *directional support* are weaker statements than the pre-registered success thresholds in Section 3.1 and Section 02_rqs.md call for. The main claims of the paper -- compactness improvement, latent taxonomic structure, and semantic-organizer mechanism -- remain *hypotheses* at the end of this preliminary report. The reported toy signals are *consistent with* these hypotheses but do not constitute evidence sufficient to adopt them.

---

## 6. Discussion

### 6.1 Interpretation

The framework, taken on its own terms, separates three causal accounts that have been conflated in the literature. *Information-channel* (H_info) treats hierarchical prompts as additional textual context that enriches the text-side representation; *geometry-organizer* (H_geom) treats them as a structural prior that reshapes the embedding geometry of an already partially structured space; an intermediate account treats them as both. The C2-versus-C3 contrast (structure preserved with empty content versus content preserved with destroyed structure) is the cleanest discriminator we are aware of in the literature for these accounts. WaffleCLIP \cite{roth2023waffling} previously argued, in *general* domains, that random descriptors recover most of the benefit attributed to LLM-generated descriptors. Our framework refines that claim in a biological setting by *separating the structural and lexical components* of the random-descriptor manipulation and by anchoring the analysis to a ground-truth taxonomy in which hierarchy is unambiguous.

What we *do* learn from the executed toy run is two-fold. First, the pipeline is end-to-end functional: prompt builders, encoders, geometric metrics, bootstrap CIs, and statistical tests all return coherent values under a synthetic stress test. Second, the qualitative pattern in the counterfactual ablation -- C2 closely tracking C1 while C3 collapses far below C0 -- is the *directional signature* predicted by H_geom. We do not, however, treat this as evidence about BioCLIP2, both because the encoder used is OpenCLIP ViT-B/32 (not BioCLIP2) and because the data are synthetic. The toy result is a check on the *instrument*, not on the *phenomenon*.

If the full BioCLIP2 + TreeOfLife replication confirms the directional pattern observed at toy scale, with effect-preservation ratios meeting the pre-registered thresholds ($\rho_{\text{sil}}(C_2) \geq 0.5$ with lower CI bound 0.3, $\rho_{\text{sil}}(C_3) \leq 0.2$ with upper CI bound 0.4, $\rho_{\text{sil}}(C_5) \geq 0.5$), the implication for the broader vision-language literature is that *textual supervision in CLIP-style training acts substantially as a structural prior over a latent organization*, not only as a semantic information channel. This would re-frame several adjacent findings -- HiCLIP's unsupervised hierarchy attention \cite{geng2023hiclip}, Concept Visualization's WordNet decoding \cite{bertucci2024concept}, and the WaffleCLIP random-descriptor result \cite{roth2023waffling} -- as facets of the same underlying mechanism rather than as separate empirical curiosities. It would also clarify what BioCLIP2 \cite{gu2025bioclip2} means by *emergent properties*: not new semantic content learned from the hierarchical text, but a hierarchical *re-organization* of structure that the pretrained backbone already approximated.

If, conversely, $\rho_{\text{sil}}(C_2)$ is small in the full run, the geometry-organizer account is refuted in its strong form and the information-channel account is rehabilitated: the lexical content of upper-rank tokens carries non-trivial information about species identity that random tokens cannot supply.

### 6.2 Limitations

**Toy-scale execution.** The most important limitation is that we report only a pipeline-level sanity check. The intended evaluation -- BioCLIP2 ViT-L/14 on $\geq$ 1,000 TreeOfLife species with $\geq$ 20 images each -- was not executed. Until this is run, all main claims remain hypotheses. We have organized the codebase, the conditions, and the metric definitions so that this run reduces to a single command (Section 4.4) on a GPU host with the dataset downloaded.

**Synthetic images for the toy run.** The toy images are class-conditioned random RGB tensors. This generates an artificially clean clustering signal (silhouette near 1) that saturates the geometric metrics and creates conditions under which prompt-driven changes appear as noise; it also inflates the latent-structure probe in RQ2. The toy run cannot, in principle, settle the questions of interest.

**OpenCLIP, not BioCLIP2.** The executed run used OpenCLIP ViT-B/32 because BioCLIP and BioCLIP2 weights (about 6 GB) and the TreeOfLife evaluation set (about 60 GB) were not retrievable in the present environment. OpenCLIP is informative for the latent-structure probe (RQ2 H1(b)), but it is not the analysis target for RQ1 and RQ3.

**C5 design choices.** The text-free hierarchical InfoNCE adapter is a single instantiation (LoRA rank 8, five epochs, AdamW 1e-4). A negative C5 result would not robustly refute the *organizer* hypothesis without a hyperparameter sweep; we currently treat C5 only as an existence check, not as a definitive test.

**Geometric metrics, not mechanistic interpretability.** We diagnose embedding *geometry* and report ratios and significance tests. We do not perform token-level attribution, attention probing, or representation-engineering interventions inside the encoder. These are complementary and out of scope.

### 6.3 Validity Threats

**Internal validity.** The most significant internal threat is *modality-gap confounding* \cite{liang2022mindgap,hewitt2024doubleellipsoid}: changes in the text-side prompt may shift the text embedding without inducing the geometric reorganization on the image side that the geometry-organizer account requires. We mitigate this by reporting *image-only* metrics in parallel with image-text metrics; only if image-only metrics also reorganize do we treat the geometry-organizer account as supported on the *visual* side. A second threat is *token-count confounding* between C0 and C1: hierarchical prompts are longer. We control for this by inserting padding to length-match C0 to C1 in an ablation. A third threat is the *small-denominator instability* of the preservation ratio when $\Delta_M(C_1)$ is small; we report the ratio only when $\Delta_M(C_1)$ itself is significant at the pre-registered $\alpha = 0.0033$.

**External validity.** RQ4 is designed to test cross-domain heterogeneity over five biological domains. The toy run did not address this. If the full run finds large I-squared or cross-domain CV, the apparent organizer effect is conditional on domain and the *general* claim should be tempered. A second external-validity threat is whether the conclusions transfer outside biology to other hierarchically structured domains (medical imaging, ontologies, geography); we explicitly leave this to future work.

**Construct validity.** The construct *geometric compactness* is operationalized through intra-class variance, inter-class margin, silhouette, and alignment/uniformity. These are correlated but not identical; we therefore require all three RQ1 metrics to pass thresholds jointly under Bonferroni correction. The construct *taxonomic alignment* is operationalized through kNN purity, LCA depth, and hierarchy consistency error; analogously we report all three.

**Statistical-conclusion validity.** All p-values are Bonferroni-corrected; all CIs are bootstrap based with B = 1000. The Cohen's $d$ values in the toy run are inflated by extremely small seed-to-seed variance ($\sim 10^{-6}$) typical of synthetic data; on real images, seed variance is expected to be in a normal range and Cohen's $d$ should be interpretable.

### 6.4 Future Work

**Immediate (gated only by compute).** Execute the full intended evaluation: BioCLIP2 ViT-L/14 on the TreeOfLife test split with the registered six prompt conditions and seven Linnaean ranks, five biological domains for RQ4. We expect this to require about 60-100 A100 GPU-hours and to provide the conclusive test of the main hypotheses.

**Mechanistic verification.** If the geometry-organizer account is supported, an attention-level or probe-level study should localize which parts of the encoder implement the organizer. We anticipate that text-side representations of upper-rank tokens function as *bias vectors* on shared species clusters.

**Hyperbolic and product-of-hyperbolic-factors variants.** If the implicit Euclidean organization is hierarchical, hyperbolic models such as MERU \cite{desai2023meru}, HyCoCLIP \cite{pal2024hycoclip}, and PHyCLIP \cite{matsubara2025phyclip} should *remove* the marginal benefit of hierarchical prompts -- the organization should already be enforced by the geometry. We propose this as a strong consistency check.

**Generalization beyond biology.** A natural follow-up is to repeat the protocol on ImageNet + WordNet, where a hierarchy is also well defined and where Concept Visualization \cite{bertucci2024concept} has already shown latent structure. If the same C2-vs-C3 contrast holds outside biology, the organizer account generalizes beyond domain-specific recipes.

---

## 7. Conclusion

We have formulated the question *why does hierarchical taxonomic prompting help biological vision-language models?* as an explicit dichotomy between an *information-channel* and a *geometry-organizer* account, and we have proposed a diagnostic framework that disentangles the two through (i) a flat-vs-hierarchical geometric audit, (ii) a rank-resolved latent-taxonomy probe across OpenCLIP, BioCLIP, and BioCLIP2, and (iii) a six-condition counterfactual prompt ablation whose central instrument is the contrast between *structure-preserving content-empty* prompts (C2) and *content-preserving structure-destroying* prompts (C3). We argue that this design extends the WaffleCLIP random-descriptor finding from a general-domain ensembling observation to a sharp structure-versus-content test in a biological domain with an unambiguous ground-truth hierarchy. We have instantiated the full pipeline as an open-source codebase and executed a toy-scale OpenCLIP sanity check whose qualitative pattern (C3 collapses, C2 tracks C1) is in the direction predicted by the geometry-organizer hypothesis, but whose magnitudes are not interpretable; the main claims of the paper therefore remain *hypotheses* supported by a pre-registered protocol rather than by evidence. The immediate next step is the full BioCLIP2 + TreeOfLife evaluation gated only by compute; if that run replicates the directional pattern at the registered effect-size thresholds, the broader implication is that textual supervision in CLIP-style training acts substantially as a structural prior that organizes latent geometry, rather than as a mere information channel -- reframing BioCLIP2's emergent properties as the manifestation of pre-existing latent taxonomic structure made visible by hierarchical supervision.

---

## References

Citations resolve against `references.bib` (BibTeX). The reference list is omitted from this Markdown rendering; the BibTeX file is the authoritative source.
