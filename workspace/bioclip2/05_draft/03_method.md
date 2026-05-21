# 3. Method

This section formalizes the diagnostic framework. Section 3.1 fixes notation and the dichotomy of hypotheses to be tested. Section 3.2 gives an overview. Section 3.3 specifies the five prompt conditions C0-C4 that are the central instrument of the *counterfactual ablation*. Section 3.4 defines the geometric and taxonomic measurements. Section 3.5 explains the key design choices.

## 3.1 Notation and Problem Setup

Let $f_v : \mathcal{X} \to \mathbb{R}^d$ and $f_t : \mathcal{T} \to \mathbb{R}^d$ denote the (frozen) image and text encoders of a vision-language model. We assume L2-normalized outputs and use cosine similarity throughout. The evaluation set $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$ contains $N$ images, each annotated with a *full Linnaean hierarchy* $y_i = (y_i^{(1)}, \dots, y_i^{(7)})$ over the ranks $r \in \{$ species, genus, family, order, class, phylum, kingdom $\}$. For each species $c$, $Z_c = \{z_i = f_v(x_i) : y_i^{(\text{species})} = c\}$ is the set of image embeddings; $\mu_c$ is its centroid; $\sigma_c$ is its within-class standard deviation.

Each *prompt condition* $C_k$ defines a deterministic mapping from a species label to a text input $t_c^{(k)} \in \mathcal{T}$; the corresponding text embedding is $w_c^{(k)} = f_t(t_c^{(k)})$. Prompt swaps do not change $f_v, f_t$; they change only the text inputs $t_c^{(k)}$ used to anchor the image embeddings.

**Dichotomy of hypotheses.** Let $M$ denote any geometric metric (e.g., silhouette) whose direction we control so that larger means better, and write $\Delta_M(C_k) = M(C_k) - M(C_0)$. The *normalized effect-preservation ratio* for condition $C_k$ relative to the normal hierarchical condition $C_1$ is

$$\rho_M(C_k) = \frac{M(C_k) - M(C_0)}{M(C_1) - M(C_0)} = \frac{\Delta_M(C_k)}{\Delta_M(C_1)}.$$

We test:

- **H_info (information-channel)**: $\rho_M(C_2) \approx 0$ and $\rho_M(C_3) \approx 1$. That is, removing the *content* of the taxonomic words destroys the benefit, while preserving content but shuffling the structural alignment leaves it intact.
- **H_geom (geometry-organizer)**: $\rho_M(C_2) \gtrsim 0.5$ and $\rho_M(C_3) \lesssim 0.2$. That is, the benefit survives content-removal so long as the *structural order* is preserved, and collapses when the structure is destroyed regardless of vocabulary.

Either hypothesis being supported above the other constitutes evidence for a particular mechanism; both ratios near $1$ or both near $0$ would constitute *inconclusive* evidence.

**A degenerate-denominator regime.** The ratio $\rho_M$ is well-behaved only when $\Delta_M(C_1)$ has the sign predicted by H_info or H_geom; in our runs it does not, because species-level silhouette under hierarchical prompting comes out *below* the flat baseline ($\Delta_M(C_1) < 0$ in every executed run). This is itself an empirical claim about the underlying mechanism (it is precisely what H_geom predicts in the saturated-species regime; see Section 5.1). When $\Delta_M(C_1) < 0$ we therefore base the decision on a *direct* contrast --- the *raw silhouette drops* of C2, C3, C4 from C0, and the *rank-resolved sign pattern* of $M(C_1) - M(C_0)$ across the seven Linnaean ranks --- rather than on $\rho_M$ alone. H_geom is supported if (a) the C0 - C3 drop is at least twice the C0 - C2 drop *and* (b) the sign of $M(C_1) - M(C_0)$ flips from positive at higher ranks to negative at species; H_info is supported if (a$'$) the C0 - C2 drop is comparable to the C0 - C3 drop. Both conditions are pre-registered and reported in Section 5.

## 3.2 Approach Overview

[Figure 1: Diagnostic framework. Top: dichotomy of hypotheses, with H_info predicting that content carries the benefit and H_geom predicting that structure does. Middle: five prompt conditions C0-C4 instantiating the orthogonal axes *content* x *structure*. Bottom: geometric audit (intra-class variance, inter-class margin, silhouette, alignment/uniformity, RankMe) and taxonomic audit (kNN purity, LCA depth, hierarchy consistency error) applied to image-only and image-text embeddings.]

The framework couples three analytical layers:

1. **Geometric audit (RQ1).** A flat-vs-hierarchical comparison on a frozen BioCLIP2 checkpoint, with intra-class variance, inter-class margin, silhouette at species level, alignment / uniformity, and RankMe.
2. **Rank-resolved latent-taxonomy probe (RQ2).** The same geometric audit replicated across seven Linnaean ranks and across OpenCLIP, BioCLIP, and BioCLIP2, with a random-taxonomy permutation control on OpenCLIP to test whether *unsupervised-with-respect-to-taxonomy* embeddings already encode partial hierarchical structure.
3. **Counterfactual prompt ablation (RQ3, central instrument).** Five prompt conditions C0-C4 designed to factor *structural alignment with the taxonomy* away from *lexical content*. We report effect-preservation ratios with bootstrap 95% confidence intervals.

A cross-dataset replication (RQ4, partial) extends 1--3 to a multi-phylum Rare Species evaluation set (400 species, 11{,}983 images) that removes the single-class confound of CUB-200, and the full five-domain meta-analysis (Aves, Insecta, Plantae, Fungi, Actinopterygii) is identified as the remaining external-validity step.

## 3.3 The Five Prompt Conditions (Counterfactual Ablation)

The five conditions are constructed to cross two axes: (a) *content* -- whether the taxonomic vocabulary in the prompt is meaningful, and (b) *structure* -- whether the prompt preserves the rank ordering aligned with the ground-truth taxonomy of the labelled species.

| Cond. | Content | Structure | Template (illustrative for *Passer domesticus*) |
|-------|---------|-----------|--------------------------------------------------|
| C0 | species only | n/a | `"a photo of Passer domesticus"` |
| C1 | real vocab | preserved | `"a photo of Animalia Chordata Aves Passeriformes Passeridae Passer domesticus"` |
| C2 | placeholder vocab | preserved | `"a photo of tax0 tax1 tax2 tax3 tax4 tax5 Passer domesticus"` |
| C3 | real vocab | destroyed | `"a photo of Plantae Arthropoda Reptilia Diptera Felidae Passer domesticus"` (upper ranks copied from random other species) |
| C4 | real vocab | shuffled order | random permutation of the seven C1 tokens per call |

C2 is the critical *structure-preserving content-empty* condition: it keeps the same number of tokens as C1, at the same positions, with the same species token, but replaces the upper six rank labels with vocabulary-external placeholders (`tax0`-`tax5`) so that no semantic cue beyond *position-in-hierarchy* remains. C3 is the *content-preserving structure-destroying* condition: upper ranks are sampled from *other* species so the words are real but their hierarchical alignment with the true label is broken (with the species token kept correct, so that the difference from C1 is *only* the alignment of upper ranks). C4 disentangles *order* from *bag-of-tokens* by randomly permuting C1's seven labels per call. Text-free hierarchical contrastive training \cite{kokilepersaud2024taxes} would form a complementary fourth axis (no text at all, hierarchical structure supervised on the image side); we deliberately keep this outside the present scope so that the H_info-vs-H_geom contrast is read solely from prompt-side manipulations, and discuss it as future work in Section 6.

This C2-versus-C3 contrast is, to our knowledge, the cleanest instrument so far proposed to separate *content* from *structure* in hierarchical prompting, and it is a direct refinement of the random-descriptor analysis of WaffleCLIP \cite{roth2023waffling} in a domain where ground-truth hierarchy is unambiguous.

### Algorithmic sketch

```
Input: frozen encoders f_v, f_t; eval set D; prompt-builder b_k for k in {0..4}
Output: per-condition metrics M_k = {intra_var, inter_margin, silhouette, ...}

for k in {0,1,2,3,4}:
    for each species c in D:
        t_c <- b_k(c, taxonomy[c])
        w_c <- L2_normalize( f_t(t_c) )
    for each image x_i in D:
        z_i <- L2_normalize( f_v(x_i) )
    compute geometric and taxonomic metrics on {z_i} (image-only)
    and on {z_i, w_c} (image-text) -> M_k

for k in {2,3,4}:
    rho_k <- (M_k - M_0) / (M_1 - M_0)      # with bootstrap CI

decision rule (primary, ratio-based):
    if  rho_2 >= 0.5 (CI lower >= 0.3) AND rho_3 <= 0.2 (CI upper <= 0.4):
            H_geom supported
    elif rho_2 < 0.3:                       H_info supported
    else:                                    inconclusive

decision rule (secondary, raw-drop + rank-sign, used when M(C1) - M(C0) < 0):
    drop_k <- M(C0) - M(C_k)                # k in {1,2,3,4}
    rank_signs <- sign( M_r(C1) - M_r(C0) ) for r in {phylum..genus, species}
    if  drop_3 >= 2 * drop_2                # structure-destruction > content-removal
        AND rank_signs are positive at higher ranks
        AND rank_signs are negative only at species:
            H_geom supported (directional)
    elif drop_2 ~= drop_3:                  # content-removal == structure-destruction
            H_info supported (directional)
    else:                                    inconclusive
```

## 3.4 Measurements

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

## 3.5 Design Choices

**Why frozen encoders.** The hypotheses concern the *embedding geometry induced by hierarchical prompts at evaluation time*, not a new training algorithm. Freezing the encoder and varying only the text-side input cleanly isolates prompt-driven effects; image-side fine-tuning would conflate prompt-mechanism evidence with adapter-learning capacity and is therefore explicitly excluded from the present protocol.

**Why placeholder tokens for C2.** Random tokens drawn from the vocabulary risk re-introducing semantics through subword pieces. We use vocabulary-external strings (`tax0`-`tax5`) that BPE-tokenize predictably and contain no semantic load, matching the *structural slot count* of C1 exactly.

**Why an image-only variant of every metric.** Modality-gap artefacts \cite{liang2022mindgap} can confound any image-text comparison. We always report an image-only counterpart so that conclusions about geometric reorganization do not depend on cross-modal alignment changes that hierarchical prompts could induce in the text branch only.

**Why effect-preservation ratios *and* raw differences.** Across metrics and datasets the absolute scales differ substantially. Normalizing by $\Delta_M(C_1) = M(C_1) - M(C_0)$ produces a quantity that is, in principle, comparable across metrics and datasets and has a transparent decision rule (Section 3.1). The same normalization, however, becomes uninterpretable when $\Delta_M(C_1)$ is small *or has the unexpected sign*. Our executed runs (Section 5) consistently exhibit $\Delta_M(C_1) < 0$ at species level on a frozen biology-pretrained encoder, which is itself the central empirical phenomenon we report. We therefore (i) report $\rho_M$ for completeness, (ii) declare it the *primary* statistic only when the denominator is positive and large enough to be informative, and (iii) fall back to the raw-drop / rank-sign criterion of Section 3.1 in the saturated-species regime that our data actually inhabit.

**Pre-registration.** The hypothesis dichotomy, the five prompt conditions, the geometric and taxonomic metrics, the primary ratio-based decision rule, and the secondary raw-drop + rank-sign decision rule of Section 3.1 were fixed *before* execution and are mirrored without modification in `02_rqs.md`. The Rare Species cross-dataset run was added to the protocol *after* CUB-200 results made the denominator-sign issue empirically clear, but it executes the same fixed measurement pipeline; we flag this as a *replication* step rather than a hypothesis-generating one and treat the multi-phylum result as confirmatory only for the sign pattern that CUB-200 already exhibited.
