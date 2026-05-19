# 5. Results

> **Caveat.** All numerical results in this section are *toy- and mock-scale sanity-check signals* obtained from OpenCLIP ViT-B/32 on a synthetic taxonomy of eight placeholder species (six images per species; 48 total images), executed on CPU. The downloads required to run BioCLIP / BioCLIP2 on the full TreeOfLife evaluation set were not available in the present environment. The numbers below validate the pipeline and report *directional* trends; they should *not* be cited as evidence for or against the main hypotheses, and a full-scale validation on BioCLIP2 + TreeOfLife is registered as immediate future work.

## 5.1 Main Results: Flat versus Hierarchical Geometric Compactness (RQ1)

[Table 1: Geometric metrics on a frozen backbone for C0 (flat species-only) and C1 (normal seven-rank hierarchical), executed on OpenCLIP ViT-B/32 with a toy taxonomy. Bootstrap CIs are based on B = 1000 resamples over three seeds.]

| Metric | C0 (flat) | C1 (hierarchical) | $\Delta$ (C1-C0) | bootstrap 95% CI | passes pre-registered threshold? |
|--------|-----------|-------------------|-------------------|--------------------|-----------------------------------|
| intra_var (lower) | 0.00239 | 0.00239 | 0 (tied) | tight | NO |
| inter_margin (higher) | 713.11 | 389.59 | -323.5 | [-324.6, -322.8] | NO (reversed) |
| silhouette (higher) | 0.9865 | 0.9740 | -0.0125 | [-0.013, -0.012] | NO (reversed) |
| RankMe | 9.79 | 9.23 | -0.57 | tight | -- |
| uniformity (lower) | -0.834 | -0.684 | +0.15 | tight | -- |

On toy data, C1 does *not* improve over C0 on any RQ1 metric. The most plausible explanation is that the synthetic images were generated from class-conditioned color means; the class signal in the image embedding is therefore so strong that silhouette is saturated near 1, and any additional text-side variation (the extra tokens of C1) couples to image embeddings only through the contrastive scoring head and acts as noise rather than as structure. We therefore *cannot* take this result as evidence against the hierarchical-compactness claim; the experiment is under-powered by construction. A full BioCLIP2 + TreeOfLife run is required to obtain meaningful signal on RQ1.

## 5.2 Ablations: Counterfactual Conditions C0-C5 (RQ3 -- central instrument)

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

## 5.3 Latent Taxonomic Structure in OpenCLIP (RQ2)

[Table 3: Species-level silhouette for OpenCLIP ViT-B/32 embeddings under true taxonomy versus random-taxonomy permutation (B = 50).]

| Source | silhouette (cosine) | z-score vs. random-taxonomy null |
|--------|----------------------|-----------------------------------|
| true labels | 0.683 | +44.9 |
| random-taxonomy permutation mean | -0.271 | (baseline) |

Even on toy data, the *unsupervised-with-respect-to-taxonomy* OpenCLIP separates true species labels from random-taxonomy controls with a very large z-score. This is consistent with the latent-structure premise of RQ2 H1(b), but the synthetic class-conditioned color signal almost certainly inflates the magnitude. Per-rank decomposition (species $\to$ kingdom) cannot be reported on the toy taxonomy because at higher ranks the number of classes is too small for silhouette to be defined; the rank-wise analysis is part of the registered intended run.

## 5.4 Statistical Integrity Notes

- Paired permutation p-values for the C0-vs-C1 comparison on the toy run are in the range 0.24-0.26, well above the Bonferroni-corrected $\alpha = 0.0033$. No metric is significant at toy scale, and we do not claim significance.
- Cohen's $d$ for silhouette in the toy run is inflated (about -2910) because the seed-to-seed standard deviation collapses to about $10^{-6}$; this is a small-sample, low-noise artefact of synthetic data and would not occur on real images.

## 5.5 Per-RQ Answers

We restate the four research questions and answer each with one of {supported, refuted, undetermined} given the evidence available.

| RQ | Toy-scale directional signal | Verdict on the registered hypothesis |
|----|------------------------------|---------------------------------------|
| **RQ1** (hierarchical prompts increase geometric compactness on a frozen biological VLM) | C1 does *not* improve over C0 on toy data; metrics are saturated by synthetic class-conditioned color. | **Undetermined** (and *not* refuted). The experiment is under-powered by construction; a BioCLIP2 + TreeOfLife replication is required. |
| **RQ2** (latent taxonomic structure exists in OpenCLIP, and the BioCLIP2 effect is concentrated at higher ranks) | OpenCLIP separates true species from random-taxonomy at z = +44.9 on toy data; per-rank decomposition not possible at toy scale. | **Directional support for H1(b)** (latent structure plausibly present); H1(a) (rank-wise decomposition) **undetermined**. |
| **RQ3** (semantic-organizer hypothesis: structure preserved $\Rightarrow$ benefit preserved; structure destroyed $\Rightarrow$ benefit collapses) | C3 silhouette collapses 0.97 $\to$ 0.10; C2 silhouette tracks C1 closely. Direction matches H_geom for both critical conditions. Preservation-ratio magnitudes are ill-conditioned at toy scale. | **Directional support; magnitudes undetermined.** Strongest qualitative signal of the toy run, but not yet a quantitative confirmation. |
| **RQ4** (cross-domain consistency over five biological domains) | Not executed (domain data not downloaded). | **Not yet evaluated.** |

We emphasize again that *supported* and *directional support* are weaker statements than the pre-registered success thresholds in Section 3.1 and Section 02_rqs.md call for. The main claims of the paper -- compactness improvement, latent taxonomic structure, and semantic-organizer mechanism -- remain *hypotheses* at the end of this preliminary report. The reported toy signals are *consistent with* these hypotheses but do not constitute evidence sufficient to adopt them.
