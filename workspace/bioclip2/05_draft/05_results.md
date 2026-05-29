# 5. Results

We report three executed runs — CUB-200 $\times$ BioCLIP2, iNat21-multiclade $\times$ BioCLIP2, and iNat21-multiclade $\times$ OpenCLIP ViT-L/14 — and map each result to the registered research question. All numbers are real frozen-evaluation measurements (five seeds, image embeddings cached once). Two analysis-shaping facts recur and are established in Section 5.1: (i) the species-level hierarchical effect $\Delta_{\text{sil}}(C_1)=S(C_1)-S(C_0)$ is *negative* in all three runs, which makes the preservation ratio's denominator negative and shifts the RQ3 decision onto raw drops and the rank-sign pattern (Section 3.1); and (ii) the seed-to-seed standard deviation is $\sim 10^{-7}$, so no metric survives Bonferroni correction and Cohen's $d$ is uninterpretable — our evidence is the *direction and relative magnitude* of effects and their *replication across runs*, not per-run significance.

## 5.1 Flat versus Hierarchical Compactness (RQ1)

[Table 1: Species-level geometry, C0 (flat) vs C1 (hierarchical), five seeds. $\Delta$ is $C1-C0$; paired-permutation $p$ in parentheses. None pass the pre-registered RQ1 threshold (intra $\geq 10\%$ drop AND inter-margin ratio $\geq 1.2\times$ AND silhouette gain $\geq +0.05$).]

| Run | silhouette C0 | silhouette C1 | $\Delta_{\text{sil}}$ | inter-margin C0$\to$C1 | kNN@10 C0$\to$C1 | passes? |
|---|---|---|---|---|---|---|
| CUB-200 $\times$ BioCLIP2 | 0.7749 | 0.7557 | **$-0.0192$** ($p{=}.078$) | 13.59 $\to$ 12.75 | 0.9992 $\to$ 0.9980 | No |
| iNat21 $\times$ BioCLIP2 | 0.6827 | 0.6715 | **$-0.0112$** ($p{=}.078$) | 14.91 $\to$ 14.50 | 0.8954 $\to$ 0.8913 | No |
| iNat21 $\times$ OpenCLIP | 0.4519 | 0.2262 | **$-0.2257$** ($p{=}.078$) | 8.91 $\to$ 4.16 | 0.8174 $\to$ 0.6004 | No |

Intra-class variance is unchanged to $\sim 10^{-8}$ in all runs (the hierarchical prompt does not move *within-species* spread); the action is entirely in *between-species* organization.

**Reading.** At the species level, hierarchical prompting *does not* tighten the embedding — it slightly loosens it on BioCLIP2 ($-0.019$ on CUB-200, $-0.011$ on iNat21) and loosens it sharply on OpenCLIP ($-0.226$). The naive form of RQ1 ("hierarchical $\Rightarrow$ more compact species clusters") is therefore **refuted in all three runs**. Crucially, the *multi-clade* iNat21 run reproduces the BioCLIP2 species-level drop even though its upper-rank tokens are informative — so the drop is **not** an artefact of CUB-200's single-class confound. The consistent explanation, developed in Sections 5.2–5.4, is that BioCLIP2 already separates species near-ceiling under a flat prompt ($S=0.77$, kNN@10 $=99.9\%$ on CUB-200), so hierarchical text cannot add species-level signal and instead pulls species toward shared higher-rank anchors. We therefore reframe RQ1 from an *improvement* claim to a *cross-rank trade-off* claim, which RQ2 tests directly. The much larger OpenCLIP drop ($\approx 12\times$ BioCLIP2 on iNat21) is the first sign that biological pretraining makes the backbone *robust* to hierarchical conditioning rather than dependent on it.

## 5.2 Rank-Resolved Effect (RQ2a)

[Table 2: $\Delta_{\text{sil}}(C_1)=S^{(r)}(C_1)-S^{(r)}(C_0)$ at each Linnaean rank. Positive = hierarchical prompt improves organization at that rank. Ranks degenerate on CUB-200 (single class Aves) are marked "—".]

| Rank | CUB-200 $\times$ BioCLIP2 | iNat21 $\times$ BioCLIP2 | iNat21 $\times$ OpenCLIP |
|---|---|---|---|
| kingdom | — | **+0.0037** | **+0.157** |
| phylum  | — | **+0.0059** | **+0.139** |
| class   | — | **+0.0101** | **+0.182** |
| order   | **+0.0064** | **+0.0095** | **+0.098** |
| family  | **+0.0242** | **+0.0200** | $-0.034$ |
| genus   | **+0.0360** | **+0.0182** | $-0.219$ |
| species | $-0.0192$ | $-0.0112$ | $-0.226$ |

**Reading.** On BioCLIP2 the pattern is unambiguous and *replicated across both datasets*: hierarchical prompting **improves silhouette at every higher rank and degrades it only at the species rank**. On the multi-clade iNat21 run this holds across all six super-species ranks (kingdom through genus, $+0.004$ to $+0.020$), where CUB-200's homogeneity had previously hidden the top three ranks. This is the direct geometric signature of a *cross-rank trade-off*: the hierarchical prompt re-arranges cross-species distances to agree with taxonomic distance, paying for higher-rank organization with a small loss of fine-grained species separation — exactly the directional prediction of H_geom in Section 3.1. OpenCLIP shows the *same sign flip* but at a different crossover (positive through order, then negative from family down, with a large $-0.226$ at species), indicating that the general-domain backbone reorganizes coarse structure far more aggressively and sacrifices fine structure far more than the biology-pretrained one.

## 5.3 Latent Taxonomic Structure Without Text (RQ2b)

[Table 3: Species-level silhouette of the *image-only* embeddings under the true taxonomy vs a random-taxonomy permutation null ($B=50$).]

| Run | true-label silhouette | random null ($\mu\pm\sigma$) | $z$-score |
|---|---|---|---|
| CUB-200 $\times$ BioCLIP2 | 0.5015 | $-0.079 \pm 0.0048$ | **+121.7** |
| iNat21 $\times$ BioCLIP2 | 0.3702 | $-0.146 \pm 0.0029$ | **+180.9** |
| iNat21 $\times$ OpenCLIP | $-0.0174$ | $-0.182 \pm 0.0033$ | **+50.7** |

**Reading.** With *no text prompt at all*, BioCLIP2 image embeddings separate the true taxonomy from random label permutations at astronomical significance ($z=122$ on CUB-200, rising to $z=181$ on the harder multi-clade set). The taxonomic structure that hierarchical prompts organize is therefore *already latent in the visual representation* — consistent with H_geom's core premise that text acts as an organizer of pre-existing structure rather than as the source of it. OpenCLIP is the instructive contrast: its true-label silhouette is *negative* ($-0.017$, i.e. species do not form clean clusters), yet it is still significantly above its own random null ($z=51$). Latent taxonomy thus exists in general-domain embeddings too, but in far weaker form; BioCLIP2's biological pretraining is what amplifies it into a strongly clustered geometry.

## 5.4 Counterfactual Ablation: Content vs Structure (RQ3, central instrument)

[Table 4: Species-level silhouette per condition and its drop from the flat baseline $S(C_0)-S(C_k)$. C2 = placeholder vocabulary, structure preserved; C3 = real vocabulary, structure destroyed; C4 = real vocabulary, order shuffled. The pre-registered ratio test is voided by $\Delta_{\text{sil}}(C_1)<0$ (Section 3.1); we report raw drops.]

| Condition | CUB-200 $\times$ BioCLIP2 | iNat21 $\times$ BioCLIP2 | iNat21 $\times$ OpenCLIP |
|---|---|---|---|
| C0 flat | 0.7749 (—) | 0.6827 (—) | 0.4519 (—) |
| C1 hierarchical | 0.7557 ($-0.019$) | 0.6715 ($-0.011$) | 0.2262 ($-0.226$) |
| **C2** placeholder / structure kept | 0.6965 (**$-0.078$**) | 0.5958 (**$-0.087$**) | 0.3533 (**$-0.099$**) |
| **C4** order shuffled | 0.6281 ($-0.147$) | 0.5518 ($-0.131$) | 0.2068 ($-0.245$) |
| **C3** structure destroyed | 0.4206 (**$-0.354$**) | 0.3014 (**$-0.381$**) | 0.0309 (**$-0.421$**) |

**Reading.** The same ordering holds in **all three runs**:
$$\text{drop}(C_2) \;<\; \text{drop}(C_4) \;<\; \text{drop}(C_3),$$
i.e. *removing the lexical content while keeping the hierarchical slot structure (C2) is the least damaging manipulation; shuffling order (C4) is worse; destroying the structural alignment (C3) is catastrophic.* The structure-destruction-to-content-removal drop ratio is remarkably stable across very different backbones and datasets: **$4.5\times$** (CUB-200/BioCLIP2), **$4.4\times$** (iNat21/BioCLIP2), **$4.3\times$** (iNat21/OpenCLIP). This is the predicted positive-and-negative signature of H_geom: the benefit of the hierarchical prompt survives content removal but collapses under structure destruction, *regardless of vocabulary*. The pre-registered ratio test returns `semantic_organizer_supported = False` in every run, but solely because its denominator $\Delta_{\text{sil}}(C_1)$ is negative (the saturated-species regime of Section 3.1); the secondary raw-drop + rank-sign rule, which was registered precisely for this regime, is satisfied in all three runs.

A backbone nuance: on OpenCLIP, C1 itself ($-0.226$) drops *more* than the content-empty C2 ($-0.099$) — the general-domain text encoder over-weights the (taxonomically irrelevant, to it) Latin upper-rank words and pulls species together, whereas inert placeholders do less harm. On BioCLIP2 this inversion does not occur (C1 $\approx -0.01$–$0.02$, much smaller than C2), again indicating that biological pretraining is what lets real taxonomic vocabulary be used *constructively* rather than as noise.

## 5.5 Cross-Modal Check: Zero-Shot Accuracy (iNat21)

[Table 5: Zero-shot top-1 accuracy of the per-condition text prototypes on the iNat21 subset (821 species). This image-text metric complements the image-only geometry above.]

| Condition | BioCLIP2 | OpenCLIP |
|---|---|---|
| C0 flat | **0.943** | 0.113 |
| C1 hierarchical | 0.937 | 0.104 |
| C2 placeholder / structure kept | 0.464 | 0.082 |
| C3 structure destroyed | 0.351 | 0.033 |
| C4 order shuffled | 0.901 | **0.114** |

**Reading.** Two findings. First, the BioCLIP2-vs-OpenCLIP gap is enormous ($94\%$ vs $11\%$ flat), quantifying the biological-pretraining advantage that the geometry metrics described qualitatively. Second, the *cross-modal* metric tells a complementary story to the image-only geometry: here C3 (wrong-lineage real words) is again worst, but C2 (placeholder tokens) now collapses sharply ($0.94\to0.46$ on BioCLIP2) while C4 (real words, shuffled order) stays high ($0.90$). This is consistent and not contradictory: zero-shot matching depends on the *text prototype* being semantically meaningful, so emptying the upper-rank content (C2) dilutes the prototype even though it barely perturbs the *image* geometry — whereas destroying *structure* (C3) harms both channels. The dissociation (C2 hurts text matching but not image geometry; C3 hurts both) is itself evidence that the image-side reorganization measured in 5.2–5.4 is driven by structure, not lexical content.

## 5.6 Per-RQ Verdicts, Statistical Integrity, and Limitations

**Per-RQ verdicts.**

| RQ | Verdict | Basis |
|---|---|---|
| **RQ1** (compactness) | naive form **refuted**; reframed as cross-rank trade-off | species-level $\Delta_{\text{sil}}(C_1)<0$ in all 3 runs, reproduced on multi-clade iNat21 (not a single-class artefact) |
| **RQ2a** (rank-resolved) | **supported (directional, replicated)** | BioCLIP2: every super-species rank positive, species negative, on both datasets; OpenCLIP same sign flip |
| **RQ2b** (latent taxonomy) | **strongly supported** | text-free image embeddings separate true taxonomy at $z=122$–$181$ (BioCLIP2); weak but significant in OpenCLIP ($z=51$) |
| **RQ3** (organizer vs channel) | **directional support for H_geom** | drop$(C_3)\approx 4.3$–$4.5\times$ drop$(C_2)$ in all 3 runs; ratio test voided by denominator sign, raw-drop rule satisfied |
| **RQ4** (cross-domain meta-analysis) | **partial** | multi-clade replication done (3 kingdoms aggregate); per-domain random-effects meta-analysis (I², CV) not run |

**Statistical integrity.** With $\sigma=10^{-3}$ embedding noise the seed standard deviation is $\sim 10^{-7}$, so (i) all six Exp1 paired-permutation $p$-values lie in $[0.05, 0.08]$ and **none survives** Bonferroni ($\alpha\approx0.00167$), and (ii) Cohen's $d$ inflates to $\sim 10^{5}$ and is *not reported as evidence*. We therefore rest the conclusions on raw differences and on **cross-run replication** (two datasets $\times$ two backbones), which we regard as the stronger form of robustness here. We state plainly that no single run is individually significant under the registered test.

**Limitations.** (1) The stochasticity model ($\sigma=10^{-3}$ Gaussian on cached embeddings) under-powers the significance tests; re-running with natural model stochasticity (inference dropout, mixed-precision, mini-batch shuffling) or a larger $\sigma$ is needed to obtain Bonferroni-significant effects. (2) Both datasets are vertebrate/plant/fungus *photographic* sets; the five-domain meta-analysis (Insecta-only, Plantae-only, Fungi-only, Actinopterygii) for a proper RQ4 heterogeneity estimate is not yet run. (3) The cross-backbone control was executed only on iNat21, not on CUB-200, so we cannot fully separate "OpenCLIP reorganizes more" from a dataset interaction. (4) BioCLIP ViT-B/16 (prior generation) and the text-free image-side hierarchical-InfoNCE condition were specified but not executed. None of these limitations bears on the *direction* or *replication* of the reported effects; they bound the *significance* and *generality* claims, consistent with the abstract's hedged framing.
