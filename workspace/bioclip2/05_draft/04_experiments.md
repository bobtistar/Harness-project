# 4. Experiments

This section describes the datasets, models, metrics, and implementation of the executed evaluation; Section 5 reports the results. The diagnostic protocol of Section 3 was run on a frozen biology-pretrained backbone (BioCLIP2 ViT-L/14) over two evaluation sets that differ in exactly the property that drives our interpretation --- *taxonomic breadth* --- and was replicated on a general-domain backbone (OpenCLIP ViT-L/14) to test backbone-dependence. All runs use real images and real GBIF-resolved taxonomies; no synthetic or mock data enter the reported numbers.

## 4.1 Datasets

We evaluate on two complementary frozen-evaluation sets, chosen so that the *rank-homogeneity confound* of Section 3.3 is present in one and structurally removed in the other.

**CUB-200-2011 (single-class).** The Caltech-UCSD Birds dataset \cite{wah2011cub}: 200 species, 11,788 images (mean $\approx 59$ images/species). We attach the seven-rank Linnaean lineage to each species via the GBIF backbone API (cached once; CC0). Because every species is a bird, the kingdom, phylum, and class tokens (`Animalia`, `Chordata`, `Aves`) are *identical across all prompts* — the rank-homogeneity confound in its strongest form — so only the order, family, genus, and species ranks carry inter-species discriminative signal. CUB-200 is therefore our *saturated, single-class* probe.

**iNaturalist-2021 multi-clade subset (multi-kingdom).** To remove the single-class confound we construct a stratified subset of the iNaturalist-2021 validation split spanning **five classes across three kingdoms** — Aves, Insecta, and Reptilia (Animalia), Plantae, and Fungi — sampling up to 15 images per species. The realized set is **821 species and 8,210 images** ($\approx 10$ images/species). Here the upper-rank tokens are *species-specific and informationally distinct*, so all seven Linnaean ranks (including kingdom) are non-degenerate and testable. This is our *multi-clade* probe; it directly tests whether the CUB-200 findings are an artefact of taxonomic homogeneity.

Preprocessing is identical for both sets: resize to $224\times224$, center crop, OpenCLIP normalization (mean $(0.4815, 0.4578, 0.4082)$, std $(0.2686, 0.2613, 0.2758)$), text truncated to 77 tokens. Both sets are used in *frozen evaluation* (no train/val split; the entire set is encoded once and reused across all conditions and seeds).

**Cross-domain (RQ4, future work).** A full five-domain meta-analysis — adding Insecta-only (BIOSCAN-1M), Plantae-only (PlantNet300K), Fungi-only (iNat21 fungi), and Actinopterygii (FishNet) as separate domains — is specified for RQ4 but not executed here; the iNat21 multi-clade run covers three kingdoms in aggregate but does not yet support a per-domain random-effects meta-analysis (Section 5.6).

## 4.2 Models

The analysis target and the cross-backbone control are both ViT-L/14, $d=768$, evaluated frozen:

- **BioCLIP2 ViT-L/14** \cite{gu2025bioclip2} (`hf:imageomics/bioclip-2`) — biology-pretrained on TreeOfLife-200M with per-rank contrastive supervision; **primary analysis target**.
- **OpenCLIP ViT-L/14** \cite{ilharco2021openclip} (`laion2b_s32b_b82k`) — general-domain pretraining on LAION-2B, *matched backbone and embedding dimension*, evaluated on the iNat21 subset to isolate the contribution of biological pretraining from that of backbone capacity.

We report the five prompt conditions C0–C4 (Section 3.3) on each (model, dataset) pair. BioCLIP ViT-B/16 \cite{stevens2024bioclip} was specified as a prior-generation comparison but is not executed in this submission; we note its absence in Section 5.6. External-method references (WaffleCLIP \cite{roth2023waffling}, CHiLS \cite{novack2023chils}, HAPrompts \cite{liang2024haprompts}, ProTeCt \cite{wu2024protect}, hierarchical cross-entropy \cite{bertinetto2020mistakes}) inform the *design* and *interpretation* of the conditions and metrics rather than serving as runnable baselines on our frozen-evaluation protocol.

## 4.3 Metrics

Per condition we compute the image-only geometric metrics (intra-class variance, inter-class margin, species-level silhouette, RankMe, uniformity), the rank-resolved silhouette and kNN purity@10 at every non-degenerate Linnaean rank, the latent-taxonomy permutation probe, and — on the iNat21 runs — the cross-modal **zero-shot top-1 accuracy** of the per-condition text prototypes. Exact definitions are in Section 3.4.

The **effect-preservation ratio** $\rho_M(C_k) = (M(C_k)-M(C_0))/(M(C_1)-M(C_0))$ is the pre-registered primary statistic for RQ3 ($k\in\{2,3,4\}$, three RQ1 metrics, bootstrap $B=1000$). As anticipated in Section 3.1 and confirmed in Section 5.1, $\Delta_M(C_1)<0$ at species level in every executed run, so we report $\rho_M$ for completeness but base the RQ3 decision on the *raw drop* $M(C_0)-M(C_k)$ and the *rank-sign pattern* of $M(C_1)-M(C_0)$.

Statistical tests:

- **Paired permutation test** ($B=1000$), paired by image identity, for the C0-vs-C1 comparison on each metric.
- **Bootstrap 95% CI** ($B=1000$) for all reported metrics and for $\rho_M$.
- **Bonferroni correction** over the six Exp1 metrics simultaneously: $\alpha = 0.01/6 \approx 0.00167$.
- **Latent-taxonomy probe**: $B=50$ random species-label permutations $\rightarrow$ $z$-score of the true-label silhouette against the permutation null.
- **Effect size**: Cohen's $d$ is computed but, as Section 5.5 documents, is *not interpretable* here because the seed-to-seed standard deviation is $\sim 10^{-7}$; we rely on raw differences and cross-run replication.

## 4.4 Implementation Details

**Stochasticity model.** Because the encoders are frozen and deterministic, run-to-run variation is induced by adding i.i.d. Gaussian noise ($\sigma=10^{-3}$) to the cached image embeddings before each seed's metric computation, simulating stochastic-sampling jitter. This $\sigma$ is deliberately small to preserve the geometry; its consequence — vanishing seed variance and inflated Cohen's $d$ — is reported transparently as a limitation (Sections 5.5, 5.6) and motivates the raw-difference-based reporting.

**Code.** The reference implementation is in `04_experiments/code/`: `prompt_variants.py` builds C0–C4; `extract_embeddings.py` encodes images/text on OpenCLIP, BioCLIP, or BioCLIP2; `metrics.py` implements all geometric/taxonomic metrics with bootstrap CIs; `run_experiment.py` orchestrates a single (model, dataset) run; `cub200_build_taxonomy.py` and `inat21_build_metadata.py` build the GBIF-resolved taxonomies. A `--mock` flag substitutes randomized embeddings for code-path validation only (never reported as a result).

**Software / hardware.** Python 3.12, PyTorch 2.4.1+cu124, `open_clip_torch` 3.2.0, NumPy 1.26.4, scikit-learn 1.8.0; single NVIDIA CUDA GPU, AMP (float16). Determinism via `numpy.random.default_rng(seed)`, `torch.manual_seed(seed)`, and CuDNN deterministic flags. Image embeddings are encoded once and cached (`*.npz`) for reuse across all conditions and seeds. **Five seeds** per run.

**Executed runs.** (i) CUB-200 $\times$ BioCLIP2 (200 species, 11,788 images; batch 128); (ii) iNat21 multi-clade $\times$ BioCLIP2 (821 species, 8,210 images; batch 64); (iii) iNat21 multi-clade $\times$ OpenCLIP ViT-L/14 (same subset; batch 64). Each run executes Exp1 (RQ1), Exp2 (RQ2 rank-resolved + latent probe), and Exp3 (RQ3 counterfactual C0–C4); the iNat21 runs additionally compute zero-shot top-1 accuracy and a t-SNE visualization of the cached image embeddings. Git hash `9c1aa34`.
