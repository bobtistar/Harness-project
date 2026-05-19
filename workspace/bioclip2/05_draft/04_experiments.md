# 4. Experiments

This section describes datasets, models, evaluation metrics, and implementation. Section 5 reports the results. We emphasize at the outset that the *executed* run is a toy-scale sanity check on OpenCLIP ViT-B/32 with eight placeholder species; the *intended* run on BioCLIP2 + TreeOfLife is fully specified but not executed in the present submission (Section 4.4 and Section 6.2).

## 4.1 Datasets

**Primary (intended).** TreeOfLife evaluation subset, derived from `imageomics/TreeOfLife-10M` and `imageomics/TreeOfLife-200M` on HuggingFace. We retain species with at least 20 images, attach the seven-rank Linnaean taxonomy already present in the metadata, and stratify a 70 / 15 / 15 train / val / test split by species. Because our analysis is frozen except for the C5 adapter, only the test split is used in conditions C0-C4; C5 fine-tunes on the train split. The target evaluation set is at least 1,000 species and at least 20,000 test images.

**Cross-domain (RQ4, intended).** Aves (NABirds + iNat21 birds), Insecta (BIOSCAN-1M), Plantae (PlantNet300K + iNat21 plants), Fungi (iNat21 fungi), Actinopterygii (FishNet). All datasets are accessed under their published research licenses; iNaturalist content is used in accordance with individual image licenses.

**Toy placeholder (executed).** Eight synthetic species across four taxonomic domains, six class-conditioned synthetic RGB images per species (48 images total), with a synthetic seven-rank taxonomy attached. This was constructed as an end-to-end pipeline sanity check and is *not* a substitute for the intended evaluation; we report it explicitly as a toy run in Section 5.

Preprocessing for all datasets: resize 224x224, center crop, OpenCLIP normalization with mean (0.48145466, 0.4578275, 0.40821073) and std (0.26862954, 0.26130258, 0.27577711); text truncation to 77 tokens.

## 4.2 Baselines

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

## 4.3 Metrics

We collect geometric metrics (intra-class variance, inter-class margin, silhouette at rank $r$, alignment, uniformity, RankMe) and taxonomic metrics (kNN purity@10 at rank $r$, LCA depth, hierarchy consistency error, mutual information $I(\text{cluster}; \text{rank-label})$). Exact definitions are given in Section 3.4 and in the experimental protocol.

The **effect-preservation ratio** $\rho_M(C_k) = (M(C_k) - M(C_0)) / (M(C_1) - M(C_0))$ is the central derived quantity for RQ3. We report it for $k \in \{2, 3, 4, 5\}$ on each of the three RQ1 geometric metrics, with bootstrap (B = 1000) 95% confidence intervals.

Statistical tests:

- **Paired permutation test** (B = 1000) for two-condition comparisons on the same image set; paired by image identity.
- **Bootstrap 95% CI** (B = 1000) for all metric reports.
- **Bonferroni correction** for the three RQ1 metrics simultaneously: $\alpha = 0.01 / 3 \approx 0.0033$.
- **Effect size**: Cohen's $d$ for paired continuous metrics; Cliff's $\delta$ for a non-parametric robustness check.
- **Meta-analysis (RQ4)**: random-effects pooled effect, I-squared heterogeneity, between-domain CV.

## 4.4 Implementation Details

**Code.** A complete reference implementation accompanies the paper (`04_experiments/code/`): `prompt_variants.py` builds C0-C5 prompts; `extract_embeddings.py` produces image and text embeddings on either OpenCLIP, BioCLIP, or BioCLIP2; `metrics.py` implements all geometric and taxonomic metrics with bootstrap confidence intervals; `run_experiment.py` orchestrates a single (model, dataset, conditions) run; `run_all.sh` and `run_all.ps1` script the full sweep. A `--mock` flag substitutes randomized embeddings for sanity-checking the metric and statistics code without any model download.

**Software.** Python 3.12.7, PyTorch 2.7.1+cu118, `open_clip_torch` 3.2.0, `transformers` 5.1.0, scikit-learn 1.5.1, NumPy 1.26.4, SciPy 1.13.1. Determinism is enforced via `numpy.random.default_rng(seed)`, `torch.manual_seed(seed)`, and CuDNN deterministic flags.

**Seeds.** Five seeds {42, 1337, 2024, 7, 314} for C3 shuffle and for any data sub-sampling; B = 1000 for bootstrap and permutation.

**Hardware.** The intended TreeOfLife run is sized for a single NVIDIA A100 40 GB, with an estimated 8-12 hours per domain at batch size 128. The *executed* toy run was performed on a CPU-only Windows 11 machine.

**Adapter for C5.** LoRA with rank 8 applied to the OpenCLIP image encoder, plus a linear projection head; objective is image-image hierarchical InfoNCE with rank-weighted positives following \cite{kokilepersaud2024taxes}; AdamW with learning rate 1e-4, batch size 256, five epochs.

**Execution status.** Code skeleton (full); mock end-to-end run (full); OpenCLIP ViT-B/32 toy run on synthetic data (full); OpenCLIP ViT-L/14 on TreeOfLife (not executed); BioCLIP and BioCLIP2 (not executed); RQ4 cross-domain replications (not executed). The unmet runs are gated by HuggingFace weight downloads (about 6 GB) and TreeOfLife evaluation data (about 60 GB) on a GPU host, both of which are unavailable in the present environment. The published code is structured so that swapping `--model bioclip2 --csv data/treeoflife_eval.csv` reproduces the full intended evaluation without code changes.
