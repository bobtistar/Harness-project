# 3. Preliminaries

This section establishes the conceptual and mathematical foundations for the diagnostic analysis that follows. Section 3.1 reviews contrastive vision-language learning. Section 3.2 describes the BioCLIP family of models and introduces the species-level saturation phenomenon that shapes our experimental observations. Section 3.3 formalizes the Linnaean taxonomic hierarchy and defines the rank homogeneity confound central to result interpretation. Section 3.4 defines the embedding geometry metrics used in Sections 4 and 5.

---

## 3.1 Contrastive Vision-Language Learning

**CLIP-style training.** Given a dataset of image-text pairs $\{(x_i, t_i)\}_{i=1}^N$, contrastive vision-language models learn an image encoder $f_v : \mathcal{X} \to \mathbb{R}^d$ and a text encoder $f_t : \mathcal{T} \to \mathbb{R}^d$ that map images and texts to a *shared* $d$-dimensional embedding space. Embeddings are L2-normalized so that cosine similarity equals the inner product:

$$\text{sim}(x, t) = \frac{f_v(x)^\top f_t(t)}{\|f_v(x)\| \|f_t(t)\|}.$$

Training minimizes a symmetric InfoNCE (cross-entropy) loss over mini-batches of size $B$:

$$\mathcal{L}_{\text{CLIP}} = -\frac{1}{2B} \sum_{i=1}^{B} \left[ \log \frac{e^{\text{sim}(x_i, t_i)/\tau}}{\sum_{j=1}^{B} e^{\text{sim}(x_i, t_j)/\tau}} + \log \frac{e^{\text{sim}(x_i, t_i)/\tau}}{\sum_{j=1}^{B} e^{\text{sim}(x_j, t_i)/\tau}} \right],$$

where $\tau > 0$ is a learned temperature. This objective simultaneously pushes each image closer to its paired text and further from all other texts in the batch, and vice versa. At convergence, semantically related images and texts cluster together in the shared space.

[Figure 1: CLIP contrastive learning objective. Left: image-text pairs drawn from a mini-batch. Right: the symmetric InfoNCE loss attracts paired (image, text) embeddings and repels unpaired ones in the shared hypersphere.]

**Zero-shot classification.** At inference, a set of $K$ class-descriptive text prompts $\{t_c\}_{c=1}^K$ are encoded once to form class prototypes $\{w_c = f_t(t_c)\}$. A test image $x$ is assigned to the class whose prototype achieves the highest cosine similarity:

$$\hat{c} = \argmax_{c \in [K]} \; \text{sim}(x, t_c).$$

The choice of $t_c$ — commonly called the *prompt* — critically governs the geometry of the class prototypes and therefore the quality of the nearest-neighbor boundaries in embedding space. This sensitivity to prompt design is the central motivation of the present work.

**Alignment and Uniformity.** \citet{wang2020understanding} decompose representation quality into two complementary scalar properties of the embedding hypersphere — alignment (cohesion of positive pairs) and uniformity (spread of the overall distribution) — which we report as auxiliary global-geometry summaries alongside the cluster-level metrics defined in Section 3.4.

---

## 3.2 BioCLIP and BioCLIP2

**BioCLIP** \citep{stevens2024bioclip} introduced hierarchical taxonomic prompts to biological vision-language learning, prepending the full Linnaean lineage (e.g., \texttt{"Animalia Chordata Aves … Passer domesticus"}) to achieve 17–20 percentage-point gains over general-domain OpenCLIP on fine-grained biological benchmarks. Qualitative analysis suggests the embedding space "conforms to the tree of life," though no formal geometric characterization is given.

**BioCLIP2** \citep{gu2025bioclip2} scales this paradigm to TreeOfLife-200M ($\approx$214 million images, $>$500,000 taxa) with a ViT-L/14 image encoder and an autoregressive text encoder, applying a separate contrastive loss at each Linnaean rank. This yields an additional $+18.1$ percentage-point gain over BioCLIP and gives rise to what the authors term **emergent properties**: inter-species embeddings align with ecological traits (beak size, habitat), while intra-species variation is preserved in orthogonal subspaces. These properties are documented qualitatively via t-SNE visualizations; the present work provides the first controlled quantitative characterization of the underlying geometric mechanisms.

**Species-level saturation.** A critical consequence of training on TreeOfLife-200M is that BioCLIP2 achieves near-ceiling species discrimination *before* any text prompt is applied. On the CUB-200-2011 benchmark (200 bird species, 11,788 images), the flat-prompt baseline (species name only) already yields a species-level silhouette of $0.775$ and a kNN purity of $99.9\%$. This *saturation regime* means that hierarchical text, which provides cross-rank structural signal, cannot meaningfully improve an already near-perfect species boundary — and may, in fact, dilute it by pulling species embeddings toward shared higher-rank anchors. Understanding this saturation is essential for correctly interpreting the geometry experiments in Section 5.

[Figure 2: BioCLIP2 architecture overview. (a) Image encoder: ViT-L/14 with frozen weights at evaluation time. (b) Text encoder: autoregressive language model consuming the full seven-rank taxonomic sequence. (c) Per-rank contrastive loss: a separate InfoNCE loss is applied at each Linnaean rank, with the image embedding pulled toward rank-specific text prototypes. (d) Hierarchical prompt construction: given a species label, the prompt is built by concatenating the lineage from kingdom down to species. (e) Saturation illustration: species-level kNN purity under flat vs. hierarchical prompts on CUB-200 and the multi-clade iNat21 subset.]

---

## 3.3 Linnaean Taxonomic Hierarchy

**Rank structure.** We adopt the standard seven-rank Linnaean classification system $\mathcal{R} = (r_1, \dots, r_7)$, ordered from coarsest to finest:

$$r_1 = \text{kingdom} \succ r_2 = \text{phylum} \succ r_3 = \text{class} \succ r_4 = \text{order} \succ r_5 = \text{family} \succ r_6 = \text{genus} \succ r_7 = \text{species}.$$

Each taxon $c$ has a unique lineage $\ell(c) = (c^{(1)}, c^{(2)}, \dots, c^{(7)}) \in \mathcal{C}^{(1)} \times \cdots \times \mathcal{C}^{(7)}$, where $\mathcal{C}^{(r)}$ is the set of taxa at rank $r$ and higher-rank taxa are shared across species within the same clade.

**Taxonomic proximity.** The *lowest common ancestor* (LCA) of two species is the deepest rank at which their lineages agree; its depth in $\{1, \dots, 7\}$ serves as the ground-truth measure of how closely related they are. This notion underlies the Exp2 latent-taxonomy probe (Section 5.4), where we test whether image embeddings reflect taxonomic proximity without any text supervision.

**Rank homogeneity confound.** A dataset-level property that directly affects the interpretation of our experiments is *rank homogeneity*: when all species in an evaluation set belong to the same taxon at some rank $r^*$, the corresponding rank-$r^*$ token is *identical* across every hierarchical prompt in the dataset. Formally, if $c^{(k)} = c'^{(k)}$ for all species pairs $(c, c')$ at some rank $r_k$, then the token at position $k$ in the hierarchical prompt carries zero inter-species discriminative information — it is constant across all prompts and thus acts as an uninformative anchor that may introduce noise into the shared embedding. The CUB-200-2011 dataset exemplifies this confound: all 200 species are birds (class Aves), so the tokens `"Animalia"`, `"Chordata"`, and `"Aves"` are shared by every prompt, effectively reducing the 7-rank hierarchical sequence to a 4-rank one (order → family → genus → species) for the purpose of inter-species discrimination. We address this confound explicitly in Section 5 by replicating all experiments on a multi-clade iNaturalist-2021 subset (821 species spanning five classes across three kingdoms — Aves, Insecta, Reptilia, Plantae, Fungi), where upper-rank tokens are species-specific and thus informationally distinct, making all seven Linnaean ranks non-degenerate.

---

## 3.4 Embedding Geometry Metrics

The following metrics characterize the geometric structure of the image embedding set $\{z_i\} \subset S^{d-1}$ (unit hypersphere) with respect to taxonomic labels. Unless stated otherwise, $Z_c = \{z_i : y_i^{(\text{species})} = c\}$ is the set of image embeddings for species $c$, $\mu_c = |Z_c|^{-1}\sum_{z \in Z_c} z$ is the (unnormalized) centroid, and $|\cdot|$ denotes set cardinality.

**Intra-class variance** $(\downarrow$ better$)$. For each species $c$, the *within-class spread* is the mean pairwise cosine distance among its embeddings:

$$\sigma_c^2 = \frac{1}{\binom{|Z_c|}{2}} \sum_{\substack{z, z' \in Z_c \\ z \neq z'}} \bigl(1 - z^\top z'\bigr).$$

We report the macro-average $\overline{\sigma^2} = |\mathcal{C}|^{-1}\sum_c \sigma_c^2$ over all species. A lower value indicates tighter, more discriminative clusters.

**Inter-class margin** $(\uparrow$ better$)$. For each species $c$, the *separation* from the nearest other species is:

$$\delta_c = \frac{\min_{c' \neq c} \bigl(1 - \hat{\mu}_c^\top \hat{\mu}_{c'}\bigr)}{\sigma_c},$$

where $\hat{\mu}_c = \mu_c / \|\mu_c\|$ is the normalized centroid. We report the macro-average $\overline{\delta}$. A higher value indicates that species centroids are well-separated relative to within-class spread.

**Silhouette score at rank $r$** $(\uparrow$ better$)$. Let $a_i^{(r)}$ be the mean cosine distance from $z_i$ to all other embeddings sharing its rank-$r$ label, and $b_i^{(r)}$ the mean cosine distance to embeddings in the nearest other rank-$r$ cluster. The silhouette coefficient is:

$$s_i^{(r)} = \frac{b_i^{(r)} - a_i^{(r)}}{\max\bigl(a_i^{(r)},\, b_i^{(r)}\bigr)}, \quad s_i^{(r)} \in [-1, 1].$$

We report the macro-average silhouette $S^{(r)} = N^{-1}\sum_i s_i^{(r)}$ at each of the seven Linnaean ranks. This allows us to separately quantify geometric organization at each level of the hierarchy.

**kNN purity at rank $r$** $(\uparrow$ better$)$. For each embedding $z_i$, let $\text{NN}_k(z_i)$ be its $k$ nearest neighbors by cosine distance. The kNN purity at rank $r$ is:

$$\text{Purity}^{(r)}_k = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{k} \bigl| \bigl\{j \in \text{NN}_k(z_i) : y_j^{(r)} = y_i^{(r)}\bigr\} \bigr|.$$

We use $k = 10$ throughout and report results at every rank.

**RankMe** $(\uparrow$ better$)$. \citet{garrido2023rankme} propose the *effective rank* of the embedding matrix as a collapse-free measure of representational quality:

$$\text{RankMe}(\mathbf{Z}) = \exp\!\left(-\sum_{k} p_k \log p_k\right), \quad p_k = \frac{\sigma_k(\mathbf{Z})}{\|\sigma(\mathbf{Z})\|_1},$$

where $\sigma(\mathbf{Z})$ is the vector of singular values of the $N \times d$ embedding matrix $\mathbf{Z}$. A higher RankMe indicates that the embedding distribution spans a larger effective subspace, guarding against representation collapse.

All metrics are computed on L2-normalized embeddings. We report *image-only* variants (using image embeddings $\{z_i\}$ alone) in parallel with *image-text* variants (using text prototypes $\{w_c\}$ as anchor) to isolate modality-gap artefacts \citep{liang2022mindgap} from genuine geometric effects.
