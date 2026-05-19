# 3. Problem Formulation and Method

## 3.1 Notation and problem setup

Let $f_{\text{img}}: \mathcal{X} \to \mathbb{R}^D$ and
$f_{\text{txt}}: \mathcal{T} \to \mathbb{R}^D$ be CLIP's image and text
encoders, respectively, with $D$ the joint embedding dimension. For a class
set $\mathcal{C} = \{c_1, \ldots, c_K\}$ specified by prompt strings $t_k =
\text{prompt}(c_k)$ (e.g. "a photo of a $c_k$"), the unnormalised CLIP score
of image $x$ on class $c_k$ is
$$
  z_k(x) \;=\; \cos\!\bigl( f_{\text{img}}(x),\, f_{\text{txt}}(t_k) \bigr) \,/\, \tau_0,
$$
where $\tau_0$ is the model's learned logit-scale. The predicted probability is
$p_k(x) = \mathrm{softmax}(z(x))_k$.

**Base-to-novel split (CoOp convention).** For each dataset $\mathcal{D}$,
we partition $\mathcal{C}$ alphabetically into $\mathcal{C}_{\text{base}}$
(first half) and $\mathcal{C}_{\text{novel}}$ (second half). The model
(zero-shot or fine-tuned) is trained on 16-shot samples from
$\mathcal{C}_{\text{base}}$ only; calibration parameters are also fit on a
held-out base-validation split (per-class cap $N=512$); at test time we
evaluate on $\mathcal{C}_{\text{base}}$, on $\mathcal{C}_{\text{novel}}$,
and on their union $\mathcal{C}_{\text{base}} \cup \mathcal{C}_{\text{novel}}$
separately.

**Calibration error.** For a probability function $p$, the (15-bin
equal-mass) ECE~\cite{guo2017calibration} is
$$
  \mathrm{ECE}(p) \;=\; \sum_{m=1}^{M} \frac{|B_m|}{n} \,\bigl|\, \mathrm{acc}(B_m) - \mathrm{conf}(B_m) \,\bigr|.
$$
We additionally report Adaptive ECE~\cite{nixon2019measuring} (equal-count
bins), MCE, Brier score, and NLL. The central quantity for RQ1 is the
*base-to-novel gap*
$$
  \Delta\mathrm{ECE} \;:=\; \mathrm{ECE}_{\text{novel}} - \mathrm{ECE}_{\text{base}}.
$$

**Text-embedding geometry summary.** Let $\bar e_k = f_{\text{txt}}(t_k)$.
We define
$$
  \tau_{\text{txt}}(\mathcal{D}) \;=\; \frac{1}{K(K-1)} \sum_{j \neq k} \cos(\bar e_j, \bar e_k),
$$
the average pairwise cosine similarity of class-name embeddings on
dataset $\mathcal{D}$. $\tau_{\text{txt}}$ is a *single scalar* per
dataset and is the explanatory variable for RQ2. Larger
$\tau_{\text{txt}}$ corresponds to more crowded class manifolds—the
fine-grained regime—and our hypothesis is that it predicts higher
union ECE.

## 3.2 Approach overview

This paper does *not* propose a new calibrator. Its contribution is a
framework that (a) puts seven existing calibrators on a single grid,
(b) tests a causal hypothesis about *why* CLIP is mis-calibrated on
fine-grained $\times$ open-vocabulary inputs, and (c) tests whether
calibration improvements transfer monotonically to selective and
conformal prediction. The four research questions are answered by four
matched experiments (§4). A schematic of the full pipeline is shown in
[Figure 1: pipeline overview—dataset $\to$ backbone $\to$ fine-tuning
recipe $\to$ logits $\to$ calibrator $\to$ {ECE, AURC, conformal-set
metrics}, with a side branch where $\tau_{\text{txt}}$ is computed from
the text-encoder output].

## 3.3 Calibrator function classes

We consider seven function classes summarised in Table~1, taking logits $z
\in \mathbb{R}^K$ and producing calibrated logits $\tilde z$. Hyperparameters
are selected on base validation by L-BFGS minimisation of NLL.

[Table 1: Calibrator function classes C0--C6, their parametric form,
trainable parameter count, and calibration data.]

- **C0 Raw** — identity. Reference baseline.
- **C1 Temperature Scaling (TS)** — $\tilde z = z / T$, $T > 0$.
  Monotone in confidence rank; does *not* change accuracy or AURC for a
  fixed-rank scorer. Implements~\cite{guo2017calibration} and the
  zero-shot variant of~\cite{levine2023zsts}.
- **C2 Vector Scaling (VS)** — $\tilde z = \mathrm{diag}(w) z + b$,
  $w, b \in \mathbb{R}^K$. Per-class temperature and bias.
- **C3 Dirichlet / matrix scaling** — $\tilde z = W \log\mathrm{softmax}(z) + b$,
  $W \in \mathbb{R}^{K \times K}$, $b \in \mathbb{R}^K$, with L2
  regularisation as in~\cite{kull2019dirichlet}.
- **C4 Distance-Aware Calibration (DAC)** — sample-wise temperature
  $T_i = T \cdot g\!\bigl(d_{\text{txt}}(\hat y_i, \mathcal{C}_{\text{base}})\bigr)$
  where $d_{\text{txt}}$ is the cosine distance to the closest base
  text embedding~\cite{wang2024dac}.
- **C5 Contrast-Aware Calibration (CAC)** — calibration weight derived
  from the contrast between fine-tuned logits $z_{\text{ft}}$ and
  zero-shot logits $z_{\text{zs}}$: $\tilde z = z_{\text{ft}} -
  \alpha \cdot (z_{\text{zs}} - z_{\text{ft}})$ with $\alpha \in
  \{0.1, 0.3, 0.5, 1.0\}$~\cite{lv2025cac}.
- **C6 Histogram Binning (HB)** — non-parametric per-class bin map
  with 15 bins; *non-monotone* and may move AURC.

The set $\{$C0, C1, C2, C3, C4, C5, C6$\}$ spans (i) no calibration,
(ii) scalar parametric, (iii) vector parametric, (iv) matrix parametric,
(v) text-distance conditional, (vi) zero-shot-contrast conditional,
(vii) non-parametric. This is the function-class spectrum referenced in
the survey~\cite{wang2025chapter}, and to our knowledge no prior work
has compared all seven on a fixed grid.

## 3.4 Design choices and their rationale

- **Hyperparameter selection on base only.** Hyperparameters for every
  calibrator are selected by base-validation NLL. Novel labels are
  *never* seen by any calibrator. This is the deployment-relevant
  regime and is the design that makes the base-to-novel ECE gap
  meaningful.
- **Fixed prompt template plus 7-template ensemble.** We report both
  the single-template (`"a photo of a {C}."`) and the 7-template
  OpenAI ensemble. The pair quantifies the interaction between
  prompt engineering and calibration~\cite{tu2024empirical}.
- **Why $\tau_{\text{txt}}$ as the explanatory variable.** DAC uses
  *sample-wise* text distances and CAC uses *sample-wise* contrast;
  neither produces a dataset-level scalar that can be regressed
  across datasets. $\tau_{\text{txt}}$ is the simplest such scalar.
  If $\tau_{\text{txt}}$ does *not* correlate with ECE, the
  geometric hypothesis is falsified. If it does, the PCA-whitening
  intervention provides the causal step.
- **Why a Pareto frontier rather than a single union-ECE ranking.**
  Operators care about base ECE *and* novel ECE separately, and the
  failure modes that DOR~\cite{wang2024dor} documented have opposite
  signs across recipes. A scalar union-ECE collapses the
  trade-off; the Pareto frontier preserves it.
- **Why a monotonicity test for downstream metrics.** AURC and
  conformal set size are not mathematically guaranteed to move with
  ECE: monotone calibrators (TS, VS) leave AURC unchanged, while
  non-monotone ones (HB, Dirichlet) may move it in either
  direction~\cite{andradeloarca2024aurc}. A separate test is therefore
  needed to know whether calibration is a *useful proxy* for
  downstream behaviour.

## 3.5 Statistical procedure

All tests are dataset-level paired across the matched configurations of
backbone and prompt regime. The hypothesis-to-test mapping is:

| Hypothesis | Test | Correction |
|-----------|------|------------|
| H1.RQ1: $\Delta\mathrm{ECE}_{\text{fg}} > 1.5 \cdot \Delta\mathrm{ECE}_{\text{coarse}}$ | paired Wilcoxon + Mann–Whitney U | Holm across (backbone, prompt) |
| H1.RQ2a: Spearman$(\tau_{\text{txt}}, \mathrm{ECE}_{\text{union}}) > 0.5$ | permutation Spearman (10k) | Holm |
| H1.RQ2b: $\Delta\mathrm{ECE}$ post-whitening $< 0$ | paired bootstrap (10k) + sign test | Holm |
| H1.RQ3: $\mathrm{ECE}_{\text{C}_k}/\mathrm{ECE}_{\text{TS}} \leq 0.7$ | paired bootstrap on relative reduction | Friedman + Nemenyi |
| H1.RQ4: Spearman$(\mathrm{ECE}, \mathrm{AURC}) \leq -0.5$ | mixed-effects regression with dataset random intercept | Holm across (AURC, set size) |

Effect size is reported as paired Cohen's $d$ or rank-biserial
correlation. The full protocol is reproduced in §4.
