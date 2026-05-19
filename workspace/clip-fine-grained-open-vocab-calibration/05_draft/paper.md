# Fine-Grained, Open-Vocabulary Calibration of CLIP: A Framework for Statistical Quantification, Causal Hypothesis Testing, Pareto Comparison, and Monotonicity of Downstream Transfer

*Draft — pending full sweep. Numerical magnitudes anchored on
\cite{wang2024dac,lv2025cac,wang2024dor} until the full
11-dataset $\times$ 2-backbone $\times$ 3-seed sweep is run.*

---

# Abstract

Confidence calibration of CLIP-family vision-language models has been studied
extensively for either *open-vocabulary* (base-to-novel) deployment or for
*coarse* image classification, but rarely under the combination of the two:
fine-grained recognition with a class space that mixes base and unseen novel
labels. We argue that this intersection is the operational regime of most
real-world deployments (species identification, sub-type triage, SKU
recognition) and is precisely where existing calibrators are weakest. We
present (i) a statistical quantification framework for the *fine-grained
specificity* of the base-to-novel calibration gap, controlling for class count
and pre-training exposure; (ii) the hypothesis, together with an interventional
test plan, that the geometry of CLIP's class-name text embeddings—captured by
a single scalar $\tau_{\text{txt}}$—is a primary causal driver of the gap; (iii)
a head-to-head Pareto comparison of seven calibrator function classes
(identity, Temperature Scaling, Vector Scaling, Dirichlet, Distance-Aware
Calibration, Contrast-Aware Calibration, Histogram Binning) under a single
fixed protocol; and (iv) a monotonicity test of ECE against downstream AURC
and split-conformal set size, with a coverage-gap diagnostic. The code path
has been verified end-to-end on a toy CIFAR-10 slice; full-sweep numbers
(11 datasets $\times$ 2 backbones $\times$ 3 seeds) are pending and reported
as expected magnitudes anchored on Wang et al. (ICML 2024), Lv et al. (2025),
and Wang et al. (DOR, 2024). The contribution is therefore a precise,
falsifiable evaluation framework and an explanatory hypothesis—not a single
new calibrator—designed to put the four sub-problems of fine-grained,
open-vocabulary calibration on the same axis for the first time.

---

# 1. Introduction

CLIP-family vision-language models~\cite{radford2021clip,jia2021align,cherti2023openclip}
have made it routine to deploy a single image encoder against a class space
that is specified at *inference time* by free-form text. The same architecture
that classifies the 1000 ImageNet labels also classifies the 200 bird species
of CUB-200, the 196 car models of Stanford Cars, or—in production—an
ever-growing catalogue of stock-keeping units, defect sub-types, or medical
sub-classes. Two properties of this regime, however, jointly stress the
model's *confidence* outputs:

1. **Fine-grained class spaces** compress the inter-class margin in both image
   and text encoders: a "Tennessee Warbler" and a "Nashville Warbler" share
   most visual cues and almost all of their textual description. This yields a
   softmax distribution that is either nearly uniform (under-confident,
   high-entropy) or nearly one-hot on a wrong class (over-confident,
   winner-take-all), depending on the fine-tuning recipe.
2. **Open-vocabulary deployment** mixes classes that were seen at fine-tuning
   time (*base*) with classes that were not (*novel*). The text embeddings of
   novel classes sit at a different position in the embedding manifold than
   those of base classes, so any calibrator that was fit on base statistics is
   being asked to extrapolate.

Each of these two problems has its own literature. For (1) the fine-grained
side, FG-CLIP~\cite{xie2025fgclip}, FineCLIP~\cite{tian2024fineclip},
BioCLIP~\cite{stevens2024bioclip}, and description-based
prompting~\cite{menon2023descriptions} have improved *accuracy*, but none
report calibration. For (2) the open-vocabulary side, recent work---
LeVine et al.~\cite{levine2023zsts}, the Distance-Aware Calibrator (DAC)
of Wang et al.~\cite{wang2024dac}, the Contrast-Aware Calibrator (CAC) of
Lv et al.~\cite{lv2025cac}, Dynamic Outlier Regularization
(DOR)~\cite{wang2024dor}, the empirical study of
Tu et al.~\cite{tu2024empirical}, and calibrated robust fine-tuning of
Oh et al.~\cite{oh2024crft}---have produced calibrators that are evaluated
on an *aggregate* of coarse and fine-grained datasets, with no statistical
test isolating the fine-grained regime, no interventional test of cause,
and no head-to-head Pareto comparison across function classes.

Our position is that these are not equivalent regimes. The combination
*fine-grained $\times$ open-vocabulary* is a distinct measurement target,
and treating it as such reveals four concrete sub-questions that the prior
literature does not currently answer:

- **RQ1.** Is the base-to-novel calibration gap $\Delta\mathrm{ECE} :=
  \mathrm{ECE}_{\text{novel}} - \mathrm{ECE}_{\text{base}}$ *specifically*
  larger on fine-grained benchmarks than on coarse benchmarks, controlling
  for class count and exposure?
- **RQ2.** Does the geometry of the class-name text embeddings—summarised by
  the mean pairwise cosine similarity $\tau_{\text{txt}}$—causally drive the
  gap, in the sense that an *intervention* that spreads the text embeddings
  (PCA whitening) lowers ECE?
- **RQ3.** Among the seven calibrator function classes that have been proposed
  for VLMs, which dominate on the union (base $\cup$ novel) Pareto frontier,
  under a single fixed evaluation protocol?
- **RQ4.** Does improvement in ECE transfer monotonically to downstream
  selective-prediction (AURC) and split-conformal-prediction (mean set size)
  metrics, or does sharpness loss break the chain?

### Contributions

- **C1 (Statistical quantification of fine-grained specificity).** We
  formulate and test the hypothesis that $\Delta\mathrm{ECE}$ on
  fine-grained CLIP deployments is at least $1.5\times$ that on matched
  coarse deployments, using a dataset-level paired Wilcoxon signed-rank
  test with Holm correction across backbones and prompt regimes.
- **C2 (Text-embedding geometry as causal hypothesis).** We propose
  $\tau_{\text{txt}}$ as a *single-scalar* explanatory variable for
  mis-calibration and provide a two-axis verification plan—Spearman
  regression across datasets *and* a PCA-whitening intervention—going
  beyond DAC and CAC, which use sample-level distances without an
  interventional test.
- **C3 (Pareto comparison of seven calibrators).** We place seven function
  classes (Raw, TS, Vector Scaling, Dirichlet, DAC, CAC, Histogram Binning)
  on the same evaluation grid (11 datasets $\times$ 2 backbones $\times$ 3
  seeds), with hyperparameters selected on base validation only, and report
  the base $\times$ novel ECE Pareto frontier together with a paired-bootstrap
  test for the $\geq 30\%$ union-ECE reduction criterion.
- **C4 (Monotonicity of $\mathrm{ECE} \to \mathrm{AURC}$ / conformal).** We
  perform a mixed-effects regression of AURC and split-conformal mean set
  size on $\mathrm{ECE}$ with dataset random intercepts, and report a
  coverage-gap diagnostic that flags calibrators which improve ECE but
  violate the $\pm 1\,\mathrm{pp}$ coverage tolerance.

### Honest scope note on numerical results

The code path of all four experiments has been verified end-to-end on a toy
CIFAR-10 slice. The full sweep over 11 datasets $\times$ 2 backbones $\times$
3 seeds was *not* executed in the current environment, because most
fine-grained datasets (CUB-200, iNat subset, ImageNet-1k, SUN397) require
manual download outside torchvision's automatic fetcher. Numerical claims
in §5 are therefore reported as *expected magnitudes* (anchored on the
published Tables of DAC~\cite{wang2024dac}, CAC~\cite{lv2025cac}, and
DOR~\cite{wang2024dor}) and flagged "(pending full sweep)". The
contribution of this paper is the framework, the falsifiable hypotheses,
the protocol, and the code—not a numerical victory claim.

### Paper organisation

§2 categorises the related work and isolates the four gaps that motivate
RQ1--RQ4. §3 introduces notation, the calibration problem under
base-to-novel splits, and the seven calibrator function classes. §4
specifies the experimental protocol, datasets, baselines, metrics, and
statistical tests. §5 reports the (pending) results against each RQ. §6
discusses interpretation, threats to validity, and limitations. §7
concludes.

---

# 2. Related Work

We organise the literature into four categories that map directly onto our
research questions: (A) CLIP-family and fine-grained vision-language
pre-training, (B) prompt learning and open-vocabulary classification,
(C) calibration of deep networks and of VLMs in particular, and
(D) selective and conformal prediction under distribution shift.

## 2.1 CLIP-family and fine-grained vision-language pre-training

CLIP~\cite{radford2021clip} and ALIGN~\cite{jia2021align} established the
dual-encoder contrastive paradigm in which a class is specified by a text
prompt at inference time. OpenCLIP~\cite{cherti2023openclip} provides public,
reproducible checkpoints that we adopt as our backbones. For fine-grained
recognition, BioCLIP~\cite{stevens2024bioclip} adapts CLIP to the tree of
life and reports $+16$--$17$ pp absolute accuracy gains on bio benchmarks;
FG-CLIP~\cite{xie2025fgclip} combines long-caption pre-training with
region-level alignment and hard-negative mining; FineCLIP~\cite{tian2024fineclip}
employs self-distilled, region-based contrastive learning; and Menon &
Vondrick~\cite{menon2023descriptions} replace single-template prompts with
LLM-generated attribute descriptions. These works improve fine-grained
*accuracy* but, with the exception of acknowledging that fine-grained scores
are "less peaked", report no calibration metrics.

*Difference from this work.* We take the fine-grained accuracy story as
given and ask the orthogonal calibration question: when CLIP is asked to
score fine-grained classes that mix base and novel labels, what is the
shape of its mis-calibration, and what causes it?

## 2.2 Prompt learning and open-vocabulary classification

CoOp~\cite{zhou2022coop} introduced learnable context tokens; CoCoOp~\cite{zhou2022cocoop}
made them image-conditioned and established the base-to-novel evaluation
split that we adopt. MaPLe~\cite{khattak2023maple} couples vision- and
language-side prompts; PromptSRC~\cite{khattak2023promptsrc} regularises
toward a fixed "a photo of" anchor; ProGrad and KgCoOp project or
regularise gradients to mitigate forgetting; CLIP-Adapter~\cite{gao2024clipadapter}
adds a residual feature adapter as a lightweight alternative.

*Difference from this work.* These methods optimise accuracy on the
base-to-novel split; we re-use their *checkpoints* but treat the
calibration of their outputs as the dependent variable, not the accuracy.

## 2.3 Calibration of deep networks and of VLMs

*General calibration.* Guo et al.~\cite{guo2017calibration} popularised
Temperature Scaling (TS) as a simple, monotone, ECE-reducing post-hoc fix.
Dirichlet calibration~\cite{kull2019dirichlet} generalises TS to a
parametric linear-on-log-probs layer. Label smoothing~\cite{mueller2019labelsmoothing}
and focal loss~\cite{mukhoti2020focal} are training-time alternatives.
Adaptive ECE~\cite{nixon2019measuring} resolves binning artefacts.
Minderer et al.~\cite{minderer2021revisiting} report that modern ViTs are
better calibrated than CNNs and that calibration scales with model size.

*Calibration of VLMs.* LeVine et al.~\cite{levine2023zsts} show that a
single learned $T$ transfers across zero-shot inference contexts. Wang et
al.'s DAC~\cite{wang2024dac} introduces a sample-wise temperature that
scales with the predicted label's textual distance to the base class set;
it reports consistent ECE reductions across 7 prompt-learning methods on
11 datasets. Lv et al.'s CAC~\cite{lv2025cac} computes a calibration
weight from the contrast between original and fine-tuned CLIP, explicitly
addressing the base/novel asymmetry. Wang et al.'s DOR~\cite{wang2024dor}
diagnoses a *sign asymmetry*—CoOp produces novel over-confidence while
KgCoOp produces base under-confidence—and proposes a regulariser. Tu et
al.~\cite{tu2024empirical} sweep over what factors actually move VLM
calibration and conclude that temperature scaling is robust even under
distribution shift. Oh et al.~\cite{oh2024crft} combine calibration with
robust fine-tuning. A 2025 chapter~\cite{wang2025chapter} surveys this
sub-field.

*Difference from this work.* The closest prior work---DAC, CAC, DOR---
evaluates on a *mixed* set of coarse and fine-grained benchmarks, with
no statistical test that isolates fine-grained specificity, no
interventional test of cause, and no single-grid head-to-head comparison
of all seven function classes. We provide all three.

## 2.4 Selective prediction and conformal prediction under shift

Geifman & El-Yaniv~\cite{geifman2017selective} defined the risk-coverage
framework and AURC. Traub et al.~\cite{traub2024selective} document common
pitfalls in evaluating selective classification and standardise the AURC
protocol; Andrade-Loarca et al.~\cite{andradeloarca2024aurc} characterise
the population AURC and finite-sample estimators. Tibshirani et
al.~\cite{tibshirani2019conformal} extend split conformal to covariate
shift via likelihood-ratio reweighting. The most relevant recent work,
Conf-OT~\cite{silvarodriguez2025confot} (CVPR 2025), demonstrates that
split conformal with transductive transfer preserves coverage on
zero-shot CLIP across 15 datasets and three non-conformity scores.

*Difference from this work.* Conf-OT establishes that coverage is
*possible*; we ask the orthogonal question of whether calibration
*improvements* (measured by ECE) transfer monotonically to AURC and
mean set size, using a mixed-effects regression with dataset random
intercepts, and we report a per-calibrator coverage-gap diagnostic.

## 2.5 Summary of gaps

| Gap | Closest prior | What is missing |
|-----|---------------|-----------------|
| Fine-grained vs. coarse $\Delta\mathrm{ECE}$ test | DAC, CAC, DOR | dataset-level paired test isolating fine-grained group |
| Causal role of text-embedding geometry | DAC (sample distance), CAC (contrast) | dataset-level scalar $\tau_{\text{txt}}$ + interventional whitening |
| Seven-class head-to-head Pareto | CAC vs. DAC pairwise | single-grid 7-way comparison with paired-bootstrap test |
| $\mathrm{ECE} \to \mathrm{AURC}$ / conformal monotonicity | Conf-OT (coverage only) | mixed-effects regression of downstream metrics on ECE |

The four rows of this table directly motivate RQ1--RQ4 in §3 and §5.

---

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

---

# 4. Experimental Protocol

This section specifies the datasets, models, baselines, metrics, hyper-
parameters, and statistical tests against which RQ1--RQ4 are evaluated.
Every experiment shares the seed set $\{0, 1, 2\}$, the OpenCLIP
backbones, and the CoOp base-to-novel split.

## 4.1 Datasets

**Fine-grained group (7).** CUB-200-2011 (200 classes), Stanford Cars
(196), FGVC-Aircraft (100), Flowers-102 (102), Oxford-IIIT Pets (37),
Food-101 (101), iNaturalist-2021 subset (200 species). These benchmarks
are characterised by small inter-class image differences and nearly
isomorphic class-name strings.

**Coarse group (4, RQ1 control).** ImageNet-1k val (1000), Caltech-101
(101), SUN397 (397), EuroSAT (10).

**Base-to-novel split.** Each class set is partitioned alphabetically:
the first half becomes $\mathcal{C}_{\text{base}}$, the second half
$\mathcal{C}_{\text{novel}}$. Three random splits (CoOp's seed1/2/3) are
averaged. Per-class test sub-sampling is capped at 50 images for the
class-balanced sensitivity analysis reported in the appendix.

**Confounder controls.** (i) Class-count matching: a 200-class
sub-sample of ImageNet/SUN397 is reported alongside the full-class
result. (ii) Sample-count matching: per-class cap to $\min(50,
\text{available})$. (iii) Pre-training exposure proxy: LAION-400M
metadata class-name token frequency reported as a covariate where
available. (iv) Prompt template: results reported for both
`"a photo of a {C}."` and the 7-template OpenAI ensemble.

## 4.2 Backbones and fine-tuning recipes

**Backbones.** OpenCLIP ViT-B/16 (`laion2b_s34b_b88k`) and ViT-L/14
(`laion2b_s32b_b82k`)~\cite{cherti2023openclip}.

**Fine-tuning recipes** (we re-use published checkpoints where
available; zero-shot is the fallback): CoOp~\cite{zhou2022coop} with
$n_{\text{ctx}}=16$, 16-shot, 50 epochs; CoCoOp~\cite{zhou2022cocoop}
with $n_{\text{ctx}}=4$ and image-conditioned context; MaPLe~\cite{khattak2023maple}
with prompt depth 9; LoRA-adapter (rank 8 on image-encoder QKV) as a
secondary recipe.

## 4.3 Baselines

We compare seven calibrator function classes (C0--C6) as listed in §3.3.
The recipe-level baselines are: B0 zero-shot CLIP; B1 zero-shot + global
TS~\cite{levine2023zsts}; B2 CoOp/CoCoOp/MaPLe + TS; B3 DAC~\cite{wang2024dac};
B4 CAC~\cite{lv2025cac}; B5 Dirichlet~\cite{kull2019dirichlet}; B6
prompt-ensemble best recipe of~\cite{tu2024empirical}. For RQ4, the
conformal baseline is Conf-OT~\cite{silvarodriguez2025confot}; the
selective-prediction baseline is the MSP threshold of~\cite{geifman2017selective},
evaluated under the AURC protocol of~\cite{traub2024selective}.

## 4.4 Metrics

**Calibration.** ECE (15-bin equal-mass), Adaptive ECE~\cite{nixon2019measuring},
classwise ECE, MCE, Brier score, NLL. Reported on
$\mathcal{C}_{\text{base}}$, $\mathcal{C}_{\text{novel}}$, and their union.

**Accuracy.** Top-1 accuracy and the difference $\Delta\mathrm{acc} =
\mathrm{acc}_{\text{cal}} - \mathrm{acc}_{\text{raw}}$, required to
satisfy $|\Delta\mathrm{acc}| \leq 0.5\,\mathrm{pp}$ for any calibrator
to be considered valid.

**Selective prediction.** AURC (Geifman-El-Yaniv; finite-sample
estimator of~\cite{andradeloarca2024aurc}), selective accuracy at
coverage $\{0.7, 0.8, 0.9\}$, and AUROC of correct-vs-incorrect
prediction.

**Conformal prediction.** Split conformal with non-conformity score
$s(x, y) = 1 - p_y(x)$, at miscoverage $\alpha = 0.1$. We report
empirical coverage on base test and on novel test, and average set
size. A coverage gap $|\hat{\mathrm{cov}} - (1 - \alpha)|$ above
$1\,\mathrm{pp}$ flags the calibrator.

## 4.5 Implementation details

**Optimisation.** SGD with $\mathrm{lr}=2 \times 10^{-3}$, momentum
$0.9$, weight decay $5 \times 10^{-4}$, cosine schedule; batch size 32
(ViT-B/16) and 16 (ViT-L/14). Calibration parameters are fit by L-BFGS,
`max_iter=200`, `tolerance_grad=1e-7`, in fp32; the forward pass uses
fp16.

**Software.** Python 3.12, PyTorch 2.7.1+cu118, open_clip_torch 3.2.0,
torchvision 0.22.1+cu118, scikit-learn 1.5.1, scipy 1.13.1, numpy
1.26.4. The pinned `requirements.txt` is shipped with the code.

**Hardware.** NVIDIA GPU with $\geq 16\,\text{GB}$ VRAM (required for
ViT-L/14 inference at batch 16). The sanity-check environment is
Windows 11, CUDA 11.8.

**Seed control.** `torch.manual_seed`, `numpy.random.seed`, and
`random.seed` are all set per run, and
`torch.backends.cudnn.deterministic = True`.

**Reproducibility logging.** Every run writes git hash, env hash, CLI
arguments, seed, dataset checksum, and output path to
`logs/<run_id>.json`.

## 4.6 Statistical tests

| Hypothesis | Test | Multiple-comparison correction |
|------------|------|-------------------------------|
| H1.RQ1 (fine-grained $\Delta\mathrm{ECE}$ larger) | paired Wilcoxon signed-rank; group Mann–Whitney U (one-sided) | Holm–Bonferroni across (backbone $\times$ prompt) |
| H1.RQ2a (correlation) | Spearman $\rho$ with 10k permutations; partial Spearman conditioning on accuracy and #classes | Holm across (dataset-level, class-level) |
| H1.RQ2b (intervention) | paired bootstrap CI (10k); one-sided sign test on per-dataset $\Delta\mathrm{ECE}$ | Holm per dataset |
| H1.RQ3 (relative reduction $\geq 30\%$) | paired bootstrap CI on $(\mathrm{ECE}_{\text{TS}} - \mathrm{ECE}_{\text{C}_k}) / \mathrm{ECE}_{\text{TS}}$; Friedman + Nemenyi for rank aggregation | Nemenyi at $\alpha = 0.05$ |
| H1.RQ4 (monotonicity) | Spearman $\rho$ within each dataset; mixed-effects regression $y \sim \mathrm{ECE} + (1 \mid \mathrm{dataset})$ with $y \in \{\mathrm{AURC}, \bar{|\mathcal{S}|}\}$ | Holm across $\{\mathrm{AURC}, \bar{|\mathcal{S}|}\}$ |

Effect sizes are reported as paired Cohen's $d$ or rank-biserial
correlation, alongside 95% bootstrap CIs.

## 4.7 Execution status

The code path of all four experiments (`exp1_diagnosis.py`,
`exp2_geometry.py`, `exp3_calibrators.py`, `exp4_downstream.py`) has been
verified end-to-end on a toy CIFAR-10 slice (128 images, ViT-B/32
OpenAI, single seed). All four scripts complete successfully and write
valid JSON to `code/outputs/*.json`. The toy numbers verify the code
path; they do not support scientific conclusions. The full sweep
(11 datasets $\times$ 2 backbones $\times$ 3 seeds $\times$ 7
calibrators) was *not* executed in the current environment because
CUB-200, iNat subset, ImageNet-1k val, and SUN397 require manual
download outside torchvision's automatic fetcher. The full-sweep
infrastructure (`run_all.sh`) is in place; only the dataset staging and
the swap from zero-shot to fine-tuned checkpoints remain.

## 4.8 Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Missing fine-tuning checkpoint | Zero-shot CLIP fallback; explicit annotation in results |
| Dataset download restrictions | torchvision-fetchable datasets first; `ImageFolder` stubs for the rest |
| ViT-L/14 OOM | Gradient checkpointing; batch reduction; fp16 |
| Multiple-comparison false positives | Holm-Bonferroni correction applied throughout |
| Calibration-set hyperparameter overfitting | Base-validation NLL only; nested CV |
| Conformal coverage violation | Zero-shot baseline coverage reported as sanity check |

---

# 5. Results

The numerical magnitudes reported in this section are *expected* values
(pending full sweep), anchored on three published references: DAC
Tables 1--3~\cite{wang2024dac}, CAC Table 2~\cite{lv2025cac}, and DOR
Table 1~\cite{wang2024dor}. Each subsection therefore lists (a) the
expected direction with its statistical decision rule and (b) the
falsification condition under which we would conclude the hypothesis is
not supported. Sanity-check numbers from the toy CIFAR-10 run are
flagged explicitly and used only to confirm the code path.

## 5.1 Main results

[Table 2: Main calibration results. Rows: 11 datasets $\times$ 2
backbones $\times$ 3 seeds; columns: $\mathrm{ECE}_{\text{base}}$,
$\mathrm{ECE}_{\text{novel}}$, $\mathrm{ECE}_{\text{union}}$, MCE, Brier,
NLL, accuracy. One block per calibrator C0--C6. (pending full sweep)]

[Figure 2: $\mathrm{ECE}_{\text{base}}$ vs. $\mathrm{ECE}_{\text{novel}}$
Pareto frontier across calibrators, averaged across datasets within
each domain group (fine-grained, coarse). (pending full sweep)]

## 5.2 Ablations

[Table 3: Ablation on $\tau_{\text{txt}}$ intervention. Rows: 11
datasets; columns: $\tau_{\text{txt}}$ pre-/post-PCA-whitening, ECE
pre-/post-whitening, accuracy delta. (pending full sweep)]

[Table 4: Prompt-regime ablation. Single template vs. 7-template
ensemble (logit/feature/probability averaging). (pending full sweep)]

## 5.3 RQ-by-RQ answers

### RQ1 — Is fine-grained $\Delta\mathrm{ECE}$ larger than coarse?

**Hypothesis (H1).** The mean per-dataset $\Delta\mathrm{ECE}$ on the
fine-grained group is at least $1.5\times$ that on the coarse group, with
paired-test $p < 0.05$ after Holm correction.

**Expected magnitude** (anchored on DAC and DOR tables):

| Group | Backbone | n | Expected $\Delta\mathrm{ECE}$ (raw) | Expected $\Delta\mathrm{ECE}$ (+TS) |
|-------|----------|---|-----:|-----:|
| Fine-grained | ViT-B/16 | 7 | $\approx +6.5$ pp | $\approx +4.0$ pp |
| Coarse       | ViT-B/16 | 4 | $\approx +2.0$ pp | $\approx +1.0$ pp |
| Fine-grained | ViT-L/14 | 7 | $\approx +5.0$ pp | $\approx +3.0$ pp |
| Coarse       | ViT-L/14 | 4 | $\approx +1.5$ pp | $\approx +0.7$ pp |

**Decision rule.** H1 is supported iff (mean ratio $\geq 1.5$) AND
($p < 0.05$ after Holm) AND the relationship survives the 200-class
sub-sampling robustness check. Otherwise H0 is retained.

**Status.** Pending full sweep. Status code: *not yet decided*.

### RQ2 — Does $\tau_{\text{txt}}$ drive ECE?

**Hypothesis (H1).** (a) Spearman$(\tau_{\text{txt}},
\mathrm{ECE}_{\text{union}}) \geq 0.55$ across 11 datasets with $p < 0.05$;
partial correlation conditioning on accuracy $\geq 0.45$. (b) PCA-whitening
of the $K \times D$ text embedding matrix reduces ECE by $\geq 1\,\mathrm{pp}$
on at least $8/11$ datasets (sign test $p < 0.01$). (c) A synthetic
interpolation from non-whitened to fully whitened yields a monotone
ECE drop.

**Sanity-check observation (toy CIFAR-10, ViT-B/32, 128 images).**
PCA-whitening decreased ECE from $0.105$ to $0.095$, a small but
correctly-signed effect. Orthogonalisation *increased* ECE to $0.341$,
consistent with DAC's warning that destructive embedding manipulations
can hurt accuracy. We therefore retain PCA-whitening as the primary
intervention and use orthogonalisation only as a destructive-control.

**Decision rule.** H1 is supported iff both the correlation criterion
and the intervention criterion are met. Falsification: either
$|\rho| < 0.3$ or the sign test fails.

**Status.** Pending full sweep. The toy result is consistent with H1
but not statistically meaningful at $n = 128$.

### RQ3 — Which calibrator wins the base$\cup$novel Pareto?

**Hypothesis (H1).** The best calibrator achieves (i) $\geq 30\%$
relative reduction in $\mathrm{ECE}_{\text{union}}$ vs. C1 (TS) with
paired-bootstrap CI excluding $0$, (ii) novel/base ECE ratio $\leq 1.5$,
and (iii) accuracy drop $\leq 0.5\,\mathrm{pp}$, simultaneously.

**Expected results** (per Wang24/Lv25 trends):

| Calibrator | base ECE | novel ECE | union ECE | $\Delta\mathrm{ECE}$ | $\Delta\mathrm{acc}$ |
|------------|---------:|----------:|----------:|---------------------:|---------------------:|
| C0 Raw           | $\approx 0.04$ | $\approx 0.13$ | $\approx 0.09$ | $+0.09$ | $0$ |
| C1 TS            | $\approx 0.03$ | $\approx 0.10$ | $\approx 0.07$ | $+0.07$ | $0$ |
| C2 Vector Scaling| $\approx 0.025$ | $\approx 0.11$ | $\approx 0.07$ | $+0.085$ | $\leq 0.3$ |
| C3 Dirichlet     | $\approx 0.025$ | $\approx 0.10$ | $\approx 0.07$ | $+0.075$ | $\leq 0.3$ |
| **C4 DAC**       | $\approx 0.028$ | $\approx 0.06$ | $\approx 0.045$ | $+0.032$ | $0$ |
| **C5 CAC**       | $\approx 0.030$ | $\approx 0.055$ | $\approx 0.043$ | $+0.025$ | $0$ |
| C6 Histogram Binning | $\approx 0.02$ | $\approx 0.13$ | $\approx 0.08$ | $+0.11$ | $\leq 0.5$ |

(All numerical entries pending full sweep.)

**Expected Pareto frontier.** DAC and CAC are expected to dominate;
CAC is expected to deliver the lowest novel ECE owing to its
zero-shot-contrast signal. Histogram Binning is expected to win on
base alone but to perform poorly on novel because its parameter count
scales with $K$ and falls back to a TS proxy on unseen classes.

**Decision rule.** Friedman test across 11 datasets per backbone;
Nemenyi post-hoc at $\alpha = 0.05$; the three-condition compound
criterion above.

**Status.** Pending full sweep.

### RQ4 — Does ECE improvement transfer monotonically to AURC and conformal set size?

**Hypothesis (H1).** Within each dataset, Spearman$(\mathrm{ECE}_{\text{union}},
\mathrm{AURC}) \leq -0.55$; mean conformal set size at $\alpha = 0.1$
decreases monotonically with ECE; empirical coverage stays within
$\pm 1\,\mathrm{pp}$ on base test and within $\pm 2\,\mathrm{pp}$ on
novel test.

**Theoretical caveat.** AURC depends only on the *ranking* of
confidences, not their values. Monotone calibrators (TS, VS) therefore
*cannot* move AURC by construction; movement is expected only for the
non-monotone calibrators (HB, Dirichlet, DAC's sample-wise temperature,
CAC's contrast reweighting). This is consistent
with~\cite{andradeloarca2024aurc} and means that AURC monotonicity
holds *within the non-monotone calibrator subset*, not across the
entire seven-class set.

**Decision rule.** Mixed-effects regression $y \sim \mathrm{ECE} +
(1 \mid \mathrm{dataset})$; fixed-effect 95% CI excluding $0$ in the
expected direction; coverage gap diagnostic table.

**Status.** Pending full sweep. The toy CIFAR-10 run confirms that
the coverage-gap diagnostic correctly flags HB on the novel slice.

## 5.4 Statistical summary

The four hypotheses above are designed so that they can all be answered
*independently*, and the rejection of any one of them is informative
in its own right. The pre-registered decision rules are summarised in
Table~5.

[Table 5: Pre-registered decision rules. Columns: RQ, hypothesis,
test, success threshold, falsification condition. (the entries are
exactly those of §4.6, restated here for self-containment.)]

---

# 6. Discussion

## 6.1 Interpretation (under the expected outcomes)

If the expected results hold, three concrete claims follow.

*First*, the fine-grained $\times$ open-vocabulary regime is
quantitatively different from the coarse $\times$ open-vocabulary
regime. The $\Delta\mathrm{ECE}$ gap is not merely larger but is
*systematically* so, and survives class-count and exposure controls.
This implies that any calibrator validated only on a mixed coarse +
fine-grained suite (which is the current practice in DAC, CAC, and
DOR) is reporting an *average* over two regimes with different
underlying physics, and may overestimate its effectiveness on the
fine-grained regime that matters in deployment.

*Second*, the geometry of CLIP's text embeddings is not just a
correlate but a *cause*: PCA-whitening is a minimally invasive
intervention that should reduce ECE without harming accuracy. This
relocates the calibration problem from a purely output-side
post-processing question to a question about pre-training and
fine-tuning objectives that determine the angular distribution of
class-name embeddings.

*Third*, on a single fixed grid, DAC and CAC are expected to
dominate the union ECE Pareto frontier, with CAC delivering the
lowest novel ECE and DAC delivering the most favourable base-novel
balance. Histogram Binning is expected to *win* on base ECE but to
*lose* on novel ECE—a clear demonstration that
parametric-count-in-$K$ calibrators do not generalise to unseen
classes without an explicit text-geometry signal.

*Fourth*, the ECE-to-downstream transfer is monotone *only*
within the non-monotone calibrator subset. For monotone
calibrators (TS, VS), ECE improvements do not propagate to AURC, so
selective-prediction quality must be measured directly rather than
inferred from ECE. This is an under-appreciated point in the prior
VLM calibration literature.

## 6.2 Limitations

1. **Execution status.** The full sweep has not been run. All
   numerical magnitudes are *expected* values anchored on prior
   tables, not measured. The contribution is the framework, the
   protocol, the falsifiable hypotheses, and the code; the
   numerical victory claims will be made only after the full sweep.
2. **Backbone scope.** We restrict to OpenCLIP ViT-B/16 and ViT-L/14.
   Generalisation to SigLIP, EVA-CLIP, or DFN families is left to
   future work but is straightforward in the same protocol.
3. **Fine-tuning recipe scope.** We focus on prompt learning (CoOp,
   CoCoOp, MaPLe) and a single adapter recipe (LoRA, rank 8). Full
   fine-tuning of CLIP is excluded because it leaves the
   open-vocabulary regime.
4. **Single non-conformity score.** For conformal prediction we fix
   $s(x, y) = 1 - p_y(x)$; APS, RAPS, and SAPS comparisons are
   deferred so that RQ4 isolates the ECE-to-set-size relationship
   rather than the choice of non-conformity score.
5. **Static evaluation.** Active-learning loops, pseudo-label
   self-training, and continual class expansion all rely on
   calibrated confidence but exercise it dynamically; we evaluate
   only the static deployment regime.

## 6.3 Threats to validity

**Internal validity.** The most serious internal threat is calibration-
set overfitting on small base validation splits. We mitigate by L2
regularisation on Dirichlet, early stopping on L-BFGS, and a nested
cross-validation report in the appendix. A second threat is that the
CoOp alphabetical base/novel split is not deeply *random* and may
correlate with semantic structure; we average across the three CoOp
seeds to mitigate.

**External validity.** Generalisation beyond OpenCLIP ViT-B/16 and
ViT-L/14 is asserted but not tested. Generalisation to non-English
class names, to long-tail class distributions, and to safety-critical
deployments (medical sub-classification, autonomous driving sub-type
recognition) is asserted as motivation but not measured.

**Construct validity.** ECE itself is a flawed metric: it has binning
artefacts, is not a proper scoring rule, and can be gamed by trivial
recalibration of confidence ranges. We mitigate by reporting Adaptive
ECE~\cite{nixon2019measuring}, Brier score, NLL, and classwise ECE in
parallel, and we accept ECE only when at least Brier and NLL move in
the same direction.

**Statistical validity.** Eleven datasets and three seeds is a modest
sample for a paired test with Holm correction across four (backbone
$\times$ prompt) combinations. We report effect sizes alongside
$p$-values to ensure that statistically significant results are also
practically significant, and we use permutation tests (10k
resamples) rather than parametric ones wherever the underlying
distribution is questionable.

**Construct validity of $\tau_{\text{txt}}$.** $\tau_{\text{txt}}$ is
a single scalar and necessarily lossy. A dataset can have low
$\tau_{\text{txt}}$ but still be fine-grained in image space (e.g. if
class names are short and semantically distant but the visual cues
overlap). We report Adaptive ECE and image-side margin distributions
as complementary characterisations.

## 6.4 Future work

- **A new conditional calibrator** explicitly fit on the
  text-geometry axis. The Pareto results of RQ3 will tell us which
  function class to use as a base, and the causal results of RQ2
  will tell us what conditioning signal to add.
- **Continual-class deployment.** A streaming setting in which new
  novel classes are added every $T$ time steps, with calibrator
  state updated online.
- **Multi-modal calibration.** Joint calibration of CLIP for
  classification *and* retrieval, where the latter has a different
  set of failure modes.
- **Theory of $\tau_{\text{txt}}$.** If the empirical correlation
  holds, the next step is to derive a closed-form bound on ECE in
  terms of the text-embedding spectrum, possibly via a Lipschitz
  argument on the softmax under crowded class prototypes.

---

# 7. Conclusion

We have argued that the fine-grained, open-vocabulary deployment regime
of CLIP-family vision-language models is a distinct calibration target
that has been under-studied: existing work evaluates on a mixed
coarse + fine-grained suite, proposes calibrators based on
sample-level signals without dataset-level explanatory variables, and
does not test the monotonicity of calibration improvements through to
downstream selective-prediction and conformal-prediction metrics. To
address this we contributed (i) a paired-test framework that
statistically isolates the fine-grained specificity of the
base-to-novel calibration gap, (ii) a single-scalar explanatory
variable $\tau_{\text{txt}}$ together with a falsifiable interventional
protocol (PCA-whitening), (iii) a head-to-head Pareto comparison of
seven calibrator function classes (Raw, TS, Vector Scaling, Dirichlet,
DAC, CAC, Histogram Binning) under a single fixed evaluation grid, and
(iv) a mixed-effects monotonicity test from ECE to AURC and conformal
set size with a per-calibrator coverage-gap diagnostic. The code path
of all four experiments has been verified end-to-end; the full sweep
(11 datasets $\times$ 2 backbones $\times$ 3 seeds) is pending and the
expected magnitudes are anchored on DAC, CAC, and DOR. We expect
fine-grained $\Delta\mathrm{ECE}$ to exceed the coarse counterpart by
at least $1.5\times$, $\tau_{\text{txt}}$ to correlate positively with
ECE and to be reduced by PCA-whitening, DAC and CAC to dominate the
Pareto frontier, and the ECE-to-downstream transfer to be monotone
only within the non-monotone calibrator subset. The framework is
designed to be falsified just as cleanly as it is to be confirmed, and
the next step is the full sweep on staged datasets together with a
new conditional calibrator fit on the $\tau_{\text{txt}}$ axis.

---

# References

See `references.bib` for the BibTeX entries. The cited keys
(in order of first appearance) are:
`radford2021clip`, `jia2021align`, `cherti2023openclip`,
`xie2025fgclip`, `tian2024fineclip`, `stevens2024bioclip`,
`menon2023descriptions`, `levine2023zsts`, `wang2024dac`,
`lv2025cac`, `wang2024dor`, `tu2024empirical`, `oh2024crft`,
`zhou2022coop`, `zhou2022cocoop`, `khattak2023maple`,
`khattak2023promptsrc`, `gao2024clipadapter`, `li2022glip`,
`guo2017calibration`, `kull2019dirichlet`, `mueller2019labelsmoothing`,
`mukhoti2020focal`, `nixon2019measuring`, `minderer2021revisiting`,
`geifman2017selective`, `traub2024selective`,
`andradeloarca2024aurc`, `tibshirani2019conformal`,
`silvarodriguez2025confot`, `wang2025chapter`.
