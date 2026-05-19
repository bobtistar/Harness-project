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
