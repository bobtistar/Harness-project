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
