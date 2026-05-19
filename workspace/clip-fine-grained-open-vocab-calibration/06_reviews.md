# Simulated Peer Review

*Venue assumed: a top-tier ML conference (NeurIPS / ICML / ICLR / CVPR) with empirical track and double-blind review. Decisions calibrated accordingly.*

---

## Reviewer 1 — Methodology Specialist

**Score (1-10)**: 5
**Recommendation**: Borderline
**Confidence (1-5)**: 4

### Summary (저자가 무엇을 했다고 이해했는지)
The authors propose an evaluation *framework* (not a new calibrator) for the fine-grained × open-vocabulary calibration regime of CLIP. Four research questions are formalised: (RQ1) a paired test that the base-to-novel ECE gap is ≥1.5× larger on fine-grained than coarse benchmarks; (RQ2) a single-scalar geometric explanatory variable τ_txt with a PCA-whitening interventional check; (RQ3) a 7-way Pareto comparison of calibrator function classes; (RQ4) a mixed-effects monotonicity test from ECE to AURC and to conformal set size. The protocol is fully specified, statistical tests are pre-registered (§3.5, §4.6), and all expected magnitudes are explicitly anchored on DAC/CAC/DOR tables and flagged "pending full sweep" (§4.7, §1 honest-scope-note, §5).

### Strengths
- S1: The hypothesis-to-test mapping table in §3.5 and §4.6 is unusually rigorous for a calibration paper — paired Wilcoxon + Holm, Friedman + Nemenyi, paired bootstrap with explicit resample counts, and a mixed-effects model with dataset random intercepts are all sensible and pre-registered.
- S2: The choice to fit calibrator hyperparameters on base-only validation (§3.4) is the deployment-relevant operating point and is consistent across all seven function classes, removing a common source of unfairness.
- S3: Effect sizes (Cohen's d, rank-biserial) are reported alongside p-values (§4.6, §6.3), which is good practice and ahead of much of the calibration literature.
- S4: The AURC theoretical caveat (§5.3 RQ4) — that monotone calibrators *cannot* move AURC by construction — is explicitly acknowledged. This is technically correct and unfortunately ignored by several published VLM calibration papers.

### Weaknesses
- W1 [§3.1, §3.5]: The choice of "1.5×" as the H1 threshold for RQ1 is **arbitrary and not justified**. Why not 1.2×, 2.0×, or a continuous Bayes factor? A pre-registered threshold needs either a power calculation (what effect would be practically meaningful for deployment?) or a citation that nominates this value. Currently it appears reverse-engineered from the DAC/DOR anchor numbers.
- W2 [§3.5]: The Spearman threshold of ρ ≥ 0.55 for RQ2a is similarly unjustified, and with n=11 datasets, the **statistical power to detect ρ=0.55 against ρ=0** is at best ~70% at α=0.05 (two-sided). After Holm correction across (backbone × prompt) the effective α drops further. The paper should either (a) report a power analysis, or (b) expand the dataset count, or (c) move to a within-dataset class-level analysis that gives more degrees of freedom.
- W3 [§3.5, RQ1 row]: Combining a paired Wilcoxon (within matched pairs) **and** a Mann-Whitney U (across groups) for the same hypothesis is methodologically unclear — fine-grained and coarse datasets are *not* paired in any meaningful sense (different label counts, different images). The paired Wilcoxon should be on (raw vs. +TS) per dataset, while the across-group comparison is the Mann-Whitney. The current table conflates them.
- W4 [§3.1, τ_txt definition]: τ_txt averages cosine similarity over all class-name pairs *including across the base/novel split*. But the geometric driver of mis-calibration is presumably the **inter-class** similarity within each split, or the cross-split similarity, not a single grand mean. A decomposition τ_txt = (τ_bb, τ_nn, τ_bn) would be far more informative and would let RQ2 distinguish "fine-grainedness" from "base-novel embedding mismatch", which are the two competing causal stories.
- W5 [§3.3 C4 DAC]: DAC is described as having a "sample-wise temperature" but the formula T_i = T · g(d_txt(...)) does not specify g. As written this could be linear, exponential, or piecewise. For the RQ3 comparison to be fair, the exact g used must match Wang24's published recipe (or be ablated).
- W6 [§4.4 accuracy gate]: The "|Δacc| ≤ 0.5 pp" gate for a calibrator to be "valid" is reasonable but the threshold is again unjustified, and what happens when DAC/CAC fail it is not specified. Does the row drop out of Table 2? Is it marked? This affects the Pareto analysis directly.
- W7 [§5.3 RQ4 caveat]: After acknowledging that TS/VS are monotone and cannot move AURC, the paper still runs a mixed-effects regression of AURC on ECE across all calibrators (§4.6). The TS rows will pull the slope toward 0 mechanically. The regression should be specified to either (a) include a calibrator-class fixed effect, or (b) be restricted to the non-monotone subset. As written it conflates two regimes.
- W8 [§4.1 confounder controls]: "LAION-400M metadata class-name token frequency" is listed as a control — but the backbones are LAION-2B-trained (`laion2b_s34b_b88k` / `laion2b_s32b_b82k`), not LAION-400M. The confounder proxy is then mis-specified. Also: the control is described as reported "where available", with no commitment to a specific subset.

### Detailed Comments
- [§3.5, line "H1.RQ1: ΔECE_fg > 1.5 · ΔECE_coarse"]: state the alternative explicitly as a ratio test on per-dataset means, give the estimator (geometric mean of ratios? mean of differences of logs?), and report the pre-registered point estimate range.
- [§4.5]: `torch.backends.cudnn.deterministic = True` is set, but `cudnn.benchmark` is not mentioned. For full reproducibility set `benchmark=False` and `use_deterministic_algorithms(True)` with the appropriate `CUBLAS_WORKSPACE_CONFIG`.
- [§4.6 RQ3 row]: Friedman + Nemenyi is the right omnibus + post-hoc, but the paired-bootstrap CI for the 30% reduction criterion is a *different* hypothesis with no shared null. Either treat them as separate hypotheses (each with its own Holm column) or pick one as primary.
- [§5.3 RQ2 toy sanity]: PCA-whitening dropped ECE 0.105 → 0.095 on 128 CIFAR images. This is well within the binning noise of 15-bin equal-mass ECE at n=128 (the per-bin variance is ~√(1/8) ≈ 0.35). I would not even cite this as suggestive evidence.
- [Table 5 / §5.4]: pre-registration is good, but the document does not explain *what happens* if 2/4 RQs fail. Does the paper still claim a contribution? A decision tree of "if RQ_i succeeds/fails then claim X" would make the framework actually falsifiable.

### Questions for Authors
- Q1: Can you derive or cite the 1.5× threshold for RQ1?
- Q2: Why not decompose τ_txt into (base-base, novel-novel, base-novel) components, which would directly disambiguate two of your hypotheses?
- Q3: For RQ4, is the mixed-effects slope reported across all calibrators or restricted to non-monotone ones? If across all, how do you correct for the TS/VS zero-slope contamination?
- Q4: What is the pre-registered behaviour if the accuracy gate fails? Drop the row, report it with an asterisk, or re-fit hyperparameters?

---

## Reviewer 2 — Novelty / Positioning

**Score (1-10)**: 4
**Recommendation**: Weak Reject
**Confidence (1-5)**: 4

### Summary (저자가 무엇을 했다고 이해했는지)
The authors do **not** propose a new calibrator, a new architecture, a new loss, or a new theoretical bound. They propose to put existing components (TS, Vector Scaling, Dirichlet, DAC, CAC, Histogram Binning) on a single evaluation grid, introduce a single scalar (τ_txt = mean pairwise cosine similarity of class-name embeddings) as an explanatory variable, and run pre-registered statistical tests. The pitch is that the **framework is the contribution**: protocol, hypotheses, statistical procedure, and code.

### Strengths
- S1: The honest scope note (§1, §4.7) about pending full sweep is refreshing and avoids the worst form of dishonesty (inventing numbers).
- S2: The 7-way Pareto comparison (§3.3) does provide a value-add over the pairwise comparisons that DAC and CAC papers report; a single-grid head-to-head with controlled hyperparameter selection has community value.
- S3: The framing of "fine-grained × open-vocabulary" as a distinct measurement target (§1) is a clean positioning statement that the prior literature has not articulated this sharply.

### Weaknesses
- W1 [§1, §2.5, contribution C1-C4]: **The novelty bar is the central concern.** "We do not propose a new calibrator" is stated explicitly (§3.2). The four contributions are: (C1) a paired statistical test, (C2) a scalar summary statistic + intervention, (C3) a 7-way comparison, (C4) a mixed-effects regression. None of these is a *method*; all are *measurements*. A measurement paper can be publishable, but the bar is then "the measurement reveals something the community did not know". The expected results (§5, §6.1) are all stated in the direction the authors already believe — "DAC and CAC are expected to dominate", "τ_txt expected to correlate positively". If the framework merely confirms what DAC/CAC/DOR already implied, the contribution collapses to "we restated published results on a tidier grid".
- W2 [§2.3, vs. DAC/CAC/DOR]: The claim that prior work lacks "a statistical test isolating the fine-grained regime, no interventional test of cause, no head-to-head Pareto" is true in the strict sense but **understates what is in those papers**. DAC's Table 3 already separates fine-grained from coarse datasets; CAC's Table 2 already shows the base/novel asymmetry; DOR explicitly discusses sign asymmetry. The delta is "with statistical tests and a Pareto plot" — which is incremental, not a positioning statement that justifies a top-tier venue.
- W3 [§3.1 τ_txt]: The τ_txt scalar is one line of code and is **dominated by what is already known**. Mean pairwise cosine of class names correlates almost mechanically with semantic granularity, which correlates with accuracy, which correlates with ECE. A simple partial-correlation conditioning on accuracy could trivially explain away the effect (and the authors do mention this in §3.5 row 2, "partial Spearman conditioning on accuracy"), but if it does explain it away, RQ2 collapses. The paper does not pre-register what conclusion is drawn in that case.
- W4 [§5.3 RQ2 toy]: PCA-whitening as a "causal" intervention is **mis-named**. Whitening is a deterministic transformation of the text embeddings; calling it an "intervention" in the Pearl/Rubin sense requires that whitening be plausibly *the* operation that distinguishes a more-calibrated counterfactual world. This is not argued. At best whitening shows that *some* function of the embedding spectrum predicts ECE — a much weaker claim than causality.
- W5 [§3.2, §6.4]: The future-work section ("a new conditional calibrator fit on the τ_txt axis") admits that the actual novel method **has not been built**. Top venues will read this as "Stage 1 of a two-paper plan, only Stage 1 submitted". The right venue for a framework + pre-registered-but-unrun protocol is a workshop or a reproducibility track, not a main conference.
- W6 [§2.5 table]: The "Closest prior" column is fair, but the "What is missing" column is in every row an *evaluative* gap, not a *conceptual* gap. Evaluative gaps are filled by running experiments, not by a paper.
- W7 [§1 contributions C1]: The 1.5× ratio claim, if confirmed, is a quantitative refinement of the qualitative observation already in DAC §4.2 ("ECE gap is more severe on fine-grained datasets"). The contribution is then the *number*, not the *finding*.

### Detailed Comments
- [§1 abstract]: The line "The contribution is therefore a precise, falsifiable evaluation framework and an explanatory hypothesis — not a single new calibrator" should be the first sentence of the paper, not buried in the abstract. Reviewers will calibrate expectations differently if this is stated up front.
- [§2.1-2.4]: Related work is thorough and well-organised; cite Murphy "Machine Learning" Ch. 14 or Vovk's conformal-prediction-as-coverage book chapter if available, for the conformal background — currently §2.4 cites only the most recent papers.
- [§5 expected magnitudes]: Using prior tables to anchor expected magnitudes is honest but is also a **double bind**: if the full sweep reproduces them, the contribution is "we re-ran"; if it doesn't, the anchors become a story of "our protocol gives different numbers, here's why" — which is itself a contribution but is not the contribution currently claimed.
- [§6.4]: Three of the four future-work bullets read as Stage 2 of this work. Pull at least one (e.g. the theoretical bound on ECE from the text-embedding spectrum) into the current paper to raise the novelty floor.

### Questions for Authors
- Q1: If RQ2a's partial correlation conditioning on accuracy drops τ_txt's effect to ρ < 0.3, do you still claim a contribution? What is the contribution then?
- Q2: What is the smallest **methodological** novelty (not measurement) in this paper that you would defend in a rebuttal? Please name one.
- Q3: Why not pair this framework with the new conditional calibrator (your own §6.4 bullet 1) and submit the joint package?
- Q4: How would you respond to the criticism that this is a "DAC reproducibility study with a Pareto plot"?

---

## Reviewer 3 — Empirical Rigor

**Score (1-10)**: 3
**Recommendation**: Reject
**Confidence (1-5)**: 5

### Summary (저자가 무엇을 했다고 이해했는지)
A protocol for evaluating CLIP calibration on fine-grained × open-vocabulary deployment, with four pre-registered RQs and matched statistical tests. The protocol is **specified but not executed**: §4.7 states explicitly that the full sweep (11 datasets × 2 backbones × 3 seeds × 7 calibrators) was *not* run, and §1, §5, §6.1, and §7 all describe results as "expected magnitudes anchored on DAC/CAC/DOR tables (pending full sweep)". The only numerical evidence is a single toy CIFAR-10 sanity check (§4.7, §5.3 RQ2) at n=128, ViT-B/32, single seed, on a dataset that is *not* even in the experimental matrix.

### Strengths
- S1: The protocol §4.1-§4.6 is genuinely detailed and would, *if executed*, produce a publishable measurement.
- S2: Explicit honest disclosure that the sweep was not run (§1 "honest scope note", §4.7, §5 intro paragraph). This is more transparent than the typical pattern of fabricating placeholder tables.
- S3: The code-path verification on a toy slice (§4.7) is a real (if minimal) artefact.

### Weaknesses
- W1 [§5 entire, §4.7, abstract]: **There are no results.** This is the central, fatal weakness. Eleven datasets × 2 backbones × 3 seeds × 7 calibrators = 462 main-table cells. Number actually populated by the authors: 0. Cells where a numerical value is shown anywhere in §5: a handful of "≈" entries (§5.3 RQ3 table), which are explicitly described as anchored on Wang24/Lv25/Wang24-DOR rather than measured. The CIFAR-10 toy sanity (n=128, ViT-B/32) is not on any of the 11 target datasets and uses a backbone not in the protocol. **An empirical paper with no measurements cannot be accepted** at a main-track venue, regardless of how clean the protocol is.
- W2 [§5.3 RQ2 toy sanity, ECE 0.105 → 0.095]: The toy numbers cited (PCA-whitening ECE drop of 0.010, orthogonalisation ECE increase to 0.341) are at n=128 with 15-bin ECE, where the **per-bin sample count is ~8** and the ECE estimator's standard error is on the order of 0.03-0.05. The "small but correctly-signed effect" is well within sampling noise and the orthogonalisation increase to 0.341 is more plausibly an artefact of empty bins than of geometry destruction. The paper acknowledges "not statistically meaningful at n=128" (§5.3) but still uses the directional sign as supporting evidence in §6.1 and §7 — this is overclaiming on negligible evidence.
- W3 [§4.7 dataset blocker]: "CUB-200, iNat subset, ImageNet-1k val, and SUN397 require manual download outside torchvision's automatic fetcher" is true but **not a publishable excuse**. These are all standard public datasets with off-the-shelf loaders in OpenCLIP, DASSL, and the CoOp repositories. The barrier is roughly one engineering day, not a fundamental obstacle. If the authors could not stage four datasets, the work is simply not ready for submission.
- W4 [§4.3 baselines]: The recipe-level baselines B0-B6 are listed but **hyperparameter ranges are not specified**. For RQ3 to be a fair Pareto comparison, each baseline must be tuned with the same compute budget on the same base-validation split. The protocol says "L-BFGS minimisation of NLL" (§3.3) for *all* calibrators, but DAC and CAC in their original papers used different selection criteria. Re-fitting them under a TS-style criterion may handicap them artificially; not re-fitting them creates an unfair advantage. The paper does not say which way it goes.
- W5 [§4.2 backbones, §4.7]: Two backbones (ViT-B/16, ViT-L/14) on one CLIP family (OpenCLIP-LAION-2B) is a narrow setting. SigLIP, EVA-CLIP, and OpenAI CLIP are all standard alternatives that the protocol could include at marginal cost. Without at least 3 backbone families the generalisation claims of §1, §6.1 are not supportable.
- W6 [§4.6 statistical power]: With **n=11 datasets and 3 seeds**, the Friedman + Nemenyi test for RQ3 has ~30 effective samples per calibrator across a 7-arm comparison. The Nemenyi critical difference at α=0.05 for k=7, N=11 is CD ≈ 2.65 ranks — i.e., calibrators must differ by more than 2.65 average ranks to be distinguishable. The expected effect sizes from DAC/CAC tables suggest that 4-5 of the 7 calibrators will be inside the CD band, and the test will be inconclusive between the top candidates. The paper does not report a power analysis and does not pre-register what is done if Nemenyi is inconclusive.
- W7 [§4.6 RQ4 mixed-effects]: A mixed-effects regression with **11 dataset-level random intercepts** and a fixed effect on ECE may not have enough between-dataset variability to identify the random-intercept variance. The model may fail to converge or collapse to a fixed-effects-only regression. The paper does not specify the fall-back. Also: clustering at the (dataset × backbone × calibrator) level introduces nested non-independence the model does not address.
- W8 [§4.3 vs §3.3]: DAC's original implementation uses sample-level text distances to *all* base classes, with a softmax temperature on the distance. The §3.3 C4 description shows distance to the *closest* base text embedding, which is a different (and weaker) formulation. If this is the DAC variant actually implemented, the comparison to published DAC numbers in §5.3 RQ3 is **not apples-to-apples** and the "DAC expected to dominate" claim is on different footing than the anchored tables.
- W9 [§5 anchoring strategy]: Using DAC/CAC/DOR tables as the source for "expected magnitudes" creates a circularity: if the eventual sweep matches them, the paper has confirmed what is already published; if it doesn't, the paper has refuted the anchors and the entire §5/§6 narrative needs to be re-written. Either outcome implies that the current draft is **a placeholder for a real paper, not a paper**.
- W10 [§4.5 reproducibility]: `cudnn.deterministic=True` is set but `cudnn.benchmark` and the deterministic CUBLAS workspace config are unmentioned (see also R1 W8). Critical for ECE measurements at the 0.01 level.

### Detailed Comments
- [§5.1 Table 2 placeholder]: The literal text "(pending full sweep)" appears as a Table 2 entry. No conference reviewer will accept a placeholder for the main result table.
- [§5.3 RQ1 expected magnitude table]: The numbers (≈+6.5 pp fine-grained vs. ≈+2.0 pp coarse) are exactly what DAC and DOR report. If the sweep reproduces these, RQ1 has confirmed published findings with a statistical test attached. If RQ1 is the *only* contribution that survives, the paper is a statistical confirmation of prior tables.
- [§5.3 RQ3 expected results table]: DAC and CAC numbers (≈0.045 and ≈0.043 union ECE) are reported to three decimal places. With only 3 seeds and 11 datasets the standard error of union ECE on this scale is approximately ±0.005 — i.e., the third decimal is not resolvable. The presentation overstates precision.
- [§6.1 four claims]: Three of the four "concrete claims" begin with "If the expected results hold". Conferences do not award contributions on a conditional.
- [§4.7]: The infrastructure being "in place" with only "dataset staging and the swap from zero-shot to fine-tuned checkpoints remaining" is, in practice, the bulk of the work. Authors should stage and run before resubmission.

### Questions for Authors
- Q1: How long would it take to run the full sweep on the staged datasets? If the answer is "<2 weeks of one-GPU-time", why was it not done before submission?
- Q2: What is the exact DAC implementation used in C4 — closest-base-distance or softmax-over-all-distances? Provide the formula and a unit test.
- Q3: For the Nemenyi CD with k=7, N=11, will any pairwise comparisons survive? If not, what is the fall-back?
- Q4: Why are zero-shot baselines (which are the realistic fallback when fine-tuning checkpoints are unavailable, §4.8) not the primary protocol with fine-tuning as a robustness check, given that you have not run the fine-tuning sweep?
- Q5: Will you commit to running the full sweep in the rebuttal period, or is this submission targeted at a venue that accepts framework-only papers?

---

## Meta-Review

**Aggregate Recommendation**: **Reject** (rebuttal-resubmit-able after a full sweep)

Scores: R1=5, R2=4, R3=3. Mean ≈ 4.0. Two of three reviewers recommend reject or weak reject; the third sits on borderline. None recommends accept.

### 합의된 강점
- Protocol-level rigor and pre-registration of statistical tests is unusually mature for this sub-field (R1 S1, S3; R2 S2; R3 S1).
- Honest scope note about pending sweep (R2 S1, R3 S2) — the paper does not fabricate numbers, which is the single most important integrity property.
- The 7-way head-to-head Pareto framing has community value (R2 S2; R3 S1).
- AURC monotonicity caveat (R1 S4) is technically correct and underappreciated in the prior VLM calibration literature.

### 합의된 약점 (저자가 우선 해결해야 할 것)
1. **No measured results.** R3 W1, R2 W1, R1 (implicit in the framing of every "Status: Pending"). The full sweep (11 × 2 × 3 × 7 = 462 cells) is not run; the only numerical artefact is a 128-image CIFAR-10 toy. **An empirical paper with zero measurements on the actual experimental matrix cannot be accepted at a top-tier venue.** This is the dominant rejection driver.
2. **Anchoring on prior tables creates circularity.** R3 W9, R2 W7, R1 (implicit). If the sweep reproduces DAC/CAC/DOR, the contribution collapses to "we restated published findings with a paired test". If it does not, the entire §5/§6 narrative must be re-written. Neither outcome supports the current draft.
3. **Novelty is measurement, not method.** R2 W1-W7, R1 (acknowledges this is by design). The paper explicitly does not propose a new calibrator. Top venues will read this as Stage 1 of a two-paper plan.
4. **Statistical-power concerns.** R1 W2, R3 W6, R3 W7. With n=11 datasets, the Spearman, Nemenyi CD, and mixed-effects estimators are all underpowered for the effect sizes anchored from DAC/CAC. A power analysis is missing.
5. **τ_txt is under-specified and possibly confounded.** R1 W4, R2 W3. A grand-mean cosine similarity collapses three distinct regimes (base-base, novel-novel, base-novel) into one scalar and is partly redundant with accuracy.
6. **Calibrator implementations may not be apples-to-apples.** R1 W5, R3 W4, R3 W8. DAC's g(·) is unspecified; whether each baseline is re-fit under the unified L-BFGS+base-NLL criterion or used as-published is unclear.
7. **Arbitrary thresholds** (1.5×, ρ ≥ 0.55, 30% reduction, |Δacc| ≤ 0.5 pp). R1 W1, W2, W6. Pre-registration is good; arbitrary pre-registration is not.

### 분기된 의견
- **R1 (Methodology, score 5)** sees the protocol as fundamentally sound but under-justified at the threshold level. R1 would accept a revision that adds power analyses, justifies thresholds, and decomposes τ_txt.
- **R2 (Novelty, score 4)** does not believe the framework-only contribution is enough for a top venue *even with full results*, and recommends pairing this with the future-work "new conditional calibrator" (§6.4 bullet 1) before resubmission. R2's reject is driven by contribution shape, not execution.
- **R3 (Empirical Rigor, score 3)** does not engage with the novelty argument at all — for R3, the question of "is this a method or a measurement paper" is moot until measurements exist. R3 would soften toward a Borderline/Weak Accept *if* the full sweep is run, the toy CIFAR sanity is removed from claim-supporting sections, and the implementation details (DAC g, Nemenyi fallback, mixed-effects nesting) are pinned down.
- **Reconciliation:** R3's "no results" critique is necessary to address before R1's "thresholds and power" or R2's "novelty shape" critiques become tractable. The execution gap dominates.

### Author Action Plan (리부탈 단계 가이드)

**Immediately answerable (rebuttal text only):**
- Justify the 1.5×, ρ ≥ 0.55, 30%, 0.5 pp thresholds with either a power calculation or a citation. (R1 W1, W2, W6)
- Specify DAC's g(·), the exact selection criterion for each baseline, and pin down whether re-fitting was done. (R1 W5, R3 W4, R3 W8)
- Clarify whether RQ4's mixed-effects regression includes a calibrator-class fixed effect or is restricted to non-monotone calibrators. (R1 W7)
- Replace "LAION-400M token frequency" with "LAION-2B token frequency" (or remove the control). (R1 W8)
- State pre-registered behaviour if accuracy gate fails and if Nemenyi CD is inconclusive. (R1 W6, R3 W6)
- State pre-registered behaviour if τ_txt's partial correlation conditioning on accuracy drops below 0.3. (R2 Q1)

**Additional experiments required (the rebuttal will not be credible without these):**
- **Run the full sweep.** 11 datasets × 2 backbones × 3 seeds × 7 calibrators. Stage CUB-200, iNat subset, ImageNet-1k val, SUN397 (one engineering day per R3 W3). Without this, no reviewer will move. (R3 W1, R2 W1, R1 implicit)
- **Replace the CIFAR-10 toy** with a real subset (e.g. CUB-200 base/novel at n=full) for §5.3 RQ2. (R3 W2)
- **Decompose τ_txt** into (τ_bb, τ_nn, τ_bn) and re-run RQ2 correlations and partial correlations. (R1 W4, R2 W3)
- **Add at least one additional backbone family** (SigLIP or OpenAI CLIP) for external validity. (R3 W5)
- **Run a power analysis** for Spearman, Nemenyi, and the mixed-effects fixed-effect estimator at the anchored effect sizes. (R1 W2, R3 W6, R3 W7)

**Body-text revisions only:**
- Remove all "(pending full sweep)" placeholders from §5; either populate or restructure §5 around the toy + protocol.
- Soften §6.1's four "if the expected results hold" claims; they cannot stand without measurements. (R3 detailed comment)
- Pull at least one piece of future-work (e.g. theory of τ_txt) into the main paper to raise the novelty floor. (R2 W5, detailed comment on §6.4)
- Move the "honest scope note" to the first paragraph of §1, not the end of §1. (R2 detailed comment on abstract)

### Decision Rationale

The paper has a real virtue — honest disclosure that the full sweep was not run — and a real fatal flaw following directly from that virtue: there are no measurements on the experimental matrix, only expected magnitudes anchored on prior published tables, plus a 128-image CIFAR-10 toy that is on a backbone (ViT-B/32 OpenAI) not even in the protocol. The protocol §3-§4 is mature enough that a competent author could run the sweep in 1-2 weeks of single-GPU time and submit a much stronger paper. As it stands, this is best characterised as a high-quality experiment design document, not an empirical contribution.

The novelty question (R2) is secondary but reinforces the reject: even with the sweep completed and the expected magnitudes reproduced, the contribution would be "we ran a tidier version of DAC's evaluation with paired tests and a Pareto plot" — incremental for a top venue. The recommended path is (a) run the sweep, (b) decompose τ_txt and add the third-bullet theoretical contribution from §6.4, (c) tighten the implementation/threshold pre-registration per R1 W1-W8, and (d) resubmit. With those four steps the paper is a credible Borderline/Weak Accept; without them it is a Reject driven entirely by R3 W1.

**Final aggregate: Reject, resubmit-after-execution.**
