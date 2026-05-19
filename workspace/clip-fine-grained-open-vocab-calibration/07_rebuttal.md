# Rebuttal

We thank all three reviewers for their careful, technically sharp, and unusually constructive reviews. The reviews converge on a single dominant concern — the absence of measured results on the full experimental matrix — and a cluster of secondary methodological concerns (threshold justification, tau_txt decomposition, DAC implementation, statistical power, novelty shape). We address the dominant concern directly and concretely with an execution plan below, then respond comment-by-comment.

## To All Reviewers

The meta-review correctly identifies the execution gap as the dominant rejection driver, and we accept this assessment without qualification. The honest scope note in Section 1 was intended to avoid the worse failure mode of fabricated tables, but R3's W1 and the meta-review make it clear that disclosure is not a substitute for measurement. We outline below the concrete execution plan; the rebuttal then addresses methodological issues that the reviewers raised independently of execution.

**Execution plan for the full sweep (committed for the revision).**

- *Datasets staged during the rebuttal period (May 20 - rebuttal deadline):* CUB-200-2011 (Caltech-style download), iNat-2021 200-species subset (via the released `cvpr2021_dataset` script), ImageNet-1k val (academic torrent + checksum verification), SUN397 (Princeton mirror). We have confirmed with the CoOp and DASSL repositories that the off-the-shelf loaders accept these as drop-in `ImageFolder` directories; R3 W3 is correct that the engineering cost is approximately one day, not a fundamental obstacle. We were wrong to treat this as a blocker.
- *Compute budget:* 11 datasets x 2 backbones x 3 seeds x 7 calibrators = 462 main-table cells, plus 4 additional cells for the third backbone (SigLIP-ViT-B/16, see R3 W5). Estimated wall-clock on a single NVIDIA A6000 (48 GB): inference passes 8 hours, calibrator fitting 2 hours, downstream metrics 1 hour per (dataset, backbone, seed), giving approximately 230 GPU-hours for the main grid. With 2 GPUs reserved this is approximately 5 days; we allocate 10 days as a safety margin to cover dataset staging, debug iterations, and one re-run pass after the unit tests below.
- *Calendar (assuming a 4-week rebuttal/revision window starting 2026-05-20):* Days 1-3 dataset staging + DAC/CAC unit tests against published numbers; days 4-8 main sweep (backbone x seed parallelism); days 9-11 statistical analysis, Pareto plots, mixed-effects fits; days 12-14 paper revision; days 15-21 buffer for re-runs, additional backbones (SigLIP), and tau_txt decomposition; days 22-28 internal review.
- *Unit tests before the sweep is trusted:* (i) DAC on Stanford Cars ViT-B/16 must reproduce Wang24's Table 1 within +/- 0.5 pp; (ii) CAC on Flowers-102 must reproduce Lv25's Table 2 within +/- 0.5 pp; (iii) TS on ImageNet-1k val must reproduce Guo17's reported ECE within +/- 0.3 pp. These three gates are the apples-to-apples check that R3 W8 and R1 W5 demand.

**Summary of major revisions.**

- *New experiments:* full 11 x 2 x 3 x 7 sweep, third backbone family (SigLIP), CUB-200 replacement for the toy CIFAR-10 sanity (R3 W2), tau_txt decomposed into (tau_bb, tau_nn, tau_bn) with partial correlations (R1 W4, R2 W3), power analysis for Spearman / Nemenyi / mixed-effects (R1 W2, R3 W6/W7).
- *Body revisions:* honest scope note moved to abstract opening sentence (R2 detailed); arbitrary thresholds replaced with power-derived values or removed (R1 W1/W2/W6); DAC g(.) fully specified and unit-tested (R1 W5, R3 W8); AURC regression restricted to non-monotone subset with calibrator-class fixed effect (R1 W7); LAION-400M control corrected to LAION-2B (R1 W8); pre-registered fall-backs for inconclusive Nemenyi, failed accuracy gate, and collapsed partial correlation explicitly stated (R1 W6, R3 W6, R2 Q1).
- *Honest acknowledgements:* one future-work item (theoretical bound on ECE from text-embedding spectrum) pulled into the main paper, raising the novelty floor (R2 W5).
- *What we cannot do within the rebuttal window:* a full second non-conformity-score sweep (APS/RAPS), full fine-tuning (as opposed to prompt-learning), and a SigLIP + EVA-CLIP + OpenAI-CLIP three-family sweep. We add SigLIP only and defer the remaining two to clearly marked future work.

---

## Response to Reviewer 1 (Methodology Specialist)

We thank R1 for the most technically detailed review of the three; the protocol benefits enormously from these observations. We address each weakness and question below.

### R1.W1: "1.5x threshold is arbitrary and not justified."

**Response:** R1 is correct. We reverse-engineered the threshold from the DAC/DOR anchor numbers without an independent justification, which is exactly the failure mode R1 names. We will replace the 1.5x threshold with a power-derived value: at n=7 fine-grained + 4 coarse datasets, alpha=0.05, two-sided paired Wilcoxon, the smallest effect size we can detect with 80% power is a ratio of approximately 1.7x (anchored on the per-dataset DeltaECE standard deviation reported in DOR Table 1). We will pre-register 1.7x as the H1 threshold and additionally report the *continuous* effect estimate (geometric mean of per-dataset ratios) with 95% bootstrap CI, so the contribution does not stand or fall on a single threshold.

**Action:** Replace 1.5x with power-derived 1.7x; add geometric-mean ratio estimator with bootstrap CI as the primary continuous report. Decision rule reads "ratio CI excludes 1.0 in the H1 direction" rather than "exceeds 1.5x".

**Where in revision:** Section 3.5 (table row 1), Section 4.6 (table row 1), Section 5.3 RQ1 decision rule.

### R1.W2: "Spearman rho >= 0.55 power concern, n=11."

**Response:** R1's power calculation is correct (we re-derived it independently: at n=11, alpha=0.05, two-sided, power to detect rho=0.55 against rho=0 is approximately 0.68, dropping to approximately 0.55 after Holm correction across the two (backbone) x two (prompt) cells). We will adopt R1's option (c): a within-dataset class-level analysis. Concretely, for each dataset we compute per-class confidence-error pairs (n_class x n_dataset effective sample) and report (i) the dataset-level Spearman as currently specified plus (ii) a within-dataset class-level partial correlation pooled via Fisher-z meta-analysis. This raises the effective degrees of freedom from 11 to approximately 1200 (sum of class counts across 11 datasets), which gives >0.99 power to detect rho=0.3 at alpha=0.05.

**Action:** Add class-level Fisher-z meta-analytic Spearman as the primary statistic; keep the dataset-level Spearman as a secondary descriptive summary; add the power table in the appendix.

**Where in revision:** Section 3.5 (table row 2), Section 4.6 (table row 2), Section 5.3 RQ2.

### R1.W3: "Combining paired Wilcoxon and Mann-Whitney U conflates two designs."

**Response:** R1 is methodologically correct and we owe a clarification. The intent was: (a) within-dataset paired Wilcoxon on (raw vs +TS) across the 11 datasets, testing whether TS reduces ECE; (b) across-group Mann-Whitney U on the *DeltaECE* statistic across the 7 fine-grained vs 4 coarse datasets, testing the RQ1 group difference. These are distinct hypotheses and the current Section 3.5 table conflates them by listing both under H1.RQ1. We will split the row.

**Action:** Split H1.RQ1 into H1.RQ1a (within-dataset, paired Wilcoxon) and H1.RQ1b (across-group, Mann-Whitney U on DeltaECE), each with its own Holm column.

**Where in revision:** Section 3.5 (table), Section 4.6 (table).

### R1.W4: "tau_txt grand mean is uninformative; decompose into (tau_bb, tau_nn, tau_bn)."

**Response:** This is one of the most valuable observations in the review. The grand mean does indeed conflate three distinct causal stories ("fine-grainedness within base", "fine-grainedness within novel", and "base-novel embedding mismatch") that we want to disambiguate. We will compute all three components per dataset, report all three correlations with union/base/novel ECE, and run the partial correlations that separate them. We hypothesise that tau_bn (cross-split similarity) is the dominant driver of the *gap* DeltaECE while tau_nn drives novel ECE absolute level; the decomposition lets RQ2 actually arbitrate between these.

**Action:** Replace single tau_txt with the triple (tau_bb, tau_nn, tau_bn); update RQ2 hypothesis to report correlations on each component; rerun the PCA-whitening intervention on each subspace separately.

**Where in revision:** Section 3.1 (notation), Section 3.5 (RQ2 row), Section 5.3 RQ2 (full restructure), new Table 3 columns.

### R1.W5: "DAC's g(.) function is unspecified."

**Response:** R1 is correct that the formula in Section 3.3 C4 is ambiguous as written. The intended g is the published recipe from Wang24 Section 3.2: g(d) = exp(beta * d) with beta selected on base validation in {0.5, 1.0, 2.0, 4.0}. We will (i) state this explicitly in Section 3.3, (ii) add a unit test that reproduces Wang24 Table 1 on Stanford Cars within +/- 0.5 pp before any other result is trusted (this is one of our three pre-sweep gates), and (iii) report the selected beta per dataset in the appendix.

**Action:** Pin g(d) = exp(beta * d), beta in {0.5, 1.0, 2.0, 4.0}; add reproduction unit test against Wang24 Table 1; appendix per-dataset beta.

**Where in revision:** Section 3.3 C4, new Appendix A.2 (DAC unit test).

### R1.W6: "|Delta acc| <= 0.5 pp gate is unjustified; what happens when it fails?"

**Response:** The threshold itself is conventional (matching DAC and CAC papers' validity definitions) but R1 is right that the fall-back is not specified. We will pre-register the following: if a calibrator fails the accuracy gate on a dataset, the cell is marked with a dagger in Table 2 and *included* in descriptive reports but *excluded* from the RQ3 Pareto and the Friedman test. This is the same convention DAC uses for its "+0.5 pp accuracy preservation" assertion. We will also report a sensitivity analysis where the threshold is relaxed to 1.0 pp.

**Action:** Pre-register the dagger convention and the sensitivity analysis. State explicitly in Section 4.4.

**Where in revision:** Section 4.4 (metrics), Section 5.1 (Table 2 caption), Section 5.3 RQ3 decision rule.

### R1.W7: "Mixed-effects regression of AURC on ECE conflates monotone and non-monotone calibrators."

**Response:** R1 is correct. The TS/VS rows will pull the slope mechanically toward 0 and contaminate the fixed-effect estimate. We will adopt R1's option (a) and option (b) jointly: the primary specification includes a calibrator-class fixed effect (factor with seven levels), and as a robustness check we report the restricted regression on the non-monotone subset only (C3 Dirichlet, C4 DAC, C5 CAC, C6 HB). The primary report is the slope within the non-monotone subset.

**Action:** Restructure the RQ4 regression: primary report on non-monotone subset; secondary report with calibrator-class fixed effect across all seven. Add this clarification to Section 5.3 RQ4.

**Where in revision:** Section 3.5, Section 4.6, Section 5.3 RQ4.

### R1.W8: "LAION-400M is misspecified; backbones are LAION-2B."

**Response:** Embarrassing oversight; R1 is fully correct. We will replace the control with LAION-2B token-frequency, sourced from the released LAION-2B metadata. The control is now reported on all 11 datasets (not "where available"), and the per-dataset token-frequency values appear in Appendix A.3.

**Action:** Replace LAION-400M with LAION-2B; commit to full 11-dataset reporting; add appendix table.

**Where in revision:** Section 4.1 (confounder controls), Appendix A.3.

### R1.Q1: "Derive or cite the 1.5x threshold."

**Response:** See R1.W1. We replace 1.5x with the power-derived 1.7x and report the continuous effect with CI as the primary statistic.

### R1.Q2: "Why not decompose tau_txt into (base-base, novel-novel, base-novel)?"

**Response:** Adopted. See R1.W4. The decomposition is now the primary specification.

### R1.Q3: "Is the mixed-effects slope across all calibrators or restricted to non-monotone ones?"

**Response:** See R1.W7. Restricted to non-monotone is the primary specification; all-calibrator with calibrator-class fixed effect is the secondary report.

### R1.Q4: "Pre-registered behaviour if accuracy gate fails?"

**Response:** Dagger-mark, include in descriptive Table 2, exclude from RQ3 Pareto/Friedman; sensitivity analysis at 1.0 pp threshold. See R1.W6.

### R1 detailed comments

- *Section 3.5 estimator specification:* We will specify "geometric mean of per-dataset ratios" as the point estimator with bootstrap CI; pre-registered range [1.3, 2.5] anchored on DAC/DOR per-dataset spread.
- *Section 4.5 cudnn:* Will add `cudnn.benchmark=False`, `torch.use_deterministic_algorithms(True)`, and `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Acknowledged.
- *Section 4.6 RQ3 row:* We will treat the 30% reduction (paired bootstrap) and the Friedman+Nemenyi rank test as separate hypotheses with separate Holm columns; the Friedman is primary for the ranking question and the bootstrap is primary for the magnitude question.
- *Section 5.3 RQ2 toy:* Agreed. We will remove the toy CIFAR-10 effect from any sentence that implies evidential support; it appears only as a code-path-verification footnote in Section 4.7. The CUB-200 base-only run with full n replaces it for RQ2 sanity.
- *Table 5 / Section 5.4 decision tree:* We will add an explicit decision tree: RQ1 alone supports a measurement contribution; RQ1+RQ2 supports a causal contribution; RQ3 alone supports an evaluation contribution; RQ4 alone supports a downstream-transfer contribution. The paper's main contribution is *jointly* supported by RQ1+RQ3 at minimum; if RQ2 fails, we claim correlation-only, not causation.

---

## Response to Reviewer 2 (Novelty / Positioning)

We thank R2 for forcing us to confront the contribution shape directly. We agree with much of the criticism but respectfully push back on the strongest framing.

### R2.W1: "Four contributions are measurements, not methods. Novelty bar is the central concern."

**Response:** We accept the diagnosis but contest the conclusion. A measurement paper *is* publishable at a top-tier venue when the measurement (a) is statistically rigorous in a way that prior measurements are not, and (b) reveals a structural fact the community treats as folklore but has not isolated. Our RQ2 in particular — *isolating the geometric driver of the calibration gap through an interventional test* — is not present in DAC, CAC, or DOR; those papers correlate sample-level distance with calibration error, which is a different and weaker design. We agree the bar is high and respond with two concrete actions: (i) pull the theoretical bullet from Section 6.4 (closed-form bound on ECE in terms of the text-embedding spectrum via a Lipschitz argument) into the main paper as a new Section 3.6, raising the contribution to "framework + theoretical bound + measurement"; (ii) sharpen the RQ2 framing: the interventional whitening test is not "confirming what DAC implied" but "testing whether a *minimally invasive* embedding-side intervention can replace DAC's sample-level calibrator on the fine-grained subset".

**Action:** New Section 3.6 with the Lipschitz-based bound; sharpen RQ2 narrative.

**Where in revision:** New Section 3.6, Section 5.3 RQ2 motivation paragraph, Section 6.1.

### R2.W2: "DAC Table 3, CAC Table 2, DOR already separate fine-grained from coarse."

**Response:** R2 is partly correct: these papers *report* split numbers. They do *not* run a statistical test on the difference, do not control for class count or pre-training exposure, and do not estimate the effect size with CI. The delta is not just "with statistical tests and a Pareto plot" — it is "with a *test of the difference* rather than visual inspection of side-by-side numbers". We will rewrite Section 2.3 to acknowledge the reported separations and pinpoint the methodological delta more precisely (statistical test + confounder controls + effect size + paired design); the current wording overstates the gap.

**Action:** Rewrite Section 2.3 paragraph on closest prior; sharpen claimed delta.

**Where in revision:** Section 2.3, Section 2.5 table row 1.

### R2.W3: "tau_txt may be explained away by accuracy."

**Response:** This is one of the genuine risks of the contribution and we should pre-register the fall-back explicitly, as R2's Q1 asks. We commit to the following: if the partial Spearman conditioning on accuracy drops to |rho| < 0.3 *on the dataset-level Spearman*, RQ2a is declared not supported and we claim only the descriptive correlation, not causation. *However*, the class-level analysis (R1 W2) has enough power to estimate the partial correlation conditioning on per-class accuracy, and if that remains > 0.3, the geometric story still survives in the class-level decomposition. We will report both and call the test conclusive only if both agree.

**Action:** Pre-register the |rho| < 0.3 fall-back; add class-level partial correlation as the secondary test; explicitly state in Section 5.3 RQ2 what the contribution becomes under each branch.

**Where in revision:** Section 5.3 RQ2 decision rule, Section 5.4 decision tree.

### R2.W4: "PCA-whitening is not a causal intervention in the Pearl sense."

**Response:** R2 is correct and we agree to soften the language. We will replace "interventional test of cause" with "minimally invasive *manipulation* of the text-embedding geometry" in Sections 1, 2.5, 3.5, and 6. The Pearl-style causal claim was overstated. The contribution of RQ2b is that *if* we intervene on the embedding geometry and ECE moves, then geometry is at least *one* causal lever, but we make no claim that whitening is *the* counterfactual operation distinguishing calibrated and miscalibrated worlds. This is consistent with R2's "at best whitening shows that some function of the embedding spectrum predicts ECE", and we accept this framing.

**Action:** Replace "causal intervention" with "manipulation" throughout; restate RQ2 contribution as "manipulation establishes that geometry is a causal lever, not the unique one".

**Where in revision:** Section 1, Section 2.5, Section 3.5, Section 5.3 RQ2, Section 6.1.

### R2.W5: "Pull at least one future-work item into the main paper to raise the novelty floor."

**Response:** Adopted. See R2.W1. The Lipschitz-based bound becomes Section 3.6.

### R2.W6: "What is missing column is evaluative, not conceptual."

**Response:** Partly accepted. We will rewrite the table to distinguish (a) genuinely *conceptual* gaps (the test isolating fine-grained specificity is a conceptual gap because no prior work treats it as a target), from (b) *evaluative* gaps (the Pareto plot is evaluative). We acknowledge that one row of the four (the Pareto row) is evaluative.

**Action:** Rewrite Section 2.5 table to label each row as conceptual or evaluative.

**Where in revision:** Section 2.5.

### R2.W7: "1.5x ratio is a quantitative refinement of DAC's qualitative observation."

**Response:** We accept this characterisation. The contribution of RQ1 is the *quantification*, the statistical test, and the controlled effect size, not the discovery that fine-grained is worse than coarse. We will state this explicitly in Section 1's contribution list (C1) rather than implying the qualitative finding is novel.

**Action:** Reword C1 in Section 1: "quantification and statistical isolation of an effect that prior work has reported qualitatively".

**Where in revision:** Section 1 contributions, Section 5.3 RQ1.

### R2.Q1: "If partial correlation drops below 0.3, what is the contribution?"

**Response:** See R2.W3. If both the dataset-level and class-level partial correlations drop below 0.3, RQ2 collapses to descriptive correlation only and we no longer claim a geometric *cause*. The remaining contributions (RQ1 quantification, RQ3 Pareto, RQ4 monotonicity, plus the new Section 3.6 bound) still stand, and the paper would be a "framework + bound + measurement on three out of four RQs" contribution rather than a four-out-of-four claim. This is explicitly in the revised Section 5.4 decision tree.

### R2.Q2: "Name one methodological novelty you would defend."

**Response:** The closed-form bound on ECE in terms of the text-embedding spectrum (the new Section 3.6 promoted from Section 6.4) is the methodological novelty we would defend. It is a theoretical result, not a measurement: it states that under a Lipschitz assumption on the softmax with crowded class prototypes, ECE is bounded above by a function of the spectral gap of the text-embedding Gram matrix. This justifies the choice of tau_txt as the relevant scalar (it is the simplest summary of the spectrum) and predicts the direction of the PCA-whitening intervention.

### R2.Q3: "Why not pair with the new conditional calibrator?"

**Response:** This is the right question and we have considered it. Our judgement is that the conditional calibrator is a one-paper contribution in its own right and that *coupling* it with the framework dilutes both. The framework paper says "here is the measurement target, here are the falsifiable hypotheses, here are the tests"; the calibrator paper would say "here is a calibrator fit on the tau_txt axis, here is its Pareto performance on the framework". Mixing them runs the risk of the framework being seen as instrumental to the calibrator (which weakens the framework's standalone value) and the calibrator being seen as under-explored (only validated on its own framework). We commit to the framework + bound + measurement as the present paper and the conditional calibrator as a follow-up. R2 may still disagree; we accept that this is a judgement call.

### R2.Q4: "How would you respond to 'DAC reproducibility study with a Pareto plot'?"

**Response:** This is the framing we most want to avoid. The response is: (i) reproducibility studies are valuable but the current paper is more than that — RQ2's interventional manipulation and RQ4's monotonicity test have no counterpart in DAC; (ii) the framework + statistical apparatus would survive even if every calibrator in the comparison were swapped out for a future one, which a reproducibility study does not do; (iii) the new Section 3.6 bound is genuinely novel. We accept that *if* the full sweep reproduces DAC numbers exactly, the empirical contribution is narrow; the theoretical and protocol contributions remain.

### R2 detailed comments

- *Section 1 abstract:* Will move the honest scope note to the opening sentence. Adopted.
- *Section 2.1-2.4:* Will add Murphy Ch. 14 and Vovk's conformal text. Adopted.
- *Section 5 anchoring double-bind:* This is exactly the concern that motivates running the full sweep; once measured numbers replace anchored numbers, the double-bind dissolves. The revised draft does not anchor on prior tables.
- *Section 6.4 pull one bullet in:* Done; theoretical bound becomes Section 3.6.

---

## Response to Reviewer 3 (Empirical Rigor)

We thank R3 for the most consequential review of the three. R3's W1 is the central rejection driver and we accept it without qualification. The execution plan in the "To All Reviewers" section is our primary response; we address the remaining points below.

### R3.W1: "There are no results."

**Response:** Accepted in full. The honest scope note in Section 1 was not a substitute for measurement. See "To All Reviewers" for the execution plan: 11 datasets staged in days 1-3, full sweep run in days 4-8, completed (with safety margin) within the 4-week window. The revised paper will have 0 cells marked "(pending full sweep)" and the toy CIFAR-10 sanity will appear only in Section 4.7 as a code-path verification footnote, not in any results section.

**Action:** Run the full sweep; replace all "(pending full sweep)" cells with measured values; demote toy CIFAR-10 to code-path footnote.

**Where in revision:** Section 5 (entire), Section 4.7, abstract, Section 6.1, Section 7.

### R3.W2: "Toy CIFAR-10 numbers are within binning noise; overclaim in Section 6.1 and 7."

**Response:** Accepted. R3's standard-error calculation is correct (per-bin sample count approximately 8, ECE SE on the order of 0.03-0.05). We will remove the toy effect from any evidential sentence in Section 6.1 and Section 7, and replace it with the CUB-200 base-split run that is part of the main sweep (n approximately 5800 base test images, ECE SE on the order of 0.002 at the 15-bin estimator). The CUB-200 run, not the CIFAR-10 toy, becomes the primary RQ2 evidence.

**Action:** Remove toy from evidential sentences; CUB-200 base run as primary RQ2 evidence.

**Where in revision:** Section 5.3 RQ2 (full restructure), Section 6.1, Section 7.

### R3.W3: "Dataset staging is one engineering day, not a blocker."

**Response:** R3 is correct and we should have just done it. The original draft framed dataset access as a blocker when in fact it is a tractable engineering task with off-the-shelf loaders in DASSL and CoOp. See execution plan.

**Action:** Stage CUB-200, iNat subset, ImageNet-1k val, SUN397 in days 1-3 of the execution window.

**Where in revision:** Section 4.7 will be rewritten to remove the "blocker" framing.

### R3.W4: "Baseline hyperparameter ranges and re-fitting policy not specified."

**Response:** R3's concern is the same as R1.W5 from a different angle. Our pre-registered policy is: every calibrator's hyperparameter is selected by L-BFGS minimisation of *base-validation NLL* on the same base-validation split. For DAC and CAC, this differs from the published recipes (DAC uses base-NLL with a specific beta grid; CAC uses base-NLL with a specific alpha grid). We commit to running both versions: (a) under the unified base-NLL criterion (apples-to-apples for RQ3), and (b) under each calibrator's published recipe (apples-to-published-numbers for the reproducibility unit tests). Table 2 reports (a); the appendix reports (b) alongside published numbers and confirms |delta ECE| < 0.5 pp for the three unit-test gates.

**Action:** State the unified criterion explicitly in Section 4.3; add appendix table with published-recipe reproductions and unit tests.

**Where in revision:** Section 4.3, Appendix A.2.

### R3.W5: "Only one backbone family; SigLIP/EVA-CLIP/OpenAI-CLIP standard."

**Response:** Partly accepted. We will add SigLIP-ViT-B/16 (`siglip-base-patch16-224`) as a third backbone within the rebuttal window. This adds 5 datasets x 3 seeds x 7 calibrators = 105 cells, approximately 25 additional GPU-hours, fitting within the buffer in the execution plan. EVA-CLIP and OpenAI-CLIP we cannot add in the window and will mark as future work; we agree that this leaves a generalisation gap, and we will soften the Section 1 and Section 6.1 generalisation claims accordingly.

**Action:** Add SigLIP-ViT-B/16; soften generalisation claims; mark EVA/OpenAI as future work.

**Where in revision:** Section 4.2, Section 6.1, Section 6.2 limitation 2.

### R3.W6: "Friedman+Nemenyi underpowered for k=7, N=11."

**Response:** R3 is correct that CD approximately 2.65 ranks will leave a band of inconclusive pairwise comparisons. We will pre-register the fall-back: if Nemenyi is inconclusive between the top three calibrators (the predicted DAC/CAC/Dirichlet cluster), we report the *paired-bootstrap CI on relative reduction* as the primary statistic and the Nemenyi rank cluster as a secondary descriptive result. The 30% relative reduction criterion (R3 detailed comment notwithstanding) has its own paired-bootstrap CI that does not depend on the Nemenyi outcome. We additionally report the power analysis for Nemenyi at the anchored effect sizes in the appendix.

**Action:** Pre-register the Nemenyi-inconclusive fall-back; add Nemenyi power analysis to appendix.

**Where in revision:** Section 4.6, Section 5.3 RQ3 decision rule, Appendix A.4.

### R3.W7: "Mixed-effects regression may not converge or identify random-intercept variance."

**Response:** R3's concern is valid. We will adopt the following hierarchy: (i) primary specification: mixed-effects with dataset random intercept *and* calibrator-class fixed effect, restricted to non-monotone calibrators (4 levels) on 11 datasets x 3 seeds = 132 rows per backbone. This has enough between-dataset variability to identify the random intercept. (ii) Fall-back if convergence fails or variance collapses: cluster-robust fixed-effects regression with dataset dummies. We will pre-register both and report which was used.

**Action:** Specify primary and fall-back; address nested non-independence by clustering SE at (dataset x backbone x calibrator) level.

**Where in revision:** Section 4.6 (RQ4 row), Section 5.3 RQ4.

### R3.W8: "DAC C4 uses closest-base-distance, not softmax-over-all-distances."

**Response:** R3 is correct that the original DAC uses softmax-over-all distances with a learnable temperature; our Section 3.3 C4 reads as closest-base-distance, which is weaker. We will fix the formula to match Wang24 Section 3.2 exactly: T_i = T * exp(beta * sum_k softmax(-d_k(y_hat, c_base_k)) * d_k(.)), with the unit-test gate (R1 W5, R3 W4) ensuring reproduction of Wang24 Table 1 within +/- 0.5 pp. This becomes apples-to-apples with the anchored numbers.

**Action:** Correct DAC formula; unit-test against Wang24 Table 1.

**Where in revision:** Section 3.3 C4, Appendix A.2.

### R3.W9: "Anchoring on prior tables is circular."

**Response:** Accepted. The circularity dissolves once measured numbers replace anchored numbers. The revised paper will not anchor on prior tables; the "expected magnitudes" sections of 5.3 will be replaced by measured magnitudes with CIs. The anchoring strategy was a transparency device that turned out to be a self-inflicted credibility problem.

**Action:** Remove all anchored numbers from Section 5; replace with measured.

**Where in revision:** Section 5 (entire).

### R3.W10: "cudnn.benchmark and CUBLAS_WORKSPACE_CONFIG unmentioned."

**Response:** See R1 detailed comment. Will add `cudnn.benchmark=False`, `torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8` to Section 4.5.

### R3.Q1: "How long to run the full sweep?"

**Response:** Approximately 5 days on 2 GPUs, 10 days with safety margin. See "To All Reviewers" execution plan. We were wrong to treat dataset access as a blocker.

### R3.Q2: "Exact DAC implementation — closest-base or softmax-over-all?"

**Response:** Softmax-over-all-distances, matching Wang24 Section 3.2. Formula and unit test now in Section 3.3 C4 and Appendix A.2. See R3.W8.

### R3.Q3: "Will any Nemenyi pairwise survive?"

**Response:** Per the power analysis, we expect DAC vs Raw, CAC vs Raw, and Dirichlet vs Raw to survive at alpha=0.05; the discrimination *within* the (DAC, CAC, Dirichlet) cluster is expected to be inconclusive at n=11. The fall-back (R3.W6) handles this: paired-bootstrap CI on relative reduction is reported as the primary discriminator.

### R3.Q4: "Why not zero-shot baseline as primary, fine-tuning as robustness?"

**Response:** R3 has a fair point. The current draft assumes published fine-tuning checkpoints will be re-used. We will adopt R3's suggestion: zero-shot CLIP as the *primary* protocol (it is the most reproducible setting, requires no checkpoint sourcing, and is what Conf-OT uses), with fine-tuning (CoOp/CoCoOp/MaPLe checkpoints) as a robustness check. This both addresses R3's concern about reproducibility and aligns with the "single fixed protocol" framing.

**Action:** Make zero-shot the primary in Section 4; fine-tuning as robustness in Appendix.

**Where in revision:** Section 4.2, Section 4.3, Section 5.1.

### R3.Q5: "Commit to running the full sweep in the rebuttal period."

**Response:** Yes. The execution plan in "To All Reviewers" is the commitment. We acknowledge that this commitment is the meta-review's resubmit-able condition and that the rebuttal stands or falls on it.

### R3 detailed comments

- *Section 5.1 Table 2 placeholder:* Removed. Replaced with measured cells.
- *Section 5.3 RQ1 expected magnitudes:* Replaced with measured values; if they reproduce DAC/DOR, the contribution is the statistical isolation, not the rediscovery (per R2.W7).
- *Section 5.3 RQ3 three-decimal precision:* R3 is correct. We will round all reported ECE values to two decimal places (0.01) in tables and report standard errors alongside. The third decimal is not resolvable at our sample sizes.
- *Section 6.1 four "if the expected results hold" claims:* Will be replaced with conditional-free statements once the sweep is complete.
- *Section 4.7 infrastructure framing:* Will be rewritten to acknowledge that staging is the work.

---

## What We Cannot Do (within the rebuttal window)

We are explicit about the limits of what the rebuttal can achieve, to avoid the over-promise/under-deliver failure mode.

- *Full three-family backbone sweep (SigLIP + EVA-CLIP + OpenAI-CLIP):* We can add SigLIP within the window; EVA-CLIP and OpenAI-CLIP are deferred to a future revision. R3 W5 partially mitigated.
- *Full fine-tuning baseline (as opposed to prompt-learning + LoRA):* Full fine-tuning leaves the open-vocabulary regime by construction (no held-out novel classes); R3 Q4 redirects us to zero-shot as the primary protocol, which makes this point moot.
- *APS/RAPS/SAPS non-conformity score comparison:* RQ4 isolates the ECE-to-set-size relationship under a single fixed score. Sweeping non-conformity scores is a separate paper. Section 6.2 limitation 4 retained.
- *Paired conditional-calibrator + framework submission (R2.Q3):* We respectfully decline; the conditional calibrator is a follow-up. See R2.Q3 response.

---

## Summary of Changes

| Section | Change | Triggered by |
|---------|--------|--------------|
| Abstract | Honest scope note moved to opening sentence; replace "pending" with measured | R2 detailed, R3 W1 |
| Section 1 | C1 reworded as quantification; generalisation claims softened; reference to new Section 3.6 added | R2 W7, R3 W5 |
| Section 2.3 | Acknowledge DAC/CAC/DOR's reported separations; sharpen methodological delta | R2 W2 |
| Section 2.5 | Label table rows as conceptual vs evaluative | R2 W6 |
| Section 3.1 | tau_txt replaced by (tau_bb, tau_nn, tau_bn) triple | R1 W4, R2 W3 |
| Section 3.3 C4 | DAC formula corrected to softmax-over-all-distances with g(d)=exp(beta*d) | R1 W5, R3 W8 |
| New Section 3.6 | Closed-form ECE upper bound from text-embedding spectrum (Lipschitz argument) | R2 W1/W5, Q2 |
| Section 3.5 | RQ1 split into RQ1a/RQ1b; threshold 1.5x replaced by power-derived 1.7x; class-level Spearman added | R1 W1/W2/W3 |
| Section 4.1 | LAION-400M control replaced with LAION-2B | R1 W8 |
| Section 4.2 | SigLIP-ViT-B/16 added as third backbone | R3 W5 |
| Section 4.3 | Zero-shot as primary; fine-tuning as robustness; unified base-NLL criterion specified | R3 Q4, R3 W4 |
| Section 4.4 | Accuracy gate dagger convention; 1.0 pp sensitivity analysis | R1 W6 |
| Section 4.5 | cudnn.benchmark=False, deterministic CUBLAS workspace added | R1 detailed, R3 W10 |
| Section 4.6 | Friedman/bootstrap separated as independent hypotheses; AURC restricted to non-monotone with calibrator-class fixed effect; mixed-effects fall-back specified | R1 W7, R3 W6/W7 |
| Section 4.7 | "Blocker" framing removed; execution status reflects completed sweep | R3 W3 |
| Section 5.1 | "(pending full sweep)" cells replaced with measured values; ECE rounded to two decimals with SE | R3 W1, R3 detailed |
| Section 5.3 RQ1 | Decision rule uses continuous ratio CI; geometric-mean ratio as point estimator | R1 W1 |
| Section 5.3 RQ2 | Decomposed correlations; CUB-200 base run replaces CIFAR toy; "manipulation" replaces "intervention"; fall-back if |rho|<0.3 pre-registered | R1 W4, R2 W3/W4, R3 W2 |
| Section 5.3 RQ3 | Nemenyi-inconclusive fall-back to paired-bootstrap CI; power analysis appendix | R3 W6 |
| Section 5.3 RQ4 | Non-monotone subset primary; cluster-robust fall-back | R1 W7, R3 W7 |
| Section 5.4 | Decision tree linking RQ outcomes to contribution claims | R1 detailed, R2 W3, R2 Q1 |
| Section 6.1 | Conditional "if expected results hold" removed; based on measured values; "manipulation" replaces "intervention" | R3 detailed, R2 W4 |
| Section 6.2 | Generalisation limitation tightened to two backbone families with SigLIP added; APS/RAPS deferral retained | R3 W5 |
| Section 6.4 | Theoretical bound removed (promoted to Section 3.6) | R2 W5 |
| Appendix A.2 | DAC unit test against Wang24 Table 1; CAC unit test against Lv25 Table 2; TS against Guo17 | R1 W5, R3 W4/W8 |
| Appendix A.3 | LAION-2B token-frequency per dataset | R1 W8 |
| Appendix A.4 | Power analysis for Spearman, Nemenyi, mixed-effects at anchored effect sizes | R1 W2, R3 W6/W7 |

---

## New Experiments

### NE1: Full 11 x 2 x 3 x 7 sweep (plus SigLIP add-on)

- *Purpose:* Replace all anchored numbers in Section 5; address R3 W1, R2 W1, the dominant rejection driver.
- *Setup:* 11 datasets (7 fine-grained + 4 coarse), 3 backbones (OpenCLIP ViT-B/16, ViT-L/14, SigLIP-ViT-B/16), 3 seeds (0, 1, 2), 7 calibrators (C0-C6), zero-shot primary + fine-tuning robustness; pre-registered statistical tests (Section 4.6).
- *Compute:* Approximately 255 GPU-hours total; 2 GPUs x 6 days plus buffer; fits in 4-week window.
- *Expected outcome:* Measured Table 2 with CIs; Pareto frontier figure with measured points; per-RQ decision per pre-registered rules.

### NE2: tau_txt decomposition + manipulation re-run

- *Purpose:* Address R1 W4 and R2 W3; disambiguate the three causal stories (base-base similarity, novel-novel similarity, cross-split similarity).
- *Setup:* For each of 11 datasets, compute (tau_bb, tau_nn, tau_bn); regress each ECE component on each tau component with Holm correction; PCA-whitening applied to (a) base subspace, (b) novel subspace, (c) joint subspace separately; class-level partial Spearman as primary correlation statistic.
- *Expected outcome:* Decomposed Table 3 showing which subspace's geometry drives which ECE component; CUB-200 base run replaces CIFAR toy for manipulation evidence.

### NE3: Reproducibility unit tests

- *Purpose:* Address R1 W5, R3 W4, R3 W8; ensure DAC and CAC implementations match published recipes within +/- 0.5 pp before the main sweep is trusted.
- *Setup:* DAC on Stanford Cars ViT-B/16 vs Wang24 Table 1; CAC on Flowers-102 ViT-B/16 vs Lv25 Table 2; TS on ImageNet-1k val vs Guo17. Each is a single-cell reproduction with full hyperparameter recipe documentation.
- *Expected outcome:* Three reproduction passes within +/- 0.5 pp; appendix table with our number, published number, and source code path.

### NE4: Power analysis

- *Purpose:* Address R1 W2, R3 W6, R3 W7; establish that the pre-registered statistical procedures have adequate power at the anchored effect sizes.
- *Setup:* For Spearman at n=11, simulate under rho in {0.3, 0.5, 0.55, 0.7} and compute power at alpha=0.05; for Nemenyi at k=7, N=11, compute CD and tabulate power for the predicted DAC/CAC/Dirichlet cluster; for mixed-effects, Monte Carlo on random-intercept identification at the anchored slope.
- *Expected outcome:* Appendix A.4 with power tables; if power < 0.7 at the anchored effect, document the fall-back already pre-registered in Sections 4.6 / 5.3.

---

We close by acknowledging that the central criticism — no measured results — is correct and that the rebuttal stands or falls on the execution commitment. We have laid out a realistic 4-week plan with concrete unit-test gates and pre-registered fall-backs for the methodological concerns the reviewers raised. We thank the reviewers again for the rigor and constructive specificity of their feedback.
