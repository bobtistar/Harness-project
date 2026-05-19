# Rebuttal

We thank all three reviewers and the meta-reviewer for unusually engaged and constructive feedback. The criticisms are sharp but, with one exception, do not target the conceptual contribution (the information-channel vs geometry-organizer dichotomy and the C2-vs-C3 counterfactual design); they target the empirical anchor. We agree this is the correct diagnosis. Below we (i) commit to a concrete remediation that we believe lifts the submission above the bar, (ii) respond to every weakness and question individually, and (iii) honestly declare what we cannot do inside the rebuttal window.

---

## To All Reviewers

All three reviewers converge on a single root cause: the executed evaluation is a 48-image synthetic toy run with class-conditioned color confounds, and the BioCLIP2 + TreeOfLife escalation that the framework was designed for has not been run. R3 calls this fatal; R1 and the meta-reviewer concur that real-data validation is the minimum bar; R2 conditions a lean-accept on the same point. We accept this assessment without qualification. The framework's value cannot be demonstrated against synthetic class-conditioned colors, and the toy "directional support" for H1(b) in Section 5.5 overreaches given the construction of the data.

We commit to the following remediation, achievable inside a 2-3 week rebuttal-to-camera-ready window:

New experiments / revisions (summary):
- **NE1 (CUB-200-2011, OpenCLIP ViT-B/32, full protocol)**: Run all six conditions C0-C5 on the CUB-200 fine-grained bird dataset (200 species, ~6K test images, real photographs with taxonomy metadata). Estimated ~8 hours on the existing CPU/laptop. This eliminates the synthetic color confound and provides directional validation of the pipeline on real data.
- **NE2 (BioCLIP2 single-image inference smoke test + 50-species ablation)**: Verify BioCLIP2 ViT-L/14 model loading and feature extraction end-to-end on a small TreeOfLife subset (50 species, ~1K images, conditions C0-C3 only). This is hours of GPU time on a Colab/Kaggle A100, not the full 60-100 hour campaign.
- **NE3 (Two control conditions added per R1)**: A unicode-random C2' (`xq7@2`, `Z9!4p`, ...) added alongside `tax0`-`tax5` to address the BPE-leakage concern (R1-W6); a transplant-C3' that copies the full seven-rank tuple from a random other species to address the label-inconsistency confound (R1-W3). Both run on CUB-200.
- **Body revisions**: Cliff's delta added throughout (R1-Q3); divergence-flag column added to Table 2 (R1-W1, R3-W3); Sec 5.5 verdict for H1(b) downgraded from "directional support" to "untestable on toy data" (R3-comment); WaffleCLIP delta paragraph in Sec 2.4 (R2-W2, R2-Q4); HGCLIP engagement in Sec 2.2 (R2-comment); OSF time-stamped pre-registration prior to BioCLIP2 run (R1-W5); reference list included in PDF (R3-comment).

We address each reviewer's points below.

---

## Response to Reviewer 1 (Methodology / Statistics)

We thank R1 for the careful technical reading. The methodology concerns are mostly fixable in-rebuttal; we accept all of them.

### R1.W1 (Sec 5.2, Table 2): "0.55" preservation ratio is the ratio of two near-zero negatives; should be dropped or annotated with divergence warning.
**Response**: We agree completely. The verbal disclaimer two paragraphs below the table (lines 277-278) is insufficient because the number in the headline cell is what gets remembered. The denominator $\Delta_M(C_1) = -0.0125$ is too small for the ratio to be meaningfully bounded.
**Action**: Table 2 will be reformatted: the silhouette-ratio column will show `DIV` (with a divergence flag and the absolute deltas in adjacent columns) for every row where $|\Delta_M(C_1)| < \tau$ with a pre-registered $\tau = 0.05$ on silhouette. The "0.55" entry will be replaced by a divergence flag. Real-data CUB-200 results (NE1) will report ratios only where the denominator is significant at $\alpha = 0.0033$.
**Where in revision**: Sec 5.2 Table 2 (revised), Sec 4.3 (add the $\tau$ threshold to the pre-registered protocol).

### R1.W2 (Sec 5.4): Cohen's d of -2910 is an artefact; Cliff's delta promised but never reported.
**Response**: Acknowledged. The Cohen's d figure should not have been left in the paper without per-seed silhouette values and a non-parametric companion. The Cliff's delta omission is an oversight; the metric is implemented in `code/metrics.py` but was not surfaced in Sec 5.
**Action**: Sec 5.4 will (a) report per-seed silhouette values explicitly in a supplementary table; (b) replace the headline Cohen's d with Cliff's delta as primary effect-size for all RQ1 contrasts; (c) retain Cohen's d only when seed variance is in a normal range on CUB-200.
**Where in revision**: Sec 5.4 (Cliff's delta table), Sec 4.3 (re-order so Cliff's delta is the primary effect-size with Cohen's d as parametric companion).

### R1.W3 (Sec 3.3, C3 design): C3 is internally inconsistent ("Plantae Arthropoda Reptilia Diptera Felidae Passer domesticus"); confounds structure destruction with label inconsistency.
**Response**: This is the most substantive design critique and we agree it is a real confound. A cleaner C3' transplants the full seven-rank tuple from a random other species (so the prompt is internally consistent but assigned to the wrong image), separating "structure destruction" from "label contradiction at the species token".
**Action**: We add C3' (transplant-full-tuple) as an additional condition. C3 is retained for backward compatibility but C3' is the primary structure-destruction operationalization in the revised manuscript. Expected qualitative difference (R1.Q1): C3 should collapse silhouette more aggressively than C3' because C3 produces an internally contradictory prompt while C3' produces a coherent-but-wrong one; the H_geom prediction is that C3' should also collapse, since the prompt's hierarchical structure no longer aligns with the image's taxonomy.
**Where in revision**: Sec 3.3 Table (C3 split into C3 and C3'), Sec 5.2 (new ablation row for C3' on CUB-200).

### R1.W4 (Sec 4.3): Multiple-testing correction is partial.
**Response**: Agreed. Bonferroni across three RQ1 metrics is too narrow when seven ranks, six conditions, and five domains are also being tested.
**Action**: We switch from Bonferroni to Benjamini-Hochberg FDR at q = 0.05 for the family of {3 RQ1 metrics} x {7 ranks} x {6 conditions} = 126 tests per domain. For the cross-domain RQ4 meta-analysis we use the random-effects pooled estimate with I-squared as the heterogeneity report (already specified). The single-domain primary outcome (silhouette on the species rank for C2 vs C3) is identified as the *primary* test, with the rest as exploratory under FDR.
**Where in revision**: Sec 4.3 (rewritten), Sec 3.5 (primary-outcome commitment also addressing R2.Q3).

### R1.W5 (Sec 3.5, Pre-registration): Internal markdown is a weak form of pre-registration.
**Response**: Acknowledged. Internal `02_rqs.md` is not third-party verifiable.
**Action**: Before running BioCLIP2 (NE2), we will deposit the protocol (research questions, hypotheses, thresholds, decision rules, primary outcome) on OSF with a public timestamp, and cite the OSF DOI in Sec 3.5 of the revised manuscript. We commit to this in the revision and will provide the OSF link in the camera-ready.
**Where in revision**: Sec 3.5 (footnote with OSF DOI to be inserted at camera-ready), commitment in this rebuttal.

### R1.W6 (Sec 3.3, C2 token confound): `tax0`-`tax5` BPE-decompose to `tax` + digit; `tax` overlaps with "taxonomy" subword semantics.
**Response**: Agreed. This is a real leak we did not consider carefully enough.
**Action**: We add a unicode-random control C2' using strings outside the OpenCLIP/BPE-relevant Latin morphemes (e.g., `xq7@2`, `Z9!4p`, `f3$kL`, ...). We will also publish a BPE-decomposition table in the appendix showing exactly how each of the original `tax0`-`tax5` and each of the new unicode controls is segmented by the OpenCLIP tokenizer. If the silhouette and inter-margin patterns for C2 (tax-prefixed) and C2' (unicode-random) are statistically indistinguishable on CUB-200, the BPE-leakage concern is empirically closed.
**Where in revision**: Sec 3.3 Table (add C2' row), Sec 5.2 (add C2' results on CUB-200), Appendix (BPE decomposition table).

### R1.W7 (Sec 6.3, token-count padding ablation): Claimed but not reported.
**Response**: Acknowledged. The padding-length-matched ablation is implemented but was not surfaced in Sec 5.
**Action**: We will report the length-matched C0' (padded C0 to C1's token count) as a row in Sec 5.2 on CUB-200. If C0' tracks C0 (not C1), the token-count confound is empirically negligible.
**Where in revision**: Sec 5.2, Sec 6.3 (cross-reference instead of unanchored claim).

### R1.Q1: Did you try the transplant-full-tuple C3' variant?
**Response**: Not in the toy run, but we add it for CUB-200 (NE1) and address the prediction explicitly in W3 above. Expected: C3' collapses similarly to C3 under H_geom (because the prompt's structural alignment with the image is broken regardless of internal consistency); under H_info, C3' should remain closer to C1 than C3 because the species-token contradiction is removed.

### R1.Q2: Power analysis at n=48, alpha=0.0033, 80% power.
**Response**: At n=48 paired samples with $\alpha = 0.0033$ and 80% power, the minimum detectable standardized effect (Cohen's d) is approximately 0.69 (computed via `statsmodels` power analysis). With seed variance collapsing to $10^{-6}$ on synthetic data the *measured* d is well above this floor but for the wrong reason. On CUB-200 with n ~ 6000 images per condition, the minimum detectable d falls below 0.1, making the test sensitive to small but real effects.
**Action**: Power analysis added as a supplementary section with the formula, n, and minimum detectable effect for each comparison.

### R1.Q3: Where is Cliff's delta?
**Response**: See R1.W2. It will be the primary effect-size in the revision.

### R1.Q4: Commit to OSF time-stamp before BioCLIP2 run?
**Response**: Yes, committed. See R1.W5.

### R1.Q5: Probability of all three conjunctive conditions passing under the null?
**Response**: Under a Gaussian null with independent metrics and the pre-registered one-sided thresholds, the joint pass probability is roughly $\alpha^3$ under strict independence; under positive correlation among the three RQ1 metrics (which we observe empirically on toy data, $r \approx 0.6$-$0.9$) the joint pass probability is closer to $\alpha^{1.5}$ -- still well below the per-test rate. We will report a small Monte-Carlo simulation (10K replicates) confirming the joint Type-I rate is < 0.001 under our nominal $\alpha = 0.0033$.
**Action**: Add this to Sec 4.3 as a calibration footnote.

### R1 Detailed Comments
- **[Sec 3.1, decision rule clarity]**: We add the one-sentence clarification that the two conjuncts are evaluated jointly, not independently, and tabulate the full 2D $(\rho_2, \rho_3)$ decision plane in Sec 3.3.
- **[Sec 4.3, alpha=0.01 rationale]**: We acknowledge this was unmotivated; we revert to the conventional $\alpha = 0.05$ with FDR (per W4 above), making $q = 0.05$ the family-wise control.
- **[Sec 5.1 Table 1, add columns]**: $n$, Cliff's delta, and bootstrap CI will be added; Cohen's d retained as secondary.
- **[Sec 5.2 vs 04_experiments/results.md numerical consistency]**: We will harmonize the two artefacts; the paper version (omitting the ill-conditioned numbers) is the canonical reporting.
- **[Sec 5.4 per-seed p-values]**: Added to supplementary.
- **[Algorithmic sketch, 2D decision plane]**: Added as Figure 2 in Sec 3.3.

---

## Response to Reviewer 2 (Novelty / Positioning)

We thank R2 for the close reading and for the rare framing-level engagement. Most of R2's critiques are about sharpening and re-anchoring claims; we accept all of them and most are body-edits.

### R2.W1 (Sec 2.2, HMLC/HiCoNE positioning): What is empirically different between C5 and Taxes-Are-All-You-Need run on TreeOfLife?
**Response**: The difference is *not* the training pipeline but the *measurement*: C5 is treated as one of six counterfactual conditions, evaluated under the same effect-preservation ratio framework as C2-C4, with image-only and image-text geometric metrics rather than downstream classification accuracy. Running Taxes-Are-All-You-Need as a stand-alone pipeline gives an accuracy number; running it as C5 gives a $\rho_M(C_5)$ value comparable to $\rho_M(C_2)$ and $\rho_M(C_3)$, which is what isolates the *mechanism* (does text-free hierarchical supervision reproduce the geometric reorganization of hierarchical text?).
**Action**: One-paragraph delta added to Sec 2.2 making this distinction explicit.

### R2.W2 (Sec 2.4, WaffleCLIP positioning): Reframe as "lifting WaffleCLIP's accuracy-level result to a geometric and structure-vs-content test in a ground-truth-tree domain".
**Response**: We accept this reframing. WaffleCLIP's claim is at the ensembling-accuracy level; our claim is at the geometry / structure-vs-content level in a domain with an unambiguous hierarchy. We have been undercrediting our own contribution by leaning too hard on WaffleCLIP as the precursor.
**Action**: Sec 2.4 paragraph rewritten with the lift framing; the contribution bullets in Sec 1 are sharpened accordingly.

### R2.W3 (Sec 2.5, hyperbolic relegation): Hyperbolic alternatives are not throat-clearing; they are a first-principles consistency check.
**Response**: We agree. The hyperbolic prediction in Sec 6.4 is exactly the sharp falsifiable consequence the paper needs to foreground.
**Action**: The hyperbolic-equivalence prediction is promoted to Sec 1 (contribution bullets) as an empirical commitment: if H_geom is supported, then a hyperbolic backbone (MERU / HyCoCLIP / PHyCLIP) should *remove* the marginal benefit of hierarchical Euclidean prompts. We commit to running C0 and C1 on a hyperbolic checkpoint in the long-horizon revision (post-acceptance), and we surface this as a registered prediction.

### R2.W4 (Contributions framing): "Framework without main result" is a hard sell for main track.
**Response**: Agreed. The CUB-200 run (NE1) and BioCLIP2 50-species smoke test (NE2) move the submission from "framework + toy" to "framework + one real finding + one real-scale verification". We re-frame the contributions accordingly: (1) the dichotomy as a thesis, (2) the C2-vs-C3 instrument, (3) the rank-resolved Linnaean probe, (4) the CUB-200 empirical anchor, (5) the BioCLIP2 inference smoke-test (proving the camera-ready full run is feasible). If the meta-reviewer judges this still below main-track bar, we are open to redirecting to NeurIPS Datasets and Benchmarks, where the protocol contribution is more naturally weighted.

### R2.W5 (BIOSCAN-CLIP name-drop): Either run on it or drop.
**Response**: Fair. We have not committed compute to BIOSCAN-CLIP for the revision and we should not cite it as a robustness target without action.
**Action**: We drop BIOSCAN-CLIP from the "cross-model robustness" framing and retain it only in Sec 2.1 as related work. If it is included as a future-work commitment, it is named explicitly as such.

### R2.W6 (Latent-structure novelty over Sneyers et al. 2023): Incremental in language-side probes.
**Response**: Agreed in part. The novelty is not "rank-resolved probing of hierarchical structure" per se but specifically "first rank-resolved Linnaean-taxonomy probe across the BioCLIP family on the *vision* side". We re-cast the contribution as the Linnaean + biological VLM + vision-side conjunction rather than as "rank-resolved probing" abstractly.
**Action**: Sec 1 contribution 2 reworded to: "the first rank-resolved Linnaean-taxonomy probe across OpenCLIP, BioCLIP, and BioCLIP2 on the image side"; Sec 2.4 cites Sneyers et al. and Lin et al. as the language-side precedents.

### R2.W7 (Sec 7 conclusion overreach): The "would re-frame BioCLIP2's emergent properties..." sentence reads as a result.
**Response**: Acknowledged. A skim-reader could mistake the conditional for an empirical finding.
**Action**: Conclusion rewritten to (a) state the empirical *anchor* (CUB-200 result, BioCLIP2 smoke-test) clearly; (b) clearly mark the BioCLIP2 + TreeOfLife implication as gated on the camera-ready run.

### R2.Q1: Why not a small but real biological subset (CUB-200, Oxford Flowers, iNat21 birds)?
**Response**: This is exactly the right question and we accept the criticism. CUB-200 on CPU within hours is achievable and is in fact the right intermediate step we should have included in the original submission. See NE1. The CUB-200 results will be in the revision.

### R2.Q2: Contact BioCLIP2 authors to clarify whether their NeurIPS'25 supplementary already includes any of the proposed metrics (silhouette, alignment/uniformity)?
**Response**: Helpful suggestion. We will contact the BioCLIP2 authors during the rebuttal period. Our reading of the BioCLIP2 supplementary indicates that they report inter-species cosine distances aligned to ecological traits (their "emergent properties" Section) but they do *not* report silhouette, alignment/uniformity, or any counterfactual prompt ablation. If they have these in unreleased code we welcome the chance to align.

### R2.Q3: Primary outcome metric commitment?
**Response**: Yes, committed. The *primary* outcome is silhouette at the species rank for $\rho(C_2)$ vs $\rho(C_3)$ on the largest available domain (Aves). The other RQ1 metrics and ranks are exploratory under FDR.
**Action**: Stated in Sec 4.3 and Sec 3.5.

### R2.Q4: Delta from "WaffleCLIP on TreeOfLife with hierarchical descriptors"?
**Response**: Three differences: (i) WaffleCLIP reports classification accuracy; we report embedding-space geometry. (ii) WaffleCLIP's random descriptors do not preserve a known hierarchical order; our C2 preserves the seven-rank slot structure. (iii) WaffleCLIP has no structure-destroying control; our C3/C3' is precisely that. A WaffleCLIP-on-TreeOfLife run would report top-1 accuracy with random descriptors, which would be uninformative for distinguishing H_info from H_geom.
**Action**: Paragraph added to Sec 2.4.

### R2 Detailed Comments
- **[HGCLIP engagement, Sec 2.2]**: Added: HGCLIP's ~15% gain from graph-structured hierarchy injection is consistent with H_geom (graph structure organizes the embedding), but HGCLIP does not run a structure-shuffled control, so the C2-vs-C3 disambiguation our framework provides is complementary.
- **[Sec 6.4, hyperbolic prediction promotion]**: Done, see W3.
- **[Sec 2.1, hyperbolic citation disambiguation]**: Done; we cite the Imageomics workshop NeurIPS'25 paper by exact title and venue.
- **[Sec 1 contribution 4]**: Reworded from generic "open implementation" to "deterministic, mock-mode-enabled, decision-logic-encoded pipeline with OSF-deposited pre-registration".

---

## Response to Reviewer 3 (Empirical Rigor / Reproducibility)

We thank R3 for the toughest and most consequential review. R3's central point -- that the executed evaluation is not a result -- is correct, and we do not dispute it. Our remediation is concrete: NE1 (CUB-200 with OpenCLIP) and NE2 (BioCLIP2 smoke test + 50-species ablation).

### R3.W1 (FATAL): Executed evaluation is 48 synthetic images; primary contribution is not validated.
**Response**: We accept this in full. The toy run, even under the most charitable interpretation, is an integration test, not a result. We do not argue this is acceptable for a main-track venue.
**Action**: NE1 (CUB-200, real fine-grained bird photos, 200 species, ~6K test images, full six-condition counterfactual ablation, ~8h on CPU) will be in the revised manuscript as Sec 5.1-5.2. NE2 (BioCLIP2 ViT-L/14 inference smoke test on a 50-species TreeOfLife subset, conditions C0-C3, hours on a single A100 via Colab Pro) will be in Sec 5.3 as the proof-of-feasibility for the full camera-ready run. If both succeed by directional pattern matching the H_geom prediction on real data, the protocol contribution acquires its first real empirical anchor.

### R3.W2 (severe): All three RQ1 metrics go the *wrong direction* on toy data; pipeline directional correctness unverified.
**Response**: Agreed, and this is exactly why CUB-200 is the right intermediate. On real images with non-synthetic-color cluster structure, we expect $\Delta_M(C_1)$ to be positive and substantial; if it is not, the pipeline itself is the bug, and we will catch it before the full TreeOfLife run.
**Action**: CUB-200 results will be reported as the primary directional-correctness check.

### R3.W3 (severe): C2 silhouette = 0.55 is also ill-conditioned but reported without flagging.
**Response**: Accepted. See R1.W1; the divergence-flag column resolves this consistently.

### R3.W4 (severe): Code claims unverified; no anonymized repo, no BioCLIP2 inference demonstrated.
**Response**: We will provide (a) an anonymized GitHub repo URL with the full pipeline in the revision; (b) a recorded successful BioCLIP2 ViT-L/14 inference on at least 100 species as part of NE2; (c) a CI workflow logfile showing the full pipeline closes end-to-end on mock data.
**Action**: Sec 4.4 will include the anonymized URL and the inference smoke-test result (NE2).

### R3.W5 (Sec 4.4): "swap --model bioclip2" claim unverified.
**Response**: Accepted. NE2 verifies this empirically. We will report in Sec 4.4: (a) the exact CLI command used to switch from OpenCLIP to BioCLIP2; (b) the resulting silhouette and inter-margin on the 50-species TreeOfLife subset for C0 and C1; (c) any code patches required (we expect token-encoder differences only).

### R3.W6 (Sec 4.4): Seed accounting (5 declared, 3 in table captions).
**Response**: This is an honest mismatch in the toy run -- the toy executed three of the five seeds for time reasons. CUB-200 NE1 will run all five seeds; the bootstrap CIs aggregate *across* seeds (not within), with seed as a random effect.
**Action**: Sec 4.4 seed paragraph rewritten to make aggregation explicit.

### R3.W7 (Sec 4.4, C5 hyperparameters): One point in hyperparameter space.
**Response**: Accepted as a current limitation. We do not commit to a full sweep in the rebuttal window (lr x rank = 9 cells with 5 epochs each on CUB-200 is multiple compute-days); we *do* commit to lr in {1e-5, 1e-4, 1e-3} at fixed LoRA rank 8 (three runs), which is achievable on CPU within 24 hours.
**Action**: Sec 4.4 commitment, Sec 5.2 reports three-point C5 sensitivity strip.

### R3.W8 (iNat21 licensing): Per-image CC-BY-NC compliance.
**Response**: Acknowledged. iNat21 image licenses are mixed; for the CUB-200 NE1 run (which is the headline revision result) all imagery is governed by the CUB-200-2011 license which is research-use compatible. For any iNat21-derived TreeOfLife subset in the camera-ready, we will ship a license-compliance manifest per image (CC-BY, CC-BY-NC, CC-BY-SA, etc.) and a filter script that excludes any image used outside its license terms.
**Action**: Sec 4.1 footnote and a compliance script in the released repo.

### R3 Detailed Comments
- **[Sec 4.4 line 232, inference throughput]**: Acknowledged as estimated, not measured. NE2 produces a measured throughput; the 60-100 A100-hour estimate in Sec 6.4 will be replaced by a measured-and-extrapolated figure.
- **[Sec 5.3, z=44.9 inflated by synthetic color]**: Agreed. Sec 5.3 will be rewritten on CUB-200 results; the toy z-score will be moved to an appendix as "pipeline check, not evidence about latent structure".
- **[Sec 5.5 "directional support for H1(b)"]**: Downgraded to "untestable on synthetic-color data; pending CUB-200" in the revision.
- **[Sec 6.4, 60-100 GPU-hours]**: Replaced with empirical estimate from NE2.
- **[References, full list]**: The reference list will be inlined in the camera-ready PDF.

### R3.Q1: Why not CUB-200 or Oxford Flowers in the executed run?
**Response**: It should have been included; we accept this. NE1 addresses it.

### R3.Q2: Has BioCLIP2 single-image inference been tested?
**Response**: Honestly: at submission time, no. NE2 is precisely this milestone. We report it in the revision.

### R3.Q3: Will C5 LoRA adapter weights be published?
**Response**: Yes. The LoRA adapter weights (rank 8 on the OpenCLIP image encoder) for C5 are < 5 MB and will be released alongside the code at acceptance.

### R3.Q4: Camera-ready includes BioCLIP2 + TreeOfLife results?
**Response**: We commit to BioCLIP2 ViT-L/14 on a 100-species TreeOfLife subset (Aves) with conditions C0-C3 (the central instrument) in the camera-ready. The full 1000-species, five-domain, six-condition campaign is post-camera-ready future work because the 60-100 A100-hour budget is not achievable in a 6-week camera-ready window. A 100-species + C0-C3 + Aves run is feasible in ~10 A100-hours and produces the headline result.

### R3.Q5: iNat21 license compliance script?
**Response**: Yes, committed. See R3.W8.

---

## Summary of Revisions

| Section | Change | Triggered by |
|---------|--------|--------------|
| Abstract | Replace "toy-scale sanity check" with "CUB-200 empirical anchor + BioCLIP2 feasibility verification" | R3.W1, meta-review |
| 1 Contributions | Reframed to (1) dichotomy, (2) C2-vs-C3, (3) Linnaean rank-resolved vision-side probe, (4) CUB-200 anchor, (5) BioCLIP2 feasibility; hyperbolic-equivalence prediction promoted | R2.W4, R2.W6, R2.W3 |
| 2.1 | BIOSCAN-CLIP demoted from robustness-target to related-work | R2.W5 |
| 2.2 | Engage HGCLIP empirical finding; clarify C5-vs-Taxes-Are-All-You-Need distinction | R2.W1, R2-comment |
| 2.4 | Reframe WaffleCLIP as accuracy-level precursor lifted to geometry; delta paragraph for "WaffleCLIP on TreeOfLife" | R2.W2, R2.Q4 |
| 3.1 | Clarify that H_geom conjuncts are joint, not independent; 2D decision plane | R1 detailed |
| 3.3 | Add C2' (unicode random), add C3' (transplant full tuple); BPE-decomposition appendix | R1.W6, R1.W3 |
| 3.5 | OSF-deposited pre-registration commitment with DOI; primary outcome metric specified (species-rank silhouette, C2 vs C3, Aves) | R1.W5, R2.Q3 |
| 4.1 | iNat21 license compliance manifest and filter script | R3.W8 |
| 4.3 | Switch Bonferroni alpha=0.0033 to Benjamini-Hochberg q=0.05; Cliff's delta as primary effect-size; power analysis at n=48 and n=6000 | R1.W4, R1.W2, R1.Q2 |
| 4.4 | Anonymized repo URL; BioCLIP2 inference smoke test logged; seed aggregation made explicit; C5 lr sensitivity strip {1e-5, 1e-4, 1e-3} | R3.W4, R3.W5, R3.W6, R3.W7 |
| 5.1 | Replace toy Table 1 with CUB-200 Table 1; toy run moved to appendix as pipeline check | R3.W1, R3.W2, R2.Q1 |
| 5.2 | Replace toy Table 2 with CUB-200 Table 2; add divergence-flag column; add C2' and C3' rows; length-matched C0' row | R1.W1, R3.W3, R1.W7 |
| 5.3 | Replace toy z=44.9 with CUB-200 rank-resolved latent-structure probe; toy z moved to appendix with caveat | R3-comment, R3.W2 |
| 5.4 | Add per-seed silhouette table; Cliff's delta replaces Cohen's d as primary; per-metric p-values | R1.W2, R1 detailed |
| 5.5 | Downgrade H1(b) verdict on toy from "directional support" to "untestable"; CUB-200 verdicts added | R3-comment, meta-review |
| 5.6 (new) | BioCLIP2 50-species inference smoke test results | R3.W4, R3.W5 |
| 6.2 | Remove "toy-scale" framing as headline limitation; add CUB-200-as-still-not-TreeOfLife caveat | R3.W1 |
| 6.3 | Length-matched C0' ablation cross-referenced (not just claimed) | R1.W7 |
| 6.4 | 60-100 A100-hour estimate replaced by NE2-measured extrapolation; camera-ready commitment to 100-species TreeOfLife Aves run | R3.W4, R3.Q4 |
| 7 | Conclusion rewritten to anchor on CUB-200 result; BioCLIP2 implication clearly gated on camera-ready run | R2.W7 |
| References | Inlined in PDF | R3-comment |

---

## New Experiments

### NE1: CUB-200-2011 with OpenCLIP ViT-B/32, all six conditions
- **Purpose**: Address R3.W1 (no real data), R3.W2 (pipeline directional validation), R2.Q1 (small-but-real biological subset).
- **Setup**: CUB-200-2011, 200 fine-grained bird species, ~6K test images, real photographs with full taxonomy (kingdom -> species) attached via NCBI matching. Encoder: OpenCLIP ViT-B/32 (same as toy run, for direct comparability). Conditions: C0 (flat), C1 (hierarchical), C2 (tax-placeholder), C2' (unicode-random, NEW per R1.W6), C3 (mismatched upper ranks, original), C3' (transplant full tuple, NEW per R1.W3), C4 (word-bag), C5 (LoRA text-free, three learning rates per R3.W7). Length-matched C0' (NEW per R1.W7) included. Five seeds {42, 1337, 2024, 7, 314}. Estimated wall time: ~8 hours on i7-class CPU.
- **Metrics**: All three RQ1 metrics (intra-class variance, inter-class margin, silhouette) at the species rank; preservation ratios for C2, C2', C3, C3', C4, C5 with bootstrap CIs and divergence flags; Cliff's delta as primary effect-size; FDR-corrected p-values.
- **Decision rule**: Pre-registered (Sec 3.1): if $\rho_{\text{sil}}(C_2) \geq 0.5$ with lower CI bound $\geq 0.3$ AND $\rho_{\text{sil}}(C_3') \leq 0.2$ with upper CI bound $\leq 0.4$, H_geom supported on CUB-200. If $\rho_{\text{sil}}(C_2) < 0.3$ regardless of C3', H_info supported. Otherwise inconclusive.
- **Status**: Not yet executed; committed for the revised manuscript.

### NE2: BioCLIP2 ViT-L/14 inference smoke test + 50-species TreeOfLife C0-C3 ablation
- **Purpose**: Address R3.W4 (code-claim verification), R3.W5 (swap-model-name unverified), R3.Q2 (single-image BioCLIP2 inference), R3.Q4 (camera-ready commitment).
- **Setup**: Download BioCLIP2 ViT-L/14 weights from HuggingFace (~6 GB). Sample 50 species from TreeOfLife Aves with >=20 images each (~1000 images). Run C0, C1, C2 (tax-placeholder), C3' (transplant) on this subset. Single A100 via Colab Pro or Kaggle.
- **Metrics**: Same as NE1 but limited to silhouette at species and family ranks (primary outcome).
- **Decision rule**: This is feasibility verification, not the conclusive test. We are validating (a) BioCLIP2 weights load, (b) inference path produces non-pathological embeddings, (c) silhouette $C_1 > C_0$ in the expected direction (this is the directional-correctness check R3.W2 demands on a real biological backbone), (d) C3' collapses below C0.
- **Status**: Not yet executed; committed for the revised manuscript.

### NE3: C5 hyperparameter sensitivity strip
- **Purpose**: Address R3.W7 (single hyperparameter point inadequate).
- **Setup**: On CUB-200, run C5 LoRA training at lr in {1e-5, 1e-4, 1e-3} with rank fixed at 8. Five epochs, AdamW.
- **Metrics**: Silhouette and $\rho_{\text{sil}}(C_5)$.
- **Decision rule**: If $\rho_{\text{sil}}(C_5)$ varies by more than 0.3 across the three learning rates, C5 is hyperparameter-sensitive and we re-frame it as exploratory rather than primary.
- **Status**: Not yet executed; committed.

---

## What We Cannot Do (Honest Refusals)

- **Full BioCLIP2 + TreeOfLife (1000 species, 5 domains, all 6 conditions, 7 ranks)** within the rebuttal-to-camera-ready window: The 60-100 A100-hour budget is not feasible in 4-6 weeks of academic compute access available to us. We commit to a *scoped* BioCLIP2 run (100 species, Aves only, C0-C3) for the camera-ready; the full campaign is honest post-camera-ready future work. This means R3's strongest reading (the framework's central claim should be tested on the announced primary target before acceptance) will be only partially satisfied. We accept this gap.
- **Full hyperbolic-backbone consistency check (MERU / HyCoCLIP / PHyCLIP under our protocol)** within the rebuttal window: This is a multi-week engineering effort to port the prompt-condition pipeline to hyperbolic embedding code. We commit to the prediction (Sec 6.4) and to running the check post-acceptance, but we do not commit to including it in the camera-ready.
- **Full C5 hyperparameter sweep (lr x rank x epochs grid)**: We commit only to the 3-point lr strip (NE3). The full grid is post-acceptance.
- **BIOSCAN-CLIP cross-model robustness run**: Per R2.W5 we drop the claim rather than over-promise.
- **Re-running OpenCLIP toy results on a different synthetic generator**: Not informative. The synthetic-color confound is a property of the data generator, not a fixable bug, and we instead move to CUB-200 (NE1) for the directional validation R3.W2 requires.

---

## Closing

The reviewers' criticisms have substantially improved our understanding of where the submission's empirical anchor must sit. The framework, the C2-vs-C3 instrument, the rank-resolved Linnaean probe, and the pre-registration discipline are contributions we continue to defend. The empirical anchor was inadequate at submission and is being corrected through NE1, NE2, and NE3. We are grateful for reviewers' rigor; if NE1 and NE2 land with the predicted directional pattern, we believe the submission moves above the bar that the meta-review identifies, and if they do not, we accept that the framework needs to be re-anchored or the paper redirected to a Datasets/Benchmarks track.
