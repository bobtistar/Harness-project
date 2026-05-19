# Simulated Peer Review

> Venue assumed: NeurIPS / CVPR / ICLR main track (full paper, 8-9 pages).
> Reviewers operate independently. Meta-reviewer aggregates after reading all three.

---

## Reviewer 1 — Methodology / Statistics Specialist

**Score (1-10)**: 4
**Recommendation**: Weak Reject
**Confidence (1-5)**: 4

### Summary
The authors formulate a dichotomy between an *information-channel* and a *geometry-organizer* account of hierarchical taxonomic prompting in biological VLMs, and propose a diagnostic protocol with three layers: (i) a flat-vs-hierarchical geometric audit (Sec 5.1), (ii) a rank-resolved latent-taxonomy probe on OpenCLIP/BioCLIP/BioCLIP2 (Sec 5.3), and (iii) a six-condition counterfactual prompt ablation C0-C5 (Sec 3.3, Sec 5.2). The central instrument is the C2-vs-C3 contrast (structure-preserving content-empty vs content-preserving structure-destroying). Decision rules are pre-registered with effect-preservation ratios and bootstrap CIs. The submission executes a toy-scale OpenCLIP ViT-B/32 sanity check; the BioCLIP2 + TreeOfLife evaluation is registered but not run.

### Strengths
- **S1**: The dichotomy in Sec 3.1 is sharply operationalized (Eq. for $\rho_M$, two-sided decision rule in Sec 3.3 algorithmic sketch lines 144-148). This is markedly more rigorous than typical "ablation-as-narrative" presentation.
- **S2**: Pre-registered thresholds ($\rho_{\text{sil}}(C_2) \geq 0.5$ with CI lower $\geq 0.3$, $\rho_{\text{sil}}(C_3) \leq 0.2$ with CI upper $\leq 0.4$) and an explicit *inconclusive* zone — this is exactly the kind of falsifiability discipline the area needs.
- **S3**: The image-only / image-text parallel reporting (Sec 3.5, Sec 6.3) correctly anticipates modality-gap confounds.
- **S4**: Bonferroni correction across the three RQ1 metrics ($\alpha = 0.0033$) is appropriately conservative for three correlated outcomes.

### Weaknesses
- **W1 (severe, Sec 5.2, Table 2)**: The C2 silhouette (0.9796) is *higher* than C1 (0.9740), and $\Delta_M(C_1) = -0.0125$ is *negative*. The reported preservation ratio $\rho_{\text{sil}}(C_2) = 0.55$ is therefore the ratio of two small negatives. The paper acknowledges this on lines 277-278 but the table on line 266 still prints `0.55` as the headline number, which is statistically misleading. *Any* small noise around zero will produce a ratio in [0, $\infty$). The bootstrap CI "tight at toy scale" (line 266) is mechanically tight because the denominator is fixed across resamples, not because the estimator is well-conditioned. This needs to be either dropped from the table or annotated with a divergence warning.
- **W2 (severe, Sec 5.4)**: Cohen's $d \approx -2910$ for silhouette (line 293). The authors call this an "artefact" of $\sigma \sim 10^{-6}$ but report it anyway. A reviewer expects: (a) the per-seed silhouette values shown explicitly, (b) a Cliff's $\delta$ as the non-parametric robustness check that is *promised* in Sec 4.3 (line 221) but not actually reported anywhere in Sec 5. Where is the Cliff's $\delta$ table?
- **W3 (Sec 3.3, C3 design)**: C3 destroys structure by reassigning upper ranks "from random other species". But the species token is kept correct (Table line 113 footnote). This means the C3 prompt is *internally inconsistent* (e.g., "Plantae Arthropoda Reptilia Diptera Felidae **Passer domesticus**"). It is not obvious that the resulting silhouette collapse (0.97 to 0.10 in Table 2) reflects "structure destruction" rather than the encoder seeing a *contradictory* prompt that pulls the text embedding off the species cluster entirely. A cleaner C3 would copy the *entire* seven-rank tuple from a random other species (so the prompt is internally consistent but assigned to the wrong image). Without this control, C3 conflates "structure" with "label correctness".
- **W4 (Sec 4.3, statistical multiple testing)**: Bonferroni is applied across the three RQ1 metrics ($\alpha/3$) but not across the six conditions, four RQs, five domains (RQ4), or seven ranks (RQ2). The effective number of tests is much larger. Either family-wise correction must be extended or an FDR procedure (Benjamini-Hochberg) should be specified.
- **W5 (Sec 3.5, "Pre-registration")**: The paper calls thresholds "pre-registered" but the registration is internal to `02_rqs.md`. There is no timestamped commit hash, OSF/AsPredicted link, or third-party registry. In the current ML-reviewing climate this is a weak form of pre-registration; please commit to a hash or third-party time-stamp in the camera-ready.
- **W6 (Sec 3.3, C2 token confound)**: `tax0`-`tax5` are claimed to be "vocabulary-external" with "no semantic load". But OpenCLIP's BPE will tokenize `tax0` as `tax` + `0`, and `tax` may pick up shared subword embeddings with "taxonomy" / "tax" — *exactly* the semantic field of interest. A BPE-decomposition table showing how each of the six placeholder tokens is segmented, and ideally a control with truly random unicode strings (e.g., `xq7@2`), would close this gap.
- **W7 (Sec 6.3, internal validity)**: The token-count confound is acknowledged on line 336 and "mitigated by padding length-match in an ablation". But this ablation is not reported anywhere in Sec 5. If it exists, show it; if not, do not claim mitigation.

### Detailed Comments
- [Sec 3.1, line 88]: H_geom is defined as $\rho_M(C_2) \gtrsim 0.5$ AND $\rho_M(C_3) \lesssim 0.2$. What about cases where $\rho_M(C_2) \approx 0.8$ but $\rho_M(C_3) \approx 0.5$? The decision rule on lines 144-148 treats this as inconclusive, which is correct, but the prose in Sec 3.1 reads as if the two conjuncts could be evaluated independently. Please add a one-sentence clarification.
- [Sec 4.3, line 220]: "Bonferroni correction for the three RQ1 metrics: $\alpha = 0.01 / 3 \approx 0.0033$". Starting from $\alpha = 0.01$ rather than the conventional 0.05 is unusual and unmotivated. Either keep 0.05 (giving 0.0167) or justify the 0.01 base rate.
- [Sec 5.1, Table 1]: The "passes pre-registered threshold? NO (reversed)" entries are honest, but the table format does not show effect sizes or sample sizes. Add columns for $n$, $d$, and Cliff's $\delta$.
- [Sec 5.2, line 268-269]: Reporting `ill-conditioned` for three of the four preservation ratios is fine, but the table still shows the numerical values (70.9, 13.2, 23.4) in `04_experiments/results.md` line 65-67. The paper version (line 268-269) omits these numbers, which is better; please ensure consistency between the two artefacts.
- [Sec 5.4, line 292]: "p-values in the range 0.24-0.26". Please report the exact p-values per metric, per seed, in a supplementary table.
- [Algorithm sketch, line 144-148]: The decision rule is unilateral on $\rho_2$ for falsifying H_info. What if $\rho_2 = 0.35$ (between 0.3 and 0.5)? The rule maps it to "inconclusive" but the prose elsewhere is less precise. Suggest tabulating the full 2D decision plane $(\rho_2, \rho_3)$.

### Questions for Authors
- **Q1**: For C3, did you also try the "transplant entire taxonomy from random species" variant I describe in W3? If so, please report; if not, what is the expected qualitative difference?
- **Q2**: The paired permutation p-value range (0.24-0.26) is reported without per-metric breakdown. With n=48 toy images, what is the *minimum effect size* the test can detect at 80% power, $\alpha = 0.0033$? A power analysis would help interpret the null toy result.
- **Q3**: The promised Cliff's $\delta$ robustness check (Sec 4.3 line 221) is not reported in Sec 5. Where is it?
- **Q4**: For pre-registration validity, can you commit to depositing the protocol on OSF with a timestamp *before* running BioCLIP2?
- **Q5**: The decision rule requires three conjunctive conditions for H_geom ($\rho_2$, $\rho_3$, $\rho_5$). What is the probability of all three passing under the null, and have you simulated this?

---

## Reviewer 2 — Novelty / Positioning

**Score (1-10)**: 6
**Recommendation**: Borderline (lean accept on framing, lean reject on execution)
**Confidence (1-5)**: 4

### Summary
The paper positions itself as a *quantitative and causal verification framework* for BioCLIP2's emergent-property claim and as a refinement of WaffleCLIP's random-descriptor finding to the structure-versus-content axis in a biological hierarchy. It distinguishes itself from BioCLIP / BioCLIP2 (which assert but do not test geometric reorganization), from HMLC / Taxes-Are-All-You-Need / HiCLIP (which propose hierarchical losses without disentangling mechanism), and from prompt-side methods CHiLS / HAPrompts / ProTeCt / HPT / HGCLIP (which improve metrics under an *informational* interpretation). The C2-vs-C3 contrast is offered as the central novel instrument.

### Strengths
- **S1**: The framing of *information-channel vs geometry-organizer* as an explicit dichotomy is, to my knowledge, novel as a stated thesis in the biological-VLM literature. BioCLIP2's "emergent properties" prose (Gu et al. 2025) does not articulate this dichotomy.
- **S2**: The relationship to WaffleCLIP (Roth et al. 2023) is correctly characterized in Sec 2.4 and Sec 6.1: WaffleCLIP showed random descriptors work in general domains but did not separate structure preservation from destruction. The proposed C2 vs C3 contrast is a clean extension and is correctly credited as such (line 119, line 314).
- **S3**: The latent-structure probe (RQ2, Sec 5.3) connects to a real gap: Concept Visualization (Bertucci et al. 2024) shows post-hoc WordNet recovery from CLIP, but no prior work runs a random-taxonomy permutation test against the *Linnaean* hierarchy on a frozen biological-VLM stack. This is a genuine contribution if executed.
- **S4**: The positioning is unusually honest. Sec 1 line 22, Sec 5 caveat on line 242, and Sec 7 line 358 explicitly demote the executed run to a "pipeline check" rather than a result. This is rare and admirable in current submissions; it raises the credibility of the framing even though it lowers the empirical claim.

### Weaknesses
- **W1 (Sec 2.2, positioning vs HMLC/HiConE)**: The paper says HMLC and *Taxes Are All You Need* are "the closest prior" to C5 (line 47) but does not explain *what would be empirically different* between running C5 and running the published Taxes-Are-All-You-Need pipeline on TreeOfLife. If the difference is "we run it as one of six conditions in a unified framework" that is incremental; if the difference is "we measure embedding geometry on the resulting model" that is meaningful but must be stated. As written, this distinction is muddled.
- **W2 (Sec 2.4, WaffleCLIP)**: The submission frames C2/C3 as a refinement of WaffleCLIP. But WaffleCLIP's main claim is on *classification accuracy ensembling*, not on geometry. The submission therefore inhabits a slightly different claim space and should not lean too hard on WaffleCLIP as the prior; otherwise reviewers will say "WaffleCLIP already did this". I recommend reframing as "we lift WaffleCLIP's accuracy-level random-descriptor result to a *geometric* and *structure-vs-content* test in a domain with a ground-truth tree". This is a genuine and defensible contribution but the paper currently undersells it by treating WaffleCLIP as a closer precursor than it is.
- **W3 (Sec 2.5, hyperbolic relegation)**: Hyperbolic alternatives (MERU, HyCoCLIP, PHyCLIP) are pushed to Sec 2.5 as "out of scope, future work". But if the central thesis is that hierarchical supervision *organizes* a latent geometry, then a hyperbolic backbone is not just a related method — it is a *first-principles alternative* that the paper's own logic predicts should remove the marginal benefit of hierarchical prompts. The paper says this on line 350 in passing but should foreground it. Right now hyperbolic feels like throat-clearing in Sec 2.5; it should be a Sec 6.4 prediction with empirical commitment.
- **W4 (Contributions framing, lines 25-29)**: The four bullets in Sec 1 read as "we propose a framework + run a toy check". For NeurIPS/CVPR/ICLR, "framework without main result" is a hard sell even with pre-registration. The contribution needs sharper teeth: either (a) one *real* finding (even if narrow), or (b) reframe explicitly as a *protocol / negative-results / RFC* paper and submit to a venue that accepts this (e.g., NeurIPS Datasets and Benchmarks, or an ICLR workshop). As submitted to a main track this is structurally underweight.
- **W5 (Sec 2.1, BIOSCAN-CLIP positioning)**: BIOSCAN-CLIP / CLIBD (Gong et al. 2024) is cited as a "cross-model robustness check" (line 41), but its image+DNA design is qualitatively different from BioCLIP2. The paper does not commit to running the diagnostic framework on BIOSCAN-CLIP. If you can run it, do; otherwise drop the reference, because as written it reads as a name-drop.
- **W6 (Sec 1, latent-structure claim novelty)**: The paper claims a "rank-resolved latent-taxonomy probe" as a contribution (line 27). Sneyers et al. 2023 and the "Probing PLMs with Hierarchy Properties" paper (2023) already do rank-resolved hierarchy probes in *language*; the lift to vision-side rank-resolved silhouette is plausible but the novelty is incremental. Either (a) emphasize the *Linnaean* + *biological VLM* axes as the novelty (which is real), or (b) drop this as a separate contribution and fold it into the umbrella framework.
- **W7 (Sec 7, conclusion overreach)**: The final sentence (line 358) claims that the implication "would re-frame BioCLIP2's emergent properties as the manifestation of pre-existing latent taxonomic structure made visible by hierarchical supervision". This is a strong and interesting reframing — but it is gated on a run that is not in the paper. A reader who skims will read this as a result.

### Detailed Comments
- [Sec 2, missing citation]: HGCLIP (Xia et al. 2024) is cited but the paper does not engage with their finding that *graph-structured* hierarchy in CLIP improves fine-grained classification by ~15%. This is highly relevant: if a graph-structured hierarchy injection helps, that supports the geometry-organizer reading. Please add 1-2 sentences in Sec 2.2 connecting HGCLIP's result to your H_geom prediction.
- [Sec 6.4, line 348-350]: The hyperbolic prediction "hierarchical Euclidean BioCLIP2 is approximately equivalent to hyperbolic backbones for ranking-style metrics" is *exactly* the kind of sharp falsifiable prediction the paper needs more of. Promote it earlier, ideally to Sec 1.
- [Sec 2.1, line 41]: "Hyperbolic variant for biological taxonomies" cites `\cite{gong2025hyperbolic}` — please specify whether this is the Imageomics workshop NeurIPS'25 paper (it appears to be). The lit review has the full citation but this section should disambiguate.
- [Sec 1, contribution 4 line 29]: "An open implementation and toy-scale execution" — for a main-track submission, an open implementation alone is not a contribution; many submissions release code. Either elevate this to "fully reproducible pipeline with mock and toy modes, complete with deterministic seeds and decision logic" with more specifics, or drop.

### Questions for Authors
- **Q1**: Why is the proposed framework not run on a *small but real* biological subset (say, 50 species from CUB-200 Birds or the BioCLIP demo data)? CUB has 200 fine-grained bird species with metadata, fits in 5 GB, and would let you run real (not synthetic) data on CPU within hours. The toy-vs-CUB step is *much* smaller than the toy-vs-BioCLIP2+TreeOfLife step you propose.
- **Q2**: Have you contacted the BioCLIP2 authors to clarify whether their NeurIPS'25 supplementary already includes any of the metrics you propose (silhouette, alignment/uniformity)? A direct correspondence might either pre-empt your contribution or strengthen it.
- **Q3**: Do you commit to a *primary* outcome metric for the camera-ready, or will you report all three RQ1 metrics symmetrically and let the reader weight them? Pre-registered work typically commits to one primary.
- **Q4**: How does the proposed framework differ from a hypothetical "run WaffleCLIP on TreeOfLife with hierarchical descriptors"? Please add a one-paragraph delta in Sec 2.4.

---

## Reviewer 3 — Empirical Rigor / Reproducibility

**Score (1-10)**: 3
**Recommendation**: Reject
**Confidence (1-5)**: 5

### Summary
This is a well-written *protocol* paper with a fully implemented pipeline whose only executed evaluation is on a synthetic 8-species, 48-image toy taxonomy with OpenCLIP ViT-B/32 on CPU. The paper does not execute the announced primary analysis (BioCLIP2 on TreeOfLife). Acceptance to a top-tier venue would set a problematic precedent.

### Strengths
- **S1**: Reproducibility scaffolding is exceptionally thorough (Sec 4.4 lines 226-236). Software versions, seeds, deterministic flags, mock mode, and command-line entry point are all specified. This is better than the median accepted paper.
- **S2**: The submission *does not hide* the toy/mock status. The opening sentence of Sec 5 is a giant CAVEAT box (line 242). `04_experiments/results.md` line 4-9 is even more explicit: "DO NOT cite the numbers". This is honest reporting and is much harder to find at NeurIPS than it should be.
- **S3**: The mock mode (Sec 4.4 line 226) is a legitimate engineering best practice — it permits CI testing of the metric code without model downloads. Reviewers can run `--mock` to verify the pipeline closes.
- **S4**: The decision logic (Sec 3.3 algorithmic sketch) is fully encoded as code, not just prose. This means a reviewer or replicator can run the pipeline and *automatically* obtain a verdict, removing post-hoc cherry-picking.

### Weaknesses
- **W1 (FATAL, Sec 5 entirely)**: The submission's executed evaluation is on **48 synthetic images** generated as class-conditioned random RGB tensors (Sec 4.1 line 190, Sec 6.2 line 326). This is not a result; it is an integration test. The primary contribution the paper *claims* in Sec 1 (a verification framework for BioCLIP2's emergent properties) is *not executed* on BioCLIP2, *not executed* on TreeOfLife, and *not executed* on any real biological imagery. Sec 5.5 line 304 honestly admits this. **This alone is sufficient for rejection from a main track.** The framework may be valuable but is not validated. NeurIPS/CVPR/ICLR do not accept protocol papers without execution outside designated tracks.
- **W2 (severe, Sec 5.1 Table 1)**: All three RQ1 metrics go the *wrong direction* on toy data (intra_var unchanged, inter_margin reversed by 323x, silhouette reversed). The paper attributes this to the synthetic data's class-conditioned color signal (line 256). But the same attribution implies the toy run cannot validate the *pipeline* in any meaningful sense either — if the pipeline produces reversed signs on data where C1 should match C0 or be near it, what guarantees that it produces sensible signs on real data? At minimum, an additional smoke test using *real* CUB-200 or Oxford Flowers with OpenCLIP would demonstrate the pipeline's directional correctness on real data without requiring BioCLIP2 weights.
- **W3 (severe, Sec 5.2)**: Three of four preservation ratios in Table 2 are flagged "ill-conditioned" because the denominator $\Delta_M(C_1)$ has the wrong sign. The fourth (C2 silhouette = 0.55) is *also* ill-conditioned by the same logic but is reported without flagging in the table headline. This is inconsistent.
- **W4 (severe, reproducibility — code claims unverified)**: Sec 4.4 line 226 references `code/prompt_variants.py`, `code/extract_embeddings.py`, `code/metrics.py`, `code/run_experiment.py`, `code/run_all.sh`, `code/run_all.ps1`. I cannot verify the code runs end-to-end on the BioCLIP2 model from the manuscript alone. For acceptance the authors must (a) publish an anonymized repository link, (b) demonstrate a successful BioCLIP2 inference on at least 100 species, even without the full counterfactual ablation.
- **W5 (Sec 4.4 line 236)**: "swapping `--model bioclip2 --csv data/treeoflife_eval.csv` reproduces the full intended evaluation without code changes". This is asserted but not demonstrated. A reviewer cannot verify this claim. If the BioCLIP2 model class differs from OpenCLIP's wrt tokenization, normalization, or feature-extraction layer indices, the code claim is wrong. Please show the BioCLIP2 inference path was tested even on a single image.
- **W6 (Sec 4.4 line 230, seeds)**: Five seeds {42, 1337, 2024, 7, 314} are specified but only "3 seeds" appear in Table 1 caption (line 247) and Table 2 caption (line 261). Which three? Are the bootstrap CIs aggregated across the three seeds or within each? The seed accounting needs to be explicit.
- **W7 (Sec 4.4 line 234, C5 design)**: LoRA rank 8, lr 1e-4, batch 256, 5 epochs is *one* point in hyperparameter space. The author acknowledges this in Sec 6.2 line 330 as a limitation, but for an ablation condition that is one of three pillars of the central thesis, a single point is inadequate. At minimum, learning rate sensitivity {1e-5, 1e-4, 1e-3} and LoRA rank {4, 8, 16} should be probed.
- **W8 (Sec 4.1 line 188, dataset licensing)**: "iNat21 content is used in accordance with individual image licenses" — but iNat21 contains CC-BY-NC images that *cannot be used* for some commercial / model-training pipelines. Has license compliance been verified per-image? For the framework to be reproducible by industrial labs this matters.

### Detailed Comments
- [Sec 4.4, line 232]: "intended TreeOfLife run is sized for a single A100 40 GB, 8-12 hours per domain at batch size 128". This estimate appears not to be verified empirically — there is no benchmark of BioCLIP2 inference throughput in this submission. Please verify or annotate as estimated.
- [Sec 5.3, Table 3]: The z-score of +44.9 against random-taxonomy is impressive at face value but on synthetic class-conditioned color images, it is uninformative. The synthetic images *trivially* expose class-color clusters which are *also* the taxonomic clusters because each species was assigned a different color. This is not "latent taxonomic structure in OpenCLIP" — it is "OpenCLIP can see colors". This must be acknowledged in Sec 5.3 explicitly; right now the caveat is only buried in Sec 5.5 (line 302).
- [Sec 5.5, line 302 "directional support for H1(b)"]: I disagree with this verdict. The synthetic data has color-class confounding such that even a *random* feature extractor would show high silhouette. "Directional support" is too strong; should be "untestable on this data, awaiting real run".
- [Sec 6.4, line 346]: "60-100 A100 GPU-hours" — is this a real estimate based on a profile of BioCLIP2 inference, or a guess? Please clarify.
- [References, line 364]: "The reference list is omitted from this Markdown rendering; the BibTeX file is the authoritative source". Reviewers cannot verify citations without the actual list. Please include in submission.

### Questions for Authors
- **Q1**: Why was a real but small dataset (CUB-200 Birds, Oxford Flowers, or even iNat21 birds subset of ~5K images) not used for the executed run? This would (a) eliminate synthetic-color confounds, (b) provide directional validation of the pipeline, (c) potentially fit on the existing CPU machine in ~24 hours. This is a much smaller step than "execute BioCLIP2 + TreeOfLife" and is the missing intermediate validation.
- **Q2**: Has the BioCLIP2 inference path been tested for a single image end-to-end (not full evaluation, just "model loads + one image embeds")? If yes, why is this not reported as a milestone in Sec 4.4 execution-status table? If no, please clarify how confident the authors are that the announced ablation will succeed.
- **Q3**: For C5, are you committed to publishing the LoRA adapter weights so that the result is reproducible? Or only the training code?
- **Q4**: The submission promises a full BioCLIP2 + TreeOfLife run "as immediate future work" (line 23, line 324). If accepted, would the camera-ready include those results? If not, the contribution gap will persist.
- **Q5**: Does the submission code respect the iNat21 image licenses (per-image CC-BY-NC etc.) when aggregating for evaluation? Reviewers expect a license-compliance script.

---

## Meta-Review

**Aggregate Recommendation**: **Reject** (with strong encouragement to resubmit after a real-data execution).

**Scores**: R1 = 4, R2 = 6, R3 = 3 → Mean = 4.33, Median = 4.

### 합의된 강점
- **Framing is sharp and original**: All three reviewers agree the information-channel vs geometry-organizer dichotomy is well-articulated and that the C2-vs-C3 design is a genuine refinement of WaffleCLIP. R1: S1 (operationalization), R2: S1 (dichotomy as thesis), R3: S2 (honest framing).
- **Honesty and pre-registration discipline**: The submission's repeated acknowledgement that results are toy-scale (Sec 1 line 22, Sec 5 caveat line 242, Sec 5.5 verdicts, Sec 7 conclusion) is unusually scrupulous. R2 (S4) and R3 (S2) both rate this as a credibility asset, and R1 (S2) notes pre-registered thresholds.
- **Reproducibility scaffolding**: Mock mode, deterministic seeds, software-version specification, command-line entry, decision rules encoded as code. R3 (S1, S3, S4) and R1 implicitly endorse. This is above-median for the venue.

### 합의된 약점 (저자가 우선 해결해야 할 것)

1. **PRIMARY: No execution on real biological data (R3-W1, R2-W4, R1 implicit)**. The submission's central claim is a verification of BioCLIP2's emergent properties. Neither BioCLIP2 nor any real biological imagery is evaluated. R3 calls this fatal. The minimum bar for resubmission is a real-data run, even at modest scale (CUB-200 with OpenCLIP, or a 50-species TreeOfLife subset with BioCLIP2). Without this the submission is a protocol/RFC and belongs in a workshop track.

2. **Preservation-ratio reporting on ill-conditioned denominators (R1-W1, R3-W3)**. The headline number `$\rho_{\text{sil}}(C_2) = 0.55$` in Sec 5.2 Table 2 is the ratio of two near-zero quantities with reversed signs. Either drop the number or annotate prominently with an explicit divergence flag. The verbal disclaimer two paragraphs later is insufficient because the table is what gets remembered.

3. **C3 confounds "structure destruction" with "label inconsistency" (R1-W3)**. The current C3 design produces internally-contradictory prompts (mismatched upper ranks + correct species token). A cleaner C3 transplants the full seven-rank tuple from a random other species. This must be added or the C3 collapse cannot be cleanly attributed to *structure destruction*.

4. **C2 placeholder tokens may smuggle subword semantics (R1-W6)**. `tax0`-`tax5` BPE-decompose; the `tax` morpheme is not semantically empty. A unicode-random control is needed.

5. **Latent-structure probe on synthetic color data is uninformative (R3-W2, R3-comment on Sec 5.3)**. The z=44.9 against random-taxonomy is a property of class-conditioned color, not of OpenCLIP's latent taxonomy. The Sec 5.5 verdict "directional support for H1(b)" must be downgraded to "untestable on this data".

6. **Statistical multiple-testing correction is partial (R1-W4)**. Bonferroni is applied across 3 RQ1 metrics but not across 7 ranks, 6 conditions, 5 domains. Either extend FWER correction or commit to an FDR procedure.

7. **Promised robustness statistics not reported (R1-W2, R1-W7)**. Cliff's $\delta$ (promised Sec 4.3) is missing from Sec 5. Token-count padding control (promised Sec 6.3) is missing from Sec 5.

### 분기된 의견
- **R2 vs R3 on overall framing**: R2 leans accept on the protocol contribution alone (score 6) provided the framing is sharpened; R3 leans reject (score 3) because the framework without execution is below main-track threshold. R1 splits the difference (score 4): the methodology is solid but partially under-delivered. **Resolution**: The framing is genuinely valuable but main-track venues (NeurIPS/CVPR/ICLR) typically require at least one main empirical result. A workshop or Datasets & Benchmarks track is a better fit *unless* one real run is added.
- **R2 vs R1 on novelty of latent-structure probe**: R2-W6 says it is incremental over Sneyers et al. 2023's language-side probes; R1 does not engage this question directly. **Resolution**: Reframe the latent-structure probe as "first rank-resolved Linnaean-taxonomy probe across the BioCLIP family" — this is a real first.
- **R1 vs R3 on the toy run's value**: R1 treats the toy run as adequate pipeline validation; R3 disputes even this, noting reversed signs on toy data make the pipeline's directional correctness on real data unverifiable. **Resolution**: R3 has the stronger point; the authors should run on real CUB-200 to validate directionality before the BioCLIP2 escalation.

### Author Action Plan (Rebuttal Stage)

**Immediate (in-rebuttal, doable in 1-2 weeks)**:
- Run the pipeline on **CUB-200 Birds with OpenCLIP ViT-B/32** (real images, 200 fine-grained classes, fits on CPU/laptop in ~12 hours). This single addition would resolve R3's primary concern about pipeline directional validation and provide a non-toy data point.
- Add a **unicode-random C2'** (`xq7@2`, `Z9!4p`, etc.) as a control alongside `tax0`-`tax5` to address R1-W6.
- Add a **transplant-C3'** condition where the full seven-rank tuple is copied from a random other species, to address R1-W3.
- Verify BioCLIP2 model loading and single-image inference end-to-end (just to confirm the swap-the-model-name claim in Sec 4.4). This is hours of work, not a full run.
- Report **Cliff's $\delta$** for all RQ1 comparisons.
- Add a **divergence flag column** to Table 2.

**Body-edit only (no new experiments)**:
- Downgrade Sec 5.5 "directional support for H1(b)" verdict.
- Add 1-paragraph in Sec 2.4 explicitly delineating C2/C3 from WaffleCLIP at the *geometric measurement* level (R2-W2).
- Add 1-2 sentences in Sec 2.2 engaging HGCLIP's empirical result (R2-comment).
- Commit to OSF/AsPredicted pre-registration timestamp (R1-W5).
- Specify primary outcome metric (R2-Q3).
- Include reference list in the submitted PDF (R3-comment).

**Long-horizon (post-acceptance / camera-ready)**:
- The full **BioCLIP2 + TreeOfLife** run. Without this the paper remains a protocol contribution. Even a 100-species subset with BioCLIP2 in the camera-ready would shift R3's recommendation upward by 2 points.
- Cross-domain RQ4 run on at least 2 domains (Aves + Insecta is the easiest).
- C5 hyperparameter sweep (R3-W7).

### Decision Rationale
The paper is intellectually valuable: the dichotomy is sharp, the C2-vs-C3 design is novel, the pre-registration discipline is exemplary, and the honesty about the toy-scale execution is admirable. However, for a main-track venue (NeurIPS/CVPR/ICLR), a paper whose primary empirical result is a 48-image synthetic toy run with metrics moving in the *reversed* direction is below the bar regardless of how well the protocol is articulated. The structural problem is not that the toy results are weak — the authors openly say so — but that *no real-data result exists* to anchor the framework. R3's reject is therefore well-grounded; R2's lean-accept is conditional on a framing the venue would not accept without execution; R1's weak-reject reflects partial methodology issues that are independently fixable.

A **single real-data run on CUB-200 or a small TreeOfLife subset** would likely move R3 to borderline and R1 to weak accept, which would lift the meta-recommendation above the bar. The encouraging path is therefore *not* "execute the full $60-100$ A100-hour BioCLIP2 + TreeOfLife campaign" but "execute a minimum real-data validation that closes the synthetic-color confound and demonstrates pipeline directional correctness". This is achievable within a typical rebuttal window. Accordingly, the current recommendation is **Reject for this submission**, with strong encouragement to resubmit after a real-data execution; the paper is closer to acceptance than the score suggests if the missing intermediate run is added.
