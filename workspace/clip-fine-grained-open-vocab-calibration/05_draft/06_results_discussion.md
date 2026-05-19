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
