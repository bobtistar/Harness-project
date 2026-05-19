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
