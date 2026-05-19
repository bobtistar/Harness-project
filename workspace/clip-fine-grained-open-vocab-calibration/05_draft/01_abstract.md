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
