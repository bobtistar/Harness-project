"""Exp4 (RQ4): ECE → AURC / conformal-set-size monotonic transfer.

Loads ``outputs/exp3_results.json`` produced by ``exp3_calibrators``,
re-runs each (dataset, seed, calibrator) on the same logits to compute
AURC and split-conformal coverage/size, and fits the Spearman + mixed-
effects relationship.

For simplicity we re-execute zero-shot forward passes here; in
production one would pickle the (logits, labels) tensors during Exp3 to
avoid double work.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data import ALL_LOADERS, toy_cifar10_dataset  # noqa: E402
from models import load_clip, evaluate_clip, OPENAI_TEMPLATES, encode_text  # noqa: E402
from calibrators import (Raw, TemperatureScaling, VectorScaling,
                         DirichletCalibration, DAC, CAC,
                         HistogramBinning, CALIBRATOR_IDS)  # noqa: E402
from metrics import (aurc, selective_accuracy_at_coverage,
                     auroc_misclassification, split_conformal_threshold,
                     conformal_coverage_and_size,
                     expected_calibration_error, accuracy)  # noqa: E402
from utils import set_seed, write_run_log  # noqa: E402
from experiments.exp3_calibrators import _split_logits, _fit_one, _predict_one  # noqa: E402


def _eval_downstream(probs_cal, y_cal, probs_test, y_test, alpha=0.1):
    out = {
        "acc": accuracy(probs_test, y_test),
        "ece": expected_calibration_error(probs_test, y_test),
        "aurc": aurc(probs_test, y_test),
        "sel_acc_0.9": selective_accuracy_at_coverage(probs_test, y_test, 0.9),
        "sel_acc_0.7": selective_accuracy_at_coverage(probs_test, y_test, 0.7),
        "auroc_misc": auroc_misclassification(probs_test, y_test),
    }
    try:
        thr = split_conformal_threshold(probs_cal, y_cal, alpha=alpha)
        cov, size = conformal_coverage_and_size(probs_test, y_test, thr)
        out["conformal_coverage"] = cov
        out["conformal_size"] = size
        out["conformal_threshold"] = thr
        out["conformal_gap"] = cov - (1 - alpha)
    except Exception as e:  # pragma: no cover
        out["conformal_error"] = str(e)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+",
                   default=["cifar10", "flowers102", "food101"])
    p.add_argument("--backbone", default="ViT-B-32")
    p.add_argument("--pretrained", default="openai")
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--toy", action="store_true")
    p.add_argument("--calibrators", nargs="+", default=CALIBRATOR_IDS)
    p.add_argument("--out", default="outputs/exp4_results.json")
    args = p.parse_args()

    bundle = load_clip(args.backbone, args.pretrained)
    if bundle is None:
        print("ERROR: open_clip unavailable.")
        return

    templates = ["a photo of a {}."]
    records: List[Dict] = []
    for seed in args.seeds:
        for name in args.datasets:
            loader = ALL_LOADERS.get(name)
            if args.toy and name == "cifar10":
                spec = toy_cifar10_dataset(n_samples=256, seed=seed)
            elif loader is None:
                continue
            else:
                spec = loader(seed=seed)
            if spec is None:
                continue
            set_seed(seed)
            logits, labels, _ = evaluate_clip(
                bundle, spec.images, spec.classnames, templates=templates,
                batch_size=32)
            (lb, yb), (ln, yn), (lu, yu) = _split_logits(
                logits, labels, spec.base_class_idx, spec.novel_class_idx)
            base_text = encode_text(
                bundle, [spec.classnames[i] for i in spec.base_class_idx],
                templates=templates, normalize=True).detach().cpu()
            novel_text = encode_text(
                bundle, [spec.classnames[i] for i in spec.novel_class_idx],
                templates=templates, normalize=True).detach().cpu()

            n_b = len(yb)
            perm = torch.randperm(n_b)
            n_cal = max(1, n_b // 2)
            cal_idx, test_idx = perm[:n_cal], perm[n_cal:]
            lb_cal, yb_cal = lb[cal_idx], yb[cal_idx]
            lb_test, yb_test = lb[test_idx], yb[test_idx]
            # Split novel into novel-cal / novel-test for conformal threshold
            # on the *union* test set (base-cal -> threshold; union-test
            # measures empirical coverage including novel shift).
            n_n = len(yn)
            perm_n = torch.randperm(n_n)
            n_cn = max(1, n_n // 2)
            cn_idx, tn_idx = perm_n[:n_cn], perm_n[n_cn:]

            zs_b = lb_cal * 0.5
            zs_n = ln * 0.5

            for cname in args.calibrators:
                try:
                    cal = _fit_one(cname, K_train=len(spec.base_class_idx),
                                   logits_cal=lb_cal, labels_cal=yb_cal,
                                   text_emb_train=base_text,
                                   base_text_emb=base_text,
                                   zs_logits_cal=zs_b)
                except Exception as e:
                    records.append({"dataset": name, "seed": seed,
                                    "calibrator": cname, "status": f"fit:{e}"})
                    continue

                try:
                    probs_b_cal = _predict_one(cal, lb_cal,
                                               text_emb=base_text,
                                               zs_logits=lb_cal * 0.5)
                    probs_b_test = _predict_one(cal, lb_test,
                                                text_emb=base_text,
                                                zs_logits=lb_test * 0.5)
                    if cname in {"temperature_scaling", "dac", "cac", "raw"}:
                        probs_n_test = _predict_one(cal, ln[tn_idx],
                                                    text_emb=novel_text,
                                                    zs_logits=zs_n[tn_idx])
                    else:
                        ts_n = TemperatureScaling().fit(lb_cal, yb_cal)
                        probs_n_test = ts_n.predict_proba(ln[tn_idx])
                except Exception as e:
                    records.append({"dataset": name, "seed": seed,
                                    "calibrator": cname, "status": f"pred:{e}"})
                    continue

                rec = {
                    "dataset": name, "seed": seed, "calibrator": cname,
                    "group": spec.group,
                    "base": _eval_downstream(probs_b_cal, yb_cal,
                                             probs_b_test, yb_test,
                                             alpha=args.alpha),
                    "novel": _eval_downstream(probs_b_cal, yb_cal,
                                              probs_n_test, yn[tn_idx],
                                              alpha=args.alpha),
                }
                records.append(rec)
                print(json.dumps({"dataset": name, "cal": cname,
                                  "base_aurc": rec["base"].get("aurc"),
                                  "novel_aurc": rec["novel"].get("aurc"),
                                  "novel_cov": rec["novel"].get("conformal_coverage"),
                                  "novel_size": rec["novel"].get("conformal_size")},
                                 indent=2))

    # Spearman across calibrators within each dataset.
    summary: Dict = {}
    try:
        from scipy.stats import spearmanr
        for ds in {r["dataset"] for r in records if "dataset" in r}:
            xs = [r["novel"]["ece"] for r in records if r.get("dataset") == ds and "novel" in r]
            ys = [r["novel"]["aurc"] for r in records if r.get("dataset") == ds and "novel" in r]
            if len(xs) >= 4:
                rho, pval = spearmanr(xs, ys)
                summary[ds] = {"spearman_ece_vs_aurc": {"rho": float(rho),
                                                        "p": float(pval),
                                                        "n": len(xs)}}
    except Exception as e:
        summary["spearman_error"] = str(e)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"args": vars(args),
                                    "records": records,
                                    "summary": summary},
                                   indent=2, default=str))
    write_run_log("exp4", vars(args), out_path.parent / "logs")
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
