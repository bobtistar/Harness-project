"""Exp2 (RQ2): inter-class text similarity τ_txt as a driver of mis-calibration.

(i) measures τ_txt per dataset, (ii) regresses ECE on τ_txt (Spearman + partial),
(iii) intervenes via PCA whitening / orthogonalisation of the text-embedding
matrix and re-computes ECE on the *same* image features.
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
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data import ALL_LOADERS, toy_cifar10_dataset  # noqa: E402
from models import (load_clip, encode_text, encode_images, zero_shot_logits,
                    tau_text, pca_whiten, orthogonalize)  # noqa: E402
from metrics import expected_calibration_error, summarize  # noqa: E402
from utils import set_seed, write_run_log  # noqa: E402


def evaluate_with_text(bundle, spec, text_emb, batch_size: int = 32):
    img_emb, labels = encode_images(bundle, spec.images, batch_size=batch_size)
    scale = float(bundle.model.logit_scale.exp().detach().cpu().item())
    logits = zero_shot_logits(img_emb, text_emb.cpu(), logit_scale=scale)
    probs = torch.softmax(logits, dim=-1)
    return probs, labels


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+",
                   default=["cifar10", "flowers102", "food101",
                            "fgvc_aircraft", "oxford_pets",
                            "caltech101", "eurosat"])
    p.add_argument("--backbone", default="ViT-B-32")
    p.add_argument("--pretrained", default="openai")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--toy", action="store_true")
    p.add_argument("--out", default="outputs/exp2_results.json")
    args = p.parse_args()
    set_seed(args.seed)

    bundle = load_clip(args.backbone, args.pretrained)
    if bundle is None:
        print("ERROR: open_clip unavailable.")
        return

    rows: List[Dict] = []
    for name in args.datasets:
        loader = ALL_LOADERS.get(name)
        if args.toy and name == "cifar10":
            spec = toy_cifar10_dataset(n_samples=128, seed=args.seed)
        elif loader is None:
            continue
        else:
            spec = loader(seed=args.seed)
        if spec is None:
            continue

        # Original text embedding.
        text_emb = encode_text(bundle, spec.classnames,
                               templates=["a photo of a {}."], normalize=True)
        text_emb = text_emb.detach().cpu()

        # Image features (computed once).
        img_emb, labels = encode_images(bundle, spec.images, batch_size=32)
        scale = float(bundle.model.logit_scale.exp().detach().cpu().item())

        # Three variants of the text-embedding matrix.
        variants = {
            "raw": text_emb,
            "pca_whiten": pca_whiten(text_emb),
            "orthogonalize": orthogonalize(text_emb),
        }
        row: Dict = {"dataset": name, "group": spec.group,
                     "n_classes": spec.num_classes,
                     "n_samples": int(len(labels))}
        for vname, t in variants.items():
            logits = zero_shot_logits(img_emb, t, logit_scale=scale)
            probs = torch.softmax(logits, dim=-1)
            stats = summarize(probs, labels)
            stats["tau_text"] = tau_text(t)
            row[vname] = stats
        row["delta_ece_pca"] = row["raw"]["ece"] - row["pca_whiten"]["ece"]
        row["delta_ece_orth"] = row["raw"]["ece"] - row["orthogonalize"]["ece"]
        rows.append(row)
        print(json.dumps({k: row[k] if not isinstance(row[k], dict)
                          else {"ece": row[k]["ece"], "tau_text": row[k]["tau_text"]}
                          for k in row}, indent=2))

    # Correlations.
    summary: Dict = {}
    if len(rows) >= 4:
        try:
            from scipy.stats import spearmanr
            tau_vals = [r["raw"]["tau_text"] for r in rows]
            ece_vals = [r["raw"]["ece"] for r in rows]
            rho, pval = spearmanr(tau_vals, ece_vals)
            summary["spearman_tau_vs_ece"] = {"rho": float(rho),
                                              "p": float(pval),
                                              "n": len(rows)}
            # Partial correlation conditioning on accuracy.
            acc_vals = [r["raw"]["acc"] for r in rows]
            try:
                from scipy.stats import spearmanr as _sp
                resid_tau = np.array(tau_vals) - np.mean(tau_vals)
                resid_ece = np.array(ece_vals) - np.mean(ece_vals)
                resid_acc = np.array(acc_vals) - np.mean(acc_vals)
                # Fit linear residuals manually.
                b1 = float(np.dot(resid_tau, resid_acc) / max(np.dot(resid_acc, resid_acc), 1e-9))
                b2 = float(np.dot(resid_ece, resid_acc) / max(np.dot(resid_acc, resid_acc), 1e-9))
                r1 = resid_tau - b1 * resid_acc
                r2 = resid_ece - b2 * resid_acc
                rho_p, p_p = _sp(r1, r2)
                summary["partial_spearman_given_acc"] = {"rho": float(rho_p),
                                                         "p": float(p_p)}
            except Exception as e:
                summary["partial_spearman_given_acc"] = f"err:{e}"
        except Exception as e:
            summary["spearman_tau_vs_ece"] = f"err:{e}"

    if rows:
        pca_deltas = [r["delta_ece_pca"] for r in rows]
        summary["mean_delta_ece_pca"] = float(np.mean(pca_deltas))
        summary["sign_test_pca_positive"] = int(sum(1 for d in pca_deltas if d > 0))
        summary["sign_test_pca_n"] = len(pca_deltas)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"args": vars(args),
                                    "rows": rows,
                                    "summary": summary}, indent=2, default=str))
    write_run_log("exp2", vars(args), out_path.parent / "logs")
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
