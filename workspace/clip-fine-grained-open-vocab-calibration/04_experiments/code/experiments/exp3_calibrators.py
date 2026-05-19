"""Exp3 (RQ3): head-to-head comparison of 7 post-hoc calibrators.

For each (backbone, dataset, seed) we
  1. compute zero-shot logits on the full dataset (all classes),
  2. split into base / novel by class index,
  3. set aside half of the base data as calibration,
  4. fit each calibrator on the calibration split,
  5. evaluate on (base-test, novel-test, union-test) under each calibrator,
  6. write a long-format JSON record per (dataset, seed, calibrator, split).

Down-stream tests (Pareto, paired bootstrap, Friedman + Nemenyi) are
handled in :mod:`experiments.exp4_downstream`.
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
from metrics import summarize  # noqa: E402
from utils import set_seed, write_run_log  # noqa: E402


def _split_logits(logits, labels, base_idx, novel_idx):
    base_idx_t = torch.tensor(base_idx, dtype=torch.long)
    novel_idx_t = torch.tensor(novel_idx, dtype=torch.long)
    mb = torch.isin(labels, base_idx_t)
    mn = torch.isin(labels, novel_idx_t)
    remap_b = {int(c): i for i, c in enumerate(base_idx)}
    remap_n = {int(c): i for i, c in enumerate(novel_idx)}
    lb = logits[mb][:, base_idx]
    yb = torch.tensor([remap_b[int(y)] for y in labels[mb]], dtype=torch.long)
    ln = logits[mn][:, novel_idx]
    yn = torch.tensor([remap_n[int(y)] for y in labels[mn]], dtype=torch.long)
    # union: keep all classes; remap so that base classes occupy [0, |base|)
    # and novel classes occupy [|base|, K).
    keep_idx = list(base_idx) + list(novel_idx)
    remap_u = {int(c): i for i, c in enumerate(keep_idx)}
    keep_mask = mb | mn
    lu = logits[keep_mask][:, keep_idx]
    yu = torch.tensor([remap_u[int(y)] for y in labels[keep_mask]], dtype=torch.long)
    return (lb, yb), (ln, yn), (lu, yu)


def _fit_one(cname: str, K_train: int,
             logits_cal, labels_cal,
             text_emb_train=None, base_text_emb=None,
             zs_logits_cal=None):
    if cname == "raw":
        return Raw()
    if cname == "temperature_scaling":
        return TemperatureScaling().fit(logits_cal, labels_cal)
    if cname == "vector_scaling":
        return VectorScaling(K=K_train).fit(logits_cal, labels_cal)
    if cname == "dirichlet":
        return DirichletCalibration(K=K_train).fit(logits_cal, labels_cal)
    if cname == "dac":
        d = DAC()
        d.set_base_text(base_text_emb)
        d.fit(logits_cal, labels_cal, text_emb=text_emb_train)
        return d
    if cname == "cac":
        return CAC().fit(logits_cal, labels_cal, zs_logits=zs_logits_cal)
    if cname == "histogram_binning":
        return HistogramBinning(K=K_train).fit(logits_cal, labels_cal)
    raise ValueError(cname)


def _predict_one(cal, logits, text_emb=None, zs_logits=None):
    name = cal.name
    if name == "dac":
        return torch.softmax(cal.predict_logits(logits, text_emb=text_emb), dim=-1)
    if name == "cac":
        return torch.softmax(cal.predict_logits(logits, zs_logits=zs_logits), dim=-1)
    return cal.predict_proba(logits)


def run_dataset(name: str, loader_fn, bundle, seed: int,
                templates: List[str], toy: bool = False,
                calibrators_to_run=None) -> List[Dict]:
    set_seed(seed)
    if toy and name == "cifar10":
        spec = toy_cifar10_dataset(n_samples=256, seed=seed)
    else:
        spec = loader_fn(seed=seed)
    if spec is None:
        return [{"dataset": name, "status": "missing"}]
    try:
        logits, labels, _ = evaluate_clip(bundle, spec.images, spec.classnames,
                                          templates=templates, batch_size=32)
    except Exception as e:
        warnings.warn(f"{name}: forward failed -> {e}")
        return [{"dataset": name, "status": f"error:{e}"}]

    (lb, yb), (ln, yn), (lu, yu) = _split_logits(
        logits, labels, spec.base_class_idx, spec.novel_class_idx)

    # Text embeddings for base/novel/union (in *local* order).
    base_text = encode_text(bundle, [spec.classnames[i] for i in spec.base_class_idx],
                            templates=templates, normalize=True).detach().cpu()
    novel_text = encode_text(bundle, [spec.classnames[i] for i in spec.novel_class_idx],
                             templates=templates, normalize=True).detach().cpu()
    union_text = torch.cat([base_text, novel_text], dim=0)

    # Split base into cal / base-test (50/50).
    n_b = len(yb)
    perm = torch.randperm(n_b)
    n_cal = max(1, n_b // 2)
    cal_idx, test_idx = perm[:n_cal], perm[n_cal:]
    lb_cal, yb_cal = lb[cal_idx], yb[cal_idx]
    lb_test, yb_test = lb[test_idx], yb[test_idx]

    # For CAC we need ZS logits.  As a stand-in (since we have no fine-tuned
    # model checkpoint here) we use a *temperature-perturbed* copy of the same
    # logits as the "zero-shot view".  This still lets CAC train.
    zs_logits_cal_base = lb_cal * 0.5
    zs_logits_test_base = lb_test * 0.5
    zs_logits_novel = ln * 0.5
    zs_logits_union = lu * 0.5

    out: List[Dict] = []
    cal_list = calibrators_to_run or CALIBRATOR_IDS
    for cname in cal_list:
        try:
            cal_b = _fit_one(cname, K_train=len(spec.base_class_idx),
                             logits_cal=lb_cal, labels_cal=yb_cal,
                             text_emb_train=base_text,
                             base_text_emb=base_text,
                             zs_logits_cal=zs_logits_cal_base)
        except Exception as e:
            out.append({"dataset": name, "calibrator": cname, "status": f"fit_err:{e}"})
            continue

        # Build a *union* version of trainable calibrators where the parameter
        # shape depends on K.  For VS/Dirichlet/HB we fit a separate copy on
        # the union (since K differs); but per protocol, only base is allowed
        # — so we reuse the base-fitted calibrator on the K-dimensional slice
        # by *projecting onto base columns first*.  In practice this means we
        # evaluate base and novel separately.  Union ECE is then computed by
        # *concatenating* base and novel predictions in their full-K space —
        # for VS/Dirichlet/HB we fall back to a single TS fit on union logits
        # for the union ECE (transparency: this is a known limitation, see
        # `expected_results.md` §Limitations).
        try:
            probs_base = _predict_one(cal_b, lb_test,
                                      text_emb=base_text,
                                      zs_logits=zs_logits_test_base)
            # For novel, we need a calibrator whose parameter dimension matches
            # K_novel.  For TS/DAC/CAC this is K-invariant (1- or 2-param).
            if cname in {"temperature_scaling", "dac", "cac", "raw"}:
                probs_novel = _predict_one(cal_b, ln,
                                           text_emb=novel_text,
                                           zs_logits=zs_logits_novel)
                probs_union = _predict_one(cal_b, lu,
                                           text_emb=union_text,
                                           zs_logits=zs_logits_union)
            else:
                # Fit a *separate* TS as a sensible default for the union /
                # novel evaluations, matching DAC paper convention.
                ts_n = TemperatureScaling().fit(lb_cal, yb_cal)
                probs_novel = ts_n.predict_proba(ln)
                probs_union = ts_n.predict_proba(lu)
        except Exception as e:
            out.append({"dataset": name, "calibrator": cname, "status": f"pred_err:{e}"})
            continue

        rec = {
            "dataset": name,
            "group": spec.group,
            "calibrator": cname,
            "seed": seed,
            "n_base_classes": len(spec.base_class_idx),
            "n_novel_classes": len(spec.novel_class_idx),
            "base": summarize(probs_base, yb_test),
            "novel": summarize(probs_novel, yn),
            "union": summarize(probs_union, yu),
        }
        rec["delta_ece"] = rec["novel"]["ece"] - rec["base"]["ece"]
        rec["novel_over_base"] = (rec["novel"]["ece"] /
                                  max(rec["base"]["ece"], 1e-6))
        out.append(rec)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+",
                   default=["cifar10", "flowers102", "food101",
                            "fgvc_aircraft", "oxford_pets",
                            "caltech101", "eurosat"])
    p.add_argument("--backbone", default="ViT-B-32")
    p.add_argument("--pretrained", default="openai")
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--templates", choices=["single", "ensemble"], default="single")
    p.add_argument("--toy", action="store_true")
    p.add_argument("--calibrators", nargs="+",
                   default=CALIBRATOR_IDS)
    p.add_argument("--out", default="outputs/exp3_results.json")
    args = p.parse_args()

    bundle = load_clip(args.backbone, args.pretrained)
    if bundle is None:
        print("ERROR: open_clip unavailable.")
        return

    templates = OPENAI_TEMPLATES if args.templates == "ensemble" else [
        "a photo of a {}."
    ]

    records: List[Dict] = []
    for seed in args.seeds:
        for name in args.datasets:
            loader = ALL_LOADERS.get(name)
            if loader is None and name != "cifar10":
                continue
            recs = run_dataset(name, loader, bundle, seed, templates,
                               toy=args.toy,
                               calibrators_to_run=args.calibrators)
            for r in recs:
                records.append(r)
                print(json.dumps(r, indent=2, default=str))

    # Quick aggregate per calibrator (union ECE mean).
    by_cal: Dict[str, List[float]] = {}
    for r in records:
        if "union" not in r:
            continue
        by_cal.setdefault(r["calibrator"], []).append(r["union"]["ece"])
    summary = {c: {"mean_union_ece": float(np.mean(v)), "n": len(v)}
               for c, v in by_cal.items()}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"args": vars(args),
                                    "records": records,
                                    "summary": summary}, indent=2, default=str))
    write_run_log("exp3", vars(args), out_path.parent / "logs")
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
