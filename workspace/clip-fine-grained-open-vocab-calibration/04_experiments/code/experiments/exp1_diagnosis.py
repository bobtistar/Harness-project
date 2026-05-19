"""Exp1 (RQ1): fine-grained vs coarse Base→Novel ΔECE.

Runs zero-shot CLIP (+ optional temperature scaling fitted on base val)
across a configurable list of datasets, splits them into base and novel
classes, and outputs a per-dataset CSV plus a paired Wilcoxon test
comparing the fine-grained and coarse groups.

Usage::
    python -m experiments.exp1_diagnosis --datasets cifar10 \
            --backbone ViT-B-32 --pretrained openai --toy
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

# Local imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data import ALL_LOADERS, toy_cifar10_dataset  # noqa: E402
from models import load_clip, evaluate_clip, OPENAI_TEMPLATES  # noqa: E402
from calibrators import TemperatureScaling, Raw  # noqa: E402
from metrics import summarize  # noqa: E402
from utils import set_seed, write_run_log  # noqa: E402


def _split_probs_by_class(logits, labels, base_idx, novel_idx):
    """Return ``(probs_base, labels_base), (probs_novel, labels_novel)``.

    Selects the appropriate class-subset columns and remaps labels to local
    indices, following CoOp's base-to-novel evaluation protocol.
    """
    base_idx_t = torch.tensor(base_idx, dtype=torch.long)
    novel_idx_t = torch.tensor(novel_idx, dtype=torch.long)
    mask_base = torch.isin(labels, base_idx_t)
    mask_novel = torch.isin(labels, novel_idx_t)

    def _remap(mask, keep_idx):
        sub_logits = logits[mask][:, keep_idx]
        # Re-index labels into [0, len(keep_idx))
        remap = {int(c): i for i, c in enumerate(keep_idx)}
        lbl = torch.tensor([remap[int(y)] for y in labels[mask]],
                           dtype=torch.long)
        return sub_logits, lbl

    return _remap(mask_base, base_idx), _remap(mask_novel, novel_idx)


def run_dataset(name: str, loader_fn, bundle, seed: int,
                templates: List[str], toy: bool = False) -> Dict:
    """Evaluate one dataset and return base/novel/union ECE."""
    set_seed(seed)
    if toy and name == "cifar10":
        spec = toy_cifar10_dataset(n_samples=128, seed=seed)
    else:
        spec = loader_fn(seed=seed)
    if spec is None:
        return {"dataset": name, "status": "missing"}
    # Pre-compute zero-shot logits.
    try:
        logits, labels, _ = evaluate_clip(
            bundle, spec.images, spec.classnames,
            templates=templates, batch_size=32,
        )
    except Exception as e:  # pragma: no cover
        warnings.warn(f"{name}: forward failed -> {e}")
        return {"dataset": name, "status": f"error:{e}"}

    (lb, yb), (ln, yn) = _split_probs_by_class(
        logits, labels, spec.base_class_idx, spec.novel_class_idx)
    if len(yb) == 0 or len(yn) == 0:
        return {"dataset": name, "status": "empty-split"}

    # Raw probs.
    raw_base = torch.softmax(lb, dim=-1)
    raw_novel = torch.softmax(ln, dim=-1)
    out: Dict = {
        "dataset": name,
        "group": spec.group,
        "n_classes": spec.num_classes,
        "n_base_classes": len(spec.base_class_idx),
        "n_novel_classes": len(spec.novel_class_idx),
        "n_base_samples": int(len(yb)),
        "n_novel_samples": int(len(yn)),
        "raw_base": summarize(raw_base, yb),
        "raw_novel": summarize(raw_novel, yn),
    }
    out["raw_delta_ece"] = out["raw_novel"]["ece"] - out["raw_base"]["ece"]

    # Temperature scaling fit on (held-out half of) base classes.
    n_b = len(yb)
    perm = torch.randperm(n_b)
    n_cal = max(1, n_b // 2)
    cal_idx, test_idx = perm[:n_cal], perm[n_cal:]
    ts = TemperatureScaling()
    ts.fit(lb[cal_idx], yb[cal_idx])
    ts_base = torch.softmax(ts.predict_logits(lb[test_idx]), dim=-1)
    ts_novel = torch.softmax(ts.predict_logits(ln), dim=-1)
    out["ts_T"] = float(ts.T.item())
    out["ts_base"] = summarize(ts_base, yb[test_idx])
    out["ts_novel"] = summarize(ts_novel, yn)
    out["ts_delta_ece"] = out["ts_novel"]["ece"] - out["ts_base"]["ece"]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+",
                   default=["cifar10", "flowers102", "food101", "fgvc_aircraft",
                            "oxford_pets", "caltech101", "eurosat"])
    p.add_argument("--backbone", default="ViT-B-32")
    p.add_argument("--pretrained", default="openai")
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--templates", choices=["single", "ensemble"], default="single")
    p.add_argument("--toy", action="store_true",
                   help="Use toy CIFAR-10 subset for sanity-check.")
    p.add_argument("--out", default="outputs/exp1_results.json")
    args = p.parse_args()

    bundle = load_clip(args.backbone, args.pretrained)
    if bundle is None:
        print("ERROR: open_clip unavailable.")
        return

    templates = OPENAI_TEMPLATES if args.templates == "ensemble" else [
        "a photo of a {}."
    ]

    results: List[Dict] = []
    for seed in args.seeds:
        for name in args.datasets:
            loader = ALL_LOADERS.get(name)
            if loader is None and name != "cifar10":
                warnings.warn(f"Unknown dataset {name}, skipping.")
                continue
            r = run_dataset(name, loader, bundle, seed, templates, toy=args.toy)
            r["seed"] = seed
            results.append(r)
            print(json.dumps(r, indent=2, default=str))

    # Aggregate group-level statistics.
    fg = [r for r in results
          if isinstance(r, dict) and r.get("group") == "fine-grained"
          and "raw_delta_ece" in r]
    cs = [r for r in results
          if isinstance(r, dict) and r.get("group") == "coarse"
          and "raw_delta_ece" in r]
    summary = {
        "fine_grained_delta_ece_mean": float(np.mean([r["raw_delta_ece"] for r in fg]))
        if fg else None,
        "coarse_delta_ece_mean": float(np.mean([r["raw_delta_ece"] for r in cs]))
        if cs else None,
        "n_fine_grained_datasets": len(fg),
        "n_coarse_datasets": len(cs),
    }
    if fg and cs:
        try:
            from scipy.stats import mannwhitneyu
            stat, p_val = mannwhitneyu([r["raw_delta_ece"] for r in fg],
                                       [r["raw_delta_ece"] for r in cs],
                                       alternative="greater")
            summary["mannwhitney_p"] = float(p_val)
        except Exception as e:
            summary["mannwhitney_p"] = f"err:{e}"

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"args": vars(args), "results": results, "summary": summary}
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    write_run_log("exp1", vars(args), out_path.parent / "logs")
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
