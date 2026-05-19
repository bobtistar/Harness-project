"""Drive Exp1–Exp4 sequentially.

Defaults are tuned for *sanity-check* speed (single seed, single small
dataset).  For headline numbers, override CLI args, e.g.::

    python run_all.py --datasets flowers102 food101 fgvc_aircraft \
        oxford_pets caltech101 eurosat --seeds 0 1 2 \
        --backbone ViT-B-16 --pretrained laion2b_s34b_b88k
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiments import exp1_diagnosis, exp2_geometry, exp3_calibrators, exp4_downstream  # noqa


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["cifar10"])
    p.add_argument("--backbone", default="ViT-B-32")
    p.add_argument("--pretrained", default="openai")
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--toy", action="store_true",
                   help="Use 128-sample CIFAR-10 slice (sanity).")
    p.add_argument("--out_dir", default="outputs")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    common = ["--datasets", *args.datasets,
              "--backbone", args.backbone,
              "--pretrained", args.pretrained,
              "--seeds", *[str(s) for s in args.seeds]]
    if args.toy:
        common.append("--toy")

    for mod, name in [(exp1_diagnosis, "exp1"), (exp2_geometry, "exp2"),
                      (exp3_calibrators, "exp3"), (exp4_downstream, "exp4")]:
        # Build argv per script.
        if name == "exp2":
            # exp2 uses --seed (singular) — use first seed only.
            argv = ["--datasets", *args.datasets,
                    "--backbone", args.backbone,
                    "--pretrained", args.pretrained,
                    "--seed", str(args.seeds[0]),
                    "--out", str(out_dir / f"{name}_results.json")]
        else:
            argv = common + ["--out", str(out_dir / f"{name}_results.json")]
        if args.toy:
            argv = list(argv)
            if "--toy" not in argv:
                argv.append("--toy")
        old_argv = sys.argv
        sys.argv = [name] + argv
        try:
            print(f"\n=== {name} ===")
            mod.main()
        except SystemExit:
            pass
        finally:
            sys.argv = old_argv


if __name__ == "__main__":
    main()
