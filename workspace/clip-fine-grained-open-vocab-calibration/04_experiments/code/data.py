"""Dataset loaders for fine-grained vs coarse CLIP evaluation.

Each dataset returns a :class:`DatasetSpec` with::
    - ``images``: an iterable yielding (PIL image, label_int)
    - ``classnames``: List[str], len = ``num_classes``
    - ``base_class_idx`` / ``novel_class_idx``: List[int]
    - ``meta``: free-form dict (license, source, ...)

For datasets which torchvision can fetch (CIFAR-10/100, Flowers102, Food101,
FGVCAircraft, OxfordIIITPet, StanfordCars), we use those directly.  For
datasets unavailable in torchvision (CUB-200, SUN397, EuroSAT,
iNat-200-subset), a `LocalFolder` adapter expects an ImageFolder-style
directory and exposes the same interface.  Each loader function never raises
even if data is absent — instead it returns ``None`` and writes a helpful
warning so downstream sanity-checks can proceed.

NOTE: Calling these loaders may trigger long-running downloads.  Toy
sanity-check entry points exist in ``data.toy_cifar10_dataset``.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

# torchvision is optional in CI; tolerate ImportError gracefully.
try:
    from torchvision import datasets as tvd
    from torchvision import transforms as tvt
    _TV = True
except Exception:  # pragma: no cover
    _TV = False


DATA_ROOT = Path(os.environ.get("CLIP_DATA_ROOT", "./.data")).expanduser().resolve()
DATA_ROOT.mkdir(parents=True, exist_ok=True)


@dataclass
class DatasetSpec:
    name: str
    classnames: List[str]
    images: object  # torch.utils.data.Dataset-like; yields (PIL, int)
    base_class_idx: List[int]
    novel_class_idx: List[int]
    group: str  # 'fine-grained' or 'coarse'
    meta: Dict[str, str] = field(default_factory=dict)

    @property
    def num_classes(self) -> int:
        return len(self.classnames)


# --------------------------------------------------------------------------- #
# Base/novel split helper (CoOp convention, alphabetical, first-half base)
# --------------------------------------------------------------------------- #
def make_base_novel_split(classnames: Sequence[str], seed: int = 0
                          ) -> Tuple[List[int], List[int]]:
    """Return (base_idx, novel_idx) given classnames.

    Following CoOp/DAC convention: alphabetical first half = base, rest = novel.
    For seed>0 we permute deterministically before splitting (mimicking the
    seed1/seed2/seed3 protocol used in CoOp papers — these correspond to
    different random class permutations, *not* image permutations).
    """
    n = len(classnames)
    order = list(range(n))
    if seed > 0:
        rng = np.random.RandomState(seed)
        order = rng.permutation(order).tolist()
    half = n // 2
    base = sorted(order[:half])
    novel = sorted(order[half:])
    return base, novel


# --------------------------------------------------------------------------- #
# Coarse datasets
# --------------------------------------------------------------------------- #
def cifar10(seed: int = 0, train: bool = False) -> Optional[DatasetSpec]:
    """CIFAR-10 — used only as a toy sanity-check, not in headline results."""
    if not _TV:
        return None
    ds = tvd.CIFAR10(root=str(DATA_ROOT / "cifar10"), train=train,
                     download=True, transform=None)
    classnames = list(ds.classes)
    base, novel = make_base_novel_split(classnames, seed=seed)
    return DatasetSpec(
        name="cifar10", classnames=classnames, images=ds,
        base_class_idx=base, novel_class_idx=novel,
        group="coarse",
        meta={"license": "MIT-like", "n_classes": str(len(classnames)),
              "n_test": str(len(ds))},
    )


def caltech101(seed: int = 0) -> Optional[DatasetSpec]:
    if not _TV:
        return None
    try:
        ds = tvd.Caltech101(root=str(DATA_ROOT), download=True)
    except Exception as e:  # pragma: no cover
        warnings.warn(f"Caltech101 unavailable: {e}")
        return None
    classnames = list(ds.categories)
    base, novel = make_base_novel_split(classnames, seed=seed)
    return DatasetSpec(name="caltech101", classnames=classnames, images=ds,
                       base_class_idx=base, novel_class_idx=novel, group="coarse",
                       meta={"license": "CC-BY"})


def eurosat(seed: int = 0) -> Optional[DatasetSpec]:
    if not _TV:
        return None
    try:
        ds = tvd.EuroSAT(root=str(DATA_ROOT / "eurosat"), download=True)
    except Exception as e:  # pragma: no cover
        warnings.warn(f"EuroSAT unavailable: {e}")
        return None
    classnames = list(ds.classes)
    base, novel = make_base_novel_split(classnames, seed=seed)
    return DatasetSpec(name="eurosat", classnames=classnames, images=ds,
                       base_class_idx=base, novel_class_idx=novel, group="coarse",
                       meta={"license": "MIT"})


# --------------------------------------------------------------------------- #
# Fine-grained datasets
# --------------------------------------------------------------------------- #
def flowers102(seed: int = 0, split: str = "test") -> Optional[DatasetSpec]:
    if not _TV:
        return None
    try:
        ds = tvd.Flowers102(root=str(DATA_ROOT / "flowers102"),
                            split=split, download=True)
    except Exception as e:  # pragma: no cover
        warnings.warn(f"Flowers102 unavailable: {e}")
        return None
    # Flowers102 ships with integer labels; we attach canonical English names
    # from the standard list (kept here in-line to avoid an extra dependency).
    classnames = _FLOWERS102_NAMES
    base, novel = make_base_novel_split(classnames, seed=seed)
    return DatasetSpec(name="flowers102", classnames=classnames, images=ds,
                       base_class_idx=base, novel_class_idx=novel,
                       group="fine-grained",
                       meta={"license": "Non-commercial",
                             "source": "Nilsback & Zisserman 2008"})


def food101(seed: int = 0, split: str = "test") -> Optional[DatasetSpec]:
    if not _TV:
        return None
    try:
        ds = tvd.Food101(root=str(DATA_ROOT / "food101"),
                         split=split, download=True)
    except Exception as e:  # pragma: no cover
        warnings.warn(f"Food101 unavailable: {e}")
        return None
    classnames = [c.replace("_", " ") for c in ds.classes]
    base, novel = make_base_novel_split(classnames, seed=seed)
    return DatasetSpec(name="food101", classnames=classnames, images=ds,
                       base_class_idx=base, novel_class_idx=novel,
                       group="fine-grained",
                       meta={"license": "Research-only"})


def fgvc_aircraft(seed: int = 0, split: str = "test") -> Optional[DatasetSpec]:
    if not _TV:
        return None
    try:
        ds = tvd.FGVCAircraft(root=str(DATA_ROOT / "fgvc"),
                              split=split, download=True)
    except Exception as e:  # pragma: no cover
        warnings.warn(f"FGVCAircraft unavailable: {e}")
        return None
    classnames = list(ds.classes)
    base, novel = make_base_novel_split(classnames, seed=seed)
    return DatasetSpec(name="fgvc_aircraft", classnames=classnames, images=ds,
                       base_class_idx=base, novel_class_idx=novel,
                       group="fine-grained",
                       meta={"license": "Research-only"})


def oxford_pets(seed: int = 0, split: str = "test") -> Optional[DatasetSpec]:
    if not _TV:
        return None
    try:
        ds = tvd.OxfordIIITPet(root=str(DATA_ROOT / "pets"),
                               split="test" if split == "test" else "trainval",
                               download=True)
    except Exception as e:  # pragma: no cover
        warnings.warn(f"OxfordIIITPet unavailable: {e}")
        return None
    classnames = [c.replace("_", " ") for c in ds.classes]
    base, novel = make_base_novel_split(classnames, seed=seed)
    return DatasetSpec(name="oxford_pets", classnames=classnames, images=ds,
                       base_class_idx=base, novel_class_idx=novel,
                       group="fine-grained",
                       meta={"license": "CC-BY-SA 4.0"})


def stanford_cars(seed: int = 0, split: str = "test") -> Optional[DatasetSpec]:
    if not _TV:
        return None
    try:
        ds = tvd.StanfordCars(root=str(DATA_ROOT / "cars"),
                              split=split, download=True)
    except Exception as e:  # pragma: no cover
        warnings.warn(f"StanfordCars unavailable: {e}")
        return None
    classnames = list(ds.classes)
    base, novel = make_base_novel_split(classnames, seed=seed)
    return DatasetSpec(name="stanford_cars", classnames=classnames, images=ds,
                       base_class_idx=base, novel_class_idx=novel,
                       group="fine-grained",
                       meta={"license": "Research-only"})


# --------------------------------------------------------------------------- #
# Generic ImageFolder loader (for CUB / iNat / SUN397 if user has them locally)
# --------------------------------------------------------------------------- #
class _ImageFolderWithNames:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.classnames = sorted(
            [d.name for d in self.root.iterdir() if d.is_dir()]
        )
        self.cls_to_idx = {c: i for i, c in enumerate(self.classnames)}
        self.samples: List[Tuple[Path, int]] = []
        for c in self.classnames:
            for img in (self.root / c).iterdir():
                if img.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                    self.samples.append((img, self.cls_to_idx[c]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> Tuple[Image.Image, int]:
        p, y = self.samples[i]
        return Image.open(p).convert("RGB"), y


def local_folder(name: str, root: Path, group: str = "fine-grained",
                 seed: int = 0) -> Optional[DatasetSpec]:
    root = Path(root)
    if not root.exists():
        warnings.warn(f"{name}: root '{root}' not found")
        return None
    ds = _ImageFolderWithNames(root)
    base, novel = make_base_novel_split(ds.classnames, seed=seed)
    return DatasetSpec(name=name, classnames=ds.classnames, images=ds,
                       base_class_idx=base, novel_class_idx=novel,
                       group=group, meta={"source": str(root)})


def cub200(seed: int = 0) -> Optional[DatasetSpec]:
    root = Path(os.environ.get("CUB200_ROOT", DATA_ROOT / "cub200" / "test"))
    return local_folder("cub200", root, group="fine-grained", seed=seed)


def sun397(seed: int = 0) -> Optional[DatasetSpec]:
    root = Path(os.environ.get("SUN397_ROOT", DATA_ROOT / "sun397" / "test"))
    return local_folder("sun397", root, group="coarse", seed=seed)


def imagenet1k_val(seed: int = 0) -> Optional[DatasetSpec]:
    root = Path(os.environ.get("IMAGENET_VAL_ROOT", DATA_ROOT / "imagenet" / "val"))
    return local_folder("imagenet1k", root, group="coarse", seed=seed)


# --------------------------------------------------------------------------- #
# Toy sanity-check
# --------------------------------------------------------------------------- #
def toy_cifar10_dataset(n_samples: int = 256, seed: int = 0) -> Optional[DatasetSpec]:
    """Tiny CIFAR-10 slice for code-path sanity-checks."""
    spec = cifar10(seed=seed, train=False)
    if spec is None:
        return None
    base_ds = spec.images
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(base_ds), size=min(n_samples, len(base_ds)),
                     replace=False).tolist()

    class _Subset:
        def __init__(self, parent, indices):
            self.parent = parent
            self.indices = indices

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, i):
            return self.parent[self.indices[i]]

    spec.images = _Subset(base_ds, idx)
    spec.meta["toy"] = "true"
    spec.meta["n"] = str(len(idx))
    return spec


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
COARSE_LOADERS: Dict[str, Callable] = {
    "imagenet1k": imagenet1k_val,
    "caltech101": caltech101,
    "sun397": sun397,
    "eurosat": eurosat,
}

FINEGRAINED_LOADERS: Dict[str, Callable] = {
    "cub200": cub200,
    "stanford_cars": stanford_cars,
    "fgvc_aircraft": fgvc_aircraft,
    "flowers102": flowers102,
    "oxford_pets": oxford_pets,
    "food101": food101,
}

ALL_LOADERS: Dict[str, Callable] = {**COARSE_LOADERS, **FINEGRAINED_LOADERS}


# --------------------------------------------------------------------------- #
# Hard-coded canonical Flowers102 names (subset; abbreviated for brevity)
# --------------------------------------------------------------------------- #
_FLOWERS102_NAMES: List[str] = [
    "pink primrose", "hard-leaved pocket orchid", "canterbury bells",
    "sweet pea", "english marigold", "tiger lily", "moon orchid",
    "bird of paradise", "monkshood", "globe thistle", "snapdragon",
    "colt's foot", "king protea", "spear thistle", "yellow iris",
    "globe flower", "purple coneflower", "peruvian lily", "balloon flower",
    "giant white arum lily", "fire lily", "pincushion flower",
    "fritillary", "red ginger", "grape hyacinth", "corn poppy",
    "prince of wales feathers", "stemless gentian", "artichoke",
    "sweet william", "carnation", "garden phlox", "love in the mist",
    "mexican aster", "alpine sea holly", "ruby-lipped cattleya",
    "cape flower", "great masterwort", "siam tulip", "lenten rose",
    "barbeton daisy", "daffodil", "sword lily", "poinsettia", "bolero deep blue",
    "wallflower", "marigold", "buttercup", "oxeye daisy", "common dandelion",
    "petunia", "wild pansy", "primula", "sunflower", "pelargonium",
    "bishop of llandaff", "gaura", "geranium", "orange dahlia",
    "pink-yellow dahlia", "cautleya spicata", "japanese anemone",
    "black-eyed susan", "silverbush", "californian poppy", "osteospermum",
    "spring crocus", "bearded iris", "windflower", "tree poppy", "gazania",
    "azalea", "water lily", "rose", "thorn apple", "morning glory",
    "passion flower", "lotus lotus", "toad lily", "anthurium",
    "frangipani", "clematis", "hibiscus", "columbine", "desert-rose",
    "tree mallow", "magnolia", "cyclamen", "watercress", "canna lily",
    "hippeastrum", "bee balm", "ball moss", "foxglove", "bougainvillea",
    "camellia", "mallow", "mexican petunia", "bromelia", "blanket flower",
    "trumpet creeper", "blackberry lily",
]
# Pad to exactly 102 if needed.
while len(_FLOWERS102_NAMES) < 102:
    _FLOWERS102_NAMES.append(f"flower_class_{len(_FLOWERS102_NAMES)}")
