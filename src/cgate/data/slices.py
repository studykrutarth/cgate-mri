"""On-disk slice cache — the data path for the CNN and ViT experts.

Why a cache exists
------------------
`ingest_cases` reduces each case to feature vectors and discards the volume,
which is right for radiomics and useless for a network that needs pixels. And
volumes cannot simply be kept in memory: BraTS 2021 is 1251 x 4 x 240 x 240 x 155
float32, roughly 550 GB.

So slices are extracted once, normalised, and written to a float16 memmap. The
memmap is opened lazily, so RAM stays flat regardless of cohort size, and the
tabular views used by the radiomics expert are untouched.

    cache/index.json          patient ids, shape, config, per-case slice indices
    cache/cases/<pid>.npy     (n_slices, n_modalities, H, W) float16

One file per case rather than one big array: the case count is not known until
the stream is exhausted, and buffering every stack to stack them at the end
would need ~9 GB of RAM for full BraTS — reintroducing the exact problem the
cache exists to solve. Per-case files are written as cases arrive, memory stays
flat, and an interrupted build can be resumed.

Slices are stored at native resolution. Resizing to the backbone's input size
happens in the Dataset, because the resize target depends on the model and
storing at 224 would inflate the phantom cache twentyfold for no benefit.

The leakage trap
----------------
The tempting way to pick slices is "the one with the largest tumour cross
section". That reads the segmentation mask, which does not exist at inference
time, and it inflates measured accuracy with information the deployed system
will never have — the same class of bug that `assert_no_leakage` guards against
for splits, and just as invisible in aggregate metrics.

Every selection strategy here is therefore **label-free**: it may look at the
image, never at the mask. `SliceCacheConfig.selection` documents which.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

SELECTION_STRATEGIES = ("geometric", "brain_area")


@dataclass
class SliceCacheConfig:
    """How slices are chosen and stored.

    n_slices        slices kept per case
    selection       'geometric'  evenly spaced through the central band
                    'brain_area' ranked by non-zero brain area (image-only)
    central_fraction  fraction of the z axis to sample from, for 'geometric'.
                    The extreme slices of a brain volume are mostly empty, so
                    sampling the full range wastes most of the budget.
    normalise       per-case, per-modality z-score inside the brain
    """

    n_slices: int = 16
    selection: str = "geometric"
    central_fraction: float = 0.6
    slice_axis: int = 2
    normalise: bool = True

    def __post_init__(self) -> None:
        if self.selection not in SELECTION_STRATEGIES:
            raise ValueError(
                f"unknown selection {self.selection!r}; expected one of "
                f"{SELECTION_STRATEGIES}"
            )
        if self.n_slices < 1:
            raise ValueError("n_slices must be >= 1")
        if not 0.0 < self.central_fraction <= 1.0:
            raise ValueError("central_fraction must be in (0, 1]")


def select_slice_indices(
    volumes: np.ndarray, config: SliceCacheConfig
) -> np.ndarray:
    """Choose slice indices from the image alone. Never sees the mask.

    Returns exactly `config.n_slices` indices, sorted ascending. If the volume
    has fewer slices than requested, indices repeat rather than the case being
    dropped — a short volume is still a valid case.
    """
    depth = volumes.shape[1 + config.slice_axis]

    if config.selection == "geometric":
        margin = (1.0 - config.central_fraction) / 2.0
        lo = int(np.floor(depth * margin))
        hi = int(np.ceil(depth * (1.0 - margin))) - 1
        lo, hi = max(0, lo), max(0, min(depth - 1, hi))
        if hi <= lo:
            lo, hi = 0, depth - 1
        idx = np.linspace(lo, hi, config.n_slices)
        return np.clip(np.round(idx).astype(int), 0, depth - 1)

    # brain_area: rank by how much signal each slice carries, image-only.
    areas = np.zeros(depth, dtype=np.float64)
    for z in range(depth):
        sl = np.take(volumes, z, axis=1 + config.slice_axis)
        areas[z] = float(np.count_nonzero(sl))
    order = np.argsort(-areas)[: config.n_slices]
    if order.size < config.n_slices:
        order = np.resize(order, config.n_slices)
    return np.sort(order)


def _extract(volumes: np.ndarray, indices: np.ndarray, config: SliceCacheConfig) -> np.ndarray:
    """(n_slices, n_modalities, H, W) for the given slice indices."""
    out = []
    for z in indices:
        sl = np.take(volumes, int(z), axis=1 + config.slice_axis)  # (C, H, W)
        out.append(sl)
    return np.stack(out, axis=0)


def _normalise(stack: np.ndarray) -> np.ndarray:
    """Per-modality z-score over brain voxels of the whole case.

    Normalising per case rather than per slice keeps relative intensity between
    slices meaningful; normalising over brain voxels rather than all voxels
    stops the large zero background from dominating the statistics.
    """
    out = stack.astype(np.float32, copy=True)
    n_mod = out.shape[1]
    for c in range(n_mod):
        vals = out[:, c]
        brain = vals[vals > 0]
        if brain.size < 8:
            continue
        mean = float(brain.mean())
        std = float(brain.std())
        if std < 1e-6:
            continue
        # Background stays at zero after normalising so the network can still
        # tell brain from air.
        mask = vals > 0
        vals[mask] = (vals[mask] - mean) / std
        out[:, c] = vals
    return out


def _case_path(root: Path, patient_id: str) -> Path:
    # Patient ids come from directory names; keep them filesystem-safe.
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(patient_id))
    return Path(root) / "cases" / f"{safe}.npy"


class SliceCacheWriter:
    """Incremental cache builder.

    Separate from `SliceCache.build` so the pipeline can feed cases through a
    single ingest pass — the stream is consumable once, and it also has to
    compute IQMs, feature views and findings from the same volumes.
    """

    def __init__(self, root: Path, config: SliceCacheConfig | None = None):
        self.root = Path(root)
        self.config = config or SliceCacheConfig()
        (self.root / "cases").mkdir(parents=True, exist_ok=True)
        self.patient_ids: list[str] = []
        self.slice_indices: list[list[int]] = []
        self.shape: tuple[int, ...] | None = None

    def add(self, case) -> None:
        vols = np.asarray(case.volumes)
        if vols.ndim != 4:
            raise ValueError(
                f"{case.patient_id}: expected (C, H, W, D) volumes, got {vols.shape}"
            )
        idx = select_slice_indices(vols, self.config)
        stack = _extract(vols, idx, self.config)
        if self.config.normalise:
            stack = _normalise(stack)

        if self.shape is None:
            self.shape = stack.shape
        elif stack.shape != self.shape:
            raise ValueError(
                f"{case.patient_id}: slice stack {stack.shape} does not match "
                f"{self.shape}. All cases must share a geometry — register them first."
            )

        np.save(_case_path(self.root, case.patient_id), stack.astype(np.float16))
        self.patient_ids.append(str(case.patient_id))
        self.slice_indices.append([int(z) for z in idx])

    def close(self) -> "SliceCache":
        if not self.patient_ids:
            raise ValueError("no cases were written to the slice cache")
        index = {
            "patient_ids": self.patient_ids,
            "shape": [len(self.patient_ids), *self.shape],
            "slice_indices": self.slice_indices,
            "config": asdict(self.config),
            # Fresh on every build, and recorded by any case table saved
            # alongside. Patient ids alone cannot tell two caches apart: a
            # second ingest into the same directory overwrites the .npy files
            # while keeping every id, so the table would reconnect happily and
            # train on one study's pixels under another's labels. Only an
            # identity stamped at write time catches that.
            "build_id": uuid.uuid4().hex,
        }
        (self.root / "index.json").write_text(json.dumps(index, indent=2), encoding="utf8")
        return SliceCache.open(self.root)


class SliceCache:
    """Lazily-memmapped slice store, one file per case."""

    def __init__(self, root: Path, index: dict):
        self.root = Path(root)
        self.index = index
        self.patient_ids: list[str] = index["patient_ids"]
        self._row_of = {pid: i for i, pid in enumerate(self.patient_ids)}
        # Memmaps are opened on first use and kept: reopening per sample would
        # dominate training time, and a memmap costs almost nothing until read.
        self._maps: dict[int, np.ndarray] = {}

    # -- construction ------------------------------------------------------
    @staticmethod
    def build(cases, root: Path, config: SliceCacheConfig | None = None) -> "SliceCache":
        """Stream cases into a cache and return it opened.

        `cases` is any iterable of objects with `.patient_id` and `.volumes`,
        i.e. both SynthCase and BratsCase.
        """
        writer = SliceCacheWriter(root, config)
        for case in cases:
            writer.add(case)
        if not writer.patient_ids:
            raise ValueError("no cases supplied to SliceCache.build")
        return writer.close()

    @staticmethod
    def open(root: Path) -> "SliceCache":
        root = Path(root)
        index_path = root / "index.json"
        if not index_path.exists():
            raise FileNotFoundError(
                f"no slice cache at {root}. Build one with SliceCache.build(...)"
            )
        index = json.loads(index_path.read_text(encoding="utf8"))
        return SliceCache(root, index)

    # -- access ------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.patient_ids)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.index["shape"])

    @property
    def n_slices(self) -> int:
        return int(self.index["shape"][1])

    @property
    def n_modalities(self) -> int:
        return int(self.index["shape"][2])

    @property
    def build_id(self) -> str | None:
        """Identity of the build that wrote this cache, or None if predates it."""
        return self.index.get("build_id")

    def rows_for(self, patient_ids) -> np.ndarray:
        """Map patient ids to cache rows, failing loudly on a miss.

        A silent miss would train the network on the wrong images while the
        labels stayed correct, which is close to undetectable downstream.
        """
        rows = []
        for pid in patient_ids:
            key = str(pid)
            if key not in self._row_of:
                raise KeyError(f"patient {key!r} is not in the slice cache at {self.root}")
            rows.append(self._row_of[key])
        return np.asarray(rows, dtype=np.int64)

    def _memmap(self, row: int):
        """The open memmap for one case, without materialising it."""
        row = int(row)
        mm = self._maps.get(row)
        if mm is None:
            path = _case_path(self.root, self.patient_ids[row])
            if not path.exists():
                raise FileNotFoundError(f"missing cached slices for row {row}: {path}")
            mm = np.load(path, mmap_mode="r")
            self._maps[row] = mm
        return mm

    def get(self, row: int) -> np.ndarray:
        """(n_slices, n_modalities, H, W) float32 for one case."""
        return np.asarray(self._memmap(row), dtype=np.float32)

    def get_slice(self, row: int, index: int) -> np.ndarray:
        """(n_modalities, H, W) float32 for one slice of one case.

        Indexes the memmap before converting, which is the whole point: `get`
        materialises every slice of the case as float32 and callers that wanted
        one slice threw away the rest. Training does that once per sample and
        inference once per (case, slice) pair, so each case was being converted
        n_slices times over -- roughly a 16x amplification at the default, and
        it left the GPU idle about 95% of the time waiting for it.
        """
        return np.asarray(self._memmap(row)[int(index)], dtype=np.float32)
