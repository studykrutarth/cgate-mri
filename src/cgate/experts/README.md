# Experts

Every expert obeys one contract, whatever its internals:

```python
expert.fit(x: np.ndarray, labels: np.ndarray) -> Expert
expert.predict_proba(x: np.ndarray) -> list[np.ndarray]   # K arrays, one per finding
expert.view: str    # which feature view it consumes: 'texture' | 'local' | 'global'
expert.name: str
```

`predict_proba` must return one probability simplex per finding in
`cgate.schema.FINDING_SCHEMA`, with `probs[j].shape == (n_cases, n_classes_j)`
and rows summing to 1. `MultiFindingExpert` handles this for anything that looks
like a scikit-learn classifier.

That contract is the only reason a gradient-boosted radiomics model and a neural
network can be fused at all — see `docs/` and `cgate/schema.py`.

## Current experts

### Real backbones — `--experts torch`

| Class | View | Model |
|---|---|---|
| `TextureRadiomicsExpert` | `texture` | HistGradientBoosting — the PyRadiomics + XGBoost expert |
| `CNNExpert` | `volume` | ResNet on 2-D slices, via timm |
| `ViTExpert` | `volume` | ViT-S/16 on 2-D slices, via timm |
| `MCDropoutExpert` | `volume` | ResNet with dropout active at inference, N passes averaged |

### scikit-learn surrogates — `--experts surrogate`

| Class | View | Model |
|---|---|---|
| `LocalPatternExpert` | `local` | MLP — CNN stand-in |
| `GlobalContextExpert` | `global` | MLP — ViT stand-in |
| `DropoutEnsembleExpert` | `local` | bagged MLPs + feature dropout — MC-dropout stand-in |
| `HomogeneousEnsemble` | `local` | 3× MLP, different seeds — deep-ensemble control |

Kept because they run anywhere with no torch and iterate in seconds, which is
useful when working on the routing/calibration/gate logic. **They are not
evidence about CNN or ViT behaviour.** Their feature views were hand-designed to
degrade under different defects, so any conclusion about routing drawn from them
is partly a statement about that design.

## How pixels reach the expert interface

The interface is tabular — `fit(x, labels)` where `x` is a feature matrix — but a
CNN needs images. The `volume` view resolves this: it is a single column of row
indices into a `SliceCache` (`cgate/data/slices.py`), built during ingest. The
expert turns those rows into pixels itself, and no other component changes.

Slice selection is **label-free by construction**: `select_slice_indices` takes
only `(volumes, config)` and there is a test asserting it has no mask parameter.
Choosing "the slice with the biggest tumour" would read a mask that does not
exist at inference and would inflate every reported number.

## Adding another torch expert

Subclass `TorchSliceExpert` and set `arch`, or implement the contract directly.
Three things to keep:

- **Reuse `align_proba`** for anything scikit-learn-shaped. A class absent from a
  training split yields a too-narrow probability array that silently misaligns
  against label indices.
- **Respect `PROB_FLOOR`.** No expert may emit exactly zero — a hard zero makes
  NLL infinite and poisons temperature scaling and every calibration metric.
- **Do not flip left-right in augmentation.** It mirrors the hemisphere and
  inverts the `laterality` label while the target stays put. `TorchExpertConfig`
  flips anterior-posterior only.

### Windows note

`torch.set_num_threads(1)` is applied on first use. scikit-learn/scipy load
Intel MKL's OpenMP runtime and torch loads its own; with both spawning thread
teams in one process, torch's backward pass dies with an access violation. Since
this pipeline runs a scikit-learn radiomics expert alongside a torch CNN, that
is the normal configuration. Capping torch's threads avoids it without the
global `KMP_DUPLICATE_LIB_OK=TRUE` workaround, which Intel documents as able to
crash or silently produce wrong results. Set `TorchExpertConfig.num_threads=None`
to opt out.
