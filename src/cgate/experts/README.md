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

| Class | View | Model | Real or surrogate |
|---|---|---|---|
| `TextureRadiomicsExpert` | `texture` | HistGradientBoosting | **real** — the PyRadiomics + XGBoost expert |
| `LocalPatternExpert` | `local` | MLP | **surrogate** for the CNN |
| `GlobalContextExpert` | `global` | MLP | **surrogate** for the ViT |
| `DropoutEnsembleExpert` | `local` | bagged MLPs + feature dropout | **surrogate** for MC dropout |
| `HomogeneousEnsemble` | `local` | 3× MLP, different seeds | **real** deep ensemble |

### Why surrogates exist

`torch` has no wheel for the Python 3.14 interpreter this repo was developed on.
Rather than block the whole project on that, the CNN and ViT are stood in by MLPs
consuming deliberately different hand-built feature views:

- `texture` — first-order and gradient statistics. Noise destroys it first.
- `local` — block-mean grid ("what is bright, and where"). Bias field breaks it.
- `global` — axis marginals and moments. Blur and ghosting break it.

The views are chosen so the three experts **fail under different acquisition
defects**, which is the phenomenon the router is supposed to exploit. Without
that, the ensemble would be three correlated models and routing could not help
by construction.

**Nothing in the code claims these are CNNs or ViTs.** They are named
`cnn_surrogate` / `vit_surrogate`, and every table that reports them says so.

## Adding the real torch experts

Create `cgate/experts/torch_experts.py`:

```python
class CNNExpert:
    name = "cnn"
    view = "volume"          # a new view returning the array itself

    def fit(self, x, labels): ...
    def predict_proba(self, x): ...   # -> list[np.ndarray], one per finding
```

Then:

1. Add a `volume` view in `cgate/data/features.py` (or bypass views and load
   volumes lazily by patient id, if memory forces it).
2. Register the class in `EXPERT_REGISTRY` and swap it into
   `build_default_experts`.
3. Nothing else changes — pipeline, router, calibration, gate and the ablation
   grid are all written against the contract above.

Two things to keep when you do:

- **Reuse `align_proba`.** scikit-learn and torch both give you only the classes
  seen in training; a rare finding absent from a split produces a
  too-narrow probability array that silently misaligns against label indices.
- **Respect `PROB_FLOOR`.** No expert may emit exactly zero for any class — a
  hard zero makes NLL infinite and poisons temperature scaling and every
  calibration metric downstream.

For genuine MC dropout, keep dropout layers active at inference and average N
stochastic forward passes; that replaces `DropoutEnsembleExpert` and the
ablation table keeps the same row (B2).
