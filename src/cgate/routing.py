"""Quality-conditioned expert routing -- the component under test.

The hypothesis: experts degrade differently as image quality drops, so fusion
weights conditioned on measured IQMs should beat fixed weights, and the gain
should concentrate in the low-quality stratum.

Model
-----
For each finding j, expert weights come from a softmax-linear map of the
standardised quality vector q:

    w_j(q) = softmax(W_j q + b_j)

trained to minimise the NLL of the fused prediction. This is deliberately the
smallest model that can express the hypothesis: it has E x (F + 1) parameters
per finding (~120 in total here), which a few hundred validation cases can fit
without overfitting, and its weights are directly readable -- "as CJV rises,
weight shifts from the ViT to radiomics" is a sentence you can defend in a viva
and check against the coefficient matrix.

Why initialisation matters
--------------------------
W and b start at exactly zero, so w_j(q) starts uniform and the untrained router
*is* the uniform heterogeneous ensemble (ablation B4). Any measured improvement
of B5 over B4 is therefore attributable to what training learned from quality
features, not to a different fusion family. If routing adds nothing, the
optimiser has the option of staying at the initialisation and the two rows tie.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .fusion import check_expert_probs, combine
from .schema import CLASS_COUNTS, N_FINDINGS

_EPS = 1e-12


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.clip(e.sum(axis=-1, keepdims=True), _EPS, None)


@dataclass
class RouterConfig:
    l2: float = 1e-2
    entropy_reg: float = 0.0
    lr: float = 5e-2
    epochs: int = 400
    patience: int = 40
    per_finding: bool = True
    seed: int = 0
    verbose: bool = False


class QualityRouter:
    """Softmax-linear router over standardised quality features."""

    def __init__(self, n_experts: int, n_features: int, config: RouterConfig | None = None):
        if n_experts < 1:
            raise ValueError("need at least one expert")
        self.n_experts = n_experts
        self.n_features = n_features
        self.config = config or RouterConfig()
        n_groups = N_FINDINGS if self.config.per_finding else 1
        self.n_groups = n_groups
        # Zero init => uniform weights => identical to the uniform ensemble.
        self.W = np.zeros((n_groups, n_experts, n_features), dtype=np.float64)
        self.b = np.zeros((n_groups, n_experts), dtype=np.float64)
        self.history_: list[dict[str, float]] = []
        self.fit_log_: dict = {}
        self.fitted_: bool = False

    # -- forward -----------------------------------------------------------
    def _group(self, j: int) -> int:
        return j if self.config.per_finding else 0

    def weights(self, q: np.ndarray) -> np.ndarray:
        """Expert weights, shape (n_cases, K, E)."""
        q = np.asarray(q, dtype=np.float64)
        if q.ndim != 2 or q.shape[1] != self.n_features:
            raise ValueError(
                f"quality features must have shape (n, {self.n_features}), got {q.shape}"
            )
        n = q.shape[0]
        out = np.zeros((n, N_FINDINGS, self.n_experts))
        for j in range(N_FINDINGS):
            g = self._group(j)
            out[:, j, :] = _softmax(q @ self.W[g].T + self.b[g])
        return out

    def fuse(
        self, q: np.ndarray, expert_probs: list[list[np.ndarray]], pool: str = "linear"
    ) -> list[np.ndarray]:
        return combine(expert_probs, self.weights(q), pool=pool)

    # -- training ----------------------------------------------------------
    def fit(
        self,
        q: np.ndarray,
        expert_probs: list[list[np.ndarray]],
        labels: np.ndarray,
        q_val: np.ndarray | None = None,
        expert_probs_val: list[list[np.ndarray]] | None = None,
        labels_val: np.ndarray | None = None,
    ) -> QualityRouter:
        """Fit on the calibration split.

        Must not be the split the experts were trained on: an expert that has
        memorised its training set looks uniformly excellent there, so the
        router would learn that quality is irrelevant. `cgate.pipeline` passes
        the dedicated calibration split.
        """
        cfg = self.config
        q = np.asarray(q, dtype=np.float64)
        n_experts, n = check_expert_probs(expert_probs)
        if n_experts != self.n_experts:
            raise ValueError(f"router built for {self.n_experts} experts, got {n_experts}")
        if q.shape[0] != n:
            raise ValueError(f"quality has {q.shape[0]} rows, expert probs have {n}")
        labels = np.asarray(labels, dtype=np.int64)

        # Pre-stack per finding: (n, C_j, E). Done once, reused every epoch.
        stacks = [
            np.stack([expert_probs[e][j] for e in range(n_experts)], axis=2)
            for j in range(N_FINDINGS)
        ]

        use_val = (
            q_val is not None and expert_probs_val is not None and labels_val is not None
        )
        if use_val:
            stacks_val = [
                np.stack([expert_probs_val[e][j] for e in range(n_experts)], axis=2)
                for j in range(N_FINDINGS)
            ]
            q_val = np.asarray(q_val, dtype=np.float64)
            labels_val = np.asarray(labels_val, dtype=np.int64)

        # Capacity check. The router has n_groups * n_experts * (n_features + 1)
        # parameters and is fitted on the calibration split, which is small by
        # construction. With per-finding weights that is 5 * 3 * 8 = 120
        # parameters against a few dozen cases -- more parameters than data, and
        # it shows up as *lower accuracy* rather than as a training-loss
        # failure, because a softmax mixture can fit calibration noise while
        # picking worse experts on new cases.
        n_params = int(self.W.size + self.b.size)
        self.fit_log_ = {
            "n_params": n_params,
            "n_samples": int(n),
            "params_per_sample": round(n_params / max(n, 1), 2),
            "overparameterised": bool(n_params > n),
        }
        if cfg.verbose and n_params > n:
            print(
                f"[router] WARNING: {n_params} parameters fitted on {n} cases. "
                f"Consider RouterConfig(per_finding=False) "
                f"({n_params // max(self.n_groups, 1)} params) or a larger l2."
            )

        mW = np.zeros_like(self.W)
        vW = np.zeros_like(self.W)
        mb = np.zeros_like(self.b)
        vb = np.zeros_like(self.b)
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        best_loss = np.inf
        best_state = (self.W.copy(), self.b.copy())
        bad_epochs = 0

        for epoch in range(1, cfg.epochs + 1):
            gW = np.zeros_like(self.W)
            gb = np.zeros_like(self.b)
            train_loss = 0.0

            for j in range(N_FINDINGS):
                g = self._group(j)
                z = q @ self.W[g].T + self.b[g]  # (n, E)
                w = _softmax(z)
                stack = stacks[j]  # (n, C, E)
                fused = np.einsum("nce,ne->nc", stack, w)  # (n, C)
                y = labels[:, j]
                rows = np.arange(n)
                py = np.clip(fused[rows, y], _EPS, None)
                train_loss += float(-np.mean(np.log(py)))

                # dL/dw_e = -p_e[y] / p_fused[y]
                dldw = -stack[rows, y, :] / py[:, None]  # (n, E)

                if cfg.entropy_reg > 0.0:
                    # Encourage the router to keep several experts alive rather
                    # than collapsing onto one; collapse would make the gate
                    # depend on a single model's failure modes.
                    logw = np.log(np.clip(w, _EPS, None))
                    dldw = dldw + cfg.entropy_reg * (logw + 1.0)

                # Backprop through softmax.
                gz = w * (dldw - np.sum(w * dldw, axis=1, keepdims=True))
                gW[g] += gz.T @ q / n
                gb[g] += gz.mean(axis=0)

            gW += cfg.l2 * self.W

            # Adam
            mW = beta1 * mW + (1 - beta1) * gW
            vW = beta2 * vW + (1 - beta2) * gW**2
            mb = beta1 * mb + (1 - beta1) * gb
            vb = beta2 * vb + (1 - beta2) * gb**2
            mW_hat = mW / (1 - beta1**epoch)
            vW_hat = vW / (1 - beta2**epoch)
            mb_hat = mb / (1 - beta1**epoch)
            vb_hat = vb / (1 - beta2**epoch)
            self.W -= cfg.lr * mW_hat / (np.sqrt(vW_hat) + eps)
            self.b -= cfg.lr * mb_hat / (np.sqrt(vb_hat) + eps)

            train_loss /= N_FINDINGS
            if use_val:
                monitor = self._loss(q_val, stacks_val, labels_val)
            else:
                monitor = train_loss
            self.history_.append({"epoch": epoch, "train": train_loss, "monitor": monitor})

            if monitor < best_loss - 1e-6:
                best_loss = monitor
                best_state = (self.W.copy(), self.b.copy())
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= cfg.patience:
                    if cfg.verbose:
                        print(f"[router] early stop at epoch {epoch}")
                    break

        # Always keep the best-monitored parameters, never the last ones.
        self.W, self.b = best_state
        self.fitted_ = True
        return self

    def _loss(
        self, q: np.ndarray, stacks: list[np.ndarray], labels: np.ndarray
    ) -> float:
        total = 0.0
        n = q.shape[0]
        rows = np.arange(n)
        for j in range(N_FINDINGS):
            g = self._group(j)
            w = _softmax(q @ self.W[g].T + self.b[g])
            fused = np.einsum("nce,ne->nc", stacks[j], w)
            py = np.clip(fused[rows, labels[:, j]], _EPS, None)
            total += float(-np.mean(np.log(py)))
        return total / N_FINDINGS

    # -- interpretation ----------------------------------------------------
    def coefficient_table(
        self, feature_names: tuple[str, ...], expert_names: tuple[str, ...]
    ) -> dict[str, dict[str, dict[str, float]]]:
        """Readable view of what the router learned.

        Positive coefficient on (finding, expert, feature) means: as that
        quality feature rises, weight shifts toward that expert. This is the
        table to inspect when claiming the router learned something sensible
        rather than noise.
        """
        from .schema import FINDING_NAMES

        out: dict[str, dict[str, dict[str, float]]] = {}
        for j, fname in enumerate(FINDING_NAMES):
            g = self._group(j)
            per_expert: dict[str, dict[str, float]] = {}
            for e, ename in enumerate(expert_names):
                per_expert[ename] = {
                    feat: float(self.W[g][e, k]) for k, feat in enumerate(feature_names)
                }
                per_expert[ename]["_bias"] = float(self.b[g][e])
            out[fname] = per_expert
            if not self.config.per_finding:
                # Weights are shared, so one group describes every finding.
                for other in FINDING_NAMES:
                    out[other] = per_expert
                break
        return out

    def mean_weights(self, q: np.ndarray) -> np.ndarray:
        """Average expert weight per finding, shape (K, E) -- a quick sanity view."""
        return self.weights(q).mean(axis=0)
