#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_models.py -- Reliability predictor model zoo behind one interface.

RELIANT: Calibrated Reliability Prediction and Budget-Constrained Portfolio
Selection for Smart Contract Security Analyzers.

--------------------------------------------------------------------------------
ROLE IN THE ARTIFACT
--------------------------------------------------------------------------------
Given a contract's alert-free features (stage 03) and a query vulnerability class,
predict, for every analyzer in the panel, how reliably that analyzer will handle
that (contract, class) -- e.g. the probability it detects the vulnerability. These
predictions are the raw scores that stage 05 turns into distribution-free
guarantees and stage 06 turns into a budget-constrained tool portfolio.

The design follows the Algorithm Selection Problem lineage (Rice 1976; SATzilla,
Xu et al. 2008): we learn a per-analyzer empirical performance model rather than a
single vulnerability detector. Concretely, one sub-model is trained per tool, each
mapping [contract features (+) query-class one-hot] -> that tool's reliability.

Three predictors share one interface:
  * ConstantPredictor  base-rate reference (a sanity floor, and a safe fallback
                       for tool/class cells with too little training signal).
  * RidgePredictor     L2-regularized linear model on standardized features --
                       the faithful linear "empirical hardness model" analogue.
  * LGBMPredictor      gradient-boosted trees (the workhorse), classifier for the
                       binary detection target, regressor for continuous recall.
  * HeteroGNNPredictor optional heterogeneous message-passing over the stage-03
                       graph, with a LAZY torch import so the entire pipeline runs
                       CPU-only without torch installed. It is strictly skippable:
                       if torch is unavailable, fit raises and the trainer skips it.

--------------------------------------------------------------------------------
WHY THIS ADDRESSES THE PRIOR REVIEW
--------------------------------------------------------------------------------
The target is analyzer reliability, learned only from features that exist before
any tool runs (stage 03 forbids tool alerts as inputs), so the models cannot
recreate the earlier circularity. Predicting per-(tool, class) reliability -- not
"is this contract vulnerable" -- is also what makes the SATzilla-style baselines
in stage 09 the *task-appropriate* comparison the reviewers asked for, instead of
vulnerability detectors that solve a different problem.

--------------------------------------------------------------------------------
TARGETS
--------------------------------------------------------------------------------
Default target is `detected` (0/1): does the tool flag the ground-truth class on
the contract? Predictors emit a reliability score in [0,1] (a detection
probability). Continuous targets (`recall`, or `1 - fnr`) are also supported; the
sub-model type adapts automatically. Missing/undefined label cells (NaN) are
masked per tool during training and never fabricated.

--------------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------------
    python3 src/04_models.py --selftest              # hermetic, no torch needed

    # quick demo once stages 02/03 have produced artifacts:
    python3 src/04_models.py --demo \
        --features artifacts/features.parquet --labels artifacts/labels.parquet

Exit codes: 0 = success; 3 = usage error.
"""

from __future__ import annotations

import argparse
import json
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__version__ = "1.0.0"
SCHEMA_VERSION = "reliant-models-1"

# Must match stage 01 / 02 / 03 vocabularies.
CANONICAL_CLASSES: Tuple[str, ...] = (
    "arithmetic", "reentrancy", "timestamp_dependency",
    "transaction_order_dependency", "tx_origin",
    "unchecked_low_level_calls", "unhandled_exceptions",
)
NODE_TYPES: Tuple[str, ...] = ("contract", "function", "modifier", "event", "state_var")
EDGE_TYPES: Tuple[str, ...] = ("contains", "inherits", "calls", "uses", "emits", "reads_writes")
# Fixed key order for per-node numeric features (union across node types).
NODE_FEAT_KEYS: Tuple[str, ...] = (
    "is_library", "is_interface", "n_bases", "n_subnodes", "visibility",
    "is_payable", "is_view_pure", "is_constructor", "is_fallback", "n_params",
    "n_calls", "n_loops", "is_mapping", "is_array",
)

DEFAULT_TARGET = "detected"
_MIN_TRAIN = 4          # below this, fall back to a constant (base-rate) sub-model


# ==============================================================================
# Dataset assembly: (features + query class) -> X ; per-tool labels -> Y
# ==============================================================================
@dataclass
class Dataset:
    """A modelling view over instances = (contract, query-class) pairs.

    X          : (n, d) feature matrix -- contract features then class one-hot.
    Y          : (n, T) per-tool target, NaN where undefined/missing.
    feature_names, tool_names, class_names, instance_ids, groups (base_id).
    graphs     : optional list of stage-03 graph dicts aligned row-wise (for GNN).
    """
    X: np.ndarray
    Y: np.ndarray
    feature_names: List[str]
    tool_names: List[str]
    class_names: List[str]
    instance_ids: List[Tuple[str, str]]
    groups: np.ndarray
    graphs: Optional[List[dict]] = field(default=None)

    def __post_init__(self) -> None:
        assert self.X.shape[0] == self.Y.shape[0] == len(self.instance_ids) == len(self.groups)
        assert self.Y.shape[1] == len(self.tool_names)
        assert self.X.shape[1] == len(self.feature_names)

    @property
    def n(self) -> int:
        return self.X.shape[0]

    def subset(self, idx: Sequence[int]) -> "Dataset":
        idx = list(idx)
        return Dataset(
            X=self.X[idx], Y=self.Y[idx],
            feature_names=self.feature_names, tool_names=self.tool_names,
            class_names=self.class_names,
            instance_ids=[self.instance_ids[i] for i in idx],
            groups=self.groups[idx],
            graphs=([self.graphs[i] for i in idx] if self.graphs is not None else None),
        )


def _target_to_float(series: pd.Series) -> pd.Series:
    """Coerce a label column to float with NaN (handles nullable boolean/float)."""
    if pd.api.types.is_bool_dtype(series) or series.dtype == "boolean":
        return series.map({True: 1.0, False: 0.0}).astype(float)
    return pd.to_numeric(series, errors="coerce").astype(float)


def assemble_dataset(features: pd.DataFrame, labels: pd.DataFrame,
                     target: str = DEFAULT_TARGET,
                     tools: Optional[Sequence[str]] = None,
                     classes: Sequence[str] = CANONICAL_CLASSES,
                     graphs_by_id: Optional[Dict[str, dict]] = None) -> Dataset:
    """Build a Dataset from stage-03 features and stage-02 labels.

    An instance is a unique (contract_id, class_canonical) with ground truth. Its
    features are the contract's alert-free features plus a one-hot of the query
    class; its labels are the target value for each tool (NaN where a tool has no
    measured outcome for that cell).
    """
    classes = list(classes)
    feat_cols = [c for c in features.columns if c.startswith("f_")]
    if not feat_cols:
        raise ValueError("features frame has no f_* columns")

    lab = labels[labels["class_canonical"].isin(classes)].copy()
    lab["_t"] = _target_to_float(lab[target])
    tool_names = list(tools) if tools is not None else sorted(lab["tool"].unique())

    # Instances, in a deterministic order.
    inst = (lab[["contract_id", "class_canonical", "base_id"]]
            .drop_duplicates()
            .sort_values(["contract_id", "class_canonical"], kind="mergesort")
            .reset_index(drop=True))

    # Per-tool target matrix via pivot, reindexed to (instances x tools).
    piv = lab.pivot_table(index=["contract_id", "class_canonical"],
                          columns="tool", values="_t", aggfunc="first")
    piv = piv.reindex(columns=tool_names)
    key_index = pd.MultiIndex.from_frame(inst[["contract_id", "class_canonical"]])
    Y = piv.reindex(index=key_index).to_numpy(dtype=float)

    # Contract feature block (join on contract_id), then class one-hot.
    fidx = features.set_index("contract_id")
    missing = set(inst["contract_id"]) - set(fidx.index)
    if missing:
        raise ValueError(f"{len(missing)} instance contract(s) lack features, "
                         f"e.g. {sorted(missing)[0]}")
    feat_block = fidx.loc[inst["contract_id"], feat_cols].to_numpy(dtype=float)
    class_pos = {c: i for i, c in enumerate(classes)}
    onehot = np.zeros((len(inst), len(classes)), dtype=float)
    for r, c in enumerate(inst["class_canonical"]):
        onehot[r, class_pos[c]] = 1.0

    X = np.hstack([feat_block, onehot])
    feature_names = list(feat_cols) + [f"class_{c}" for c in classes]
    instance_ids = list(zip(inst["contract_id"], inst["class_canonical"]))
    groups = inst["base_id"].to_numpy()

    graphs = None
    if graphs_by_id is not None:
        graphs = [graphs_by_id.get(cid, {"n_nodes": 0, "nodes": [], "edges": [],
                                         "edge_types": list(EDGE_TYPES)})
                  for cid in inst["contract_id"]]

    return Dataset(X=X, Y=Y, feature_names=feature_names, tool_names=tool_names,
                   class_names=classes, instance_ids=instance_ids, groups=groups,
                   graphs=graphs)


def load_graphs_jsonl(path: Path) -> Dict[str, dict]:
    """Load stage-03 graphs.jsonl into {contract_id: graph}."""
    out: Dict[str, dict] = {}
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                g = json.loads(line)
                out[g["contract_id"]] = g
    return out


# ==============================================================================
# Predictor interface + tabular sub-model helpers
# ==============================================================================
def _clip01(a: np.ndarray) -> np.ndarray:
    return np.clip(a, 0.0, 1.0)


class _ConstSub:
    """A constant sub-model returning a fixed value (base rate)."""
    def __init__(self, value: float):
        self.value = float(value)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(X.shape[0], self.value, dtype=float)


class ReliabilityPredictor(ABC):
    """Predict a reliability score in [0,1] for each tool of the panel.

    fit(X, Y) with Y of shape (n, T) possibly containing NaN; predict(X) returns
    (n, T) in [0,1]. Subclasses that need the graph set requires_graphs = True and
    accept a `graphs=` argument.
    """
    name: str = "base"
    requires_graphs: bool = False

    def __init__(self) -> None:
        self.tool_names: List[str] = []
        self.n_features_: int = 0

    @abstractmethod
    def fit(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> "ReliabilityPredictor":
        ...

    @abstractmethod
    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        ...

    def fit_dataset(self, ds: Dataset) -> "ReliabilityPredictor":
        self.tool_names = list(ds.tool_names)
        kw = {"graphs": ds.graphs} if self.requires_graphs else {}
        return self.fit(ds.X, ds.Y, **kw)

    def predict_dataset(self, ds: Dataset) -> np.ndarray:
        kw = {"graphs": ds.graphs} if self.requires_graphs else {}
        return self.predict(ds.X, **kw)

    def save(self, path) -> None:
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path) -> "ReliabilityPredictor":
        with open(path, "rb") as fh:
            return pickle.load(fh)


class ConstantPredictor(ReliabilityPredictor):
    """Predict each tool's training base rate (a task-appropriate sanity floor)."""
    name = "constant"

    def fit(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> "ConstantPredictor":
        self.n_features_ = X.shape[1]
        self.rates_ = np.array([
            np.nanmean(Y[:, t]) if np.any(~np.isnan(Y[:, t])) else 0.5
            for t in range(Y.shape[1])], dtype=float)
        return self

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        return np.tile(_clip01(self.rates_), (X.shape[0], 1))


class RidgePredictor(ReliabilityPredictor):
    """Standardized L2 linear model per tool (SATzilla-faithful linear analogue)."""
    name = "ridge"

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def fit(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> "RidgePredictor":
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        self.n_features_ = X.shape[1]
        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X)
        self.models_: List[object] = []
        for t in range(Y.shape[1]):
            mask = ~np.isnan(Y[:, t])
            y = Y[mask, t]
            if mask.sum() < _MIN_TRAIN or np.unique(y).size < 2:
                self.models_.append(_ConstSub(y.mean() if y.size else 0.5))
            else:
                self.models_.append(Ridge(alpha=self.alpha).fit(Xs[mask], y))
        return self

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        Xs = self.scaler_.transform(X)
        cols = [_clip01(np.asarray(m.predict(Xs), dtype=float)) for m in self.models_]
        return np.vstack(cols).T


class LGBMPredictor(ReliabilityPredictor):
    """Gradient-boosted trees per tool; classifier for binary detection targets."""
    name = "lightgbm"

    def __init__(self, params: Optional[dict] = None, seed: int = 0):
        super().__init__()
        self.seed = seed
        self.params = params or dict(
            n_estimators=300, num_leaves=15, learning_rate=0.05,
            min_child_samples=5, subsample=0.9, subsample_freq=1,
            colsample_bytree=0.9, reg_lambda=1.0)

    def _binary(self, y: np.ndarray) -> bool:
        return np.all(np.isin(np.unique(y), (0.0, 1.0)))

    def fit(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> "LGBMPredictor":
        import lightgbm as lgb  # lazy: only needed if this predictor is used
        self.n_features_ = X.shape[1]
        common = dict(random_state=self.seed, n_jobs=1, verbosity=-1,
                      deterministic=True, force_row_wise=True, **self.params)
        self.models_: List[object] = []
        self.kinds_: List[str] = []
        for t in range(Y.shape[1]):
            mask = ~np.isnan(Y[:, t])
            y = Y[mask, t]
            if mask.sum() < _MIN_TRAIN or np.unique(y).size < 2:
                self.models_.append(_ConstSub(y.mean() if y.size else 0.5))
                self.kinds_.append("const")
            elif self._binary(y):
                m = lgb.LGBMClassifier(**common).fit(X[mask], y.astype(int))
                self.models_.append(m)
                self.kinds_.append("clf")
            else:
                m = lgb.LGBMRegressor(**common).fit(X[mask], y)
                self.models_.append(m)
                self.kinds_.append("reg")
        return self

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        cols = []
        for m, kind in zip(self.models_, self.kinds_):
            if kind == "clf":
                cols.append(_clip01(m.predict_proba(X)[:, 1]))
            else:
                cols.append(_clip01(np.asarray(m.predict(X), dtype=float)))
        return np.vstack(cols).T


# ==============================================================================
# Graph -> arrays (pure numpy; testable without torch)
# ==============================================================================
def graph_to_arrays(graph: dict) -> dict:
    """Convert a stage-03 graph dict to numpy arrays for the GNN.

    node_type : (N,)   int type ids
    node_feat : (N, F) float features = [type one-hot | NODE_FEAT_KEYS values]
    edge_index: (E, 2) int (src, dst)
    edge_type : (E,)   int edge-type ids
    """
    nodes = graph.get("nodes", [])
    n = len(nodes)
    f_dim = len(NODE_TYPES) + len(NODE_FEAT_KEYS)
    node_type = np.zeros(n, dtype=np.int64)
    node_feat = np.zeros((n, f_dim), dtype=np.float32)
    type_pos = {t: i for i, t in enumerate(NODE_TYPES)}
    for i, nd in enumerate(nodes):
        ti = type_pos.get(nd.get("ntype"), 0)
        node_type[i] = ti
        node_feat[i, ti] = 1.0
        feat = nd.get("feat", {}) or {}
        for j, k in enumerate(NODE_FEAT_KEYS):
            node_feat[i, len(NODE_TYPES) + j] = float(feat.get(k, 0) or 0)
    edges = graph.get("edges", [])
    if edges:
        arr = np.asarray(edges, dtype=np.int64)
        edge_index = arr[:, :2]
        edge_type = arr[:, 2]
    else:
        edge_index = np.zeros((0, 2), dtype=np.int64)
        edge_type = np.zeros((0,), dtype=np.int64)
    return {"node_type": node_type, "node_feat": node_feat,
            "edge_index": edge_index, "edge_type": edge_type, "n_nodes": n}


# ==============================================================================
# Optional heterogeneous GNN (LAZY torch; strictly skippable; server-only here)
# ==============================================================================
class HeteroGNNPredictor(ReliabilityPredictor):
    """Typed message-passing over the contract graph, then an MLP head per tool.

    Torch is imported lazily inside fit/predict. If torch is not installed, fit
    raises RuntimeError and the trainer (stage 07) simply skips this predictor --
    the rest of the pipeline is unaffected. Implemented in pure PyTorch (no
    torch_geometric dependency). Because a CPU sandbox may lack torch, this class
    is exercised end-to-end on the project server; here the graph->tensor path and
    the skip behaviour are unit-tested.
    """
    name = "hetero_gnn"
    requires_graphs = True

    def __init__(self, hidden: int = 32, layers: int = 2, epochs: int = 60,
                 lr: float = 1e-2, seed: int = 0):
        super().__init__()
        self.hidden = hidden
        self.layers = layers
        self.epochs = epochs
        self.lr = lr
        self.seed = seed
        self._module = None
        self._n_class = 0

    @staticmethod
    def _require_torch():
        try:
            import torch
            return torch
        except Exception as exc:  # pragma: no cover - torch absent in sandbox
            raise RuntimeError(
                "HeteroGNNPredictor requires PyTorch, which is not installed. "
                "Install torch to enable the optional GNN, or run without it "
                "(the Ridge/LightGBM pipeline is fully self-contained).") from exc

    def _build_module(self, torch, node_feat_dim: int, n_class: int, n_tools: int):
        nn = torch.nn

        class TypedMP(nn.Module):
            def __init__(self, dim, n_etypes):
                super().__init__()
                self.self_lin = nn.Linear(dim, dim)
                self.rel_lin = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n_etypes)])
                self.act = nn.ReLU()

            def forward(self, h, edge_index, edge_type):
                out = self.self_lin(h)
                for r in range(len(self.rel_lin)):
                    sel = (edge_type == r)
                    if sel.any():
                        src = edge_index[sel, 0]
                        dst = edge_index[sel, 1]
                        msg = self.rel_lin[r](h.index_select(0, src))
                        agg = torch.zeros_like(h)
                        agg.index_add_(0, dst, msg)
                        out = out + agg
                return self.act(out)

        class Net(nn.Module):
            def __init__(self, fdim, hidden, layers, n_etypes, n_class, n_tools):
                super().__init__()
                self.enc = nn.Linear(fdim, hidden)
                self.mp = nn.ModuleList([TypedMP(hidden, n_etypes) for _ in range(layers)])
                self.head = nn.Sequential(
                    nn.Linear(hidden + n_class, hidden), nn.ReLU(),
                    nn.Linear(hidden, n_tools))

            def forward(self, node_feat, edge_index, edge_type, class_vec):
                h = self.enc(node_feat)
                for layer in self.mp:
                    h = layer(h, edge_index, edge_type)
                g = h.mean(dim=0) if h.shape[0] > 0 else torch.zeros(h.shape[1])
                z = torch.cat([g, class_vec], dim=-1)
                return self.head(z)

        return Net(node_feat_dim, self.hidden, self.layers,
                   len(EDGE_TYPES), n_class, n_tools)

    def _to_tensors(self, torch, graph: dict):
        a = graph_to_arrays(graph)
        return (torch.tensor(a["node_feat"], dtype=torch.float32),
                torch.tensor(a["edge_index"], dtype=torch.long),
                torch.tensor(a["edge_type"], dtype=torch.long))

    def fit(self, X: np.ndarray, Y: np.ndarray, graphs: Optional[List[dict]] = None,
            **kwargs) -> "HeteroGNNPredictor":
        torch = self._require_torch()
        if graphs is None or len(graphs) != X.shape[0]:
            raise ValueError("HeteroGNNPredictor.fit requires one graph per row")
        torch.manual_seed(self.seed)
        self.n_features_ = X.shape[1]
        self._n_class = len(CANONICAL_CLASSES)
        class_mat = X[:, -self._n_class:]
        node_feat_dim = len(NODE_TYPES) + len(NODE_FEAT_KEYS)
        n_tools = Y.shape[1]
        self._module = self._build_module(torch, node_feat_dim, self._n_class, n_tools)
        opt = torch.optim.Adam(self._module.parameters(), lr=self.lr)
        Yt = torch.tensor(np.nan_to_num(Y, nan=0.0), dtype=torch.float32)
        Mt = torch.tensor(~np.isnan(Y), dtype=torch.float32)
        tensors = [self._to_tensors(torch, g) for g in graphs]
        cvecs = torch.tensor(class_mat, dtype=torch.float32)
        bce = torch.nn.BCEWithLogitsLoss(reduction="none")
        self._module.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            logits = torch.stack([
                self._module(nf, ei, et, cvecs[i])
                for i, (nf, ei, et) in enumerate(tensors)])
            loss = (bce(logits, Yt) * Mt).sum() / Mt.sum().clamp(min=1)
            loss.backward()
            opt.step()
        return self

    def predict(self, X: np.ndarray, graphs: Optional[List[dict]] = None,
                **kwargs) -> np.ndarray:
        torch = self._require_torch()
        if self._module is None:
            raise RuntimeError("predict called before fit")
        class_mat = X[:, -self._n_class:]
        cvecs = torch.tensor(class_mat, dtype=torch.float32)
        self._module.eval()
        out = []
        with torch.no_grad():
            for i, g in enumerate(graphs):
                nf, ei, et = self._to_tensors(torch, g)
                logits = self._module(nf, ei, et, cvecs[i])
                out.append(torch.sigmoid(logits).numpy())
        return _clip01(np.vstack(out))


# ==============================================================================
# Factory
# ==============================================================================
def build_predictor(name: str, **kwargs) -> ReliabilityPredictor:
    name = name.lower()
    if name == "constant":
        return ConstantPredictor()
    if name == "ridge":
        return RidgePredictor(**kwargs)
    if name in ("lightgbm", "lgbm"):
        return LGBMPredictor(**kwargs)
    if name in ("hetero_gnn", "gnn"):
        return HeteroGNNPredictor(**kwargs)
    raise ValueError(f"unknown predictor {name!r}")


# ==============================================================================
# Hermetic self-test
# ==============================================================================
def _synth_dataset(n: int = 240, T: int = 4, seed: int = 0) -> Dataset:
    """Synthetic (X, Y) with a learnable per-tool signal and some masked cells."""
    rng = np.random.default_rng(seed)
    d = 6
    X = rng.normal(size=(n, d)).astype(float)
    classes = list(CANONICAL_CLASSES[:3])
    onehot = np.zeros((n, len(classes)))
    ci = rng.integers(0, len(classes), size=n)
    onehot[np.arange(n), ci] = 1.0
    Xf = np.hstack([X, onehot])
    # Each tool detects when a tool-specific linear combo exceeds a threshold.
    W = rng.normal(size=(Xf.shape[1], T))
    logits = Xf @ W
    probs = 1 / (1 + np.exp(-logits))
    Y = (rng.uniform(size=(n, T)) < probs).astype(float)
    # Mask ~15% of cells to NaN (missing tool outcomes).
    Y[rng.uniform(size=(n, T)) < 0.15] = np.nan
    groups = rng.integers(0, n // 3, size=n)  # base_id-like groups
    return Dataset(X=Xf, Y=Y,
                   feature_names=[f"f_{i}" for i in range(d)] + [f"class_{c}" for c in classes],
                   tool_names=[f"tool{i}" for i in range(T)],
                   class_names=classes,
                   instance_ids=[(f"c{i}", classes[ci[i]]) for i in range(n)],
                   groups=groups)


def _auc(y: np.ndarray, s: np.ndarray) -> float:
    """AUC on non-NaN entries; 0.5 when degenerate (single class or all-tied)."""
    from sklearn.metrics import roc_auc_score
    m = ~np.isnan(y)
    y, s = y[m], s[m]
    if y.size == 0 or np.unique(y).size < 2:
        return 0.5
    return float(roc_auc_score(y, s))


def run_selftest() -> int:
    print(f"RELIANT 04_models self-test (v{__version__})")
    ds = _synth_dataset()

    # deterministic split by group (mimics stage-07 leakage-safe CV)
    uniq = np.unique(ds.groups)
    tr_groups = set(uniq[: int(0.7 * len(uniq))])
    tr = [i for i in range(ds.n) if ds.groups[i] in tr_groups]
    te = [i for i in range(ds.n) if ds.groups[i] not in tr_groups]
    dtr, dte = ds.subset(tr), ds.subset(te)

    # --- interface + shapes + range for tabular predictors ---------------------
    for name in ("constant", "ridge", "lightgbm"):
        p = build_predictor(name).fit_dataset(dtr)
        P = p.predict_dataset(dte)
        assert P.shape == (dte.n, len(dte.tool_names)), f"{name} bad shape"
        assert np.all((P >= 0) & (P <= 1)), f"{name} out of [0,1]"

    # --- learners beat the base rate on held-out AUC ---------------------------
    const = ConstantPredictor().fit_dataset(dtr)
    ridge = RidgePredictor(alpha=1.0).fit_dataset(dtr)
    lgbm = LGBMPredictor().fit_dataset(dtr)
    Pc, Pr, Pl = (const.predict_dataset(dte), ridge.predict_dataset(dte),
                  lgbm.predict_dataset(dte))
    auc_c = np.mean([_auc(dte.Y[:, t], Pc[:, t]) for t in range(len(dte.tool_names))])
    auc_r = np.mean([_auc(dte.Y[:, t], Pr[:, t]) for t in range(len(dte.tool_names))])
    auc_l = np.mean([_auc(dte.Y[:, t], Pl[:, t]) for t in range(len(dte.tool_names))])
    print(f"  held-out mean AUC: constant={auc_c:.3f} ridge={auc_r:.3f} lgbm={auc_l:.3f}")
    assert abs(auc_c - 0.5) < 0.06, "constant predictor should be ~0.5 AUC"
    assert auc_r > 0.6, f"ridge failed to learn (AUC {auc_r:.3f})"
    assert auc_l > 0.6, f"lightgbm failed to learn (AUC {auc_l:.3f})"

    # --- masked cells never fabricated: a fully-missing tool -> base-rate 0.5 ---
    Ym = dtr.Y.copy(); Ym[:, 0] = np.nan
    d0 = Dataset(dtr.X, Ym, dtr.feature_names, dtr.tool_names, dtr.class_names,
                 dtr.instance_ids, dtr.groups)
    p0 = RidgePredictor().fit_dataset(d0).predict_dataset(dte)
    assert np.allclose(p0[:, 0], 0.5), "empty tool column should predict base 0.5"

    # --- save / load round-trip -------------------------------------------------
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        fp = Path(tmp) / "m.pkl"
        ridge.save(fp)
        loaded = ReliabilityPredictor.load(fp)
        assert np.allclose(loaded.predict_dataset(dte), Pr)

    # --- assemble_dataset from frames ------------------------------------------
    features = pd.DataFrame({
        "contract_id": ["a", "b", "c"],
        "dataset": ["solidifi"] * 3, "base_id": ["b1", "b1", "b2"],
        "parse_method": ["ast"] * 3,
        "f_x": [1.0, 2.0, 3.0], "f_y": [0.0, 1.0, 0.0]})
    labels = pd.DataFrame({
        "contract_id": ["a", "a", "b", "c"],
        "dataset": ["solidifi"] * 4,
        "base_id": ["b1", "b1", "b1", "b2"],
        "class_canonical": ["reentrancy", "reentrancy", "arithmetic", "reentrancy"],
        "tool": ["slither", "mythril", "slither", "slither"],
        "detected": pd.array([True, False, True, pd.NA], dtype="boolean")})
    dd = assemble_dataset(features, labels, target="detected")
    assert dd.n == 3 and dd.tool_names == ["mythril", "slither"]
    # class one-hot present + correct width (2 features + 7 classes)
    assert dd.X.shape[1] == 2 + len(CANONICAL_CLASSES)
    # instance (a, reentrancy): slither detected -> 1, mythril -> 0
    ai = dd.instance_ids.index(("a", "reentrancy"))
    assert dd.Y[ai, dd.tool_names.index("slither")] == 1.0
    assert dd.Y[ai, dd.tool_names.index("mythril")] == 0.0
    # instance (c, reentrancy): slither is NA -> NaN
    ci = dd.instance_ids.index(("c", "reentrancy"))
    assert np.isnan(dd.Y[ci, dd.tool_names.index("slither")])

    # --- graph_to_arrays (pure numpy) ------------------------------------------
    g = {"contract_id": "g", "edge_types": list(EDGE_TYPES), "n_nodes": 3,
         "nodes": [{"id": 0, "ntype": "contract", "feat": {"is_library": 1, "n_subnodes": 2}},
                   {"id": 1, "ntype": "function", "feat": {"visibility": 2, "n_calls": 3}},
                   {"id": 2, "ntype": "state_var", "feat": {"is_mapping": 1}}],
         "edges": [[0, 1, EDGE_TYPES.index("contains")],
                   [1, 2, EDGE_TYPES.index("reads_writes")]]}
    arr = graph_to_arrays(g)
    assert arr["node_feat"].shape == (3, len(NODE_TYPES) + len(NODE_FEAT_KEYS))
    assert arr["node_type"][0] == NODE_TYPES.index("contract")
    assert arr["edge_index"].shape == (2, 2) and arr["edge_type"].shape == (2,)
    # type one-hot set correctly for the function node
    assert arr["node_feat"][1, NODE_TYPES.index("function")] == 1.0

    # --- GNN is strictly skippable when torch is absent ------------------------
    gnn = HeteroGNNPredictor()
    assert gnn.requires_graphs is True
    try:
        import torch
        torch_present = torch is not None
    except Exception:
        torch_present = False
    if not torch_present:
        raised = False
        try:
            gnn.fit(dtr.X, dtr.Y, graphs=[g] * dtr.n)
        except RuntimeError as exc:
            raised = "torch" in str(exc).lower()
        assert raised, "GNN must raise a clear RuntimeError when torch is missing"
        print("  torch absent -> GNN correctly reports skippable.")
    else:  # pragma: no cover - only on a torch-enabled host
        print("  torch present -> GNN path available (not asserted here).")

    print("  all invariants passed.")
    return 0


# ==============================================================================
# CLI (optional demo on real artifacts)
# ==============================================================================
def do_demo(args) -> int:
    features = pd.read_parquet(args.features, engine="pyarrow")
    labels = pd.read_parquet(args.labels, engine="pyarrow")
    ds = assemble_dataset(features, labels, target=args.target)
    print(f"instances={ds.n}  tools={len(ds.tool_names)}  features={ds.X.shape[1]}")

    uniq = np.unique(ds.groups)
    rng = np.random.default_rng(0)
    rng.shuffle(uniq)
    tr_groups = set(uniq[: int(0.7 * len(uniq))])
    tr = [i for i in range(ds.n) if ds.groups[i] in tr_groups]
    te = [i for i in range(ds.n) if ds.groups[i] not in tr_groups]
    dtr, dte = ds.subset(tr), ds.subset(te)

    for name in ("constant", "ridge", "lightgbm"):
        p = build_predictor(name).fit_dataset(dtr)
        P = p.predict_dataset(dte)
        aucs = [_auc(dte.Y[:, t], P[:, t]) for t in range(len(ds.tool_names))]
        print(f"  {name:10} held-out mean AUC = {np.nanmean(aucs):.3f}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Reliability predictor model zoo (stage 04).")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--selftest", action="store_true")
    mode.add_argument("--demo", action="store_true")
    p.add_argument("--features", type=str, default="artifacts/features.parquet")
    p.add_argument("--labels", type=str, default="artifacts/labels.parquet")
    p.add_argument("--target", type=str, default=DEFAULT_TARGET)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.demo:
        return do_demo(args)
    return run_selftest()


if __name__ == "__main__":
    raise SystemExit(main())
