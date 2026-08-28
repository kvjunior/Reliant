#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
09_baselines.py -- External SATzilla-style predictor + formal selection oracles.

RELIANT: Calibrated Reliability Prediction and Budget-Constrained Portfolio
Selection for Smart Contract Security Analyzers.

--------------------------------------------------------------------------------
ROLE IN THE ARTIFACT -- AND WHY IT STRENGTHENS THE COMPARISONS
--------------------------------------------------------------------------------
The prior submission's comparison was judged misleading because the baselines
solved a different task (vulnerability detection) than the model (performance
prediction). This stage supplies the missing task-appropriate references, of two
kinds:

  (A) An external, recognized ALGORITHM-SELECTION predictor. SATzilla (Xu, Hutter,
      Hoos & Leyton-Brown, 2008), in the empirical-hardness tradition of Rice
      (1976), builds a per-algorithm empirical performance model from instance
      features and selects accordingly. We provide a faithful per-tool
      empirical-performance-model baseline (a random forest per analyzer), which
      plugs into the stage-07 protocol exactly like our own predictors and lets
      RQ1 compare RELIANT against an established method on the SAME task.

  (B) Formal SELECTION baselines and an ORACLE, scored only on ground-truth cells:
        run-all      run every analyzer (the detection ceiling at full cost);
        best-1       always the single most reliable analyzer (naive selection);
        random-k     a random k-analyzer panel (chance selection). For fairness,
                     k defaults to RELIANT's own realized mean panel size at the
                     50% budget (read from results/rq3_selection_economics.json
                     when present, else 3) and can be pinned with --random-k;
        oracle       perfect-foresight selection -- for each contract the cheapest
                     analyzer that actually detects -- an UPPER bound no real
                     selector can beat. RELIANT's realized detection must fall
                     between random-k/best-1 and this oracle, which is exactly how
                     the paper positions it.

These are provided as a reusable module: (A) is a drop-in predictor for stage 07
(so `--models ... satzilla` includes it in the cross-validated RQ1 table), and (B)
is a set of pure functions this stage's --run mode scores on the stage-07
predictions to emit results/baselines.json for stage 10.

--------------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------------
    python3 src/09_baselines.py --run --artifacts artifacts --out results
    python3 src/09_baselines.py --run --random-k 3     # pin the random panel size
    python3 src/09_baselines.py --selftest

Exit codes: 0 = success; 3 = usage error.
"""

from __future__ import annotations

import argparse
import json
import sys
import importlib.util
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__version__ = "1.1.0"
SCHEMA_VERSION = "reliant-baselines-1"

# Fallback per-contract costs (seconds), used ONLY when measured medians from
# artifacts/tool_timings.parquet are unavailable for a tool. Keys mirror stage
# 02's DEFAULT_PANEL exactly (7 tools; semgrep-c3a9f40 was removed from the
# panel in stage 02 v1.1.0 because its pinned findings.yaml covers none of the
# seven canonical classes) -- the self-test asserts this sync so the two files
# cannot drift.
DEFAULT_TOOL_COSTS: Dict[str, float] = {
    "slither-0.11.3": 6.0, "mythril-0.24.8": 45.0, "oyente": 30.0,
    "smartcheck": 4.0, "securify2": 20.0, "conkas": 25.0,
    "confuzzius": 90.0,
}
_FALLBACK_COST = 20.0


def load_stage(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _agg(values: Sequence[float]) -> Dict[str, object]:
    a = np.asarray([v for v in values if v == v], dtype=float)
    if a.size == 0:
        return {"mean": None, "std": None, "n": 0}
    return {"mean": float(a.mean()), "std": float(a.std()), "n": int(a.size)}


def _tool_cost_vector(tools: Sequence[str], costs_map: Dict[str, float]) -> np.ndarray:
    return np.array([costs_map.get(t, _FALLBACK_COST) for t in tools], dtype=float)


# ==============================================================================
# (A) SATzilla-style empirical performance model (drop-in stage-04 predictor)
# ==============================================================================
class SATzillaPredictor:
    """Per-analyzer empirical performance model in the SATzilla tradition.

    A separate random-forest model predicts each analyzer's reliability from the
    alert-free instance features (classifier for the binary `detected` target,
    regressor for a continuous reliability target). Duck-typed to the stage-04
    ReliabilityPredictor interface (fit/predict/fit_dataset/predict_dataset,
    requires_graphs) so stage 07 can cross-validate it alongside our own models.
    """

    requires_graphs = False

    def __init__(self, seed: int = 0, n_estimators: int = 200,
                 min_samples_leaf: int = 2, max_depth: Optional[int] = None):
        self.seed = seed
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.tool_names: List[str] = []
        self.models_: List[Optional[Tuple[str, object]]] = []
        self.base_: np.ndarray = np.zeros(0)
        self.binary_: bool = True

    def fit(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> "SATzillaPredictor":
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        T = Y.shape[1]
        finite = Y[~np.isnan(Y)]
        self.binary_ = bool(finite.size and np.all(np.isin(finite, (0.0, 1.0))))
        self.models_ = []
        self.base_ = np.full(T, 0.5, dtype=float)
        for t in range(T):
            m = ~np.isnan(Y[:, t])
            yt, Xt = Y[m, t], X[m]
            if yt.size:
                self.base_[t] = float(yt.mean())
            if Xt.shape[0] < 5 or (self.binary_ and np.unique(yt).size < 2):
                self.models_.append(None)               # degenerate -> base rate
                continue
            if self.binary_:
                clf = RandomForestClassifier(
                    n_estimators=self.n_estimators, min_samples_leaf=self.min_samples_leaf,
                    max_depth=self.max_depth, random_state=self.seed, n_jobs=1)
                clf.fit(Xt, yt.astype(int))
                self.models_.append(("clf", clf))
            else:
                reg = RandomForestRegressor(
                    n_estimators=self.n_estimators, min_samples_leaf=self.min_samples_leaf,
                    max_depth=self.max_depth, random_state=self.seed, n_jobs=1)
                reg.fit(Xt, yt)
                self.models_.append(("reg", reg))
        return self

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        n, T = X.shape[0], len(self.models_)
        out = np.zeros((n, T), dtype=float)
        for t, mt in enumerate(self.models_):
            if mt is None:
                out[:, t] = self.base_[t]
            elif mt[0] == "clf":
                clf = mt[1]
                proba = clf.predict_proba(X)
                if proba.shape[1] == 2:
                    out[:, t] = proba[:, 1]
                else:                                    # single observed class
                    out[:, t] = float(clf.classes_[0])
            else:
                out[:, t] = np.clip(mt[1].predict(X), 0.0, 1.0)
        return out

    def fit_dataset(self, ds) -> "SATzillaPredictor":
        self.tool_names = list(ds.tool_names)
        return self.fit(ds.X, ds.Y)

    def predict_dataset(self, ds) -> np.ndarray:
        return self.predict(ds.X)


def build_baseline_predictor(name: str, seed: int = 0):
    """Factory for stage 07: baseline predictor names handled by this module."""
    if name in ("satzilla", "satzilla_rf", "epm"):
        return SATzillaPredictor(seed=seed)
    raise ValueError(f"unknown baseline predictor {name!r}")


# ==============================================================================
# (B) Selection oracles and baselines (scored on ground-truth cells only)
# ==============================================================================
def _clean(detected: np.ndarray) -> np.ndarray:
    """Binary detection matrix with unknowns (NaN) treated as not-detected."""
    return (np.nan_to_num(np.asarray(detected, dtype=float), nan=0.0) >= 0.5).astype(float)


def run_all(detected: np.ndarray, costs: np.ndarray) -> Dict[str, float]:
    """Run every analyzer: the detection ceiling at full cost."""
    D = _clean(detected)
    return {"detection": float(np.mean(np.any(D >= 0.5, axis=1))),
            "cost": float(costs.sum()), "panel_size": float(D.shape[1])}


def best_single_tool(detected: np.ndarray, costs: np.ndarray) -> Dict[str, float]:
    """Always the single most reliable analyzer (naive selection)."""
    D = _clean(detected)
    rates = D.mean(axis=0)
    t = int(np.argmax(rates))
    return {"tool_index": t, "detection": float(rates[t]),
            "cost": float(costs[t]), "panel_size": 1.0}


def random_k(detected: np.ndarray, costs: np.ndarray, k: int,
             reps: int = 200, seed: int = 0) -> Dict[str, float]:
    """A random k-analyzer panel (chance selection), averaged over draws."""
    D = _clean(detected)
    T = D.shape[1]
    k = max(1, min(k, T))
    rng = np.random.default_rng(seed)
    dets, cs = [], []
    for _ in range(reps):
        sub = rng.choice(T, size=k, replace=False)
        dets.append(float(np.mean(np.any(D[:, sub] >= 0.5, axis=1))))
        cs.append(float(costs[sub].sum()))
    return {"k": k, "detection": float(np.mean(dets)),
            "cost": float(np.mean(cs)), "panel_size": float(k)}


def oracle_selection(detected: np.ndarray, costs: np.ndarray,
                     budget: Optional[float] = None) -> Dict[str, float]:
    """Perfect-foresight selection: the cheapest analyzer that actually detects.

    For each contract the oracle knows the outcome, so it runs the single cheapest
    analyzer that detects it (cost = min cost among detecting analyzers), subject
    to the budget. No real selector can exceed this detection at this cost -- it is
    the upper bound that positions RELIANT from above.
    """
    D = _clean(detected)
    n, T = D.shape
    detected_flag = np.zeros(n, dtype=float)
    paid = np.zeros(n, dtype=float)
    for i in range(n):
        det_tools = np.where(D[i] >= 0.5)[0]
        if det_tools.size == 0:
            continue                                    # undetectable by any tool
        min_cost = float(costs[det_tools].min())
        if budget is None or min_cost <= budget + 1e-9:
            detected_flag[i] = 1.0
            paid[i] = min_cost
    det = float(detected_flag.mean())
    mean_cost = float(paid[detected_flag > 0].mean()) if detected_flag.any() else 0.0
    return {"detection": det, "cost_per_detection": mean_cost,
            "mean_cost": float(paid.mean()), "panel_size": 1.0}


def score_selection_baselines(detected: np.ndarray, costs: np.ndarray,
                              panel_size: int, budget_fracs: Sequence[float],
                              reps: int = 200, seed: int = 0) -> dict:
    """All selection baselines at once, for one seed's detection matrix."""
    run_all_cost = float(costs.sum())
    out = {
        "run_all": run_all(detected, costs),
        "best_1": best_single_tool(detected, costs),
        "random_k": random_k(detected, costs, panel_size, reps, seed),
        "oracle_unbounded": oracle_selection(detected, costs, None),
    }
    out["oracle_by_budget"] = {
        f"{int(f*100)}pct_budget": oracle_selection(detected, costs, f * run_all_cost)
        for f in budget_fracs}
    return out


# ==============================================================================
# --run: score baselines on stage-07 predictions -> results/baselines.json
# ==============================================================================
def _detected_matrix(ds_slice: pd.DataFrame, tools: List[str]) -> np.ndarray:
    piv = ds_slice.pivot_table(index=["contract_id", "class_canonical"],
                               columns="tool", values="y_true", aggfunc="first")
    return piv.reindex(columns=tools).to_numpy(float)


def _costs_from_artifacts(art: Path) -> Dict[str, float]:
    """Per-tool costs: DEFAULT_TOOL_COSTS updated with measured medians.

    Medians are taken over SUCCESSFUL runs only (durations of failed / timed-out
    runs measure the failure mode, not the tool's cost), and they OVERRIDE the
    defaults per tool rather than replacing the whole map, so a tool with no
    successful timing keeps its documented fallback instead of silently
    degrading to _FALLBACK_COST.
    """
    costs_map = dict(DEFAULT_TOOL_COSTS)
    timings = art / "tool_timings.parquet"
    if timings.exists():
        t = pd.read_parquet(timings, engine="pyarrow")
        if "status" in t.columns:
            t = t[t["status"] == "success"]
        med = t.groupby("tool")["duration_s"].median()
        costs_map.update({k: float(v) for k, v in med.items() if v == v})
    return costs_map


def _derive_random_k(out_dir: Path, explicit: Optional[int]) -> Tuple[int, str]:
    """Choose the random-panel size k, with recorded provenance.

    Priority: (1) an explicit --random-k (pinning for reproducibility);
    (2) RELIANT's own realized mean panel size at the 50% budget from
    results/rq3_selection_economics.json -- the fair chance-selection reference
    is a random panel of the SAME size the calibrated selector actually uses;
    (3) the documented default of 3 (the mid-panel size) when stage 08 has not
    run yet. Returns (k, source_description).
    """
    if explicit is not None:
        return max(1, int(explicit)), "explicit --random-k"
    rq3 = out_dir / "rq3_selection_economics.json"
    if rq3.exists():
        try:
            frontier = json.loads(rq3.read_text(encoding="utf-8")).get(
                "budget_frontier", {})
            for key in ("50pct_budget", *sorted(frontier)):
                mean = (frontier.get(key, {}).get("mean_panel_size", {})
                        or {}).get("mean")
                if mean is not None and mean == mean:
                    return (max(1, int(round(float(mean)))),
                            f"rq3 mean_panel_size at {key}")
        except (json.JSONDecodeError, OSError):
            pass
    return 3, "default (stage-08 RQ3 not available)"


def _satzilla_auc_if_present(preds: pd.DataFrame) -> Optional[dict]:
    """If stage 07 was run with the SATzilla model, report its per-tool AUC."""
    from sklearn.metrics import roc_auc_score
    names = [m for m in preds["model"].unique() if m in ("satzilla", "satzilla_rf")]
    if not names:
        return None
    name = names[0]
    dm = preds[preds.model == name]
    tools = sorted(dm["tool"].unique())
    vals = []
    for seed in sorted(dm["seed"].unique()):
        for fold in sorted(dm[dm.seed == seed]["fold"].unique()):
            dsf = dm[(dm.seed == seed) & (dm.fold == fold)]
            for t in tools:
                sub = dsf[dsf.tool == t]
                y = sub["y_true"].to_numpy(float)
                p = sub["y_pred"].to_numpy(float)
                m = ~np.isnan(y)
                if m.sum() and np.unique(y[m]).size > 1:
                    vals.append(float(roc_auc_score(y[m], p[m])))
    return {"model": name, "mean_tool_auc": _agg(vals)}


def do_run(args) -> int:
    art = Path(args.artifacts)
    out = Path(args.out)
    preds = pd.read_parquet(art / "predictions.parquet", engine="pyarrow")
    costs_map = _costs_from_artifacts(art)

    tools = sorted(preds["tool"].unique())
    costs = _tool_cost_vector(tools, costs_map)
    models = set(preds["model"].unique())
    ref_model = "lightgbm" if "lightgbm" in models else sorted(models)[0]
    dm = preds[preds.model == ref_model]
    budget_fracs = (0.25, 0.5, 0.75)
    random_k_size, random_k_source = _derive_random_k(out, args.random_k)

    # aggregate baseline scores across seeds
    per_seed = {"run_all_det": [], "best1_det": [], "randomk_det": [],
                "randomk_cost": [], "oracle_det": [], "oracle_cost": [],
                "best1_cost": []}
    oracle_budget = {f"{int(f*100)}pct_budget": {"det": [], "cost": []}
                     for f in budget_fracs}
    for seed in sorted(dm["seed"].unique()):
        det = _detected_matrix(dm[dm.seed == seed], tools)
        s = score_selection_baselines(det, costs, random_k_size, budget_fracs,
                                      seed=seed)
        per_seed["run_all_det"].append(s["run_all"]["detection"])
        per_seed["best1_det"].append(s["best_1"]["detection"])
        per_seed["best1_cost"].append(s["best_1"]["cost"])
        per_seed["randomk_det"].append(s["random_k"]["detection"])
        per_seed["randomk_cost"].append(s["random_k"]["cost"])
        per_seed["oracle_det"].append(s["oracle_unbounded"]["detection"])
        per_seed["oracle_cost"].append(s["oracle_unbounded"]["cost_per_detection"])
        for f in budget_fracs:
            key = f"{int(f*100)}pct_budget"
            oracle_budget[key]["det"].append(s["oracle_by_budget"][key]["detection"])
            oracle_budget[key]["cost"].append(s["oracle_by_budget"][key]["mean_cost"])

    result = {
        "schema_version": SCHEMA_VERSION, "version": __version__,
        "reference_model_for_detection": ref_model,
        "tools": tools,
        "costs_used": {t: float(c) for t, c in zip(tools, costs)},
        "run_all_cost": float(costs.sum()),
        "random_k": {"k": random_k_size, "source": random_k_source},
        "selection_baselines": {
            "run_all_detection": _agg(per_seed["run_all_det"]),
            "best_1_detection": _agg(per_seed["best1_det"]),
            "best_1_cost": _agg(per_seed["best1_cost"]),
            "random_k_detection": _agg(per_seed["randomk_det"]),
            "random_k_mean_cost": _agg(per_seed["randomk_cost"]),
            "oracle_unbounded_detection": _agg(per_seed["oracle_det"]),
            "oracle_cost_per_detection": _agg(per_seed["oracle_cost"]),
            "oracle_by_budget": {
                k: {"detection": _agg(v["det"]), "mean_cost": _agg(v["cost"])}
                for k, v in oracle_budget.items()},
        },
        "note": ("Oracle is a perfect-foresight upper bound; RELIANT (stage 08 RQ3) "
                 "should sit between best-1/random-k and this oracle."),
    }
    sat = _satzilla_auc_if_present(preds)
    if sat is not None:
        result["satzilla_predictor"] = sat

    out.mkdir(parents=True, exist_ok=True)
    (out / "baselines.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    b = result["selection_baselines"]
    print(f"Wrote {out}/baselines.json")
    print(f"  run-all detection = {b['run_all_detection']['mean']:.3f} "
          f"(cost {result['run_all_cost']:.0f})")
    print(f"  best-1 detection  = {b['best_1_detection']['mean']:.3f} "
          f"(cost {b['best_1_cost']['mean']:.0f})")
    print(f"  random-k detection= {b['random_k_detection']['mean']:.3f} "
          f"(k={random_k_size}, {random_k_source})")
    print(f"  ORACLE detection  = {b['oracle_unbounded_detection']['mean']:.3f} "
          f"(cost/detection {b['oracle_cost_per_detection']['mean']:.1f})")
    if sat is not None:
        print(f"  SATzilla mean tool AUC = {sat['mean_tool_auc']['mean']}")
    return 0


# ==============================================================================
# Hermetic self-test
# ==============================================================================
def _synth_dataset(m4, n: int = 150, seed: int = 0):
    rng = np.random.default_rng(seed)
    classes = list(m4.CANONICAL_CLASSES)[:3]
    tools = [f"tool{i}" for i in range(4)]
    contract_ids, base_ids, cls, sig = [], [], [], []
    for c in range(n):
        contract_ids.append(f"c{c}")
        base_ids.append(f"b{c // 3}")
        cls.append(classes[c % 3])
        sig.append(rng.normal())
    sig = np.asarray(sig)
    feat = pd.DataFrame({"contract_id": contract_ids, "dataset": "solidifi",
                         "base_id": base_ids, "parse_method": "ast",
                         "f_a": sig, "f_b": rng.uniform(0, 3, n)})
    off = rng.uniform(-0.4, 0.4, len(tools))
    rows = []
    for cid, k, s in zip(contract_ids, cls, sig):
        for ti, t in enumerate(tools):
            p = 1 / (1 + np.exp(-(1.5 * s + off[ti])))
            rows.append((cid, "solidifi", f"b{int(cid[1:]) // 3}", k, t,
                         bool(rng.uniform() < p)))
    lab = pd.DataFrame(rows, columns=["contract_id", "dataset", "base_id",
                                      "class_canonical", "tool", "detected"])
    lab["detected"] = lab["detected"].astype("boolean")
    ds = m4.assemble_dataset(feat, lab, target="detected", tools=tools)
    return ds


def run_selftest() -> int:
    print(f"RELIANT 09_baselines self-test (v{__version__})")
    src = Path(__file__).resolve().parent
    m4 = load_stage(src / "04_models.py", "reliant_stage04")

    # --- SATzilla predictor learns and is a clean drop-in ----------------------
    ds = _synth_dataset(m4)
    uniq = np.unique(ds.groups)
    rng = np.random.default_rng(0)
    rng.shuffle(uniq)
    cut = int(0.7 * len(uniq))
    tr = [i for i in range(ds.n) if ds.groups[i] in set(uniq[:cut])]
    te = [i for i in range(ds.n) if ds.groups[i] in set(uniq[cut:])]
    dtr, dte = ds.subset(tr), ds.subset(te)
    sat = build_baseline_predictor("satzilla", seed=0).fit_dataset(dtr)
    P = sat.predict_dataset(dte)
    assert P.shape == (dte.n, len(ds.tool_names))
    from sklearn.metrics import roc_auc_score
    aucs = []
    for t in range(P.shape[1]):
        y = dte.Y[:, t]
        m = ~np.isnan(y)
        if np.unique(y[m]).size > 1:
            aucs.append(roc_auc_score(y[m], P[m, t]))
    print(f"  SATzilla mean per-tool AUC = {np.mean(aucs):.3f}")
    assert np.mean(aucs) > 0.6, "SATzilla predictor failed to learn"
    # determinism
    P2 = build_baseline_predictor("satzilla", seed=0).fit_dataset(dtr).predict_dataset(dte)
    assert np.allclose(P, P2), "SATzilla predictor not deterministic"

    # --- selection oracle ordering: oracle >= run-all >= best-1 >= random-k -----
    rng = np.random.default_rng(1)
    T = 5
    n = 400
    costs = np.array([5, 8, 12, 20, 30], dtype=float)
    # correlated-ish detection with heterogeneous tool strengths
    base = rng.uniform(0.2, 0.6, (n, 1))
    strength = rng.uniform(-0.1, 0.3, (1, T))
    D = (rng.uniform(size=(n, T)) < np.clip(base + strength, 0, 1)).astype(float)
    ra = run_all(D, costs)
    b1 = best_single_tool(D, costs)
    rk1 = random_k(D, costs, k=1, seed=3)
    rk2 = random_k(D, costs, k=2, seed=3)
    orc = oracle_selection(D, costs, None)
    print(f"  detection: oracle={orc['detection']:.3f} run-all={ra['detection']:.3f} "
          f"best-1={b1['detection']:.3f} random-1={rk1['detection']:.3f} "
          f"random-2={rk2['detection']:.3f}")
    # unbounded oracle detects exactly what run-all does (any tool suffices)
    assert abs(orc["detection"] - ra["detection"]) < 1e-9, "oracle == run-all detection"
    # run-all is the ceiling; adding tools never hurts
    assert ra["detection"] >= b1["detection"] - 1e-9, "run-all >= best-1"
    assert ra["detection"] >= rk2["detection"] - 1e-9, "run-all >= random-2"
    assert rk2["detection"] >= rk1["detection"] - 1e-9, "more tools >= fewer (avg)"
    # best single >= a random single (max rate >= mean rate)
    assert b1["detection"] >= rk1["detection"] - 1e-9, "best-1 >= random-1"
    # oracle reaches that detection far cheaper than running everything
    assert orc["cost_per_detection"] <= ra["cost"], "oracle cheaper than run-all"
    print(f"  oracle cost/detection={orc['cost_per_detection']:.1f} << run-all cost={ra['cost']:.0f}")

    # --- budgeted oracle is monotone in budget ---------------------------------
    scores = score_selection_baselines(D, costs, panel_size=2,
                                       budget_fracs=(0.1, 0.3, 0.6), seed=0)
    dets = [scores["oracle_by_budget"][k]["detection"]
            for k in ("10pct_budget", "30pct_budget", "60pct_budget")]
    assert dets[0] <= dets[1] + 1e-9 <= dets[2] + 1e-9, "oracle detection must rise with budget"
    print(f"  oracle detection vs budget: {dets[0]:.3f} <= {dets[1]:.3f} <= {dets[2]:.3f}")

    # JSON serializable
    json.dumps(scores)

    # --- random-k derivation: explicit > rq3-derived > default -----------------
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        assert _derive_random_k(out_dir, 5) == (5, "explicit --random-k")
        k, srcd = _derive_random_k(out_dir, None)
        assert k == 3 and "default" in srcd
        (out_dir / "rq3_selection_economics.json").write_text(json.dumps({
            "budget_frontier": {
                "25pct_budget": {"mean_panel_size": {"mean": 1.4}},
                "50pct_budget": {"mean_panel_size": {"mean": 2.6}}}}),
            encoding="utf-8")
        k, srcd = _derive_random_k(out_dir, None)
        assert k == 3 and "50pct_budget" in srcd, (k, srcd)  # round(2.6) = 3
        assert _derive_random_k(out_dir, 1) == (1, "explicit --random-k")

        # --- cost loading: success-only medians UPDATE the defaults ------------
        art = out_dir / "art"
        art.mkdir()
        cm = _costs_from_artifacts(art)                 # no timings file
        assert cm == DEFAULT_TOOL_COSTS
        t = pd.DataFrame({
            "contract_id": ["c1", "c2", "c3", "c4"],
            "tool": ["oyente", "oyente", "oyente", "conkas"],
            "duration_s": [10.0, 12.0, 500.0, None],
            "status": ["success", "success", "error", "missing"],
        })
        t.to_parquet(art / "tool_timings.parquet", engine="pyarrow", index=False)
        cm = _costs_from_artifacts(art)
        assert cm["oyente"] == 11.0, cm["oyente"]       # median over successes only
        assert cm["conkas"] == DEFAULT_TOOL_COSTS["conkas"]  # no success -> default
        assert cm["mythril-0.24.8"] == DEFAULT_TOOL_COSTS["mythril-0.24.8"]

    # --- drift tripwires: cost map synced to stage 02 panel & stage 08 copy ----
    m2 = load_stage(src / "02_ground_truth.py", "reliant_stage02")
    assert set(DEFAULT_TOOL_COSTS) == set(m2.DEFAULT_PANEL), (
        "DEFAULT_TOOL_COSTS keys must equal stage 02's DEFAULT_PANEL")
    m8 = load_stage(src / "08_evaluate.py", "reliant_stage08")
    for tool in m2.DEFAULT_PANEL:
        assert m8.DEFAULT_TOOL_COSTS.get(tool) == DEFAULT_TOOL_COSTS[tool], (
            f"cost for {tool} differs between stages 08 and 09")
    print("  cross-file sync: costs == stage-08 copy on the stage-02 panel.")

    print("  all invariants passed.")
    return 0


# ==============================================================================
# CLI
# ==============================================================================
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Baselines: SATzilla predictor + oracles (stage 09).")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--selftest", action="store_true")
    p.add_argument("--artifacts", type=str, default="artifacts")
    p.add_argument("--out", type=str, default="results")
    p.add_argument("--random-k", type=int, default=None,
                   help="Random-panel size for the chance baseline. Default: "
                        "RELIANT's realized mean panel size at the 50%% budget "
                        "(from results/rq3_selection_economics.json), else 3.")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.run:
        return do_run(args)
    return run_selftest()


if __name__ == "__main__":
    raise SystemExit(main())
