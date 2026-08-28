#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
07_train.py -- Reproducibility spine: leakage-safe training, calibration, and dumps.

RELIANT: Calibrated Reliability Prediction and Budget-Constrained Portfolio
Selection for Smart Contract Security Analyzers.

--------------------------------------------------------------------------------
ROLE IN THE ARTIFACT
--------------------------------------------------------------------------------
One entry point that turns the corpus, features, and labels into everything the
evaluation stage needs, with the leakage discipline the paper depends on. It:

  1. assembles instances = (contract, query-class) with per-tool labels (stage 04);
  2. runs grouped K-fold cross-validation keyed on base_id, so every instance
     gets an OUT-OF-FOLD prediction and no near-duplicate (the seven SolidiFI
     variants of a base) ever straddles train and test;
  3. within each fold, splits off a calibration set (also by base_id) and fits the
     stage-05 conformal certificates (marginal split, Mondrian-by-class, and a
     one-sided FNR upper bound for the portfolio);
  4. repeats across seeds for stochastic models and reports mean +/- spread;
  5. additionally runs the cross-benchmark generalization split (train on the
     synthetic SolidiFI corpus, test on the real-world smartbugs-curated corpus)
     that stage 08 uses for the distribution-shift question;
  6. dumps predictions, split assignments, and a run summary to artifacts/.

It also hosts the shared module loader that imports the numbered stages (their
names are not valid module identifiers, and stages 04/06 define dataclasses that
require the module to be registered in sys.modules before execution).

--------------------------------------------------------------------------------
WHY THIS ADDRESSES THE PRIOR REVIEW
--------------------------------------------------------------------------------
Every downstream number is produced under one auditable, leakage-safe protocol:
grouped CV by base_id (no train/test contamination from injected variants),
out-of-fold predictions (no optimistic in-sample fit), conformal calibration on a
held-out split (honest intervals), and a fixed seed set (bit-reproducible). This
is the machinery that lets stages 08-10 report accuracy, coverage, and selection
economics that a reviewer can regenerate exactly.

--------------------------------------------------------------------------------
OUTPUTS (to --out, default artifacts/)
--------------------------------------------------------------------------------
    predictions.parquet        per (seed, fold, model, instance, tool): the label,
                               the point prediction, split & Mondrian intervals,
                               and the one-sided FNR upper bound.
    predictions_shift.parquet  SolidiFI -> curated cross-benchmark predictions.
    splits.json                per (seed, fold) base_id assignments (train/cal/test).
    train_meta.json            config, per-model OOF accuracy & coverage, rho,
                               timings, and corpus provenance (registry
                               fingerprint) when --registry is supplied.

--------------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------------
    python3 src/07_train.py --train \
        --registry data/registry.parquet \
        --features artifacts/features.parquet --labels artifacts/labels.parquet \
        --graphs artifacts/graphs.jsonl \
        --target detected --alpha 0.1 --folds 5 --seeds 3 \
        --models constant ridge lightgbm satzilla --out artifacts

    python3 src/07_train.py --selftest      # hermetic; synthetic features+labels

Exit codes: 0 = success; 3 = usage error.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__version__ = "1.1.0"
SCHEMA_VERSION = "reliant-train-1"

# Schema of predictions.parquet. Stage 08 carries the reciprocal copy and
# both self-tests assert equality, so the contract cannot drift silently.
PRED_COLUMNS: Tuple[str, ...] = (
    "seed", "fold", "model", "contract_id", "dataset", "base_id",
    "class_canonical", "tool", "y_true", "y_pred",
    "lo_split", "hi_split", "lo_mond", "hi_mond", "fnr_upper",
)
_MIN_CAL = 5  # below this many calibration points for a tool -> trivial certificate


# ==============================================================================
# Shared stage loader (registers in sys.modules; required for dataclass stages)
# ==============================================================================
def load_stage(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod                     # BEFORE exec (dataclass resolution)
    spec.loader.exec_module(mod)                # type: ignore[union-attr]
    return mod


def load_stages(src_dir: Path):
    m4 = load_stage(src_dir / "04_models.py", "reliant_stage04")
    m5 = load_stage(src_dir / "05_conformal.py", "reliant_stage05")
    m6 = load_stage(src_dir / "06_portfolio.py", "reliant_stage06")
    return m4, m5, m6


# ==============================================================================
# Leakage-safe grouped splitting
# ==============================================================================
def group_kfold(groups: Sequence[object], k: int, seed: int
                ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """K folds partitioning unique groups (base_ids); instances follow their group.

    k is clamped to the number of distinct groups so no fold can receive an empty
    test set (which would produce an unusable subset downstream); with fewer
    groups than requested folds this degrades gracefully to leave-one-group-out.
    """
    g = np.asarray(groups, dtype=object)
    uniq = np.unique(g)
    if uniq.size == 0:
        raise ValueError("group_kfold: no groups to split")
    k = max(1, min(int(k), int(uniq.size)))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    fold_groups = np.array_split(uniq, k)
    out = []
    for f in range(k):
        test_g = set(fold_groups[f].tolist())
        test_idx = np.array([i for i in range(g.size) if g[i] in test_g], dtype=int)
        dev_idx = np.array([i for i in range(g.size) if g[i] not in test_g], dtype=int)
        out.append((dev_idx, test_idx))
    return out


def group_holdout(idx: np.ndarray, groups_all: np.ndarray, frac_cal: float,
                  seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Split dev indices into (train, calibration) by base_id, disjoint by group.

    At least one group is always left for training: with a single dev group the
    calibration set is empty and calibrate_fold falls back to trivial (maximally
    conservative) certificates rather than fitting on nothing.
    """
    sub = np.unique(groups_all[idx])
    rng = np.random.default_rng(seed + 1)
    rng.shuffle(sub)
    n_cal = max(1, int(round(frac_cal * len(sub))))
    n_cal = min(n_cal, max(0, len(sub) - 1))     # never starve the training set
    cal_g = set(sub[:n_cal].tolist())
    cal_idx = np.array([i for i in idx if groups_all[i] in cal_g], dtype=int)
    tr_idx = np.array([i for i in idx if groups_all[i] not in cal_g], dtype=int)
    return tr_idx, cal_idx


# ==============================================================================
# Predictor construction + per-fold conformal calibration
# ==============================================================================
# Baseline predictor names implemented by stage 09 (the SATzilla-style EPM).
_STAGE09_MODELS = ("satzilla", "satzilla_rf", "epm")
SUPPORTED_MODELS: Tuple[str, ...] = (
    "constant", "ridge", "lightgbm", "hetero_gnn", *_STAGE09_MODELS)


def make_predictor(m4, model: str, seed: int):
    if model in ("lightgbm", "lgbm"):
        return m4.build_predictor("lightgbm", seed=seed)
    if model in ("hetero_gnn", "gnn"):
        return m4.build_predictor("hetero_gnn", seed=seed)
    if model == "ridge":
        return m4.build_predictor("ridge")
    if model == "constant":
        return m4.build_predictor("constant")
    if model not in _STAGE09_MODELS:
        raise ValueError(
            f"unknown model {model!r}; supported: {', '.join(SUPPORTED_MODELS)}")
    # Baseline predictors live in stage 09; load lazily so stage 07 can
    # cross-validate them alongside our own models. A failure here is a real
    # import/environment problem, not a bad name, so report it as such.
    try:
        src_dir = Path(__file__).resolve().parent
        m9 = load_stage(src_dir / "09_baselines.py", "reliant_stage09")
        return m9.build_baseline_predictor(model, seed=seed)
    except Exception as exc:
        raise RuntimeError(
            f"model {model!r} is implemented in 09_baselines.py but could not be "
            f"loaded: {type(exc).__name__}: {exc}") from exc


def calibrate_fold(m5, y_cal: np.ndarray, pred_cal: np.ndarray,
                   cls_cal: np.ndarray, pred_test: np.ndarray,
                   cls_test: np.ndarray, alpha: float) -> Dict[str, np.ndarray]:
    """Per-tool conformal certificates for a fold's test instances.

    Returns arrays (n_test, T): split two-sided [lo, hi], Mondrian-by-class
    two-sided [lo, hi], and a one-sided Mondrian FNR upper bound.
    """
    n_te, T = pred_test.shape
    keys = ("lo_split", "hi_split", "lo_mond", "hi_mond", "fnr_upper")
    out = {k: np.full((n_te, T), np.nan) for k in keys}
    for t in range(T):
        yc, pc, pt = y_cal[:, t], pred_cal[:, t], pred_test[:, t]
        mask = ~np.isnan(yc)
        if int(mask.sum()) < _MIN_CAL:
            out["lo_split"][:, t] = 0.0
            out["hi_split"][:, t] = 1.0
            out["lo_mond"][:, t] = 0.0
            out["hi_mond"][:, t] = 1.0
            out["fnr_upper"][:, t] = 1.0        # no info -> maximally conservative
            continue
        sc = m5.SplitConformal(alpha=alpha, mode="two_sided").calibrate(yc[mask], pc[mask])
        lo, hi = sc.interval(pt)
        out["lo_split"][:, t], out["hi_split"][:, t] = lo, hi

        mc = m5.MondrianConformal(alpha=alpha, mode="two_sided", min_per_group=10)
        mc.calibrate(yc[mask], pc[mask], cls_cal[mask])
        mlo, mhi = mc.interval(pt, cls_test)
        out["lo_mond"][:, t], out["hi_mond"][:, t] = mlo, mhi

        # one-sided upper bound on FNR (= 1 - reliability) for the portfolio
        mcu = m5.MondrianConformal(alpha=alpha, mode="upper", min_per_group=10)
        mcu.calibrate(1.0 - yc[mask], 1.0 - pc[mask], cls_cal[mask])
        _, fu = mcu.interval(1.0 - pt, cls_test)
        out["fnr_upper"][:, t] = fu
    return out


def _emit_rows(rows: List[dict], seed: int, fold: int, model: str,
               ds_sub, cid2ds: Dict[str, str], Pte: np.ndarray,
               bounds: Dict[str, np.ndarray]) -> None:
    tools = ds_sub.tool_names
    for r, (cid, cls) in enumerate(ds_sub.instance_ids):
        base = ds_sub.groups[r]
        for t, tool in enumerate(tools):
            yt = ds_sub.Y[r, t]
            rows.append({
                "seed": seed, "fold": fold, "model": model,
                "contract_id": cid, "dataset": cid2ds.get(cid, ""),
                "base_id": base, "class_canonical": cls, "tool": tool,
                "y_true": float(yt) if not np.isnan(yt) else np.nan,
                "y_pred": float(Pte[r, t]),
                "lo_split": float(bounds["lo_split"][r, t]),
                "hi_split": float(bounds["hi_split"][r, t]),
                "lo_mond": float(bounds["lo_mond"][r, t]),
                "hi_mond": float(bounds["hi_mond"][r, t]),
                "fnr_upper": float(bounds["fnr_upper"][r, t]),
            })


# ==============================================================================
# Cross-validation
# ==============================================================================
def _assert_no_leakage(groups: np.ndarray, tr_idx: np.ndarray, cal_idx: np.ndarray,
                       test_idx: np.ndarray, where: str) -> None:
    """Hard guarantee that no base_id crosses a split boundary.

    Raised explicitly rather than via `assert` so the guard survives `python -O`:
    grouped-split integrity is the paper's central reproducibility claim and must
    never be silently optimized away.
    """
    tr, cal, te = set(groups[tr_idx]), set(groups[cal_idx]), set(groups[test_idx])
    for a, b, msg in ((tr, te, "train/test"), (cal, te, "cal/test"),
                      (tr, cal, "train/cal")):
        shared = a & b
        if shared:
            raise AssertionError(
                f"base_id leakage across {msg} at {where}: "
                f"{sorted(map(str, shared))[:5]} ...")


def run_cv(dataset, cid2ds: Dict[str, str], m4, m5, models: Sequence[str],
           seeds: Sequence[int], k: int, alpha: float,
           has_torch: bool) -> Tuple[pd.DataFrame, dict]:
    """Grouped K-fold CV across seeds; returns OOF predictions + split record."""
    rows: List[dict] = []
    splits_record: Dict[str, dict] = {}
    groups = dataset.groups
    for seed in seeds:
        folds = group_kfold(groups, k, seed)
        for fi, (dev_idx, test_idx) in enumerate(folds):
            tr_idx, cal_idx = group_holdout(dev_idx, groups, 0.25, seed)
            _assert_no_leakage(groups, tr_idx, cal_idx, test_idx,
                               f"seed{seed}_fold{fi}")

            dtr, dcal, dte = dataset.subset(tr_idx), dataset.subset(cal_idx), dataset.subset(test_idx)
            cls_cal = np.array([c for _, c in dcal.instance_ids], dtype=object)
            cls_te = np.array([c for _, c in dte.instance_ids], dtype=object)
            splits_record[f"seed{seed}_fold{fi}"] = {
                "train_base_ids": sorted(set(map(str, groups[tr_idx]))),
                "cal_base_ids": sorted(set(map(str, groups[cal_idx]))),
                "test_base_ids": sorted(set(map(str, groups[test_idx]))),
            }
            for model in models:
                if model in ("hetero_gnn", "gnn") and not has_torch:
                    continue                     # skip GNN when torch is absent
                pred = make_predictor(m4, model, seed).fit_dataset(dtr)
                Pcal = pred.predict_dataset(dcal)
                Pte = pred.predict_dataset(dte)
                bounds = calibrate_fold(m5, dcal.Y, Pcal, cls_cal, Pte, cls_te, alpha)
                _emit_rows(rows, seed, fi, model, dte, cid2ds, Pte, bounds)

    df = pd.DataFrame(rows, columns=list(PRED_COLUMNS))
    df = df.sort_values(["seed", "fold", "model", "contract_id", "class_canonical", "tool"],
                        kind="mergesort").reset_index(drop=True)
    return df, splits_record


# ==============================================================================
# Cross-benchmark generalization: SolidiFI -> curated
# ==============================================================================
def run_cross_benchmark(features: pd.DataFrame, labels: pd.DataFrame,
                        cid2ds: Dict[str, str], m4, m5, models: Sequence[str],
                        seeds: Sequence[int], alpha: float,
                        target: str, has_torch: bool,
                        graphs_by_id: Optional[dict]) -> pd.DataFrame:
    """Train on SolidiFI, calibrate on a held-out SolidiFI split, test on curated."""
    sol_ids = set(features.loc[features.dataset == "solidifi", "contract_id"])
    cur_ids = set(features.loc[features.dataset == "sb_curated", "contract_id"])
    lab_sol = labels[labels.contract_id.isin(sol_ids)]
    lab_cur = labels[labels.contract_id.isin(cur_ids)]
    if lab_sol.empty or lab_cur.empty:
        return pd.DataFrame(columns=list(PRED_COLUMNS))

    tools = sorted(set(lab_sol.tool.unique()) & set(lab_cur.tool.unique()))
    ds_sol = m4.assemble_dataset(features, lab_sol, target=target, tools=tools,
                                 graphs_by_id=graphs_by_id)
    ds_cur = m4.assemble_dataset(features, lab_cur, target=target, tools=tools,
                                 graphs_by_id=graphs_by_id)
    cls_cur = np.array([c for _, c in ds_cur.instance_ids], dtype=object)

    rows: List[dict] = []
    for seed in seeds:
        tr_idx, cal_idx = group_holdout(np.arange(ds_sol.n), ds_sol.groups, 0.25, seed)
        dtr, dcal = ds_sol.subset(tr_idx), ds_sol.subset(cal_idx)
        cls_cal = np.array([c for _, c in dcal.instance_ids], dtype=object)
        for model in models:
            if model in ("hetero_gnn", "gnn") and not has_torch:
                continue
            pred = make_predictor(m4, model, seed).fit_dataset(dtr)
            Pcal = pred.predict_dataset(dcal)
            Pcur = pred.predict_dataset(ds_cur)
            bounds = calibrate_fold(m5, dcal.Y, Pcal, cls_cal, Pcur, cls_cur, alpha)
            _emit_rows(rows, seed, -1, model, ds_cur, cid2ds, Pcur, bounds)
    df = pd.DataFrame(rows, columns=list(PRED_COLUMNS))
    return df.sort_values(["seed", "model", "contract_id", "class_canonical", "tool"],
                          kind="mergesort").reset_index(drop=True)


# ==============================================================================
# Summary (OOF accuracy + coverage + rho) for the metadata
# ==============================================================================
def _auc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    m = ~np.isnan(y)
    y, s = y[m], s[m]
    if y.size == 0 or np.unique(y).size < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def _oof_metric(sub: pd.DataFrame, binary: bool) -> Tuple[float, str]:
    """OOF accuracy computed the SAME way stage 08's RQ1 computes it.

    AUC is evaluated within each (seed, fold, tool) cell and then averaged, never
    pooled across tools. Pooling would let a per-tool base-rate predictor act as a
    cross-tool ranker and score well above chance (0.63 rather than 0.50 on this
    corpus), so train_meta.json and results/rq1_prediction_accuracy.json would
    disagree for every model. Averaging per cell keeps the two artifacts
    numerically identical and the base-rate baseline honestly at chance.
    """
    if not binary:
        y = sub["y_true"].to_numpy(float)
        p = sub["y_pred"].to_numpy(float)
        m = ~np.isnan(y)
        return float(np.sqrt(np.nanmean((y[m] - p[m]) ** 2))), "rmse"
    vals: List[float] = []
    for _, cell in sub.groupby(["seed", "fold", "tool"], observed=True):
        a = _auc(cell["y_true"].to_numpy(float), cell["y_pred"].to_numpy(float))
        if a == a:
            vals.append(a)
    return (float(np.mean(vals)) if vals else float("nan")), "auc"


def summarize(df: pd.DataFrame, m6, target: str) -> dict:
    """Per-model OOF accuracy, marginal coverage (split vs Mondrian), and rho."""
    out: Dict[str, dict] = {}
    binary = set(np.unique(df["y_true"].dropna())) <= {0.0, 1.0}
    for model in sorted(df["model"].unique()):
        sub = df[df.model == model]
        y = sub["y_true"].to_numpy(float)
        m = ~np.isnan(y)
        cov_split = float(np.mean((y[m] >= sub["lo_split"].to_numpy(float)[m] - 1e-9) &
                                  (y[m] <= sub["hi_split"].to_numpy(float)[m] + 1e-9)))
        cov_mond = float(np.mean((y[m] >= sub["lo_mond"].to_numpy(float)[m] - 1e-9) &
                                 (y[m] <= sub["hi_mond"].to_numpy(float)[m] + 1e-9)))
        # worst-class Mondrian coverage
        worst = 1.0
        for cls, g in sub[m].groupby("class_canonical"):
            yy = g["y_true"].to_numpy(float)
            cc = float(np.mean((yy >= g["lo_mond"].to_numpy(float) - 1e-9) &
                               (yy <= g["hi_mond"].to_numpy(float) + 1e-9)))
            worst = min(worst, cc)
        acc, metric_name = _oof_metric(sub, binary)
        out[model] = {
            "oof_metric": metric_name,
            "oof_value": round(acc, 4) if acc == acc else None,
            "oof_metric_note": ("mean AUC over (seed, fold, tool) cells -- matches "
                                "stage 08 RQ1 exactly" if binary else "pooled RMSE"),
            "coverage_split_marginal": round(cov_split, 4),
            "coverage_mondrian_marginal": round(cov_mond, 4),
            "coverage_mondrian_worst_class": round(worst, 4),
            "n_predictions": int(len(sub)),
        }
    # rho from the best available model's OOF miss matrix (conditional on FNR).
    ref = "lightgbm" if "lightgbm" in out else sorted(out)[0]
    rho = _estimate_rho_from_predictions(df[df.model == ref], m6)
    return {"per_model": out, "reference_model": ref, "miss_correlation_rho": rho}


def _estimate_rho_from_predictions(sub: pd.DataFrame, m6) -> Optional[float]:
    """Estimate miss-correlation from OOF predictions (conditional on predicted FNR)."""
    piv_y = sub.pivot_table(index=["contract_id", "class_canonical"],
                            columns="tool", values="y_true", aggfunc="first")
    piv_p = sub.pivot_table(index=["contract_id", "class_canonical"],
                            columns="tool", values="y_pred", aggfunc="first")
    piv_y, piv_p = piv_y.align(piv_p, join="inner")
    Y = piv_y.to_numpy(float)
    P = piv_p.to_numpy(float)
    keep = ~np.isnan(Y).any(axis=1)
    Y, P = Y[keep], P[keep]
    if Y.shape[0] < 5 or Y.shape[1] < 2:
        return None
    miss = (Y < 0.5).astype(float)
    fnr = np.clip(1.0 - P, 1e-6, 1 - 1e-6)
    return float(m6.estimate_miss_correlation(miss, fnr_matrix=fnr))


# ==============================================================================
# Orchestration + dump
# ==============================================================================
def _cid_to_dataset(features: pd.DataFrame) -> Dict[str, str]:
    return dict(zip(features["contract_id"], features["dataset"]))


def _check_against_registry(registry_path: Optional[str], features: pd.DataFrame,
                            labels: pd.DataFrame) -> Optional[dict]:
    """Validate features/labels against the stage-01 registry; return provenance.

    Catches the silent-corruption case where features or labels were built from a
    different corpus snapshot than the registry (e.g. a re-download changed the
    contract set): contract_ids must be a subset of the registry, and each
    contract's base_id must agree, since base_id is the grouping key that makes
    the whole evaluation leakage-safe. Returns the corpus fingerprint for
    train_meta.json, or None when no registry is supplied.
    """
    if not registry_path or not Path(registry_path).exists():
        return None
    reg = pd.read_parquet(registry_path, engine="pyarrow")
    reg_ids = set(reg["contract_id"])
    for name, frame in (("features", features), ("labels", labels)):
        unknown = set(frame["contract_id"]) - reg_ids
        if unknown:
            raise AssertionError(
                f"{name} contain {len(unknown)} contract_id(s) absent from the "
                f"registry (corpus drift): {sorted(map(str, unknown))[:5]} ...")
    reg_base = dict(zip(reg["contract_id"], reg["base_id"]))
    mismatch = [c for c, b in zip(features["contract_id"], features["base_id"])
                if reg_base.get(c, b) != b]
    if mismatch:
        raise AssertionError(
            f"base_id disagrees with the registry for {len(mismatch)} contract(s); "
            f"grouped splits would not be leakage-safe: {mismatch[:5]} ...")
    meta_path = Path(registry_path).with_name("registry_meta.json")
    prov: dict = {"registry": str(registry_path), "n_registry_contracts": int(len(reg))}
    if meta_path.exists():
        try:
            rm = json.loads(meta_path.read_text(encoding="utf-8"))
            # Stage 01 writes 'corpus_fingerprint_sha256'; accept the short alias
            # too so provenance survives a future rename rather than silently
            # recording null.
            prov["corpus_fingerprint"] = (rm.get("corpus_fingerprint_sha256")
                                          or rm.get("corpus_fingerprint"))
            prov["registry_schema_version"] = rm.get("schema_version")
            prov["registry_generated_utc"] = rm.get("generated_utc")
        except (json.JSONDecodeError, OSError):
            pass
    return prov


def do_train(args) -> int:
    src_dir = Path(__file__).resolve().parent
    m4, m5, m6 = load_stages(src_dir)

    features = pd.read_parquet(args.features, engine="pyarrow")
    labels = pd.read_parquet(args.labels, engine="pyarrow")
    provenance = _check_against_registry(args.registry, features, labels)
    cid2ds = _cid_to_dataset(features)
    graphs_by_id = None
    want_gnn = any(x in ("hetero_gnn", "gnn") for x in args.models)
    if args.graphs and Path(args.graphs).exists() and want_gnn:
        graphs_by_id = m4.load_graphs_jsonl(Path(args.graphs))

    try:
        import torch  # noqa: F401
        has_torch = torch is not None
    except Exception:
        has_torch = False
    if want_gnn and not has_torch:
        sys.stderr.write("[train] torch not installed; skipping the GNN model.\n")

    ds = m4.assemble_dataset(features, labels, target=args.target,
                             graphs_by_id=graphs_by_id)
    seeds = list(range(args.seeds))
    t0 = time.time()
    preds, splits = run_cv(ds, cid2ds, m4, m5, args.models, seeds,
                           args.folds, args.alpha, has_torch)
    shift = run_cross_benchmark(features, labels, cid2ds, m4, m5, args.models,
                                seeds, args.alpha, args.target, has_torch, graphs_by_id)
    elapsed = time.time() - t0

    summary = summarize(preds, m6, args.target)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(out_dir / "predictions.parquet", engine="pyarrow", index=False)
    shift.to_parquet(out_dir / "predictions_shift.parquet", engine="pyarrow", index=False)
    (out_dir / "splits.json").write_text(json.dumps(splits, indent=1), encoding="utf-8")
    meta = {
        "schema_version": SCHEMA_VERSION,
        "generator_version": __version__,
        "config": {"target": args.target, "alpha": args.alpha, "folds": args.folds,
                   "seeds": seeds, "models": list(args.models)},
        "n_instances": int(ds.n), "n_tools": len(ds.tool_names),
        "tools": list(ds.tool_names),
        "elapsed_s": round(elapsed, 2),
        "summary": summary,
        "n_shift_rows": int(len(shift)),
    }
    if provenance:
        meta["corpus_provenance"] = provenance
    (out_dir / "train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Wrote {out_dir}/predictions.parquet  ({len(preds)} rows, "
          f"{ds.n} instances x {len(ds.tool_names)} tools)")
    print(f"  cross-benchmark rows: {len(shift)}  |  elapsed {elapsed:.1f}s")
    for model, s in summary["per_model"].items():
        print(f"  {model:10} {s['oof_metric']}={s['oof_value']}  "
              f"cov(split)={s['coverage_split_marginal']:.3f}  "
              f"cov(mondrian)={s['coverage_mondrian_marginal']:.3f}  "
              f"worst-class={s['coverage_mondrian_worst_class']:.3f}")
    print(f"  miss-correlation rho = {summary['miss_correlation_rho']}")
    return 0


# ==============================================================================
# Hermetic self-test (synthetic features + labels; no torch, no Docker)
# ==============================================================================
def _synth_frames(n_contracts: int = 120, seed: int = 0):
    """Feature-dependent binary labels over synthetic contracts, base_id-grouped."""
    rng = np.random.default_rng(seed)
    classes = ["arithmetic", "reentrancy", "timestamp_dependency"]
    tools = [f"tool{i}" for i in range(4)]
    # base_id groups: 3 contracts per base
    contract_ids, base_ids, cls_list = [], [], []
    for c in range(n_contracts):
        contract_ids.append(f"c{c}")
        base_ids.append(f"b{c // 3}")
        cls_list.append(classes[c % len(classes)])
    f1 = rng.normal(size=n_contracts)
    f2 = rng.uniform(0, 5, size=n_contracts)
    features = pd.DataFrame({
        "contract_id": contract_ids, "dataset": ["solidifi"] * n_contracts,
        "base_id": base_ids, "parse_method": ["ast"] * n_contracts,
        "f_sig": f1, "f_aux": f2})
    # curated block so cross-benchmark has a test set
    m = 30
    cur_ids = [f"u{c}" for c in range(m)]
    fc1 = rng.normal(size=m)
    features = pd.concat([features, pd.DataFrame({
        "contract_id": cur_ids, "dataset": ["sb_curated"] * m,
        "base_id": cur_ids, "parse_method": ["ast"] * m,
        "f_sig": fc1, "f_aux": rng.uniform(0, 5, m)})], ignore_index=True)

    rows = []
    tool_off = rng.uniform(-0.5, 0.5, len(tools))
    for cid, cls, sig in zip(contract_ids + cur_ids,
                             cls_list + [classes[i % 3] for i in range(m)],
                             list(f1) + list(fc1)):
        base = cid if cid.startswith("u") else f"b{int(cid[1:]) // 3}"
        for ti, tool in enumerate(tools):
            p = 1 / (1 + np.exp(-(1.4 * sig + tool_off[ti])))
            det = rng.uniform() < p
            rows.append((cid, "solidifi" if cid.startswith("c") else "sb_curated",
                         base, cls, tool, bool(det)))
    labels = pd.DataFrame(rows, columns=["contract_id", "dataset", "base_id",
                                         "class_canonical", "tool", "detected"])
    labels["detected"] = labels["detected"].astype("boolean")
    return features, labels


def run_selftest() -> int:
    print(f"RELIANT 07_train self-test (v{__version__})")
    src_dir = Path(__file__).resolve().parent
    m4, m5, m6 = load_stages(src_dir)
    features, labels = _synth_frames()
    cid2ds = _cid_to_dataset(features)
    ds = m4.assemble_dataset(features, labels, target="detected")

    # --- grouped CV produces OOF predictions with no base_id leakage -----------
    preds, splits = run_cv(ds, cid2ds, m4, m5, ["constant", "ridge", "lightgbm"],
                           seeds=[0, 1], k=4, alpha=0.1, has_torch=False)
    assert list(preds.columns) == list(PRED_COLUMNS)
    # every instance appears once per (seed, model) as OOF
    n_inst = ds.n
    per = preds.groupby(["seed", "model"]).apply(
        lambda g: g[["contract_id", "class_canonical"]].drop_duplicates().shape[0],
        include_groups=False)
    assert (per == n_inst).all(), f"OOF coverage incomplete: {per.to_dict()}"
    # split record: train/cal/test base_ids disjoint
    for key, rec in splits.items():
        s_tr, s_cal, s_te = set(rec["train_base_ids"]), set(rec["cal_base_ids"]), set(rec["test_base_ids"])
        assert not (s_tr & s_te) and not (s_cal & s_te) and not (s_tr & s_cal), key

    # --- determinism: identical seeds -> identical predictions -----------------
    preds2, _ = run_cv(ds, cid2ds, m4, m5, ["constant", "ridge", "lightgbm"],
                       seeds=[0, 1], k=4, alpha=0.1, has_torch=False)
    assert preds.equals(preds2), "CV is not reproducible across identical runs"

    # --- conformal certificates achieve coverage out-of-fold -------------------
    summ = summarize(preds, m6, "detected")
    lgb = summ["per_model"]["lightgbm"]
    print(f"  lightgbm OOF AUC={lgb['oof_value']} cov(split)={lgb['coverage_split_marginal']:.3f} "
          f"cov(mondrian)={lgb['coverage_mondrian_marginal']:.3f}")
    assert lgb["oof_value"] > 0.6, "predictor failed to learn OOF signal"
    assert lgb["coverage_split_marginal"] >= 0.85, "split conformal under-covers OOF"
    assert lgb["coverage_mondrian_marginal"] >= 0.85, "Mondrian under-covers OOF"
    assert summ["miss_correlation_rho"] is not None

    # --- fnr_upper is a valid one-sided bound (>= realized miss on average) -----
    m = ~preds["y_true"].isna()
    realized_miss = (preds.loc[m, "y_true"] < 0.5).to_numpy(float)
    fnr_up = preds.loc[m, "fnr_upper"].to_numpy(float)
    # coverage: P(miss_indicator <= fnr_upper) should be high (upper bound holds)
    frac_ok = float(np.mean(realized_miss <= fnr_up + 1e-9))
    print(f"  one-sided FNR bound holds on {frac_ok:.3f} of cells (target ~0.90)")
    assert frac_ok >= 0.85, "FNR upper bound violated too often"

    # --- cross-benchmark produces curated predictions --------------------------
    shift = run_cross_benchmark(features, labels, cid2ds, m4, m5,
                                ["ridge", "lightgbm"], seeds=[0], alpha=0.1,
                                target="detected", has_torch=False, graphs_by_id=None)
    assert len(shift) > 0 and set(shift["dataset"]) == {"sb_curated"}
    print(f"  cross-benchmark rows: {len(shift)} (all curated)")

    # --- split guards: degenerate k, non-starved train, explicit leakage error --
    g_small = np.array(["b0", "b0", "b1", "b1"], dtype=object)
    folds_small = group_kfold(g_small, k=10, seed=0)      # k clamped to 2 groups
    assert len(folds_small) == 2
    assert all(te.size > 0 and dev.size > 0 for dev, te in folds_small), \
        "no fold may have an empty test or dev set"
    tr1, cal1 = group_holdout(np.arange(2), np.array(["b0", "b0"], dtype=object),
                              0.25, 0)
    assert tr1.size == 2 and cal1.size == 0, "single dev group must keep training data"
    tr2, cal2 = group_holdout(np.arange(4), g_small, 0.5, 0)
    assert tr2.size > 0 and cal2.size > 0 and \
        not (set(g_small[tr2]) & set(g_small[cal2])), "train/cal must be group-disjoint"
    try:                                                   # guard survives -O
        _assert_no_leakage(g_small, np.array([0, 1]), np.array([2]), np.array([1, 2]),
                           "unit-test")
        raise SystemExit("FAIL: leakage guard did not fire")
    except AssertionError as exc:
        assert "leakage" in str(exc)
    print("  split guards: k clamped, train never starved, leakage raises.")

    # --- unknown model vs stage-09 model are distinguished ---------------------
    try:
        make_predictor(m4, "not_a_model", 0)
        raise SystemExit("FAIL: unknown model accepted")
    except ValueError as exc:
        assert "supported" in str(exc)
    assert make_predictor(m4, "satzilla", 0) is not None, "stage-09 model must load"

    # --- registry consistency check catches corpus drift -----------------------
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        rp = Path(tmp) / "registry.parquet"
        reg_ok = pd.DataFrame({
            "contract_id": features["contract_id"],
            "base_id": features["base_id"],
            "dataset": features["dataset"]})
        reg_ok.to_parquet(rp, engine="pyarrow", index=False)
        (Path(tmp) / "registry_meta.json").write_text(
            json.dumps({"corpus_fingerprint_sha256": "abc123",
                        "schema_version": "x"}),
            encoding="utf-8")
        prov = _check_against_registry(str(rp), features, labels)
        assert prov["corpus_fingerprint"] == "abc123"
        assert _check_against_registry(None, features, labels) is None
        reg_bad = reg_ok.copy()
        reg_bad.loc[0, "base_id"] = "WRONG"
        reg_bad.to_parquet(rp, engine="pyarrow", index=False)
        try:
            _check_against_registry(str(rp), features, labels)
            raise SystemExit("FAIL: base_id drift not detected")
        except AssertionError as exc:
            assert "base_id" in str(exc)
        reg_short = reg_ok.iloc[1:].copy()
        reg_short.to_parquet(rp, engine="pyarrow", index=False)
        try:
            _check_against_registry(str(rp), features, labels)
            raise SystemExit("FAIL: missing contract not detected")
        except AssertionError as exc:
            assert "absent from the registry" in str(exc)
    print("  registry check: fingerprint recorded, base_id + corpus drift rejected.")

    # --- cross-file tripwire: PRED_COLUMNS matches stage 08 --------------------
    m8 = load_stage(src_dir / "08_evaluate.py", "reliant_stage08")
    assert tuple(m8.PRED_COLUMNS) == PRED_COLUMNS, \
        "PRED_COLUMNS drift between stages 07 and 08"
    # ...and the OOF metric is computed identically, so train_meta.json and
    # results/rq1_prediction_accuracy.json can never disagree for any model.
    rq1 = m8.rq1_prediction_accuracy(preds)["per_model"]
    for model, s in summ["per_model"].items():
        theirs = rq1[model]["mean_tool_auc"]["mean"]
        # oof_value is stored rounded to 4 dp, so compare at that resolution.
        assert abs(s["oof_value"] - round(theirs, 4)) < 1e-9, (
            f"{model}: stage-07 OOF AUC {s['oof_value']} != stage-08 RQ1 {theirs}")
    assert abs(summ["per_model"]["constant"]["oof_value"] - 0.5) < 0.05, \
        "base-rate model must sit at chance; pooled AUC would inflate it"
    print(f"  cross-file sync: PRED_COLUMNS == stage 08; OOF AUC identical for "
          f"{len(summ['per_model'])} models (base-rate at chance).")

    print("  all invariants passed.")
    return 0


# ==============================================================================
# CLI
# ==============================================================================
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Reproducibility spine (stage 07).")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--train", action="store_true")
    mode.add_argument("--selftest", action="store_true")
    p.add_argument("--registry", type=str, default="data/registry.parquet")
    p.add_argument("--features", type=str, default="artifacts/features.parquet")
    p.add_argument("--labels", type=str, default="artifacts/labels.parquet")
    p.add_argument("--graphs", type=str, default="artifacts/graphs.jsonl")
    p.add_argument("--target", type=str, default="detected")
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--models", nargs="+",
                   default=["constant", "ridge", "lightgbm"],
                   help="Subset of: " + " ".join(SUPPORTED_MODELS) +
                        " (hetero_gnn requires torch and is skipped without it)")
    p.add_argument("--out", type=str, default="artifacts")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.train:
        return do_train(args)
    return run_selftest()


if __name__ == "__main__":
    raise SystemExit(main())
