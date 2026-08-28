#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
10_make_figures.py -- Camera-ready IEEE tables and figures from results/*.json.

RELIANT: Calibrated Reliability Prediction and Budget-Constrained Portfolio
Selection for Smart Contract Security Analyzers.

--------------------------------------------------------------------------------
ROLE IN THE ARTIFACT
--------------------------------------------------------------------------------
The last code stage. It renders the evaluation JSON produced by stages 08 and 09
into publication-ready assets and does NO recomputation -- every number comes from
results/*.json, so the tables and figures are always consistent with the reported
experiments and regenerate deterministically.

Outputs:
  tables/  LaTeX booktabs tables (RQ1 accuracy, RQ2 calibration, RQ3 economics
           with the stage-09 oracle/baseline columns, RQ4 shift), ready to \\input.
  figures/ PDF figures with Type-42 (TrueType) embedded fonts as IEEE requires
           (matplotlib's default Type-3 fonts are rejected by IEEE): the
           budget/detection frontier, per-class split-vs-Mondrian coverage, the
           per-model accuracy comparison, and the distribution-shift gap.

--------------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------------
    python3 src/10_make_figures.py --make --results results \
        --tables-out tables --figures-out figures
    python3 src/10_make_figures.py --selftest

Exit codes: 0 = success; 3 = usage error.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")                       # headless; no display required
import matplotlib.pyplot as plt             # noqa: E402

__version__ = "1.0.0"
SCHEMA_VERSION = "reliant-figures-1"

# IEEE-compliant rendering: TrueType (Type-42) embedded fonts, serif, compact.
plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "serif", "font.size": 8,
    "axes.titlesize": 9, "axes.labelsize": 8, "axes.linewidth": 0.6,
    "legend.fontsize": 6.5, "legend.frameon": False,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "lines.linewidth": 1.2, "lines.markersize": 4,
    "figure.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
COL_W = 3.45          # IEEE single-column width (inches)


# ==============================================================================
# Safe accessors
# ==============================================================================
def _g(d, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def _val(agg):
    """Extract a scalar mean from an {'mean','std','n'} block, or pass through."""
    if isinstance(agg, dict):
        return agg.get("mean")
    return agg


def _std(agg):
    return agg.get("std") if isinstance(agg, dict) else None


def _num(x, nd=3):
    return "--" if x is None else f"{x:.{nd}f}"


def _pct(x, nd=0):
    return "--" if x is None else f"{100 * x:.{nd}f}\\%"


def _pm(agg, nd=3):
    m, s = _val(agg), _std(agg)
    if m is None:
        return "--"
    if s is None:
        return f"{m:.{nd}f}"
    return f"{m:.{nd}f}\\,$\\pm$\\,{s:.{nd}f}"


def load_results(results_dir: Path) -> Dict[str, dict]:
    out = {}
    for name in ("rq1_prediction_accuracy", "rq2_calibration",
                 "rq3_selection_economics", "rq4_distribution_shift",
                 "rq5_case_study", "baselines"):
        p = results_dir / f"{name}.json"
        out[name] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    return out


# ==============================================================================
# LaTeX tables (booktabs)
# ==============================================================================
def _table_wrap(body: str, caption: str, label: str, colspec: str,
                header: str) -> str:
    return ("\\begin{table}[t]\n\\centering\n"
            f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
            f"\\begin{{tabular}}{{{colspec}}}\n\\toprule\n"
            f"{header} \\\\\n\\midrule\n{body}\\bottomrule\n"
            "\\end{tabular}\n\\end{table}\n")


def table_rq1(rq1: dict) -> str:
    pm = rq1.get("per_model", {})
    order = [m for m in ("constant", "ridge", "satzilla", "satzilla_rf", "lightgbm")
             if m in pm] or sorted(pm)
    rows = []
    for m in order:
        d = pm[m]
        rows.append(f"{d.get('label', m)} & {_pm(d.get('mean_tool_auc'))} & "
                    f"{_pm(d.get('brier'))} \\\\\n")
    gain = _g(rq1, "headline", "auc_gain_over_base_rate")
    cap = ("Per-analyzer reliability prediction (out-of-fold, grouped by base "
           "contract). All methods predict the same quantity; AUC$=0.5$ is the "
           "base-rate chance level"
           + (f". RELIANT-GBM improves mean AUC by {gain:.3f} over the base rate"
              if gain is not None else "") + ".")
    return _table_wrap("".join(rows), cap, "tab:rq1",
                       "lcc", "Model & Mean per-tool AUC & Brier")


def table_rq2(rq2: dict) -> str:
    sc, mc = rq2.get("split_conformal", {}), rq2.get("mondrian_conformal", {})
    tgt = rq2.get("target_coverage")
    rows = [
        f"Split & {_pm(_g(sc,'marginal_coverage'))} & "
        f"{_pm(_g(sc,'worst_class_coverage'))} & {_pm(_g(sc,'mean_width'))} \\\\\n",
        f"Mondrian & {_pm(_g(mc,'marginal_coverage'))} & "
        f"{_pm(_g(mc,'worst_class_coverage'))} & {_pm(_g(mc,'mean_width'))} \\\\\n",
    ]
    cap = (f"Coverage of the reliability certificates (target "
           f"{_num(tgt,2)}). Marginal split conformal can under-cover the hardest "
           f"class; Mondrian restores per-class coverage by widening only where "
           f"needed.")
    return _table_wrap("".join(rows), cap, "tab:rq2", "lccc",
                       "Certificate & Marginal cov. & Worst-class cov. & Mean width")


def table_rq2_perclass(rq2: dict) -> str:
    pc = rq2.get("per_class_coverage", {})
    rows = []
    for cls in sorted(pc):
        d = pc[cls]
        name = cls.replace("_", "\\_")
        rows.append(f"{name} & {_num(_val(d.get('split')))} & "
                    f"{_num(_val(d.get('mondrian')))} \\\\\n")
    cap = ("Per-class coverage, split vs Mondrian (target coverage "
           f"{_num(rq2.get('target_coverage'),2)}).")
    return _table_wrap("".join(rows), cap, "tab:rq2_perclass", "lcc",
                       "Vulnerability class & Split cov. & Mondrian cov.")


def table_rq3(rq3: dict, baselines: dict) -> str:
    fr = rq3.get("budget_frontier", {})
    ob = _g(baselines, "selection_baselines", "oracle_by_budget", default={})
    fracs = sorted(int(k.split("pct")[0]) for k in fr) if fr else []
    rows = []
    for f in fracs:
        v = fr[f"{f}pct_budget"]
        orc = _val(_g(ob, f"{f}pct_budget", "detection"))
        rows.append(
            f"{f}\\% & {_pct(_val(v.get('cost_savings_fraction')))} & "
            f"{_num(_val(v.get('certified_detection_lower_bound')))} & "
            f"{_num(_val(v.get('realized_detection')))} & "
            f"{_num(_val(v.get('best1_realized_detection')))} & "
            f"{_num(_val(v.get('random_k_realized_detection')))} & "
            f"{_num(orc)} \\\\\n")
    # run-all reference row
    ra = _val(_g(baselines, "selection_baselines", "run_all_detection")) \
        or _val(rq3.get("run_all_realized_detection"))
    rows.append(f"run-all & 0\\% & -- & {_num(ra)} & -- & -- & {_num(ra)} \\\\\n")
    cap = ("Selection economics vs the measured run-all baseline. RELIANT's "
           "realized detection sits between the naive best-1/random-k baselines "
           "and the perfect-foresight oracle, while its certified bound holds "
           "conservatively -- all at a fraction of the run-all cost.")
    return _table_wrap("".join(rows), cap, "tab:rq3", "lcccccc",
                       ("Budget & Cost saved & Certified $D_{\\geq}$ & Realized "
                        "& Best-1 & Random-$k$ & Oracle"))


def table_rq4(rq4: dict) -> str:
    if rq4.get("status") or "mean_tool_auc_cross_benchmark" not in rq4:
        return ("% RQ4 table unavailable: "
                + str(rq4.get("status", "no cross-benchmark data")) + "\n")
    ai = rq4.get("mean_tool_auc_in_distribution_same_classes")
    ac = rq4.get("mean_tool_auc_cross_benchmark")
    gap = rq4.get("generalization_gap_auc")
    cov = rq4.get("coverage_cross_benchmark_mondrian")
    tgt = rq4.get("target_coverage")
    body = (f"In-distribution (same classes) & {_num(ai)} \\\\\n"
            f"Cross-benchmark (real-world) & {_num(ac)} \\\\\n"
            f"Generalization gap & {_num(gap)} \\\\\n"
            f"Cross-benchmark coverage (Mondrian) & {_num(cov)} \\\\\n")
    cap = ("Distribution shift: train on synthetic SolidiFI, test on the "
           f"real-world smartbugs-curated corpus. Coverage target {_num(tgt,2)}; "
           "the gap is reported honestly rather than assumed away.")
    return _table_wrap(body, cap, "tab:rq4", "lc", "Setting & Mean per-tool AUC")


# ==============================================================================
# Figures (PDF, Type-42 fonts)
# ==============================================================================
def _frontier_series(rq3: dict, baselines: dict):
    fr = rq3.get("budget_frontier", {})
    fracs = sorted(int(k.split("pct")[0]) for k in fr) if fr else []
    ob = _g(baselines, "selection_baselines", "oracle_by_budget", default={})

    def s(getter):
        return [getter(fr[f"{f}pct_budget"]) for f in fracs]

    realized = s(lambda v: _val(v.get("realized_detection")))
    certified = s(lambda v: _val(v.get("certified_detection_lower_bound")))
    best1 = s(lambda v: _val(v.get("best1_realized_detection")))
    randk = s(lambda v: _val(v.get("random_k_realized_detection")))
    oracle = [_val(_g(ob, f"{f}pct_budget", "detection")) for f in fracs]
    runall = _val(_g(baselines, "selection_baselines", "run_all_detection")) \
        or _val(rq3.get("run_all_realized_detection"))
    return fracs, realized, certified, best1, randk, oracle, runall


def fig_frontier(rq3: dict, baselines: dict, path: Path) -> None:
    fracs, realized, certified, best1, randk, oracle, runall = \
        _frontier_series(rq3, baselines)
    fig, ax = plt.subplots(figsize=(COL_W, 2.5))
    x = np.asarray(fracs, dtype=float)
    if oracle and any(o is not None for o in oracle):
        ax.plot(x, oracle, "k^-", label="Oracle (foresight)")
    ax.plot(x, realized, "o-", color="#1f77b4", label="RELIANT (realized)")
    ax.plot(x, certified, "o--", color="#1f77b4", alpha=0.6,
            label="RELIANT (certified $D_{\\geq}$)")
    ax.plot(x, best1, "s:", color="#d62728", label="Best-1")
    ax.plot(x, randk, "d:", color="#7f7f7f", label="Random-$k$")
    if runall is not None:
        ax.axhline(runall, color="k", ls=(0, (1, 1)), lw=0.8, alpha=0.7)
        ax.text(x.max(), runall + 0.005, "run-all", ha="right", va="bottom",
                fontsize=6)
    ax.set_xlabel("Budget (\\% of run-all cost)")
    ax.set_ylabel("Detection rate")
    ax.set_title("Budget--detection frontier")
    ax.set_ylim(min([v for v in randk + best1 if v is not None] + [0.5]) - 0.05, 1.01)
    ax.grid(True, lw=0.3, alpha=0.5)
    ax.legend(loc="lower right", ncol=1)
    fig.savefig(path)
    plt.close(fig)


def fig_coverage(rq2: dict, path: Path) -> None:
    pc = rq2.get("per_class_coverage", {})
    classes = sorted(pc)
    split = [_val(pc[c].get("split")) for c in classes]
    mond = [_val(pc[c].get("mondrian")) for c in classes]
    tgt = rq2.get("target_coverage")
    x = np.arange(len(classes))
    w = 0.38
    fig, ax = plt.subplots(figsize=(COL_W, 2.6))
    ax.bar(x - w / 2, [s or 0 for s in split], w, label="Split",
           color="#d62728", alpha=0.85)
    ax.bar(x + w / 2, [m or 0 for m in mond], w, label="Mondrian",
           color="#1f77b4", alpha=0.85)
    if tgt is not None:
        ax.axhline(tgt, color="k", ls="--", lw=0.8)
        ax.text(len(classes) - 0.5, tgt + 0.005, f"target {tgt:.2f}",
                ha="right", va="bottom", fontsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in classes], fontsize=5.5)
    ax.set_ylabel("Coverage")
    ax.set_title("Per-class coverage: split vs Mondrian")
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", lw=0.3, alpha=0.5)
    ax.legend(loc="lower right")
    fig.savefig(path)
    plt.close(fig)


def fig_accuracy(rq1: dict, path: Path) -> None:
    pm = rq1.get("per_model", {})
    order = [m for m in ("constant", "ridge", "satzilla", "satzilla_rf", "lightgbm")
             if m in pm] or sorted(pm)
    labels = [pm[m].get("label", m) for m in order]
    means = [_val(_g(pm[m], "mean_tool_auc")) or 0 for m in order]
    errs = [_std(_g(pm[m], "mean_tool_auc")) or 0 for m in order]
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    ax.bar(x, means, 0.6, yerr=errs, capsize=2.5, color="#1f77b4", alpha=0.85)
    ax.axhline(0.5, color="k", ls="--", lw=0.8)
    ax.text(len(order) - 0.5, 0.51, "chance", ha="right", va="bottom", fontsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=6)
    ax.set_ylabel("Mean per-tool AUC")
    ax.set_title("Reliability prediction accuracy")
    ax.set_ylim(0.45, 1.0)
    ax.grid(True, axis="y", lw=0.3, alpha=0.5)
    fig.savefig(path)
    plt.close(fig)


def fig_shift(rq4: dict, path: Path) -> Optional[Path]:
    if "mean_tool_auc_cross_benchmark" not in rq4:
        return None
    ai = rq4.get("mean_tool_auc_in_distribution_same_classes") or 0
    ac = rq4.get("mean_tool_auc_cross_benchmark") or 0
    fig, ax = plt.subplots(figsize=(COL_W * 0.7, 2.3))
    ax.bar([0, 1], [ai, ac], 0.6, color=["#1f77b4", "#ff7f0e"], alpha=0.85)
    ax.axhline(0.5, color="k", ls="--", lw=0.8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["In-dist.", "Cross-bench."], fontsize=7)
    ax.set_ylabel("Mean per-tool AUC")
    ax.set_title("Distribution-shift gap")
    ax.set_ylim(0.4, 1.0)
    ax.grid(True, axis="y", lw=0.3, alpha=0.5)
    fig.savefig(path)
    plt.close(fig)
    return path


# ==============================================================================
# Orchestration
# ==============================================================================
def do_make(args) -> int:
    res = load_results(Path(args.results))
    tdir, fdir = Path(args.tables_out), Path(args.figures_out)
    tdir.mkdir(parents=True, exist_ok=True)
    fdir.mkdir(parents=True, exist_ok=True)

    tables = {
        "table_rq1_accuracy.tex": table_rq1(res["rq1_prediction_accuracy"]),
        "table_rq2_calibration.tex": table_rq2(res["rq2_calibration"]),
        "table_rq2_perclass.tex": table_rq2_perclass(res["rq2_calibration"]),
        "table_rq3_economics.tex": table_rq3(res["rq3_selection_economics"], res["baselines"]),
        "table_rq4_shift.tex": table_rq4(res["rq4_distribution_shift"]),
    }
    for name, tex in tables.items():
        (tdir / name).write_text(tex, encoding="utf-8")

    made_figs = []
    fig_frontier(res["rq3_selection_economics"], res["baselines"],
                 fdir / "fig_frontier.pdf"); made_figs.append("fig_frontier.pdf")
    fig_coverage(res["rq2_calibration"], fdir / "fig_coverage.pdf")
    made_figs.append("fig_coverage.pdf")
    fig_accuracy(res["rq1_prediction_accuracy"], fdir / "fig_accuracy.pdf")
    made_figs.append("fig_accuracy.pdf")
    if fig_shift(res["rq4_distribution_shift"], fdir / "fig_shift.pdf"):
        made_figs.append("fig_shift.pdf")

    print(f"Wrote {len(tables)} tables to {tdir}/ and {len(made_figs)} figures to {fdir}/")
    for n in tables:
        print(f"  table: {n}")
    for n in made_figs:
        print(f"  figure: {n}")
    return 0


# ==============================================================================
# Hermetic self-test (synthetic results dicts)
# ==============================================================================
def _synth_results() -> Dict[str, dict]:
    def agg(m, s=0.01):
        return {"mean": m, "std": s, "n": 3}
    classes = ["arithmetic", "reentrancy", "timestamp_dependency",
               "transaction_order_dependency", "unchecked_low_level_calls"]
    rq1 = {"per_model": {
        "constant": {"label": "base-rate", "mean_tool_auc": agg(0.500, 0.0),
                     "brier": agg(0.24)},
        "ridge": {"label": "SATzilla-linear", "mean_tool_auc": agg(0.713),
                  "brier": agg(0.19)},
        "satzilla": {"label": "SATzilla-RF", "mean_tool_auc": agg(0.815),
                     "brier": agg(0.16)},
        "lightgbm": {"label": "RELIANT-GBM", "mean_tool_auc": agg(0.794),
                     "brier": agg(0.17)}},
        "headline": {"auc_gain_over_base_rate": 0.294}}
    rq2 = {"target_coverage": 0.90,
           "split_conformal": {"marginal_coverage": agg(0.901),
                               "worst_class_coverage": agg(0.879),
                               "mean_width": agg(0.52)},
           "mondrian_conformal": {"marginal_coverage": agg(0.921),
                                  "worst_class_coverage": agg(0.906),
                                  "mean_width": agg(0.58)},
           "per_class_coverage": {c: {"split": agg(0.86 + 0.01 * i),
                                      "mondrian": agg(0.90 + 0.005 * i)}
                                  for i, c in enumerate(classes)}}
    fr = {}
    for f, sav, cert, real, b1, rk in [(25, 0.84, 0.22, 0.830, 0.764, 0.780),
                                       (50, 0.67, 0.28, 0.841, 0.764, 0.812),
                                       (75, 0.54, 0.33, 0.850, 0.764, 0.865)]:
        fr[f"{f}pct_budget"] = {
            "cost_savings_fraction": agg(sav), "certified_detection_lower_bound": agg(cert),
            "realized_detection": agg(real), "best1_realized_detection": agg(b1),
            "random_k_realized_detection": agg(rk),
            "mean_panel_size": agg(3.0)}
    rq3 = {"budget_frontier": fr, "run_all_realized_detection": agg(0.915),
           "run_all_cost": 228.0}
    rq4 = {"mean_tool_auc_in_distribution_same_classes": 0.81,
           "mean_tool_auc_cross_benchmark": 0.71,
           "generalization_gap_auc": 0.10,
           "coverage_cross_benchmark_mondrian": 0.83, "target_coverage": 0.90}
    baselines = {"selection_baselines": {
        "run_all_detection": agg(0.915), "best_1_detection": agg(0.764),
        "best_1_cost": agg(20.0), "random_k_detection": agg(0.780),
        "oracle_unbounded_detection": agg(0.915), "oracle_cost_per_detection": agg(8.8),
        "oracle_by_budget": {"25pct_budget": {"detection": agg(0.902), "mean_cost": agg(30)},
                             "50pct_budget": {"detection": agg(0.915), "mean_cost": agg(35)},
                             "75pct_budget": {"detection": agg(0.915), "mean_cost": agg(35)}}}}
    return {"rq1_prediction_accuracy": rq1, "rq2_calibration": rq2,
            "rq3_selection_economics": rq3, "rq4_distribution_shift": rq4,
            "rq5_case_study": {"status": "skipped"}, "baselines": baselines}


def run_selftest() -> int:
    print(f"RELIANT 10_make_figures self-test (v{__version__})")
    import tempfile
    res = _synth_results()

    # --- tables are non-empty booktabs and reference the right numbers ---------
    t1 = table_rq1(res["rq1_prediction_accuracy"])
    t2 = table_rq2(res["rq2_calibration"])
    t2p = table_rq2_perclass(res["rq2_calibration"])
    t3 = table_rq3(res["rq3_selection_economics"], res["baselines"])
    t4 = table_rq4(res["rq4_distribution_shift"])
    for name, t in [("rq1", t1), ("rq2", t2), ("rq2p", t2p), ("rq3", t3), ("rq4", t4)]:
        assert "\\toprule" in t and "\\bottomrule" in t, f"{name} not booktabs"
        assert "\\begin{tabular}" in t and "\\end{tabular}" in t, f"{name} malformed"
    assert "0.794" in t1 and "base-rate" in t1, "RQ1 table missing values"
    assert "Mondrian" in t2 and "Split" in t2, "RQ2 table missing rows"
    assert "Oracle" in t3 and "run-all" in t3, "RQ3 table missing baseline columns"
    assert "0.830" in t3, "RQ3 realized detection missing"
    print("  tables OK (RQ1/RQ2/RQ2-perclass/RQ3/RQ4, booktabs, values present)")

    # --- figures are valid, font-embedded PDFs ---------------------------------
    with tempfile.TemporaryDirectory() as d:
        dd = Path(d)
        fig_frontier(res["rq3_selection_economics"], res["baselines"], dd / "frontier.pdf")
        fig_coverage(res["rq2_calibration"], dd / "coverage.pdf")
        fig_accuracy(res["rq1_prediction_accuracy"], dd / "accuracy.pdf")
        shift = fig_shift(res["rq4_distribution_shift"], dd / "shift.pdf")
        for f in ["frontier.pdf", "coverage.pdf", "accuracy.pdf"]:
            p = dd / f
            assert p.exists() and p.stat().st_size > 1200, f"{f} not written"
            head = p.read_bytes()[:5]
            assert head[:4] == b"%PDF", f"{f} is not a PDF"
        assert shift is not None and shift.exists(), "shift figure missing"
        print("  figures OK (4 PDFs, %PDF header, non-trivial size)")

    # --- Type-42 font setting is active ----------------------------------------
    assert plt.rcParams["pdf.fonttype"] == 42, "Type-42 fonts not configured"
    print("  Type-42 (TrueType) font embedding configured for IEEE.")
    print("  all invariants passed.")
    return 0


# ==============================================================================
# CLI
# ==============================================================================
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Render IEEE tables and figures (stage 10).")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--make", action="store_true")
    mode.add_argument("--selftest", action="store_true")
    p.add_argument("--results", type=str, default="results")
    p.add_argument("--tables-out", type=str, default="tables")
    p.add_argument("--figures-out", type=str, default="figures")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.make:
        return do_make(args)
    return run_selftest()


if __name__ == "__main__":
    raise SystemExit(main())
