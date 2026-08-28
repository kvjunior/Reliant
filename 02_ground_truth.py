#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_ground_truth.py -- Analyzer-reliability label core (the measured target).

RELIANT: Calibrated Reliability Prediction and Budget-Constrained Portfolio
Selection for Smart Contract Security Analyzers.

--------------------------------------------------------------------------------
ROLE IN THE ARTIFACT
--------------------------------------------------------------------------------
This stage turns raw analyzer output into the quantity RELIANT predicts: for
every (contract, tool, vulnerability class) it measures whether -- and how well
-- the tool detected the ground-truth vulnerability, and how long it took. These
per-instance reliabilities are the *labels* for stage 04 (prediction) and the
inputs to the portfolio guarantee in stage 06. It writes four artifacts:

    artifacts/labels.parquet        one row per (contract, tool, gt-class)
    artifacts/labels_wide.parquet   reliability matrix (contract x class) x tool
    artifacts/tool_timings.parquet  wall-clock duration per (contract, tool)
    artifacts/labels_meta.json      panel, tool versions, taxonomy map, coverage

It has two clearly separated halves:

  * LABEL CORE (pure Python, deterministic, runs anywhere, covered by --selftest):
    given the ground truth and the analyzers' normalized findings, compute the
    reliability metrics. This is the scientific content and is fully tested here.

  * SERVER DRIVER (--run-tools): a thin, documented wrapper that invokes the
    version-pinned SmartBugs 2.0 execution framework (Docker) to produce those
    findings once. It needs Docker + the tool images and therefore runs on the
    project server, not in a CPU sandbox. Its command construction is unit-tested
    even where the tools cannot be executed.

--------------------------------------------------------------------------------
WHAT "RELIABILITY" MEANS HERE (precise definitions)
--------------------------------------------------------------------------------
Fix a contract C, a tool T, and a ground-truth class K present in C with injected
/ annotated source lines L (|L| = m). Let R be T's findings on C whose taxonomy
classification (from SmartBugs' per-tool findings.yaml -> SWC/DASP -> canonical)
is relevant to K, and let D be the subset of R that carry a source line.

    detected      = 1 if |R| >= 1            (tool flagged class K anywhere on C)
    TP            = # of lines in L matched one-to-one to a line in D within
                    LINE_TOLERANCE (greedy nearest)
    FN            = m - TP
    FP_proxy      = |D| - TP                 (class-K alerts not at a known line)
    recall        = TP / m                   (per-instance detection rate)
    precision     = TP / |D|                 (|D| > 0)
    FNR           = 1 - recall               (the miss rate we chiefly predict)
    FPR_proxy     = 1 - precision            (honestly a proxy: true negatives are
                                              undefined at line granularity)

`detected` is tool-agnostic and always defined, so it is the primary modelling
target and the atom of the portfolio detection guarantee. Line-level recall /
precision refine it and are set to NaN when a tool reports relevant findings but
no source line (e.g. bytecode-level tools); they remain valid for the default
panel, whose eight tools all emit lines. FP_proxy is called a proxy because a
base contract may contain genuine class-K issues that were not injected /
annotated, and a real-world annotation set may be incomplete.

--------------------------------------------------------------------------------
WHY THIS ADDRESSES THE PRIOR REVIEW
--------------------------------------------------------------------------------
The measured target is analyzer *reliability*, not vulnerability presence, so no
tool alert is ever used as a model feature (that happens in stage 03/04) -- the
circularity that sank the earlier submission is impossible by construction. The
metrics are defined against the tool's own normalized taxonomy and scored only on
cells that actually have ground truth (each SolidiFI contract injects exactly one
class, so six of seven per-contract cells are structurally empty and are never
fabricated). Real wall-clock durations feed the honest run-all-tools baseline in
stage 06, replacing the earlier paper's undefended time-savings claim.

--------------------------------------------------------------------------------
GROUND TRUTH (verified structure; see stage 01)
--------------------------------------------------------------------------------
SolidiFI : buggy_contracts/<Class>/BugLog_<N>.csv -- injected lines are the
           integer `loc` column ONLY (the 'bug type' column is UTF-7-mangled);
           the class is the directory name. All injected bugs in a contract share
           that one class.
curated  : vulnerabilities.json -- per contract a list of {lines:[...], category};
           lines are grouped by canonical class.

--------------------------------------------------------------------------------
TOOL PANEL (default; version-pinned to the uploaded SmartBugs 2.0 snapshot)
--------------------------------------------------------------------------------
slither-0.11.3, mythril-0.24.8, oyente, smartcheck, securify2, conkas,
confuzzius. Five are the classic tools evaluated by SolidiFI / Durieux et al.
(continuity) and two are modern (recency); all seven emit source lines, all carry
a findings.yaml classification, and together they cover all seven canonical
classes with methodological diversity (static / symbolic / hybrid fuzzing).
Pinning versions defuses the "abandoned tools" criticism. semgrep-c3a9f40 is
deliberately NOT in the default panel: in the pinned snapshot its findings.yaml
classifies rules only to codes outside the seven canonical classes (empty /
SWC-124 / DASP-2), so it is structurally incapable of detecting any studied
class; it can be re-enrolled explicitly via --panel.

--------------------------------------------------------------------------------
TASK -> CONTRACT RESOLUTION (the round-trip contract between the two halves)
--------------------------------------------------------------------------------
--run-tools hands SmartBugs *absolute* file paths and writes run_manifest.json
mapping, for every contract, its absolute posix path AND its dataset-relative
path(s) to the contract_id. --build-labels resolves each recorded task filename
via, in order: (1) a direct manifest hit, (2) the longest path-suffix hit in the
manifest, (3) the longest path-suffix hit against registry relpaths (with and
without the 'dataset/' prefix), (4) a unique basename. The self-test round-trips
the exact writer format through the reader so the two halves cannot drift.

--------------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------------
    # (server) run the analyzers once via SmartBugs 2.0 to produce findings:
    python3 src/02_ground_truth.py --run-tools \
        --registry data/registry.parquet --uploads /mnt/user-data/uploads \
        --smartbugs /path/to/smartbugs --results-dir artifacts/sb_runs

    # (anywhere) build reliability labels from an existing results directory:
    python3 src/02_ground_truth.py --build-labels \
        --registry data/registry.parquet --uploads /mnt/user-data/uploads \
        --results-dir artifacts/sb_runs --tools-dir /path/to/smartbugs/tools \
        --out artifacts

    # hermetic self-test (no datasets, no Docker, no writes):
    python3 src/02_ground_truth.py --selftest

Exit codes: 0 = success; 2 = data/verification failure; 3 = usage error.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple

import pandas as pd
import yaml  # PyYAML; used only to read SmartBugs findings.yaml classification.

__version__ = "1.2.0"
SCHEMA_VERSION = "reliant-labels-1"

# ------------------------------------------------------------------------------
# Tool panel (version-pinned to the uploaded SmartBugs 2.0 snapshot)
# ------------------------------------------------------------------------------
# semgrep-c3a9f40 is intentionally excluded: in this snapshot its findings.yaml
# classifies rules only to codes outside the seven canonical classes (empty /
# SWC-124 / DASP-2), so it can never register a detection for any studied class
# (verified empirically: 0/7 class coverage). A structurally-dead panel member
# would only distort the run-all cost baseline. Re-enroll it via --panel if a
# future snapshot gains in-scope classifications.
DEFAULT_PANEL: Tuple[str, ...] = (
    "slither-0.11.3",
    "mythril-0.24.8",
    "oyente",
    "smartcheck",
    "securify2",
    "conkas",
    "confuzzius",
)

# Optional hand-authored classification for tools without a findings.yaml (e.g.
# manticore-0.3.7, whose finding names are decoded dynamically). Empty by default;
# the default panel is fully covered by findings.yaml.
MANUAL_CLASSIFICATION: Dict[str, Dict[str, str]] = {}

# ------------------------------------------------------------------------------
# Taxonomy: SWC / DASP codes -> canonical RELIANT classes
# ------------------------------------------------------------------------------
# SWC-104 (unchecked call return value) is the closest code for BOTH SolidiFI
# "Unchecked-Send" and "Unhandled-Exceptions"; tools do not distinguish them, so
# a SWC-104/DASP-4 finding is relevant to both canonical classes. DASP-2 (access
# control) is deliberately NOT mapped to tx_origin: it is far broader than the
# tx.origin authorization pattern, so only the specific SWC-115 code maps there.
SWC_TO_CANON: Dict[str, Set[str]] = {
    "SWC-101": {"arithmetic"},
    "SWC-107": {"reentrancy"},
    "SWC-116": {"timestamp_dependency"},
    "SWC-114": {"transaction_order_dependency"},
    "SWC-115": {"tx_origin"},
    "SWC-104": {"unchecked_low_level_calls", "unhandled_exceptions"},
}
DASP_TO_CANON: Dict[str, Set[str]] = {
    "DASP-3": {"arithmetic"},
    "DASP-1": {"reentrancy"},
    "DASP-8": {"timestamp_dependency"},
    "DASP-7": {"transaction_order_dependency"},
    "DASP-4": {"unchecked_low_level_calls", "unhandled_exceptions"},
}
CANONICAL_CLASSES: Tuple[str, ...] = (
    "arithmetic", "reentrancy", "timestamp_dependency",
    "transaction_order_dependency", "tx_origin",
    "unchecked_low_level_calls", "unhandled_exceptions",
)

LINE_TOLERANCE_DEFAULT = 0  # exact-line matching; injection logs are line-precise.

LABELS_COLUMNS: Tuple[str, ...] = (
    "contract_id", "dataset", "base_id", "class_canonical",
    "tool", "tool_version",
    "n_injected", "n_relevant_findings", "n_detected_lines",
    "tp", "fp_proxy", "fn",
    "recall", "precision", "fnr", "fpr_proxy",
    "detected", "detected_at_line",
    "line_info", "status", "duration_s",
)
TIMINGS_COLUMNS: Tuple[str, ...] = (
    "contract_id", "dataset", "tool", "tool_version",
    "duration_s", "status", "exit_code",
)

_CODE_RE = re.compile(r"(SWC-\d+|DASP-\d+)", re.IGNORECASE)


# ==============================================================================
# Interop with stage 01 (single source of truth for dataset resolution + schema)
# ==============================================================================
def load_stage01(this_file: Optional[str] = None):
    """Import 01_download_data.py by path (its name is not a valid module id)."""
    here = Path(this_file or __file__).resolve().parent
    path = here / "01_download_data.py"
    if not path.exists():
        raise FileNotFoundError(f"cannot locate stage 01 at {path}")
    spec = importlib.util.spec_from_file_location("reliant_stage01", path)
    mod = importlib.util.module_from_spec(spec)
    # Register BEFORE exec so decorators that resolve the module by name (e.g.
    # dataclasses, functools) work if stage 01 ever gains them; also makes
    # repeated loads share one module object.
    sys.modules["reliant_stage01"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def tool_version(tool_id: str) -> str:
    """Extract a version/commit tag from a SmartBugs tool id ('slither-0.11.3'
    -> '0.11.3', 'semgrep-c3a9f40' -> 'c3a9f40', 'oyente' -> '')."""
    m = re.search(r"-([0-9][0-9A-Za-z.\-]*|[0-9a-f]{6,})$", tool_id)
    return m.group(1) if m else ""


# ==============================================================================
# Ground-truth parsing
# ==============================================================================
def parse_solidifi_loc(buglog_path: Path) -> List[int]:
    """Return injected line numbers from a SolidiFI BugLog CSV.

    Only the integer `loc` (first) column is trusted; the 'bug type' column is
    UTF-7-mangled and is never read. latin-1 guarantees no decode error.
    """
    locs: List[int] = []
    with buglog_path.open("r", encoding="latin-1", newline="") as fh:
        for row in csv.reader(fh):
            if row and row[0].strip().isdigit():
                locs.append(int(row[0].strip()))
    return locs


def load_curated_index(curated_root: Path) -> Dict[str, dict]:
    """Index vulnerabilities.json by dataset-relative posix path."""
    with (curated_root / "vulnerabilities.json").open("r", encoding="utf-8") as fh:
        records = json.load(fh)
    return {str(Path(r["path"]).as_posix()): r for r in records}


def curated_gt_for(record: dict, canonicalize) -> Dict[str, List[int]]:
    """Group a curated contract's annotated lines by canonical class."""
    out: Dict[str, List[int]] = {}
    for v in record.get("vulnerabilities", []):
        canon = canonicalize(str(v.get("category", "")))
        if canon == "unmapped":
            continue
        lines = [int(x) for x in v.get("lines", []) if isinstance(x, (int, float))]
        out.setdefault(canon, [])
        out[canon].extend(lines)
    # De-duplicate while keeping determinism.
    return {k: sorted(set(v)) for k, v in out.items()}


def ground_truth_for_row(row: pd.Series, solidifi_root: Path,
                         curated_root: Path, curated_index: Dict[str, dict],
                         canonicalize) -> Dict[str, List[int]]:
    """Return {canonical_class: [ground-truth lines]} for one registry row."""
    if row["dataset"] == "solidifi":
        # ground_truth_ref is the BugLog path relative to the SolidiFI root.
        buglog = solidifi_root / str(row["ground_truth_ref"])
        locs = sorted(set(parse_solidifi_loc(buglog))) if buglog.exists() else []
        cls = str(row["class_canonical"])
        return {cls: locs} if locs else {cls: []}
    if row["dataset"] == "sb_curated":
        rec = curated_index.get(str(row["relpath"]))
        return curated_gt_for(rec, canonicalize) if rec else {}
    return {}


# ==============================================================================
# Classification map: tool finding name -> set of canonical classes
# ==============================================================================
def parse_classification_codes(text: str) -> Set[str]:
    """Extract normalized SWC-/DASP- codes from a classification string."""
    return {c.upper() for c in _CODE_RE.findall(text or "")}


def codes_to_canon(codes: Set[str]) -> Set[str]:
    """Map a set of SWC/DASP codes to the canonical classes they imply."""
    canon: Set[str] = set()
    for c in codes:
        canon |= SWC_TO_CANON.get(c, set())
        canon |= DASP_TO_CANON.get(c, set())
    return canon


def _classification_of(entry) -> str:
    """Pull the 'classification' text out of a findings.yaml value (which may be
    a dict, a string, or None)."""
    if isinstance(entry, dict):
        return str(entry.get("classification", "") or "")
    if isinstance(entry, str):
        return entry
    return ""


def load_findings_yaml_map(tool_dir: Path) -> Dict[str, Set[str]]:
    """Read tools/<tool>/findings.yaml -> {finding_name_lower: {canonical...}}."""
    fpath = tool_dir / "findings.yaml"
    if not fpath.exists():
        return {}
    # Some tool snapshots ship findings.yaml with stray TAB characters (e.g. a
    # trailing tab after a classification value in semgrep-c3a9f40). YAML forbids
    # tabs, so normalize them to spaces and strip trailing whitespace before
    # parsing; fall back to an empty map (surfaced via diagnostics) if still bad.
    raw = fpath.read_text(encoding="utf-8")
    sanitized = "\n".join(line.replace("\t", " ").rstrip() for line in raw.splitlines())
    try:
        data = yaml.safe_load(sanitized) or {}
    except yaml.YAMLError:
        return {}
    mapping: Dict[str, Set[str]] = {}
    if isinstance(data, dict):
        for name, entry in data.items():
            canon = codes_to_canon(parse_classification_codes(_classification_of(entry)))
            if canon:
                mapping[str(name).strip().lower()] = canon
    return mapping


def build_finding_class_map(tools_dir: Path, panel: Tuple[str, ...]
                            ) -> Tuple[Dict[str, Dict[str, Set[str]]], dict]:
    """Build {tool: {finding_name_lower: {canonical...}}} for the panel.

    Combines each tool's findings.yaml with any MANUAL_CLASSIFICATION override.
    Also returns a small summary for the metadata (mapped-name counts per class).
    """
    result: Dict[str, Dict[str, Set[str]]] = {}
    summary: Dict[str, dict] = {}
    for tool in panel:
        tmap = load_findings_yaml_map(tools_dir / tool)
        for name, cls in MANUAL_CLASSIFICATION.get(tool, {}).items():
            tmap[name.strip().lower()] = codes_to_canon(parse_classification_codes(cls))
        result[tool] = tmap
        per_class: Dict[str, int] = {}
        for canon_set in tmap.values():
            for c in canon_set:
                per_class[c] = per_class.get(c, 0) + 1
        summary[tool] = {"n_finding_names": len(tmap),
                         "names_per_class": dict(sorted(per_class.items()))}
    return result, summary


# ==============================================================================
# SmartBugs results loader
# ==============================================================================
def _coerce_line(value) -> Optional[int]:
    """Coerce a finding's line field to a positive int, or None."""
    if value is None:
        return None
    if isinstance(value, bool):  # guard: bool is a subclass of int
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float):
        return int(value) if value > 0 else None
    if isinstance(value, str) and value.strip().isdigit():
        n = int(value.strip())
        return n if n > 0 else None
    return None


def load_task(task_dir: Path,
              task_log_name: str = "smartbugs.json",
              parser_output_name: str = "result.json") -> Optional[dict]:
    """Load one SmartBugs task directory into a normalized dict, or None.

    Reads the task log (filename, tool id/mode, duration, exit code) and the
    parser output (findings). Mirrors the fields SmartBugs' own results2csv uses.
    """
    tlog_p = task_dir / task_log_name
    pout_p = task_dir / parser_output_name
    if not tlog_p.exists() or not pout_p.exists():
        return None
    try:
        tlog = json.loads(tlog_p.read_text(encoding="utf-8"))
        pout = json.loads(pout_p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    tool = tlog.get("tool", {}) or {}
    result = tlog.get("result", {}) or {}
    exit_code = result.get("exit_code", tlog.get("exit_code"))
    errors = pout.get("errors") or []
    fails = pout.get("fails") or []
    findings_raw = pout.get("findings", []) or []

    # Status is derived from the PARSER's own outcome (fails / errors), not from
    # the raw process exit code. Several analyzers signal "findings present" via
    # a non-zero exit: slither-0.11.3 exits 255 whenever it reports anything, so
    # keying on exit_code would mark every successful analysis as an error, drop
    # the tool from success-only cost medians, and silently substitute a fallback
    # cost. A run that produced parseable output with no recorded failure is a
    # success even when the exit code is non-zero; a run that produced neither
    # output nor findings AND exited non-zero is an error.
    if fails or errors:
        status = "error"
    elif exit_code in (0, None) or findings_raw:
        status = "success"
    else:
        status = "error"

    findings = []
    for f in findings_raw:
        findings.append({"name": str(f.get("name", "")),
                         "line": _coerce_line(f.get("line"))})
    return {
        "tool_id": tool.get("id", ""),
        "tool_mode": tool.get("mode", ""),
        "filename": tlog.get("filename", ""),
        "duration_s": result.get("duration", tlog.get("duration")),
        "exit_code": exit_code,
        "status": status,
        "n_errors": len(errors),
        "findings": findings,
    }


def iter_results(results_dir: Path) -> Iterator[Tuple[Path, dict]]:
    """Yield (task_dir, task) for every SmartBugs result.json under results_dir."""
    for pout in sorted(results_dir.rglob("result.json")):
        task = load_task(pout.parent)
        if task is not None:
            yield pout.parent, task


def _walk_path_suffixes(fn: str) -> Iterator[str]:
    """Yield '/'-component suffixes of a normalized path, longest first.

    '/a/b/c.sol' -> 'a/b/c.sol', 'b/c.sol', 'c.sol'. Component-aligned matching
    (rather than raw str.endswith) prevents false hits such as 'x/ab.sol'
    matching '.../zx/ab.sol'.
    """
    parts = [p for p in fn.split("/") if p]
    for i in range(len(parts)):
        yield "/".join(parts[i:])


def _build_suffix_index(registry: pd.DataFrame) -> Dict[str, str]:
    """Map both 'relpath' and 'dataset/relpath' -> contract_id.

    The bare relpath keys are what make ABSOLUTE task filenames resolvable (the
    recorded path never contains the literal dataset name, only the extraction
    directory); the dataset-prefixed keys keep results generated with
    dataset-relative filenames resolvable too. A key claimed by two different
    contracts is dropped (resolution then falls through to the next rung), and
    uniqueness of the surviving keys is what makes a suffix hit unambiguous.
    """
    index: Dict[str, str] = {}
    ambiguous: set = set()
    for r in registry.itertuples(index=False):
        for key in (str(r.relpath), f"{r.dataset}/{r.relpath}"):
            prev = index.get(key)
            if prev is not None and prev != r.contract_id:
                ambiguous.add(key)
            else:
                index[key] = r.contract_id
    for key in ambiguous:
        index.pop(key, None)
    return index


def map_task_to_contract(filename: str, registry: pd.DataFrame,
                         suffix_index: Dict[str, str],
                         run_manifest: Optional[Dict[str, str]] = None,
                         task_key: Optional[str] = None) -> Optional[str]:
    """Resolve the contract_id a SmartBugs task analyzed.

    Resolution ladder (first hit wins):
      1. run_manifest[task_key] / run_manifest[filename]  -- direct driver record
         (the driver stores absolute posix paths AND dataset-relative paths);
      2. longest component-suffix of `filename` found in run_manifest -- absorbs
         SmartBugs path normalization (relative vs absolute, leading './');
      3. longest component-suffix found in the registry suffix index
         (bare relpath or 'dataset/relpath');
      4. unique basename across the registry.
    Returns None when nothing matches unambiguously (counted as unmapped).
    """
    fn = str(filename).replace("\\", "/")
    if run_manifest:
        if task_key and task_key in run_manifest:
            return run_manifest[task_key]
        if fn in run_manifest:
            return run_manifest[fn]
        for suffix in _walk_path_suffixes(fn):
            if suffix in run_manifest:
                return run_manifest[suffix]
    for suffix in _walk_path_suffixes(fn):
        if suffix in suffix_index:
            return suffix_index[suffix]
    base = os.path.basename(fn)
    by_base = [r.contract_id for r in registry.itertuples(index=False)
               if os.path.basename(str(r.relpath)) == base]
    return by_base[0] if len(by_base) == 1 else None


# ==============================================================================
# Metric core
# ==============================================================================
def greedy_line_match(gt_lines: List[int], det_lines: List[int], tol: int) -> int:
    """One-to-one greedy matching count between ground-truth and detected lines.

    Each ground-truth line matches at most one detected line (nearest within
    `tol`) and vice versa. For tol == 0 this is exact-line matching.
    """
    gt = sorted(gt_lines)
    det = sorted(det_lines)
    used = [False] * len(det)
    tp = 0
    for g in gt:
        best_j, best_d = -1, None
        for j, d in enumerate(det):
            if used[j]:
                continue
            dist = abs(d - g)
            if dist <= tol and (best_d is None or dist < best_d):
                best_d, best_j = dist, j
        if best_j >= 0:
            used[best_j] = True
            tp += 1
    return tp


def relevant_by_class(findings: List[dict],
                      tool_map: Dict[str, Set[str]]) -> Dict[str, dict]:
    """Bucket a tool's findings by canonical class relevance.

    Returns {canonical_class: {'lines': [...], 'count': n}}, where a finding
    relevant to two classes (e.g. SWC-104 -> unchecked + unhandled) is counted
    under both.
    """
    out: Dict[str, dict] = {}
    for f in findings:
        classes = tool_map.get(str(f["name"]).strip().lower(), set())
        for canon in classes:
            b = out.setdefault(canon, {"lines": [], "count": 0})
            b["count"] += 1
            if f["line"] is not None:
                b["lines"].append(int(f["line"]))
    return out


def compute_label_rows(row: pd.Series, tool: str, task: Optional[dict],
                       gt_by_class: Dict[str, List[int]],
                       tool_map: Dict[str, Set[str]], tol: int) -> List[dict]:
    """Compute one label row per ground-truth class of a (contract, tool)."""
    version = tool_version(tool)
    rows: List[dict] = []

    if task is None:
        status, duration = "missing", None
        rel: Dict[str, dict] = {}
    else:
        status, duration = task["status"], task["duration_s"]
        rel = relevant_by_class(task["findings"], tool_map)

    for cls, gt_lines in gt_by_class.items():
        m = len(gt_lines)
        bucket = rel.get(cls, {"lines": [], "count": 0})
        n_rel = int(bucket["count"])
        det_lines = list(bucket["lines"])
        n_det_lines = len(det_lines)

        if status == "missing":
            # Tool result absent: outcome unknown, not a measured miss.
            detected = pd.NA
            tp = fp_proxy = fn = pd.NA
            recall = precision = fnr = fpr_proxy = pd.NA
            detected_at_line = pd.NA
            line_info = "none"
        else:
            detected = n_rel >= 1
            if n_rel > 0 and n_det_lines == 0:
                # Relevant but line-less (e.g. bytecode tool): line metrics N/A.
                tp = fp_proxy = fn = pd.NA
                recall = precision = fnr = fpr_proxy = pd.NA
                detected_at_line = pd.NA
                line_info = "coarse"
            else:
                tp = greedy_line_match(gt_lines, det_lines, tol)
                fn = m - tp
                fp_proxy = n_det_lines - tp
                recall = (tp / m) if m > 0 else pd.NA
                precision = (tp / n_det_lines) if n_det_lines > 0 else pd.NA
                fnr = (1.0 - recall) if m > 0 else pd.NA
                fpr_proxy = (1.0 - precision) if n_det_lines > 0 else pd.NA
                detected_at_line = tp >= 1
                line_info = "line" if n_det_lines > 0 else "none"

        rows.append({
            "contract_id": row["contract_id"],
            "dataset": row["dataset"],
            "base_id": row["base_id"],
            "class_canonical": cls,
            "tool": tool,
            "tool_version": version,
            "n_injected": m,
            "n_relevant_findings": n_rel,
            "n_detected_lines": n_det_lines,
            "tp": tp, "fp_proxy": fp_proxy, "fn": fn,
            "recall": recall, "precision": precision,
            "fnr": fnr, "fpr_proxy": fpr_proxy,
            "detected": detected, "detected_at_line": detected_at_line,
            "line_info": line_info, "status": status, "duration_s": duration,
        })
    return rows


def _coerce_label_dtypes(labels: pd.DataFrame) -> pd.DataFrame:
    """Cast label columns to nullable dtypes so NaN/NA survive a Parquet round-trip."""
    int_cols = ["n_injected", "n_relevant_findings", "n_detected_lines",
                "tp", "fp_proxy", "fn"]
    float_cols = ["recall", "precision", "fnr", "fpr_proxy", "duration_s"]
    bool_cols = ["detected", "detected_at_line"]
    for c in int_cols:
        labels[c] = labels[c].astype("Int64")
    for c in float_cols:
        labels[c] = labels[c].astype("Float64")
    for c in bool_cols:
        labels[c] = labels[c].astype("boolean")
    return labels


def build_labels(registry: pd.DataFrame,
                 results_by_key: Dict[Tuple[str, str], dict],
                 gt_by_contract: Dict[str, Dict[str, List[int]]],
                 tool_class_map: Dict[str, Dict[str, Set[str]]],
                 panel: Tuple[str, ...], tol: int
                 ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Assemble the long labels table and the per-(contract, tool) timings table.

    `results_by_key` maps (contract_id, tool) -> normalized task dict.
    `gt_by_contract` maps contract_id -> {class: [lines]}.
    """
    label_rows: List[dict] = []
    timing_rows: List[dict] = []
    # Indexed lookup (drop=False keeps contract_id available as a column).
    reg_by_id = registry.set_index("contract_id", drop=False)

    for cid, gt_by_class in gt_by_contract.items():
        row = reg_by_id.loc[cid]
        if not gt_by_class:  # e.g. curated contract with only 'unmapped' classes
            continue
        for tool in panel:
            task = results_by_key.get((cid, tool))
            label_rows.extend(
                compute_label_rows(row, tool, task, gt_by_class, tool_class_map[tool], tol))
            timing_rows.append({
                "contract_id": cid,
                "dataset": row["dataset"],
                "tool": tool,
                "tool_version": tool_version(tool),
                "duration_s": task["duration_s"] if task else None,
                "status": task["status"] if task else "missing",
                "exit_code": task["exit_code"] if task else None,
            })

    labels = pd.DataFrame(label_rows, columns=list(LABELS_COLUMNS))
    labels = labels.sort_values(
        ["contract_id", "class_canonical", "tool"], kind="mergesort"
    ).reset_index(drop=True)
    labels = _coerce_label_dtypes(labels)
    timings = pd.DataFrame(timing_rows, columns=list(TIMINGS_COLUMNS))
    timings = timings.sort_values(
        ["contract_id", "tool"], kind="mergesort").reset_index(drop=True)
    return labels, timings


def pivot_reliability(labels: pd.DataFrame, value: str = "detected") -> pd.DataFrame:
    """Reliability matrix: (contract_id, class_canonical) x tool -> value."""
    wide = labels.pivot_table(
        index=["contract_id", "dataset", "base_id", "class_canonical"],
        columns="tool", values=value, aggfunc="first")
    wide = wide.reset_index()
    wide.columns = [c if isinstance(c, str) else str(c) for c in wide.columns]
    return wide


# ==============================================================================
# Output writer
# ==============================================================================
def write_outputs(labels: pd.DataFrame, timings: pd.DataFrame,
                  out_dir: Path, meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(out_dir / "labels.parquet", engine="pyarrow", index=False)
    wide = pivot_reliability(labels, "detected")
    wide.to_parquet(out_dir / "labels_wide.parquet", engine="pyarrow", index=False)
    timings.to_parquet(out_dir / "tool_timings.parquet", engine="pyarrow", index=False)
    (out_dir / "labels_meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=False), encoding="utf-8")


def summarize_labels(labels: pd.DataFrame, timings: pd.DataFrame,
                     panel: Tuple[str, ...], tol: int,
                     class_map_summary: dict) -> dict:
    """Build the metadata / coverage report for labels_meta.json."""
    scored = labels[labels["status"] != "missing"]
    det = scored.dropna(subset=["detected"])
    per_tool = {}
    for tool in panel:
        sub = det[det.tool == tool]
        per_tool[tool] = {
            "version": tool_version(tool),
            "n_instances": int(len(sub)),
            "detection_rate": (round(float(sub["detected"].mean()), 4)
                               if len(sub) else None),
        }
    instances = labels[["contract_id", "class_canonical"]].drop_duplicates()
    by_class = instances["class_canonical"].value_counts().sort_index().to_dict()
    return {
        "schema_version": SCHEMA_VERSION,
        "generator": os.path.basename(__file__),
        "generator_version": __version__,
        "line_tolerance": tol,
        "panel": list(panel),
        "canonical_classes": list(CANONICAL_CLASSES),
        "swc_to_canonical": {k: sorted(v) for k, v in SWC_TO_CANON.items()},
        "dasp_to_canonical": {k: sorted(v) for k, v in DASP_TO_CANON.items()},
        "counts": {
            "n_label_rows": int(len(labels)),
            "n_instances": int(len(instances)),
            "instances_by_class": {k: int(v) for k, v in by_class.items()},
            "n_missing_results": int((labels["status"] == "missing").sum()),
        },
        "detection_rate_by_tool": per_tool,
        "finding_class_map_summary": class_map_summary,
        "notes": [
            "detected is the tool-agnostic primary target; recall/precision are "
            "line-level refinements (NaN when a tool is relevant but line-less).",
            "FP_proxy / FPR_proxy are proxies: true negatives are undefined at "
            "line granularity and annotation sets may be incomplete.",
            "SWC-104/DASP-4 map to BOTH unchecked_low_level_calls and "
            "unhandled_exceptions (tools do not distinguish them).",
            "Scored only on (contract, class) cells with ground truth; each "
            "SolidiFI contract has exactly one such class.",
        ],
    }


# ==============================================================================
# Server driver (SmartBugs 2.0 via Docker) -- not run in a CPU sandbox
# ==============================================================================
def build_smartbugs_cmd(panel: Tuple[str, ...], files: List[str],
                        results_dir: str, timeout: int, processes: int,
                        smartbugs_bin: str = "smartbugs") -> List[str]:
    """Construct the SmartBugs 2.0 CLI invocation (unit-tested, not executed here)."""
    cmd = [smartbugs_bin, "-t", *panel, "-f", *files,
           "--results", results_dir,
           "--timeout", str(timeout),
           "--processes", str(processes),
           "--json"]
    return cmd


def build_run_manifest(registry: pd.DataFrame, solidifi_root: Path,
                       curated_root: Path) -> Tuple[List[str], Dict[str, str]]:
    """Return (files_to_analyze, run_manifest) for the analyzer run.

    The manifest maps, for every contract, its ABSOLUTE posix path (exactly what
    is handed to SmartBugs, hence what its task logs record) plus its bare
    relpath and 'dataset/relpath' -> contract_id. Storing all three makes label
    attribution robust to how the framework normalizes paths. Extracted from
    run_tools so the self-test can round-trip the exact writer format through
    collect_results_by_key without Docker.
    """
    files: List[str] = []
    manifest: Dict[str, str] = {}

    def _claim(key: str, cid: str) -> None:
        prev = manifest.get(key)
        if prev is not None and prev != cid:
            raise AssertionError(
                f"run_manifest key {key!r} claimed by both {prev} and {cid}")
        manifest[key] = cid

    for r in registry.itertuples(index=False):
        root = solidifi_root if r.dataset == "solidifi" else curated_root
        fpath = (root / str(r.relpath)).resolve()
        files.append(str(fpath))
        cid = str(r.contract_id)
        _claim(fpath.as_posix(), cid)
        _claim(str(r.relpath), cid)
        _claim(f"{r.dataset}/{r.relpath}", cid)
    return files, manifest


def run_tools(registry: pd.DataFrame, solidifi_root: Path, curated_root: Path,
              panel: Tuple[str, ...], results_dir: Path, smartbugs_bin: str,
              timeout: int = 600, processes: int = 1) -> int:
    """Run the analyzer panel over the corpus via SmartBugs (server-only).

    Writes run_manifest.json (see build_run_manifest) so labels can be attributed
    unambiguously. Requires Docker + tool images; this function intentionally
    shells out and is not exercised by --selftest (its two halves --
    build_run_manifest and build_smartbugs_cmd -- are).
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    files, manifest = build_run_manifest(registry, solidifi_root, curated_root)
    (results_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")

    cmd = build_smartbugs_cmd(panel, files, str(results_dir), timeout,
                              processes, smartbugs_bin)
    print(f"[run-tools] {len(files)} contracts x {len(panel)} tools via SmartBugs")
    print("  " + " ".join(cmd[:8]) + " ... (%d files)" % len(files))
    try:
        proc = subprocess.run(cmd, check=False)
    except FileNotFoundError:
        sys.stderr.write(
            f"error: {smartbugs_bin!r} not found. Run this on the server with "
            "SmartBugs 2.0 + Docker installed, or pass --smartbugs-bin.\n")
        return 3
    return proc.returncode


# ==============================================================================
# Build-labels orchestration
# ==============================================================================
def collect_results_by_key(results_dir: Path, registry: pd.DataFrame,
                           panel: Tuple[str, ...]
                           ) -> Tuple[Dict[Tuple[str, str], dict], dict]:
    """Load all SmartBugs tasks and key them by (contract_id, tool)."""
    suffix_index = _build_suffix_index(registry)
    rm_path = results_dir / "run_manifest.json"
    run_manifest = (json.loads(rm_path.read_text(encoding="utf-8"))
                    if rm_path.exists() else None)
    panel_set = set(panel)
    by_key: Dict[Tuple[str, str], dict] = {}
    n_tasks = n_unmapped = n_offpanel = 0
    for task_dir, task in iter_results(results_dir):
        n_tasks += 1
        tool = task["tool_id"]
        if tool not in panel_set:
            n_offpanel += 1
            continue
        task_key = f"{tool}:{task['filename']}"
        cid = map_task_to_contract(task["filename"], registry, suffix_index,
                                   run_manifest, task_key)
        if cid is None:
            n_unmapped += 1
            continue
        by_key[(cid, tool)] = task
    diag = {"n_tasks": n_tasks, "n_unmapped": n_unmapped, "n_offpanel": n_offpanel,
            "n_keyed": len(by_key)}
    return by_key, diag


def unmapped_finding_names(results_by_key: Dict[Tuple[str, str], dict],
                           tool_class_map: Dict[str, Dict[str, Set[str]]],
                           limit: int = 25) -> Dict[str, List[str]]:
    """Per tool, finding names present in results but absent from the class map.

    Surfacing these on the server makes any parser/findings.yaml drift visible so
    a MANUAL_CLASSIFICATION entry can be added; unmapped names are otherwise
    silently treated as irrelevant to every canonical class.
    """
    seen: Dict[str, Set[str]] = {}
    for (_, tool), task in results_by_key.items():
        for f in task["findings"]:
            seen.setdefault(tool, set()).add(str(f["name"]).strip().lower())
    out: Dict[str, List[str]] = {}
    for tool, names in seen.items():
        missing = sorted(names - set(tool_class_map.get(tool, {}).keys()))
        if missing:
            out[tool] = missing[:limit]
    return out


def do_build_labels(args) -> int:
    stage01 = load_stage01()
    canonicalize = stage01.canonicalize

    reg_path = Path(args.registry)
    if not reg_path.exists():
        sys.stderr.write(f"error: registry not found: {reg_path}\n")
        return 3
    registry = pd.read_parquet(reg_path, engine="pyarrow")

    # Resolve dataset roots (needed to read ground truth) exactly as stage 01 does.
    work = Path(args.out) / "_work"
    solidifi = args.solidifi or (Path(args.uploads) / "SolidiFI-benchmark-master.zip"
                                 if args.uploads else None)
    curated = args.curated or (Path(args.uploads) / "smartbugs-curated-main.zip"
                               if args.uploads else None)
    if solidifi is None or curated is None:
        sys.stderr.write("error: provide --uploads or --solidifi/--curated.\n")
        return 3
    solidifi_root = stage01.resolve_dataset_dir(Path(solidifi), "buggy_contracts", work, "solidifi")
    curated_root = stage01.resolve_dataset_dir(Path(curated), "vulnerabilities.json", work, "sb_curated")

    # Ground truth per contract.
    curated_index = load_curated_index(curated_root)
    gt_by_contract: Dict[str, Dict[str, List[int]]] = {}
    for _, row in registry.iterrows():
        gt = ground_truth_for_row(row, solidifi_root, curated_root, curated_index, canonicalize)
        gt = {k: v for k, v in gt.items() if v}  # keep classes with >=1 gt line
        if gt:
            gt_by_contract[row["contract_id"]] = gt

    # Classification map from the SmartBugs tools directory.
    tools_dir = Path(args.tools_dir) if args.tools_dir else None
    if tools_dir is None or not tools_dir.exists():
        sys.stderr.write(
            "error: --tools-dir (SmartBugs 'tools/' directory) is required to map "
            "findings to canonical classes.\n")
        return 3
    panel = tuple(args.panel) if args.panel else DEFAULT_PANEL
    tool_class_map, class_map_summary = build_finding_class_map(tools_dir, panel)

    # Load analyzer results.
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        sys.stderr.write(f"error: results dir not found: {results_dir}\n")
        return 3
    results_by_key, diag = collect_results_by_key(results_dir, registry, panel)
    print(f"[build-labels] loaded {diag['n_keyed']} (contract,tool) results "
          f"({diag['n_unmapped']} unmapped, {diag['n_offpanel']} off-panel).")

    labels, timings = build_labels(
        registry, results_by_key, gt_by_contract, tool_class_map, panel, args.tolerance)
    meta = summarize_labels(labels, timings, panel, args.tolerance, class_map_summary)
    diag["unmapped_finding_names"] = unmapped_finding_names(results_by_key, tool_class_map)
    meta["results_diagnostics"] = diag
    write_outputs(labels, timings, Path(args.out), meta)

    c = meta["counts"]
    print(f"\nWrote {args.out}/labels.parquet  ({c['n_label_rows']} rows, "
          f"{c['n_instances']} instances)")
    print("  instances by class:", c["instances_by_class"])
    print("  detection rate by tool:")
    for tool, d in meta["detection_rate_by_tool"].items():
        print(f"      {tool:<20} n={d['n_instances']:<4} rate={d['detection_rate']}")
    return 0


# ==============================================================================
# Hermetic self-test
# ==============================================================================
def run_selftest() -> int:
    """Exercise the label core on fully synthetic inputs and assert every metric."""
    print(f"RELIANT 02_ground_truth self-test (v{__version__})")

    # --- classification plumbing -----------------------------------------------
    assert codes_to_canon(parse_classification_codes("SWC-101, DASP-3")) == {"arithmetic"}
    assert codes_to_canon({"SWC-104"}) == {"unchecked_low_level_calls", "unhandled_exceptions"}
    assert codes_to_canon({"SWC-115"}) == {"tx_origin"}
    assert codes_to_canon({"DASP-2"}) == set()  # access-control not mapped to tx_origin
    assert tool_version("slither-0.11.3") == "0.11.3"
    assert tool_version("semgrep-c3a9f40") == "c3a9f40"
    assert tool_version("oyente") == ""

    # --- findings.yaml reader (note: '#NN' after a space is a YAML comment) ----
    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp) / "toolX"
        td.mkdir()
        (td / "findings.yaml").write_text(
            "Integer Overflow:\n    classification: SWC-101, DASP-3\n"
            "Reentrancy:\n    classification: SWC-107, DASP-1\n"
            # Mirrors securify2's real layout: '#17' is a comment, so the key is
            # 'Possibly unsafe usage of tx-origin' (which is what the parser emits).
            "Possibly unsafe usage of tx-origin: #17\n    classification: SWC-115\n",
            encoding="utf-8")
        m = load_findings_yaml_map(td)
        assert m["integer overflow"] == {"arithmetic"}
        assert m["reentrancy"] == {"reentrancy"}
        assert m["possibly unsafe usage of tx-origin"] == {"tx_origin"}

    # --- line matching ----------------------------------------------------------
    assert greedy_line_match([10, 20, 30], [10, 20], 0) == 2
    assert greedy_line_match([10, 20], [11, 19], 0) == 0
    assert greedy_line_match([10, 20], [11, 19], 1) == 2      # tolerance
    assert greedy_line_match([10, 10], [10], 0) == 1          # one-to-one

    # --- metric core on a synthetic contract -----------------------------------
    tool_map = {"reentrancy": {"reentrancy"}, "integer overflow": {"arithmetic"}}
    row = pd.Series({"contract_id": "c1", "dataset": "solidifi",
                     "base_id": "b1", "class_canonical": "reentrancy"})
    gt = {"reentrancy": [22, 36, 51]}  # m = 3

    # (a) tool detects 2 of 3 at correct lines, plus one spurious class-K alert.
    task_a = {"status": "success", "duration_s": 3.2, "exit_code": 0,
              "findings": [{"name": "Reentrancy", "line": 22},
                           {"name": "Reentrancy", "line": 36},
                           {"name": "Reentrancy", "line": 99}]}
    ra = compute_label_rows(row, "toolA", task_a, gt, tool_map, 0)[0]
    assert ra["n_injected"] == 3 and ra["tp"] == 2 and ra["fn"] == 1
    assert ra["fp_proxy"] == 1 and abs(ra["recall"] - 2/3) < 1e-9
    assert abs(ra["precision"] - 2/3) < 1e-9 and ra["detected"] is True
    assert ra["detected_at_line"] is True and ra["line_info"] == "line"

    # (b) tool reports nothing relevant -> genuine miss.
    task_b = {"status": "success", "duration_s": 1.0, "exit_code": 0,
              "findings": [{"name": "Integer Overflow", "line": 5}]}
    rb = compute_label_rows(row, "toolB", task_b, gt, tool_map, 0)[0]
    assert rb["detected"] is False and rb["tp"] == 0 and rb["fn"] == 3
    assert abs(rb["recall"] - 0.0) < 1e-9 and abs(rb["fnr"] - 1.0) < 1e-9

    # (c) relevant but line-less (bytecode-style) -> coarse detection, NaN lines.
    task_c = {"status": "success", "duration_s": 8.0, "exit_code": 0,
              "findings": [{"name": "Reentrancy", "line": None}]}
    rc = compute_label_rows(row, "toolC", task_c, gt, tool_map, 0)[0]
    assert rc["detected"] is True and rc["line_info"] == "coarse"
    assert pd.isna(rc["recall"]) and pd.isna(rc["tp"])

    # (d) missing result -> outcome unknown (NA), not a measured miss.
    rd = compute_label_rows(row, "toolD", None, gt, tool_map, 0)[0]
    assert rd["status"] == "missing" and pd.isna(rd["detected"])

    # (e) SWC-104 finding scores for BOTH unchecked and unhandled classes.
    tool_map2 = {"unchecked call": {"unchecked_low_level_calls", "unhandled_exceptions"}}
    row_uh = pd.Series({"contract_id": "c2", "dataset": "solidifi",
                        "base_id": "b2", "class_canonical": "unhandled_exceptions"})
    task_e = {"status": "success", "duration_s": 2.0, "exit_code": 0,
              "findings": [{"name": "Unchecked Call", "line": 12}]}
    re_ = compute_label_rows(row_uh, "toolE", task_e, {"unhandled_exceptions": [12]},
                             tool_map2, 0)[0]
    assert re_["detected"] is True and re_["tp"] == 1

    # --- end-to-end build + pivot on a tiny registry ---------------------------
    registry = pd.DataFrame([
        {"contract_id": "c1", "dataset": "solidifi", "base_id": "b1",
         "class_canonical": "reentrancy", "relpath": "x/c1.sol"},
        {"contract_id": "c2", "dataset": "sb_curated", "base_id": "c2",
         "class_canonical": "arithmetic", "relpath": "dataset/arithmetic/c2.sol"},
    ])
    gt_all = {"c1": {"reentrancy": [22, 36]}, "c2": {"arithmetic": [7]}}
    tmap_all = {"tk": {"reentrancy": {"reentrancy"}, "integer overflow": {"arithmetic"}}}
    results = {
        ("c1", "tk"): {"status": "success", "duration_s": 2.0, "exit_code": 0,
                       "findings": [{"name": "Reentrancy", "line": 22}]},
        ("c2", "tk"): {"status": "success", "duration_s": 1.5, "exit_code": 0,
                       "findings": [{"name": "Integer Overflow", "line": 7}]},
    }
    labels, timings = build_labels(registry, results, gt_all, tmap_all, ("tk",), 0)
    assert len(labels) == 2 and list(labels.columns) == list(LABELS_COLUMNS)
    assert len(timings) == 2
    wide = pivot_reliability(labels, "detected")
    assert "tk" in wide.columns and len(wide) == 2
    assert bool(wide.loc[wide.contract_id == "c1", "tk"].iloc[0]) is True

    # Determinism of the long table.
    labels2, _ = build_labels(registry, results, gt_all, tmap_all, ("tk",), 0)
    assert labels.equals(labels2)

    # --- server-driver command construction (Docker not exercised here) --------
    cmd = build_smartbugs_cmd(("slither-0.11.3", "oyente"),
                              ["/a/x.sol", "/b/y.sol"], "out/run", 600, 2,
                              smartbugs_bin="/opt/sb/smartbugs")
    assert cmd[0] == "/opt/sb/smartbugs"
    assert "-t" in cmd and "slither-0.11.3" in cmd and "oyente" in cmd
    assert "-f" in cmd and "/a/x.sol" in cmd
    assert cmd[cmd.index("--results") + 1] == "out/run"
    assert cmd[cmd.index("--timeout") + 1] == "600"
    assert cmd[cmd.index("--processes") + 1] == "2"
    assert build_smartbugs_cmd(("oyente",), ["f.sol"], "r", 1, 1)[0] == "smartbugs"

    # --- task -> contract resolution: suffix walk unit behavior ----------------
    assert list(_walk_path_suffixes("/a/b/c.sol")) == ["a/b/c.sol", "b/c.sol", "c.sol"]
    reg_amb = pd.DataFrame([
        # SolidiFI-style: same basename in two class dirs (ambiguous basename).
        {"contract_id": "s1", "dataset": "solidifi",
         "relpath": "buggy_contracts/Re-entrancy/buggy_1.sol"},
        {"contract_id": "s2", "dataset": "solidifi",
         "relpath": "buggy_contracts/Overflow-Underflow/buggy_1.sol"},
        {"contract_id": "k1", "dataset": "sb_curated",
         "relpath": "dataset/reentrancy/dao.sol"},
    ])
    idx = _build_suffix_index(reg_amb)
    assert idx["buggy_contracts/Re-entrancy/buggy_1.sol"] == "s1"
    assert idx["solidifi/buggy_contracts/Overflow-Underflow/buggy_1.sol"] == "s2"
    # Component alignment: 'x/ab.sol' must not match '.../zx/ab.sol'.
    idx_align = _build_suffix_index(pd.DataFrame(
        [{"contract_id": "cA", "dataset": "d", "relpath": "x/ab.sol"}]))
    assert map_task_to_contract("/root/zx/ab.sol", pd.DataFrame(
        [{"contract_id": "cA", "dataset": "d", "relpath": "x/ab.sol"}]),
        idx_align) == "cA"  # resolved by unique basename, NOT a false suffix hit
    # An ABSOLUTE SolidiFI path resolves via relpath suffix despite the wrapper
    # dir and the 7-way-ambiguous basename; no manifest needed.
    fn_abs = "/srv/work/solidifi/SolidiFI-benchmark-master/buggy_contracts/Re-entrancy/buggy_1.sol"
    assert map_task_to_contract(fn_abs, reg_amb, idx) == "s1"
    # Ambiguous basename with no suffix/manifest info -> None (never guesses).
    assert map_task_to_contract("/elsewhere/buggy_1.sol", reg_amb, idx) is None

    # --- WRITER -> READER round-trip (the exact server format, no Docker) ------
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        sroot = root / "extract" / "SolidiFI-benchmark-master"
        croot = root / "extract" / "smartbugs-curated-main"
        for _, r in reg_amb.iterrows():
            base = sroot if r["dataset"] == "solidifi" else croot
            p = base / r["relpath"]
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("pragma solidity ^0.5.0;\ncontract C {}\n", encoding="utf-8")
        files, manifest = build_run_manifest(reg_amb, sroot, croot)
        assert len(files) == 3 and all(Path(f).is_absolute() for f in files)
        # Every absolute path, bare relpath, and dataset/relpath is claimable.
        assert manifest[str(Path(files[0]).as_posix())] == "s1"
        assert manifest["dataset/reentrancy/dao.sol"] == "k1"
        rdir = root / "runs"
        for i, (f, cid, tool) in enumerate(
                zip(files, ("s1", "s2", "k1"), ("oyente", "oyente", "conkas"))):
            td = rdir / tool / f"task{i}"
            td.mkdir(parents=True)
            (td / "smartbugs.json").write_text(json.dumps(
                {"filename": f, "tool": {"id": tool, "mode": "solc"},
                 "result": {"duration": 1.0, "exit_code": 0}}), encoding="utf-8")
            (td / "result.json").write_text(json.dumps(
                {"findings": [], "errors": [], "fails": []}), encoding="utf-8")
        (rdir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        by_key, diag = collect_results_by_key(rdir, reg_amb, ("oyente", "conkas"))
        assert diag["n_unmapped"] == 0, f"server-format tasks unmapped: {diag}"
        assert set(by_key) == {("s1", "oyente"), ("s2", "oyente"), ("k1", "conkas")}
        # Reader still works with NO manifest (suffix rung alone).
        (rdir / "run_manifest.json").unlink()
        by_key2, diag2 = collect_results_by_key(rdir, reg_amb, ("oyente", "conkas"))
        assert diag2["n_unmapped"] == 0 and set(by_key2) == set(by_key)

    # --- REGRESSION: exit-255-on-findings must NOT be treated as failure -----
    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp) / "slither-0.11.3" / "t0"
        td.mkdir(parents=True)
        (td / "smartbugs.json").write_text(json.dumps(
            {"filename": "d/x.sol", "tool": {"id": "slither-0.11.3", "mode": "solc"},
             "result": {"duration": 0.82, "exit_code": 255}}), encoding="utf-8")
        (td / "result.json").write_text(json.dumps(
            {"findings": [{"name": "reentrancy-eth", "line": 12}],
             "errors": [], "fails": []}), encoding="utf-8")
        rec = load_task(td)
        assert rec["status"] == "success", (
            "exit 255 WITH findings and no fails must be a success -- otherwise "
            "slither is excluded from success-only cost medians")
        assert rec["duration_s"] == 0.82
        # a genuine failure is still an error
        (td / "result.json").write_text(json.dumps(
            {"findings": [], "errors": ["boom"], "fails": ["crashed"]}), encoding="utf-8")
        assert load_task(td)["status"] == "error"
        # non-zero exit, no findings, no fails -> error
        (td / "result.json").write_text(json.dumps(
            {"findings": [], "errors": [], "fails": []}), encoding="utf-8")
        assert load_task(td)["status"] == "error"
    print("  exit-255-on-findings handled as success (slither cost preserved).")

    print("  all invariants passed.")
    return 0


# ==============================================================================
# CLI
# ==============================================================================
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Analyzer-reliability label core (stage 02).")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--run-tools", action="store_true",
                      help="Server: run the analyzer panel via SmartBugs 2.0.")
    mode.add_argument("--build-labels", action="store_true",
                      help="Build reliability labels from a results directory.")
    mode.add_argument("--selftest", action="store_true",
                      help="Run the hermetic self-test and exit.")

    p.add_argument("--registry", type=str, default="data/registry.parquet")
    p.add_argument("--uploads", type=str, default=None,
                   help="Directory with the dataset archives (auto-fills roots).")
    p.add_argument("--solidifi", type=str, default=None)
    p.add_argument("--curated", type=str, default=None)
    p.add_argument("--smartbugs", type=str, default=None,
                   help="Path to the SmartBugs framework (its tools/ dir is used "
                        "for the classification map).")
    p.add_argument("--tools-dir", type=str, default=None,
                   help="SmartBugs 'tools/' directory (defaults to "
                        "<--smartbugs>/tools).")
    p.add_argument("--smartbugs-bin", type=str, default="smartbugs",
                   help="SmartBugs executable for --run-tools.")
    p.add_argument("--results-dir", type=str, default="artifacts/sb_runs")
    p.add_argument("--panel", nargs="+", default=None,
                   help=f"Override the tool panel (default: {' '.join(DEFAULT_PANEL)}).")
    p.add_argument("--tolerance", type=int, default=LINE_TOLERANCE_DEFAULT,
                   help="Line-match tolerance (default 0 = exact).")
    p.add_argument("--out", type=str, default="artifacts")
    return p


def _resolve_tools_dir(args) -> None:
    if args.tools_dir is None and args.smartbugs:
        args.tools_dir = str(Path(args.smartbugs) / "tools")


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    _resolve_tools_dir(args)

    # Default action (no mode flag) is the self-test: safe, no side effects.
    if args.selftest or not (args.run_tools or args.build_labels):
        return run_selftest()

    if args.run_tools:
        stage01 = load_stage01()
        registry = pd.read_parquet(args.registry, engine="pyarrow")
        work = Path(args.out) / "_work"
        solidifi = args.solidifi or (Path(args.uploads) / "SolidiFI-benchmark-master.zip")
        curated = args.curated or (Path(args.uploads) / "smartbugs-curated-main.zip")
        sroot = stage01.resolve_dataset_dir(Path(solidifi), "buggy_contracts", work, "solidifi")
        croot = stage01.resolve_dataset_dir(Path(curated), "vulnerabilities.json", work, "sb_curated")
        panel = tuple(args.panel) if args.panel else DEFAULT_PANEL
        return run_tools(registry, sroot, croot, panel, Path(args.results_dir),
                         args.smartbugs_bin)

    return do_build_labels(args)


if __name__ == "__main__":
    raise SystemExit(main())
