#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_download_data.py -- Corpus acquisition, integrity verification, and registry construction.

RELIANT: Calibrated Reliability Prediction and Budget-Constrained Portfolio
Selection for Smart Contract Security Analyzers.

--------------------------------------------------------------------------------
ROLE IN THE ARTIFACT
--------------------------------------------------------------------------------
This is the foundational, reproducibility-critical stage of the pipeline. It
ingests the two *core* datasets, canonicalizes every contract into a single
typed table (``data/registry.parquet``), and pins the exact corpus bytes with a
content SHA-256 manifest so that every downstream number in the paper is
regenerable from a clean checkout. It writes three artifacts:

    data/registry.parquet    one row per Solidity contract (the unit of analysis)
    data/registry_meta.json  provenance, taxonomy, counts, corpus fingerprint
    data/manifest.sha256     per-file content hashes (the reproducibility anchor)

Nothing here trains, predicts, or runs any analysis tool; keeping acquisition
separate from modelling is what makes the corpus auditable.

--------------------------------------------------------------------------------
WHY LOCAL-SNAPSHOT-FIRST (AND NOT A LIVE DOWNLOAD)
--------------------------------------------------------------------------------
GitHub "archive" tarballs/zips for a branch are *not* byte-stable: the same
branch re-downloaded later can differ (recompression, submodule state, .git
attributes), so hashing a freshly downloaded archive is a poor reproducibility
guarantee. Instead we ingest a *pinned local snapshot* (the zips the authors
provide / the user has archived) and compute an independent content hash over
the Solidity sources themselves. Upstream provenance -- repository URL, the
recommended pinned commit, and the citation -- is recorded in the metadata, but
the scientific guarantee is the content manifest, which is exact and offline.
This design also runs with no network access at all.

--------------------------------------------------------------------------------
DATASETS (verified structure; see also the metadata this script emits)
--------------------------------------------------------------------------------
SolidiFI-benchmark  (TRAIN + injected ground truth)
    buggy_contracts/<Class>/buggy_<N>.sol        50 contracts per class, N in 1..50
    buggy_contracts/<Class>/BugLog_<N>.csv       injection log (authoritative)
    7 classes x 50 = 350 contracts.
    * The BugLog "bug type" column is UTF-7-mangled (e.g. "Re+AC0-erntrancy"),
      so the *directory name* is authoritative for the class and only the `loc`
      column (the injected line) is trusted. This script therefore never reads
      the "bug type" column.
    * `buggy_<N>.sol` is derived from the *same base contract* across all seven
      class directories (empirically verified: buggy_1 == EIP20Interface,
      buggy_2 == CareerOnToken, ... in every class dir). Hence the
      leakage-safe split key is base_id = N: all seven injected variants of a
      base must fall in the same train/calibration/test fold. Getting this wrong
      silently leaks near-identical contracts across folds.
    * The shipped results/<Tool>/ directory is inconsistent across tools (for
      some tools the per-contract BugLog equals the injection log, for others it
      does not), so RELIANT does NOT use it. Detection outcomes are regenerated
      from scratch in 02_ground_truth.py via SmartBugs 2.0.

smartbugs-curated   (real-world TEST set)
    dataset/<dasp_category>/*.sol                143 real annotated contracts
    vulnerabilities.json                         line-level labels keyed by path
    Each contract is unique (no injected variants), so its base_id is itself.

DeFiHackLabs and the SmartBugs framework are recorded for provenance only; the
former is used solely for the RQ5 real-exploit case study (it contains Foundry
PoC scripts, not labelled prediction instances) and is not enrolled here.

--------------------------------------------------------------------------------
HOW THIS STAGE ADDRESSES THE PRIOR REVIEW
--------------------------------------------------------------------------------
The registry stores, for every contract, a canonical vulnerability class
harmonized across the two datasets' different taxonomies, plus a ground-truth
reference and an explicit base_id. This is the substrate that lets later stages
(a) predict tool reliability from contract-only features (no tool alerts ever
enter the corpus), and (b) evaluate honestly on a real-world test set rather
than only on synthetic injected bugs. It also gives IEEE Transactions on
Reliability exactly the reproducible provenance its guidelines expect.

--------------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------------
    # Build the registry from local snapshots (zip files OR extracted dirs):
    python3 src/01_download_data.py \
        --solidifi /path/SolidiFI-benchmark-master.zip \
        --curated  /path/smartbugs-curated-main.zip \
        --smartbugs /path/smartbugs-master.zip \
        --defihacklabs /path/DeFiHackLabs-main.zip \
        --out data

    # Convenience: point at a directory that contains the four archives:
    python3 src/01_download_data.py --uploads /mnt/user-data/uploads --out data

    # Re-verify an existing registry (structure + manifest only):
    python3 src/01_download_data.py --out data --verify-only

    # Full content verification: also re-hash every .sol from disk:
    python3 src/01_download_data.py --out data --verify-only --uploads /path/to/archives

    # Hermetic self-test (no datasets, no writes; safe to run anywhere):
    python3 src/01_download_data.py --selftest

Exit codes: 0 = success; 2 = integrity/verification failure; 3 = usage error.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import pandas as pd

# ------------------------------------------------------------------------------
# Versioning (bumped when the registry schema or taxonomy changes)
# ------------------------------------------------------------------------------
__version__ = "1.1.0"
SCHEMA_VERSION = "reliant-registry-1"

# Registry column order (also the schema contract for every downstream stage).
COLUMNS: Tuple[str, ...] = (
    "contract_id",           # stable, human-readable unique id
    "dataset",               # "solidifi" | "sb_curated"
    "category_raw",          # source taxonomy label (directory / DASP category)
    "class_canonical",       # harmonized RELIANT class, or "unmapped"
    "base_id",               # leakage-safe grouping key for train/calib/test
    "relpath",               # path of the .sol relative to its dataset root
    "ground_truth_ref",      # where the label lives (resolved in stage 02)
    "has_ground_truth",      # bool: a resolvable label exists
    "n_ground_truth_items",  # injected bugs (SolidiFI) / annotations (curated)
    "sha256",                # content hash of the .sol (integrity)
    "n_bytes",               # file size in bytes
    "n_lines",               # number of source lines
    "pragma_solidity",       # normalized solidity version constraint, or ""
    "source_url",            # upstream repository (provenance)
)

# ------------------------------------------------------------------------------
# Canonical vulnerability taxonomy
# ------------------------------------------------------------------------------
# RELIANT predicts per-(tool, class) reliability. The only classes it can *train*
# on are the seven SolidiFI injected classes, so those define the canonical set.
# The curated (DASP) taxonomy is mapped INTO it where a defensible correspondence
# exists; categories with no clean SolidiFI counterpart are marked "unmapped" and
# are excluded from the seven-class reliability targets (they can still serve for
# feature-distribution / detection analysis). Every mapping decision is explicit.
CANONICAL_CLASSES: Tuple[str, ...] = (
    "arithmetic",                    # SWC-101 integer overflow / underflow
    "reentrancy",                    # SWC-107
    "timestamp_dependency",          # SWC-116 block-value dependence
    "transaction_order_dependency",  # SWC-114 (a.k.a. front-running / race)
    "tx_origin",                     # SWC-115 authorization via tx.origin
    "unchecked_low_level_calls",     # SWC-104 unchecked send / call return value
    "unhandled_exceptions",          # unpropagated call exceptions
)

# Keys are lower-cased raw labels from *either* dataset.
CLASS_CANONICAL_MAP: Dict[str, str] = {
    # -- SolidiFI directory names -------------------------------------------------
    "overflow-underflow":        "arithmetic",
    "re-entrancy":               "reentrancy",
    "timestamp-dependency":      "timestamp_dependency",
    "tod":                       "transaction_order_dependency",
    "tx.origin":                 "tx_origin",
    "unchecked-send":            "unchecked_low_level_calls",
    "unhandled-exceptions":      "unhandled_exceptions",
    # -- smartbugs-curated DASP categories ---------------------------------------
    "arithmetic":                "arithmetic",
    "reentrancy":                "reentrancy",
    "time_manipulation":         "timestamp_dependency",
    # front-running (SWC-114) and transaction-order-dependency describe the same
    # ordering/race hazard under different names; the correspondence is standard
    # but approximate, and is disclosed as such in the paper.
    "front_running":             "transaction_order_dependency",
    "unchecked_low_level_calls": "unchecked_low_level_calls",
    # No clean SolidiFI counterpart -> excluded from the seven-class targets:
    "access_control":            "unmapped",  # broader than tx.origin
    "bad_randomness":            "unmapped",
    "denial_of_service":         "unmapped",
    "short_addresses":           "unmapped",
    "other":                     "unmapped",
}


def canonicalize(raw: str) -> str:
    """Map a raw dataset category label to the canonical RELIANT class."""
    return CLASS_CANONICAL_MAP.get(raw.strip().lower(), "unmapped")


# ------------------------------------------------------------------------------
# Upstream provenance (recorded in metadata; not used for the integrity guarantee)
# ------------------------------------------------------------------------------
PROVENANCE: Dict[str, Dict[str, Optional[str]]] = {
    "solidifi": {
        "name": "SolidiFI-benchmark",
        "repo": "https://github.com/DependableSystemsLab/SolidiFI-benchmark",
        "recommended_commit": "4b0573e1b3f7031396de6f48f7f3e7380222ad3a",
        "archive": "SolidiFI-benchmark-master.zip",
        "marker": "buggy_contracts",
        "license": "MIT",
        "citation": ("Ghaleb & Pattabiraman, ISSTA 2020 -- How Effective Are "
                     "Smart Contract Analysis Tools? Evaluating Smart Contract "
                     "Static Analysis Tools Using Bug Injection."),
    },
    "sb_curated": {
        "name": "smartbugs-curated",
        "repo": "https://github.com/smartbugs/smartbugs-curated",
        "recommended_commit": "230e649123477eff332742a59a1c7cc6dc286cab",
        "archive": "smartbugs-curated-main.zip",
        "marker": "vulnerabilities.json",
        "license": "See repository (curated set; per-contract licenses vary).",
        "citation": ("Durieux, Ferreira, Abreu & Cruz, ICSE 2020 -- Empirical "
                     "Review of Automated Analysis Tools on 47,587 Ethereum "
                     "Smart Contracts; Ferreira et al. -- SmartBugs: A Framework "
                     "to Analyze Solidity Smart Contracts."),
    },
    # Framework + case-study corpus: provenance only, never enrolled here.
    "smartbugs": {
        "name": "SmartBugs 2.0 (execution framework)",
        "repo": "https://github.com/smartbugs/smartbugs",
        "recommended_commit": "7ddaf6c1f5374c933481fde767b853393e9831b5",
        "archive": "smartbugs-master.zip",
        "marker": "sb",
        "license": "See repository.",
        "citation": ("di Angelo, Durieux, Ferreira & Salzer, ASE 2023 -- "
                     "SmartBugs 2.0: An Execution Framework for Weakness "
                     "Detection in Ethereum Smart Contracts."),
    },
    "defihacklabs": {
        "name": "DeFiHackLabs",
        "repo": "https://github.com/SunWeb3Sec/DeFiHackLabs",
        "recommended_commit": None,  # moving corpus; pinned only by content hash
        "archive": "DeFiHackLabs-main.zip",
        "marker": "src",
        "license": "See repository.",
        "citation": ("SunWeb3Sec -- DeFiHackLabs (real-world DeFi exploit PoCs; "
                     "used for the RQ5 case study only)."),
    },
}

# Expected corpus sizes, asserted after a real build to catch silent truncation.
EXPECTED_COUNTS = {"solidifi": 350, "sb_curated": 143, "solidifi_bases": 50}

_PRAGMA_RE = re.compile(r"pragma\s+solidity\s+([^;]+);", re.IGNORECASE)
_BASE_IDX_RE = re.compile(r"buggy_(\d+)\.sol$", re.IGNORECASE)


# ==============================================================================
# Low-level helpers
# ==============================================================================
def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """Streaming SHA-256 of a file (chunked to respect a small RAM budget)."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def extract_pragma(text: str) -> str:
    """Return the first normalized `pragma solidity` version constraint, or ""."""
    m = _PRAGMA_RE.search(text)
    if not m:
        return ""
    # Collapse internal whitespace so ">=0.4.22  <0.6.0" == ">=0.4.22 <0.6.0".
    return re.sub(r"\s+", " ", m.group(1).strip())


def count_source_lines(raw: bytes) -> int:
    """Number of source lines. latin-1 never raises, and line count is
    encoding-independent for newline handling, so this is safe on any bytes."""
    if not raw:
        return 0
    text = raw.decode("latin-1")
    n = text.count("\n")
    # Count a final unterminated line as a line.
    if not text.endswith("\n"):
        n += 1
    return n


def count_injected_bugs(buglog_path: Path) -> int:
    """Count injected-bug rows in a SolidiFI BugLog CSV.

    Robust to the UTF-7 mojibake in the "bug type" column and to stray blank
    lines: we count only rows whose first field (`loc`) is an integer, which is
    exactly one row per injected bug. We never read the mangled type column.
    """
    n = 0
    # latin-1 guarantees no decode error; we only inspect the first column.
    with buglog_path.open("r", encoding="latin-1", newline="") as fh:
        for row in csv.reader(fh):
            if row and row[0].strip().isdigit():
                n += 1
    return n


def parse_base_index(sol_name: str) -> Optional[int]:
    """Extract N from 'buggy_<N>.sol' (the SolidiFI base-contract index)."""
    m = _BASE_IDX_RE.search(sol_name)
    return int(m.group(1)) if m else None


# ==============================================================================
# Safe archive handling
# ==============================================================================
def _is_within(base: Path, target: Path) -> bool:
    """True iff `target` resolves inside `base` (zip-slip guard)."""
    try:
        target.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def safe_extract_zip(zip_path: Path, dest: Path) -> None:
    """Extract a zip while rejecting path traversal and absolute members."""
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            # Reject absolute paths and any member escaping the destination.
            if member.startswith(("/", "\\")) or ".." in Path(member).parts:
                raise ValueError(f"Unsafe path in archive {zip_path.name}: {member!r}")
            out_path = dest / member
            if not _is_within(dest, out_path):
                raise ValueError(f"Zip-slip attempt in {zip_path.name}: {member!r}")
        zf.extractall(dest)


_MARKER_SEARCH_MAX_DEPTH = 4


def _find_marker_dir(root: Path, marker: str,
                     max_depth: int = _MARKER_SEARCH_MAX_DEPTH) -> Optional[Path]:
    """Locate the dataset root by searching for a marker file/dir.

    Handles the common GitHub layout where the real content sits one level down
    inside a "<repo>-<branch>/" wrapper directory. The fallback is a genuinely
    depth-bounded breadth-first scan (at most `max_depth` levels below `root`):
    an unbounded rglob over a large extracted corpus is both slow and
    non-deterministic in cost, and every layout we support puts the marker within
    two levels. Directories are visited in sorted order so resolution is
    reproducible.
    """
    if (root / marker).exists():
        return root
    frontier = [root]
    for _ in range(max_depth):
        next_frontier: List[Path] = []
        for parent in frontier:
            try:
                children = sorted(p for p in parent.iterdir() if p.is_dir())
            except (PermissionError, OSError):
                continue
            for child in children:
                if (child / marker).exists():
                    return child
                next_frontier.append(child)
        if not next_frontier:
            break
        frontier = next_frontier
    return None


def resolve_dataset_dir(src: Path, marker: str, workdir: Path, key: str) -> Path:
    """Return a directory containing `marker`, extracting a zip if needed.

    `src` may be a directory (used as-is) or a .zip (extracted under `workdir`).
    Extraction is idempotent: an existing extraction is reused.
    """
    if not src.exists():
        raise FileNotFoundError(f"{key}: path does not exist: {src}")

    if src.is_dir():
        found = _find_marker_dir(src, marker)
        if found is None:
            raise FileNotFoundError(
                f"{key}: could not find marker {marker!r} under directory {src}")
        return found

    if src.suffix.lower() == ".zip":
        target = workdir / key
        if not target.exists():
            safe_extract_zip(src, target)
        found = _find_marker_dir(target, marker)
        if found is None:
            raise FileNotFoundError(
                f"{key}: could not find marker {marker!r} inside archive {src}")
        return found

    raise ValueError(f"{key}: expected a directory or a .zip, got {src}")


# ==============================================================================
# Dataset iterators -> registry records
# ==============================================================================
def iter_solidifi(root: Path) -> Iterator[dict]:
    """Yield one registry record per SolidiFI buggy contract.

    `root` is the directory containing `buggy_contracts/`.
    """
    bc = root / "buggy_contracts"
    class_dirs = sorted(p for p in bc.iterdir() if p.is_dir())
    if not class_dirs:
        raise FileNotFoundError(f"No class directories under {bc}")

    url = str(PROVENANCE["solidifi"]["repo"])
    for cdir in class_dirs:
        category_raw = cdir.name  # authoritative class label (not the CSV column)
        canonical = canonicalize(category_raw)
        for sol in sorted(cdir.glob("buggy_*.sol")):
            idx = parse_base_index(sol.name)
            if idx is None:
                # Skip anything not matching the buggy_<N>.sol contract pattern.
                continue
            raw = sol.read_bytes()
            buglog = cdir / f"BugLog_{idx}.csv"
            has_gt = buglog.exists()
            yield {
                "contract_id": f"solidifi__{category_raw}__buggy_{idx}",
                "dataset": "solidifi",
                "category_raw": category_raw,
                "class_canonical": canonical,
                # Same base contract across all seven classes -> group by index.
                "base_id": f"solidifi_base_{idx:02d}",
                "relpath": str(sol.relative_to(root).as_posix()),
                "ground_truth_ref": (
                    str(buglog.relative_to(root).as_posix()) if has_gt else None),
                "has_ground_truth": bool(has_gt),
                "n_ground_truth_items": count_injected_bugs(buglog) if has_gt else 0,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "n_bytes": len(raw),
                "n_lines": count_source_lines(raw),
                "pragma_solidity": extract_pragma(raw.decode("latin-1")),
                "source_url": url,
            }


def _load_curated_labels(root: Path) -> Dict[str, dict]:
    """Index vulnerabilities.json by the contract's dataset-relative path."""
    vj = root / "vulnerabilities.json"
    with vj.open("r", encoding="utf-8") as fh:
        records = json.load(fh)
    index: Dict[str, dict] = {}
    for rec in records:
        # Normalize the stored path to posix relative form for a robust join.
        index[str(Path(rec["path"]).as_posix())] = rec
    return index


def iter_curated(root: Path) -> Iterator[dict]:
    """Yield one registry record per smartbugs-curated contract.

    `root` is the directory containing `vulnerabilities.json` and `dataset/`.
    """
    labels = _load_curated_labels(root)
    dataset_dir = root / "dataset"
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"No dataset/ directory under {root}")

    url = str(PROVENANCE["sb_curated"]["repo"])
    seen_paths = set()
    for sol in sorted(dataset_dir.rglob("*.sol")):
        relpath = str(sol.relative_to(root).as_posix())
        seen_paths.add(relpath)
        # category = the immediate parent directory (primary DASP category).
        category_raw = sol.parent.name
        canonical = canonicalize(category_raw)
        raw = sol.read_bytes()
        rec = labels.get(relpath)
        if rec is not None:
            n_items = len(rec.get("vulnerabilities", []))
            gt_ref = f"vulnerabilities.json#{relpath}"
            has_gt = True
        else:
            n_items, gt_ref, has_gt = 0, None, False
        stem = sol.stem
        yield {
            "contract_id": f"sbcur__{category_raw}__{stem}",
            "dataset": "sb_curated",
            "category_raw": category_raw,
            "class_canonical": canonical,
            # Each curated contract is unique -> it is its own base.
            "base_id": f"sbcur__{category_raw}__{stem}",
            "relpath": relpath,
            "ground_truth_ref": gt_ref,
            "has_ground_truth": has_gt,
            "n_ground_truth_items": n_items,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "n_bytes": len(raw),
            "n_lines": count_source_lines(raw),
            "pragma_solidity": extract_pragma(raw.decode("latin-1")),
            "source_url": url,
        }

    # Warn (do not fail) if the JSON references files not present on disk.
    missing = set(labels) - seen_paths
    if missing:
        sys.stderr.write(
            f"[warn] {len(missing)} path(s) in vulnerabilities.json have no .sol "
            f"on disk (e.g. {sorted(missing)[0]})\n")


# ==============================================================================
# Registry assembly + integrity artifacts
# ==============================================================================
def build_registry(solidifi_dir: Path, curated_dir: Path) -> pd.DataFrame:
    """Construct the deterministic contract registry from both datasets."""
    records: List[dict] = []
    records.extend(iter_solidifi(solidifi_dir))
    records.extend(iter_curated(curated_dir))
    if not records:
        raise RuntimeError("No contracts discovered; check dataset paths.")

    df = pd.DataFrame.from_records(records, columns=list(COLUMNS))
    # Determinism: a stable row order independent of filesystem iteration order.
    df = df.sort_values("contract_id", kind="mergesort").reset_index(drop=True)

    _validate_structure(df)
    return df


def _validate_structure(df: pd.DataFrame) -> None:
    """Assert invariants that must hold for ANY registry (full or partial).

    These are corpus-size-independent so that the same check guards both a
    freshly built registry and one being re-verified from disk, and so that the
    hermetic self-test (which uses a tiny fixture) exercises the identical code.
    """
    # Schema.
    if list(df.columns) != list(COLUMNS):
        raise AssertionError(f"Column mismatch: {list(df.columns)}")
    if len(df) == 0:
        raise AssertionError("Registry is empty")
    # Unique ids.
    dups = df["contract_id"][df["contract_id"].duplicated()].tolist()
    if dups:
        raise AssertionError(f"Duplicate contract_id(s): {dups[:5]}")
    # No missing values in structurally required columns.
    for col in ("sha256", "relpath", "contract_id", "base_id", "dataset"):
        if df[col].isna().any() or (df[col].astype(str).str.len() == 0).any():
            raise AssertionError(f"Empty values in required column {col!r}")
    # sha256 must look like a 64-hex digest.
    if not df["sha256"].astype(str).str.fullmatch(r"[0-9a-f]{64}").all():
        raise AssertionError("Malformed sha256 value(s) in registry")
    # Canonical labels are in-vocabulary (or the explicit sentinel).
    allowed = set(CANONICAL_CLASSES) | {"unmapped"}
    bad = sorted(set(df["class_canonical"]) - allowed)
    if bad:
        raise AssertionError(f"Unknown canonical class(es): {bad}")

    # SolidiFI leakage invariant (holds for any subset): a base_id must contain
    # at most ONE contract per source class directory. Keyed on category_raw
    # rather than class_canonical so that a future SolidiFI release adding a
    # directory outside the canonical taxonomy (which would map several dirs to
    # "unmapped") reports the real problem instead of a spurious duplicate-class
    # error. This is what guarantees base_id is a sound split key.
    sol = df[df.dataset == "solidifi"]
    if len(sol) > 0:
        per_base = sol.groupby("base_id")
        span = per_base["category_raw"].nunique()
        size = per_base.size()
        if not (span == size).all():
            offenders = size[span != size].index.tolist()[:5]
            raise AssertionError(
                f"SolidiFI base_id(s) with two contracts from the same class "
                f"directory (base_id is not a sound split key): {offenders}")


def _assert_full_corpus(df: pd.DataFrame) -> None:
    """Assert the exact expected magnitudes of the REAL corpus.

    Called once at build time on the complete datasets to catch silent
    truncation (a partial checkout, a half-extracted zip, a missing class dir).
    Skipped when the user deliberately builds a subset (--allow-partial).
    """
    counts = df["dataset"].value_counts().to_dict()
    if counts.get("solidifi") != EXPECTED_COUNTS["solidifi"]:
        raise AssertionError(
            f"Expected {EXPECTED_COUNTS['solidifi']} SolidiFI contracts, found "
            f"{counts.get('solidifi')}")
    if counts.get("sb_curated") != EXPECTED_COUNTS["sb_curated"]:
        raise AssertionError(
            f"Expected {EXPECTED_COUNTS['sb_curated']} curated contracts, found "
            f"{counts.get('sb_curated')}")
    n_bases = df.loc[df.dataset == "solidifi", "base_id"].nunique()
    if n_bases != EXPECTED_COUNTS["solidifi_bases"]:
        raise AssertionError(
            f"Expected {EXPECTED_COUNTS['solidifi_bases']} SolidiFI base ids, "
            f"found {n_bases}")
    # In the full corpus every base must carry exactly the seven class variants.
    sizes = df[df.dataset == "solidifi"].groupby("base_id").size()
    if not (sizes == len(CANONICAL_CLASSES)).all():
        raise AssertionError(
            "Some SolidiFI base is missing class variants (expected 7 each)")


def _corpus_manifest_lines(df: pd.DataFrame) -> List[str]:
    """Deterministic 'sha256  dataset/relpath' lines, sorted by path."""
    lines = [f"{row.sha256}  {row.dataset}/{row.relpath}"
             for row in df.itertuples(index=False)]
    return sorted(lines)


def _corpus_fingerprint(manifest_lines: List[str]) -> str:
    """A single 64-hex fingerprint of the entire corpus (hash of the manifest)."""
    h = hashlib.sha256()
    for line in manifest_lines:
        h.update(line.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def write_outputs(df: pd.DataFrame, out_dir: Path,
                  provenance_used: List[str]) -> dict:
    """Write registry.parquet, manifest.sha256, and registry_meta.json."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) The typed registry.
    reg_path = out_dir / "registry.parquet"
    df.to_parquet(reg_path, engine="pyarrow", index=False)

    # 2) The content manifest (the reproducibility anchor).
    manifest_lines = _corpus_manifest_lines(df)
    (out_dir / "manifest.sha256").write_text(
        "\n".join(manifest_lines) + "\n", encoding="utf-8")
    fingerprint = _corpus_fingerprint(manifest_lines)

    # 3) Human- and machine-readable metadata.
    by_class = (df["class_canonical"].value_counts().sort_index().to_dict())
    by_dataset_category = (
        df.groupby(["dataset", "category_raw"]).size().sort_index()
        .rename("n").reset_index().to_dict(orient="records"))
    meta = {
        "schema_version": SCHEMA_VERSION,
        "generator": os.path.basename(__file__),
        "generator_version": __version__,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "corpus_fingerprint_sha256": fingerprint,
        "counts": {
            "total": int(len(df)),
            "by_dataset": {k: int(v) for k, v in
                           df["dataset"].value_counts().sort_index().items()},
            "by_class_canonical": {k: int(v) for k, v in by_class.items()},
            "n_base_ids": int(df["base_id"].nunique()),
            "n_with_ground_truth": int(df["has_ground_truth"].sum()),
            "by_dataset_category": by_dataset_category,
        },
        "registry_columns": list(COLUMNS),
        "canonical_classes": list(CANONICAL_CLASSES),
        "class_canonical_map": CLASS_CANONICAL_MAP,
        "provenance": {k: PROVENANCE[k] for k in provenance_used},
        "notes": [
            "Integrity guarantee is manifest.sha256 over the .sol sources; the "
            "recommended_commit fields are provenance only (GitHub archive bytes "
            "are not stable).",
            "SolidiFI class is taken from the directory name; the BugLog 'bug "
            "type' column is UTF-7-mangled and is never read.",
            "SolidiFI base_id groups the seven injected variants of each of the "
            "50 base contracts; use it as the train/calibration/test split key.",
            "The shipped SolidiFI results/ directory is inconsistent across "
            "tools and is not used; detection labels are regenerated in stage 02 "
            "via SmartBugs 2.0.",
            "DeFiHackLabs and the SmartBugs framework are recorded for provenance "
            "only and are not enrolled as prediction instances.",
        ],
    }
    (out_dir / "registry_meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=False), encoding="utf-8")
    return meta


# ==============================================================================
# Verification mode
# ==============================================================================
def verify_sources(df: pd.DataFrame, roots: Dict[str, Path]) -> Tuple[int, List[str]]:
    """Re-hash every .sol on disk and compare against the registry's sha256.

    roots maps dataset key ("solidifi" / "sb_curated") to the directory the
    registry's relpaths are relative to. Returns (n_checked, problems).
    This is the only check that can detect an edited or truncated source file;
    the manifest cross-check alone cannot, because the manifest is derived from
    the registry rather than from the bytes on disk.
    """
    problems: List[str] = []
    checked = 0
    for row in df.itertuples(index=False):
        root = roots.get(row.dataset)
        if root is None:
            continue
        path = root / str(row.relpath)
        if not path.exists():
            problems.append(f"missing on disk: {row.dataset}/{row.relpath}")
            continue
        digest = sha256_file(path)
        checked += 1
        if digest != row.sha256:
            problems.append(
                f"content changed: {row.dataset}/{row.relpath} "
                f"(registry {row.sha256[:12]}..., disk {digest[:12]}...)")
    return checked, problems


def verify_registry(out_dir: Path,
                    source_roots: Optional[Dict[str, Path]] = None) -> bool:
    """Check an existing registry for drift; returns True iff everything agrees.

    Two levels of checking:

    1. ALWAYS (offline, no datasets needed) -- the registry's structural
       invariants, exact agreement between manifest.sha256 and the registry's own
       hash set, the recorded corpus fingerprint, and the recorded row count.
       This detects a hand-edited manifest, a mismatched metadata file, or a
       registry that no longer satisfies the leakage/schema invariants. It does
       NOT read the .sol files, so on its own it cannot see an edited source.

    2. WHEN `source_roots` IS GIVEN -- every .sol is re-hashed from disk and
       compared against the registry (see verify_sources). This is the full
       content guarantee and is what --verify-only performs when the dataset
       paths (or --uploads) are supplied.
    """
    reg_path = out_dir / "registry.parquet"
    man_path = out_dir / "manifest.sha256"
    if not reg_path.exists() or not man_path.exists():
        sys.stderr.write(f"[verify] missing registry or manifest in {out_dir}\n")
        return False

    df = pd.read_parquet(reg_path, engine="pyarrow")
    try:
        _validate_structure(df)
    except AssertionError as exc:
        sys.stderr.write(f"[verify] registry structure invalid: {exc}\n")
        return False

    # Cross-check: the manifest must be exactly the registry's own hash set.
    manifest_on_disk = man_path.read_text(encoding="utf-8").strip().splitlines()
    manifest_from_reg = _corpus_manifest_lines(df)
    if manifest_on_disk != manifest_from_reg:
        sys.stderr.write(
            "[verify] manifest.sha256 does not match registry contents "
            "(corpus drift or a hand-edited manifest)\n")
        return False

    fp = _corpus_fingerprint(manifest_from_reg)
    meta_path = out_dir / "registry_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        # Fingerprint recorded at build time must match the recomputed one.
        if meta.get("corpus_fingerprint_sha256") not in (None, fp):
            sys.stderr.write("[verify] corpus fingerprint mismatch vs metadata\n")
            return False
        # Recorded counts must still match the parquet (catches drift where a row
        # was added/removed but the manifest was regenerated in lockstep).
        rec_total = meta.get("counts", {}).get("total")
        if rec_total is not None and rec_total != len(df):
            sys.stderr.write(
                f"[verify] row count {len(df)} != metadata total {rec_total}\n")
            return False

    if source_roots:
        checked, problems = verify_sources(df, source_roots)
        if problems:
            sys.stderr.write(
                f"[verify] {len(problems)} source integrity problem(s):\n")
            for p in problems[:10]:
                sys.stderr.write(f"    {p}\n")
            if len(problems) > 10:
                sys.stderr.write(f"    ... and {len(problems) - 10} more\n")
            return False
        sys.stderr.write(
            f"[verify] OK -- {len(df)} contracts, {checked} sources re-hashed "
            f"from disk, fingerprint {fp[:16]}...\n")
        return True

    sys.stderr.write(
        f"[verify] OK (structure + manifest only; pass --uploads or "
        f"--solidifi/--curated to re-hash sources) -- {len(df)} contracts, "
        f"fingerprint {fp[:16]}...\n")
    return True


# ==============================================================================
# Hermetic self-test (no datasets, no writes to the real output dir)
# ==============================================================================
def _make_synthetic_corpus(root: Path) -> Tuple[Path, Path]:
    """Create a tiny fake SolidiFI + curated tree to exercise every code path."""
    # --- Synthetic SolidiFI: 2 classes x 2 bases = 4 contracts -----------------
    sdir = root / "SolidiFI-like"
    for cls in ("Re-entrancy", "Overflow-Underflow"):
        d = sdir / "buggy_contracts" / cls
        d.mkdir(parents=True)
        for n in (1, 2):
            # Same base body across classes so base_id sharing is testable.
            body = (f"pragma solidity ^0.5.0;\n\ncontract Base{n} {{\n"
                    f"    function f() public {{}}\n}}\n")
            (d / f"buggy_{n}.sol").write_text(body, encoding="utf-8")
            # BugLog with UTF-7-mangled type column + 2 injected rows.
            (d / f"BugLog_{n}.csv").write_text(
                "loc,length,bug type,approach\n"
                "5,8,Re+AC0-erntrancy,code snippet injection\n"
                "9,8,Re+AC0-erntrancy,code snippet injection\n",
                encoding="utf-8")

    # --- Synthetic curated: 2 categories, one labelled, one unmapped -----------
    cdir = root / "curated-like"
    (cdir / "dataset" / "reentrancy").mkdir(parents=True)
    (cdir / "dataset" / "bad_randomness").mkdir(parents=True)
    (cdir / "dataset" / "reentrancy" / "dao.sol").write_text(
        "pragma solidity ^0.4.24;\ncontract DAO {}\n", encoding="utf-8")
    (cdir / "dataset" / "bad_randomness" / "lottery.sol").write_text(
        "pragma solidity ^0.4.24;\ncontract Lottery {}\n", encoding="utf-8")
    (cdir / "vulnerabilities.json").write_text(json.dumps([
        {"name": "dao.sol", "path": "dataset/reentrancy/dao.sol",
         "pragma": "0.4.24", "source": "test",
         "vulnerabilities": [{"lines": [2], "category": "reentrancy"}]},
        {"name": "lottery.sol", "path": "dataset/bad_randomness/lottery.sol",
         "pragma": "0.4.24", "source": "test",
         "vulnerabilities": [{"lines": [2], "category": "bad_randomness"}]},
    ]), encoding="utf-8")
    return sdir, cdir


def run_selftest() -> int:
    """Exercise the full build on synthetic data and assert every invariant."""
    print(f"RELIANT 01_download_data self-test (v{__version__})")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        sdir, cdir = _make_synthetic_corpus(root)

        # Resolve exactly as the real CLI would (via marker discovery).
        work = root / "_work"
        sroot = resolve_dataset_dir(sdir, "buggy_contracts", work, "solidifi")
        croot = resolve_dataset_dir(cdir, "vulnerabilities.json", work, "sb_curated")

        # build_registry enforces only structural invariants, so the tiny
        # fixture builds without any relaxation of production magnitudes.
        df = build_registry(sroot, croot)

        # --- Invariants ---------------------------------------------------------
        assert len(df) == 6, f"expected 6 rows, got {len(df)}"
        assert list(df.columns) == list(COLUMNS), "schema drift"
        assert df["contract_id"].is_unique, "ids not unique"

        # base_id sharing: base 1 appears in both SolidiFI classes.
        base1 = df[df.base_id == "solidifi_base_01"]
        assert set(base1["category_raw"]) == {"Re-entrancy", "Overflow-Underflow"}, \
            "base_id must group the same base across classes"

        # Canonical mapping (including the 'unmapped' sentinel).
        cmap = dict(zip(df.category_raw, df.class_canonical))
        assert cmap["Re-entrancy"] == "reentrancy"
        assert cmap["Overflow-Underflow"] == "arithmetic"
        assert cmap["reentrancy"] == "reentrancy"
        assert cmap["bad_randomness"] == "unmapped"

        # Ground-truth counting ignores the mangled type column.
        sol_gt = df[df.dataset == "solidifi"]["n_ground_truth_items"].tolist()
        assert all(x == 2 for x in sol_gt), f"injected-bug count wrong: {sol_gt}"

        # Curated join + line labels.
        cur = df[df.dataset == "sb_curated"].set_index("category_raw")
        assert cur.loc["reentrancy", "has_ground_truth"] == True  # noqa: E712
        assert cur.loc["reentrancy", "n_ground_truth_items"] == 1

        # Pragma extraction + line counting.
        assert (df["pragma_solidity"] == "^0.5.0").sum() == 4
        assert (df["n_lines"] > 0).all()

        # Determinism: identical rebuild yields identical fingerprint.
        df2 = build_registry(sroot, croot)
        fp1 = _corpus_fingerprint(_corpus_manifest_lines(df))
        fp2 = _corpus_fingerprint(_corpus_manifest_lines(df2))
        assert fp1 == fp2, "corpus fingerprint is not deterministic"

        # Round-trip: write to a temp dir and verify() must pass.
        out = root / "out"
        write_outputs(df, out, ["solidifi", "sb_curated"])
        assert verify_registry(out), "verify_registry failed on fresh output"

        # --- REGRESSION: content verification must catch an EDITED source ------
        # The manifest is derived from the registry, so the offline check alone
        # cannot see a tampered .sol; only re-hashing from disk can.
        roots = {"solidifi": sroot, "sb_curated": croot}
        n_checked, problems = verify_sources(df, roots)
        assert n_checked == len(df) and not problems, (n_checked, problems)
        assert verify_registry(out, source_roots=roots), "full verify failed"
        victim = sroot / "buggy_contracts" / "Re-entrancy" / "buggy_1.sol"
        original = victim.read_text(encoding="utf-8")
        victim.write_text(original + "// tampered\n", encoding="utf-8")
        assert not verify_registry(out, source_roots=roots), \
            "an edited source file was NOT detected"
        _, problems = verify_sources(df, roots)
        assert len(problems) == 1 and "content changed" in problems[0]
        victim.write_text(original, encoding="utf-8")          # restore
        assert verify_registry(out, source_roots=roots), "restore should verify"
        # a source deleted from disk is reported too
        victim.unlink()
        _, problems = verify_sources(df, roots)
        assert len(problems) == 1 and "missing on disk" in problems[0]
        victim.write_text(original, encoding="utf-8")
        print("  content verification detects edited and missing sources.")

        # --- depth-bounded marker search ---------------------------------------
        nested = root / "deep"
        (nested / "a" / "b" / "MARKER").mkdir(parents=True)
        assert _find_marker_dir(nested, "MARKER") == nested / "a" / "b"
        assert _find_marker_dir(nested, "MARKER", max_depth=1) is None, \
            "search must respect its depth bound"
        assert _find_marker_dir(nested, "NO_SUCH_MARKER") is None

        # --- zip-slip guard rejects traversal members ---------------------------
        bad_zip = root / "evil.zip"
        with zipfile.ZipFile(bad_zip, "w") as zf:
            zf.writestr("../escape.sol", "pragma solidity ^0.5.0;")
        try:
            safe_extract_zip(bad_zip, root / "evil_out")
            raise AssertionError("zip-slip member was not rejected")
        except ValueError:
            pass

        # --- structural invariant is keyed on the source class directory --------
        bad = df.copy()
        sol_rows = bad.index[bad.dataset == "solidifi"].tolist()
        bad.loc[sol_rows[1], "base_id"] = bad.loc[sol_rows[0], "base_id"]
        bad.loc[sol_rows[1], "category_raw"] = bad.loc[sol_rows[0], "category_raw"]
        try:
            _validate_structure(bad)
            raise AssertionError("duplicate class-dir within a base not caught")
        except AssertionError as exc:
            assert "not a sound split key" in str(exc), exc

        # A corrupted manifest must be rejected.
        (out / "manifest.sha256").write_text("deadbeef  x/y.sol\n", encoding="utf-8")
        assert not verify_registry(out), "verify must reject a bad manifest"

    # Also sanity-check the taxonomy tables themselves.
    assert set(CLASS_CANONICAL_MAP.values()) - {"unmapped"} == set(CANONICAL_CLASSES)
    print("  all invariants passed.")
    return 0


# ==============================================================================
# CLI
# ==============================================================================
def _default_from_uploads(uploads: Path, key: str) -> Optional[Path]:
    """Map a dataset key to its archive inside an --uploads directory."""
    archive = PROVENANCE[key]["archive"]
    cand = uploads / str(archive)
    return cand if cand.exists() else None


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build the RELIANT contract registry from local dataset "
                    "snapshots (stage 01).")
    p.add_argument("--solidifi", type=Path, default=None,
                   help="Path to SolidiFI-benchmark (.zip or extracted dir).")
    p.add_argument("--curated", type=Path, default=None,
                   help="Path to smartbugs-curated (.zip or extracted dir).")
    p.add_argument("--smartbugs", type=Path, default=None,
                   help="Path to the SmartBugs framework (provenance only).")
    p.add_argument("--defihacklabs", type=Path, default=None,
                   help="Path to DeFiHackLabs (provenance only; RQ5 case study).")
    p.add_argument("--uploads", type=Path, default=None,
                   help="Directory containing the dataset archives; used to "
                        "auto-fill any --solidifi/--curated not given explicitly.")
    p.add_argument("--out", type=Path, default=Path("data"),
                   help="Output directory for registry/manifest/metadata.")
    p.add_argument("--verify-only", action="store_true",
                   help="Verify an existing registry; no rebuild. Checks "
                        "structure + manifest offline, and additionally "
                        "re-hashes every source file when --uploads or "
                        "--solidifi/--curated is supplied.")
    p.add_argument("--allow-partial", action="store_true",
                   help="Skip the exact full-corpus size assertion (use when "
                        "deliberately building on a subset of the datasets).")
    p.add_argument("--selftest", action="store_true",
                   help="Run the hermetic self-test and exit.")
    return p


def _resolve_source_roots(args) -> Dict[str, Path]:
    """Best-effort resolution of dataset roots for content verification.

    Returns {} when no dataset location was supplied, in which case verification
    falls back to the structural + manifest checks and says so explicitly.
    """
    solidifi = args.solidifi
    curated = args.curated
    if args.uploads is not None:
        solidifi = solidifi or _default_from_uploads(args.uploads, "solidifi")
        curated = curated or _default_from_uploads(args.uploads, "sb_curated")
    roots: Dict[str, Path] = {}
    work = args.out / "_work"
    for key, src, marker in (("solidifi", solidifi, "buggy_contracts"),
                             ("sb_curated", curated, "vulnerabilities.json")):
        if src is None:
            continue
        try:
            work.mkdir(parents=True, exist_ok=True)
            roots[key] = resolve_dataset_dir(Path(src), marker, work, key)
        except (FileNotFoundError, ValueError) as exc:
            sys.stderr.write(f"[verify] cannot resolve {key}: {exc}\n")
    return roots


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.selftest:
        return run_selftest()

    if args.verify_only:
        roots = _resolve_source_roots(args)
        return 0 if verify_registry(args.out, source_roots=roots or None) else 2

    # Resolve dataset locations (explicit flags win; --uploads fills the rest).
    solidifi = args.solidifi
    curated = args.curated
    if args.uploads is not None:
        solidifi = solidifi or _default_from_uploads(args.uploads, "solidifi")
        curated = curated or _default_from_uploads(args.uploads, "sb_curated")

    if solidifi is None or curated is None:
        sys.stderr.write(
            "error: both --solidifi and --curated are required (or provide "
            "--uploads pointing at a directory that contains the archives).\n")
        return 3

    work = args.out / "_work"
    work.mkdir(parents=True, exist_ok=True)
    try:
        solidifi_dir = resolve_dataset_dir(solidifi, "buggy_contracts", work, "solidifi")
        curated_dir = resolve_dataset_dir(curated, "vulnerabilities.json", work, "sb_curated")
    except (FileNotFoundError, ValueError) as exc:
        sys.stderr.write(f"error resolving datasets: {exc}\n")
        return 3

    print("Building RELIANT contract registry ...")
    print(f"  SolidiFI : {solidifi_dir}")
    print(f"  curated  : {curated_dir}")

    df = build_registry(solidifi_dir, curated_dir)

    if not args.allow_partial:
        try:
            _assert_full_corpus(df)
        except AssertionError as exc:
            sys.stderr.write(
                f"error: corpus is not complete ({exc}).\n"
                f"  The expected magnitudes {EXPECTED_COUNTS} describe the pinned\n"
                f"  snapshots in config.yaml; smartbugs-curated in particular has\n"
                f"  grown over time, so a newer checkout will legitimately differ.\n"
                f"  Re-extract the pinned archives, or pass --allow-partial to\n"
                f"  build on whatever is present (the fingerprint still pins it).\n")
            return 2

    provenance_used = ["solidifi", "sb_curated"]
    # Record (but do not enroll) the framework / case-study corpora if provided.
    if args.smartbugs is not None:
        provenance_used.append("smartbugs")
    if args.defihacklabs is not None:
        provenance_used.append("defihacklabs")

    meta = write_outputs(df, args.out, provenance_used)

    # Human-readable summary.
    c = meta["counts"]
    print(f"\nWrote {args.out}/registry.parquet  ({c['total']} contracts)")
    print(f"  by dataset : {c['by_dataset']}")
    print(f"  base ids   : {c['n_base_ids']}  |  with ground truth: "
          f"{c['n_with_ground_truth']}")
    print("  by canonical class:")
    for cls, n in c["by_class_canonical"].items():
        print(f"      {cls:<30} {n}")
    print(f"  corpus fingerprint : {meta['corpus_fingerprint_sha256'][:16]}...")
    print(f"  metadata           : {args.out}/registry_meta.json")
    print(f"  manifest           : {args.out}/manifest.sha256")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
