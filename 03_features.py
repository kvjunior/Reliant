#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_features.py -- Alert-free feature extraction (tabular + heterogeneous graph).

RELIANT: Calibrated Reliability Prediction and Budget-Constrained Portfolio
Selection for Smart Contract Security Analyzers.

--------------------------------------------------------------------------------
ROLE IN THE ARTIFACT -- AND WHY IT IS THE CIRCULARITY KILLER
--------------------------------------------------------------------------------
The prior submission was rejected in part for CIRCULARITY: it fed raw tool alerts
into the model that was supposed to predict tool behaviour "before running the
tools." This stage makes that mistake impossible by construction. Every feature
is computed from the contract's own source, abstract syntax tree, and
control-/call-structure -- artefacts that exist BEFORE any analyzer is invoked.
No file produced by stage 02 (labels, findings, timings) is ever read here, and a
runtime guard refuses any input that even looks like tool output. The extractor's
only inputs are data/registry.parquet (source-file locations) and the .sol files.

It writes:
    artifacts/features.parquet        one row per contract, ~60 numeric features
    artifacts/graphs.jsonl            one heterogeneous graph per contract (JSON)
    artifacts/features_meta.json      feature dictionary, guard record, coverage

Downstream, stage 04 combines these per-contract features with a query class to
predict per-(contract, tool, class) reliability; the tabular view feeds the Ridge
and LightGBM predictors and the graph feeds the optional Hetero-GNN.

--------------------------------------------------------------------------------
FEATURE DESIGN (all source-derived; none is a tool output)
--------------------------------------------------------------------------------
Six groups, chosen because they plausibly govern how reliably a given analyzer
detects a given class -- e.g. whether reentrancy is even detectable depends on the
presence and shape of external low-level calls; timestamp findings depend on
block-value usage; arithmetic findings depend on arithmetic operators and the
compiler version (pre-0.8 has no overflow checks):
  (1) size & lexical      lines, tokens, comment ratio, nesting depth
  (2) pragma / version    parsed solc major/minor/patch and constraint style
  (3) contract structure  contracts / libraries / interfaces / inheritance
  (4) function aggregates  visibility & mutability mix, sizes, parameters
  (5) control flow         if / loops / ternary / arithmetic & logic ops / calls
  (6) security tokens      low-level calls, .value/.send/.transfer, tx.origin,
                           block timestamp/number/hash, selfdestruct, assembly,
                           require/assert, keccak/ecrecover, mappings, new

Lexical token counts are taken from a comment-/string-stripped copy of the source
so patterns inside comments or string literals are never miscounted. Structural
counts come from the AST. The two are complementary and both alert-free.

--------------------------------------------------------------------------------
HETEROGENEOUS GRAPH
--------------------------------------------------------------------------------
Node types : contract, function, modifier, event, state_var.
Edge types : contains (contract->member), inherits (contract->contract),
             calls (function->function, intra-file), uses (function->modifier),
             emits (function->event), reads_writes (function->state_var).
Each node carries a small numeric feature dict. Graphs are emitted as one compact
JSON object per line (keyed by contract_id), streamable by stage 04.

--------------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------------
    python3 src/03_features.py --extract \
        --registry data/registry.parquet --uploads /mnt/user-data/uploads \
        --out artifacts

    python3 src/03_features.py --extract ... --limit 40   # quick subset

    python3 src/03_features.py --selftest                 # hermetic, no datasets

Exit codes: 0 = success; 2 = alert-free guard tripped; 3 = usage error.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path
from statistics import mean
from typing import Dict, Iterator, List, Optional, Tuple

import pandas as pd

__version__ = "1.1.0"
SCHEMA_VERSION = "reliant-features-1"

# ------------------------------------------------------------------------------
# Alert-free guard: nothing here may reference stage-02 tool output.
# ------------------------------------------------------------------------------
# Any input path or registry column containing one of these tokens indicates a
# tool-output artefact leaking into feature extraction, which would recreate the
# circularity the prior paper was rejected for. Two families:
#   * label/metric vocabulary (a labels table or metric column offered as input);
#   * ANALYZER NAMES and run-output markers -- tool outputs live in paths named
#     after the tools ("slither_results.json", "sb_runs/", the results/ tree
#     SolidiFI itself ships), so a registry or output path referencing one is
#     the clearest possible signal of circularity. The panel tools of stage 02
#     are all listed; "result"/"sb_run" cover the run directories.
FORBIDDEN_TOKENS: Tuple[str, ...] = (
    "label", "finding", "detect", "fnr", "fpr", "recall", "precision",
    "alert", "timing", "duration", "tp", "fp_proxy",
    "slither", "mythril", "oyente", "smartcheck", "securify", "conkas",
    "confuzzius", "semgrep", "manticore", "result", "sb_run",
)

# Registry columns the extractor is permitted to consume (all source-only).
ALLOWED_REGISTRY_COLUMNS: Tuple[str, ...] = (
    "contract_id", "dataset", "category_raw", "class_canonical", "base_id",
    "relpath", "ground_truth_ref", "has_ground_truth", "n_ground_truth_items",
    "sha256", "n_bytes", "n_lines", "pragma_solidity", "source_url",
)

# ------------------------------------------------------------------------------
# Feature dictionary (fixed order == the schema contract for stage 04)
# ------------------------------------------------------------------------------
FEATURE_NAMES: Tuple[str, ...] = (
    # (1) size & lexical
    "f_n_bytes", "f_n_lines", "f_n_code_lines", "f_n_comment_lines",
    "f_comment_ratio", "f_max_line_len", "f_n_tokens", "f_max_nesting_depth",
    # (2) pragma / version
    "f_solc_major", "f_solc_minor", "f_solc_patch", "f_pragma_caret",
    "f_pragma_range",
    # (3) contract structure
    "f_n_contracts", "f_n_libraries", "f_n_interfaces", "f_n_functions",
    "f_n_modifiers", "f_n_events", "f_n_structs", "f_n_enums", "f_n_state_vars",
    "f_n_imports", "f_max_bases", "f_n_inherit_edges",
    # (4) function aggregates
    "f_n_public_fn", "f_n_external_fn", "f_n_internal_fn", "f_n_private_fn",
    "f_n_payable_fn", "f_n_view_pure_fn", "f_n_constructor", "f_has_fallback",
    "f_n_modifier_uses", "f_mean_fn_stmts", "f_max_fn_stmts", "f_mean_fn_params",
    "f_max_fn_params",
    # (5) control flow
    "f_n_if", "f_n_for", "f_n_while", "f_n_loops", "f_n_ternary",
    "f_n_binop_arith", "f_n_binop_logic", "f_n_calls", "f_cyclomatic_proxy",
    # (6) security tokens (lexical, on stripped source)
    "f_n_low_level_call", "f_n_call_value", "f_n_delegatecall",
    "f_n_staticcall", "f_n_send", "f_n_transfer", "f_n_selfdestruct",
    "f_n_tx_origin", "f_n_msg_sender", "f_n_msg_value", "f_n_timestamp",
    "f_n_block_number", "f_n_blockhash", "f_n_assembly", "f_n_require",
    "f_n_assert", "f_n_revert_throw", "f_n_keccak_sha3", "f_n_ecrecover",
    "f_n_mapping", "f_n_new", "f_n_payable_kw", "f_low_level_total",
)

# The subset of FEATURE_NAMES derived from the AST (zeroed on parse failure).
# f_has_fallback is intentionally NOT here: it is computed lexically so it is
# robust to parser quirks (0.5 fallbacks are unnamed functions the parser mangles).
AST_FEATURE_NAMES: Tuple[str, ...] = (
    "f_n_contracts", "f_n_libraries", "f_n_interfaces", "f_n_functions",
    "f_n_modifiers", "f_n_events", "f_n_structs", "f_n_enums", "f_n_state_vars",
    "f_n_imports", "f_max_bases", "f_n_inherit_edges",
    "f_n_public_fn", "f_n_external_fn", "f_n_internal_fn", "f_n_private_fn",
    "f_n_payable_fn", "f_n_view_pure_fn", "f_n_constructor", "f_n_modifier_uses",
    "f_mean_fn_stmts", "f_max_fn_stmts", "f_mean_fn_params", "f_max_fn_params",
    "f_n_if", "f_n_for", "f_n_while", "f_n_loops", "f_n_ternary",
    "f_n_binop_arith", "f_n_binop_logic", "f_n_calls", "f_cyclomatic_proxy",
)

# Node / edge vocabularies for the heterogeneous graph.
NODE_TYPES: Tuple[str, ...] = ("contract", "function", "modifier", "event", "state_var")
EDGE_TYPES: Tuple[str, ...] = ("contains", "inherits", "calls", "uses", "emits", "reads_writes")

_VISIBILITY_CODE = {"default": 0, "public": 1, "external": 2, "internal": 3, "private": 4}

# ------------------------------------------------------------------------------
# Lexical token patterns (matched on comment-/string-stripped source)
# ------------------------------------------------------------------------------
_LEX = {
    "f_n_low_level_call": re.compile(r"\.\s*call\b"),
    "f_n_call_value": re.compile(r"\.\s*(?:call\s*\.\s*value|value)\s*\(|\.\s*call\s*\{"),
    "f_n_delegatecall": re.compile(r"\.\s*delegatecall\b"),
    "f_n_staticcall": re.compile(r"\.\s*staticcall\b"),
    "f_n_send": re.compile(r"\.\s*send\s*\("),
    "f_n_transfer": re.compile(r"\.\s*transfer\s*\("),
    "f_n_selfdestruct": re.compile(r"\b(?:selfdestruct|suicide)\s*\("),
    "f_n_tx_origin": re.compile(r"\btx\s*\.\s*origin\b"),
    "f_n_msg_sender": re.compile(r"\bmsg\s*\.\s*sender\b"),
    "f_n_msg_value": re.compile(r"\bmsg\s*\.\s*value\b"),
    "f_n_timestamp": re.compile(r"\bblock\s*\.\s*timestamp\b|\bnow\b"),
    "f_n_block_number": re.compile(r"\bblock\s*\.\s*number\b"),
    "f_n_blockhash": re.compile(r"\bblockhash\s*\(|\bblock\s*\.\s*blockhash\b"),
    "f_n_assembly": re.compile(r"\bassembly\b"),
    "f_n_require": re.compile(r"\brequire\s*\("),
    "f_n_assert": re.compile(r"\bassert\s*\("),
    "f_n_revert_throw": re.compile(r"\brevert\b|\bthrow\b"),
    "f_n_keccak_sha3": re.compile(r"\bkeccak256\s*\(|\bsha3\s*\("),
    "f_n_ecrecover": re.compile(r"\becrecover\s*\("),
    "f_n_mapping": re.compile(r"\bmapping\s*\("),
    "f_n_new": re.compile(r"\bnew\s+"),
    "f_n_payable_kw": re.compile(r"\bpayable\b"),
}
_TOKEN_RE = re.compile(r"[A-Za-z_]\w*")
_PRAGMA_RE = re.compile(r"(\d+)\.(\d+)\.(\d+)")
# Fallback across versions: unnamed `function ()`, or 0.6+ `fallback(` / `receive(`.
_FALLBACK_RE = re.compile(r"\bfunction\s*\(\s*\)|\bfallback\s*\(|\breceive\s*\(")


# ==============================================================================
# Interop with stage 01 (dataset resolution / canonicalization)
# ==============================================================================
def load_stage01():
    here = Path(__file__).resolve().parent
    path = here / "01_download_data.py"
    if not path.exists():
        raise FileNotFoundError(f"cannot locate stage 01 at {path}")
    spec = importlib.util.spec_from_file_location("reliant_stage01", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# ==============================================================================
# Source sanitizing + lexical / pragma features
# ==============================================================================
def strip_comments_and_strings(src: str) -> str:
    """Remove // and /* */ comments and "..."/'...' literals, char by char.

    Keeps newlines so line structure (and nesting depth) is preserved. This makes
    lexical token counts robust: a '.call' inside a comment or string is ignored.
    """
    out: List[str] = []
    i, n = 0, len(src)
    state = "code"  # code | line_comment | block_comment | dq_string | sq_string
    while i < n:
        c = src[i]
        nxt = src[i + 1] if i + 1 < n else ""
        if state == "code":
            if c == "/" and nxt == "/":
                state = "line_comment"; i += 2; continue
            if c == "/" and nxt == "*":
                state = "block_comment"; i += 2; continue
            if c == '"':
                state = "dq_string"; i += 1; continue
            if c == "'":
                state = "sq_string"; i += 1; continue
            out.append(c); i += 1
        elif state == "line_comment":
            if c == "\n":
                state = "code"; out.append("\n")
            i += 1
        elif state == "block_comment":
            if c == "*" and nxt == "/":
                state = "code"; i += 2; continue
            if c == "\n":
                out.append("\n")
            i += 1
        elif state == "dq_string":
            if c == "\\":
                i += 2; continue
            if c == '"':
                state = "code"
            elif c == "\n":
                out.append("\n")
            i += 1
        elif state == "sq_string":
            if c == "\\":
                i += 2; continue
            if c == "'":
                state = "code"
            elif c == "\n":
                out.append("\n")
            i += 1
    return "".join(out)


def line_stats(raw: str) -> Dict[str, float]:
    """Total / code / comment line counts and max line length from raw source."""
    lines = raw.splitlines()
    n_total = len(lines)
    n_comment = 0
    n_code = 0
    in_block = False
    for ln in lines:
        s = ln.strip()
        is_comment_line = False
        if in_block:
            is_comment_line = True
            if "*/" in s:
                in_block = False
        elif s.startswith("/*"):
            is_comment_line = True
            if "*/" not in s:
                in_block = True
        elif s.startswith("//"):
            is_comment_line = True
        if is_comment_line:
            n_comment += 1
        elif s:
            n_code += 1
    max_len = max((len(ln) for ln in lines), default=0)
    return {"n_lines": n_total, "n_code_lines": n_code,
            "n_comment_lines": n_comment, "max_line_len": max_len}


def max_brace_nesting(stripped: str) -> int:
    """Maximum { } nesting depth in comment-/string-stripped source."""
    depth = mx = 0
    for c in stripped:
        if c == "{":
            depth += 1
            mx = max(mx, depth)
        elif c == "}":
            depth = max(0, depth - 1)
    return mx


def pragma_features(pragma: str) -> Dict[str, float]:
    """Parse a solidity version constraint into numeric features."""
    m = _PRAGMA_RE.search(pragma or "")
    major, minor, patch = (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else (0, 0, 0)
    return {
        "f_solc_major": major, "f_solc_minor": minor, "f_solc_patch": patch,
        "f_pragma_caret": 1 if "^" in (pragma or "") else 0,
        "f_pragma_range": 1 if (">" in (pragma or "") or "<" in (pragma or "")) else 0,
    }


def lexical_features(stripped: str) -> Dict[str, float]:
    """Count security-relevant tokens on the stripped source."""
    feats = {name: len(rx.findall(stripped)) for name, rx in _LEX.items()}
    feats["f_n_tokens"] = len(_TOKEN_RE.findall(stripped))
    feats["f_has_fallback"] = 1 if _FALLBACK_RE.search(stripped) else 0
    feats["f_low_level_total"] = (
        feats["f_n_low_level_call"] + feats["f_n_delegatecall"]
        + feats["f_n_staticcall"] + feats["f_n_send"] + feats["f_n_transfer"])
    return feats


# ==============================================================================
# AST walking + structural features
# ==============================================================================
def parse_ast(src: str):
    """Parse Solidity source to an AST dict, or None on failure (lazy import)."""
    try:
        from solidity_parser import parser as sp
    except Exception:  # pragma: no cover - dependency missing
        return None
    try:
        return sp.parse(src, loc=False)
    except Exception:
        return None


def walk(node) -> Iterator[dict]:
    """Yield every dict node (with a 'type') in an AST subtree."""
    if isinstance(node, dict):
        if "type" in node:
            yield node
        for v in node.values():
            yield from walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from walk(v)


def _count_types(node) -> Dict[str, int]:
    acc: Dict[str, int] = {}
    for nd in walk(node):
        t = nd.get("type")
        acc[t] = acc.get(t, 0) + 1
    return acc


def _binop_counts(node) -> Tuple[int, int]:
    """(arithmetic, logical) BinaryOperation counts in a subtree."""
    arith = logic = 0
    for nd in walk(node):
        if nd.get("type") == "BinaryOperation":
            op = nd.get("operator")
            if op in ("+", "-", "*", "/", "%", "**"):
                arith += 1
            elif op in ("&&", "||"):
                logic += 1
    return arith, logic


def _contract_kind_flags(kind: str) -> Tuple[int, int]:
    return (1 if kind == "library" else 0, 1 if kind == "interface" else 0)


def ast_features(ast) -> Tuple[Dict[str, float], List[dict]]:
    """Structural features from the AST, plus the raw contract node list."""
    contracts = [c for c in ast.get("children", []) if c.get("type") == "ContractDefinition"]
    n_imports = sum(1 for c in ast.get("children", []) if c.get("type") == "ImportDirective")

    n_lib = n_iface = 0
    n_functions = n_modifiers = n_events = n_structs = n_enums = n_state_vars = 0
    n_public = n_external = n_internal = n_private = n_payable = n_viewpure = 0
    n_constructor = n_modifier_uses = 0
    max_bases = n_inherit_edges = 0
    fn_stmt_counts: List[int] = []
    fn_param_counts: List[int] = []
    n_if = n_for = n_while = n_ternary = n_calls = 0
    arith_total = logic_total = 0

    for c in contracts:
        lib, iface = _contract_kind_flags(str(c.get("kind", "contract")))
        n_lib += lib
        n_iface += iface
        bases = c.get("baseContracts", []) or []
        max_bases = max(max_bases, len(bases))
        n_inherit_edges += len(bases)
        for sn in c.get("subNodes", []) or []:
            t = sn.get("type")
            if t == "FunctionDefinition":
                n_functions += 1
                vis = sn.get("visibility") or "default"
                n_public += vis == "public"
                n_external += vis == "external"
                n_internal += vis == "internal"
                n_private += vis == "private"
                sm = sn.get("stateMutability")
                n_payable += sm == "payable"
                n_viewpure += sm in ("view", "pure", "constant")
                n_constructor += 1 if sn.get("isConstructor") else 0
                n_modifier_uses += len(sn.get("modifiers", []) or [])
                params = sn.get("parameters")
                fn_param_counts.append(_param_count(params))
                body = sn.get("body")
                types = _count_types(body) if body else {}
                fn_stmt_counts.append(sum(v for k, v in types.items()
                                          if k.endswith("Statement")))
                n_if += types.get("IfStatement", 0)
                n_for += types.get("ForStatement", 0)
                n_while += types.get("WhileStatement", 0) + types.get("DoWhileStatement", 0)
                n_ternary += types.get("Conditional", 0)
                n_calls += types.get("FunctionCall", 0)
                a, lg = _binop_counts(body) if body else (0, 0)
                arith_total += a
                logic_total += lg
            elif t == "ModifierDefinition":
                n_modifiers += 1
            elif t == "EventDefinition":
                n_events += 1
            elif t == "StructDefinition":
                n_structs += 1
            elif t == "EnumDefinition":
                n_enums += 1
            elif t == "StateVariableDeclaration":
                n_state_vars += len(sn.get("variables", []) or [])

    n_loops = n_for + n_while
    feats = {
        "f_n_contracts": len(contracts), "f_n_libraries": n_lib,
        "f_n_interfaces": n_iface, "f_n_functions": n_functions,
        "f_n_modifiers": n_modifiers, "f_n_events": n_events,
        "f_n_structs": n_structs, "f_n_enums": n_enums,
        "f_n_state_vars": n_state_vars, "f_n_imports": n_imports,
        "f_max_bases": max_bases, "f_n_inherit_edges": n_inherit_edges,
        "f_n_public_fn": n_public, "f_n_external_fn": n_external,
        "f_n_internal_fn": n_internal, "f_n_private_fn": n_private,
        "f_n_payable_fn": n_payable, "f_n_view_pure_fn": n_viewpure,
        "f_n_constructor": n_constructor,
        "f_n_modifier_uses": n_modifier_uses,
        "f_mean_fn_stmts": round(mean(fn_stmt_counts), 3) if fn_stmt_counts else 0,
        "f_max_fn_stmts": max(fn_stmt_counts) if fn_stmt_counts else 0,
        "f_mean_fn_params": round(mean(fn_param_counts), 3) if fn_param_counts else 0,
        "f_max_fn_params": max(fn_param_counts) if fn_param_counts else 0,
        "f_n_if": n_if, "f_n_for": n_for, "f_n_while": n_while,
        "f_n_loops": n_loops, "f_n_ternary": n_ternary,
        "f_n_binop_arith": arith_total, "f_n_binop_logic": logic_total,
        "f_n_calls": n_calls,
        "f_cyclomatic_proxy": 1 + n_if + n_loops + n_ternary + logic_total,
    }
    return feats, contracts


def _param_count(params) -> int:
    """Robustly count parameters across parser shapes (list or ParameterList)."""
    if params is None:
        return 0
    if isinstance(params, list):
        return len(params)
    if isinstance(params, dict):
        p = params.get("parameters")
        return len(p) if isinstance(p, list) else 0
    return 0


def _is_fallback(sn: dict) -> bool:
    """True for a fallback function.

    Handles both the 0.6+ `fallback` keyword (isFallback flag) and the 0.4/0.5
    unnamed-function form `function () external payable { ... }`, where the parser
    leaves the name empty and does not set isFallback.
    """
    if sn.get("isFallback"):
        return True
    name = sn.get("name")
    return (not sn.get("isConstructor")) and (name is None or name == "")


def _fn_name(sn: dict) -> str:
    """Stable display name for a function node."""
    if sn.get("isConstructor"):
        return "<constructor>"
    if _is_fallback(sn):
        return "<fallback>"
    return sn.get("name") or "<anon>"


def zero_ast_features() -> Dict[str, float]:
    """AST feature block set to zero (used when parsing fails: lexical fallback)."""
    return {k: 0 for k in AST_FEATURE_NAMES}


# ==============================================================================
# Heterogeneous graph construction
# ==============================================================================
def _member_name(call_node: dict) -> Optional[str]:
    """Best-effort callee name for a FunctionCall (Identifier or MemberAccess)."""
    expr = call_node.get("expression")
    if not isinstance(expr, dict):          # newer ASTs may store a scalar here
        return None
    t = expr.get("type")
    if t == "Identifier":
        return expr.get("name")
    if t == "MemberAccess":
        return expr.get("memberName")
    return None


def build_graph(contract_id: str, contracts: List[dict]) -> dict:
    """Build the per-contract heterogeneous graph (nodes + typed edges)."""
    nodes: List[dict] = []
    node_index: Dict[str, int] = {}  # unique key -> node id
    etype_id = {e: i for i, e in enumerate(EDGE_TYPES)}
    edges: List[Tuple[int, int, int]] = []

    def add_node(key: str, ntype: str, feats: Dict[str, float]) -> int:
        if key in node_index:
            return node_index[key]
        nid = len(nodes)
        node_index[key] = nid
        nodes.append({"id": nid, "ntype": ntype, "name": key, "feat": feats})
        return nid

    # First pass: declare all nodes so intra-file references resolve.
    contract_members: List[Tuple[dict, int]] = []
    for c in contracts:
        cname = c.get("name", "?")
        ckey = f"contract:{cname}"
        subs = c.get("subNodes", []) or []
        cfeat = {
            "is_library": 1 if c.get("kind") == "library" else 0,
            "is_interface": 1 if c.get("kind") == "interface" else 0,
            "n_bases": len(c.get("baseContracts", []) or []),
            "n_subnodes": len(subs),
        }
        cid = add_node(ckey, "contract", cfeat)
        contract_members.append((c, cid))
        for sn in subs:
            t = sn.get("type")
            if t == "FunctionDefinition":
                fname = _fn_name(sn)
                types = _count_types(sn.get("body")) if sn.get("body") else {}
                add_node(f"function:{cname}.{fname}", "function", {
                    "visibility": _VISIBILITY_CODE.get(sn.get("visibility") or "default", 0),
                    "is_payable": 1 if sn.get("stateMutability") == "payable" else 0,
                    "is_view_pure": 1 if sn.get("stateMutability") in ("view", "pure", "constant") else 0,
                    "is_constructor": 1 if sn.get("isConstructor") else 0,
                    "is_fallback": 1 if sn.get("isFallback") else 0,
                    "n_params": _param_count(sn.get("parameters")),
                    "n_calls": types.get("FunctionCall", 0),
                    "n_loops": types.get("ForStatement", 0) + types.get("WhileStatement", 0),
                })
            elif t == "ModifierDefinition":
                add_node(f"modifier:{cname}.{sn.get('name')}", "modifier",
                         {"n_params": _param_count(sn.get("parameters"))})
            elif t == "EventDefinition":
                add_node(f"event:{cname}.{sn.get('name')}", "event",
                         {"n_params": _param_count(sn.get("parameters"))})
            elif t == "StateVariableDeclaration":
                for v in sn.get("variables", []) or []:
                    tn = v.get("typeName")
                    tn = tn if isinstance(tn, dict) else {}
                    add_node(f"state_var:{cname}.{v.get('name')}", "state_var", {
                        "visibility": _VISIBILITY_CODE.get(v.get("visibility") or "default", 0),
                        "is_mapping": 1 if tn.get("type") == "Mapping" else 0,
                        "is_array": 1 if tn.get("type") == "ArrayTypeName" else 0,
                    })

    # Function-name -> node id, for resolving intra-file call edges by name.
    fn_by_name: Dict[str, int] = {}
    for key, nid in node_index.items():
        if key.startswith("function:"):
            fn_by_name.setdefault(key.split(".", 1)[-1], nid)

    # Second pass: edges.
    for c, cid in contract_members:
        cname = c.get("name", "?")
        # inherits
        for b in c.get("baseContracts", []) or []:
            bn = b.get("baseName") if isinstance(b, dict) else None
            bname = bn.get("namePath") if isinstance(bn, dict) else None
            if bname:
                bid = add_node(f"contract:{bname}", "contract",
                               {"is_library": 0, "is_interface": 0,
                                "n_bases": 0, "n_subnodes": 0})
                edges.append((cid, bid, etype_id["inherits"]))
        for sn in c.get("subNodes", []) or []:
            t = sn.get("type")
            if t == "FunctionDefinition":
                fname = _fn_name(sn)
                fkey = f"function:{cname}.{fname}"
                fid = node_index[fkey]
                edges.append((cid, fid, etype_id["contains"]))
                # uses modifier
                for mo in sn.get("modifiers", []) or []:
                    mkey = f"modifier:{cname}.{mo.get('name')}"
                    if mkey in node_index:
                        edges.append((fid, node_index[mkey], etype_id["uses"]))
                # body-derived edges
                body = sn.get("body")
                if body:
                    for nd in walk(body):
                        nt = nd.get("type")
                        if nt == "FunctionCall":
                            callee = _member_name(nd)
                            if callee and callee in fn_by_name:
                                edges.append((fid, fn_by_name[callee], etype_id["calls"]))
                        elif nt == "EmitStatement":
                            ec = nd.get("eventCall")
                            ev = ec.get("expression") if isinstance(ec, dict) else None
                            ev = ev if isinstance(ev, dict) else {}
                            ename = ev.get("name") or ev.get("memberName")
                            ekey = f"event:{cname}.{ename}"
                            if ename and ekey in node_index:
                                edges.append((fid, node_index[ekey], etype_id["emits"]))
                        elif nt == "Identifier":
                            svkey = f"state_var:{cname}.{nd.get('name')}"
                            if svkey in node_index:
                                edges.append((fid, node_index[svkey], etype_id["reads_writes"]))
            elif t == "ModifierDefinition":
                edges.append((cid, node_index[f"modifier:{cname}.{sn.get('name')}"],
                              etype_id["contains"]))
            elif t == "EventDefinition":
                edges.append((cid, node_index[f"event:{cname}.{sn.get('name')}"],
                              etype_id["contains"]))
            elif t == "StateVariableDeclaration":
                for v in sn.get("variables", []) or []:
                    edges.append((cid, node_index[f"state_var:{cname}.{v.get('name')}"],
                                  etype_id["contains"]))

    # De-duplicate edges deterministically.
    uniq_edges = sorted(set(edges))
    return {
        "contract_id": contract_id,
        "node_types": list(NODE_TYPES),
        "edge_types": list(EDGE_TYPES),
        "n_nodes": len(nodes),
        "n_edges": len(uniq_edges),
        "nodes": nodes,
        "edges": [list(e) for e in uniq_edges],
    }


# ==============================================================================
# Per-contract extraction
# ==============================================================================
def extract_one(contract_id: str, raw: str, pragma: str) -> Tuple[Dict[str, float], dict, str]:
    """Return (feature_row, graph, parse_method) for one contract source."""
    stripped = strip_comments_and_strings(raw)
    feats: Dict[str, float] = {}
    feats["f_n_bytes"] = len(raw.encode("utf-8", errors="ignore"))
    ls = line_stats(raw)
    feats["f_n_lines"] = ls["n_lines"]
    feats["f_n_code_lines"] = ls["n_code_lines"]
    feats["f_n_comment_lines"] = ls["n_comment_lines"]
    feats["f_comment_ratio"] = round(
        ls["n_comment_lines"] / ls["n_lines"], 4) if ls["n_lines"] else 0.0
    feats["f_max_line_len"] = ls["max_line_len"]
    feats["f_max_nesting_depth"] = max_brace_nesting(stripped)
    feats.update(pragma_features(pragma))
    feats.update(lexical_features(stripped))

    ast = parse_ast(raw)
    if ast is not None:
        astf, contracts = ast_features(ast)
        feats.update(astf)
        graph = build_graph(contract_id, contracts)
        parse_method = "ast"
    else:
        feats.update(zero_ast_features())
        graph = {"contract_id": contract_id, "node_types": list(NODE_TYPES),
                 "edge_types": list(EDGE_TYPES), "n_nodes": 0, "n_edges": 0,
                 "nodes": [], "edges": []}
        parse_method = "lexical_fallback"

    # Enforce the exact schema (no missing / no stray feature).
    missing = [k for k in FEATURE_NAMES if k not in feats]
    if missing:
        raise AssertionError(f"feature(s) not computed: {missing}")
    row = {k: feats[k] for k in FEATURE_NAMES}
    return row, graph, parse_method


# ==============================================================================
# Alert-free guards
# ==============================================================================
def assert_paths_alert_free(paths: List[str]) -> None:
    for p in paths:
        low = str(p).lower()
        for tok in FORBIDDEN_TOKENS:
            if tok in low:
                raise PermissionError(
                    f"alert-free guard: input path {p!r} references forbidden "
                    f"token {tok!r} (tool output must not enter feature extraction)")


def assert_registry_alert_free(df: pd.DataFrame) -> None:
    extra = [c for c in df.columns if c not in ALLOWED_REGISTRY_COLUMNS]
    if extra:
        raise PermissionError(
            f"alert-free guard: registry has non-source columns {extra}; refusing "
            f"to derive features from anything that could encode tool output")


def assert_features_alert_free() -> None:
    for name in FEATURE_NAMES:
        low = name.lower()
        for tok in FORBIDDEN_TOKENS:
            if tok in low:
                raise AssertionError(
                    f"feature {name!r} contains forbidden token {tok!r}")


# ==============================================================================
# Corpus extraction
# ==============================================================================
def extract_corpus(registry: pd.DataFrame, roots: Dict[str, Path],
                   limit: Optional[int] = None
                   ) -> Tuple[pd.DataFrame, List[dict], dict]:
    """Extract features + graphs for every contract in the registry."""
    assert_registry_alert_free(registry)
    assert_features_alert_free()

    rows: List[dict] = []
    graphs: List[dict] = []
    parse_methods: Dict[str, int] = {}
    df = registry if limit is None else registry.head(limit)

    for r in df.itertuples(index=False):
        root = roots.get(r.dataset)
        if root is None:
            continue
        src = (root / r.relpath).read_text(encoding="latin-1")
        feat_row, graph, method = extract_one(r.contract_id, src, r.pragma_solidity)
        parse_methods[method] = parse_methods.get(method, 0) + 1
        rows.append({"contract_id": r.contract_id, "dataset": r.dataset,
                     "base_id": r.base_id, "parse_method": method, **feat_row})
        graphs.append(graph)

    features = pd.DataFrame(rows, columns=(
        ["contract_id", "dataset", "base_id", "parse_method"] + list(FEATURE_NAMES)))
    features = features.sort_values("contract_id", kind="mergesort").reset_index(drop=True)

    diag = {
        "n_contracts": int(len(features)),
        "parse_methods": parse_methods,
        "graph_stats": {
            "mean_nodes": round(float(mean([g["n_nodes"] for g in graphs])), 2) if graphs else 0,
            "mean_edges": round(float(mean([g["n_edges"] for g in graphs])), 2) if graphs else 0,
            "max_nodes": max([g["n_nodes"] for g in graphs], default=0),
        },
    }
    return features, graphs, diag


def write_outputs(features: pd.DataFrame, graphs: List[dict],
                  out_dir: Path, diag: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    features.to_parquet(out_dir / "features.parquet", engine="pyarrow", index=False)
    with (out_dir / "graphs.jsonl").open("w", encoding="utf-8") as fh:
        for g in sorted(graphs, key=lambda x: x["contract_id"]):
            fh.write(json.dumps(g, separators=(",", ":")) + "\n")
    meta = {
        "schema_version": SCHEMA_VERSION,
        "generator": Path(__file__).name,
        "generator_version": __version__,
        "n_features": len(FEATURE_NAMES),
        "feature_names": list(FEATURE_NAMES),
        "node_types": list(NODE_TYPES),
        "edge_types": list(EDGE_TYPES),
        "alert_free_guarantee": {
            "inputs": "data/registry.parquet (+ .sol sources) only",
            "forbidden_tokens": list(FORBIDDEN_TOKENS),
            "note": ("No stage-02 output (labels/findings/timings) is read; every "
                     "feature is derivable before any analyzer runs. This makes "
                     "the prior circularity impossible by construction."),
        },
        "diagnostics": diag,
    }
    (out_dir / "features_meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=False), encoding="utf-8")


# ==============================================================================
# Hermetic self-test
# ==============================================================================
_SAMPLE_SOL = """\
pragma solidity ^0.5.0;

// a comment mentioning tx.origin that must NOT be counted
contract Base { }

contract Wallet is Base {
    mapping(address => uint) public balances;   // state var
    address owner;
    event Paid(address to, uint amt);

    modifier onlyOwner() { require(msg.sender == owner); _; }

    constructor() public { owner = msg.sender; }

    function () external payable { }             // fallback

    function withdraw(uint amt) public onlyOwner {
        require(amt <= balances[msg.sender]);
        if (amt > 0) {
            (bool ok, ) = msg.sender.call.value(amt)("");   // low-level call+value
            require(ok);
            balances[msg.sender] -= amt;                    // arithmetic + state write
            emit Paid(msg.sender, amt);
        }
        for (uint i = 0; i < amt; i++) { helper(); }        // loop + intra call
    }

    function helper() internal view returns (uint) {
        return block.timestamp + now;            // timestamp usage (x2)
    }
}
"""


def run_selftest() -> int:
    print(f"RELIANT 03_features self-test (v{__version__})")

    # --- comment/string stripping keeps tokens honest --------------------------
    stripped = strip_comments_and_strings(_SAMPLE_SOL)
    assert "a comment mentioning tx.origin" not in stripped
    # tx.origin appears ONLY in a comment -> must be zero after stripping.
    assert len(_LEX["f_n_tx_origin"].findall(stripped)) == 0

    # --- full extraction on the sample -----------------------------------------
    row, graph, method = extract_one("t1", _SAMPLE_SOL, "^0.5.0")
    assert method == "ast"
    assert set(row.keys()) == set(FEATURE_NAMES)

    # pragma parsed
    assert row["f_solc_major"] == 0 and row["f_solc_minor"] == 5 and row["f_pragma_caret"] == 1
    # structure
    assert row["f_n_contracts"] == 2 and row["f_n_state_vars"] == 2
    assert row["f_n_events"] == 1 and row["f_n_modifiers"] == 1
    assert row["f_n_inherit_edges"] == 1 and row["f_max_bases"] == 1
    assert row["f_has_fallback"] == 1 and row["f_n_constructor"] == 1
    # security tokens (from code, not the comment)
    assert row["f_n_low_level_call"] >= 1 and row["f_n_call_value"] >= 1
    assert row["f_n_msg_sender"] >= 3 and row["f_n_require"] >= 3
    assert row["f_n_timestamp"] == 2          # block.timestamp + now
    assert row["f_n_mapping"] == 1
    # control flow
    assert row["f_n_if"] >= 1 and row["f_n_loops"] >= 1 and row["f_n_calls"] >= 1
    assert row["f_cyclomatic_proxy"] >= 3

    # --- graph correctness ------------------------------------------------------
    ntypes = [n["ntype"] for n in graph["nodes"]]
    assert ntypes.count("contract") == 2          # Base + Wallet
    assert "function" in ntypes and "modifier" in ntypes and "event" in ntypes
    etypes = {EDGE_TYPES[e[2]] for e in graph["edges"]}
    assert "contains" in etypes and "inherits" in etypes
    assert "uses" in etypes                        # withdraw uses onlyOwner
    assert "emits" in etypes                       # withdraw emits Paid
    assert "calls" in etypes                       # withdraw -> helper
    assert "reads_writes" in etypes                # balances referenced
    # every edge index is valid
    for s, d, e in graph["edges"]:
        assert 0 <= s < graph["n_nodes"] and 0 <= d < graph["n_nodes"]
        assert 0 <= e < len(EDGE_TYPES)

    # --- alert-free guards ------------------------------------------------------
    assert_features_alert_free()  # our own feature names are clean
    ok_reg = pd.DataFrame(columns=list(ALLOWED_REGISTRY_COLUMNS))
    assert_registry_alert_free(ok_reg)
    poisoned = ok_reg.copy()
    poisoned["detected"] = []        # a tool-output column
    try:
        assert_registry_alert_free(poisoned)
        raise AssertionError("guard failed to reject a labels column")
    except PermissionError:
        pass
    try:
        assert_paths_alert_free(["artifacts/labels.parquet"])
        raise AssertionError("guard failed to reject a labels path")
    except PermissionError:
        pass
    # Analyzer-named / run-output paths are the clearest circularity signal:
    # the registry must never be pointed at tool output.
    for bad_path in ("data/slither_results.json", "sb_runs/mythril/task0",
                     "SolidiFI-benchmark-master/results/Oyente"):
        try:
            assert_paths_alert_free([bad_path])
            raise AssertionError(f"guard failed to reject {bad_path!r}")
        except PermissionError:
            pass
    # ...while the standard legitimate inputs stay accepted.
    assert_paths_alert_free(["data/registry.parquet", "artifacts",
                             "uploads/smartbugs-curated-main.zip"])

    # --- lexical fallback path (force a parse failure deterministically) -------
    # The ANTLR parser is lenient (it logs and returns a best-effort AST rather
    # than raising), so we force parse_ast to None to exercise the fallback that
    # protects against a missing/failing parser.
    saved_parse = globals()["parse_ast"]
    globals()["parse_ast"] = lambda _src: None
    try:
        row2, graph2, method2 = extract_one("bad", _SAMPLE_SOL, "^0.5.0")
    finally:
        globals()["parse_ast"] = saved_parse
    assert method2 == "lexical_fallback" and set(row2.keys()) == set(FEATURE_NAMES)
    assert graph2["n_nodes"] == 0
    # AST-derived features zeroed; lexical/size/pragma still computed.
    assert row2["f_n_contracts"] == 0 and row2["f_n_functions"] == 0
    assert row2["f_has_fallback"] == 1 and row2["f_n_low_level_call"] >= 1

    # --- determinism ------------------------------------------------------------
    r3, g3, _ = extract_one("t1", _SAMPLE_SOL, "^0.5.0")
    assert r3 == row and g3["edges"] == graph["edges"]

    print("  all invariants passed.")
    return 0


# ==============================================================================
# CLI
# ==============================================================================
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Alert-free feature extraction (stage 03).")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--extract", action="store_true", help="Extract features + graphs.")
    mode.add_argument("--selftest", action="store_true", help="Run the self-test.")
    p.add_argument("--registry", type=str, default="data/registry.parquet")
    p.add_argument("--uploads", type=str, default=None)
    p.add_argument("--solidifi", type=str, default=None)
    p.add_argument("--curated", type=str, default=None)
    p.add_argument("--out", type=str, default="artifacts")
    p.add_argument("--limit", type=int, default=None,
                   help="Extract only the first N contracts (for quick checks).")
    return p


def do_extract(args) -> int:
    stage01 = load_stage01()
    reg_path = Path(args.registry)
    assert_paths_alert_free([str(reg_path), str(args.out)])
    if not reg_path.exists():
        sys.stderr.write(f"error: registry not found: {reg_path}\n")
        return 3
    registry = pd.read_parquet(reg_path, engine="pyarrow")

    work = Path(args.out) / "_work"
    solidifi = args.solidifi or (Path(args.uploads) / "SolidiFI-benchmark-master.zip"
                                 if args.uploads else None)
    curated = args.curated or (Path(args.uploads) / "smartbugs-curated-main.zip"
                               if args.uploads else None)
    if solidifi is None or curated is None:
        sys.stderr.write("error: provide --uploads or --solidifi/--curated.\n")
        return 3
    roots = {
        "solidifi": stage01.resolve_dataset_dir(Path(solidifi), "buggy_contracts", work, "solidifi"),
        "sb_curated": stage01.resolve_dataset_dir(Path(curated), "vulnerabilities.json", work, "sb_curated"),
    }

    print(f"Extracting alert-free features for "
          f"{len(registry) if args.limit is None else min(args.limit, len(registry))} contracts ...")
    features, graphs, diag = extract_corpus(registry, roots, args.limit)
    write_outputs(features, graphs, Path(args.out), diag)

    print(f"\nWrote {args.out}/features.parquet  "
          f"({diag['n_contracts']} contracts x {len(FEATURE_NAMES)} features)")
    print(f"  parse methods : {diag['parse_methods']}")
    print(f"  graph (mean)  : {diag['graph_stats']['mean_nodes']} nodes, "
          f"{diag['graph_stats']['mean_edges']} edges")
    print(f"  graphs        : {args.out}/graphs.jsonl")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.selftest or not args.extract:
        return run_selftest()
    return do_extract(args)


if __name__ == "__main__":
    raise SystemExit(main())
