# RELIANT

**Analyzers as Imperfect Inspectors: Certified Reliability Prediction and Cost-Bounded Redundant Panel Selection for Smart Contract Security**

Anonymous artifact accompanying a submission to *IEEE Transactions on Reliability*.

---

## What this is

Smart contract analyzers are imperfect inspectors: their error rates depend on the
contract under analysis, and their misses are positively correlated. A panel of
them is therefore a *k*-out-of-*n* redundant inspection system, not a set of
independent trials.

RELIANT predicts per-(contract, analyzer, class) reliability from **alert-free**
features, wraps each prediction in a **distribution-free conformal certificate**,
and selects a **cost-bounded panel** whose joint detection guarantee is computed
with a one-factor Gaussian copula rather than a false independence assumption.

On 493 contracts, 7 version-pinned analyzers and 7 vulnerability classes:

| Result | Value |
|---|---|
| Full-panel ("run-all") detection / cost | 0.9190 / 283.2 s per contract |
| RELIANT at 50 % budget | **0.8534 detection for 107.6 s** — 93 % of the detection for 38 % of the cost |
| Conformal coverage (marginal, α = 0.10) | 0.8956, inside the theoretical band |
| Per-class calibration, worst class | 0.8724 → **0.8929**, mean interval width 0.7739 → **0.6811** |
| Pooled conditional miss correlation ρ | 0.1399 (per class: 0.0000 – 0.5717) |

We also report where the method does **not** win. A per-class prior matches the
learned predictor (−0.0018, interval contains zero), and certificate coverage
falls to 0.6858 under synthetic-to-real distribution shift. See
[Reported negative results](#reported-negative-results).

---

## Repository layout

```
.
├── config.yaml              # single source of truth for every stage setting
├── requirements.txt
├── run_all.sh               # stages 01 → 10 end to end
├── src/
│   ├── 01_download_data.py  # corpus registry, SHA-256 manifest, base-id grouping
│   ├── 02_ground_truth.py   # analyzer runs → findings → SWC/DASP → labels
│   ├── 03_features.py       # 70 alert-free features + heterogeneous graph
│   ├── 04_models.py         # per-analyzer empirical performance models
│   ├── 05_conformal.py      # split / Mondrian / one-sided FNR certificates
│   ├── 06_portfolio.py      # copula joint-miss bound + budget-constrained selection
│   ├── 07_train.py          # leakage-safe grouped CV, out-of-fold predictions
│   ├── 08_evaluate.py       # RQ1–RQ5
│   ├── 09_baselines.py      # SATzilla predictor, selection oracles
│   └── 10_make_figures.py   # LaTeX tables + figures
├── data/                    # stage 01 output (registry, manifest)
├── artifacts/               # features, labels, predictions  (git-ignored, large)
├── results/                 # RQ JSON
├── tables/                  # LaTeX tables
└── figures/                 # PDFs
```

Every stage is a standalone script with a `--selftest` mode and no hidden state.

---

## Requirements

Python **3.12** (developed and validated on 3.12.3). Install:

```bash
pip install -r requirements.txt
# in a restricted sandbox:  pip install --break-system-packages -r requirements.txt
```

Core dependencies: `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`,
`pyarrow`, `lightgbm`, `PyYAML`, `solidity-parser`.

`torch` is **optional**. The heterogeneous-GNN predictor in stage 04 runs only if
PyTorch is importable; every other stage runs, and the GNN is skipped with a
notice, when it is absent. **No result reported in the paper uses the GNN.**

**Docker** is required only for stage 02 `--run-tools`, which executes the seven
analyzers under SmartBugs 2.0. Everything downstream runs on a single CPU core
with peak memory below 4 GB.

### Input archives

Place these four archives in `uploads/`:

| Archive | Used for |
|---|---|
| `SolidiFI-benchmark-master.zip` | 350 injected-bug contracts (primary ground truth) |
| `smartbugs-curated-main.zip` | 143 real annotated contracts (cross-benchmark test) |
| `smartbugs-master.zip` | analyzer `findings.yaml` classification maps |
| `DeFiHackLabs-main.zip` | RQ5 deployment profile (optional) |

---

## Quick start

Verify the installation without any data — all ten stages self-test hermetically
in well under a minute:

```bash
for f in src/0*.py src/10_*.py; do
  printf '%-24s ' "$(basename "$f")"
  python3 "$f" --selftest >/dev/null 2>&1 && echo PASS || echo FAIL
done
```

Build the corpus registry and the alert-free features (no Docker needed):

```bash
python3 src/01_download_data.py --uploads uploads --out data
python3 src/03_features.py --extract --registry data/registry.parquet \
        --uploads uploads --out artifacts
```

Expected stage 01 output:

```
Wrote data/registry.parquet  (493 contracts)
  by dataset : {'sb_curated': 143, 'solidifi': 350}
  base ids   : 193  |  with ground truth: 493
  corpus fingerprint : 53b04aed239d40a6...
```

Expected stage 03 output:

```
Wrote artifacts/features.parquet  (493 contracts x 70 features)
  parse methods : {'ast': 493}
```

All 493 contracts parse via AST with no regular-expression fallback.

Then run everything:

```bash
bash run_all.sh
```

`run_all.sh` reuses `artifacts/labels.parquet` if present. To generate labels
from a SmartBugs results directory:

```bash
SB_RUNS=/path/to/smartbugs/results TOOLS_DIR=/path/to/smartbugs/tools bash run_all.sh
```

---

## Pipeline

| Stage | Input | Output | Notes |
|---|---|---|---|
| **01** `download_data` | four archives | `registry.parquet`, `manifest.sha256` | class from the SolidiFI *directory name*, not the UTF-7-mangled BugLog column |
| **02** `ground_truth` | SmartBugs runs | `labels.parquet` | findings → `findings.yaml` → SWC/DASP → canonical class |
| **03** `features` | `.sol` sources | `features.parquet`, `graphs.jsonl` | **alert-free**; runtime guard rejects analyzer-derived columns |
| **04** `models` | features + labels | fitted predictors | one sub-model per analyzer, input `[φ(C) ⊕ onehot(k)]` |
| **05** `conformal` | calibration split | thresholds | split, Mondrian-by-class, one-sided FNR upper bound |
| **06** `portfolio` | certified FNRs, costs | panel `S*` | one-factor Gaussian copula, exhaustive over 2⁷ subsets |
| **07** `train` | all of the above | `predictions.parquet` | grouped 5-fold CV × 3 seeds, keyed on `base_id` |
| **08** `evaluate` | predictions | `results/rq*.json` | RQ1–RQ5 |
| **09** `baselines` | predictions | `baselines.json` | SATzilla-RF, oracles |
| **10** `make_figures` | results | LaTeX + PDFs | Type-42 embedded fonts |

### Canonical settings (`config.yaml`)

```yaml
labels:     {target: detected, line_tolerance: 0}
conformal:  {alpha: 0.10}                     # target coverage 0.90
training:   {folds: 5, seeds: 3}
portfolio:  {budget_fractions: [0.25, 0.50, 0.75], guarantee_target: 0.80}
case_study: {sample: 25, query_class: reentrancy, budget_fraction: 0.5}
```

---

## Design decisions that affect the numbers

These are stated here because they change how the results should be read.

**Operational reliability.** A run that crashes, times out, or fails to compile
scores as a miss (`detected = 0`). This is what a practitioner experiences — the
budget is spent and nothing comes back — but it folds *capability* and
*robustness* together. Oyente fails on 19.3 % of runs and Securify2 on 11.8 %,
so their measured rates are not pure capability measures.

**Contract-level detection.** `detected` is ≥ 1 relevant finding *anywhere* in
the contract, not necessarily at the injected line. Line-level matching at zero
tolerance is computed as an alternative target and is the recommended sensitivity
analysis; the contract-level definition is retained because it is defined for
analyzers that report at bytecode granularity, which line matching is not.

**Leakage.** SolidiFI injects seven classes into a common set of base contracts,
so the seven variants of a base are near-duplicates. Every split is grouped on
`base_id` and the invariant is asserted at runtime on every fold.

**Slither exit codes.** Slither 0.11.3 exits 255 whenever it reports findings.
Run status is derived from the output parser, not the process exit code;
otherwise every successful Slither analysis would be discarded as an error.

**Semgrep exclusion.** `semgrep-c3a9f40` is excluded from the panel. In the
pinned snapshot its rule set classifies into **2 of our 7 classes** (8 reentrancy
rules under SWC-107/DASP-1, 3 arithmetic rules under SWC-101/DASP-3) and its
rules target protocol-level DeFi patterns rather than the code-level fault shapes
in this corpus.
⚠️ Source comments in `02_ground_truth.py` and `config.yaml` still describe this
as "0/7 coverage"; that wording is wrong and is corrected in the paper. Fix the
comments before release.

---

## Reported negative results

The paper reports every claim it tested, including those the evidence does not
support. The same applies here.

| Claim | Verdict |
|---|---|
| Reliability predictable from alert-free features | supported (AUC 0.786 vs 0.500 chance) |
| … beyond a per-class prior | **not supported** (−0.0018, interval contains zero) |
| … better than SATzilla-RF, or better calibrated | **not supported** (−0.0070; Brier 0.1825 vs 0.1475) |
| Certificates attain nominal coverage | supported |
| Per-class calibration repairs under-coverage and narrows intervals | supported |
| Cheaper than exhaustive analysis at comparable detection | supported (93 % of detection for 38 % of cost) |
| Higher detection than the best single analyzer | supported (0.853 vs 0.586) |
| … and cheaper than it | **not supported** (107.6 s vs 58.0 s) |
| Coverage survives distribution shift | **not supported** (0.686 vs 0.900 target) |

---

## Known gaps in this artifact

Stated plainly so reviewers do not have to discover them.

1. **`class_prior` predictor is not in `04_models.py`.** The per-class base-rate
   baseline that drives the headline RQ1 verdict is reported in the paper but is
   not registered in `build_predictor`. It must be added for the RQ1 comparison
   to be reproducible from this tree.
2. **The cluster bootstrap is not implemented.** Every confidence interval in the
   paper cites a cluster bootstrap over `base_id` with 500 resamples; no
   bootstrap or resampling code exists in `src/`. Two of the four paired
   differences in Fig. 4(b) are consequently drawn without intervals.
3. **`_random_k_detection` is not budget-matched.** It samples a uniform random
   *k*-subset and never filters by cost; its realized mean cost (201.7 s) exceeds
   the 50 % cap (141.6 s) by 42 %. The paper reports it as *unconstrained* and
   flags a budget-feasible variant as future work.
4. **RQ5 analyzes exploit harnesses, not victim contracts.** DeFiHackLabs ships
   Foundry proof-of-concept files that fork mainnet state and reach the vulnerable
   contract through an interface declaration; the vulnerable code is on chain and
   absent from the repository. RQ5 is therefore a *deployment-cost profile*, not
   a detection measurement, and the paper says so.
5. **The per-fold calibration size `n_c` is not recorded** in the results, so the
   conformal band in Fig. 5(a) is drawn for a stated assumed value.
6. **Per-class ρ is not consumed by the selector.** Selection uses the pooled
   conditional ρ = 0.1399 while the per-class range spans 0.0000 – 0.5717, and
   two classes rest on fewer than half the available analyzer pairs.

---

## Reproducing the paper's figures

Figures are generated from the reported results, not from the raw pipeline, so
they can be rebuilt without Docker.

```bash
python3 make_figs.py          # Fig. 1, 4, 5, 6, 7, 8  (matplotlib, Type-42 fonts)
pdflatex fig2.tex             # Fig. 2  architecture   (TikZ)
pdflatex fig3.tex             # Fig. 3  redundancy model (TikZ + pgfplots)
```

`fig2_spec.json` carries a node-by-node and edge-by-edge provenance map for the
architecture diagram. `fig3_curve.dat` holds the 31 points of the copula bound,
produced by evaluating the joint-miss equation with Gauss–Hermite quadrature on
the measured false-negative rates of a four-analyzer panel; at ρ = 0 it returns
0.925351, matching the independence product exactly.

---

## Runtime

| Step | Cost |
|---|---|
| Ten self-tests | < 1 min, no data |
| Stage 01 registry | ~1 min |
| Stage 03 features (493 contracts) | ~3 min |
| Stages 04–10 | ~10 min, single core |
| Stage 02 label generation | **~39 h single-threaded** (283 s × 493 contracts), Docker required |

Label generation dominates. Ship or reuse `artifacts/labels.parquet` wherever
possible; `run_all.sh` reuses it automatically.

---

## Determinism

Seeds are fixed, splits are grouped and asserted, and the corpus is pinned by a
SHA-256 content manifest over every source file. The corpus fingerprint is
recorded in `registry_meta.json` and re-checked in `train_meta.json`, so a run
against a different corpus fails loudly rather than silently.

---

## Licence

MIT. See `LICENSE`.

The bundled corpora keep their own licences: SolidiFI, SmartBugs and
SmartBugs-curated, and DeFiHackLabs are redistributed under their respective
upstream terms and are not covered by the licence above.
