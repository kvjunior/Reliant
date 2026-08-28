#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_conformal.py -- Calibrated reliability certificates (distribution-free coverage).

RELIANT: Calibrated Reliability Prediction and Budget-Constrained Portfolio
Selection for Smart Contract Security Analyzers.

--------------------------------------------------------------------------------
ROLE IN THE ARTIFACT -- AND WHY IT FIXES "WHY IS THE NUMBER GOOD?"
--------------------------------------------------------------------------------
The prior submission reported a point error (MAE) with no context for whether it
was good, and offered no notion of guarantee. This stage replaces a bare point
estimate of analyzer reliability with a *certified interval*: given a calibration
sample, it wraps any predictor's score in a bound that provably covers the true
reliability with probability at least 1 - alpha, with NO distributional
assumptions beyond exchangeability (split-conformal prediction; Vovk et al. 2005;
Lei et al. 2018). For the portfolio (stage 06) it emits a one-sided bound -- a
certified UPPER bound on a tool's miss rate (equivalently a LOWER bound on its
detection reliability) -- so the panel's detection guarantee is conservative by
construction rather than by hope.

Three calibrators, increasing in adaptivity, share one small numpy core:
  * SplitConformal    marginal coverage >= 1 - alpha (two-sided or one-sided).
  * MondrianConformal group-conditional: coverage >= 1 - alpha WITHIN each group
                      (e.g. per vulnerability class), which marginal coverage does
                      not guarantee -- a class the model handles poorly cannot be
                      silently under-covered.
  * CQR               conformalized quantile regression: intervals whose width
                      adapts to local difficulty, calibrated to keep the coverage
                      guarantee (Romano et al. 2019).

--------------------------------------------------------------------------------
GUARANTEE (finite-sample, distribution-free)
--------------------------------------------------------------------------------
With n calibration scores and a nonconformity score s(x, y), the conformal
threshold is the k-th smallest score, k = ceil((n + 1)(1 - alpha)). Then for an
exchangeable test point, P(s(X, Y) <= threshold) >= 1 - alpha, and coverage is
also upper-bounded by 1 - alpha + 1/(n + 1). Everything below is a specialization
of this fact to two-sided intervals, one-sided reliability bounds, per-group
thresholds (Mondrian), and quantile-regression residuals (CQR).

The target is a reliability value in [0, 1] (a detection probability, or recall /
1 - FNR from stage 02); bounds are clipped to [0, 1]. Calibrators are model-
agnostic: they consume (label, prediction) arrays, so they wrap any stage-04
ReliabilityPredictor without being coupled to it.

--------------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------------
    python3 src/05_conformal.py --selftest     # verifies empirical coverage

Exit codes: 0 = success; 3 = usage error.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__version__ = "1.1.0"
SCHEMA_VERSION = "reliant-conformal-1"

# Reliability lives in [0, 1]; all emitted bounds are clipped to this range.
# Clipping is coverage-safe precisely BECAUSE the target is in [0, 1]: shrinking
# lo up to 0 or hi down to 1 can never exclude a y that lies inside [0, 1].
_LO, _HI = 0.0, 1.0

# Nonconformity modes:
#   "two_sided" -> interval [pred - q, pred + q]           (covers y)
#   "upper"     -> upper bound  U = pred + q, P(y <= U) >= 1 - alpha
#   "lower"     -> lower bound  L = pred - q, P(y >= L) >= 1 - alpha
MODES = ("two_sided", "upper", "lower")


# ==============================================================================
# Conformal core (pure numpy)
# ==============================================================================
def conformal_quantile(scores: Sequence[float], alpha: float) -> float:
    """Finite-sample conformal threshold: the ceil((n+1)(1-alpha))-th smallest score.

    Returns +inf when the calibration set is too small to certify at this alpha
    (which yields a trivially valid, infinitely wide interval rather than a false
    guarantee). This is exactly the quantity giving P(score <= threshold) >= 1-a.

    Non-finite scores (NaN from an unobserved label, +-inf) are DROPPED before the
    order statistic is taken, and n is the count of finite scores. This is not
    cosmetic: numpy sorts NaN to the END of the array, so a calibration set with
    even a few NaNs would place the k-th smallest ON a NaN, making the threshold
    NaN and every emitted bound NaN -- a silent, total loss of coverage rather
    than a visible error. Dropping them instead calibrates on the observed
    exchangeable subsample, which is the intended semantics.
    """
    s = np.asarray(scores, dtype=float).ravel()
    s = s[np.isfinite(s)]
    s = np.sort(s)
    n = s.size
    if n == 0:
        return np.inf
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return np.inf
    return float(s[k - 1])


def _scores(y: np.ndarray, pred: np.ndarray, mode: str) -> np.ndarray:
    """Nonconformity scores for the given mode."""
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    if mode == "two_sided":
        return np.abs(y - pred)
    if mode == "upper":          # want P(y <= U): penalize y above pred
        return y - pred
    if mode == "lower":          # want P(y >= L): penalize y below pred
        return pred - y
    raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")


def _apply(pred: np.ndarray, q: float, mode: str) -> Tuple[np.ndarray, np.ndarray]:
    """Turn a threshold q into (lo, hi) bounds for the given mode, clipped to [0,1]."""
    pred = np.asarray(pred, dtype=float)
    if mode == "two_sided":
        lo, hi = pred - q, pred + q
    elif mode == "upper":
        lo, hi = np.full_like(pred, _LO), pred + q
    else:  # lower
        lo, hi = pred - q, np.full_like(pred, _HI)
    return np.clip(lo, _LO, _HI), np.clip(hi, _LO, _HI)


# ==============================================================================
# Split conformal (marginal coverage)
# ==============================================================================
@dataclass
class SplitConformal:
    """Marginal split-conformal calibrator: coverage >= 1 - alpha overall."""
    alpha: float = 0.1
    mode: str = "two_sided"
    q_: Optional[float] = None
    n_cal_: int = 0          # finite calibration scores actually used

    def calibrate(self, y_cal: np.ndarray, pred_cal: np.ndarray) -> "SplitConformal":
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        s = _scores(y_cal, pred_cal, self.mode)
        self.n_cal_ = int(np.isfinite(s).sum())
        self.q_ = conformal_quantile(s, self.alpha)
        return self

    def interval(self, pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.q_ is None:
            raise RuntimeError("call calibrate() before interval()")
        return _apply(pred, self.q_, self.mode)

    def bound(self, pred: np.ndarray) -> np.ndarray:
        """The certified one-sided bound (upper for mode='upper', lower for 'lower').

        Deliberately refuses two_sided: returning one end of a two-sided interval
        would look like a one-sided certificate while carrying only the weaker
        two-sided guarantee (P(y <= hi) is not controlled at 1 - alpha there).
        Callers that want an end point should use interval() explicitly.
        """
        if self.mode == "two_sided":
            raise ValueError(
                "bound() is only defined for one-sided modes; a two_sided "
                "interval's endpoint does not carry a one-sided guarantee. "
                "Use interval(), or calibrate with mode='upper'/'lower'.")
        lo, hi = self.interval(pred)
        return hi if self.mode == "upper" else lo


# ==============================================================================
# Mondrian (group-conditional) conformal
# ==============================================================================
@dataclass
class MondrianConformal:
    """Group-conditional conformal: coverage >= 1 - alpha within each group.

    A separate threshold is fit per group key (e.g. vulnerability class or tool),
    so a group the model predicts poorly cannot be masked by easy groups -- the
    property marginal split-conformal lacks. Groups with too few calibration
    points fall back to the pooled threshold (recorded in `fallback_groups_`).
    """
    alpha: float = 0.1
    mode: str = "two_sided"
    min_per_group: int = 20
    q_by_group_: Optional[Dict[object, float]] = None
    q_global_: Optional[float] = None
    fallback_groups_: Tuple[object, ...] = ()
    n_by_group_: Optional[Dict[object, int]] = None
    unseen_groups_: Tuple[object, ...] = ()

    def calibrate(self, y_cal: np.ndarray, pred_cal: np.ndarray,
                  groups_cal: Sequence[object]) -> "MondrianConformal":
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        y_cal = np.asarray(y_cal, dtype=float)
        pred_cal = np.asarray(pred_cal, dtype=float)
        groups_cal = np.asarray(groups_cal, dtype=object)
        if not (y_cal.size == pred_cal.size == groups_cal.size):
            raise ValueError("y_cal, pred_cal and groups_cal must have equal length")
        all_scores = _scores(y_cal, pred_cal, self.mode)
        self.q_global_ = conformal_quantile(all_scores, self.alpha)
        self.q_by_group_ = {}
        self.n_by_group_ = {}
        fallbacks: List[object] = []
        for g in np.unique(groups_cal):
            m = groups_cal == g
            # Count only usable (finite-score) points, so a group padded with
            # unobserved labels cannot pass min_per_group on paper and then
            # calibrate on far fewer effective points.
            n_eff = int(np.isfinite(all_scores[m]).sum())
            self.n_by_group_[g] = n_eff
            if n_eff >= self.min_per_group:
                self.q_by_group_[g] = conformal_quantile(all_scores[m], self.alpha)
            else:
                self.q_by_group_[g] = self.q_global_
                fallbacks.append(g)
        self.fallback_groups_ = tuple(fallbacks)
        self.unseen_groups_ = ()
        return self

    def interval(self, pred: np.ndarray,
                 groups: Sequence[object]) -> Tuple[np.ndarray, np.ndarray]:
        """Per-group bounds. Vectorized: one numpy pass per distinct group.

        Groups never seen during calibration fall back to the pooled threshold
        (still valid marginally) and are recorded in `unseen_groups_` so a silent
        fallback is auditable rather than invisible.
        """
        if self.q_by_group_ is None:
            raise RuntimeError("call calibrate() before interval()")
        pred = np.asarray(pred, dtype=float)
        groups = np.asarray(groups, dtype=object)
        if pred.size != groups.size:
            raise ValueError("pred and groups must have equal length")
        lo = np.empty_like(pred)
        hi = np.empty_like(pred)
        unseen: List[object] = []
        for g in np.unique(groups):
            m = groups == g
            if g in self.q_by_group_:
                q = self.q_by_group_[g]
            else:
                q = self.q_global_
                unseen.append(g)
            lo[m], hi[m] = _apply(pred[m], q, self.mode)
        if unseen:
            self.unseen_groups_ = tuple(unseen)
        return lo, hi


# ==============================================================================
# Conformalized Quantile Regression (adaptive width)
# ==============================================================================
@dataclass
class CQR:
    """Conformalize predicted conditional quantiles for adaptive-width intervals.

    Given lower/upper quantile predictions q_lo(x), q_hi(x) (e.g. from gradient-
    boosted quantile regressors), the conformity score is
        E = max(q_lo - y, y - q_hi),
    and the (1-alpha) conformal quantile of E on calibration widens the band to
    restore coverage >= 1 - alpha (Romano, Patterson & Candes 2019). This is
    model-agnostic: it consumes quantile predictions, not a specific model.
    """
    alpha: float = 0.1
    q_: Optional[float] = None
    n_cal_: int = 0

    def calibrate(self, y_cal: np.ndarray, qlo_cal: np.ndarray,
                  qhi_cal: np.ndarray) -> "CQR":
        y_cal = np.asarray(y_cal, dtype=float)
        qlo_cal = np.asarray(qlo_cal, dtype=float)
        qhi_cal = np.asarray(qhi_cal, dtype=float)
        if not (y_cal.size == qlo_cal.size == qhi_cal.size):
            raise ValueError("y_cal, qlo_cal and qhi_cal must have equal length")
        # Romano et al. (2019), Eq. (6): E_i = max{q_lo(X_i) - Y_i, Y_i - q_hi(X_i)}
        E = np.maximum(qlo_cal - y_cal, y_cal - qhi_cal)
        self.n_cal_ = int(np.isfinite(E).sum())
        self.q_ = conformal_quantile(E, self.alpha)
        return self

    def interval(self, qlo: np.ndarray, qhi: np.ndarray
                 ) -> Tuple[np.ndarray, np.ndarray]:
        """Romano et al. (2019), Eq. (7): [q_lo - Q_{1-alpha}(E), q_hi + Q_{1-alpha}(E)]."""
        if self.q_ is None:
            raise RuntimeError("call calibrate() before interval()")
        qlo = np.asarray(qlo, dtype=float)
        qhi = np.asarray(qhi, dtype=float)
        return np.clip(qlo - self.q_, _LO, _HI), np.clip(qhi + self.q_, _LO, _HI)


def fit_quantile_models(X: np.ndarray, y: np.ndarray, alpha: float = 0.1,
                        params: Optional[dict] = None, seed: int = 0):
    """Train LGBM lower/upper quantile regressors for CQR (lazy LightGBM import).

    Returns (model_lo, model_hi) predicting the alpha/2 and 1 - alpha/2 quantiles.
    Kept separate from the conformal math so the calibrators stay dependency-free
    and testable; only this helper needs LightGBM.
    """
    import lightgbm as lgb
    base = dict(n_estimators=300, num_leaves=15, learning_rate=0.05,
                min_child_samples=5, random_state=seed, n_jobs=1, verbosity=-1,
                deterministic=True, force_row_wise=True)
    if params:
        base.update(params)
    lo = lgb.LGBMRegressor(objective="quantile", alpha=alpha / 2, **base).fit(X, y)
    hi = lgb.LGBMRegressor(objective="quantile", alpha=1 - alpha / 2, **base).fit(X, y)
    return lo, hi


# ==============================================================================
# Coverage / width diagnostics
# ==============================================================================
def coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """Empirical fraction of y within [lo, hi] (non-NaN entries only)."""
    y = np.asarray(y, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    m = ~np.isnan(y)
    if m.sum() == 0:
        return float("nan")
    inside = (y[m] >= lo[m] - 1e-12) & (y[m] <= hi[m] + 1e-12)
    return float(inside.mean())


def one_sided_coverage(y: np.ndarray, bound: np.ndarray, mode: str) -> float:
    """P(y <= bound) for mode='upper', or P(y >= bound) for mode='lower'."""
    y = np.asarray(y, dtype=float)
    bound = np.asarray(bound, dtype=float)
    m = ~np.isnan(y)
    if m.sum() == 0:
        return float("nan")
    if mode == "upper":
        return float((y[m] <= bound[m] + 1e-12).mean())
    return float((y[m] >= bound[m] - 1e-12).mean())


def mean_width(lo: np.ndarray, hi: np.ndarray) -> float:
    return float(np.mean(np.asarray(hi, float) - np.asarray(lo, float)))


def coverage_by_group(y: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                      groups: Sequence[object]) -> Dict[object, float]:
    groups = np.asarray(groups, dtype=object)
    out: Dict[object, float] = {}
    for g in np.unique(groups):
        m = groups == g
        out[g] = coverage(np.asarray(y)[m], np.asarray(lo)[m], np.asarray(hi)[m])
    return out


# ==============================================================================
# Hermetic self-test (empirical coverage is the whole point)
# ==============================================================================
def _heteroscedastic(n: int, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two groups with very different noise; pred = true mean (residual = noise)."""
    rng = np.random.default_rng(seed)
    g = rng.integers(0, 2, size=n)
    mu = np.where(g == 0, 0.5, 0.5)
    sigma = np.where(g == 0, 0.02, 0.30)          # group 1 is much noisier
    y = np.clip(mu + rng.normal(0, 1, n) * sigma, 0, 1)
    pred = mu.astype(float)                        # unbiased point predictor
    return y, pred, g


def run_selftest() -> int:
    print(f"RELIANT 05_conformal self-test (v{__version__})")
    alpha = 0.1
    target = 1 - alpha

    # --- conformal_quantile finite-sample behaviour ----------------------------
    assert np.isinf(conformal_quantile([], alpha))
    # too few points for the level -> +inf (trivially valid, not a false claim)
    assert np.isinf(conformal_quantile([0.3], alpha))
    q = conformal_quantile(np.linspace(0, 1, 100), alpha)
    assert 0.85 <= q <= 0.95, f"unexpected quantile {q}"

    # --- (1) split conformal: marginal coverage ~ target -----------------------
    yc, pc, gc = _heteroscedastic(4000, 1)
    yt, pt, gt = _heteroscedastic(8000, 2)
    sc = SplitConformal(alpha=alpha, mode="two_sided").calibrate(yc, pc)
    lo, hi = sc.interval(pt)
    cov = coverage(yt, lo, hi)
    print(f"  split marginal coverage = {cov:.3f} (target {target:.2f})")
    assert cov >= target - 0.02, f"split under-covers marginally: {cov:.3f}"

    # split conformal UNDER-covers the noisy group (motivates Mondrian)
    cov_split_g = coverage_by_group(yt, lo, hi, gt)
    print(f"  split coverage by group = "
          f"{{0: {cov_split_g[0]:.3f}, 1: {cov_split_g[1]:.3f}}}")
    assert cov_split_g[1] < target - 0.05, "expected split to under-cover noisy group"

    # --- (2) Mondrian: coverage within EACH group ------------------------------
    mc = MondrianConformal(alpha=alpha, mode="two_sided", min_per_group=20)
    mc.calibrate(yc, pc, gc)
    mlo, mhi = mc.interval(pt, gt)
    cov_mondrian_g = coverage_by_group(yt, mlo, mhi, gt)
    print(f"  Mondrian coverage by group = "
          f"{{0: {cov_mondrian_g[0]:.3f}, 1: {cov_mondrian_g[1]:.3f}}}")
    assert min(cov_mondrian_g.values()) >= target - 0.03, \
        "Mondrian must cover every group"
    # and it does so by widening the interval only where needed
    w0 = mean_width(mlo[gt == 0], mhi[gt == 0])
    w1 = mean_width(mlo[gt == 1], mhi[gt == 1])
    assert w1 > w0, "Mondrian should widen the noisy group's interval"
    print(f"  Mondrian mean width: easy={w0:.3f} noisy={w1:.3f} (adaptive)")

    # --- (3) one-sided reliability bounds hold at the guarantee ----------------
    up = SplitConformal(alpha=alpha, mode="upper").calibrate(yc, pc)
    U = up.bound(pt)
    cu = one_sided_coverage(yt, U, "upper")
    lw = SplitConformal(alpha=alpha, mode="lower").calibrate(yc, pc)
    L = lw.bound(pt)
    cl = one_sided_coverage(yt, L, "lower")
    print(f"  one-sided: P(y<=U)={cu:.3f}  P(y>=L)={cl:.3f} (target {target:.2f})")
    assert cu >= target - 0.02 and cl >= target - 0.02
    assert np.all(U <= 1.0 + 1e-9) and np.all(L >= -1e-9), "bounds must lie in [0,1]"

    # --- (4) CQR: coverage restored, width adapts across groups ----------------
    # Synthetic conditional quantiles: q_lo/q_hi = mean -/+ 1.0*sigma_hat, where a
    # single global sigma_hat mis-specifies the heteroscedastic truth; CQR fixes it.
    ycq, pcq, gcq = _heteroscedastic(5000, 3)
    ytq, ptq, gtq = _heteroscedastic(8000, 4)
    s_hat = 0.10
    qlo_c, qhi_c = pcq - s_hat, pcq + s_hat
    qlo_t, qhi_t = ptq - s_hat, ptq + s_hat
    cqr = CQR(alpha=alpha).calibrate(ycq, qlo_c, qhi_c)
    clo, chi = cqr.interval(qlo_t, qhi_t)
    cov_cqr = coverage(ytq, clo, chi)
    print(f"  CQR marginal coverage = {cov_cqr:.3f} (target {target:.2f})")
    assert cov_cqr >= target - 0.02, "CQR must restore coverage"

    # --- (5) exact coverage guarantee on a clean exchangeable stream -----------
    # Uniform residuals -> coverage lies in [1-alpha, 1-alpha + 1/(n+1)].
    rng = np.random.default_rng(9)
    covs = []
    for rep in range(200):
        ycal = rng.uniform(0, 1, 500)
        pcal = np.full(500, 0.5)
        ytst = rng.uniform(0, 1, 2000)
        ptst = np.full(2000, 0.5)
        cc = SplitConformal(alpha=alpha, mode="two_sided").calibrate(ycal, pcal)
        l, h = cc.interval(ptst)
        covs.append(coverage(ytst, l, h))
    mean_cov = float(np.mean(covs))
    print(f"  average coverage over 200 calibrations = {mean_cov:.3f} "
          f"(should be >= {target:.2f})")
    assert mean_cov >= target - 0.01, f"average coverage below guarantee: {mean_cov}"

    # --- API guards -------------------------------------------------------------
    try:
        SplitConformal(mode="bogus").calibrate(yc, pc)
        raise AssertionError("bad mode not rejected")
    except ValueError:
        pass
    try:
        SplitConformal(alpha=alpha).interval(pt)  # before calibrate
        raise AssertionError("interval before calibrate not rejected")
    except RuntimeError:
        pass
    # bound() must refuse two_sided: an endpoint is not a one-sided certificate.
    try:
        SplitConformal(alpha=alpha, mode="two_sided").calibrate(yc, pc).bound(pt)
        raise AssertionError("bound() on two_sided not rejected")
    except ValueError:
        pass

    # --- REGRESSION: NaN calibration labels must not destroy the certificate ----
    # numpy sorts NaN last, so an unfiltered k-th order statistic would land ON a
    # NaN and make every bound NaN (coverage 0.0) rather than raising.
    yc_nan = yc.copy()
    yc_nan[: yc.size // 10] = np.nan                 # 10% unobserved labels
    sc_nan = SplitConformal(alpha=alpha, mode="two_sided").calibrate(yc_nan, pc)
    assert np.isfinite(sc_nan.q_), "NaN labels produced a NaN threshold"
    assert sc_nan.n_cal_ == yc.size - yc.size // 10, sc_nan.n_cal_
    lo_n, hi_n = sc_nan.interval(pt)
    cov_nan = coverage(yt, lo_n, hi_n)
    print(f"  NaN-robust: 10% unobserved labels -> coverage {cov_nan:.3f} "
          f"(n_cal={sc_nan.n_cal_}, threshold finite)")
    assert cov_nan >= target - 0.03, cov_nan
    # ...and the same for Mondrian, whose per-group counts must be EFFECTIVE
    mc_nan = MondrianConformal(alpha=alpha, mode="two_sided",
                               min_per_group=20).calibrate(yc_nan, pc, gc)
    assert all(np.isfinite(q) for q in mc_nan.q_by_group_.values())
    assert sum(mc_nan.n_by_group_.values()) == sc_nan.n_cal_

    # --- REGRESSION: vectorized Mondrian == the per-element reference ----------
    ref_lo = np.empty_like(pt)
    ref_hi = np.empty_like(pt)
    for i in range(pt.size):                          # explicit per-element loop
        q_i = mc.q_by_group_.get(gt[i], mc.q_global_)
        l_i, h_i = _apply(pt[i:i + 1], q_i, mc.mode)
        ref_lo[i], ref_hi[i] = l_i[0], h_i[0]
    assert np.allclose(ref_lo, mlo) and np.allclose(ref_hi, mhi), \
        "vectorized Mondrian diverges from the per-element reference"
    print("  vectorized Mondrian matches the per-element reference exactly.")

    # --- unseen test group falls back to the pooled threshold, and is recorded --
    mc_u = MondrianConformal(alpha=alpha, min_per_group=20).calibrate(yc, pc, gc)
    lo_u, hi_u = mc_u.interval(np.full(5, 0.5),
                               np.array(["NEVER_SEEN"] * 5, dtype=object))
    assert mc_u.unseen_groups_ == ("NEVER_SEEN",), mc_u.unseen_groups_
    exp_lo, exp_hi = _apply(np.full(5, 0.5), mc_u.q_global_, mc_u.mode)
    assert np.allclose(lo_u, exp_lo) and np.allclose(hi_u, exp_hi), \
        "unseen group must use exactly the pooled threshold"
    print("  unseen group -> pooled threshold, recorded in unseen_groups_.")

    print("  all invariants passed.")
    return 0


# ==============================================================================
# Optional demo on real artifacts (predictor + conformal on a grouped holdout)
# ==============================================================================
def do_demo(args) -> int:  # pragma: no cover - exercised via stage 07 normally
    import importlib.util
    import sys
    import pandas as pd

    def load_mod(path, name):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    m4 = load_mod(str(args.models), "reliant_stage04")
    features = pd.read_parquet(args.features, engine="pyarrow")
    labels = pd.read_parquet(args.labels, engine="pyarrow")
    ds = m4.assemble_dataset(features, labels, target=args.target)

    uniq = np.unique(ds.groups)
    rng = np.random.default_rng(0)
    rng.shuffle(uniq)
    a, b = int(0.5 * len(uniq)), int(0.75 * len(uniq))
    tr = [i for i in range(ds.n) if ds.groups[i] in set(uniq[:a])]
    cal = [i for i in range(ds.n) if ds.groups[i] in set(uniq[a:b])]
    te = [i for i in range(ds.n) if ds.groups[i] in set(uniq[b:])]
    dtr, dcal, dte = ds.subset(tr), ds.subset(cal), ds.subset(te)

    pred = m4.build_predictor("lightgbm").fit_dataset(dtr)
    Pcal, Pte = pred.predict_dataset(dcal), pred.predict_dataset(dte)

    # Flatten across tools for a marginal certificate on the detection target.
    yc = dcal.Y.ravel(); pc = Pcal.ravel()
    yt = dte.Y.ravel(); pt = Pte.ravel()
    m = ~np.isnan(yc); mt = ~np.isnan(yt)
    sc = SplitConformal(alpha=args.alpha, mode="two_sided").calibrate(yc[m], pc[m])
    lo, hi = sc.interval(pt[mt])
    print(f"instances={ds.n} tools={len(ds.tool_names)}")
    print(f"split-conformal marginal coverage on holdout = "
          f"{coverage(yt[mt], lo, hi):.3f} (target {1-args.alpha:.2f}), "
          f"mean width = {mean_width(lo, hi):.3f}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Conformal reliability certificates (stage 05).")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--selftest", action="store_true")
    mode.add_argument("--demo", action="store_true")
    p.add_argument("--models", type=str, default="src/04_models.py")
    p.add_argument("--features", type=str, default="artifacts/features.parquet")
    p.add_argument("--labels", type=str, default="artifacts/labels.parquet")
    p.add_argument("--target", type=str, default="detected")
    p.add_argument("--alpha", type=float, default=0.1)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.demo:
        return do_demo(args)
    return run_selftest()


if __name__ == "__main__":
    raise SystemExit(main())
