#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WPB_DEEP_PATTERN_LAB.py
=======================

Numerical research harness for the quartic zero-repair / predecessor open problem.

The script deliberately uses MULTIPLE independent "eyes" on the data:

  1. exact feasibility / residual audit,
  2. root-cost / scaling diagnostics,
  3. active-set / facet-transition diagnostics,
  4. scaling-collapse diagnostics for predecessor tails,
  5. optimal-face uniqueness audit,
  6. DREAM6-ZR support-pattern compression,
  7. anti-diagonal routing profiles,
  8. child-law / selector-motif compression,
  9. cross-depth recurrence diagnostics.

It does NOT turn any numerical pattern into a theorem.  Its purpose is to create
high-quality data from which a mathematically exact candidate rule can be inferred
and then proved separately.

Two related but distinct objects are kept separate:

A) Homogeneous predecessor orbit
       B_{n+1} in argmin_A { E A : Q*A >=_st P*B_n }.
   This is the obstacle problem from PROBLEM.pdf.

B) Full history-dependent DREAM6-ZR finite-depth trees.
   These exploit Bellman-fiber freedom and need not have the same scaling law as A.

Subcommands
-----------
1) Deep predecessor data:
   python WPB_DEEP_PATTERN_LAB.py predecessor \
       --nmax 1000 --K 128 --out-dir pred1000

2) Run DREAM6-ZR on deeper finite depths with the SAME frozen architecture:
   python WPB_DEEP_PATTERN_LAB.py zr-run \
       --zr-script DREAM6_ZR_v01.py \
       --depths 1 2 3 4 5 6 \
       --out-dir zr_deep

3) Analyze existing DREAM6-ZR NPZ/JSON files:
   python WPB_DEEP_PATTERN_LAB.py zr-analyze \
       --input-dir . --max-depth 6 --out-dir zr_analysis

4) Combine existing predecessor and ZR outputs into one report:
   python WPB_DEEP_PATTERN_LAB.py synthesize \
       --pred-json pred1000/predecessor.json \
       --zr-json zr_analysis/zr_multiview.json \
       --out combined_report.json

Dependencies
------------
numpy, scipy, matplotlib (plots optional).
DREAM6-ZR running additionally requires torch because DREAM6_ZR_v01.py does.

Important
---------
- No verifier feedback is fed into DREAM6-ZR dynamics.
- The predecessor computation is a separate exact finite LP recursion.
- "Facet transition cost" below is reported only via explicit numerical proxies;
  PROBLEM.pdf does not define a canonical transition cost.
- If the predecessor argmin is not numerically unique, the script reports this
  instead of pretending that D is single-valued.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linprog

P = np.asarray([1/8, 1/4, 1/4, 1/4, 1/8], dtype=np.float64)
Q = np.asarray([7/64, 5/16, 5/32, 5/16, 7/64], dtype=np.float64)

B8 = np.asarray([
    0.53562333995590705,
    0.42450840479209673,
    0.039868255251995686,
], dtype=np.float64)

KNOWN_DLE_M8 = {
    1: 0.562605012644505,
    2: 0.589547279384718,
    3: 0.601704229318789,
    4: 0.607609305902225,
    5: 0.61096453,
}

PRED_REFERENCE = {
    5: 0.5047111111111112,
    10: 0.7304679700992335,
    20: 1.0612621328465828,
    40: 1.4808148769739475,
    80: 2.028557484297186,
    100: 2.2284806931361,
    200: 3.09531,
}


# ============================================================================
# Generic helpers
# ============================================================================

def json_dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def stable_hash_bool(arr: np.ndarray) -> str:
    a = np.ascontiguousarray(np.asarray(arr, dtype=np.uint8))
    return hashlib.sha256(a.tobytes()).hexdigest()[:20]


def stable_hash_quantized(arr: np.ndarray, eps: float) -> str:
    x = np.asarray(arr, dtype=np.float64)
    q = np.rint(x / float(eps)).astype(np.int64)
    return hashlib.sha256(np.ascontiguousarray(q).tobytes()).hexdigest()[:20]


def safe_support_end(b: np.ndarray, tol: float = 1e-12) -> int:
    idx = np.flatnonzero(np.asarray(b) > tol)
    return int(idx[-1] + 1) if idx.size else 0


def powers_of_two_up_to(n: int) -> set[int]:
    out = set()
    k = 1
    while k <= n:
        out.add(k)
        k *= 2
    return out


# ============================================================================
# Part A: homogeneous predecessor obstacle problem
# ============================================================================

@dataclass
class PredMatrices:
    K: int
    A_ub: np.ndarray
    cq: np.ndarray
    cp: np.ndarray
    MP: np.ndarray


def build_predecessor_matrices(K: int) -> PredMatrices:
    """
    Variables are a_1,...,a_K.  Extend every tail by
        a_k=1 for k<=0,  a_k=0 for k>K.
    Dominance is checked for k=1,...,K+4.
    """
    if K < 8:
        raise ValueError("K should be at least 8")

    A_dom = np.zeros((K + 4, K), dtype=np.float64)
    cq = np.zeros(K + 4, dtype=np.float64)
    cp = np.zeros(K + 4, dtype=np.float64)
    MP = np.zeros((K + 4, K), dtype=np.float64)

    for row, k in enumerate(range(1, K + 5)):
        for i, q in enumerate(Q):
            t = k - i
            if t <= 0:
                cq[row] += q
            elif t <= K:
                # -L_Q a <= const_Q - L_P b
                A_dom[row, t - 1] -= q

        for i, p in enumerate(P):
            t = k - i
            if t <= 0:
                cp[row] += p
            elif t <= K:
                MP[row, t - 1] += p

    # monotonicity a_{k+1} <= a_k
    A_mon = np.zeros((K - 1, K), dtype=np.float64)
    for k in range(K - 1):
        A_mon[k, k + 1] = 1.0
        A_mon[k, k] = -1.0

    return PredMatrices(
        K=K,
        A_ub=np.vstack([A_dom, A_mon]),
        cq=cq,
        cp=cp,
        MP=MP,
    )


def predecessor_rhs(mats: PredMatrices, b: np.ndarray) -> np.ndarray:
    rhs_p = mats.cp + mats.MP @ b
    return np.concatenate([
        mats.cq - rhs_p,
        np.zeros(mats.K - 1, dtype=np.float64),
    ])


def solve_predecessor_step(
    mats: PredMatrices,
    b: np.ndarray,
    *,
    tol: float,
) -> Tuple[np.ndarray, float, dict]:
    """
    Strict solve with independent residual verification and method fallbacks.
    We prefer to STOP rather than silently continue a numerically bad orbit.
    """
    b_ub = predecessor_rhs(mats, b)
    objective = np.ones(mats.K, dtype=np.float64)

    methods = ("highs", "highs-ipm", "highs-ds")
    attempts = []

    for method in methods:
        options = {
            "primal_feasibility_tolerance": max(tol, 1e-10),
            "dual_feasibility_tolerance": max(tol, 1e-10),
        }
        if method in ("highs", "highs-ipm"):
            options["ipm_optimality_tolerance"] = max(1e-12, tol * 0.01)

        res = linprog(
            objective,
            A_ub=mats.A_ub,
            b_ub=b_ub,
            bounds=(0.0, 1.0),
            method=method,
            options=options,
        )

        if not res.success:
            attempts.append({
                "method": method,
                "success": False,
                "message": str(res.message),
            })
            continue

        a = np.asarray(res.x, dtype=np.float64)
        slack = b_ub - mats.A_ub @ a
        max_violation = float(max(0.0, -np.min(slack)))
        monotone_violation = float(
            max(0.0, np.max(a[1:] - a[:-1])) if mats.K > 1 else 0.0
        )
        bound_violation = float(max(
            0.0,
            -float(np.min(a)),
            float(np.max(a) - 1.0),
        ))

        attempts.append({
            "method": method,
            "success": True,
            "objective": float(res.fun),
            "max_constraint_violation": max_violation,
            "monotone_violation": monotone_violation,
            "bound_violation": bound_violation,
        })

        accept_tol = max(5e-9, 50.0 * tol)
        if (
            max_violation <= accept_tol
            and monotone_violation <= accept_tol
            and bound_violation <= accept_tol
        ):
            return a, float(res.fun), {
                "method": method,
                "attempts": attempts,
                "max_constraint_violation": max_violation,
                "monotone_violation": monotone_violation,
                "bound_violation": bound_violation,
            }

    raise RuntimeError(
        "predecessor LP failed strict verification: "
        + json.dumps(attempts, indent=2)
    )


def predecessor_active_signature(
    mats: PredMatrices,
    b_prev: np.ndarray,
    a: np.ndarray,
    *,
    active_tol: float,
) -> dict:
    b_ub = predecessor_rhs(mats, b_prev)
    slack = b_ub - mats.A_ub @ a

    ndom = mats.K + 4
    dom_active = slack[:ndom] <= active_tol
    mon_active = slack[ndom:] <= active_tol

    return {
        "dominance_active_count": int(np.sum(dom_active)),
        "monotone_active_count": int(np.sum(mon_active)),
        "signature_hash": stable_hash_bool(np.concatenate([dom_active, mon_active])),
        "dominance_active": dom_active,
        "monotone_active": mon_active,
        "min_slack": float(np.min(slack)),
    }


def tail_features(b: np.ndarray) -> dict:
    b = np.asarray(b, dtype=np.float64)
    K = len(b)
    k = np.arange(1, K + 1, dtype=np.float64)

    # Tail identities:
    # E J = sum b_k
    # E J^2 = sum (2k-1)b_k
    # E J^3 = sum (3k^2-3k+1)b_k
    mean = float(np.sum(b))
    m2 = float(np.sum((2.0 * k - 1.0) * b))
    m3 = float(np.sum((3.0 * k * k - 3.0 * k + 1.0) * b))

    # Difference hierarchy with left extension b_0=1 and a zero right pad.
    ext = np.concatenate([[1.0], b, np.zeros(5, dtype=np.float64)])
    d1 = np.diff(ext)
    d2 = np.diff(ext, n=2)
    d3 = np.diff(ext, n=3)
    d4 = np.diff(ext, n=4)

    eps = 1e-15
    interior = np.clip(b, eps, 1.0 - eps)
    entropy = float(np.sum(
        -interior * np.log(interior)
        - (1.0 - interior) * np.log(1.0 - interior)
    ))

    return {
        "mean": mean,
        "second_moment": m2,
        "third_moment": m3,
        "d2_l1": float(np.sum(np.abs(d2))),
        "d2_l2sq": float(np.dot(d2, d2)),
        "d3_l1": float(np.sum(np.abs(d3))),
        "d3_l2sq": float(np.dot(d3, d3)),
        "d4_l1": float(np.sum(np.abs(d4))),
        "d4_l2sq": float(np.dot(d4, d4)),
        "tail_entropy": entropy,
        "support_end": safe_support_end(b),
        "b1": float(b[0]) if K else 0.0,
    }


def optimal_face_probe(
    mats: PredMatrices,
    b_prev: np.ndarray,
    optimum: float,
    *,
    face_eps: float,
    probe_count: int,
) -> dict:
    """
    Numerical set-valuedness audit for D(B).

    Adds sum a_k <= optimum + face_eps and probes the optimal face in fixed,
    deterministic directions.  Large diameter means the argmin is not being
    observed as a single point at this tolerance.
    """
    base_rhs = predecessor_rhs(mats, b_prev)
    A_face = np.vstack([mats.A_ub, np.ones((1, mats.K), dtype=np.float64)])
    b_face = np.concatenate([base_rhs, [float(optimum + face_eps)]])

    directions = []

    # Coordinate directions concentrated where the current support lives.
    stride = max(1, mats.K // max(1, probe_count // 2))
    for j in range(0, mats.K, stride):
        e = np.zeros(mats.K, dtype=np.float64)
        e[j] = 1.0
        directions.append(e)
        if len(directions) >= probe_count // 2:
            break

    # Deterministic sinusoidal directions probe collective degrees of freedom.
    grid = np.arange(1, mats.K + 1, dtype=np.float64)
    freq = 1
    while len(directions) < probe_count:
        v = np.sin(np.pi * freq * grid / (mats.K + 1.0))
        v /= max(1e-30, np.linalg.norm(v))
        directions.append(v)
        freq += 1

    widths = []
    for c in directions:
        lo = linprog(
            c,
            A_ub=A_face,
            b_ub=b_face,
            bounds=(0.0, 1.0),
            method="highs",
        )
        hi = linprog(
            -c,
            A_ub=A_face,
            b_ub=b_face,
            bounds=(0.0, 1.0),
            method="highs",
        )
        if not (lo.success and hi.success):
            continue
        widths.append(float((-hi.fun) - lo.fun))

    return {
        "face_eps": float(face_eps),
        "probe_count_requested": int(probe_count),
        "probe_count_completed": int(len(widths)),
        "max_directional_width": float(max(widths)) if widths else None,
        "median_directional_width": float(np.median(widths)) if widths else None,
        "width_over_face_eps": (
            float(max(widths) / face_eps) if widths and face_eps > 0 else None
        ),
    }


def profile_collapse_rmse(
    b_n: np.ndarray,
    b_2n: np.ndarray,
    spatial_ratio: float,
    *,
    tail_floor: float = 1e-9,
) -> float:
    """
    If b_n(k) ~ F(k / s_n), spatial_ratio=s_{2n}/s_n.
    Compare b_n(k) with b_{2n}(spatial_ratio*k).
    """
    b_n = np.asarray(b_n, dtype=np.float64)
    b_2n = np.asarray(b_2n, dtype=np.float64)

    maxk = safe_support_end(b_n, tail_floor)
    if maxk < 2:
        return float("nan")

    x = np.arange(1, maxk + 1, dtype=np.float64)
    xp = spatial_ratio * x

    grid2 = np.arange(1, len(b_2n) + 1, dtype=np.float64)
    y2 = np.interp(xp, grid2, b_2n, left=b_2n[0], right=0.0)
    y1 = b_n[:maxk]

    return float(np.sqrt(np.mean((y1 - y2) ** 2)))


def best_power_collapse(
    b_n: np.ndarray,
    b_2n: np.ndarray,
    *,
    alpha_min: float = 0.0,
    alpha_max: float = 0.8,
    alpha_steps: int = 161,
) -> dict:
    alphas = np.linspace(alpha_min, alpha_max, alpha_steps)
    rmses = np.asarray([
        profile_collapse_rmse(b_n, b_2n, 2.0 ** float(a))
        for a in alphas
    ])
    finite = np.isfinite(rmses)
    if not np.any(finite):
        return {"alpha": None, "rmse": None}
    j = np.flatnonzero(finite)[np.argmin(rmses[finite])]
    return {"alpha": float(alphas[j]), "rmse": float(rmses[j])}


def regression_models(ns: np.ndarray, ms: np.ndarray) -> dict:
    """
    Descriptive late-window fits only.  These are NOT asymptotic model selection.
    """
    ns = np.asarray(ns, dtype=np.float64)
    ms = np.asarray(ms, dtype=np.float64)

    mask = ns >= max(10.0, np.quantile(ns, 0.5))
    x = ns[mask]
    y = ms[mask]

    def fit_linear(design: np.ndarray) -> dict:
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        pred = design @ coef
        resid = y - pred
        rmse = float(np.sqrt(np.mean(resid ** 2)))
        scale = float(max(1e-30, np.std(y)))
        return {
            "coef": coef.tolist(),
            "rmse": rmse,
            "rmse_over_std": float(rmse / scale),
        }

    log_fit = fit_linear(np.column_stack([np.log(x), np.ones_like(x)]))
    sqrt_fit = fit_linear(np.column_stack([np.sqrt(x), np.ones_like(x)]))

    # grid-search power alpha with affine intercept
    alpha_grid = np.linspace(0.05, 0.75, 141)
    best = None
    for alpha in alpha_grid:
        r = fit_linear(np.column_stack([x ** alpha, np.ones_like(x)]))
        if best is None or r["rmse"] < best["rmse"]:
            best = {"alpha": float(alpha), **r}

    return {
        "window_n_min": int(np.min(x)),
        "window_n_max": int(np.max(x)),
        "log_affine": log_fit,
        "sqrt_affine": sqrt_fit,
        "best_power_affine": best,
    }


def run_predecessor(args) -> int:
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    mats = build_predecessor_matrices(args.K)
    b = np.zeros(args.K, dtype=np.float64)

    snapshot_set = powers_of_two_up_to(args.nmax)
    snapshot_set.update(
        int(x) for x in args.snapshots.split(",") if x.strip()
    )
    snapshot_set = {n for n in snapshot_set if 1 <= n <= args.nmax}

    rows = []
    snapshots: Dict[int, np.ndarray] = {}
    transition_events = []
    uniqueness = {}

    previous_signature = None
    previous_features = tail_features(b)
    started = time.perf_counter()

    for n in range(1, args.nmax + 1):
        b_prev = b.copy()
        a, mean, solve_meta = solve_predecessor_step(
            mats, b_prev, tol=args.lp_tol
        )
        b = a

        features = tail_features(b)
        active = predecessor_active_signature(
            mats, b_prev, b, active_tol=args.active_tol
        )

        signature_changed = (
            previous_signature is not None
            and active["signature_hash"] != previous_signature
        )

        row = {
            "n": int(n),
            "mean": float(mean),
            "increment": float(mean - previous_features["mean"]),
            "support_end": int(features["support_end"]),
            "dominance_active_count": active["dominance_active_count"],
            "monotone_active_count": active["monotone_active_count"],
            "active_signature": active["signature_hash"],
            "signature_changed": bool(signature_changed),
            "tail_l1_move": float(np.sum(np.abs(b - b_prev))),
            "solver_method": solve_meta["method"],
            "max_constraint_violation": solve_meta["max_constraint_violation"],
            "d2_l1": features["d2_l1"],
            "d3_l1": features["d3_l1"],
            "d4_l1": features["d4_l1"],
            "d4_l2sq": features["d4_l2sq"],
            "tail_entropy": features["tail_entropy"],
            "second_moment": features["second_moment"],
            "third_moment": features["third_moment"],
        }
        rows.append(row)

        if signature_changed:
            # Explicit proxies only; no claim that this is the canonical
            # "transition cost" requested in Problem 3.
            transition_events.append({
                "n": int(n),
                "mean_increment_proxy": row["increment"],
                "tail_l1_move_proxy": row["tail_l1_move"],
                "active_count_change_proxy": int(
                    row["dominance_active_count"]
                    + row["monotone_active_count"]
                    - rows[-2]["dominance_active_count"]
                    - rows[-2]["monotone_active_count"]
                ),
                "new_signature": row["active_signature"],
            })

        previous_signature = active["signature_hash"]
        previous_features = features

        if n in snapshot_set:
            snapshots[n] = b.copy()

        if args.uniqueness_every > 0 and (
            n <= 4 or n % args.uniqueness_every == 0
        ):
            uniqueness[n] = optimal_face_probe(
                mats,
                b_prev,
                mean,
                face_eps=args.face_eps,
                probe_count=args.face_probes,
            )

        if n in PRED_REFERENCE:
            print(
                f"[predecessor] n={n:5d}"
                f" m={mean:.15g}"
                f" reference={PRED_REFERENCE[n]:.15g}"
                f" diff={mean-PRED_REFERENCE[n]:+.3e}"
                f" support={features['support_end']}"
                f" face={'CHANGE' if signature_changed else 'same'}"
            )
        elif n in snapshot_set:
            print(
                f"[predecessor] n={n:5d}"
                f" m={mean:.15g}"
                f" support={features['support_end']}"
                f" face={'CHANGE' if signature_changed else 'same'}"
            )

        # Scientific cutoff audit.
        if features["support_end"] >= args.K - args.support_guard:
            raise RuntimeError(
                f"support reached {features['support_end']} with K={args.K}; "
                "increase --K before interpreting later iterates"
            )

    # Scaling diagnostics from exact stored rows.
    mean_by_n = {int(r["n"]): float(r["mean"]) for r in rows}
    dyadic = []
    for n in sorted(mean_by_n):
        if 2 * n not in mean_by_n:
            continue
        mn = mean_by_n[n]
        m2n = mean_by_n[2 * n]
        item = {
            "n": n,
            "m_n": mn,
            "m_2n": m2n,
            "alpha_doubling": float(math.log(m2n / mn) / math.log(2.0)),
            "d_sqrt": float((m2n - mn) / math.sqrt(n)),
            "doubling_increment": float(m2n - mn),
            "m_over_sqrt_n": float(mn / math.sqrt(n)),
            "m_over_log_n": float(mn / math.log(n)) if n > 1 else None,
        }
        if n in snapshots and 2 * n in snapshots:
            power = best_power_collapse(snapshots[n], snapshots[2 * n])
            item["best_profile_collapse_alpha"] = power["alpha"]
            item["best_profile_collapse_rmse"] = power["rmse"]
            item["sqrt_profile_rmse"] = profile_collapse_rmse(
                snapshots[n], snapshots[2 * n], math.sqrt(2.0)
            )
            if n > 1:
                item["log_profile_rmse"] = profile_collapse_rmse(
                    snapshots[n],
                    snapshots[2 * n],
                    math.log(2 * n) / math.log(n),
                )
        dyadic.append(item)

    ns = np.asarray([r["n"] for r in rows], dtype=np.float64)
    ms = np.asarray([r["mean"] for r in rows], dtype=np.float64)

    # Transition burst summary.
    transition_ns = np.asarray(
        [e["n"] for e in transition_events], dtype=np.int64
    )
    burst_gaps = np.diff(transition_ns) if len(transition_ns) >= 2 else np.asarray([])

    summary = {
        "problem": "homogeneous quartic predecessor",
        "P": P.tolist(),
        "Q": Q.tolist(),
        "nmax": int(args.nmax),
        "K": int(args.K),
        "runtime_seconds": float(time.perf_counter() - started),
        "final_mean": float(rows[-1]["mean"]),
        "final_support_end": int(rows[-1]["support_end"]),
        "reference_differences": {
            str(n): (
                float(mean_by_n[n] - ref)
                if n in mean_by_n else None
            )
            for n, ref in PRED_REFERENCE.items()
        },
        "facet_transition_count_proxy": int(len(transition_events)),
        "facet_transition_fraction_proxy": float(
            len(transition_events) / max(1, args.nmax - 1)
        ),
        "median_steps_between_signature_changes": (
            float(np.median(burst_gaps)) if burst_gaps.size else None
        ),
        "dyadic_scaling": dyadic,
        "late_window_regressions": regression_models(ns, ms),
        "optimal_face_probes": {str(k): v for k, v in uniqueness.items()},
        "interpretation_warning": (
            "All scaling fits and transition costs are diagnostics only. "
            "They do not prove an asymptotic exponent or a Lyapunov theorem."
        ),
    }

    # CSV
    csv_path = out / "predecessor_trajectory.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    npz_kwargs = {
        "n": np.asarray([r["n"] for r in rows], dtype=np.int64),
        "mean": ms,
    }
    for n, snap in snapshots.items():
        npz_kwargs[f"tail_n{n}"] = snap
    np.savez_compressed(out / "predecessor_snapshots.npz", **npz_kwargs)

    json_dump(out / "predecessor.json", summary)

    # Compact text report.
    report_lines = []
    report_lines.append("HOMOGENEOUS PREDECESSOR MULTIVIEW REPORT")
    report_lines.append("=" * 78)
    report_lines.append(f"nmax={args.nmax} K={args.K}")
    report_lines.append(
        f"final mean={summary['final_mean']:.15g}, "
        f"support_end={summary['final_support_end']}"
    )
    report_lines.append(
        "facet-signature changes (proxy) = "
        f"{summary['facet_transition_count_proxy']}"
    )
    report_lines.append("")
    report_lines.append("Dyadic diagnostics:")
    for d in dyadic:
        if d["n"] < 5:
            continue
        report_lines.append(
            f"n={d['n']:5d} -> {2*d['n']:5d} "
            f"alpha={d['alpha_doubling']:.6f} "
            f"d_sqrt={d['d_sqrt']:.6f} "
            f"m/sqrt(n)={d['m_over_sqrt_n']:.6f} "
            + (
                f"collapse_alpha={d.get('best_profile_collapse_alpha'):.4f}"
                if d.get("best_profile_collapse_alpha") is not None
                else ""
            )
        )
    report_lines.append("")
    report_lines.append("Late-window descriptive fits:")
    report_lines.append(json.dumps(summary["late_window_regressions"], indent=2))
    (out / "predecessor_report.txt").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )

    if not args.no_plots:
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(8, 5))
            plt.plot(ns, ms, label="m_n")
            plt.plot(ns, ms / np.sqrt(ns), label="m_n / sqrt(n)")
            plt.xlabel("n")
            plt.ylabel("diagnostic")
            plt.title("Quartic predecessor trajectory")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out / "predecessor_scaling.png", dpi=160)
            plt.close()

            alx = [d["n"] for d in dyadic if d["n"] >= 2]
            aly = [d["alpha_doubling"] for d in dyadic if d["n"] >= 2]
            plt.figure(figsize=(8, 5))
            plt.plot(alx, aly, marker="o")
            plt.xscale("log")
            plt.xlabel("n")
            plt.ylabel("log(m_2n/m_n)/log 2")
            plt.title("Doubling exponent diagnostic")
            plt.tight_layout()
            plt.savefig(out / "predecessor_doubling_exponent.png", dpi=160)
            plt.close()
        except Exception as exc:
            print(f"[warning] plots skipped: {exc}")

    print(f"[done] predecessor data -> {out}")
    return 0


# ============================================================================
# Part B: DREAM6-ZR deep finite-depth solver orchestration
# ============================================================================

def run_zr_depths(args) -> int:
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    zr_script = Path(args.zr_script)
    if not zr_script.exists():
        raise FileNotFoundError(zr_script)

    run_log = []

    for depth in args.depths:
        depth = int(depth)
        json_path = out / f"DREAM6_ZR_r{depth}.json"
        npz_path = out / f"DREAM6_ZR_r{depth}.npz"

        cmd = [
            sys.executable,
            str(zr_script),
            "--depth", str(depth),
            "--M", str(args.M),
            "--iterations", str(args.iterations),
            "--learning-rate", str(args.learning_rate),
            "--root-weight", str(args.root_weight),
            "--temperature-stop", str(args.temperature_stop),
            "--branch-weight", str(args.branch_weight),
            "--concurrence-weight", str(args.concurrence_weight),
            "--leaf-weight", str(args.leaf_weight),
            "--entropy-weight", str(args.entropy_weight),
            "--readout-topk-k", str(args.topk),
            "--verbose-every", str(args.verbose_every),
            "--json-out", str(json_path),
            "--npz-out", str(npz_path),
        ]
        if depth <= args.reference_max_depth:
            cmd.append("--reference-full-lp")

        print("=" * 100)
        print(f"[zr-run] depth={depth}")
        print(" ".join(cmd))

        started = time.perf_counter()
        proc = subprocess.run(cmd, text=True)
        elapsed = time.perf_counter() - started

        entry = {
            "depth": depth,
            "returncode": int(proc.returncode),
            "seconds": float(elapsed),
            "json": str(json_path),
            "npz": str(npz_path),
        }
        run_log.append(entry)
        json_dump(out / "zr_run_log.json", run_log)

        if proc.returncode != 0:
            print(f"[zr-run] depth={depth} FAILED; stopping without retuning.")
            break

    print(f"[done] zr runs -> {out}")
    return 0


# ============================================================================
# Part C: DREAM6-ZR multiview pattern analysis
# ============================================================================

def build_tree_nodes(depth: int):
    nodes: List[Tuple[int, ...]] = [()]
    for d in range(1, depth + 1):
        nodes.extend(itertools.product(range(5), repeat=d))
    idx = {h: i for i, h in enumerate(nodes)}
    internals = [h for h in nodes if len(h) < depth]
    return nodes, idx, internals


def parse_depth_from_name(path: Path) -> Optional[int]:
    m = re.search(r"_r(\d+)", path.name)
    return int(m.group(1)) if m else None


def node_moment_signature(a: np.ndarray, eps: float) -> str:
    j = np.arange(len(a), dtype=np.float64)
    mean = float(a @ j)
    m2 = float(a @ (j * j))
    c0 = float(a[0])
    c1 = float(a[:2].sum())
    support = int(safe_support_end(a, 1e-12))
    v = np.asarray([mean, m2, c0, c1, float(support)], dtype=np.float64)
    return stable_hash_quantized(v, eps)


def zr_depth_fingerprint(npz_path: Path, json_path: Optional[Path], *, motif_eps: float):
    depth = parse_depth_from_name(npz_path)
    if depth is None:
        raise ValueError(f"cannot parse depth from {npz_path.name}")

    zf = np.load(npz_path)
    required = ("A_exact", "z_exact", "K_support")
    for key in required:
        if key not in zf:
            raise ValueError(f"{npz_path} missing {key}")

    A = np.asarray(zf["A_exact"], dtype=np.float64)
    Z = np.asarray(zf["z_exact"], dtype=np.float64)
    K_support = np.asarray(zf["K_support"], dtype=np.bool_)
    valid = np.asarray(zf["valid"], dtype=np.bool_)

    nodes, idx, internals = build_tree_nodes(depth)

    meta = {}
    if json_path is not None and json_path.exists():
        meta = json.loads(json_path.read_text(encoding="utf-8"))

    by_level = []
    for d in range(depth):
        internal_ids = [
            ii for ii, h in enumerate(internals) if len(h) == d
        ]
        node_ids = [idx[internals[ii]] for ii in internal_ids]

        laws = A[node_ids]
        zloc = Z[internal_ids]
        ksup = K_support[internal_ids]

        # Selector motifs: pure combinatorial support patterns.
        k_hashes = [
            stable_hash_bool(ksup[q] & valid)
            for q in range(len(internal_ids))
        ]

        # Exactified transport support motifs.
        z_hashes = [
            stable_hash_bool(zloc[q] > 1e-12)
            for q in range(len(internal_ids))
        ]

        law_support_hashes = [
            stable_hash_bool(laws[q] > 1e-12)
            for q in range(len(node_ids))
        ]

        law_moment_hashes = [
            node_moment_signature(laws[q], motif_eps)
            for q in range(len(node_ids))
        ]

        # Aggregate carry law.
        carry_mean = laws.mean(axis=0)

        # Aggregate state-conditional drift.
        zz = zloc.sum(axis=0)  # sum over nodes
        drift = np.zeros(A.shape[1], dtype=np.float64)
        for j in range(A.shape[1]):
            den = float(np.sum(zz[j]))
            if den > 1e-14:
                num = 0.0
                for y in range(5):
                    for x in range(5):
                        num += zz[j, y, x] * float(y - x)
                drift[j] = num / den

        # Aggregate anti-diagonal branch routing.
        max_s = A.shape[1] - 1 + 4
        anti = np.zeros((max_s + 1, 5), dtype=np.float64)
        for j in range(A.shape[1]):
            for y in range(5):
                s = j + y
                anti[s] += zz[j, y]
        den = anti.sum(axis=1, keepdims=True)
        anti_profile = np.divide(
            anti,
            den,
            out=np.zeros_like(anti),
            where=den > 0,
        )

        by_level.append({
            "absolute_depth": int(d),
            "remaining_depth": int(depth - d),
            "node_count": int(len(node_ids)),
            "selector_support_motif_count": int(len(set(k_hashes))),
            "z_support_motif_count": int(len(set(z_hashes))),
            "law_support_motif_count": int(len(set(law_support_hashes))),
            "law_moment_motif_count": int(len(set(law_moment_hashes))),
            "selector_compression_ratio": float(
                len(set(k_hashes)) / max(1, len(k_hashes))
            ),
            "law_moment_compression_ratio": float(
                len(set(law_moment_hashes)) / max(1, len(law_moment_hashes))
            ),
            "mean_carry_law": carry_mean.tolist(),
            "state_drift": drift.tolist(),
            "anti_profile": anti_profile.tolist(),
        })

    root_mean = float(A[0] @ np.arange(A.shape[1], dtype=np.float64))

    root_anti = np.asarray(by_level[0]["anti_profile"], dtype=np.float64)

    return {
        "depth": int(depth),
        "npz": str(npz_path),
        "json": str(json_path) if json_path else None,
        "root_mean": root_mean,
        "known_DLE_M8": KNOWN_DLE_M8.get(depth),
        "root_premium_vs_known": (
            float(root_mean - KNOWN_DLE_M8[depth])
            if depth in KNOWN_DLE_M8 else None
        ),
        "exactification_success": (
            meta.get("exactification", {}).get("success") if meta else None
        ),
        "verification_pass": (
            meta.get("verification", {}).get("pass") if meta else None
        ),
        "selected_fraction": (
            meta.get("readout", {}).get("selected_fraction") if meta else None
        ),
        "mean_reversion": meta.get("mean_reversion_diagnostic") if meta else None,
        "by_level": by_level,
        "root_anti_profile": root_anti.tolist(),
    }


def anti_profile_distance(a: np.ndarray, b: np.ndarray) -> dict:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = max(a.shape[0], b.shape[0])
    aa = np.zeros((n, 5), dtype=np.float64)
    bb = np.zeros((n, 5), dtype=np.float64)
    aa[:a.shape[0]] = a
    bb[:b.shape[0]] = b

    active = (aa.sum(axis=1) > 0) | (bb.sum(axis=1) > 0)
    if not np.any(active):
        return {"mean_row_l1": 0.0, "frobenius": 0.0}
    return {
        "mean_row_l1": float(
            np.mean(np.sum(np.abs(aa[active] - bb[active]), axis=1))
        ),
        "frobenius": float(np.linalg.norm(aa - bb)),
    }


def analyze_zr(args) -> int:
    inp = Path(args.input_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    files = []
    for path in inp.glob(args.npz_glob):
        d = parse_depth_from_name(path)
        if d is None or d > args.max_depth:
            continue
        files.append((d, path))
    files.sort()

    if not files:
        raise FileNotFoundError(
            f"no files matching {inp / args.npz_glob}"
        )

    fps = []
    for depth, npz_path in files:
        json_path = inp / f"DREAM6_ZR_r{depth}.json"
        fp = zr_depth_fingerprint(
            npz_path,
            json_path if json_path.exists() else None,
            motif_eps=args.motif_eps,
        )
        fps.append(fp)

    cross = []

    # Root-profile stabilization across total depth.
    for a, b in zip(fps[:-1], fps[1:]):
        dist = anti_profile_distance(
            np.asarray(a["root_anti_profile"]),
            np.asarray(b["root_anti_profile"]),
        )
        cross.append({
            "kind": "root_anti_profile",
            "depth_a": a["depth"],
            "depth_b": b["depth"],
            **dist,
        })

    # Same remaining-depth comparison: does a boundary-layer motif recur?
    lookup = {fp["depth"]: fp for fp in fps}
    for q in range(1, max(fp["depth"] for fp in fps) + 1):
        available = []
        for fp in fps:
            level = next(
                (x for x in fp["by_level"] if x["remaining_depth"] == q),
                None,
            )
            if level is not None:
                available.append((fp["depth"], level))
        for (ra, la), (rb, lb) in zip(available[:-1], available[1:]):
            dist = anti_profile_distance(
                np.asarray(la["anti_profile"]),
                np.asarray(lb["anti_profile"]),
            )
            carry_l1 = float(np.sum(np.abs(
                np.asarray(la["mean_carry_law"])
                - np.asarray(lb["mean_carry_law"])
            )))
            drift_l1 = float(np.sum(np.abs(
                np.asarray(la["state_drift"])
                - np.asarray(lb["state_drift"])
            )))
            cross.append({
                "kind": "same_remaining_depth",
                "remaining_depth": int(q),
                "depth_a": int(ra),
                "depth_b": int(rb),
                "carry_l1": carry_l1,
                "drift_l1": drift_l1,
                **dist,
            })

    # Compression summary by total depth.
    compression = []
    for fp in fps:
        total_nodes = sum(x["node_count"] for x in fp["by_level"])
        total_selector_motifs = sum(
            x["selector_support_motif_count"] for x in fp["by_level"]
        )
        total_law_motifs = sum(
            x["law_moment_motif_count"] for x in fp["by_level"]
        )
        compression.append({
            "depth": fp["depth"],
            "internal_nodes": int(total_nodes),
            "sum_selector_motif_counts_by_level": int(total_selector_motifs),
            "sum_law_moment_motif_counts_by_level": int(total_law_motifs),
            "selector_motifs_per_internal_node": float(
                total_selector_motifs / max(1, total_nodes)
            ),
            "law_motifs_per_internal_node": float(
                total_law_motifs / max(1, total_nodes)
            ),
        })

    report = {
        "object": "full history-dependent DREAM6-ZR finite-depth trees",
        "depth_fingerprints": fps,
        "cross_depth_comparisons": cross,
        "compression_summary": compression,
        "interpretation_warning": (
            "A stable numerical motif is a candidate construction, not a proof. "
            "Exactification verifies each finite tree but does not prove a scale-recursive law."
        ),
    }
    json_dump(out / "zr_multiview.json", report)

    # Compact CSV summary.
    with (out / "zr_depth_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fh:
        fields = [
            "depth",
            "root_mean",
            "known_DLE_M8",
            "root_premium_vs_known",
            "verification_pass",
            "selected_fraction",
        ]
        wr = csv.DictWriter(fh, fieldnames=fields)
        wr.writeheader()
        for fp in fps:
            wr.writerow({k: fp.get(k) for k in fields})

    lines = []
    lines.append("DREAM6-ZR MULTIVIEW PATTERN REPORT")
    lines.append("=" * 78)
    for fp in fps:
        lines.append(
            f"r={fp['depth']} root={fp['root_mean']:.15g} "
            f"premium={fp['root_premium_vs_known']} "
            f"verify={fp['verification_pass']}"
        )
        for level in fp["by_level"]:
            lines.append(
                f"  d={level['absolute_depth']} q={level['remaining_depth']} "
                f"nodes={level['node_count']} "
                f"Kmotifs={level['selector_support_motif_count']} "
                f"Zmotifs={level['z_support_motif_count']} "
                f"lawMotifs={level['law_moment_motif_count']}"
            )
    lines.append("")
    lines.append("Root anti-diagonal stabilization:")
    for item in cross:
        if item["kind"] == "root_anti_profile":
            lines.append(
                f"  r={item['depth_a']}->{item['depth_b']} "
                f"mean-row-L1={item['mean_row_l1']:.6g}"
            )
    (out / "zr_multiview_report.txt").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    if not args.no_plots:
        try:
            import matplotlib.pyplot as plt

            ds = np.asarray([fp["depth"] for fp in fps], dtype=float)
            ms = np.asarray([fp["root_mean"] for fp in fps], dtype=float)

            plt.figure(figsize=(8, 5))
            plt.plot(ds, ms, marker="o")
            plt.xlabel("finite depth r")
            plt.ylabel("exactified root mean")
            plt.title("DREAM6-ZR exactified root mean")
            plt.tight_layout()
            plt.savefig(out / "zr_root_mean.png", dpi=160)
            plt.close()

            root_dist = [
                item for item in cross
                if item["kind"] == "root_anti_profile"
            ]
            if root_dist:
                x = [item["depth_b"] for item in root_dist]
                y = [item["mean_row_l1"] for item in root_dist]
                plt.figure(figsize=(8, 5))
                plt.plot(x, y, marker="o")
                plt.xlabel("new total depth r")
                plt.ylabel("root anti-profile mean row L1 change")
                plt.title("Cross-depth stabilization of root anti-diagonal routing")
                plt.tight_layout()
                plt.savefig(out / "zr_root_antidiagonal_stability.png", dpi=160)
                plt.close()
        except Exception as exc:
            print(f"[warning] plots skipped: {exc}")

    print(f"[done] zr multiview analysis -> {out}")
    return 0


# ============================================================================
# Part D: synthesis without conflating the two problems
# ============================================================================

def synthesize(args) -> int:
    pred = json.loads(Path(args.pred_json).read_text(encoding="utf-8"))
    zr = json.loads(Path(args.zr_json).read_text(encoding="utf-8"))

    combined = {
        "homogeneous_predecessor": {
            "nmax": pred.get("nmax"),
            "final_mean": pred.get("final_mean"),
            "dyadic_scaling": pred.get("dyadic_scaling"),
            "late_window_regressions": pred.get("late_window_regressions"),
            "facet_transition_count_proxy": pred.get("facet_transition_count_proxy"),
            "optimal_face_probes": pred.get("optimal_face_probes"),
        },
        "full_causal_zr": {
            "depth_fingerprints": zr.get("depth_fingerprints"),
            "cross_depth_comparisons": zr.get("cross_depth_comparisons"),
            "compression_summary": zr.get("compression_summary"),
        },
        "do_not_conflate": (
            "The predecessor orbit is one homogeneous prefix-compatible construction. "
            "The DREAM6-ZR finite-depth trees use history-dependent Bellman-fiber freedom. "
            "Different scaling behavior is therefore logically possible and informative."
        ),
        "questions_targeted": {
            "Problem_2_growth": (
                "Use predecessor dyadic exponents, m/sqrt(n), m/log(n), "
                "late-window fits and profile-collapse exponents."
            ),
            "Problem_3_active_set_complexity": (
                "Use active-signature changes, burst gaps, and explicit transition proxies."
            ),
            "Problem_4_scaling_limit": (
                "Use tail profile-collapse scans and compare log/sqrt/power spatial scalings."
            ),
            "Problem_5_Lyapunov": (
                "Use stored tail difference energies and moment/entropy features; "
                "these are candidate diagnostics only."
            ),
            "Bellman_fiber_question": (
                "Use DREAM6-ZR selector motif counts, child-law motif counts, "
                "anti-diagonal recurrence and cross-depth compression."
            ),
        },
    }

    out = Path(args.out)
    json_dump(out, combined)
    print(f"[done] combined synthesis -> {out}")
    return 0


# ============================================================================
# CLI
# ============================================================================

def build_parser():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("predecessor")
    p.add_argument("--nmax", type=int, default=1000)
    p.add_argument("--K", type=int, default=128)
    p.add_argument("--lp-tol", type=float, default=1e-10)
    p.add_argument("--active-tol", type=float, default=1e-9)
    p.add_argument("--support-guard", type=int, default=8)
    p.add_argument(
        "--snapshots",
        type=str,
        default="5,10,20,40,80,100,200,500,1000",
    )
    p.add_argument("--uniqueness-every", type=int, default=100)
    p.add_argument("--face-eps", type=float, default=1e-9)
    p.add_argument("--face-probes", type=int, default=12)
    p.add_argument("--out-dir", type=Path, default=Path("pred_data"))
    p.add_argument("--no-plots", action="store_true")
    p.set_defaults(func=run_predecessor)

    z = sub.add_parser("zr-run")
    z.add_argument("--zr-script", type=Path, required=True)
    z.add_argument("--depths", type=int, nargs="+", default=[1,2,3,4,5,6])
    z.add_argument("--M", type=int, default=8)
    z.add_argument("--iterations", type=int, default=3000)
    z.add_argument("--learning-rate", type=float, default=0.02)
    z.add_argument("--root-weight", type=float, default=0.02)
    z.add_argument("--temperature-stop", type=float, default=0.6)
    z.add_argument("--branch-weight", type=float, default=20.0)
    z.add_argument("--concurrence-weight", type=float, default=60.0)
    z.add_argument("--leaf-weight", type=float, default=150.0)
    z.add_argument("--entropy-weight", type=float, default=0.0)
    z.add_argument("--topk", type=int, default=4)
    z.add_argument("--verbose-every", type=int, default=1000)
    z.add_argument("--reference-max-depth", type=int, default=4)
    z.add_argument("--out-dir", type=Path, default=Path("zr_deep"))
    z.set_defaults(func=run_zr_depths)

    a = sub.add_parser("zr-analyze")
    a.add_argument("--input-dir", type=Path, default=Path("."))
    a.add_argument("--npz-glob", type=str, default="DREAM6_ZR_r*.npz")
    a.add_argument("--max-depth", type=int, default=12)
    a.add_argument("--motif-eps", type=float, default=1e-4)
    a.add_argument("--out-dir", type=Path, default=Path("zr_analysis"))
    a.add_argument("--no-plots", action="store_true")
    a.set_defaults(func=analyze_zr)

    s = sub.add_parser("synthesize")
    s.add_argument("--pred-json", type=Path, required=True)
    s.add_argument("--zr-json", type=Path, required=True)
    s.add_argument("--out", type=Path, default=Path("combined_report.json"))
    s.set_defaults(func=synthesize)

    return ap


def main():
    ap = build_parser()
    args = ap.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
