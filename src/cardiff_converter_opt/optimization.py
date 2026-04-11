"""Pareto sweep via weighted-sum scalarisation (MATLAB ``pareto_optimal`` loop).

The original CVX programs are not available; we use ``scipy.optimize.minimize`` with
box bounds on ``(A_sw, f_sw, I_ripple, ΔT_j)``, matching the ANN features in Step3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import minimize

from .parameters import DesignParameters
from .physics import component_breakdown, total_loss, total_volume_m3


@dataclass
class ParetoPoint:
    pareto_weight: float  # same role as MATLAB ``pareto_optimal`` (0..1)
    area_sw: float
    f_sw: float
    i_ripple_pp: float
    delta_t_j: float
    p_total: float
    v_total_m3: float
    p_semiconductor: float
    p_inductor: float
    p_capacitor: float
    v_switch_m3: float
    v_inductor_m3: float
    v_capacitor_m3: float


def _pack(x: np.ndarray) -> tuple[float, float, float, float]:
    return float(x[0]), float(x[1]), float(x[2]), float(x[3])


def compute_normalisation_bounds(
    p: DesignParameters,
    n_samples: int = 4000,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Return ``(loss_max, vol_max)`` for normalising the weighted objective."""

    rng = rng or np.random.default_rng(0)
    a_lo, a_hi = 0.4 * p.area_swref, 4.0 * p.area_swref
    losses: list[float] = []
    vols: list[float] = []
    for _ in range(n_samples):
        a = rng.uniform(a_lo, a_hi)
        f = rng.uniform(p.min_frp, p.max_frp)
        ir = rng.uniform(p.min_Irp, p.max_Irp)
        dt = rng.uniform(5.0, p.chg_tempmax)
        losses.append(total_loss(p, a, f, ir, dt))
        vols.append(total_volume_m3(p, a, f, ir))
    return max(losses), max(vols)


def optimize_weighted(
    p: DesignParameters,
    pareto_weight: float,
    loss_max: float,
    vol_max: float,
    x0: np.ndarray | None = None,
) -> ParetoPoint:
    """Minimise ``w * P/loss_max + (1-w) * V/vol_max`` subject to box constraints."""

    w = float(np.clip(pareto_weight, 0.0, 1.0))
    if w >= 1.0:
        w = 1.0 - 1e-9
    bounds = [
        (0.35 * p.area_swref, 4.5 * p.area_swref),
        (p.min_frp, p.max_frp),
        (p.min_Irp, p.max_Irp),
        (1.0, p.chg_tempmax),
    ]
    if x0 is None:
        x0 = np.array(
            [
                p.area_swref,
                0.5 * (p.min_frp + p.max_frp),
                0.5 * (p.min_Irp + p.max_Irp),
                0.5 * p.chg_tempmax,
            ],
            dtype=float,
        )

    def objective(x: np.ndarray) -> float:
        a, f, ir, dt = _pack(x)
        pl = total_loss(p, a, f, ir, dt)
        vol = total_volume_m3(p, a, f, ir)
        return w * pl / loss_max + (1.0 - w) * vol / vol_max

    res = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        options={"maxiter": 400, "ftol": 1e-9},
    )
    if not res.success:
        # Fall back to best-effort
        pass
    a, f, ir, dt = _pack(res.x)
    (ps, pi, pc), (vs, vi, vc) = component_breakdown(p, a, f, ir, dt)
    return ParetoPoint(
        pareto_weight=w,
        area_sw=a,
        f_sw=f,
        i_ripple_pp=ir,
        delta_t_j=dt,
        p_total=total_loss(p, a, f, ir, dt),
        v_total_m3=total_volume_m3(p, a, f, ir),
        p_semiconductor=ps,
        p_inductor=pi,
        p_capacitor=pc,
        v_switch_m3=vs,
        v_inductor_m3=vi,
        v_capacitor_m3=vc,
    )


def pareto_sweep(
    p: DesignParameters | None = None,
    weights: np.ndarray | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> list[ParetoPoint]:
    """Replicate ``for pareto_optimal = 0:0.05:1`` from MATLAB."""

    p = p or DesignParameters()
    lm, vm = compute_normalisation_bounds(p)
    if weights is None:
        wts = np.linspace(0.0, 1.0, 21)
    else:
        wts = np.asarray(weights, dtype=float)
    out: list[ParetoPoint] = []
    x_prev: np.ndarray | None = None
    for i, w in enumerate(wts):
        if progress:
            progress(i + 1, len(wts))
        ww = float(w)
        if ww >= 1.0:
            ww = 0.99999
        po = optimize_weighted(p, ww, lm, vm, x0=x_prev)
        x_prev = np.array([po.area_sw, po.f_sw, po.i_ripple_pp, po.delta_t_j])
        out.append(po)
    return out


def efficiency_and_power_density(p: DesignParameters, point: ParetoPoint):
    """Return ``(efficiency_pct, power_density_kw_per_dm3)`` as in MATLAB plots."""

    pout = p.p_out_nominal - point.p_total
    eff = 100.0 * pout / p.p_out_nominal
    v_dm3 = point.v_total_m3 * 1000.0  # 1 m³ = 1000 dm³
    rho = (pout / 1000.0) / max(v_dm3, 1e-18)
    return eff, rho
