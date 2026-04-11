"""Semi-analytical loss and volume models for the buck VSI design variables.

The original repository omitted ``cvx_initialization`` / ``cvx_optimization``. This module
implements a consistent physics-based decomposition into semiconductor, inductor, and
capacitor contributions so the Python pipeline can run end-to-end.

Decision vector ``x = [A_sw, f_sw, I_ripple, deltaT]`` (SI units) matches the ANN inputs
in MATLAB Step3.
"""

from __future__ import annotations

import numpy as np

from .parameters import DesignParameters


def inductance_buck(p: DesignParameters, f_sw: float, i_ripple_pp: float) -> float:
    """Average inductance [H] for peak–peak ripple ``i_ripple_pp`` at frequency ``f_sw``."""
    d = p.duty
    return p.out_vge * (1.0 - d) / (f_sw * max(i_ripple_pp, 1e-9))


def output_capacitance(p: DesignParameters, f_sw: float, i_ripple_pp: float) -> float:
    """Output capacitor [F] for voltage ripple ``del_vge_out`` (triangle ripple model)."""
    return i_ripple_pp / (8.0 * f_sw * max(p.del_vge_out, 1e-12))


def semiconductor_loss(
    p: DesignParameters,
    area_sw: float,
    f_sw: float,
    i_ripple_pp: float,
    delta_t_j: float,
) -> float:
    """Total semiconductor loss [W] for ``n`` parallel legs worth of area ``area_sw`` each."""

    n = float(p.num_switches_per_leg)
    a = max(area_sw, 1e-12)
    ar = p.area_swref / a
    io = p.out_ct
    d = p.duty
    # RMS inductor current (triangular ripple on DC)
    i_l_rms_sq = io * io + (i_ripple_pp**2) / 12.0
    i_l_rms = np.sqrt(max(i_l_rms_sq, 0.0))
    # Buck high-side / low-side RMS (first-harmonic approximation)
    i_hs_rms = i_l_rms * np.sqrt(d)
    i_ls_rms = i_l_rms * np.sqrt(1.0 - d)
    rds = p.rdson_area_swref / a
    temp_factor = 1.0 + p.alpha_val * delta_t_j
    p_cond = n * rds * temp_factor * (i_hs_rms**2 + i_ls_rms**2)

    vin = p.in_vge
    # Switching energy scales ref−1 with area (faster edge, smaller die) — first order
    e_sw_cycle = (
        0.5 * p.ton_toff * io * (vin + p.frd_vge)
        + 0.25 * p.toff_ton * (vin + p.frd_vge)
    )
    p_sw = n * f_sw * e_sw_cycle * ar
    coss = p.cossref_over_area * a
    p_coss = n * f_sw * 0.5 * coss * vin**2
    # Reverse recovery and gate drive losses scale ~ f_sw
    p_rr = n * f_sw * p.qrrref_over_area * a * vin
    p_g = n * f_sw * 2.0 * p.qgref_over_area * a * p.vge_gs

    return float(p_cond + p_sw + p_coss + p_rr + p_g)


def inductor_loss(
    p: DesignParameters,
    f_sw: float,
    i_ripple_pp: float,
) -> float:
    """Inductor copper + core loss aggregate [W] (MATLAB coefficient structure).

    Cross-terms are normalised by the ripple current span and a 100 kHz frequency
    reference so coefficients remain O(1) when ``f_sw`` is expressed in hertz.
    """

    l = inductance_buck(p, f_sw, i_ripple_pp)
    io = p.out_ct
    p3 = p.pind3_loss_coeff * (io**2)
    ir_n = i_ripple_pp / max(io, 1e-9)
    f_n = f_sw / 100e3
    pl = (
        p.pind1_loss
        + p.pind2_loss * f_sw
        + p3
        + p.pind_coe1 * ir_n
        + p.pind_coe2 * (ir_n**2)
        + p.pind_coe3 * f_n * ir_n
        + p.pind_coe4 * (l / (l + 2e-3))
    )
    return float(max(pl, 0.0))


def capacitor_loss(p: DesignParameters, i_ripple_pp: float) -> float:
    """Output capacitor ESR + dielectric loss [W] (simplified from MATLAB comments)."""
    p_cap_coeff = 3e2 * ((p.tan_del * 8.0 * p.del_vge_out) / (24.0 * np.pi))
    return float(p_cap_coeff * i_ripple_pp)


def switch_volume_m3(p: DesignParameters, area_sw: float) -> float:
    """Semiconductor + package volume [m³].

    Calibrated so that at ``area_sw == area_swref`` and ``n = 2`` the aggregate matches
    the ~74 cm³ switch share in the reference MATLAB pie charts.
    """

    n = float(p.num_switches_per_leg)
    v_per_device_ref_m3 = 37e-6  # 37 cm³ per active switch at the reference die area
    scale = area_sw / p.area_swref
    return n * v_per_device_ref_m3 * scale


def inductor_volume_m3(p: DesignParameters, f_sw: float, i_ripple_pp: float) -> float:
    """Magnetic component volume [m³].

    Posynomial-style scaling ``1/(f · Iripple)`` with a reference operating point taken
    near the middle of the feasible box (150 kHz, ~1 A ripple).
    """

    f_ref = 150e3
    ir_ref = 1.0
    v_ref_m3 = 72e-6  # 72 cm³ at the reference point (MATLAB pie chart scale)
    return float(v_ref_m3 * (f_ref * ir_ref) / max(f_sw * i_ripple_pp, 1e-9))


def capacitor_volume_m3(p: DesignParameters, f_sw: float, i_ripple_pp: float) -> float:
    """DC-link / output capacitor bank volume [m³]."""

    c_out = output_capacitance(p, f_sw, i_ripple_pp)
    c_ref = output_capacitance(p, 150e3, 1.0)
    v_ref_m3 = 69e-6  # 69 cm³ at the reference point (MATLAB pie chart scale)
    return float(v_ref_m3 * (c_out / max(c_ref, 1e-18)))


def total_loss(
    p: DesignParameters,
    area_sw: float,
    f_sw: float,
    i_ripple_pp: float,
    delta_t_j: float,
) -> float:
    return (
        semiconductor_loss(p, area_sw, f_sw, i_ripple_pp, delta_t_j)
        + inductor_loss(p, f_sw, i_ripple_pp)
        + capacitor_loss(p, i_ripple_pp)
    )


def total_volume_m3(
    p: DesignParameters,
    area_sw: float,
    f_sw: float,
    i_ripple_pp: float,
) -> float:
    v = (
        switch_volume_m3(p, area_sw)
        + inductor_volume_m3(p, f_sw, i_ripple_pp)
        + capacitor_volume_m3(p, f_sw, i_ripple_pp)
    )
    if p.include_heat_sink_volume:
        v += p.heat_sink_vol_m3
    return v


def component_breakdown(
    p: DesignParameters,
    area_sw: float,
    f_sw: float,
    i_ripple_pp: float,
    delta_t_j: float,
):
    """Return (P_sem, P_ind, P_cap) and (V_sw, V_ind, V_cap) for reporting."""
    ps = semiconductor_loss(p, area_sw, f_sw, i_ripple_pp, delta_t_j)
    pi = inductor_loss(p, f_sw, i_ripple_pp)
    pc = capacitor_loss(p, i_ripple_pp)
    vs = switch_volume_m3(p, area_sw)
    vi = inductor_volume_m3(p, f_sw, i_ripple_pp)
    vc = capacitor_volume_m3(p, f_sw, i_ripple_pp)
    return (ps, pi, pc), (vs, vi, vc)
