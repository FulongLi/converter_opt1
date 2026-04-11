"""Design parameters transcribed from the MATLAB Step2 script (Nov 2024)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DesignParameters:
    """Single-phase buck-type VSI case (430 V → 230 V, ~1 kW)."""

    # Topology / switches
    num_switches_per_leg: int = 2
    min_sws: int = 1
    max_sws: int = 2
    vge_switch: int = 900  # V rating class

    # Device data @ reference (900 V class block in MATLAB)
    h: float = 0.02879  # m, package height scale
    rthja_ref: float = 150.0  # K/W (junction–ambient ref)
    rdson_area_swref: float = 9.348e-6  # Ω·m² (Rds_on × area at ref)
    cossref_over_area: float = 8.9912e-7  # F/m²
    qgref_over_area: float = 2.1930e-4  # C/m²
    qrrref_over_area: float = 1.1e-3  # C/m²
    alpha_val: float = 0.003636364  # 1/K, Rds temp coefficient
    area_swref: float = 4.56e-5  # m², reference die area
    ton_toff: float = 5.93e-9  # s, combined switching time scale
    toff_ton: float = 3.42e-10  # s
    frd_vge: float = 0.8
    volume_resistance: float = 80.0  # thermal model constant (see thesis)
    r_theta_sa: float = 0.1  # K/W, sink to ambient (used with volume_resistance)

    # Electrical operating point
    in_vge: float = 430.0
    out_vge: float = 230.0
    out_ct: float = 4.35  # A, DC output current
    amb_temp: float = 25.0
    max_jn_temp: float = 100.0
    out_rp: float = 0.0158  # % output voltage ripple (cap)
    vf: float = 1.2  # PCB / packing factor on volumes
    min_rp: float = 10.0  # % min inductor current ripple
    max_rp: float = 50.0  # % max inductor current ripple
    min_frp: float = 50e3  # Hz
    max_frp: float = 300e3  # Hz
    vge_gs: float = 18.0  # V gate drive

    # Inductor loss polynomial / limit (MATLAB)
    pind1_loss: float = 0.02401
    pind2_loss: float = 6.381e-10
    pind3_loss_coeff: float = 0.002242  # multiplies out_ct**2 in MATLAB
    pind_coe1: float = 0.1302
    pind_coe2: float = 0.06675
    pind_coe3: float = 0.2853
    pind_coe4: float = 2.774
    indct_loss_max: float = 0.5  # W, cap used in original model

    tan_del: float = 0.02  # cap loss tangent proxy

    # Geometry coefficient r (MATLAB eq. 34)
    r_geom: float = 2.443750e-1

    # Output power for efficiency (MATLAB uses 1 kW nominal)
    p_out_nominal: float = 1000.0  # W
    # MATLAB pie charts sum only switch / inductor / capacitor volumes
    include_heat_sink_volume: bool = False

    @property
    def duty(self) -> float:
        return self.out_vge / self.in_vge

    @property
    def chg_tempmax(self) -> float:
        return self.max_jn_temp - self.amb_temp

    @property
    def del_vge_out(self) -> float:
        return self.out_vge * (self.out_rp / 100.0)

    @property
    def min_Irp(self) -> float:
        return (self.min_rp / 100.0) * self.out_ct

    @property
    def max_Irp(self) -> float:
        return (self.max_rp / 100.0) * self.out_ct

    @property
    def heat_sink_vol_m3(self) -> float:
        """Raw thermal volume constant from MATLAB (often used as constraint, not in pie charts)."""
        return self.volume_resistance / self.r_theta_sa

    @property
    def temp_rise_coeff(self) -> float:
        """MATLAB: temp_rise = Rthja_ref * area_swref / 2."""
        return self.rthja_ref * self.area_swref / 2.0
