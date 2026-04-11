"""Command-line entry points."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from joblib import dump
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler

from .optimization import efficiency_and_power_density, pareto_sweep
from .parameters import DesignParameters


def main_pareto() -> None:
    ap = argparse.ArgumentParser(description="Pareto sweep (MATLAB Step2 analogue).")
    ap.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for PNG figures",
    )
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    p = DesignParameters()
    pts = pareto_sweep(p)
    eff = np.array([efficiency_and_power_density(p, x)[0] for x in pts])
    rho = np.array([efficiency_and_power_density(p, x)[1] for x in pts])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(rho, eff, "--*", linewidth=2, markersize=6)
    ax.set_xlabel(r"Power density (kW/dm$^{3}$)", fontsize=12)
    ax.set_ylabel("Efficiency (%)", fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p_eff = args.output_dir / "pareto_efficiency_vs_power_density.png"
    fig.savefig(p_eff, dpi=150)
    print(f"Wrote {p_eff}")

    idx = min(5, len(pts) - 1)
    z = pts[idx]
    v_cm3 = 1e6 * np.array([z.v_switch_m3, z.v_inductor_m3, z.v_capacitor_m3])
    fig2, ax2 = plt.subplots(figsize=(5, 5))
    ax2.pie(
        v_cm3,
        labels=[
            f"Switches ({v_cm3[0]:.1f} cm³)",
            f"Inductors ({v_cm3[1]:.1f} cm³)",
            f"Capacitors ({v_cm3[2]:.1f} cm³)",
        ],
        autopct="%1.1f%%",
    )
    ax2.set_title(f"Volume {1e6 * z.v_total_m3:.1f} cm³")
    p_pie_v = args.output_dir / "pie_volume.png"
    fig2.savefig(p_pie_v, dpi=150)
    print(f"Wrote {p_pie_v}")

    l = np.array([z.p_semiconductor, z.p_inductor, z.p_capacitor])
    fig3, ax3 = plt.subplots(figsize=(5, 5))
    ax3.pie(
        l,
        labels=[
            f"Semiconductors ({l[0]:.2f} W)",
            f"Inductors ({l[1]:.2f} W)",
            f"Capacitors ({l[2]:.2f} W)",
        ],
        autopct="%1.1f%%",
    )
    ax3.set_title(f"Power loss {z.p_total:.2f} W")
    p_pie_l = args.output_dir / "pie_loss.png"
    fig3.savefig(p_pie_l, dpi=150)
    print(f"Wrote {p_pie_l}")


def main_train_ann() -> None:
    ap = argparse.ArgumentParser(description="Train ANN surrogate (MATLAB Step3 analogue).")
    ap.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("models"),
        help="Where to save netdata.npz and ann.joblib",
    )
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    p = DesignParameters()
    pts = pareto_sweep(p)
    x = np.array(
        [[pt.area_sw, pt.f_sw, pt.i_ripple_pp, pt.delta_t_j] for pt in pts],
        dtype=float,
    )
    y = np.array(
        [[1e9 * pt.v_total_m3, pt.p_total] for pt in pts],
        dtype=float,
    )

    np.savez(
        args.output_dir / "netdata.npz",
        Asw_designs=x[:, 0],
        fripple_designs=x[:, 1],
        Iripple_designs=x[:, 2],
        deltaT_designs=x[:, 3],
        vol_designs=y[:, 0],
        loss_designs=y[:, 1],
    )
    print(f"Wrote {args.output_dir / 'netdata.npz'}")

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    xt = scaler_x.fit_transform(x)
    yt = scaler_y.fit_transform(y)
    mlp = MLPRegressor(
        hidden_layer_sizes=(15, 12, 10),
        activation="relu",
        solver="adam",
        alpha=0.01,
        max_iter=8000,
        random_state=42,
        tol=1e-4,
    )
    mlp.fit(xt, yt)
    dump(
        {"mlp": mlp, "scaler_x": scaler_x, "scaler_y": scaler_y},
        args.output_dir / "ann.joblib",
    )
    print(f"Wrote {args.output_dir / 'ann.joblib'}")
