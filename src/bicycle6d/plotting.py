"""Plots for the dynamic-bicycle DOB-CLF-ECBF-QP experiment."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import ExperimentConfig
from .metrics import MethodMetrics, completion_index


COLORS = {
    "nominal": "#1f77b4",
    "dob_cbf": "#2ca02c",
    "tunable_dob_cbf": "#9467bd",
    "state_dob_cbf": "#d95f02",
}

LINESTYLES = {
    "nominal": (0, (4.0, 2.0)),
    "dob_cbf": "-",
    "tunable_dob_cbf": (0, (3.0, 1.6)),
    "state_dob_cbf": (0, (5.0, 2.0, 1.2, 2.0)),
}

SHORT_LABELS = {
    "nominal": "Nominal CLF-ECBF-QP",
    "dob_cbf": "DOB-CLF-ECBF-QP",
    "tunable_dob_cbf": "Tunable DOB-CLF-ECBF-QP",
    "state_dob_cbf": "State-dependent DOB-CLF-ECBF-QP",
}

BAR_LABELS = {
    "nominal": "Nominal\nCLF-ECBF",
    "dob_cbf": "DOB\nCLF-ECBF",
    "tunable_dob_cbf": "Tunable\nDOB-CLF",
    "state_dob_cbf": "State-dep.\nDOB-CLF",
}

plt.rcParams.update(
    {
        "font.size": 13,
        "font.weight": "medium",
        "axes.titlesize": 15,
        "axes.titleweight": "semibold",
        "axes.labelsize": 13,
        "axes.labelweight": "medium",
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "figure.titlesize": 15,
        "text.color": "#111111",
        "axes.labelcolor": "#111111",
        "axes.edgecolor": "#111111",
        "xtick.color": "#111111",
        "ytick.color": "#111111",
    }
)


def _ordered_keys(results: dict[str, dict]) -> list[str]:
    return [key for key in COLORS if key in results]


def _completion_stop(res: dict, cfg: ExperimentConfig) -> int:
    idx = completion_index(res, cfg.bicycle.finish_x)
    return idx + 1 if idx is not None else len(res["t"])


def _trajectory_slice(res: dict, cfg: ExperimentConfig) -> slice:
    return slice(None, _completion_stop(res, cfg))


def plot_trajectory(results: dict[str, dict], cfg: ExperimentConfig, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    car = cfg.bicycle
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    theta = np.linspace(0.0, 2.0 * np.pi, 256)
    ax.axhline(car.lane_half_width, color="#555555", linestyle="--", linewidth=1.0, alpha=0.55)
    ax.axhline(-car.lane_half_width, color="#555555", linestyle="--", linewidth=1.0, alpha=0.55)
    for obs0, obs_v, axes in zip(car.obstacles, car.obstacle_vels, car.axes):
        center = obs0
        ax.fill(
            center[0] + axes[0] * np.cos(theta),
            center[1] + axes[1] * np.sin(theta),
            color="#222222",
            alpha=0.18,
        )
        if np.linalg.norm(obs_v) > 0.0:
            future = obs0 + obs_v * 4.2
            ax.plot([obs0[0], future[0]], [obs0[1], future[1]], color="#222222", alpha=0.25)
    for key in _ordered_keys(results):
        res = results[key]
        trajectory_slice = _trajectory_slice(res, cfg)
        ax.plot(
            res["x"][trajectory_slice],
            res["y"][trajectory_slice],
            color=COLORS[key],
            linestyle=LINESTYLES[key],
            linewidth=1.6,
            label=SHORT_LABELS[key],
        )
    ax.axvline(car.finish_x, color="black", linewidth=1.0, linestyle=":", alpha=0.75)
    ax.set_title("Dynamic-bicycle trajectories", fontsize=9.5, pad=3)
    ax.set_xlabel("global X (m)", fontsize=8.5, labelpad=2)
    ax.set_ylabel("global Y (m)", fontsize=8.5, labelpad=0)
    ax.tick_params(axis="both", labelsize=8, pad=1)
    # Road-scale trajectories are easier to compare with an expanded lateral axis.
    # The safety metrics still use the true X/Y coordinates and elliptic barriers.
    all_x = np.concatenate([res["x"][_trajectory_slice(res, cfg)] for res in results.values()])
    all_y = np.concatenate([res["y"][_trajectory_slice(res, cfg)] for res in results.values()])
    x_min = min(float(np.min(all_x)), float(np.min(car.obstacles[:, 0] - car.axes[:, 0])))
    x_max = max(float(np.max(all_x)), float(np.max(car.obstacles[:, 0] + car.axes[:, 0])))
    y_min = min(float(np.min(all_y)), -car.lane_half_width, float(np.min(car.obstacles[:, 1] - car.axes[:, 1])))
    y_max = max(float(np.max(all_y)), car.lane_half_width, float(np.max(car.obstacles[:, 1] + car.axes[:, 1])))
    ax.set_xlim(x_min - 2.0, x_max + 2.0)
    ax.set_ylim(y_min - 0.4, y_max + 0.4)
    ax.grid(True, alpha=0.24)
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.005, 0.005),
        fontsize=6.2,
        frameon=False,
        borderpad=0.0,
        handlelength=1.55,
        handletextpad=0.4,
        labelspacing=0.22,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_safety(results: dict[str, dict], cfg: ExperimentConfig, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(11.6, 7.8), sharex=True)
    for axis in axes:
        axis.axhline(0.0, color="black", linewidth=1.0)
        axis.grid(True, alpha=0.24)
    for key in _ordered_keys(results):
        res = results[key]
        stop = min(_completion_stop(res, cfg), int(cfg.bicycle.safety_plot_horizon / cfg.bicycle.dt))
        t = res["t"][:stop]
        axes[0].plot(
            t,
            res["h"][:stop],
            color=COLORS[key],
            linestyle=LINESTYLES[key],
            linewidth=2.0,
            label=SHORT_LABELS[key],
        )
        axes[1].plot(
            t,
            res["psi1"][:stop],
            color=COLORS[key],
            linestyle=LINESTYLES[key],
            linewidth=2.0,
            label=SHORT_LABELS[key],
        )
    axes[0].set_title("Minimum safety value under relaxed CLF-CBF-QP")
    axes[1].set_title("Minimum first ECBF auxiliary state")
    axes[0].set_ylabel("min h")
    axes[1].set_ylabel(r"min $\psi_1$")
    axes[1].set_xlabel("time (s)")
    axes[0].legend(ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def plot_dob_estimation(results: dict[str, dict], cfg: ExperimentConfig, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(11.6, 7.8), sharex=True)
    for key in _ordered_keys(results):
        res = results[key]
        if np.max(res["bar_e"]) <= 0.0:
            continue
        stop = _completion_stop(res, cfg)
        t = res["t"][:stop]
        error = np.max(np.abs(res["b_true"][:stop] - res["b_hat"][:stop]), axis=1)
        ebar = np.max(res["bar_e"][:stop], axis=1)
        rho = np.max(res["rho"][:stop], axis=1)
        axes[0].plot(t, error, color=COLORS[key], linestyle=":", linewidth=1.5, alpha=0.85)
        axes[0].plot(
            t,
            ebar,
            color=COLORS[key],
            linestyle=LINESTYLES[key],
            linewidth=2.0,
            label=SHORT_LABELS[key],
        )
        axes[1].plot(
            t,
            rho,
            color=COLORS[key],
            linestyle=LINESTYLES[key],
            linewidth=2.0,
            label=SHORT_LABELS[key],
        )
    axes[0].set_title("Maximum DOB channel error and certified bound")
    axes[1].set_title("Maximum robust ECBF margin")
    axes[0].set_ylabel(r"$|\beta-\hat{\beta}|$, $\bar e$")
    axes[1].set_ylabel(r"$\rho$")
    axes[1].set_xlabel("time (s)")
    for axis in axes:
        axis.grid(True, alpha=0.24)
        axis.legend(ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def plot_clf_performance(results: dict[str, dict], cfg: ExperimentConfig, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(11.6, 7.8), sharex=True)
    for key in _ordered_keys(results):
        res = results[key]
        if "clf_value" not in res or "clf_slack" not in res:
            continue
        stop = _completion_stop(res, cfg)
        t = res["t"][:stop]
        axes[0].plot(
            t,
            res["clf_value"][:stop],
            color=COLORS[key],
            linestyle=LINESTYLES[key],
            linewidth=2.0,
            label=SHORT_LABELS[key],
        )
        axes[1].plot(
            t,
            res["clf_slack"][:stop],
            color=COLORS[key],
            linestyle=LINESTYLES[key],
            linewidth=2.0,
            label=SHORT_LABELS[key],
        )
    axes[0].set_title("Tracking CLF value in the relaxed CLF-CBF-QP")
    axes[1].set_title("CLF relaxation required by hard safety constraints")
    axes[0].set_ylabel(r"$V(x)$")
    axes[1].set_ylabel(r"$s_{\mathrm{clf}}$")
    axes[1].set_xlabel("time (s)")
    for axis in axes:
        axis.grid(True, alpha=0.24)
        axis.legend(ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def plot_metric_bars(metrics: list[MethodMetrics], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    key_by_label = {label: key for key, label in SHORT_LABELS.items()}
    ordered = [(key_by_label[item.name], item) for item in metrics if item.name in key_by_label]
    x = np.arange(len(ordered))
    labels = [BAR_LABELS[key] for key, _ in ordered]
    colors = [COLORS[key] for key, _ in ordered]
    panels = [
        ([item.mean_rho for _, item in ordered], r"mean $\rho$", "Mean robust margin"),
        ([item.clf_slack_integral for _, item in ordered], r"$\int s_{\mathrm{clf}}\,dt$", "Integrated CLF relaxation"),
        ([item.final_lane_error for _, item in ordered], "|final y| (m)", "Final lane error"),
        ([item.control_energy for _, item in ordered], r"$\int \|u\|^2 dt$", "Control energy"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 8.0))
    for axis, (values, ylabel, title) in zip(axes.flat, panels):
        axis.bar(x, values, color=colors, edgecolor="white", linewidth=0.8)
        axis.set_xticks(x, labels)
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(True, axis="y", alpha=0.24)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(axis="x", labelsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240)
    plt.close(fig)
