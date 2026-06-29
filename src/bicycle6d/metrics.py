"""Metrics for the dynamic-bicycle DOB-CLF-ECBF-QP experiment."""

from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path

import numpy as np


STEER_CONTROL_VARIATION_SCALE = 12.0


@dataclass(frozen=True)
class MethodMetrics:
    name: str
    min_h: float
    violation_pct: float
    min_psi1: float
    final_x: float
    final_lane_error: float
    completion_margin: float
    control_energy: float
    control_variation: float
    mean_abs_steer: float
    mean_clf_value: float
    max_clf_value: float
    clf_slack_integral: float
    mean_clf_slack: float
    max_clf_slack: float
    active_clf_slack_pct: float
    mean_rho: float
    max_rho: float
    mean_beta_h_bound: float
    max_beta_h_bound: float
    min_qp_margin: float
    max_observer_error: float
    max_error_bound: float
    error_bound_violation_pct: float
    infeasible_count: int
    active_ecbf_pct: float
    completed: bool
    completion_time: float


def completion_index(result: dict, finish_x: float) -> int | None:
    completed = np.flatnonzero(result["x"] >= finish_x)
    if completed.size == 0:
        return None
    return int(completed[0])


def compute_metrics(result: dict, dt: float, finish_x: float) -> MethodMetrics:
    finish_idx = completion_index(result, finish_x)
    stop = finish_idx + 1 if finish_idx is not None else len(result["t"])
    h = result["h"][:stop]
    psi1 = result["psi1"][:stop]
    rho_raw = result["rho"][:stop]
    bar_e_raw = result["bar_e"][:stop]
    beta_h_raw = result["b_h_bound"][:stop]
    rho = np.max(rho_raw, axis=1) if rho_raw.ndim > 1 else rho_raw
    bar_e = np.max(bar_e_raw, axis=1) if bar_e_raw.ndim > 1 else bar_e_raw
    beta_h_bound = np.max(beta_h_raw, axis=1) if beta_h_raw.ndim > 1 else beta_h_raw
    u = np.column_stack([result["u_ax"][:stop], result["u_delta"][:stop]])
    scaled_u = np.column_stack(
        [
            result["u_ax"][:stop],
            STEER_CONTROL_VARIATION_SCALE * result["u_delta"][:stop],
        ]
    )
    du = np.linalg.norm(np.diff(scaled_u, axis=0), axis=1)
    clf_value = result.get("clf_value", np.zeros(stop, dtype=float))[:stop]
    clf_slack = result.get("clf_slack", np.zeros(stop, dtype=float))[:stop]
    error = np.abs(result["b_true"][:stop] - result["b_hat"][:stop])
    error_plot = np.max(error, axis=1) if error.ndim > 1 else error
    has_error_certificate = bool(np.max(bar_e) > 0.0)
    bound_violation_pct = (
        float(np.mean(error_plot > bar_e + 1e-7) * 100.0)
        if has_error_certificate
        else float("nan")
    )
    final_x = float(result["x"][stop - 1])
    completion_margin = final_x - finish_x
    return MethodMetrics(
        name=result["name"],
        min_h=float(np.min(h)),
        violation_pct=float(np.mean(h < 0.0) * 100.0),
        min_psi1=float(np.min(psi1)),
        final_x=final_x,
        final_lane_error=float(abs(result["y"][stop - 1])),
        completion_margin=float(completion_margin),
        control_energy=float(np.sum(np.sum(u**2, axis=1)) * dt),
        control_variation=float(np.sum(du)),
        mean_abs_steer=float(np.mean(np.abs(result["u_delta"][:stop]))),
        mean_clf_value=float(np.mean(clf_value)),
        max_clf_value=float(np.max(clf_value)),
        clf_slack_integral=float(np.sum(clf_slack) * dt),
        mean_clf_slack=float(np.mean(clf_slack)),
        max_clf_slack=float(np.max(clf_slack)),
        active_clf_slack_pct=float(np.mean(clf_slack > 1e-6) * 100.0),
        mean_rho=float(np.mean(rho)),
        max_rho=float(np.max(rho)),
        mean_beta_h_bound=float(np.mean(beta_h_bound)),
        max_beta_h_bound=float(np.max(beta_h_bound)),
        min_qp_margin=float(np.min(result["lhs_nominal"][:stop])),
        max_observer_error=float(np.max(error_plot)),
        max_error_bound=float(np.max(bar_e)),
        error_bound_violation_pct=bound_violation_pct,
        infeasible_count=int(result["infeasible_count"]),
        active_ecbf_pct=float(np.mean(result["ecbf_active"][:stop]) * 100.0),
        completed=bool(completion_margin >= 0.0),
        completion_time=float(result["t"][finish_idx]) if finish_idx is not None else float("nan"),
    )


def write_summary(metrics: list[MethodMetrics], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(MethodMetrics.__dataclass_fields__.keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for item in metrics:
            writer.writerow({field: getattr(item, field) for field in fields})
