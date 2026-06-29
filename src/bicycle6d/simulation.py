"""Simulation loop for the dynamic-bicycle DOB-CLF-ECBF-QP experiment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ExperimentConfig, global_channel_lipschitz_bounds
from .controller import (
    barrier_dots,
    barriers,
    constraint_labels,
    disturbance_coefficients,
    ecbf_filter,
    inertial_velocity,
    known_h_ddots,
    local_channel_lipschitz_bounds,
    local_initial_error_bounds,
    split_state,
    state_derivative_known,
)
from .observer import RelativeDegreeTwoDOB, StaticEstimator


@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str
    kind: str


METHODS: list[MethodSpec] = [
    MethodSpec("nominal", "Nominal CLF-ECBF-QP", "nominal"),
    MethodSpec("dob_cbf", "DOB-CLF-ECBF-QP", "dob_cbf"),
    MethodSpec("tunable_dob_cbf", "Tunable DOB-CLF-ECBF-QP", "tunable_dob_cbf"),
    MethodSpec("state_dob_cbf", "State-dependent DOB-CLF-ECBF-QP", "state_dob_cbf"),
]


class EstimatorBank:
    """One scalar DOB per highest-order safety channel."""

    def __init__(self, estimators: list[RelativeDegreeTwoDOB | StaticEstimator]) -> None:
        self.estimators = estimators

    def _require_length(self, values: np.ndarray, name: str) -> None:
        if len(values) != len(self.estimators):
            raise ValueError(
                f"{name} has length {len(values)}, expected {len(self.estimators)}"
            )

    @property
    def b_hat(self) -> np.ndarray:
        return np.array([estimator.b_hat for estimator in self.estimators], dtype=float)

    @property
    def rho(self) -> np.ndarray:
        return np.array([estimator.rho for estimator in self.estimators], dtype=float)

    @property
    def bar_e(self) -> np.ndarray:
        return np.array([estimator.bar_e for estimator in self.estimators], dtype=float)

    def initialize(self, h_dot_initial: np.ndarray) -> None:
        self._require_length(h_dot_initial, "h_dot_initial")
        for estimator, h_dot in zip(self.estimators, h_dot_initial):
            estimator.initialize(float(h_dot))

    def flow_update(
        self,
        h_dot_next: np.ndarray,
        known_h_ddot: np.ndarray,
        dt: float,
        b_h_bounds: np.ndarray | None = None,
    ) -> None:
        self._require_length(h_dot_next, "h_dot_next")
        self._require_length(known_h_ddot, "known_h_ddot")
        if b_h_bounds is None:
            b_h_iter = [None] * len(self.estimators)
        else:
            self._require_length(b_h_bounds, "b_h_bounds")
            b_h_iter = [float(value) for value in b_h_bounds]
        for estimator, h_dot_i, known_i, b_h_i in zip(
            self.estimators,
            h_dot_next,
            known_h_ddot,
            b_h_iter,
        ):
            estimator.flow_update(float(h_dot_i), float(known_i), dt, b_h_i)


def uses_state_dependent_certificate(kind: str) -> bool:
    return kind == "state_dob_cbf"


def method_channel_lipschitz_bounds(
    state: np.ndarray,
    t: float,
    kind: str,
    cfg: ExperimentConfig,
) -> np.ndarray:
    n_constraints = len(constraint_labels(cfg.bicycle))
    if kind == "nominal":
        return np.zeros(n_constraints, dtype=float)
    if kind in {"dob_cbf", "tunable_dob_cbf"}:
        return global_channel_lipschitz_bounds(cfg.bicycle, cfg.disturbance)
    if kind == "state_dob_cbf":
        return local_channel_lipschitz_bounds(state, t, cfg)
    raise ValueError(f"unknown method kind: {kind}")


def robust_margin(
    estimator: EstimatorBank,
    kind: str,
    psi1: np.ndarray | None = None,
    cfg: ExperimentConfig | None = None,
) -> np.ndarray:
    if kind == "nominal":
        return np.zeros_like(estimator.bar_e)
    if kind == "tunable_dob_cbf":
        if psi1 is None or cfg is None:
            raise ValueError("tunable DOB-CBF requires psi1 and config")
        psi_pos = np.maximum(np.asarray(psi1, dtype=float), 0.0)
        scale_shape = (psi_pos / (psi_pos + cfg.bicycle.tunable_margin_psi_scale)) ** 2
        scale = 1.0 + (cfg.bicycle.tunable_margin_max_scale - 1.0) * scale_shape
        return estimator.rho / np.maximum(scale, 1.0)
    if kind in {"dob_cbf", "state_dob_cbf"}:
        return estimator.rho
    raise ValueError(f"unknown method kind: {kind}")


def make_estimator(kind: str, cfg: ExperimentConfig) -> EstimatorBank:
    n_constraints = len(constraint_labels(cfg.bicycle))
    if kind == "nominal":
        return EstimatorBank([StaticEstimator() for _ in range(n_constraints)])
    obs = cfg.observer
    if kind not in {"dob_cbf", "tunable_dob_cbf", "state_dob_cbf"}:
        raise ValueError(f"unknown method kind: {kind}")
    initial_bounds = local_initial_error_bounds(cfg.bicycle.state0, 0.0, cfg)
    if kind in {"dob_cbf", "tunable_dob_cbf"}:
        channel_bounds = global_channel_lipschitz_bounds(cfg.bicycle, cfg.disturbance)
    else:
        channel_bounds = local_channel_lipschitz_bounds(cfg.bicycle.state0, 0.0, cfg)
    return EstimatorBank(
        [
            RelativeDegreeTwoDOB(
                gain=obs.gain,
                b_h_bound=float(channel_bound),
                initial_error_bound=float(initial_bound),
                epsilon=obs.epsilon,
            )
            for initial_bound, channel_bound in zip(initial_bounds, channel_bounds)
        ]
    )


def run_method(spec: MethodSpec, cfg: ExperimentConfig) -> dict:
    bicycle = cfg.bicycle
    estimator = make_estimator(spec.kind, cfg)
    n_steps = int(bicycle.horizon / bicycle.dt)
    t_values = np.arange(n_steps) * bicycle.dt
    state = bicycle.state0.copy()
    estimator.initialize(barrier_dots(state, 0.0, bicycle))
    previous_u = None
    rows: dict[str, list] = {
        "t": [],
        "x": [],
        "y": [],
        "psi": [],
        "vx": [],
        "vy": [],
        "r": [],
        "speed": [],
        "disturbance_x": [],
        "disturbance_y": [],
        "disturbance_r": [],
        "b_true": [],
        "b_hat": [],
        "bar_e": [],
        "rho": [],
        "b_h_bound": [],
        "h": [],
        "h_all": [],
        "h_dot": [],
        "psi1": [],
        "u_ax": [],
        "u_delta": [],
        "u_ref_ax": [],
        "u_ref_delta": [],
        "clf_value": [],
        "clf_slack": [],
        "clf_residual": [],
        "rhs": [],
        "lhs_nominal": [],
        "ecbf_active": [],
        "feasible": [],
    }
    infeasible_count = 0
    for t in t_values:
        x_pos, y_pos, psi, vx, vy, yaw_rate = split_state(state)
        disturbance = cfg.disturbance.value(float(t), x_pos)
        q = disturbance_coefficients(state, float(t), bicycle)
        b_true = q @ disturbance[:2]
        current_b_h = method_channel_lipschitz_bounds(state, float(t), spec.kind, cfg)
        h_all = barriers(state, float(t), bicycle)
        h_dot = barrier_dots(state, float(t), bicycle)
        psi1_all = h_dot + bicycle.lambda_1 * h_all
        rho = robust_margin(estimator, spec.kind, psi1_all, cfg)
        control = ecbf_filter(state, float(t), estimator.b_hat, rho, bicycle, previous_u)
        if not control.feasible:
            infeasible_count += 1
        rows["t"].append(float(t))
        rows["x"].append(x_pos)
        rows["y"].append(y_pos)
        rows["psi"].append(psi)
        rows["vx"].append(vx)
        rows["vy"].append(vy)
        rows["r"].append(yaw_rate)
        rows["speed"].append(float(np.linalg.norm(inertial_velocity(state))))
        rows["disturbance_x"].append(float(disturbance[0]))
        rows["disturbance_y"].append(float(disturbance[1]))
        rows["disturbance_r"].append(float(disturbance[2]))
        rows["b_true"].append(b_true)
        rows["b_hat"].append(estimator.b_hat)
        rows["bar_e"].append(estimator.bar_e)
        rows["rho"].append(rho)
        rows["b_h_bound"].append(current_b_h)
        rows["h"].append(float(np.min(h_all)))
        rows["h_all"].append(h_all)
        rows["h_dot"].append(h_dot)
        rows["psi1"].append(float(np.min(psi1_all)))
        rows["u_ax"].append(float(control.u[0]))
        rows["u_delta"].append(float(control.u[1]))
        rows["u_ref_ax"].append(float(control.u_ref[0]))
        rows["u_ref_delta"].append(float(control.u_ref[1]))
        rows["clf_value"].append(float(control.clf_value))
        rows["clf_slack"].append(float(control.clf_slack))
        rows["clf_residual"].append(float(control.clf_residual))
        rows["rhs"].append(control.rhs)
        rows["lhs_nominal"].append(float(control.lhs_nominal))
        rows["ecbf_active"].append(control.active)
        rows["feasible"].append(control.feasible)

        known = known_h_ddots(state, control.u, float(t), bicycle)
        state_dot = state_derivative_known(state, control.u, bicycle)
        state_dot[3:6] += disturbance
        next_state = state + bicycle.dt * state_dot
        next_t = float(t + bicycle.dt)
        next_b_h = method_channel_lipschitz_bounds(next_state, next_t, spec.kind, cfg)
        interval_b_h = np.maximum(current_b_h, next_b_h)
        estimator.flow_update(
            barrier_dots(next_state, next_t, bicycle),
            known,
            bicycle.dt,
            interval_b_h if uses_state_dependent_certificate(spec.kind) else None,
        )
        previous_u = control.u
        state = next_state
    result = {key: np.array(value) for key, value in rows.items()}
    result["name"] = spec.label
    result["key"] = spec.key
    result["constraint_labels"] = constraint_labels(bicycle)
    result["infeasible_count"] = infeasible_count
    return result


def run_all(cfg: ExperimentConfig) -> dict[str, dict]:
    return {spec.key: run_method(spec, cfg) for spec in METHODS}
