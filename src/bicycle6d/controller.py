"""Reference-tracking ECBF-QP controller for a disturbed dynamic bicycle."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import numpy as np
import osqp
from scipy.linalg import solve_continuous_are
from scipy import sparse

from .config import BicycleConfig, ExperimentConfig


@dataclass(frozen=True)
class ControlResult:
    u: np.ndarray
    u_ref: np.ndarray
    rhs: np.ndarray
    lhs_nominal: float
    feasible: bool
    active: bool
    clf_value: float
    clf_slack: float
    clf_residual: float


def split_state(state: np.ndarray) -> tuple[float, float, float, float, float, float]:
    return tuple(float(value) for value in state)


def rotation(psi: float) -> np.ndarray:
    c = np.cos(psi)
    s = np.sin(psi)
    return np.array([[c, -s], [s, c]], dtype=float)


def position(state: np.ndarray) -> np.ndarray:
    return np.asarray(state[:2], dtype=float)


def inertial_velocity(state: np.ndarray) -> np.ndarray:
    _, _, psi, vx, vy, _ = split_state(state)
    return rotation(psi) @ np.array([vx, vy], dtype=float)


def obstacle_positions(t: float, cfg: BicycleConfig) -> np.ndarray:
    return cfg.obstacles + t * cfg.obstacle_vels


def obstacle_velocities(t: float, cfg: BicycleConfig) -> np.ndarray:
    return cfg.obstacle_vels


def slip_angles(state: np.ndarray, steer: float, cfg: BicycleConfig) -> tuple[float, float]:
    _, _, _, vx, vy, yaw_rate = split_state(state)
    vx_eff = max(abs(vx), cfg.vx_floor)
    alpha_f = steer - (vy + cfg.lf * yaw_rate) / vx_eff
    alpha_r = -(vy - cfg.lr * yaw_rate) / vx_eff
    return float(alpha_f), float(alpha_r)


def tire_forces(state: np.ndarray, steer: float, cfg: BicycleConfig) -> tuple[float, float]:
    alpha_f, alpha_r = slip_angles(state, steer, cfg)
    return cfg.cornering_front * alpha_f, cfg.cornering_rear * alpha_r


def state_derivative_known(state: np.ndarray, u: np.ndarray, cfg: BicycleConfig) -> np.ndarray:
    _, _, psi, vx, vy, yaw_rate = split_state(state)
    ax, steer = (float(u[0]), float(u[1]))
    fyf, fyr = tire_forces(state, steer, cfg)
    p_dot = inertial_velocity(state)
    return np.array(
        [
            p_dot[0],
            p_dot[1],
            yaw_rate,
            ax + yaw_rate * vy,
            -yaw_rate * vx + (fyf + fyr) / cfg.mass,
            (cfg.lf * fyf - cfg.lr * fyr) / cfg.yaw_inertia,
        ],
        dtype=float,
    )


def pddot_known_zero(state: np.ndarray, cfg: BicycleConfig) -> np.ndarray:
    _, _, psi, _, _, _ = split_state(state)
    fyf0, fyr0 = tire_forces(state, 0.0, cfg)
    body_accel = np.array([0.0, (fyf0 + fyr0) / cfg.mass], dtype=float)
    return rotation(psi) @ body_accel


def pddot_input_matrix(state: np.ndarray, cfg: BicycleConfig) -> np.ndarray:
    _, _, psi, _, _, _ = split_state(state)
    body_matrix = np.array(
        [
            [1.0, 0.0],
            [0.0, cfg.cornering_front / cfg.mass],
        ],
        dtype=float,
    )
    return rotation(psi) @ body_matrix


def constraint_labels(cfg: BicycleConfig) -> list[str]:
    return [f"obstacle_{idx + 1}" for idx in range(len(cfg.obstacle_centers))]


def barriers(state: np.ndarray, t: float, cfg: BicycleConfig) -> np.ndarray:
    p = position(state)
    rel = p - obstacle_positions(t, cfg)
    return np.sum((rel / cfg.axes) ** 2, axis=1) - 1.0


def barrier_dots(state: np.ndarray, t: float, cfg: BicycleConfig) -> np.ndarray:
    p = position(state)
    p_dot = inertial_velocity(state)
    rel = p - obstacle_positions(t, cfg)
    rel_dot = p_dot - obstacle_velocities(t, cfg)
    return 2.0 * np.sum(rel * rel_dot / (cfg.axes**2), axis=1)


def control_coefficients(state: np.ndarray, t: float, cfg: BicycleConfig) -> np.ndarray:
    p = position(state)
    rel = p - obstacle_positions(t, cfg)
    pddot_u = pddot_input_matrix(state, cfg)
    scaled_rel = rel / (cfg.axes**2)
    return 2.0 * scaled_rel @ pddot_u


def known_h_ddots(state: np.ndarray, u: np.ndarray, t: float, cfg: BicycleConfig) -> np.ndarray:
    p = position(state)
    p_dot = inertial_velocity(state)
    rel = p - obstacle_positions(t, cfg)
    rel_dot = p_dot - obstacle_velocities(t, cfg)
    pddot = pddot_known_zero(state, cfg) + pddot_input_matrix(state, cfg) @ u
    return 2.0 * np.sum(rel_dot**2 / (cfg.axes**2), axis=1) + 2.0 * (
        (rel / (cfg.axes**2)) @ pddot
    )


def disturbance_coefficients(state: np.ndarray, t: float, cfg: BicycleConfig) -> np.ndarray:
    p = position(state)
    rel = p - obstacle_positions(t, cfg)
    _, _, psi, _, _, _ = split_state(state)
    rot_t = rotation(psi).T
    scaled_rel = rel / (cfg.axes**2)
    return 2.0 * (rot_t @ scaled_rel.T).T


def local_channel_lipschitz_bounds(state: np.ndarray, t: float, cfg: ExperimentConfig) -> np.ndarray:
    bicycle = cfg.bicycle
    p = position(state)
    p_dot = inertial_velocity(state)
    _, _, _, _, _, yaw_rate = split_state(state)
    rel = p - obstacle_positions(t, bicycle)
    rel_dot = p_dot - obstacle_velocities(t, bicycle)
    scaled_rel = rel / (bicycle.axes**2)
    scaled_rel_dot = rel_dot / (bicycle.axes**2)
    circle_q_norm = 2.0 * np.linalg.norm(scaled_rel, axis=1)
    circle_qdot_bound = 2.0 * (
        np.linalg.norm(scaled_rel_dot, axis=1)
        + abs(yaw_rate) * np.linalg.norm(scaled_rel, axis=1)
    )
    return (
        circle_qdot_bound * cfg.disturbance.xy_norm_bound
        + circle_q_norm * cfg.disturbance.xy_lipschitz_bound
    )


def local_initial_error_bounds(state: np.ndarray, t: float, cfg: ExperimentConfig) -> np.ndarray:
    q_norm = np.linalg.norm(disturbance_coefficients(state, t, cfg.bicycle), axis=1)
    return q_norm * cfg.disturbance.xy_norm_bound


def reference_input(state: np.ndarray, cfg: BicycleConfig) -> np.ndarray:
    _, y_pos, psi, vx, vy, yaw_rate = split_state(state)
    ax = cfg.speed_kp * (cfg.desired_speed - vx)
    steer = (
        -cfg.steer_y_gain * y_pos
        - cfg.steer_heading_gain * psi
        - cfg.steer_vy_gain * vy
        - cfg.steer_yaw_gain * yaw_rate
    )
    u = np.array([ax, steer], dtype=float)
    return np.minimum(np.maximum(u, cfg.lower_u), cfg.upper_u)


@lru_cache(maxsize=16)
def _linearized_lqr_clf_matrix(
    mass: float,
    yaw_inertia: float,
    lf: float,
    lr: float,
    cornering_front: float,
    cornering_rear: float,
    desired_speed: float,
    q_y: float,
    q_heading: float,
    q_speed: float,
    q_vy: float,
    q_yaw_rate: float,
    r_ax: float,
    r_steer: float,
) -> np.ndarray:
    speed = max(desired_speed, 1e-3)
    a_matrix = np.zeros((5, 5), dtype=float)
    b_matrix = np.zeros((5, 2), dtype=float)
    a_matrix[0, 1] = speed
    a_matrix[0, 3] = 1.0
    a_matrix[1, 4] = 1.0
    b_matrix[2, 0] = 1.0
    a_matrix[3, 3] = -(cornering_front + cornering_rear) / (mass * speed)
    a_matrix[3, 4] = (
        (-cornering_front * lf + cornering_rear * lr) / (mass * speed) - speed
    )
    b_matrix[3, 1] = cornering_front / mass
    a_matrix[4, 3] = (-lf * cornering_front + lr * cornering_rear) / (
        yaw_inertia * speed
    )
    a_matrix[4, 4] = -(
        lf**2 * cornering_front + lr**2 * cornering_rear
    ) / (yaw_inertia * speed)
    b_matrix[4, 1] = lf * cornering_front / yaw_inertia
    q_matrix = np.diag([q_y, q_heading, q_speed, q_vy, q_yaw_rate])
    r_matrix = np.diag([r_ax, r_steer])
    return solve_continuous_are(a_matrix, b_matrix, q_matrix, r_matrix)


def clf_matrix(cfg: BicycleConfig) -> np.ndarray:
    return _linearized_lqr_clf_matrix(
        cfg.mass,
        cfg.yaw_inertia,
        cfg.lf,
        cfg.lr,
        cfg.cornering_front,
        cfg.cornering_rear,
        cfg.desired_speed,
        cfg.clf_y_weight,
        cfg.clf_heading_weight,
        cfg.clf_speed_weight,
        cfg.clf_vy_weight,
        cfg.clf_yaw_rate_weight,
        cfg.clf_ax_weight,
        cfg.clf_steer_weight,
    )


def clf_error(state: np.ndarray, cfg: BicycleConfig) -> np.ndarray:
    _, y_pos, psi, vx, vy, yaw_rate = split_state(state)
    return np.array(
        [y_pos, psi, vx - cfg.desired_speed, vy, yaw_rate],
        dtype=float,
    )


def clf_coefficients(state: np.ndarray, cfg: BicycleConfig) -> tuple[float, np.ndarray, float]:
    """Return V, c0, c1 for Vdot = c0 + c1 @ u under the nominal model."""
    error = clf_error(state, cfg)
    p_matrix = clf_matrix(cfg)
    gradient = np.zeros(6, dtype=float)
    local_gradient = 2.0 * p_matrix @ error
    gradient[1] = local_gradient[0]
    gradient[2] = local_gradient[1]
    gradient[3] = local_gradient[2]
    gradient[4] = local_gradient[3]
    gradient[5] = local_gradient[4]
    zero_u = np.zeros(2, dtype=float)
    f0 = state_derivative_known(state, zero_u, cfg)
    g_columns = np.column_stack(
        [
            state_derivative_known(state, np.array([1.0, 0.0], dtype=float), cfg) - f0,
            state_derivative_known(state, np.array([0.0, 1.0], dtype=float), cfg) - f0,
        ]
    )
    return float(error @ p_matrix @ error), np.array(gradient @ g_columns), float(gradient @ f0)


def _build_clf_constraints(
    cbf_a: np.ndarray,
    cbf_b: np.ndarray,
    clf_a: np.ndarray,
    clf_upper: float,
    cfg: BicycleConfig,
) -> tuple[sparse.csc_matrix, np.ndarray, np.ndarray]:
    rows = [
        np.array([1.0, 0.0, 0.0], dtype=float),
        np.array([0.0, 1.0, 0.0], dtype=float),
        *[np.array([row[0], row[1], 0.0], dtype=float) for row in cbf_a],
        np.array([clf_a[0], clf_a[1], -1.0], dtype=float),
        np.array([0.0, 0.0, 1.0], dtype=float),
    ]
    lower = [
        cfg.lower_u[0],
        cfg.lower_u[1],
        *[float(value_b) for value_b in cbf_b],
        -np.inf,
        0.0,
    ]
    upper = [
        cfg.upper_u[0],
        cfg.upper_u[1],
        *[np.inf for _ in cbf_b],
        float(clf_upper),
        np.inf,
    ]
    return sparse.csc_matrix(np.vstack(rows)), np.array(lower, dtype=float), np.array(upper, dtype=float)


def _osqp_solve(
    p_matrix: sparse.csc_matrix,
    q_vector: np.ndarray,
    a_matrix: sparse.csc_matrix,
    lower: np.ndarray,
    upper: np.ndarray,
    eps_abs: float = 1e-8,
    eps_rel: float = 1e-8,
) -> Optional[np.ndarray]:
    problem = osqp.OSQP()
    problem.setup(
        P=p_matrix,
        q=q_vector,
        A=a_matrix,
        l=lower,
        u=upper,
        verbose=False,
        polish=False,
        eps_abs=eps_abs,
        eps_rel=eps_rel,
        max_iter=10_000,
    )
    result = problem.solve()
    solved = result.info.status in {"solved", "solved inaccurate"}
    if not solved or result.x is None or not np.all(np.isfinite(result.x)):
        return None
    return np.asarray(result.x, dtype=float)


def _solve_clf_qp(
    state: np.ndarray,
    u_ref: np.ndarray,
    u_prev: np.ndarray,
    cbf_a: np.ndarray,
    cbf_b: np.ndarray,
    cfg: BicycleConfig,
) -> tuple[np.ndarray, bool, bool, float, float, float]:
    v_value, clf_a, clf_const = clf_coefficients(state, cfg)
    clf_upper = -cfg.clf_rate * v_value - clf_const
    a_matrix, lower, upper = _build_clf_constraints(
        cbf_a,
        cbf_b,
        clf_a,
        clf_upper,
        cfg,
    )
    solution = _osqp_solve(
        sparse.diags(
            [
                cfg.ax_weight + cfg.smooth_ax_weight,
                cfg.steer_weight + cfg.smooth_steer_weight,
                cfg.clf_slack_weight,
            ],
            format="csc",
        ),
        np.array(
            [
                -cfg.ax_weight * u_ref[0] - cfg.smooth_ax_weight * u_prev[0],
                -cfg.steer_weight * u_ref[1] - cfg.smooth_steer_weight * u_prev[1],
                0.0,
            ],
            dtype=float,
        ),
        a_matrix,
        lower,
        upper,
    )
    if solution is None:
        residual = float(clf_const + clf_a @ u_ref + cfg.clf_rate * v_value)
        return u_ref, False, False, float(v_value), float("inf"), residual
    u = solution[:2]
    slack = max(0.0, float(solution[2]))
    residual = float(clf_const + clf_a @ u + cfg.clf_rate * v_value)
    feasible = bool(
        np.all(u >= cfg.lower_u - 1e-5)
        and np.all(u <= cfg.upper_u + 1e-5)
        and np.all(cbf_a @ u >= cbf_b - 1e-5)
        and float(clf_const + clf_a @ u - slack) <= -cfg.clf_rate * v_value + 1e-5
    )
    active = bool(np.any(cbf_a @ u <= cbf_b + 1e-6))
    return u, feasible, active, float(v_value), slack, residual


def ecbf_filter(
    state: np.ndarray,
    t: float,
    b_hat: np.ndarray,
    rho: np.ndarray,
    cfg: BicycleConfig,
    u_prev: Optional[np.ndarray] = None,
) -> ControlResult:
    h = barriers(state, t, cfg)
    h_dot = barrier_dots(state, t, cfg)
    known_without_control = known_h_ddots(state, np.zeros(2), t, cfg)
    a = control_coefficients(state, t, cfg)
    rhs = rho - known_without_control - cfg.k1 * h_dot - cfg.k0 * h - b_hat
    u_ref = reference_input(state, cfg)
    previous = u_ref if u_prev is None else np.asarray(u_prev, dtype=float)
    u, feasible, active, v_value, clf_slack, clf_residual = _solve_clf_qp(
        state,
        u_ref,
        previous,
        a,
        rhs,
        cfg,
    )
    return ControlResult(
        u=u,
        u_ref=u_ref,
        rhs=rhs,
        lhs_nominal=float(np.min(a @ u - rhs)),
        feasible=feasible,
        active=active,
        clf_value=v_value,
        clf_slack=clf_slack,
        clf_residual=clf_residual,
    )
