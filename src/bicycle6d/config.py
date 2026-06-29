"""Configuration for the dynamic-bicycle DOB-CLF-ECBF-QP experiment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .disturbance import BodyFrameDisturbance


@dataclass(frozen=True)
class BicycleConfig:
    """Six-state dynamic bicycle and ECBF constants."""

    dt: float = 0.01
    horizon: float = 13.0
    initial_state: tuple[float, float, float, float, float, float] = (
        0.0,
        0.0,
        0.0,
        8.0,
        0.0,
        0.0,
    )
    finish_x: float = 84.0
    mass: float = 1500.0
    yaw_inertia: float = 2250.0
    lf: float = 1.2
    lr: float = 1.6
    cornering_front: float = 50000.0
    cornering_rear: float = 52000.0
    vx_floor: float = 3.0
    lane_half_width: float = 3.6
    obstacle_centers: tuple[tuple[float, float], ...] = ((30, 1.35), (74, -1.20))
    obstacle_velocities: tuple[tuple[float, float], ...] = ((0.0, 0.0), (0.0, 0.0))
    obstacle_axes: tuple[tuple[float, float], ...] = ((8.4, 2.28), (8.4, 2.28))
    lambda_1: float = 0.85
    lambda_2: float = 1.25
    desired_speed: float = 8.5
    speed_kp: float = 0.9
    steer_y_gain: float = 0.22
    steer_heading_gain: float = 1.40
    steer_vy_gain: float = 0.12
    steer_yaw_gain: float = 0.42
    ax_weight: float = 1.0
    steer_weight: float = 10.0
    clf_rate: float = 0.25
    clf_slack_weight: float = 2.0
    smooth_ax_weight: float = 0.15
    smooth_steer_weight: float = 28.0
    clf_y_weight: float = 1.0
    clf_heading_weight: float = 2.0
    clf_speed_weight: float = 0.20
    clf_vy_weight: float = 0.50
    clf_yaw_rate_weight: float = 0.50
    clf_ax_weight: float = 0.50
    clf_steer_weight: float = 5.0
    u_min: tuple[float, float] = (-6.0, -0.52)
    u_max: tuple[float, float] = (2.5, 0.52)
    global_channel_coefficient_bound: float = 170.0
    relative_speed_bound: float = 25.0
    yaw_rate_bound: float = 0.9
    tunable_margin_max_scale: float = 8.0
    tunable_margin_psi_scale: float = 1.0
    safety_plot_horizon: float = 10.4

    @property
    def state0(self) -> np.ndarray:
        return np.array(self.initial_state, dtype=float)

    @property
    def obstacles(self) -> np.ndarray:
        return np.array(self.obstacle_centers, dtype=float)

    @property
    def obstacle_vels(self) -> np.ndarray:
        return np.array(self.obstacle_velocities, dtype=float)

    @property
    def axes(self) -> np.ndarray:
        return np.array(self.obstacle_axes, dtype=float)

    @property
    def lower_u(self) -> np.ndarray:
        return np.array(self.u_min, dtype=float)

    @property
    def upper_u(self) -> np.ndarray:
        return np.array(self.u_max, dtype=float)

    @property
    def k0(self) -> float:
        return self.lambda_1 * self.lambda_2

    @property
    def k1(self) -> float:
        return self.lambda_1 + self.lambda_2


@dataclass(frozen=True)
class ObserverConfig:
    """DOB parameters for highest-order ECBF channels."""

    gain: float
    b_h_bound: float
    epsilon: float = 0.08

    @property
    def steady_error_bound(self) -> float:
        return self.b_h_bound / self.gain


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level experiment configuration."""

    bicycle: BicycleConfig
    disturbance: BodyFrameDisturbance
    observer: ObserverConfig
    output_dir: Path


def global_channel_lipschitz_bounds(
    bicycle: BicycleConfig,
    disturbance: BodyFrameDisturbance,
) -> np.ndarray:
    """Return scene-level per-channel bounds for |dot b_i|."""
    circle_qdot = 2.0 * (
        bicycle.relative_speed_bound
        + bicycle.yaw_rate_bound * 0.5 * bicycle.global_channel_coefficient_bound
    )
    circle_bound = (
        circle_qdot * disturbance.xy_norm_bound
        + bicycle.global_channel_coefficient_bound * disturbance.xy_lipschitz_bound
    )
    return np.full(len(bicycle.obstacle_centers), circle_bound, dtype=float)


def global_channel_lipschitz_bound(
    bicycle: BicycleConfig,
    disturbance: BodyFrameDisturbance,
) -> float:
    """Return the largest scene-level channel derivative bound."""
    return float(np.max(global_channel_lipschitz_bounds(bicycle, disturbance)))


def default_config() -> ExperimentConfig:
    root = Path(__file__).resolve().parents[2]
    bicycle = BicycleConfig()
    disturbance = BodyFrameDisturbance()
    dob_gain = 70.0
    b_h_bound = global_channel_lipschitz_bound(bicycle, disturbance)
    observer = ObserverConfig(
        gain=dob_gain,
        b_h_bound=b_h_bound,
    )
    return ExperimentConfig(
        bicycle=bicycle,
        disturbance=disturbance,
        observer=observer,
        output_dir=root / "outputs" / "bicycle6d",
    )
