"""Continuous body-frame residual disturbances for the dynamic bicycle."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BodyFrameDisturbance:
    """Unknown residual acceleration and yaw-acceleration disturbance."""

    bias: tuple[float, float, float] = (0.03, 1.25, 0.02)
    amplitude: tuple[float, float, float] = (0.12, 0.52, 0.018)
    omega: tuple[float, float, float] = (0.70, 1.10, 0.90)
    phase: tuple[float, float, float] = (0.40, 0.37, 2.10)
    reversal_center_x: float = 50.0
    reversal_width: float = 28.0
    position_speed_bound: float = 10.0

    def _base_value(self, t: float) -> np.ndarray:
        bias = np.array(self.bias, dtype=float)
        amplitude = np.array(self.amplitude, dtype=float)
        omega = np.array(self.omega, dtype=float)
        phase = np.array(self.phase, dtype=float)
        return bias + amplitude * np.sin(omega * t + phase)

    def spatial_sign(self, x_position: float) -> float:
        z = (x_position - self.reversal_center_x) / self.reversal_width
        if z <= -1.0:
            return 1.0
        if z >= 1.0:
            return -1.0
        return float(-np.sin(0.5 * np.pi * z))

    def value(self, t: float, x_position: float) -> np.ndarray:
        return self.spatial_sign(x_position) * self._base_value(t)

    @property
    def max_spatial_sign_derivative(self) -> float:
        return float(0.5 * np.pi / self.reversal_width)

    @property
    def xy_norm_bound(self) -> float:
        bias = np.array(self.bias[:2], dtype=float)
        amplitude = np.array(self.amplitude[:2], dtype=float)
        return float(np.linalg.norm(bias) + np.linalg.norm(amplitude))

    @property
    def xy_lipschitz_bound(self) -> float:
        amplitude = np.array(self.amplitude[:2], dtype=float)
        omega = np.array(self.omega[:2], dtype=float)
        time_bound = np.linalg.norm(amplitude * omega)
        spatial_bound = self.position_speed_bound * self.xy_norm_bound * self.max_spatial_sign_derivative
        return float(time_bound + spatial_bound)
