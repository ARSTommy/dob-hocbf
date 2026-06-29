"""Scalar DOBs for highest-order relative-degree-two ECBF channels."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Optional


@dataclass
class RelativeDegreeTwoDOB:
    """First-order DOB for y = dot h, where dot y = known + b(t)."""

    gain: float
    b_h_bound: float
    initial_error_bound: float
    epsilon: float
    b_hat: float = 0.0
    xi: float = 0.0
    bar_e: float = 0.0

    def __post_init__(self) -> None:
        if self.gain <= 0.0:
            raise ValueError("gain must be positive")
        if self.b_h_bound < 0.0:
            raise ValueError("b_h_bound must be nonnegative")
        if self.initial_error_bound < 0.0:
            raise ValueError("initial_error_bound must be nonnegative")
        self.bar_e = self.initial_error_bound

    @property
    def rho(self) -> float:
        return self.bar_e + self.epsilon

    def initialize(self, h_dot_initial: float, b_hat_initial: float = 0.0) -> None:
        self.b_hat = b_hat_initial
        self.xi = self.gain * h_dot_initial - self.b_hat

    def flow_update(
        self,
        h_dot_next: float,
        known_h_ddot: float,
        dt: float,
        b_h_bound: Optional[float] = None,
    ) -> None:
        self.xi += dt * self.gain * (known_h_ddot + self.b_hat)
        self.b_hat = self.gain * h_dot_next - self.xi

        current_bound = self.b_h_bound if b_h_bound is None else b_h_bound
        if current_bound < 0.0:
            raise ValueError("b_h_bound must be nonnegative")
        decay = exp(-self.gain * dt)
        steady = current_bound / self.gain
        self.bar_e = steady + (self.bar_e - steady) * decay
        if self.bar_e < 0.0:
            self.bar_e = 0.0


@dataclass
class StaticEstimator:
    """No-DOB baseline with a fixed estimate and fixed margin."""

    b_hat: float = 0.0
    rho: float = 0.0
    bar_e: float = 0.0

    def initialize(self, h_dot_initial: float, b_hat_initial: float = 0.0) -> None:
        return None

    def flow_update(
        self,
        h_dot_next: float,
        known_h_ddot: float,
        dt: float,
        b_h_bound: Optional[float] = None,
    ) -> None:
        return None
