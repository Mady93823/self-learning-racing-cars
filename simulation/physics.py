"""Vectorised top-down car physics.

A kinematic bicycle model with a grip limit layered on top. The body can yaw
slightly faster than the tyres can bend the velocity vector, so a car entering a
corner too fast slides wide. That is what makes the throttle output matter: fast
laps need braking, and smoke appears because of real lateral slip rather than
because an effect was scripted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from settings import PhysicsSettings


@dataclass
class CarState:
    """Struct of arrays: index ``i`` across every field describes one car."""

    x: np.ndarray
    y: np.ndarray
    heading: np.ndarray
    forward_speed: np.ndarray
    lateral_speed: np.ndarray
    steer: np.ndarray           # current wheel position in [-1, 1]
    throttle: np.ndarray        # current pedal position in [-1, 1]; negative brakes
    yaw_rate: np.ndarray

    @classmethod
    def zeros(cls, count: int) -> "CarState":
        return cls(*(np.zeros(count) for _ in range(8)))


def step(state: CarState, ids: np.ndarray, steer_cmd: np.ndarray, throttle_cmd: np.ndarray,
         dt: float, cfg: PhysicsSettings) -> None:
    """Advance the cars listed in ``ids`` by one fixed tick. Commands are per-car arrays."""
    if ids.size == 0:
        return

    # Actuators move at a finite rate, so a jittery network still produces smooth motion.
    steer_limit = math.radians(cfg.steer_rate_deg) / math.radians(cfg.max_steer_deg) * dt
    steer = state.steer[ids] + np.clip(steer_cmd[ids] - state.steer[ids], -steer_limit, steer_limit)
    throttle_limit = cfg.throttle_rate * dt
    throttle = state.throttle[ids] + np.clip(
        throttle_cmd[ids] - state.throttle[ids], -throttle_limit, throttle_limit)

    heading = state.heading[ids]
    forward = state.forward_speed[ids]
    lateral = state.lateral_speed[ids]

    # Longitudinal: engine or brakes, linear drag, plus scrub while sliding.
    accel = np.where(throttle >= 0.0, throttle * cfg.engine_accel, throttle * cfg.brake_decel)
    accel = accel - cfg.drag * forward - cfg.slide_scrub * np.abs(lateral)
    forward = np.maximum(forward + accel * dt, 0.0)  # no reverse gear

    # Yaw: bicycle model, capped by what the tyres can roughly support at this speed.
    wheel_angle = steer * math.radians(cfg.max_steer_deg)
    wanted_yaw = forward * np.tan(wheel_angle) / cfg.wheelbase
    yaw_cap = cfg.yaw_overdrive * cfg.lateral_grip / np.maximum(forward, 40.0)
    yaw = np.clip(wanted_yaw, -yaw_cap, yaw_cap)

    # World velocity before the car rotates.
    cos_h, sin_h = np.cos(heading), np.sin(heading)
    vx = forward * cos_h - lateral * sin_h
    vy = forward * sin_h + lateral * cos_h

    heading = heading + yaw * dt
    cos_n, sin_n = np.cos(heading), np.sin(heading)

    # Re-express the velocity in the rotated body frame, which is where slip appears...
    forward = np.maximum(vx * cos_n + vy * sin_n, 0.0)
    lateral = -vx * sin_n + vy * cos_n
    # ...and let the tyres remove only as much of it as the grip limit allows.
    removal = cfg.lateral_grip * dt
    lateral = np.sign(lateral) * np.maximum(np.abs(lateral) - removal, 0.0)

    vx = forward * cos_n - lateral * sin_n
    vy = forward * sin_n + lateral * cos_n

    state.x[ids] += vx * dt
    state.y[ids] += vy * dt
    state.heading[ids] = heading
    state.forward_speed[ids] = forward
    state.lateral_speed[ids] = lateral
    state.steer[ids] = steer
    state.throttle[ids] = throttle
    state.yaw_rate[ids] = yaw
