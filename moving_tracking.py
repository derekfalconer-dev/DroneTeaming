"""Shared moving-object tracking primitives for the drone simulator.

This module intentionally contains no drone-name, drone-ID-pair, playbook, or
task-ownership logic. Scenario helpers below construct repeatable geometry for
the demonstration, but association and filtering operate only on reports.

Estimator model
---------------
State vector: x = [East, North, velocity East, velocity North].

Transition for elapsed time dt:
F = [[1, 0, dt, 0],
     [0, 1, 0, dt],
     [0, 0, 1, 0],
     [0, 0, 0, 1]]

Position-only reports use H = [[1, 0, 0, 0], [0, 1, 0, 0]].

Process noise Q assumes white acceleration noise with variance
MOVING_TRACK_ACCELERATION_NOISE_MPS2**2:
Q = q * [[dt^4/4, 0, dt^3/2, 0],
         [0, dt^4/4, 0, dt^3/2],
         [dt^3/2, 0, dt^2, 0],
         [0, dt^3/2, 0, dt^2]]

Reports in this increment are independent noisy position measurements with
source-local mocked track continuity. Real perception trackers may introduce
correlated errors and would require a revised fusion strategy.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass


DEFAULT_SCENARIO_ID = "seeded_stationary_patrol"
MOVING_HANDOFF_SCENARIO_ID = "moving_person_handoff"
MOVING_HANDOFF_SCENARIO_NAME = "Jogging person handoff"
MOVING_PERSON_SPEED_MPS = 4.0
TRACK_REPORT_SCHEMA_VERSION = "moving-point-track-v1"
MOVING_TRACK_REPORT_TYPE = "moving_point_track"
MOVING_TRACK_BUFFER_SECONDS = 1.0
MOVING_TRACK_BUFFER_LIMIT = 512
MOVING_TRACK_TRACKED_SECONDS = 3.0
MOVING_TRACK_LOST_SECONDS = 10.0
MOVING_TRACK_TRAIL_SECONDS = 18.0
MOVING_TRACK_INITIAL_VELOCITY_VARIANCE = 64.0
MOVING_TRACK_ACCELERATION_NOISE_MPS2 = 1.1
MOVING_TRACK_POSITION_UNCERTAINTY_FLOOR_M2 = 4.0


@dataclass(frozen=True)
class ScenarioPose:
    east_m: float
    north_m: float
    heading_deg: float


def new_run_id(seed: int) -> str:
    return f"run-{seed}-{uuid.uuid4().hex[:12]}"


def moving_person_position(simulation_time: float) -> tuple[float, float]:
    """Private truth trajectory for the demonstration jogging person.

    Truth is used only by simulation generation and evaluation. It is never sent
    as a shared object identity or used by association decisions.
    """
    return -170.0 + (MOVING_PERSON_SPEED_MPS * simulation_time), 0.0 + (10.0 * math.sin(simulation_time / 18.0))


def moving_person_trajectory(max_seconds: int = 95) -> list[dict[str, float]]:
    return [
        {"time": float(second), "east_m": moving_person_position(float(second))[0], "north_m": moving_person_position(float(second))[1]}
        for second in range(max_seconds + 1)
    ]


def moving_handoff_background_points() -> list[tuple[str, str, float, float]]:
    return [
        ("moving-bg-west-crate", "storage crate", -360.0, -185.0),
        ("moving-bg-south-generator", "generator", -35.0, -260.0),
        ("moving-bg-east-antenna", "antenna", 360.0, 185.0),
        ("moving-bg-north-case", "equipment case", 20.0, 285.0),
    ]


def moving_handoff_pose_for(
    drone_id: int,
    initial_observer_id: int,
    next_observer_id: int,
) -> ScenarioPose:
    """Return static camera geometry for the handoff scenario.

    The selected IDs only shape coverage. The tracker never receives them as
    identity hints and can associate reports in any arrival order.
    """
    if drone_id == initial_observer_id:
        return ScenarioPose(-220.0, -35.0, 90.0)

    if drone_id == next_observer_id:
        return ScenarioPose(220.0, 35.0, 270.0)

    return ScenarioPose(0.0, 390.0, 0.0)


def mat_identity(size: int) -> list[list[float]]:
    return [[1.0 if row == col else 0.0 for col in range(size)] for row in range(size)]


def mat_transpose(matrix: list[list[float]]) -> list[list[float]]:
    return [list(row) for row in zip(*matrix, strict=True)]


def mat_mul(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
    return [
        [
            sum(left[row][index] * right[index][col] for index in range(len(right)))
            for col in range(len(right[0]))
        ]
        for row in range(len(left))
    ]


def mat_vec_mul(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [sum(matrix[row][col] * vector[col] for col in range(len(vector))) for row in range(len(matrix))]


def mat_add(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
    return [
        [left[row][col] + right[row][col] for col in range(len(left[0]))]
        for row in range(len(left))
    ]


def mat_sub(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
    return [
        [left[row][col] - right[row][col] for col in range(len(left[0]))]
        for row in range(len(left))
    ]


def state_transition(dt: float) -> list[list[float]]:
    return [
        [1.0, 0.0, dt, 0.0],
        [0.0, 1.0, 0.0, dt],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def process_noise(dt: float) -> list[list[float]]:
    q = MOVING_TRACK_ACCELERATION_NOISE_MPS2 * MOVING_TRACK_ACCELERATION_NOISE_MPS2
    dt2 = dt * dt
    dt3 = dt2 * dt
    dt4 = dt2 * dt2
    return [
        [q * dt4 / 4.0, 0.0, q * dt3 / 2.0, 0.0],
        [0.0, q * dt4 / 4.0, 0.0, q * dt3 / 2.0],
        [q * dt3 / 2.0, 0.0, q * dt2, 0.0],
        [0.0, q * dt3 / 2.0, 0.0, q * dt2],
    ]


def predict_state(
    state: list[float],
    covariance: list[list[float]],
    from_time: float,
    to_time: float,
) -> tuple[list[float], list[list[float]]]:
    dt = max(0.0, to_time - from_time)
    transition = state_transition(dt)
    predicted_state = mat_vec_mul(transition, state)
    predicted_covariance = mat_add(mat_mul(mat_mul(transition, covariance), mat_transpose(transition)), process_noise(dt))
    return predicted_state, predicted_covariance


def position_measurement_matrix() -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ]


def det_2x2(matrix: list[list[float]]) -> float:
    return (matrix[0][0] * matrix[1][1]) - (matrix[0][1] * matrix[1][0])


def inverse_2x2(matrix: list[list[float]]) -> list[list[float]]:
    determinant = det_2x2(matrix)
    if determinant <= 0.0 or not math.isfinite(determinant):
        raise ValueError("Matrix is not positive definite.")

    return [
        [matrix[1][1] / determinant, -matrix[0][1] / determinant],
        [-matrix[1][0] / determinant, matrix[0][0] / determinant],
    ]


def innovation_for_position(
    state: list[float],
    covariance: list[list[float]],
    measurement: tuple[float, float],
    measurement_covariance: list[list[float]],
) -> tuple[list[float], list[list[float]], float]:
    residual = [measurement[0] - state[0], measurement[1] - state[1]]
    h = position_measurement_matrix()
    innovation_covariance = mat_add(mat_mul(mat_mul(h, covariance), mat_transpose(h)), measurement_covariance)
    inv_s = inverse_2x2(innovation_covariance)
    d2 = (
        residual[0] * ((inv_s[0][0] * residual[0]) + (inv_s[0][1] * residual[1]))
        + residual[1] * ((inv_s[1][0] * residual[0]) + (inv_s[1][1] * residual[1]))
    )
    return residual, innovation_covariance, d2


def update_with_position(
    state: list[float],
    covariance: list[list[float]],
    measurement: tuple[float, float],
    measurement_covariance: list[list[float]],
) -> tuple[list[float], list[list[float]], float]:
    residual, innovation_covariance, d2 = innovation_for_position(state, covariance, measurement, measurement_covariance)
    h = position_measurement_matrix()
    inv_s = inverse_2x2(innovation_covariance)
    kalman_gain = mat_mul(mat_mul(covariance, mat_transpose(h)), inv_s)
    correction = mat_vec_mul(kalman_gain, residual)
    updated_state = [state[index] + correction[index] for index in range(4)]
    i_minus_kh = mat_sub(mat_identity(4), mat_mul(kalman_gain, h))
    updated_covariance = mat_mul(i_minus_kh, covariance)
    return updated_state, updated_covariance, d2


def initialize_track(
    east_m: float,
    north_m: float,
    position_covariance: list[list[float]],
) -> tuple[list[float], list[list[float]]]:
    return (
        [east_m, north_m, 0.0, 0.0],
        [
            [max(position_covariance[0][0], MOVING_TRACK_POSITION_UNCERTAINTY_FLOOR_M2), position_covariance[0][1], 0.0, 0.0],
            [position_covariance[1][0], max(position_covariance[1][1], MOVING_TRACK_POSITION_UNCERTAINTY_FLOOR_M2), 0.0, 0.0],
            [0.0, 0.0, MOVING_TRACK_INITIAL_VELOCITY_VARIANCE, 0.0],
            [0.0, 0.0, 0.0, MOVING_TRACK_INITIAL_VELOCITY_VARIANCE],
        ],
    )


def horizontal_uncertainty_m(covariance: list[list[float]]) -> float:
    return math.sqrt(max(0.0, covariance[0][0] + covariance[1][1]))


def moving_lifecycle(last_measurement_time: float, display_time: float) -> str:
    age = max(0.0, display_time - last_measurement_time)

    if age <= MOVING_TRACK_TRACKED_SECONDS:
        return "tracked"

    if age <= MOVING_TRACK_LOST_SECONDS:
        return "coasting"

    return "lost"
