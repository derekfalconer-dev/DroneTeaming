#!/usr/bin/env python3
"""Independent simulated drone process with telemetry, mapping, and peer sharing."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import http.server
import json
import math
import os
import random
import signal
import sys
import socket
import socketserver
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections import deque
from dataclasses import asdict, dataclass

from moving_tracking import (
    DEFAULT_SCENARIO_ID,
    MOVING_HANDOFF_SCENARIO_ID,
    MOVING_HANDOFF_SCENARIO_NAME,
    MOVING_TRACK_BUFFER_LIMIT,
    MOVING_TRACK_BUFFER_SECONDS,
    MOVING_TRACK_LOST_SECONDS,
    MOVING_TRACK_REPORT_TYPE,
    MOVING_TRACK_TRACKED_SECONDS,
    MOVING_TRACK_TRAIL_SECONDS,
    TRACK_REPORT_SCHEMA_VERSION,
    horizontal_uncertainty_m,
    initialize_track,
    innovation_for_position,
    moving_handoff_background_points,
    moving_handoff_pose_for,
    moving_lifecycle,
    moving_person_position,
    moving_person_trajectory,
    new_run_id,
    predict_state,
    update_with_position,
)
from team_configuration import (
    TeamConfigurationError,
    TeamMemberConfiguration,
    load_team_configuration,
    validate_advertised_team_member,
)


SCHEMA_VERSION = "3.0"
APPLICATION_BUILD = "priority-response-stage-a-2026-11-06"
SCENARIO_NAME = "Seeded noisy overlapping moving-search patrol"
DEFAULT_SCENARIO_SEED = 20261106
SCENARIO_OBJECT_COUNT = 24
MIN_PRIORITY_OBJECT_COUNT = 1
MAX_PRIORITY_OBJECT_COUNT = 5
RANDOM_PRIORITY_OBJECT_COUNT = 0
TASK_TYPE_ISR_OBSERVE = "isr_observe"
TASK_TYPE_PAYLOAD_DROP = "payload_drop"
SUPPORTED_TASK_TYPES = {TASK_TYPE_ISR_OBSERVE, TASK_TYPE_PAYLOAD_DROP}
PAYLOAD_PENDING_STATUS = "pending_workflow"
PAYLOAD_REQUEST_REPORTED_STATUS = "reported"
PAYLOAD_APPROACHING_STATUS = "approaching_payload"
PAYLOAD_AWAITING_LOCAL_ACQUISITION_STATUS = "awaiting_local_acquisition"
PAYLOAD_WAITING_FOR_TRACK_STATUS = "waiting_for_track"
PAYLOAD_EXECUTION_STATUSES = {
    PAYLOAD_APPROACHING_STATUS,
    PAYLOAD_AWAITING_LOCAL_ACQUISITION_STATUS,
    PAYLOAD_WAITING_FOR_TRACK_STATUS,
}
PAYLOAD_STATE_HEARTBEAT_SECONDS = 5.0
DEFAULT_PAYLOAD_REQUESTS_ENABLED = False
DEFAULT_PAYLOAD_TYPE = "rescue_kit"
DEFAULT_PAYLOAD_QUANTITY = 1
DEFAULT_PAYLOAD_STANDOFF_M = 45.0
PLAYBOOK_REPORT_CONTINUE = "report_continue"
PLAYBOOK_OBSERVE_ORBIT = "observe_orbit"
PLAYBOOK_ORBIT_FOLLOW = "orbit_follow"
ENABLED_PLAYBOOKS = {PLAYBOOK_REPORT_CONTINUE, PLAYBOOK_OBSERVE_ORBIT}
SUPPORTED_PLAYBOOKS = ENABLED_PLAYBOOKS | {PLAYBOOK_ORBIT_FOLLOW}
DEFAULT_PLAYBOOK = PLAYBOOK_REPORT_CONTINUE
AVAILABLE_SCENARIOS = {DEFAULT_SCENARIO_ID, MOVING_HANDOFF_SCENARIO_ID}
DEFAULT_INITIAL_OBSERVER_ID = 1
DEFAULT_NEXT_OBSERVER_ID = 2
TASK_ELECTION_SECONDS = 1.5
ORBIT_RADIUS_M = 85.0
ORBIT_ANGULAR_RATE_DEG_PER_SECOND = 9.0
MAX_RESPONSE_SPEED_MPS = 38.0
MAX_RESPONSE_TURN_RATE_DEG_PER_SECOND = 55.0
SCENARIO_BOUND_EAST_M = 620.0
SCENARIO_BOUND_NORTH_M = 520.0
PRIORITY_SIGNATURE_PREFIX = "mock-search-signature"
RESPONSE_LIMITATION_NOTE = (
    "Stage A supports deterministic ownership and stationary observe-and-orbit only. "
    "Objective identity is the mocked per-object search signature from accepted observations; "
    "local map IDs and hidden truth IDs are never used as cross-node task identity. "
    "Connected-peer claim convergence is supported, but radio partition recovery, owner-failure "
    "reassignment, moving-object tracking, and orbit-and-follow are future Stage B work."
)
OBSERVATION_TRACE_RETENTION = 256
NETWORK_THROUGHPUT_WINDOW_SECONDS = 5.0
STALE_AFTER_SECONDS = 6.0
TRAIL_SAMPLE_INTERVAL_SECONDS = 0.5
TRAIL_RETENTION_SECONDS = 30.0
PATROL_WAYPOINT_INTERVAL_SECONDS = 5.0
CAMERA_RANGE_M = 260.0
CAMERA_HALF_ANGLE_DEG = 24.0
ASSOCIATION_GATE_D2 = 9.21
ASSOCIATION_AMBIGUITY_D2_DELTA = 0.75
CONTRIBUTION_MAX_AGE_SECONDS = 8.0
COVARIANCE_INTERSECTION_STEPS = 101
SENSOR_BIAS_ENABLED = False
SENSOR_SYSTEMATIC_VARIANCE_M2 = 0.0
EARTH_METERS_PER_DEG_LAT = 111_111.0
MAP_REFERENCE_LATITUDE = 37.7752
MAP_REFERENCE_LONGITUDE = -122.4230
MEASUREMENT_FRAME = (
    "Local tangent-plane metric frame anchored at "
    f"{MAP_REFERENCE_LATITUDE:.7f},{MAP_REFERENCE_LONGITUDE:.7f}; "
    "position vector order is [East, North] in meters. Geographic latitude/longitude "
    "is converted using a small-area equirectangular approximation."
)


DRONE_PROFILES: dict[int, dict[str, float | str]] = {
    1: {
        "color": "red",
        "latitude": 37.7749,
        "longitude": -122.4194,
        "heading_deg": 45.0,
    },
    2: {
        "color": "blue",
        "latitude": 37.7797,
        "longitude": -122.4262,
        "heading_deg": 210.0,
    },
    3: {
        "color": "green",
        "latitude": 37.7695,
        "longitude": -122.4284,
        "heading_deg": 300.0,
    },
}


OBJECT_LABELS = (
    "vehicle",
    "person",
    "storage crate",
    "generator",
    "antenna",
    "equipment case",
)


@dataclass(frozen=True)
class PatrolRoute:
    seed: int
    drone_id: int
    waypoints_m: list[tuple[float, float]]
    waypoint_interval_seconds: float


@dataclass(frozen=True)
class SensorModel:
    sensor_id: str
    sigma_major_m: float
    sigma_minor_m: float
    axis_deg: float
    bias_east_m: float = 0.0
    bias_north_m: float = 0.0


SENSOR_MODELS: dict[int, SensorModel] = {
    1: SensorModel("eo-red", sigma_major_m=8.0, sigma_minor_m=4.5, axis_deg=20.0),
    2: SensorModel("eo-blue", sigma_major_m=10.5, sigma_minor_m=5.5, axis_deg=-35.0),
    3: SensorModel("eo-green", sigma_major_m=13.0, sigma_minor_m=7.0, axis_deg=55.0),
}


@dataclass(frozen=True)
class DroneStatus:
    schema_version: str
    app_build: str
    scenario_name: str
    scenario_seed: int
    id: int
    name: str
    state: str
    latitude: float
    longitude: float
    altitude_m: float
    speed_mps: float
    heading_deg: float
    camera_yaw_deg: float
    battery_percent: float
    timestamp: float
    paused: bool
    sharing_enabled: bool
    selected_playbook: str
    playbook_config_version: int
    requested_priority_count: int
    priority_object_count: int
    camera_footprint: list[dict[str, float]]
    flight_trail: list[dict[str, float]]
    response_task: dict[str, object] | None
    response_tasks: list[dict[str, object]]
    response_queue_depth: int
    orbit_radius_m: float
    current_mode: str
    assigned_objective: str | None
    diagnostics: dict[str, object]
    team_configuration: dict[str, object]


@dataclass(frozen=True)
class Observation:
    id: str
    scenario_run_id: str
    source_drone_id: int
    source_drone_name: str
    source_color: str
    sensor_id: str
    source_track_id: str
    timestamp: float
    measurement_frame: str
    east_m: float
    north_m: float
    latitude: float
    longitude: float
    label: str
    classification_score: float
    covariance: list[list[float]]
    priority_evidence: dict[str, object] | None = None
    measurement_type: str = "point"
    object_id: str = ""
    report_schema_version: str = "legacy-observation-v1"
    source_node_session_id: str = ""
    report_id: str = ""
    state_epoch: float | None = None
    velocity_east_mps: float | None = None
    velocity_north_mps: float | None = None
    state_covariance: list[list[float]] | None = None
    sensor_track_status: str = "active"
    simulated: bool = True
    supporting_observation_ids: list[str] | None = None


def meters_per_deg_lon(latitude: float) -> float:
    return EARTH_METERS_PER_DEG_LAT * math.cos(math.radians(latitude))


def offset_lat_lon(latitude: float, longitude: float, east_m: float, north_m: float) -> tuple[float, float]:
    return (
        latitude + (north_m / EARTH_METERS_PER_DEG_LAT),
        longitude + (east_m / meters_per_deg_lon(latitude)),
    )


def local_m_to_lat_lon(east_m: float, north_m: float) -> tuple[float, float]:
    return offset_lat_lon(MAP_REFERENCE_LATITUDE, MAP_REFERENCE_LONGITUDE, east_m, north_m)


def catmull_rom(p0: float, p1: float, p2: float, p3: float, t: float) -> float:
    return 0.5 * (
        (2.0 * p1)
        + ((-p0 + p2) * t)
        + ((2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t * t)
        + ((-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t * t * t)
    )


def catmull_rom_derivative(p0: float, p1: float, p2: float, p3: float, t: float) -> float:
    return 0.5 * (
        (-p0 + p2)
        + (2.0 * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t)
        + (3.0 * (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t * t)
    )


def generate_patrol_routes(seed: int) -> dict[int, PatrolRoute]:
    rng = random.Random(seed)
    shared_east_m = rng.uniform(-35.0, 35.0)
    shared_north_m = rng.uniform(-25.0, 25.0)

    base_routes: dict[int, list[tuple[float, float]]] = {
        1: [
            (-390.0, -170.0),
            (-245.0, -80.0),
            (-125.0, -28.0),
            (5.0, -26.0),
            (58.0, -72.0),
            (260.0, -175.0),
            (455.0, -245.0),
            (300.0, -55.0),
            (50.0, -28.0),
            (-170.0, -125.0),
        ],
        2: [
            (-405.0, 170.0),
            (-250.0, 82.0),
            (-125.0, 28.0),
            (6.0, 28.0),
            (92.0, 22.0),
            (178.0, 24.0),
            (390.0, 150.0),
            (505.0, 250.0),
            (125.0, 100.0),
            (-190.0, 135.0),
        ],
        3: [
            (130.0, -415.0),
            (68.0, -245.0),
            (26.0, -92.0),
            (3.0, 128.0),
            (-42.0, 282.0),
            (-230.0, 430.0),
            (-365.0, 245.0),
            (-78.0, 126.0),
            (44.0, -90.0),
            (255.0, -300.0),
        ],
    }

    routes: dict[int, PatrolRoute] = {}

    for drone_id, waypoints in base_routes.items():
        jittered_waypoints = []

        for waypoint_index, (east_m, north_m) in enumerate(waypoints):
            preserve_overlap = waypoint_index in {2, 3, 4, 8}
            waypoint_jitter_east = 0.0 if preserve_overlap else rng.uniform(-18.0, 18.0)
            waypoint_jitter_north = 0.0 if preserve_overlap else rng.uniform(-18.0, 18.0)
            jittered_waypoints.append(
                (
                    east_m + shared_east_m + waypoint_jitter_east,
                    north_m + shared_north_m + waypoint_jitter_north,
                ),
            )

        routes[drone_id] = PatrolRoute(
            seed=seed,
            drone_id=drone_id,
            waypoints_m=jittered_waypoints,
            waypoint_interval_seconds=PATROL_WAYPOINT_INTERVAL_SECONDS,
        )

    return routes


def patrol_pose_on_route(route: PatrolRoute, simulation_time: float) -> tuple[float, float, float]:
    waypoints = route.waypoints_m
    route_period = len(waypoints) * route.waypoint_interval_seconds
    route_time = simulation_time % route_period
    segment_index = int(route_time // route.waypoint_interval_seconds)
    t = (route_time % route.waypoint_interval_seconds) / route.waypoint_interval_seconds

    p0 = waypoints[(segment_index - 1) % len(waypoints)]
    p1 = waypoints[segment_index % len(waypoints)]
    p2 = waypoints[(segment_index + 1) % len(waypoints)]
    p3 = waypoints[(segment_index + 2) % len(waypoints)]

    east_m = catmull_rom(p0[0], p1[0], p2[0], p3[0], t)
    north_m = catmull_rom(p0[1], p1[1], p2[1], p3[1], t)
    velocity_east = catmull_rom_derivative(p0[0], p1[0], p2[0], p3[0], t)
    velocity_north = catmull_rom_derivative(p0[1], p1[1], p2[1], p3[1], t)
    latitude, longitude = local_m_to_lat_lon(east_m, north_m)
    heading = math.degrees(math.atan2(velocity_east, velocity_north)) % 360.0
    return latitude, longitude, heading


def patrol_pose_for(drone_id: int, simulation_time: float, seed: int = DEFAULT_SCENARIO_SEED) -> tuple[float, float, float]:
    return patrol_pose_on_route(generate_patrol_routes(seed)[drone_id], simulation_time)


def covariance_from_sensor_model(sensor_model: SensorModel) -> list[list[float]]:
    axis_rad = math.radians(sensor_model.axis_deg)
    cos_axis = math.cos(axis_rad)
    sin_axis = math.sin(axis_rad)
    major_variance = sensor_model.sigma_major_m * sensor_model.sigma_major_m
    minor_variance = sensor_model.sigma_minor_m * sensor_model.sigma_minor_m

    covariance_ee = (cos_axis * cos_axis * major_variance) + (sin_axis * sin_axis * minor_variance)
    covariance_nn = (sin_axis * sin_axis * major_variance) + (cos_axis * cos_axis * minor_variance)
    covariance_en = cos_axis * sin_axis * (major_variance - minor_variance)

    if SENSOR_BIAS_ENABLED:
        covariance_ee += SENSOR_SYSTEMATIC_VARIANCE_M2
        covariance_nn += SENSOR_SYSTEMATIC_VARIANCE_M2

    return [
        [covariance_ee, covariance_en],
        [covariance_en, covariance_nn],
    ]


def matrix_det_2x2(matrix: list[list[float]]) -> float:
    return (matrix[0][0] * matrix[1][1]) - (matrix[0][1] * matrix[1][0])


def is_valid_covariance(matrix: list[list[float]]) -> bool:
    if len(matrix) != 2 or len(matrix[0]) != 2 or len(matrix[1]) != 2:
        return False

    values = [matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]]
    if not all(math.isfinite(value) for value in values):
        return False

    if abs(matrix[0][1] - matrix[1][0]) > 1e-6:
        return False

    return matrix[0][0] > 0.0 and matrix_det_2x2(matrix) > 0.0


def solve_2x2(matrix: list[list[float]], vector: tuple[float, float]) -> tuple[float, float]:
    determinant = matrix_det_2x2(matrix)
    if determinant <= 0.0 or not math.isfinite(determinant):
        raise ValueError("Matrix is not positive definite.")

    return (
        ((matrix[1][1] * vector[0]) - (matrix[0][1] * vector[1])) / determinant,
        ((-matrix[1][0] * vector[0]) + (matrix[0][0] * vector[1])) / determinant,
    )


def inverse_2x2(matrix: list[list[float]]) -> list[list[float]]:
    col0 = solve_2x2(matrix, (1.0, 0.0))
    col1 = solve_2x2(matrix, (0.0, 1.0))
    return [
        [col0[0], col1[0]],
        [col0[1], col1[1]],
    ]


def add_2x2(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
    return [
        [left[0][0] + right[0][0], left[0][1] + right[0][1]],
        [left[1][0] + right[1][0], left[1][1] + right[1][1]],
    ]


def covariance_intersection_two(
    first_mean: tuple[float, float],
    first_covariance: list[list[float]],
    second_mean: tuple[float, float],
    second_covariance: list[list[float]],
) -> tuple[tuple[float, float], list[list[float]], float]:
    """Fuse two possibly correlated stationary-object estimates using covariance intersection.

    The bounded deterministic grid search chooses the weight that minimizes log(det(P)).
    Pairwise covariance intersection for more than two contributions is applied in stable
    source-track order by the caller, so the result can depend on contribution ordering.
    """
    first_information = inverse_2x2(first_covariance)
    second_information = inverse_2x2(second_covariance)
    best: tuple[float, float, tuple[float, float], list[list[float]]] | None = None
    objective_tolerance = 1e-12

    for index in range(COVARIANCE_INTERSECTION_STEPS):
        weight = index / (COVARIANCE_INTERSECTION_STEPS - 1)
        information = [
            [
                (weight * first_information[0][0]) + ((1.0 - weight) * second_information[0][0]),
                (weight * first_information[0][1]) + ((1.0 - weight) * second_information[0][1]),
            ],
            [
                (weight * first_information[1][0]) + ((1.0 - weight) * second_information[1][0]),
                (weight * first_information[1][1]) + ((1.0 - weight) * second_information[1][1]),
            ],
        ]
        covariance = inverse_2x2(information)
        information_vector = (
            (weight * ((first_information[0][0] * first_mean[0]) + (first_information[0][1] * first_mean[1])))
            + ((1.0 - weight) * ((second_information[0][0] * second_mean[0]) + (second_information[0][1] * second_mean[1]))),
            (weight * ((first_information[1][0] * first_mean[0]) + (first_information[1][1] * first_mean[1])))
            + ((1.0 - weight) * ((second_information[1][0] * second_mean[0]) + (second_information[1][1] * second_mean[1]))),
        )
        mean = (
            (covariance[0][0] * information_vector[0]) + (covariance[0][1] * information_vector[1]),
            (covariance[1][0] * information_vector[0]) + (covariance[1][1] * information_vector[1]),
        )
        objective = math.log(matrix_det_2x2(covariance))

        if (
            best is None
            or objective < best[0] - objective_tolerance
            or (abs(objective - best[0]) <= objective_tolerance and weight < best[1])
        ):
            # Equal or numerically indistinguishable objectives choose the smaller weight.
            # Because x = P[w I1 x1 + (1-w) I2 x2], this deterministic tie-break favors
            # the second estimate when covariances provide no log-det preference.
            best = (objective, weight, mean, covariance)

    if best is None:
        raise ValueError("Covariance-intersection search produced no estimate.")

    return best[2], best[3], best[0]


def mahalanobis_squared(
    residual: tuple[float, float],
    covariance: list[list[float]],
) -> float:
    solved = solve_2x2(covariance, residual)
    return (residual[0] * solved[0]) + (residual[1] * solved[1])


def stable_gaussian_pair(seed: int, *parts: object) -> tuple[float, float]:
    raw_seed = ":".join(str(part) for part in (seed, *parts))
    digest = hashlib.sha256(raw_seed.encode("utf-8")).hexdigest()
    rng = random.Random(int(digest[:16], 16))
    return rng.gauss(0.0, 1.0), rng.gauss(0.0, 1.0)


def source_local_track_id(seed: int, drone_id: int, truth_object_id: str) -> str:
    raw_id = f"{seed}:{drone_id}:{truth_object_id}"
    return hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:12]


def noisy_measurement_for_truth(
    seed: int,
    drone_id: int,
    truth_object_id: str,
    observation_time_bucket: int,
    latitude: float,
    longitude: float,
) -> tuple[float, float, list[list[float]]]:
    sensor_model = SENSOR_MODELS[drone_id]
    covariance = covariance_from_sensor_model(sensor_model)
    true_east_m, true_north_m = lat_lon_to_local_m(
        MAP_REFERENCE_LATITUDE,
        MAP_REFERENCE_LONGITUDE,
        latitude,
        longitude,
    )
    z0, z1 = stable_gaussian_pair(seed, drone_id, truth_object_id, observation_time_bucket)
    a = covariance[0][0]
    b = covariance[1][0]
    c = covariance[1][1]
    l00 = math.sqrt(a)
    l10 = b / l00
    l11 = math.sqrt(max(c - (l10 * l10), 0.0))

    bias_east_m = sensor_model.bias_east_m if SENSOR_BIAS_ENABLED else 0.0
    bias_north_m = sensor_model.bias_north_m if SENSOR_BIAS_ENABLED else 0.0

    return (
        true_east_m + bias_east_m + (l00 * z0),
        true_north_m + bias_north_m + (l10 * z0) + (l11 * z1),
        covariance,
    )


def resolve_priority_object_count(seed: int, requested_priority_count: int = RANDOM_PRIORITY_OBJECT_COUNT) -> int:
    if requested_priority_count == RANDOM_PRIORITY_OBJECT_COUNT:
        return random.Random(f"{seed}:priority-count").randint(MIN_PRIORITY_OBJECT_COUNT, MAX_PRIORITY_OBJECT_COUNT)

    return max(MIN_PRIORITY_OBJECT_COUNT, min(MAX_PRIORITY_OBJECT_COUNT, requested_priority_count))


def priority_signature_for(seed: int, priority_index: int) -> str:
    return f"{PRIORITY_SIGNATURE_PREFIX}-{hashlib.sha256(f'{seed}:{priority_index}'.encode('utf-8')).hexdigest()[:10]}"


def task_id_for_priority_signature(
    scenario_seed: int,
    signature_id: str,
    task_type: str = TASK_TYPE_ISR_OBSERVE,
) -> str:
    if task_type == TASK_TYPE_ISR_OBSERVE:
        return hashlib.sha256(f"{scenario_seed}:{signature_id}".encode("utf-8")).hexdigest()[:18]

    return hashlib.sha256(f"{scenario_seed}:{task_type}:{signature_id}".encode("utf-8")).hexdigest()[:18]


def task_id_for_contact(scenario_seed: int, contact_ref: str, task_type: str) -> str:
    return task_id_for_priority_signature(scenario_seed, contact_ref, task_type)


def generate_moving_handoff_objects() -> list[dict[str, float | str | bool]]:
    objects: list[dict[str, float | str | bool]] = []

    for object_id, label, east_m, north_m in moving_handoff_background_points():
        latitude, longitude = local_m_to_lat_lon(east_m, north_m)
        objects.append(
            {
                "id": object_id,
                "label": label,
                "latitude": latitude,
                "longitude": longitude,
                "priority": False,
                "moving": False,
            },
        )

    east_m, north_m = moving_person_position(0.0)
    latitude, longitude = local_m_to_lat_lon(east_m, north_m)
    objects.append(
        {
            "id": "private-moving-person",
            "label": "person",
            "latitude": latitude,
            "longitude": longitude,
            "priority": False,
            "moving": True,
            "trajectory": moving_person_trajectory(),
        },
    )
    return objects


def generate_ground_truth_objects(
    seed: int,
    requested_priority_count: int = RANDOM_PRIORITY_OBJECT_COUNT,
    scenario_id: str = DEFAULT_SCENARIO_ID,
) -> list[dict[str, float | str | bool]]:
    if scenario_id == MOVING_HANDOFF_SCENARIO_ID:
        return generate_moving_handoff_objects()

    rng = random.Random(seed)
    routes = generate_patrol_routes(seed)
    shared_east_m = routes[1].waypoints_m[3][0]
    shared_north_m = (routes[1].waypoints_m[3][1] + routes[2].waypoints_m[3][1]) / 2.0
    priority_count = resolve_priority_object_count(seed, requested_priority_count)
    priority_index = 0
    objects: list[dict[str, float | str | bool]] = []

    def add_object(
        internal_id: str,
        label: str,
        east_m: float,
        north_m: float,
        *,
        priority: bool = False,
    ) -> None:
        nonlocal priority_index
        latitude, longitude = local_m_to_lat_lon(east_m, north_m)
        item: dict[str, float | str | bool] = {
            "id": internal_id,
            "label": label,
            "latitude": latitude,
            "longitude": longitude,
            "priority": priority,
        }

        if priority:
            item["priority_signature"] = priority_signature_for(seed, priority_index)
            item["priority_rank"] = priority_index + 1
            priority_index += 1

        objects.append(item)

    priority_candidates = [
        ("shared-rb-00", "vehicle", shared_east_m + 45.0, shared_north_m),
        ("shared-rb-01", "generator", shared_east_m + 82.0, shared_north_m + 8.0),
        ("shared-rbg-02", "antenna", shared_east_m + 8.0, shared_north_m + 88.0),
        ("red-route-03", "equipment case", routes[1].waypoints_m[4][0] + 70.0, routes[1].waypoints_m[4][1] + 12.0),
        ("blue-route-04", "storage crate", routes[2].waypoints_m[4][0] + 70.0, routes[2].waypoints_m[4][1] - 12.0),
    ]

    for candidate_index, (suffix, label, east_m, north_m) in enumerate(priority_candidates):
        add_object(
            f"truth-{seed}-priority-{suffix}",
            label,
            east_m,
            north_m,
            priority=candidate_index < priority_count,
        )

    object_index = 3
    for drone_id in sorted(DRONE_PROFILES):
        for local_index, future_time in enumerate((1.0, 7.0, 18.0, 34.0, 48.0, 64.0, 82.0)):
            latitude, longitude, heading = patrol_pose_on_route(routes[drone_id], future_time)
            east_m, north_m = lat_lon_to_local_m(MAP_REFERENCE_LATITUDE, MAP_REFERENCE_LONGITUDE, latitude, longitude)
            along_track_m = rng.uniform(75.0, 145.0)
            cross_track_m = rng.uniform(-42.0, 42.0)
            heading_rad = math.radians(heading)
            object_east_m = (east_m + (math.sin(heading_rad) * along_track_m)) + (math.cos(heading_rad) * cross_track_m)
            object_north_m = (north_m + (math.cos(heading_rad) * along_track_m)) - (math.sin(heading_rad) * cross_track_m)
            label = OBJECT_LABELS[(local_index + drone_id + rng.randrange(len(OBJECT_LABELS))) % len(OBJECT_LABELS)]
            add_object(f"truth-{seed}-{drone_id}-{local_index:02d}", label, object_east_m, object_north_m)
            object_index += 1

    while len(objects) < SCENARIO_OBJECT_COUNT:
        drone_id = ((len(objects) - 3) % len(DRONE_PROFILES)) + 1
        latitude, longitude, heading = patrol_pose_on_route(routes[drone_id], 95.0 + (len(objects) * 4.0))
        east_m, north_m = lat_lon_to_local_m(MAP_REFERENCE_LATITUDE, MAP_REFERENCE_LONGITUDE, latitude, longitude)
        heading_rad = math.radians(heading)
        object_east_m = east_m + (math.sin(heading_rad) * rng.uniform(90.0, 145.0)) + rng.uniform(-35.0, 35.0)
        object_north_m = north_m + (math.cos(heading_rad) * rng.uniform(90.0, 145.0)) + rng.uniform(-35.0, 35.0)
        label = OBJECT_LABELS[rng.randrange(len(OBJECT_LABELS))]
        add_object(f"truth-{seed}-fill-{len(objects):02d}", label, object_east_m, object_north_m)

    return objects[:SCENARIO_OBJECT_COUNT]


def lat_lon_to_local_m(
    reference_latitude: float,
    reference_longitude: float,
    latitude: float,
    longitude: float,
) -> tuple[float, float]:
    return (
        (longitude - reference_longitude) * meters_per_deg_lon(reference_latitude),
        (latitude - reference_latitude) * EARTH_METERS_PER_DEG_LAT,
    )


def normalize_angle_deg(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def move_toward(current: float, target: float, maximum_delta: float) -> float:
    if abs(target - current) <= maximum_delta:
        return target

    return current + (maximum_delta if target > current else -maximum_delta)


def move_angle_toward(current_deg: float, target_deg: float, maximum_delta_deg: float) -> float:
    delta = normalize_angle_deg(target_deg - current_deg)

    if abs(delta) <= maximum_delta_deg:
        return target_deg % 360.0

    return (current_deg + (maximum_delta_deg if delta > 0 else -maximum_delta_deg)) % 360.0


class DroneSimulator:
    """Own one drone's telemetry, observation store, object map, and peer transport."""

    def __init__(
        self,
        drone_id: int,
        name: str,
        peers: list[str],
        scenario_seed: int = DEFAULT_SCENARIO_SEED,
        requested_priority_count: int = RANDOM_PRIORITY_OBJECT_COUNT,
        selected_playbook: str = DEFAULT_PLAYBOOK,
        api_host: str = "127.0.0.1",
        api_port: int = 0,
        scenario_id: str = DEFAULT_SCENARIO_ID,
        initial_observer_id: int = DEFAULT_INITIAL_OBSERVER_ID,
        next_observer_id: int = DEFAULT_NEXT_OBSERVER_ID,
        scenario_run_id: str | None = None,
        team_configuration: dict[int, TeamMemberConfiguration] | None = None,
        payload_requests_enabled: bool = DEFAULT_PAYLOAD_REQUESTS_ENABLED,
        requested_payload_type: str = DEFAULT_PAYLOAD_TYPE,
        requested_payload_quantity: int = DEFAULT_PAYLOAD_QUANTITY,
        payload_standoff_m: float = DEFAULT_PAYLOAD_STANDOFF_M,
    ) -> None:
        profile = DRONE_PROFILES[drone_id]
        self.drone_id = drone_id
        self.name = name
        self.color = str(profile["color"])
        self.base_latitude = float(profile["latitude"])
        self.base_longitude = float(profile["longitude"])
        self.base_heading_deg = float(profile["heading_deg"])
        self.peers = peers
        self.scenario_seed = scenario_seed
        self.requested_priority_count = requested_priority_count
        self.selected_playbook = selected_playbook if selected_playbook in ENABLED_PLAYBOOKS else DEFAULT_PLAYBOOK
        self.api_host = api_host
        self.api_port = api_port
        self.hostname = socket.gethostname()
        self.process_id = os.getpid()
        self.playbook_config_version = 0
        self.scenario_id = scenario_id if scenario_id in AVAILABLE_SCENARIOS else DEFAULT_SCENARIO_ID
        self.priority_object_count = (
            0
            if self.scenario_id == MOVING_HANDOFF_SCENARIO_ID
            else resolve_priority_object_count(scenario_seed, requested_priority_count)
        )
        self.initial_observer_id = initial_observer_id if initial_observer_id in DRONE_PROFILES else DEFAULT_INITIAL_OBSERVER_ID
        self.next_observer_id = next_observer_id if next_observer_id in DRONE_PROFILES else DEFAULT_NEXT_OBSERVER_ID
        if self.next_observer_id == self.initial_observer_id:
            self.next_observer_id = next(
                candidate_id
                for candidate_id in sorted(DRONE_PROFILES)
                if candidate_id != self.initial_observer_id
            )
        self.scenario_run_id = scenario_run_id or str(scenario_seed)
        self.team_configuration = (
            team_configuration[drone_id]
            if team_configuration is not None and drone_id in team_configuration
            else TeamMemberConfiguration.legacy_default()
        )
        self.payload_requests_enabled = bool(payload_requests_enabled)
        self.requested_payload_type = requested_payload_type
        self.requested_payload_quantity = max(1, int(requested_payload_quantity))
        self.payload_standoff_m = max(1.0, float(payload_standoff_m))
        self.node_session_id = f"{self.drone_id}-{uuid.uuid4().hex[:12]}"
        self.routes = generate_patrol_routes(scenario_seed)
        self.truth_objects = generate_ground_truth_objects(scenario_seed, requested_priority_count, self.scenario_id)

        self.lock = threading.RLock()
        self.last_wall_time = time.monotonic()
        self.simulation_time = 0.0
        self.flight_trail: list[dict[str, float]] = []
        self.next_trail_sample_time = 0.0
        self.paused = False
        self.sharing_enabled = False
        self.observations: dict[str, Observation] = {}
        self.objects: dict[str, dict[str, object]] = {}
        self.track_to_object: dict[str, str] = {}
        self.unresolved_observations: dict[str, dict[str, object]] = {}
        self.moving_report_buffer: list[Observation] = []
        self.moving_report_ids: set[str] = set()
        self.moving_committed_report_ids: set[str] = set()
        self.moving_rejected_report_count = 0
        self.moving_processing_epoch = -math.inf
        self.response_tasks: dict[str, dict[str, object]] = {}
        self.response_task_queue: list[str] = []
        self.active_response_task_id: str | None = None
        self.claims: dict[str, dict[int, dict[str, object]]] = {}
        self.outbound_claims: list[dict[str, object]] = []
        self.cancelled_priority_signatures: set[str] = set()
        self.cancelled_task_ids: set[str] = set()
        self.payload_reservations_by_task: dict[str, dict[str, object]] = {}
        self.response_east_m: float | None = None
        self.response_north_m: float | None = None
        self.response_heading_deg: float | None = None
        self.response_camera_yaw_deg: float | None = None
        self.return_east_m: float | None = None
        self.return_north_m: float | None = None
        self.return_heading_deg: float | None = None
        self.orbit_phase_deg = 0.0
        self.last_local_observation: Observation | None = None
        self.local_observations_generated = 0
        self.peer_observations_sent = 0
        self.peer_observations_received = 0
        self.unique_observations_accepted = 0
        self.duplicate_observations_ignored = 0
        self.validation_rejects = 0
        self.observation_trace_records: dict[str, dict[str, object]] = {}
        self.observation_trace_order: deque[str] = deque()
        self.observation_trace_evictions = 0
        self.peer_observation_messages_sent = 0
        self.peer_observation_messages_received = 0
        self.peer_claim_messages_sent = 0
        self.peer_claim_messages_received = 0
        self.peer_payload_bytes_sent = 0
        self.peer_payload_bytes_received = 0
        self.peer_payload_byte_events: deque[tuple[float, int, str]] = deque()
        self.peer_send_failures = 0
        self.most_recent_received_observation_id: str | None = None
        self.most_recent_received_source: str | None = None
        self.most_recent_sent_observation_id: str | None = None
        self.peer_connection_state: dict[str, dict[str, object]] = {
            peer: {
                "peer": peer,
                "state": "not_contacted",
                "last_error": None,
                "last_success_time": None,
                "last_success_wall_time": None,
                "last_rtt_ms": None,
                "observation_messages_sent": 0,
                "claim_messages_sent": 0,
                "payload_bytes_sent": 0,
                "send_failures": 0,
                "latest_error": None,
            }
            for peer in peers
        }

    def _advance_locked(self, update_flight_trail: bool = True) -> None:
        now = time.monotonic()
        elapsed = now - self.last_wall_time

        if not math.isfinite(elapsed) or elapsed < 0.0:
            elapsed = 0.0

        if not self.paused:
            self.simulation_time += elapsed
            self._reconcile_eligible_work_locked()
            self._resolve_response_tasks_locked()
            self._update_response_guidance_locked(elapsed)
            self._update_payload_guidance_locked(elapsed)
            self._update_return_to_patrol_locked(elapsed)
            self._process_moving_report_buffer_locked()

            if update_flight_trail:
                self._update_flight_trail_locked()
        self.last_wall_time = now

    def _pose_at_locked(self) -> tuple[float, float, float]:
        if self._has_local_execution_control_locked():
            assert self.response_east_m is not None
            assert self.response_north_m is not None
            assert self.response_heading_deg is not None
            latitude, longitude = local_m_to_lat_lon(self.response_east_m, self.response_north_m)
            return latitude, longitude, self.response_heading_deg

        if self.return_east_m is not None and self.return_north_m is not None and self.return_heading_deg is not None:
            latitude, longitude = local_m_to_lat_lon(self.return_east_m, self.return_north_m)
            return latitude, longitude, self.return_heading_deg

        if self.scenario_id == MOVING_HANDOFF_SCENARIO_ID:
            pose = moving_handoff_pose_for(self.drone_id, self.initial_observer_id, self.next_observer_id)
            latitude, longitude = local_m_to_lat_lon(pose.east_m, pose.north_m)
            return latitude, longitude, pose.heading_deg

        return patrol_pose_on_route(self.routes[self.drone_id], self.simulation_time)

    def _pose_at_time_locked(self, simulation_time: float) -> tuple[float, float, float]:
        if self.scenario_id == MOVING_HANDOFF_SCENARIO_ID:
            pose = moving_handoff_pose_for(self.drone_id, self.initial_observer_id, self.next_observer_id)
            latitude, longitude = local_m_to_lat_lon(pose.east_m, pose.north_m)
            return latitude, longitude, pose.heading_deg

        return patrol_pose_on_route(self.routes[self.drone_id], simulation_time)

    def _refresh_moving_truth_objects_locked(self) -> None:
        if self.scenario_id != MOVING_HANDOFF_SCENARIO_ID:
            return

        east_m, north_m = moving_person_position(self.simulation_time)
        latitude, longitude = local_m_to_lat_lon(east_m, north_m)

        for truth_object in self.truth_objects:
            if bool(truth_object.get("moving", False)):
                truth_object["latitude"] = latitude
                truth_object["longitude"] = longitude
                truth_object["east_m"] = east_m
                truth_object["north_m"] = north_m
                truth_object["trajectory"] = moving_person_trajectory()

    def _update_flight_trail_locked(self) -> None:
        while self.next_trail_sample_time <= self.simulation_time:
            latitude, longitude, _heading = self._pose_at_locked()
            self.flight_trail.append(
                {
                    "latitude": latitude,
                    "longitude": longitude,
                    "timestamp": self.next_trail_sample_time,
                },
            )
            self.next_trail_sample_time += TRAIL_SAMPLE_INTERVAL_SECONDS

        cutoff = self.simulation_time - TRAIL_RETENTION_SECONDS
        self.flight_trail = [
            point
            for point in self.flight_trail
            if float(point["timestamp"]) > cutoff
        ][-int((TRAIL_RETENTION_SECONDS / TRAIL_SAMPLE_INTERVAL_SECONDS) + 2):]

    def _camera_yaw_locked(self) -> float:
        if self._has_local_response_control_locked() and self.response_camera_yaw_deg is not None:
            return self.response_camera_yaw_deg

        _latitude, _longitude, heading = self._pose_at_locked()
        return heading

    def _camera_footprint_locked(self) -> list[dict[str, float]]:
        latitude, longitude, _heading = self._pose_at_locked()
        camera_yaw = self._camera_yaw_locked()
        points = [{"latitude": latitude, "longitude": longitude}]

        for angle in (camera_yaw - CAMERA_HALF_ANGLE_DEG, camera_yaw + CAMERA_HALF_ANGLE_DEG):
            angle_rad = math.radians(angle)
            east_m = math.sin(angle_rad) * CAMERA_RANGE_M
            north_m = math.cos(angle_rad) * CAMERA_RANGE_M
            point_latitude, point_longitude = offset_lat_lon(latitude, longitude, east_m, north_m)
            points.append({"latitude": point_latitude, "longitude": point_longitude})

        return points

    def _is_in_camera_footprint_locked(self, latitude: float, longitude: float) -> bool:
        drone_latitude, drone_longitude, _heading = self._pose_at_locked()
        heading = self._camera_yaw_locked()
        east_m, north_m = lat_lon_to_local_m(drone_latitude, drone_longitude, latitude, longitude)
        distance_m = math.hypot(east_m, north_m)

        if distance_m > CAMERA_RANGE_M:
            return False

        bearing_deg = math.degrees(math.atan2(east_m, north_m)) % 360.0
        return abs(normalize_angle_deg(bearing_deg - heading)) <= CAMERA_HALF_ANGLE_DEG

    def status(self) -> DroneStatus:
        with self.lock:
            self._advance_locked()
            latitude, longitude, heading = self._pose_at_locked()
            altitude = 95.0 + (self.drone_id * 12.0) + (math.sin(self.simulation_time / 6.0) * 6.0)
            speed = 2.5 + abs(math.cos(self.simulation_time / 5.0))
            battery = max(0.0, 100.0 - (self.simulation_time * 0.02) - (self.drone_id * 1.25))

            return DroneStatus(
                schema_version=SCHEMA_VERSION,
                app_build=APPLICATION_BUILD,
                scenario_name=MOVING_HANDOFF_SCENARIO_NAME if self.scenario_id == MOVING_HANDOFF_SCENARIO_ID else SCENARIO_NAME,
                scenario_seed=self.scenario_seed,
                id=self.drone_id,
                name=self.name,
                state="online",
                latitude=latitude,
                longitude=longitude,
                altitude_m=altitude,
                speed_mps=speed,
                heading_deg=heading,
                camera_yaw_deg=self._camera_yaw_locked(),
                battery_percent=battery,
                timestamp=self.simulation_time,
                paused=self.paused,
                sharing_enabled=self.sharing_enabled,
                selected_playbook=self.selected_playbook,
                playbook_config_version=self.playbook_config_version,
                requested_priority_count=self.requested_priority_count,
                priority_object_count=self.priority_object_count,
                camera_footprint=self._camera_footprint_locked(),
                flight_trail=list(self.flight_trail),
                response_task=self._visible_response_task_locked(),
                response_tasks=self._visible_response_tasks_locked(),
                response_queue_depth=len(self.response_task_queue),
                orbit_radius_m=ORBIT_RADIUS_M,
                current_mode=self._current_mode_locked(),
                assigned_objective=self._assigned_objective_locked(),
                diagnostics=self._diagnostics_locked(),
                team_configuration=self.team_configuration.to_dict(),
            )

    def truth_payload(self) -> dict[str, object]:
        with self.lock:
            self._advance_locked()
            self._refresh_moving_truth_objects_locked()
            return {
                "schema_version": SCHEMA_VERSION,
                "app_build": APPLICATION_BUILD,
                "scenario_name": MOVING_HANDOFF_SCENARIO_NAME if self.scenario_id == MOVING_HANDOFF_SCENARIO_ID else SCENARIO_NAME,
                "scenario_seed": self.scenario_seed,
                "scenario_id": self.scenario_id,
                "scenario_run_id": self.scenario_run_id,
                "initial_observer_id": self.initial_observer_id,
                "next_observer_id": self.next_observer_id,
                "selected_playbook": self.selected_playbook,
                "playbook_config_version": self.playbook_config_version,
                "requested_priority_count": self.requested_priority_count,
                "priority_object_count": self.priority_object_count,
                "simulation_time": self.simulation_time,
                "object_count": len(self.truth_objects),
                "priority_discovered_count": self._priority_discovered_count_locked(),
                "response_task": self._visible_response_task_locked(),
                "response_tasks": self._visible_response_tasks_locked(),
                "response_queue_depth": len(self.response_task_queue),
                "orbit_radius_m": ORBIT_RADIUS_M,
                "payload_requests_enabled": self.payload_requests_enabled,
                "requested_payload_type": self.requested_payload_type,
                "requested_payload_quantity": self.requested_payload_quantity,
                "payload_standoff_m": self.payload_standoff_m,
                "payload_reservations": dict(self.payload_reservations_by_task),
                "response_limitation_note": RESPONSE_LIMITATION_NOTE,
                "objects": list(self.truth_objects),
            }

    def objects_payload(self) -> dict[str, object]:
        with self.lock:
            self._advance_locked()
            objects = []

            for item in self.objects.values():
                if item.get("tracking_model") == "constant_velocity":
                    self._refresh_moving_track_payload_locked(item)
                else:
                    self._refresh_fused_object_locked(item)
                object_payload = dict(item)
                age_seconds = max(0.0, self.simulation_time - float(object_payload["last_seen"]))
                object_payload["age_seconds"] = age_seconds
                object_payload["stale_after_seconds"] = STALE_AFTER_SECONDS
                object_payload["contribution_max_age_seconds"] = CONTRIBUTION_MAX_AGE_SECONDS
                object_payload["stale"] = age_seconds >= STALE_AFTER_SECONDS
                objects.append(object_payload)

            return {
                "schema_version": SCHEMA_VERSION,
                "app_build": APPLICATION_BUILD,
                "scenario_name": MOVING_HANDOFF_SCENARIO_NAME if self.scenario_id == MOVING_HANDOFF_SCENARIO_ID else SCENARIO_NAME,
                "scenario_seed": self.scenario_seed,
                "scenario_id": self.scenario_id,
                "scenario_run_id": self.scenario_run_id,
                "initial_observer_id": self.initial_observer_id,
                "next_observer_id": self.next_observer_id,
                "drone_id": self.drone_id,
                "drone_name": self.name,
                "simulation_time": self.simulation_time,
                "sharing_enabled": self.sharing_enabled,
                "paused": self.paused,
                "selected_playbook": self.selected_playbook,
                "playbook_config_version": self.playbook_config_version,
                "requested_priority_count": self.requested_priority_count,
                "priority_object_count": self.priority_object_count,
                "response_task": self._visible_response_task_locked(),
                "response_tasks": self._visible_response_tasks_locked(),
                "response_queue_depth": len(self.response_task_queue),
                "orbit_radius_m": ORBIT_RADIUS_M,
                "payload_requests_enabled": self.payload_requests_enabled,
                "requested_payload_type": self.requested_payload_type,
                "requested_payload_quantity": self.requested_payload_quantity,
                "payload_standoff_m": self.payload_standoff_m,
                "payload_reservations": dict(self.payload_reservations_by_task),
                "objects": sorted(
                    objects,
                    key=lambda item: str(item["object_id"]),
                ),
                "unresolved_observation_count": len(self.unresolved_observations),
                "unresolved_observations": list(self.unresolved_observations.values())[-25:],
                "moving_report_buffer_depth": len(self.moving_report_buffer),
                "moving_rejected_report_count": self.moving_rejected_report_count,
            }

    def control_payload(self) -> dict[str, object]:
        with self.lock:
            return {
                "schema_version": SCHEMA_VERSION,
                "app_build": APPLICATION_BUILD,
                "scenario_name": MOVING_HANDOFF_SCENARIO_NAME if self.scenario_id == MOVING_HANDOFF_SCENARIO_ID else SCENARIO_NAME,
                "scenario_seed": self.scenario_seed,
                "scenario_id": self.scenario_id,
                "scenario_run_id": self.scenario_run_id,
                "initial_observer_id": self.initial_observer_id,
                "next_observer_id": self.next_observer_id,
                "drone_id": self.drone_id,
                "paused": self.paused,
                "sharing_enabled": self.sharing_enabled,
                "selected_playbook": self.selected_playbook,
                "playbook_config_version": self.playbook_config_version,
                "requested_priority_count": self.requested_priority_count,
                "priority_object_count": self.priority_object_count,
                "simulation_time": self.simulation_time,
                "observation_count": len(self.observations),
                "object_count": len(self.objects),
                "response_task": self._visible_response_task_locked(),
                "response_tasks": self._visible_response_tasks_locked(),
                "response_queue_depth": len(self.response_task_queue),
                "unresolved_observation_count": len(self.unresolved_observations),
                "payload_requests_enabled": self.payload_requests_enabled,
                "requested_payload_type": self.requested_payload_type,
                "requested_payload_quantity": self.requested_payload_quantity,
                "payload_standoff_m": self.payload_standoff_m,
                "payload_reservations": dict(self.payload_reservations_by_task),
                "diagnostics": self._diagnostics_locked(),
                "team_configuration": self.team_configuration.to_dict(),
            }

    def set_paused(self, paused: bool) -> None:
        with self.lock:
            self._advance_locked()
            self.paused = paused
            self.last_wall_time = time.monotonic()

    def set_sharing_enabled(self, sharing_enabled: bool) -> None:
        with self.lock:
            self.sharing_enabled = sharing_enabled

    def set_simulation_time(self, simulation_time: float) -> bool:
        if not math.isfinite(simulation_time):
            return False

        with self.lock:
            if simulation_time < self.simulation_time:
                return False

            self.simulation_time = simulation_time
            self.last_wall_time = time.monotonic()
            self._process_moving_report_buffer_locked()
            return True

    def set_playbook(self, selected_playbook: str, config_version: int | None = None) -> bool:
        if selected_playbook not in ENABLED_PLAYBOOKS:
            return False

        with self.lock:
            self._advance_locked(update_flight_trail=False)
            next_version = max(self.playbook_config_version + 1, int(config_version or 0))

            if selected_playbook == self.selected_playbook and next_version <= self.playbook_config_version:
                return True

            previous_playbook = self.selected_playbook
            self.selected_playbook = selected_playbook
            self.playbook_config_version = next_version

            if previous_playbook == PLAYBOOK_REPORT_CONTINUE and selected_playbook == PLAYBOOK_OBSERVE_ORBIT:
                self.outbound_claims.clear()
                self._make_discovered_priorities_available_locked()
            elif previous_playbook == PLAYBOOK_OBSERVE_ORBIT and selected_playbook == PLAYBOOK_REPORT_CONTINUE:
                self._release_active_response_tasks_locked()

            return True

    def configure_payload_requests(
        self,
        enabled: bool | None = None,
        payload_type: str | None = None,
        quantity: int | None = None,
        stand_off_m: float | None = None,
    ) -> None:
        with self.lock:
            if enabled is not None:
                self.payload_requests_enabled = bool(enabled)

            if payload_type is not None:
                payload_type = payload_type.strip()
                if payload_type:
                    self.requested_payload_type = payload_type

            if quantity is not None:
                self.requested_payload_quantity = max(1, int(quantity))

            if stand_off_m is not None:
                self.payload_standoff_m = max(1.0, float(stand_off_m))

    def reset(
        self,
        scenario_seed: int | None = None,
        requested_priority_count: int | None = None,
        selected_playbook: str | None = None,
        scenario_id: str | None = None,
        initial_observer_id: int | None = None,
        next_observer_id: int | None = None,
        scenario_run_id: str | None = None,
    ) -> None:
        with self.lock:
            if scenario_seed is not None:
                self.scenario_seed = scenario_seed

            if requested_priority_count is not None:
                self.requested_priority_count = requested_priority_count

            if selected_playbook is not None:
                self.selected_playbook = selected_playbook if selected_playbook in ENABLED_PLAYBOOKS else DEFAULT_PLAYBOOK

            if scenario_id is not None:
                self.scenario_id = scenario_id if scenario_id in AVAILABLE_SCENARIOS else DEFAULT_SCENARIO_ID

            candidate_initial_observer_id = self.initial_observer_id
            candidate_next_observer_id = self.next_observer_id

            if initial_observer_id is not None and initial_observer_id in DRONE_PROFILES:
                candidate_initial_observer_id = initial_observer_id

            if next_observer_id is not None and next_observer_id in DRONE_PROFILES:
                candidate_next_observer_id = next_observer_id

            if candidate_next_observer_id == candidate_initial_observer_id:
                candidate_next_observer_id = next(
                    candidate_id
                    for candidate_id in sorted(DRONE_PROFILES)
                    if candidate_id != candidate_initial_observer_id
                )

            self.initial_observer_id = candidate_initial_observer_id
            self.next_observer_id = candidate_next_observer_id

            self.scenario_run_id = scenario_run_id or new_run_id(self.scenario_seed)
            self.node_session_id = f"{self.drone_id}-{uuid.uuid4().hex[:12]}"
            self.priority_object_count = (
                0
                if self.scenario_id == MOVING_HANDOFF_SCENARIO_ID
                else resolve_priority_object_count(self.scenario_seed, self.requested_priority_count)
            )
            self.routes = generate_patrol_routes(self.scenario_seed)
            self.truth_objects = generate_ground_truth_objects(self.scenario_seed, self.requested_priority_count, self.scenario_id)
            self.last_wall_time = time.monotonic()
            self.simulation_time = 0.0
            self.flight_trail.clear()
            self.next_trail_sample_time = 0.0
            self.paused = False
            self.sharing_enabled = False
            self.observations.clear()
            self.objects.clear()
            self.track_to_object.clear()
            self.unresolved_observations.clear()
            self.moving_report_buffer.clear()
            self.moving_report_ids.clear()
            self.moving_committed_report_ids.clear()
            self.moving_rejected_report_count = 0
            self.moving_processing_epoch = -math.inf
            self.response_tasks.clear()
            self.response_task_queue.clear()
            self.active_response_task_id = None
            self.claims.clear()
            self.outbound_claims.clear()
            self.cancelled_priority_signatures.clear()
            self.cancelled_task_ids.clear()
            self.payload_reservations_by_task.clear()
            self.response_east_m = None
            self.response_north_m = None
            self.response_heading_deg = None
            self.response_camera_yaw_deg = None
            self.return_east_m = None
            self.return_north_m = None
            self.return_heading_deg = None
            self.orbit_phase_deg = 0.0
            self.last_local_observation = None
            self.local_observations_generated = 0
            self.peer_observations_sent = 0
            self.peer_observations_received = 0
            self.unique_observations_accepted = 0
            self.duplicate_observations_ignored = 0
            self.validation_rejects = 0
            self.observation_trace_records.clear()
            self.observation_trace_order.clear()
            self.observation_trace_evictions = 0
            self.peer_observation_messages_sent = 0
            self.peer_observation_messages_received = 0
            self.peer_claim_messages_sent = 0
            self.peer_claim_messages_received = 0
            self.peer_payload_bytes_sent = 0
            self.peer_payload_bytes_received = 0
            self.peer_payload_byte_events.clear()
            self.peer_send_failures = 0
            self.most_recent_received_observation_id = None
            self.most_recent_received_source = None
            self.most_recent_sent_observation_id = None

            for peer_state in self.peer_connection_state.values():
                peer_state["state"] = "not_contacted"
                peer_state["last_error"] = None
                peer_state["last_success_time"] = None
                peer_state["last_success_wall_time"] = None
                peer_state["last_rtt_ms"] = None
                peer_state["observation_messages_sent"] = 0
                peer_state["claim_messages_sent"] = 0
                peer_state["payload_bytes_sent"] = 0
                peer_state["send_failures"] = 0
                peer_state["latest_error"] = None

    def capture_observations(self) -> list[Observation]:
        new_observations: list[Observation] = []

        with self.lock:
            self._advance_locked()

            if self.paused:
                return []

            self._refresh_moving_truth_objects_locked()

            for truth_object in self.truth_objects:
                latitude = float(truth_object["latitude"])
                longitude = float(truth_object["longitude"])

                if not self._is_in_camera_footprint_locked(latitude, longitude):
                    continue

                truth_object_id = str(truth_object["id"])
                observation_time_bucket = int(self.simulation_time)
                source_track_id = source_local_track_id(self.scenario_seed, self.drone_id, truth_object_id)
                observation_id = self._observation_id(source_track_id, observation_time_bucket)

                if observation_id in self.observations:
                    continue

                east_m, north_m, covariance = noisy_measurement_for_truth(
                    self.scenario_seed,
                    self.drone_id,
                    truth_object_id,
                    observation_time_bucket,
                    latitude,
                    longitude,
                )
                measured_latitude, measured_longitude = local_m_to_lat_lon(east_m, north_m)
                score = 0.88
                sensor_model = SENSOR_MODELS[self.drone_id]
                priority_signature = truth_object.get("priority_signature")
                priority_evidence = None

                if bool(truth_object.get("priority", False)) and priority_signature:
                    priority_evidence = {
                        "signature_id": str(priority_signature),
                        "signature_type": "mocked_search_signature_match",
                        "match_score": 0.97,
                        "model": "stage-a-mock-signature-v1",
                    }

                observation = Observation(
                    id=observation_id,
                    scenario_run_id=self.scenario_run_id,
                    source_drone_id=self.drone_id,
                    source_drone_name=self.name,
                    source_color=self.color,
                    sensor_id=sensor_model.sensor_id,
                    source_track_id=source_track_id,
                    timestamp=float(observation_time_bucket),
                    measurement_frame=MEASUREMENT_FRAME,
                    east_m=east_m,
                    north_m=north_m,
                    latitude=measured_latitude,
                    longitude=measured_longitude,
                    label=str(truth_object["label"]),
                    classification_score=score,
                    covariance=covariance,
                    priority_evidence=priority_evidence,
                    measurement_type=MOVING_TRACK_REPORT_TYPE if bool(truth_object.get("moving", False)) else "point",
                    object_id=f"{sensor_model.sensor_id}:{source_track_id}",
                    report_schema_version=TRACK_REPORT_SCHEMA_VERSION if bool(truth_object.get("moving", False)) else "legacy-observation-v1",
                    source_node_session_id=self.node_session_id,
                    report_id=observation_id,
                    state_epoch=float(observation_time_bucket),
                    sensor_track_status="active",
                    simulated=True,
                    supporting_observation_ids=[observation_id],
                )

                if self._apply_observation_locked(observation):
                    self.local_observations_generated += 1
                    self.last_local_observation = observation
                    new_observations.append(observation)

        return new_observations

    def receive_observation(self, payload: dict[str, object], payload_byte_count: int = 0) -> bool:
        with self.lock:
            self.peer_observations_received += 1
            self.peer_observation_messages_received += 1
            self._record_peer_payload_bytes_locked(payload_byte_count, "received")

        try:
            observation = self._observation_from_payload(payload)
        except (KeyError, TypeError, ValueError, IndexError):
            with self.lock:
                self.validation_rejects += 1
            return False

        with self.lock:
            self._advance_locked()
            self.most_recent_received_observation_id = observation.id
            self.most_recent_received_source = observation.source_drone_name
            return self._apply_observation_locked(observation)

    def broadcast_observation(self, observation: Observation) -> None:
        with self.lock:
            sharing_enabled = self.sharing_enabled

        if not sharing_enabled:
            return

        payload = json.dumps(asdict(observation)).encode("utf-8")

        for peer in self.peers:
            request = urllib.request.Request(
                f"{peer}/observations",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            started_at = time.monotonic()

            try:
                with urllib.request.urlopen(request, timeout=0.4):
                    pass
            except (OSError, urllib.error.URLError) as error:
                with self.lock:
                    self.peer_send_failures += 1
                    self.peer_connection_state[peer]["state"] = "error"
                    self.peer_connection_state[peer]["last_error"] = str(error)
                    self.peer_connection_state[peer]["latest_error"] = str(error)
                    self.peer_connection_state[peer]["send_failures"] = int(self.peer_connection_state[peer]["send_failures"]) + 1
                continue

            elapsed_ms = (time.monotonic() - started_at) * 1000.0
            with self.lock:
                self.peer_observations_sent += 1
                self.peer_observation_messages_sent += 1
                self._record_peer_payload_bytes_locked(len(payload), "sent")
                self.most_recent_sent_observation_id = observation.id
                self.peer_connection_state[peer]["state"] = "ok"
                self.peer_connection_state[peer]["last_error"] = None
                self.peer_connection_state[peer]["latest_error"] = None
                self.peer_connection_state[peer]["last_success_time"] = self.simulation_time
                self.peer_connection_state[peer]["last_success_wall_time"] = time.monotonic()
                self.peer_connection_state[peer]["last_rtt_ms"] = elapsed_ms
                self.peer_connection_state[peer]["observation_messages_sent"] = int(self.peer_connection_state[peer]["observation_messages_sent"]) + 1
                self.peer_connection_state[peer]["payload_bytes_sent"] = int(self.peer_connection_state[peer]["payload_bytes_sent"]) + len(payload)

    def redeliver_last_observation(self) -> bool:
        with self.lock:
            observation = self.last_local_observation

        if observation is None:
            return False

        self.broadcast_observation(observation)
        return True

    def redeliver_observation(self, observation_id: str) -> bool:
        with self.lock:
            observation = self.observations.get(observation_id)

        if observation is None:
            return False

        self.broadcast_observation(observation)
        return True

    def _observation_from_payload(self, payload: dict[str, object]) -> Observation:
        covariance = payload["covariance"]  # type: ignore[assignment]

        return Observation(
            id=str(payload["id"]),
            scenario_run_id=str(payload["scenario_run_id"]),
            source_drone_id=int(payload["source_drone_id"]),
            source_drone_name=str(payload["source_drone_name"]),
            source_color=str(payload["source_color"]),
            sensor_id=str(payload["sensor_id"]),
            source_track_id=str(payload["source_track_id"]),
            timestamp=float(payload["timestamp"]),
            measurement_frame=str(payload["measurement_frame"]),
            east_m=float(payload["east_m"]),
            north_m=float(payload["north_m"]),
            latitude=float(payload["latitude"]),
            longitude=float(payload["longitude"]),
            label=str(payload["label"]),
            classification_score=float(payload["classification_score"]),
            covariance=[
                [float(covariance[0][0]), float(covariance[0][1])],  # type: ignore[index]
                [float(covariance[1][0]), float(covariance[1][1])],  # type: ignore[index]
            ],
            priority_evidence=payload.get("priority_evidence") if isinstance(payload.get("priority_evidence"), dict) else None,
            measurement_type=str(payload.get("measurement_type", "point")),
            object_id=str(payload.get("object_id", "")),
            report_schema_version=str(payload.get("report_schema_version", "legacy-observation-v1")),
            source_node_session_id=str(payload.get("source_node_session_id", "")),
            report_id=str(payload.get("report_id", payload.get("id", ""))),
            state_epoch=float(payload["state_epoch"]) if payload.get("state_epoch") is not None else None,
            velocity_east_mps=float(payload["velocity_east_mps"]) if payload.get("velocity_east_mps") is not None else None,
            velocity_north_mps=float(payload["velocity_north_mps"]) if payload.get("velocity_north_mps") is not None else None,
            state_covariance=payload.get("state_covariance") if isinstance(payload.get("state_covariance"), list) else None,  # type: ignore[arg-type]
            sensor_track_status=str(payload.get("sensor_track_status", "active")),
            simulated=bool(payload.get("simulated", True)),
            supporting_observation_ids=[
                str(item)
                for item in payload.get("supporting_observation_ids", [])
            ] if isinstance(payload.get("supporting_observation_ids"), list) else None,
        )

    def _diagnostics_locked(self) -> dict[str, object]:
        return {
            "local_observations_generated": self.local_observations_generated,
            "peer_observations_sent": self.peer_observations_sent,
            "peer_observations_received": self.peer_observations_received,
            "unique_observations_accepted": self.unique_observations_accepted,
            "duplicates_ignored": self.duplicate_observations_ignored,
            "validation_rejects": self.validation_rejects,
            "unresolved_observation_count": len(self.unresolved_observations),
            "association_gate_d2": ASSOCIATION_GATE_D2,
            "association_ambiguity_d2_delta": ASSOCIATION_AMBIGUITY_D2_DELTA,
            "contribution_max_age_seconds": CONTRIBUTION_MAX_AGE_SECONDS,
            "measurement_frame": MEASUREMENT_FRAME,
            "peer_connection_state": list(self.peer_connection_state.values()),
            "most_recent_received_observation_id": self.most_recent_received_observation_id,
            "most_recent_received_source": self.most_recent_received_source,
            "most_recent_sent_observation_id": self.most_recent_sent_observation_id,
            "selected_playbook": self.selected_playbook,
            "playbook_config_version": self.playbook_config_version,
            "priority_object_count": self.priority_object_count,
            "priority_discovered_count": self._priority_discovered_count_locked(),
            "response_task": self._visible_response_task_locked(),
            "response_tasks": self._visible_response_tasks_locked(),
            "response_queue_depth": len(self.response_task_queue),
            "orbit_radius_m": ORBIT_RADIUS_M,
            "network": self._network_payload_locked(),
            "current_mode": self._current_mode_locked(),
            "assigned_objective": self._assigned_objective_locked(),
            "response_limitation_note": RESPONSE_LIMITATION_NOTE,
            "payload_drop": {
                "requests_enabled": self.payload_requests_enabled,
                "requested_payload_type": self.requested_payload_type,
                "requested_payload_quantity": self.requested_payload_quantity,
                "stand_off_m": self.payload_standoff_m,
                "reserved_inventory": dict(self.payload_reservations_by_task),
                "available_inventory_after_reservations": self._available_payload_inventory_locked(),
                "note": "Increment 2 supports approach to a shared contact estimate only; release, inventory consumption, and delivery confirmation are not implemented.",
            },
            "moving_tracking": {
                "report_schema_version": TRACK_REPORT_SCHEMA_VERSION,
                "buffer_seconds": MOVING_TRACK_BUFFER_SECONDS,
                "buffer_limit": MOVING_TRACK_BUFFER_LIMIT,
                "tracked_seconds": MOVING_TRACK_TRACKED_SECONDS,
                "lost_seconds": MOVING_TRACK_LOST_SECONDS,
                "rejected_reports": self.moving_rejected_report_count,
                "buffer_depth": len(self.moving_report_buffer),
                "notes": "Constant-velocity Kalman filter; local identity continuity is mocked. Real image-based tracking is outside this increment.",
            },
        }

    def _task_contact_ref(self, task: dict[str, object]) -> str:
        return str(task.get("contact_ref") or task.get("signature_id") or "")

    def _task_type(self, task: dict[str, object]) -> str:
        return str(task.get("task_type") or TASK_TYPE_ISR_OBSERVE)

    def _task_is_executable_by_this_node_locked(self, task: dict[str, object], excluding_task_id: str | None = None) -> bool:
        task_type = self._task_type(task)

        if not self.team_configuration.available:
            return False

        if task_type == TASK_TYPE_ISR_OBSERVE:
            return (
                "isr_observe" in self.team_configuration.capabilities
                and not self._drone_has_active_assignment_locked(self.drone_id, excluding_task_id or str(task.get("task_id", "")))
            )

        if task_type == TASK_TYPE_PAYLOAD_DROP:
            requested_payload_type = str(task.get("requested_payload_type", ""))
            requested_payload_quantity = int(task.get("requested_payload_quantity", 0) or 0)
            return (
                "payload_drop" in self.team_configuration.capabilities
                and requested_payload_quantity > 0
                and self._available_payload_inventory_locked(excluding_task_id or str(task.get("task_id", ""))).get(requested_payload_type, 0) >= requested_payload_quantity
                and not self._drone_has_active_assignment_locked(self.drone_id, excluding_task_id or str(task.get("task_id", "")))
            )

        return False

    def _available_payload_inventory_locked(self, excluding_task_id: str | None = None) -> dict[str, int]:
        available = dict(self.team_configuration.payload_inventory)
        for task_id, reservation in self.payload_reservations_by_task.items():
            if excluding_task_id is not None and task_id == excluding_task_id:
                continue

            payload_type = str(reservation.get("payload_type", ""))
            quantity = int(reservation.get("quantity", 0) or 0)
            available[payload_type] = max(0, available.get(payload_type, 0) - quantity)

        return available

    def _reserve_payload_inventory_locked(self, task: dict[str, object]) -> None:
        if self._task_type(task) != TASK_TYPE_PAYLOAD_DROP:
            return

        task_id = str(task["task_id"])
        payload_type = str(task.get("requested_payload_type", ""))
        quantity = int(task.get("requested_payload_quantity", 0) or 0)

        if quantity <= 0 or not payload_type:
            return

        existing = self.payload_reservations_by_task.get(task_id)
        if existing is not None:
            task["reservation_state"] = "reserved"
            task["reserved_payload_quantity"] = int(existing.get("quantity", quantity))
            return

        if self._available_payload_inventory_locked(task_id).get(payload_type, 0) < quantity:
            task["reservation_state"] = "reservation_failed"
            task["event"] = "Payload assignment could not reserve sufficient inventory."
            return

        self.payload_reservations_by_task[task_id] = {
            "payload_type": payload_type,
            "quantity": quantity,
            "reserved_at": self.simulation_time,
        }
        task["reservation_state"] = "reserved"
        task["reserved_payload_quantity"] = quantity

    def _release_payload_reservation_locked(self, task: dict[str, object]) -> None:
        if self._task_type(task) != TASK_TYPE_PAYLOAD_DROP:
            return

        task_id = str(task.get("task_id", ""))
        self.payload_reservations_by_task.pop(task_id, None)
        task["reservation_state"] = "released"
        task["reserved_payload_quantity"] = 0

    def _task_payload_metadata(
        self,
        task_type: str,
        requested_payload_type: str | None = None,
        requested_payload_quantity: int | None = None,
        contact_ref: str | None = None,
    ) -> dict[str, object]:
        if task_type != TASK_TYPE_PAYLOAD_DROP:
            return {}

        return {
            "requested_payload_type": requested_payload_type or "",
            "requested_payload_quantity": int(requested_payload_quantity or 0),
            "release_authorization_required": True,
            "contact_ref": contact_ref or "",
            "reservation_state": "not_reserved",
            "reserved_payload_quantity": 0,
            "payload_standoff_m": self.payload_standoff_m,
            "execution_note": "Approach only; terminal guidance, authorization, release, inventory consumption, and delivery confirmation are out of scope.",
        }

    def _new_response_task_locked(
        self,
        *,
        contact_ref: str,
        task_type: str,
        object_label: str,
        estimated_east_m: float,
        estimated_north_m: float,
        estimated_latitude: float,
        estimated_longitude: float,
        last_observation_at: float,
        status: str,
        event: str,
        requested_payload_type: str | None = None,
        requested_payload_quantity: int | None = None,
    ) -> dict[str, object]:
        if task_type not in SUPPORTED_TASK_TYPES:
            raise ValueError(f"Unsupported task type: {task_type}")

        task_id = task_id_for_contact(self.scenario_seed, contact_ref, task_type)
        task = {
            "task_id": task_id,
            "scenario_run_id": str(self.scenario_seed),
            "task_type": task_type,
            "contact_ref": contact_ref,
            "signature_id": contact_ref,
            "object_label": object_label,
            "display_label": self._task_display_label(object_label),
            "status": status,
            "selected_playbook": self.selected_playbook,
            "playbook_config_version": self.playbook_config_version,
            "created_at": self.simulation_time,
            "updated_at": self.simulation_time,
            "last_observation_at": last_observation_at,
            "election_deadline": self.simulation_time + TASK_ELECTION_SECONDS,
            "estimated_east_m": estimated_east_m,
            "estimated_north_m": estimated_north_m,
            "estimated_latitude": estimated_latitude,
            "estimated_longitude": estimated_longitude,
            "assigned_drone_id": None,
            "assigned_drone_name": None,
            "orbit_radius_m": ORBIT_RADIUS_M,
            "revision": 0,
            "assignment_epoch": 1 if status == "electing" else 0,
            "event": event,
            "limitation": RESPONSE_LIMITATION_NOTE,
        }
        task.update(self._task_payload_metadata(task_type, requested_payload_type, requested_payload_quantity, contact_ref))
        return task

    def _payload_request_metadata_matches(self, task: dict[str, object], payload_type: str, quantity: int) -> bool:
        return (
            self._task_type(task) == TASK_TYPE_PAYLOAD_DROP
            and str(task.get("requested_payload_type", "")) == payload_type
            and int(task.get("requested_payload_quantity", 0) or 0) == int(quantity)
        )

    def _ensure_payload_request_for_priority_object_locked(self, object_map: dict[str, object]) -> dict[str, object] | None:
        if not self.payload_requests_enabled:
            return None

        if "isr_observe" not in self.team_configuration.capabilities or not self.team_configuration.available:
            return None

        signature_id = str(object_map.get("priority_signature") or "")
        if not signature_id:
            return None

        task_id = task_id_for_contact(self.scenario_seed, signature_id, TASK_TYPE_PAYLOAD_DROP)
        if task_id in self.cancelled_task_ids:
            return None

        if task_id in self.response_tasks:
            task = self.response_tasks[task_id]
            if not self._payload_request_metadata_matches(task, self.requested_payload_type, self.requested_payload_quantity):
                task["metadata_conflict"] = True
                task["event"] = "Conflicting payload request metadata rejected; existing request was not overwritten."
                return None

            task["last_observation_at"] = max(float(task.get("last_observation_at", 0.0)), float(object_map.get("last_seen", 0.0)))
            task["updated_at"] = self.simulation_time
            return task

        task = self._new_response_task_locked(
            contact_ref=signature_id,
            task_type=TASK_TYPE_PAYLOAD_DROP,
            object_label=str(object_map.get("label", "payload objective")),
            estimated_east_m=float(object_map["east_m"]),
            estimated_north_m=float(object_map["north_m"]),
            estimated_latitude=float(object_map["latitude"]),
            estimated_longitude=float(object_map["longitude"]),
            last_observation_at=float(object_map["last_seen"]),
            status=PAYLOAD_REQUEST_REPORTED_STATUS,
            event="ISR contact evidence accepted; payload-drop request published.",
            requested_payload_type=self.requested_payload_type,
            requested_payload_quantity=self.requested_payload_quantity,
        )
        task["assignment_epoch"] = 1
        task["requesting_drone_id"] = self.drone_id
        task["requesting_drone_name"] = self.name
        self.response_tasks[task_id] = task
        self.outbound_claims.append(self._task_claim_payload_locked(task, PAYLOAD_REQUEST_REPORTED_STATUS))
        self._make_task_allocatable_locked(task_id, task)
        return task

    def create_dormant_payload_task_for_contact(
        self,
        contact_ref: str,
        requested_payload_type: str,
        requested_payload_quantity: int,
        object_label: str = "payload objective",
    ) -> dict[str, object]:
        with self.lock:
            task = self._new_response_task_locked(
                contact_ref=contact_ref,
                task_type=TASK_TYPE_PAYLOAD_DROP,
                object_label=object_label,
                estimated_east_m=0.0,
                estimated_north_m=0.0,
                estimated_latitude=MAP_REFERENCE_LATITUDE,
                estimated_longitude=MAP_REFERENCE_LONGITUDE,
                last_observation_at=self.simulation_time,
                status=PAYLOAD_PENDING_STATUS,
                event="Payload delivery workflow is not implemented in this increment.",
                requested_payload_type=requested_payload_type,
                requested_payload_quantity=requested_payload_quantity,
            )
            self.response_tasks[str(task["task_id"])] = task
            return dict(task)

    def _make_discovered_priorities_available_locked(self) -> None:
        for object_map in self.objects.values():
            if bool(object_map.get("priority", False)) and object_map.get("priority_signature"):
                self._ensure_response_task_for_priority_object_locked(object_map)

        for task_id, task in sorted(self.response_tasks.items()):
            if str(task.get("task_id")) in self.cancelled_task_ids:
                continue

            if task.get("status") != "reported":
                continue

            task["playbook_config_version"] = self.playbook_config_version
            task["updated_at"] = self.simulation_time
            task["owner_task_state"] = None

            if self._task_type(task) != TASK_TYPE_ISR_OBSERVE:
                continue

            if self.active_response_task_id is None:
                task["status"] = "electing"
                task["election_deadline"] = self.simulation_time + TASK_ELECTION_SECONDS
                self.active_response_task_id = task_id
                claim = self._task_claim_payload_locked(task)
                self.claims.setdefault(task_id, {})[self.drone_id] = claim
                self.outbound_claims.append(claim)
            else:
                task["status"] = "queued"
                if task_id not in self.response_task_queue:
                    self.response_task_queue.append(task_id)

    def _task_display_label(self, label: object) -> str:
        return f"PRIORITY {str(label or 'Objective').replace('_', ' ').title()}"

    def _ensure_response_task_for_priority_object_locked(self, object_map: dict[str, object]) -> None:
        signature_id = str(object_map["priority_signature"])

        if task_id_for_contact(self.scenario_seed, signature_id, TASK_TYPE_ISR_OBSERVE) in self.cancelled_task_ids:
            return

        task_id = task_id_for_contact(self.scenario_seed, signature_id, TASK_TYPE_ISR_OBSERVE)

        if task_id in self.response_tasks:
            return

        east_m = float(object_map["east_m"])
        north_m = float(object_map["north_m"])
        infeasible = abs(east_m) + ORBIT_RADIUS_M > SCENARIO_BOUND_EAST_M or abs(north_m) + ORBIT_RADIUS_M > SCENARIO_BOUND_NORTH_M
        local_available = self.active_response_task_id is None
        status = "queued" if not local_available else "electing"
        task = self._new_response_task_locked(
            contact_ref=signature_id,
            task_type=TASK_TYPE_ISR_OBSERVE,
            object_label=str(object_map.get("label", "objective")),
            estimated_east_m=east_m,
            estimated_north_m=north_m,
            estimated_latitude=float(object_map["latitude"]),
            estimated_longitude=float(object_map["longitude"]),
            last_observation_at=float(object_map["last_seen"]),
            status="infeasible" if infeasible else status,
            event="Previously discovered priority evidence became eligible under observe-and-orbit.",
        )
        self.response_tasks[task_id] = task

        if infeasible:
            task["event"] = "Requested orbit is infeasible within scenario bounds."
            return

        if local_available and self._task_is_executable_by_this_node_locked(task):
            self.active_response_task_id = task_id
            claim = self._task_claim_payload_locked(task)
            self.claims.setdefault(task_id, {})[self.drone_id] = claim
            self.outbound_claims.append(claim)
        else:
            self.response_task_queue.append(task_id)

    def _start_return_to_patrol_locked(self) -> None:
        if (
            self.response_east_m is not None
            and self.response_north_m is not None
            and self.response_heading_deg is not None
        ):
            self.return_east_m = self.response_east_m
            self.return_north_m = self.response_north_m
            self.return_heading_deg = self.response_heading_deg
            return

        latitude, longitude, heading = self._pose_at_locked()
        east_m, north_m = lat_lon_to_local_m(
            MAP_REFERENCE_LATITUDE,
            MAP_REFERENCE_LONGITUDE,
            latitude,
            longitude,
        )
        self.return_east_m = east_m
        self.return_north_m = north_m
        self.return_heading_deg = heading

    def _release_active_response_tasks_locked(self) -> None:
        for task_id, task in self.response_tasks.items():
            if str(task.get("task_id")) in self.cancelled_task_ids:
                continue

            if task.get("assigned_drone_id") == self.drone_id and task.get("status") in {"transitioning", "orbiting"}:
                self._start_return_to_patrol_locked()
                self.outbound_claims.append(self._task_claim_payload_locked(task, "released"))

            if task.get("status") in {"transitioning", "orbiting", "assigned_remote", "electing", "queued"}:
                self._release_current_assignment_locked(task, "released")
                task["playbook_config_version"] = self.playbook_config_version

            if task_id in self.response_task_queue:
                self.response_task_queue.remove(task_id)

        self.active_response_task_id = None
        self._clear_response_guidance_locked()

    def _observation_id(self, source_track_id: str, observation_time_bucket: int) -> str:
        raw_id = f"{self.drone_id}:{SENSOR_MODELS[self.drone_id].sensor_id}:{source_track_id}:{observation_time_bucket}"
        return hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:24]

    def _track_key(self, observation: Observation) -> str:
        return f"{observation.source_drone_id}:{observation.sensor_id}:{observation.source_track_id}"

    def _validate_observation_locked(self, observation: Observation) -> bool:
        if observation.measurement_type not in {"point", MOVING_TRACK_REPORT_TYPE}:
            return False

        if observation.scenario_run_id != self.scenario_run_id:
            return False

        if not all(
            math.isfinite(value)
            for value in (observation.east_m, observation.north_m, observation.latitude, observation.longitude)
        ):
            return False

        if observation.measurement_type == MOVING_TRACK_REPORT_TYPE:
            if observation.report_schema_version != TRACK_REPORT_SCHEMA_VERSION:
                return False

            if observation.state_epoch is None or not math.isfinite(observation.state_epoch):
                return False

        return is_valid_covariance(observation.covariance)

    def _candidate_gate_d2(self, object_map: dict[str, object], observation: Observation) -> float:
        covariance = object_map["covariance"]  # type: ignore[assignment]
        if not is_valid_covariance(covariance):  # type: ignore[arg-type]
            raise ValueError("Invalid map covariance.")

        residual = (
            observation.east_m - float(object_map["east_m"]),
            observation.north_m - float(object_map["north_m"]),
        )
        innovation_covariance = add_2x2(covariance, observation.covariance)  # type: ignore[arg-type]
        return mahalanobis_squared(residual, innovation_covariance)

    def _record_unresolved_locked(self, observation: Observation, reason: str, d2: float | None = None) -> None:
        self.unresolved_observations[observation.id] = {
            "id": observation.id,
            "reason": reason,
            "source_drone_id": observation.source_drone_id,
            "source_drone_name": observation.source_drone_name,
            "sensor_id": observation.sensor_id,
            "source_track_id": observation.source_track_id,
            "timestamp": observation.timestamp,
            "label": observation.label,
            "latitude": observation.latitude,
            "longitude": observation.longitude,
            "east_m": observation.east_m,
            "north_m": observation.north_m,
            "d2": d2,
        }
        self._record_observation_trace_locked(observation, "unresolved", None, reason)

    def _select_association_locked(self, observation: Observation) -> str | None:
        track_key = self._track_key(observation)
        continuity_object_id = self.track_to_object.get(track_key)

        if continuity_object_id is not None:
            object_map = self.objects.get(continuity_object_id)

            if object_map is None:
                self.track_to_object.pop(track_key, None)
            elif object_map["label"] == observation.label:
                d2 = self._candidate_gate_d2(object_map, observation)

                if d2 <= ASSOCIATION_GATE_D2:
                    return continuity_object_id

                self._record_unresolved_locked(observation, "track_continuity_spatial_rejection", d2)
                return None

        candidates: list[tuple[float, str]] = []
        for object_id, object_map in self.objects.items():
            if object_map["label"] != observation.label:
                continue

            try:
                d2 = self._candidate_gate_d2(object_map, observation)
            except ValueError:
                continue

            if d2 <= ASSOCIATION_GATE_D2:
                candidates.append((d2, object_id))

        if not candidates:
            return None

        candidates.sort(key=lambda item: (item[0], item[1]))
        best_d2, best_object_id = candidates[0]

        if len(candidates) > 1:
            second_d2, _second_object_id = candidates[1]

            if second_d2 - best_d2 <= ASSOCIATION_AMBIGUITY_D2_DELTA:
                self._record_unresolved_locked(observation, "ambiguous_association", best_d2)
                return None

        return best_object_id

    def _new_local_object_id_locked(self, observation: Observation) -> str:
        raw_id = f"{self.drone_id}:{observation.id}:{len(self.objects)}"
        return f"{self.drone_id}-map-{hashlib.sha256(raw_id.encode('utf-8')).hexdigest()[:10]}"

    def _contribution_from_observation(self, observation: Observation) -> dict[str, object]:
        return {
            "track_key": self._track_key(observation),
            "observation_id": observation.id,
            "source_drone_id": observation.source_drone_id,
            "source_drone_name": observation.source_drone_name,
            "source_color": observation.source_color,
            "sensor_id": observation.sensor_id,
            "source_track_id": observation.source_track_id,
            "timestamp": observation.timestamp,
            "latitude": observation.latitude,
            "longitude": observation.longitude,
            "east_m": observation.east_m,
            "north_m": observation.north_m,
            "classification_score": observation.classification_score,
            "covariance": observation.covariance,
            "priority_evidence": observation.priority_evidence,
            "scenario_run_id": observation.scenario_run_id,
            "measurement_frame": observation.measurement_frame,
            "measurement_type": observation.measurement_type,
            "object_id": observation.object_id,
            "report_schema_version": observation.report_schema_version,
            "report_id": observation.report_id or observation.id,
            "source_node_session_id": observation.source_node_session_id,
            "state_epoch": observation.state_epoch,
            "sensor_track_status": observation.sensor_track_status,
            "simulated": observation.simulated,
            "supporting_observation_ids": observation.supporting_observation_ids or [],
            "raw_observation": asdict(observation),
        }

    def _refresh_fused_object_locked(self, object_map: dict[str, object]) -> None:
        contributions = object_map["contributions"]  # type: ignore[assignment]
        active_contributions = [
            contribution
            for contribution in contributions.values()  # type: ignore[union-attr]
            if self.simulation_time - float(contribution["timestamp"]) <= CONTRIBUTION_MAX_AGE_SECONDS
        ]
        active_contributions.sort(
            key=lambda contribution: (
                int(contribution["source_drone_id"]),
                str(contribution["sensor_id"]),
                str(contribution["source_track_id"]),
            ),
        )

        if active_contributions:
            first = active_contributions[0]
            mean = (float(first["east_m"]), float(first["north_m"]))
            covariance = first["covariance"]  # type: ignore[assignment]

            for contribution in active_contributions[1:]:
                mean, covariance, _objective = covariance_intersection_two(
                    mean,
                    covariance,  # type: ignore[arg-type]
                    (float(contribution["east_m"]), float(contribution["north_m"])),
                    contribution["covariance"],  # type: ignore[arg-type]
                )

            latitude, longitude = local_m_to_lat_lon(mean[0], mean[1])
            object_map["east_m"] = mean[0]
            object_map["north_m"] = mean[1]
            object_map["latitude"] = latitude
            object_map["longitude"] = longitude
            object_map["covariance"] = covariance
            object_map["uncertainty_m"] = math.sqrt(covariance[0][0] + covariance[1][1])
            object_map["last_seen"] = max(float(contribution["timestamp"]) for contribution in active_contributions)
            object_map["active_contribution_count"] = len(active_contributions)
            active_sources = {int(contribution["source_drone_id"]) for contribution in active_contributions}
            active_source_names = {str(contribution["source_drone_name"]) for contribution in active_contributions}
            object_map["sources"] = sorted(active_sources)
            object_map["source_names"] = sorted(active_source_names)
        else:
            object_map["active_contribution_count"] = 0
            object_map["sources"] = []
            object_map["source_names"] = []

        historical_sources = {
            int(contribution["source_drone_id"])
            for contribution in contributions.values()  # type: ignore[union-attr]
        }
        historical_source_names = {
            str(contribution["source_drone_name"])
            for contribution in contributions.values()  # type: ignore[union-attr]
        }
        object_map["historical_sources"] = sorted(historical_sources)
        object_map["historical_source_names"] = sorted(historical_source_names)
        object_map["measurements"] = sorted(
            list(contributions.values()),  # type: ignore[union-attr]
            key=lambda contribution: (
                int(contribution["source_drone_id"]),
                str(contribution["sensor_id"]),
                str(contribution["source_track_id"]),
            ),
        )
        object_map["own_sensor"] = self.drone_id in historical_sources
        object_map["teammate_sourced"] = any(source != self.drone_id for source in historical_sources)

    def _apply_observation_locked(self, observation: Observation) -> bool:
        if observation.measurement_type == MOVING_TRACK_REPORT_TYPE:
            return self._buffer_moving_report_locked(observation)

        if observation.id in self.observations:
            self.duplicate_observations_ignored += 1
            self._record_observation_trace_locked(observation, "duplicate_ignored", None, "Observation ID already retained by this node.")
            return False

        if not self._validate_observation_locked(observation):
            self.validation_rejects += 1
            self._record_observation_trace_locked(observation, "rejected", None, "Observation failed schema, run, finite-value, or covariance validation.")
            return False

        track_key = self._track_key(observation)
        associated_object_id = self._select_association_locked(observation)
        create_new_object = associated_object_id is None and observation.id not in self.unresolved_observations

        if associated_object_id is None and not create_new_object:
            self.observations[observation.id] = observation
            self.unique_observations_accepted += 1
            self._record_observation_trace_locked(observation, "unresolved", None, "Accepted but not associated to a local map object.")
            return False

        if create_new_object:
            associated_object_id = self._new_local_object_id_locked(observation)
            self.objects[associated_object_id] = {
                "local_object_id": associated_object_id,
                "object_id": associated_object_id,
                "label": observation.label,
                "latitude": observation.latitude,
                "longitude": observation.longitude,
                "east_m": observation.east_m,
                "north_m": observation.north_m,
                "classification_score": observation.classification_score,
                "covariance": observation.covariance,
                "uncertainty_m": math.sqrt(observation.covariance[0][0] + observation.covariance[1][1]),
                "first_seen": observation.timestamp,
                "last_seen": observation.timestamp,
                "sources": [],
                "source_names": [],
                "historical_sources": [],
                "historical_source_names": [],
                "own_sensor": False,
                "teammate_sourced": False,
                "observation_count": 0,
                "observation_ids": [],
                "active_contribution_count": 0,
                "contributions": {},
                "measurements": [],
                "priority": False,
                "priority_evidence": None,
                "priority_signature": None,
            }

        object_map = self.objects[associated_object_id]
        contributions = object_map["contributions"]  # type: ignore[assignment]
        prior_contribution = contributions.get(track_key)  # type: ignore[union-attr]

        if prior_contribution is not None and observation.timestamp <= float(prior_contribution["timestamp"]):
            self.duplicate_observations_ignored += 1
            self.observations[observation.id] = observation
            self._record_observation_trace_locked(observation, "duplicate_ignored", associated_object_id, "Older or equal source-track contribution ignored.")
            return False

        contributions[track_key] = self._contribution_from_observation(observation)  # type: ignore[index]
        self.track_to_object[track_key] = associated_object_id
        self.observations[observation.id] = observation
        self.unique_observations_accepted += 1
        object_map["first_seen"] = min(float(object_map["first_seen"]), observation.timestamp)
        object_map["classification_score"] = max(
            float(object_map["classification_score"]),
            observation.classification_score,
        )
        object_map["observation_count"] = int(object_map["observation_count"]) + 1
        object_map["observation_ids"] = [*list(object_map["observation_ids"]), observation.id]

        if observation.priority_evidence:
            object_map["priority"] = True
            object_map["priority_evidence"] = observation.priority_evidence
            object_map["priority_signature"] = str(observation.priority_evidence["signature_id"])

        self._refresh_fused_object_locked(object_map)
        self._record_observation_trace_locked(observation, "accepted", associated_object_id, "Observation contributed to this node's local map.")

        if observation.priority_evidence:
            self._ensure_response_task_locked(observation, object_map)
            self._ensure_payload_request_for_priority_object_locked(object_map)

        return True

    def _buffer_moving_report_locked(self, observation: Observation) -> bool:
        if observation.id in self.observations or observation.id in self.moving_report_ids or observation.id in self.moving_committed_report_ids:
            self.duplicate_observations_ignored += 1
            self._record_observation_trace_locked(observation, "duplicate_ignored", None, "Moving point-track report ID already retained by this node.")
            return False

        if not self._validate_observation_locked(observation):
            self.validation_rejects += 1
            self._record_observation_trace_locked(observation, "rejected", None, "Moving point-track report failed schema, run, timestamp, finite-value, or covariance validation.")
            return False

        if float(observation.state_epoch or observation.timestamp) < self.moving_processing_epoch:
            self.moving_rejected_report_count += 1
            self.observations[observation.id] = observation
            self._record_observation_trace_locked(observation, "rejected_delayed", None, "Moving point-track report arrived older than the committed processing epoch.")
            return False

        self.moving_report_ids.add(observation.id)
        self.moving_report_buffer.append(observation)
        self.observations[observation.id] = observation

        if len(self.moving_report_buffer) > MOVING_TRACK_BUFFER_LIMIT:
            overflow = sorted(
                self.moving_report_buffer,
                key=lambda report: (
                    float(report.state_epoch or report.timestamp),
                    int(report.source_drone_id),
                    str(report.sensor_id),
                    str(report.report_id or report.id),
                ),
            )[0]
            self.moving_report_buffer.remove(overflow)
            self.moving_report_ids.discard(overflow.id)
            self.moving_rejected_report_count += 1
            self._record_observation_trace_locked(overflow, "rejected_buffer_overflow", None, "Moving point-track reorder buffer overflowed.")

        return True

    def _moving_candidate_gate_d2_locked(self, object_map: dict[str, object], observation: Observation) -> float:
        state = object_map["filter_state"]  # type: ignore[assignment]
        covariance = object_map["filter_covariance"]  # type: ignore[assignment]
        predicted_state, predicted_covariance = predict_state(
            state,  # type: ignore[arg-type]
            covariance,  # type: ignore[arg-type]
            float(object_map["filter_epoch"]),
            float(observation.state_epoch or observation.timestamp),
        )
        _residual, _innovation_covariance, d2 = innovation_for_position(
            predicted_state,
            predicted_covariance,
            (observation.east_m, observation.north_m),
            observation.covariance,
        )
        return d2

    def _select_moving_association_locked(self, observation: Observation) -> str | None:
        track_key = self._track_key(observation)
        continuity_object_id = self.track_to_object.get(track_key)

        if continuity_object_id is not None:
            object_map = self.objects.get(continuity_object_id)

            if object_map is None:
                self.track_to_object.pop(track_key, None)
            elif object_map.get("tracking_model") == "constant_velocity" and object_map["label"] == observation.label:
                d2 = self._moving_candidate_gate_d2_locked(object_map, observation)

                if d2 <= ASSOCIATION_GATE_D2:
                    return continuity_object_id

                self._record_unresolved_locked(observation, "moving_track_continuity_spatial_rejection", d2)
                return None

        candidates: list[tuple[float, str]] = []
        for object_id, object_map in self.objects.items():
            if object_map.get("tracking_model") != "constant_velocity" or object_map["label"] != observation.label:
                continue

            try:
                d2 = self._moving_candidate_gate_d2_locked(object_map, observation)
            except ValueError:
                continue

            if d2 <= ASSOCIATION_GATE_D2:
                candidates.append((d2, object_id))

        if not candidates:
            return None

        candidates.sort(key=lambda item: (item[0], item[1]))
        best_d2, best_object_id = candidates[0]

        if len(candidates) > 1:
            second_d2, _second_object_id = candidates[1]

            if second_d2 - best_d2 <= ASSOCIATION_AMBIGUITY_D2_DELTA:
                self._record_unresolved_locked(observation, "moving_ambiguous_association", best_d2)
                return None

        return best_object_id

    def _process_moving_report_buffer_locked(self) -> None:
        if not self.moving_report_buffer:
            return

        eligible_cutoff = self.simulation_time - MOVING_TRACK_BUFFER_SECONDS
        eligible = [
            report
            for report in self.moving_report_buffer
            if float(report.state_epoch or report.timestamp) <= eligible_cutoff
        ]

        if not eligible:
            return

        for report in sorted(
            eligible,
            key=lambda item: (
                float(item.state_epoch or item.timestamp),
                int(item.source_drone_id),
                str(item.sensor_id),
                str(item.report_id or item.id),
            ),
        ):
            if report not in self.moving_report_buffer:
                continue

            self.moving_report_buffer.remove(report)
            self.moving_report_ids.discard(report.id)
            self._commit_moving_report_locked(report)

    def _commit_moving_report_locked(self, observation: Observation) -> bool:
        report_epoch = float(observation.state_epoch or observation.timestamp)

        if report_epoch < self.moving_processing_epoch:
            self.moving_rejected_report_count += 1
            self._record_observation_trace_locked(observation, "rejected_delayed", None, "Moving point-track report was older than the committed processing epoch.")
            return False

        associated_object_id = self._select_moving_association_locked(observation)

        if associated_object_id is None and observation.id in self.unresolved_observations:
            self._record_observation_trace_locked(observation, "unresolved", None, "Moving report accepted but not associated to a local moving track.")
            return False

        if associated_object_id is None:
            associated_object_id = self._new_local_object_id_locked(observation)
            filter_state, filter_covariance = initialize_track(observation.east_m, observation.north_m, observation.covariance)
            self.objects[associated_object_id] = {
                "local_object_id": associated_object_id,
                "object_id": associated_object_id,
                "label": observation.label,
                "tracking_model": "constant_velocity",
                "motion_class": "moving_person",
                "latitude": observation.latitude,
                "longitude": observation.longitude,
                "east_m": observation.east_m,
                "north_m": observation.north_m,
                "predicted_east_m": observation.east_m,
                "predicted_north_m": observation.north_m,
                "velocity_east_mps": 0.0,
                "velocity_north_mps": 0.0,
                "classification_score": observation.classification_score,
                "covariance": observation.covariance,
                "state_covariance": filter_covariance,
                "uncertainty_m": horizontal_uncertainty_m(filter_covariance),
                "first_seen": report_epoch,
                "last_seen": report_epoch,
                "last_measurement_east_m": observation.east_m,
                "last_measurement_north_m": observation.north_m,
                "last_measurement_at": report_epoch,
                "filter_epoch": report_epoch,
                "filter_state": filter_state,
                "filter_covariance": filter_covariance,
                "sources": [],
                "source_names": [],
                "historical_sources": [],
                "historical_source_names": [],
                "source_track_ids": [],
                "own_sensor": False,
                "teammate_sourced": False,
                "observation_count": 0,
                "observation_ids": [],
                "active_contribution_count": 0,
                "contributions": {},
                "measurements": [],
                "track_trail": [],
                "track_status": "tracked",
                "prediction_age_seconds": 0.0,
                "last_measurement_age_seconds": 0.0,
                "priority": False,
                "priority_evidence": None,
                "priority_signature": None,
            }
        else:
            object_map = self.objects[associated_object_id]
            predicted_state, predicted_covariance = predict_state(
                object_map["filter_state"],  # type: ignore[arg-type]
                object_map["filter_covariance"],  # type: ignore[arg-type]
                float(object_map["filter_epoch"]),
                report_epoch,
            )
            updated_state, updated_covariance, _d2 = update_with_position(
                predicted_state,
                predicted_covariance,
                (observation.east_m, observation.north_m),
                observation.covariance,
            )
            object_map["filter_state"] = updated_state
            object_map["filter_covariance"] = updated_covariance
            object_map["filter_epoch"] = report_epoch

        object_map = self.objects[associated_object_id]
        if "filter_state" not in object_map:
            object_map["filter_state"], object_map["filter_covariance"] = initialize_track(observation.east_m, observation.north_m, observation.covariance)
            object_map["filter_epoch"] = report_epoch

        state = object_map["filter_state"]  # type: ignore[assignment]
        covariance = object_map["filter_covariance"]  # type: ignore[assignment]
        object_map["east_m"] = float(state[0])  # type: ignore[index]
        object_map["north_m"] = float(state[1])  # type: ignore[index]
        object_map["predicted_east_m"] = float(state[0])  # type: ignore[index]
        object_map["predicted_north_m"] = float(state[1])  # type: ignore[index]
        object_map["velocity_east_mps"] = float(state[2])  # type: ignore[index]
        object_map["velocity_north_mps"] = float(state[3])  # type: ignore[index]
        latitude, longitude = local_m_to_lat_lon(float(state[0]), float(state[1]))  # type: ignore[index]
        object_map["latitude"] = latitude
        object_map["longitude"] = longitude
        object_map["state_covariance"] = covariance
        object_map["covariance"] = [[covariance[0][0], covariance[0][1]], [covariance[1][0], covariance[1][1]]]  # type: ignore[index]
        object_map["uncertainty_m"] = horizontal_uncertainty_m(covariance)  # type: ignore[arg-type]
        object_map["classification_score"] = max(float(object_map["classification_score"]), observation.classification_score)
        object_map["last_seen"] = report_epoch
        object_map["last_measurement_at"] = report_epoch
        object_map["last_measurement_east_m"] = observation.east_m
        object_map["last_measurement_north_m"] = observation.north_m
        object_map["first_seen"] = min(float(object_map["first_seen"]), report_epoch)
        object_map["observation_count"] = int(object_map["observation_count"]) + 1
        object_map["observation_ids"] = [*list(object_map["observation_ids"]), observation.id]
        object_map["track_trail"] = [
            *[
                point
                for point in object_map.get("track_trail", [])  # type: ignore[union-attr]
                if report_epoch - float(point["timestamp"]) <= MOVING_TRACK_TRAIL_SECONDS
            ],
            {"east_m": float(state[0]), "north_m": float(state[1]), "timestamp": report_epoch},  # type: ignore[index]
        ]

        contribution_key = f"{self._track_key(observation)}:{observation.id}"
        contributions = object_map["contributions"]  # type: ignore[assignment]
        contributions[contribution_key] = self._contribution_from_observation(observation)  # type: ignore[index]
        self.track_to_object[self._track_key(observation)] = associated_object_id
        self.unique_observations_accepted += 1
        self.moving_committed_report_ids.add(observation.id)
        self.moving_processing_epoch = max(self.moving_processing_epoch, report_epoch)
        self._refresh_moving_track_payload_locked(object_map)
        self._record_observation_trace_locked(observation, "accepted", associated_object_id, "Moving point-track report updated this node's constant-velocity track.")
        return True

    def _refresh_moving_track_payload_locked(self, object_map: dict[str, object]) -> None:
        state = object_map["filter_state"]  # type: ignore[assignment]
        covariance = object_map["filter_covariance"]  # type: ignore[assignment]
        predicted_state, predicted_covariance = predict_state(
            state,  # type: ignore[arg-type]
            covariance,  # type: ignore[arg-type]
            float(object_map["filter_epoch"]),
            self.simulation_time,
        )
        predicted_latitude, predicted_longitude = local_m_to_lat_lon(predicted_state[0], predicted_state[1])
        contributions = object_map["contributions"]  # type: ignore[assignment]
        historical_sources = {
            int(contribution["source_drone_id"])
            for contribution in contributions.values()  # type: ignore[union-attr]
        }
        historical_source_names = {
            str(contribution["source_drone_name"])
            for contribution in contributions.values()  # type: ignore[union-attr]
        }
        recent_contributions = [
            contribution
            for contribution in contributions.values()  # type: ignore[union-attr]
            if self.simulation_time - float(contribution["timestamp"]) <= MOVING_TRACK_TRACKED_SECONDS
        ]
        recent_sources = {int(contribution["source_drone_id"]) for contribution in recent_contributions}
        recent_source_names = {str(contribution["source_drone_name"]) for contribution in recent_contributions}
        source_track_ids = sorted({
            f"{contribution['source_drone_name']}:{contribution['sensor_id']}:{contribution['source_track_id']}"
            for contribution in contributions.values()  # type: ignore[union-attr]
        })
        status = moving_lifecycle(float(object_map["last_measurement_at"]), self.simulation_time)
        object_map["predicted_east_m"] = predicted_state[0]
        object_map["predicted_north_m"] = predicted_state[1]
        object_map["latitude"] = predicted_latitude
        object_map["longitude"] = predicted_longitude
        object_map["velocity_east_mps"] = predicted_state[2]
        object_map["velocity_north_mps"] = predicted_state[3]
        object_map["state_covariance"] = predicted_covariance
        object_map["covariance"] = [[predicted_covariance[0][0], predicted_covariance[0][1]], [predicted_covariance[1][0], predicted_covariance[1][1]]]
        object_map["uncertainty_m"] = horizontal_uncertainty_m(predicted_covariance)
        object_map["track_status"] = status
        object_map["prediction_age_seconds"] = max(0.0, self.simulation_time - float(object_map["filter_epoch"]))
        object_map["last_measurement_age_seconds"] = max(0.0, self.simulation_time - float(object_map["last_measurement_at"]))
        object_map["sources"] = sorted(recent_sources)
        object_map["source_names"] = sorted(recent_source_names)
        object_map["historical_sources"] = sorted(historical_sources)
        object_map["historical_source_names"] = sorted(historical_source_names)
        object_map["source_track_ids"] = source_track_ids
        object_map["active_contribution_count"] = len(recent_contributions)
        object_map["measurements"] = sorted(
            list(contributions.values()),  # type: ignore[union-attr]
            key=lambda contribution: (
                float(contribution["timestamp"]),
                int(contribution["source_drone_id"]),
                str(contribution["sensor_id"]),
                str(contribution["source_track_id"]),
            ),
        )[-40:]
        object_map["own_sensor"] = self.drone_id in historical_sources
        object_map["teammate_sourced"] = any(source != self.drone_id for source in historical_sources)
        object_map["handoff_timeline"] = self._moving_handoff_timeline_locked(object_map)

    def _moving_handoff_timeline_locked(self, object_map: dict[str, object]) -> list[dict[str, str]]:
        measurements = object_map.get("measurements", [])
        if not measurements:
            return []

        first = measurements[0]  # type: ignore[index]
        recent_sources = list(object_map.get("source_names", []))
        historical_sources = list(object_map.get("historical_source_names", []))
        events = [{"state": "Acquired", "detail": f"Acquired by {first['source_drone_name']}"}]

        if len(recent_sources) > 1:
            events.append({"state": "Overlap", "detail": f"Overlapping observations: {', '.join(recent_sources)}"})

        if len(historical_sources) > 1 and recent_sources:
            events.append({"state": "Continued", "detail": f"Coverage continued by {', '.join(recent_sources)}"})

        status = str(object_map.get("track_status", "tracked"))
        if status == "coasting":
            events.append({"state": "Coasting", "detail": "Coasting from prediction; no fresh accepted measurement."})
        elif status == "lost":
            events.append({"state": "Lost", "detail": "Track lost; prediction is retained only as historical context."})

        return events

    def _record_peer_payload_bytes_locked(self, byte_count: int, direction: str) -> None:
        if byte_count <= 0:
            return

        now = time.monotonic()

        if direction == "sent":
            self.peer_payload_bytes_sent += byte_count
        else:
            self.peer_payload_bytes_received += byte_count

        self.peer_payload_byte_events.append((now, byte_count, direction))
        self._prune_peer_payload_byte_events_locked(now)

    def _prune_peer_payload_byte_events_locked(self, now: float | None = None) -> None:
        cutoff = (now if now is not None else time.monotonic()) - NETWORK_THROUGHPUT_WINDOW_SECONDS

        while self.peer_payload_byte_events and self.peer_payload_byte_events[0][0] < cutoff:
            self.peer_payload_byte_events.popleft()

    def _payload_bytes_per_second_locked(self) -> dict[str, float]:
        now = time.monotonic()
        self._prune_peer_payload_byte_events_locked(now)
        sent = sum(byte_count for _timestamp, byte_count, direction in self.peer_payload_byte_events if direction == "sent")
        received = sum(byte_count for _timestamp, byte_count, direction in self.peer_payload_byte_events if direction == "received")
        return {
            "sent": sent / NETWORK_THROUGHPUT_WINDOW_SECONDS,
            "received": received / NETWORK_THROUGHPUT_WINDOW_SECONDS,
        }

    def _record_observation_trace_locked(
        self,
        observation: Observation,
        outcome: str,
        associated_object_id: str | None,
        detail: str,
    ) -> None:
        if observation.id not in self.observation_trace_records:
            self.observation_trace_order.append(observation.id)

        self.observation_trace_records[observation.id] = {
            "observation_id": observation.id,
            "scenario_run_id": observation.scenario_run_id,
            "originated_locally": observation.source_drone_id == self.drone_id,
            "source_drone_id": observation.source_drone_id,
            "source_drone_name": observation.source_drone_name,
            "sensor_id": observation.sensor_id,
            "source_track_id": observation.source_track_id,
            "outcome": outcome,
            "detail": detail,
            "associated_local_object_id": associated_object_id,
            "trace_recorded_wall_time": time.monotonic(),
            "trace_recorded_simulation_time": self.simulation_time,
        }

        while len(self.observation_trace_order) > OBSERVATION_TRACE_RETENTION:
            evicted_id = self.observation_trace_order.popleft()
            self.observation_trace_records.pop(evicted_id, None)
            self.observation_trace_evictions += 1

    def trace_observation_payload(self, observation_id: str) -> dict[str, object]:
        with self.lock:
            self._advance_locked()
            record = self.observation_trace_records.get(observation_id)

            return {
                "schema_version": SCHEMA_VERSION,
                "app_build": APPLICATION_BUILD,
                "drone_id": self.drone_id,
                "drone_name": self.name,
                "observation_id": observation_id,
                "retained_trace_count": len(self.observation_trace_records),
                "trace_retention_limit": OBSERVATION_TRACE_RETENTION,
                "trace_evictions": self.observation_trace_evictions,
                "record": dict(record) if record is not None else None,
                "status": str(record["outcome"]) if record is not None else "not_seen_or_no_longer_retained",
            }

    def _peer_reachability_state(self, peer_state: dict[str, object]) -> str:
        last_success_wall_time = peer_state.get("last_success_wall_time")

        if peer_state.get("state") == "error":
            return "failing"

        if isinstance(last_success_wall_time, float) and time.monotonic() - last_success_wall_time <= 10.0:
            return "recently_reachable"

        return "unknown"

    def _network_payload_locked(self) -> dict[str, object]:
        rates = self._payload_bytes_per_second_locked()
        peers = []

        for peer_state in self.peer_connection_state.values():
            item = dict(peer_state)
            item["reachability"] = self._peer_reachability_state(peer_state)
            peers.append(item)

        return {
            "node_id": self.drone_id,
            "node_name": self.name,
            "hostname": self.hostname,
            "pid": self.process_id,
            "app_build": APPLICATION_BUILD,
            "api_address": f"http://{self.api_host}:{self.api_port}",
            "configured_peers": list(self.peers),
            "peer_connection_state": peers,
            "observation_messages_sent": self.peer_observation_messages_sent,
            "observation_messages_received": self.peer_observation_messages_received,
            "claim_messages_sent": self.peer_claim_messages_sent,
            "claim_messages_received": self.peer_claim_messages_received,
            "payload_bytes_sent": self.peer_payload_bytes_sent,
            "payload_bytes_received": self.peer_payload_bytes_received,
            "payload_bytes_per_second": rates,
            "payload_accounting": "JSON payload only; dashboard polling is excluded from peer-traffic counters.",
            "send_failures": self.peer_send_failures,
            "duplicate_observations_ignored": self.duplicate_observations_ignored,
            "validation_rejects": self.validation_rejects,
            "trace_retention_limit": OBSERVATION_TRACE_RETENTION,
            "trace_retained_count": len(self.observation_trace_records),
            "trace_evictions": self.observation_trace_evictions,
            "wall_time_source": "time.monotonic",
        }

    def network_payload(self) -> dict[str, object]:
        with self.lock:
            return self._network_payload_locked()

    def _priority_discovered_count_locked(self) -> int:
        return len(
            {
                str(item.get("priority_signature"))
                for item in self.objects.values()
                if bool(item.get("priority", False)) and item.get("priority_signature")
            },
        )

    def _visible_response_task_locked(self) -> dict[str, object] | None:
        if self.active_response_task_id is not None:
            task = self.response_tasks.get(self.active_response_task_id)
            return dict(task) if task is not None else None

        visible_tasks = self._visible_response_tasks_locked()
        return visible_tasks[0] if visible_tasks else None

    def _visible_response_tasks_locked(self) -> list[dict[str, object]]:
        return [
            dict(task)
            for task in sorted(
                self.response_tasks.values(),
                key=lambda item: (str(item["signature_id"]), str(item["task_id"])),
            )
        ]

    def _current_mode_locked(self) -> str:
        task = self.response_tasks.get(self.active_response_task_id or "")

        if task is None:
            return "patrol"

        if task.get("status") in {"transitioning", "orbiting", *PAYLOAD_EXECUTION_STATUSES}:
            return str(task["status"])

        return "patrol"

    def _assigned_objective_locked(self) -> str | None:
        task = self.response_tasks.get(self.active_response_task_id or "")

        if task is None or task.get("status") not in {"transitioning", "orbiting", *PAYLOAD_EXECUTION_STATUSES}:
            return None

        return str(task["signature_id"])

    def _has_local_response_control_locked(self) -> bool:
        task = self.response_tasks.get(self.active_response_task_id or "")

        return bool(
            task
            and self._task_type(task) == TASK_TYPE_ISR_OBSERVE
            and task.get("assigned_drone_id") == self.drone_id
            and task.get("status") in {"transitioning", "orbiting"}
            and self.selected_playbook == PLAYBOOK_OBSERVE_ORBIT
            and int(task.get("playbook_config_version", self.playbook_config_version)) == self.playbook_config_version
        )

    def _has_local_payload_control_locked(self) -> bool:
        task = self.response_tasks.get(self.active_response_task_id or "")

        return bool(
            task
            and self._task_type(task) == TASK_TYPE_PAYLOAD_DROP
            and task.get("assigned_drone_id") == self.drone_id
            and task.get("status") in PAYLOAD_EXECUTION_STATUSES
        )

    def _has_local_execution_control_locked(self) -> bool:
        return self._has_local_response_control_locked() or self._has_local_payload_control_locked()

    def _clear_response_guidance_locked(self) -> None:
        self.response_east_m = None
        self.response_north_m = None
        self.response_heading_deg = None
        self.response_camera_yaw_deg = None

    def _initialize_payload_response_control_locked(self, task: dict[str, object]) -> None:
        task_id = str(task["task_id"])

        if (
            self.active_response_task_id == task_id
            and self.response_east_m is not None
            and self.response_north_m is not None
            and self.response_heading_deg is not None
            and self.response_camera_yaw_deg is not None
        ):
            return

        latitude, longitude, heading = self._pose_at_locked()
        response_east_m, response_north_m = lat_lon_to_local_m(
            MAP_REFERENCE_LATITUDE,
            MAP_REFERENCE_LONGITUDE,
            latitude,
            longitude,
        )
        self.active_response_task_id = task_id
        self.response_east_m = response_east_m
        self.response_north_m = response_north_m
        self.response_heading_deg = heading
        self.response_camera_yaw_deg = heading
        task["payload_standoff_m"] = float(task.get("payload_standoff_m", self.payload_standoff_m))

    def _initialize_local_response_control_locked(self, task: dict[str, object]) -> None:
        task_id = str(task["task_id"])

        if (
            self.active_response_task_id == task_id
            and self.response_east_m is not None
            and self.response_north_m is not None
            and self.response_heading_deg is not None
            and self.response_camera_yaw_deg is not None
        ):
            return

        latitude, longitude, heading = self._pose_at_locked()
        response_east_m, response_north_m = lat_lon_to_local_m(
            MAP_REFERENCE_LATITUDE,
            MAP_REFERENCE_LONGITUDE,
            latitude,
            longitude,
        )
        self.active_response_task_id = task_id
        self.response_east_m = response_east_m
        self.response_north_m = response_north_m
        self.response_heading_deg = heading
        self.response_camera_yaw_deg = heading

        estimate_east_m = float(task["estimated_east_m"])
        estimate_north_m = float(task["estimated_north_m"])
        delta_east_m = response_east_m - estimate_east_m
        delta_north_m = response_north_m - estimate_north_m

        if math.hypot(delta_east_m, delta_north_m) > 0.0:
            self.orbit_phase_deg = math.degrees(math.atan2(delta_north_m, delta_east_m)) % 360.0

    def _task_state_update_key(
        self,
        status: str,
        owner_drone_id: object,
        owner_drone_name: object,
    ) -> tuple[int, int, str]:
        precedence = {
            "cancelled": 8,
            "released": 7,
            PAYLOAD_AWAITING_LOCAL_ACQUISITION_STATUS: 6,
            PAYLOAD_APPROACHING_STATUS: 5,
            PAYLOAD_WAITING_FOR_TRACK_STATUS: 4,
            "orbiting": 4,
            "transitioning": 3,
            "reported": 2,
            "assigned_remote": 1,
        }.get(status, 0)

        try:
            owner_id = int(owner_drone_id) if owner_drone_id is not None else 1_000_000_000
        except (TypeError, ValueError):
            owner_id = 1_000_000_000

        return (-precedence, owner_id, str(owner_drone_name or ""))

    def _claim_cost_to_estimate_locked(self, east_m: float, north_m: float) -> float:
        latitude, longitude, _heading = self._pose_at_locked()
        drone_east_m, drone_north_m = lat_lon_to_local_m(
            MAP_REFERENCE_LATITUDE,
            MAP_REFERENCE_LONGITUDE,
            latitude,
            longitude,
        )
        return math.hypot(east_m - drone_east_m, north_m - drone_north_m)

    def _task_claim_payload_locked(self, task: dict[str, object], status: str = "claiming") -> dict[str, object]:
        east_m = float(task["estimated_east_m"])
        north_m = float(task["estimated_north_m"])
        revision = int(task.get("revision", 0)) + (0 if status == "claiming" else 1)
        task_type = self._task_type(task)
        contact_ref = self._task_contact_ref(task)

        if status != "claiming":
            task["revision"] = revision

        payload = {
            "schema_version": SCHEMA_VERSION,
            "app_build": APPLICATION_BUILD,
            "scenario_run_id": str(self.scenario_seed),
            "task_id": str(task["task_id"]),
            "task_type": task_type,
            "contact_ref": contact_ref,
            "signature_id": str(task["signature_id"]),
            "object_label": str(task.get("object_label", "objective")),
            "display_label": str(task.get("display_label", self._task_display_label(task.get("object_label", "objective")))),
            "claimant_drone_id": self.drone_id,
            "claimant_drone_name": self.name,
            "claimant_team_configuration": self.team_configuration.to_dict(),
            "claimant_payload_reservations": dict(self.payload_reservations_by_task),
            "claimant_available_payload_inventory": self._available_payload_inventory_locked(),
            "owner_drone_id": task.get("assigned_drone_id"),
            "owner_drone_name": task.get("assigned_drone_name"),
            "status": status,
            "task_state": status,
            "revision": revision,
            "assignment_epoch": int(task.get("assignment_epoch", 0)),
            "playbook": self.selected_playbook,
            "playbook_config_version": self.playbook_config_version,
            "cost": self._claim_cost_to_estimate_locked(east_m, north_m),
            "created_at": float(task["created_at"]),
            "updated_at": self.simulation_time,
            "estimated_east_m": east_m,
            "estimated_north_m": north_m,
            "estimated_latitude": float(task["estimated_latitude"]),
            "estimated_longitude": float(task["estimated_longitude"]),
            "orbit_radius_m": ORBIT_RADIUS_M,
            "election_deadline": float(task["election_deadline"]),
        }

        if task_type == TASK_TYPE_PAYLOAD_DROP:
            payload.update(
                {
                    "requested_payload_type": str(task.get("requested_payload_type", "")),
                    "requested_payload_quantity": int(task.get("requested_payload_quantity", 0) or 0),
                    "release_authorization_required": bool(task.get("release_authorization_required", True)),
                    "reservation_state": str(task.get("reservation_state", "not_reserved")),
                    "reserved_payload_quantity": int(task.get("reserved_payload_quantity", 0) or 0),
                    "payload_standoff_m": float(task.get("payload_standoff_m", self.payload_standoff_m)),
                },
            )

        return payload

    def _ensure_response_task_locked(self, observation: Observation, object_map: dict[str, object]) -> None:
        assert observation.priority_evidence is not None
        signature_id = str(observation.priority_evidence["signature_id"])
        task_id = task_id_for_contact(self.scenario_seed, signature_id, TASK_TYPE_ISR_OBSERVE)

        if task_id in self.cancelled_task_ids:
            return

        if task_id in self.response_tasks:
            task = self.response_tasks[task_id]
            task["object_label"] = str(object_map.get("label", task.get("object_label", observation.label)))
            task["display_label"] = self._task_display_label(task["object_label"])
            task["last_observation_at"] = max(float(task.get("last_observation_at", 0.0)), observation.timestamp)
            task["updated_at"] = self.simulation_time
            return

        east_m = float(object_map["east_m"])
        north_m = float(object_map["north_m"])
        infeasible = abs(east_m) + ORBIT_RADIUS_M > SCENARIO_BOUND_EAST_M or abs(north_m) + ORBIT_RADIUS_M > SCENARIO_BOUND_NORTH_M
        local_available = self.active_response_task_id is None
        status = "queued" if not local_available else "electing"
        task = self._new_response_task_locked(
            contact_ref=signature_id,
            task_type=TASK_TYPE_ISR_OBSERVE,
            object_label=observation.label,
            estimated_east_m=east_m,
            estimated_north_m=north_m,
            estimated_latitude=float(object_map["latitude"]),
            estimated_longitude=float(object_map["longitude"]),
            last_observation_at=observation.timestamp,
            status="infeasible" if infeasible else status,
            event="Priority evidence accepted; deterministic claim election started.",
        )
        self.response_tasks[task_id] = task

        if infeasible:
            task["event"] = "Requested orbit is infeasible within scenario bounds."
            return

        if local_available and self._task_is_executable_by_this_node_locked(task):
            self.active_response_task_id = task_id
            claim = self._task_claim_payload_locked(task)
            self.claims.setdefault(task_id, {})[self.drone_id] = claim
            self.outbound_claims.append(claim)
        else:
            self.response_task_queue.append(task_id)

    def _drone_has_active_assignment_locked(self, drone_id: int, excluding_task_id: str | None = None) -> bool:
        for task_id, task in self.response_tasks.items():
            if task_id == excluding_task_id:
                continue

            if task.get("assigned_drone_id") != drone_id:
                continue

            if task.get("status") in {"transitioning", "orbiting", "assigned_remote", *PAYLOAD_EXECUTION_STATUSES}:
                return True

        return False

    def _claim_is_current_and_available_locked(self, claim: dict[str, object], task: dict[str, object]) -> bool:
        if str(claim.get("scenario_run_id")) != str(self.scenario_seed):
            return False

        if str(claim.get("task_id")) != str(task.get("task_id")):
            return False

        if str(claim.get("task_state", claim.get("status", "claiming"))) != "claiming":
            return False

        if int(claim.get("assignment_epoch", task.get("assignment_epoch", 0))) != int(task.get("assignment_epoch", 0)):
            return False

        if int(claim.get("playbook_config_version", self.playbook_config_version)) != int(task.get("playbook_config_version", self.playbook_config_version)):
            return False

        claimant_drone_id = int(claim["claimant_drone_id"])
        task_type = self._task_type(task)
        advertised = claim.get("claimant_team_configuration")

        if isinstance(advertised, dict):
            try:
                advertised_config = TeamMemberConfiguration.from_mapping(advertised)
            except TeamConfigurationError:
                return False

            if task_type == TASK_TYPE_ISR_OBSERVE and ("isr_observe" not in advertised_config.capabilities or not advertised_config.available):
                return False
            if task_type == TASK_TYPE_PAYLOAD_DROP:
                requested_payload_type = str(task.get("requested_payload_type", ""))
                requested_payload_quantity = int(task.get("requested_payload_quantity", 0) or 0)
                if (
                    "payload_drop" not in advertised_config.capabilities
                    or not advertised_config.available
                    or requested_payload_quantity <= 0
                    or advertised_config.payload_inventory.get(requested_payload_type, 0) < requested_payload_quantity
                ):
                    return False
        elif task_type != TASK_TYPE_ISR_OBSERVE:
            return False

        if claimant_drone_id == self.drone_id and not self._task_is_executable_by_this_node_locked(task, str(task["task_id"])):
            return False

        return not self._drone_has_active_assignment_locked(claimant_drone_id, str(task["task_id"]))

    def _release_current_assignment_locked(self, task: dict[str, object], owner_task_state: str = "released") -> None:
        task["previous_assigned_drone_id"] = task.get("assigned_drone_id")
        task["previous_assigned_drone_name"] = task.get("assigned_drone_name")
        task["assigned_drone_id"] = None
        task["assigned_drone_name"] = None
        task["owner_task_state"] = owner_task_state
        task["status"] = "reported"
        task["updated_at"] = self.simulation_time

    def _make_task_allocatable_locked(self, task_id: str, task: dict[str, object]) -> None:
        if self.selected_playbook != PLAYBOOK_OBSERVE_ORBIT:
            return

        if str(task.get("task_id")) in self.cancelled_task_ids:
            return

        if task.get("status") not in {"reported", "released"}:
            return

        if self.active_response_task_id is None and self._task_is_executable_by_this_node_locked(task):
            task["status"] = "electing"
            task["assignment_epoch"] = int(task.get("assignment_epoch", 0)) + 1
            task["election_deadline"] = self.simulation_time + TASK_ELECTION_SECONDS
            task["playbook_config_version"] = self.playbook_config_version
            task["updated_at"] = self.simulation_time
            if task.get("owner_task_state") != "released":
                task["owner_task_state"] = None
            self.active_response_task_id = task_id
            claim = self._task_claim_payload_locked(task)
            self.claims.setdefault(task_id, {})[self.drone_id] = claim
            self.outbound_claims.append(claim)
        elif task_id not in self.response_task_queue:
            task["status"] = "queued"
            task["playbook_config_version"] = self.playbook_config_version
            task["updated_at"] = self.simulation_time
            if task.get("owner_task_state") != "released":
                task["owner_task_state"] = None
            self.response_task_queue.append(task_id)

    def _reconcile_eligible_work_locked(self) -> None:
        if self.selected_playbook != PLAYBOOK_OBSERVE_ORBIT:
            return

        for task_id, task in sorted(self.response_tasks.items()):
            self._make_task_allocatable_locked(task_id, task)

    def _record_claim_locked(self, payload: dict[str, object]) -> bool:
        if str(payload.get("scenario_run_id")) != str(self.scenario_seed):
            return False

        task_id = str(payload["task_id"])
        claimant_drone_id = int(payload["claimant_drone_id"])
        task_type = str(payload.get("task_type", TASK_TYPE_ISR_OBSERVE))
        contact_ref = str(payload.get("contact_ref", payload.get("signature_id", "")))

        if task_type not in SUPPORTED_TASK_TYPES:
            return False

        expected_task_id = task_id_for_contact(self.scenario_seed, contact_ref, task_type)
        if task_id != expected_task_id:
            return False

        if task_id in self.cancelled_task_ids:
            return False

        if "claimant_team_configuration" in payload:
            try:
                validate_advertised_team_member(payload["claimant_team_configuration"])
            except TeamConfigurationError:
                return False
        east_m = float(payload["estimated_east_m"])
        north_m = float(payload["estimated_north_m"])
        latitude, longitude = local_m_to_lat_lon(east_m, north_m)
        payload_status = str(payload.get("task_state", payload.get("status", "claiming")))
        signature_id = str(payload.get("signature_id", contact_ref))
        revision = int(payload.get("revision", 0))

        if task_id not in self.response_tasks:
            self.response_tasks[task_id] = self._new_response_task_locked(
                contact_ref=contact_ref,
                task_type=task_type,
                object_label=str(payload.get("object_label", "objective")),
                estimated_east_m=east_m,
                estimated_north_m=north_m,
                estimated_latitude=latitude,
                estimated_longitude=longitude,
                last_observation_at=float(payload.get("updated_at", self.simulation_time)),
                status=PAYLOAD_REQUEST_REPORTED_STATUS if task_type == TASK_TYPE_PAYLOAD_DROP else "queued",
                event="Peer task request received; local deterministic election joined if this node is eligible.",
                requested_payload_type=str(payload.get("requested_payload_type", "")) if task_type == TASK_TYPE_PAYLOAD_DROP else None,
                requested_payload_quantity=int(payload.get("requested_payload_quantity", 0) or 0) if task_type == TASK_TYPE_PAYLOAD_DROP else None,
            )
            self.response_tasks[task_id]["election_deadline"] = max(self.simulation_time + TASK_ELECTION_SECONDS, float(payload.get("election_deadline", self.simulation_time)))
            self.response_tasks[task_id]["assignment_epoch"] = int(payload.get("assignment_epoch", 0))

            self.response_tasks[task_id]["requesting_drone_id"] = payload.get("requesting_drone_id", payload.get("claimant_drone_id"))
            self.response_tasks[task_id]["requesting_drone_name"] = payload.get("requesting_drone_name", payload.get("claimant_drone_name"))

            if self.active_response_task_id is None and self._task_is_executable_by_this_node_locked(self.response_tasks[task_id]):
                self.response_tasks[task_id]["status"] = "electing"
                self.active_response_task_id = task_id
                local_claim = self._task_claim_payload_locked(self.response_tasks[task_id])
                self.claims.setdefault(task_id, {})[self.drone_id] = local_claim
                self.outbound_claims.append(local_claim)
            elif task_id not in self.response_task_queue:
                self.response_task_queue.append(task_id)

        task = self.response_tasks[task_id]

        if self._task_type(task) != task_type or self._task_contact_ref(task) != contact_ref:
            return False

        if task_type == TASK_TYPE_PAYLOAD_DROP:
            requested_payload_type = str(payload.get("requested_payload_type", ""))
            requested_payload_quantity = int(payload.get("requested_payload_quantity", 0) or 0)
            if task_id in self.response_tasks and not self._payload_request_metadata_matches(self.response_tasks[task_id], requested_payload_type, requested_payload_quantity):
                self.response_tasks[task_id]["metadata_conflict"] = True
                self.response_tasks[task_id]["event"] = "Conflicting payload request metadata rejected; existing request was not overwritten."
                return False

        if (
            task_type == TASK_TYPE_PAYLOAD_DROP
            and payload_status == PAYLOAD_REQUEST_REPORTED_STATUS
            and payload.get("owner_drone_id") is None
        ):
            incoming_epoch = int(payload.get("assignment_epoch", task.get("assignment_epoch", 0)))
            current_epoch = int(task.get("assignment_epoch", 0))
            incoming_config_version = int(payload.get("playbook_config_version", task.get("playbook_config_version", 0)))
            current_config_version = int(task.get("playbook_config_version", 0))
            current_revision = int(task.get("revision", 0))

            if incoming_epoch < current_epoch:
                return False

            if revision < current_revision:
                return False

            if incoming_config_version < current_config_version:
                return False

            if task.get("status") in {"cancelled", "assigned_remote", *PAYLOAD_EXECUTION_STATUSES}:
                return False

            task["assignment_epoch"] = max(current_epoch, incoming_epoch)
            task["playbook_config_version"] = max(current_config_version, incoming_config_version)
            task["updated_at"] = self.simulation_time
            task["event"] = "Peer payload-drop request received; local election remains eligible if this node can execute it."
            self._make_task_allocatable_locked(task_id, task)
            return True

        if payload_status in {"transitioning", "orbiting", "reported", "cancelled", "released", *PAYLOAD_EXECUTION_STATUSES}:
            current_revision = int(task.get("revision", 0))
            incoming_epoch = int(payload.get("assignment_epoch", task.get("assignment_epoch", 0)))
            current_epoch = int(task.get("assignment_epoch", 0))

            if incoming_epoch < current_epoch:
                return False

            if revision < current_revision:
                return False

            incoming_key = self._task_state_update_key(
                payload_status,
                payload.get("owner_drone_id"),
                payload.get("owner_drone_name"),
            )
            current_key = self._task_state_update_key(
                str(task.get("owner_task_state", task.get("status", "queued"))),
                task.get("assigned_drone_id"),
                task.get("assigned_drone_name"),
            )

            if revision == current_revision and incoming_key > current_key:
                return False

            owner_drone_id = payload.get("owner_drone_id")
            owner_drone_name = payload.get("owner_drone_name")
            current_owner_id = task.get("assigned_drone_id")
            claimant_drone_id = int(payload.get("claimant_drone_id", -1))

            if payload_status in {"released", "reported"} and current_owner_id is not None:
                try:
                    current_owner_int = int(current_owner_id)
                except (TypeError, ValueError):
                    current_owner_int = -1

                try:
                    payload_owner_int = int(owner_drone_id) if owner_drone_id is not None else -1
                except (TypeError, ValueError):
                    payload_owner_int = -1

                if current_owner_int not in {payload_owner_int, claimant_drone_id}:
                    return False

            if payload_status in {"transitioning", "orbiting", *PAYLOAD_EXECUTION_STATUSES} and owner_drone_id is not None:
                try:
                    incoming_owner_int = int(owner_drone_id)
                except (TypeError, ValueError):
                    return False

                if incoming_owner_int == self.drone_id and not self._task_is_executable_by_this_node_locked(task, task_id):
                    return False

                if incoming_owner_int == self.drone_id and self._drone_has_active_assignment_locked(self.drone_id, task_id):
                    self._make_task_allocatable_locked(task_id, task)
                    return False

            task["revision"] = revision
            task["assignment_epoch"] = max(current_epoch, incoming_epoch)
            task["assigned_drone_id"] = owner_drone_id
            task["assigned_drone_name"] = owner_drone_name
            task["object_label"] = str(payload.get("object_label", task.get("object_label", "objective")))
            task["display_label"] = str(payload.get("display_label", task.get("display_label", self._task_display_label(task.get("object_label", "objective")))))
            task["updated_at"] = self.simulation_time
            task["owner_task_state"] = payload_status
            task["playbook_config_version"] = int(payload.get("playbook_config_version", task.get("playbook_config_version", 0)))
            task["event"] = f"Peer task state received: {payload_status}."

            if payload_status == "cancelled":
                task["previous_assigned_drone_id"] = task.get("assigned_drone_id")
                task["previous_assigned_drone_name"] = task.get("assigned_drone_name")
                task["assigned_drone_id"] = None
                task["assigned_drone_name"] = None
                self._release_payload_reservation_locked(task)
                task["status"] = "cancelled"
                self.cancelled_task_ids.add(task_id)

                if task_id in self.response_task_queue:
                    self.response_task_queue.remove(task_id)

                if self.active_response_task_id == task_id:
                    self.active_response_task_id = None
                    self._clear_response_guidance_locked()
            elif payload_status in {"released", "reported"}:
                self._release_payload_reservation_locked(task)
                self._release_current_assignment_locked(task, payload_status)

                if self.active_response_task_id == task_id:
                    self.active_response_task_id = None
                    self._clear_response_guidance_locked()

                if payload_status == "released":
                    self._make_task_allocatable_locked(task_id, task)
                else:
                    self._promote_next_queued_task_locked()
            elif int(owner_drone_id) == self.drone_id:
                if self._task_type(task) == TASK_TYPE_PAYLOAD_DROP:
                    self._initialize_payload_response_control_locked(task)
                    self._reserve_payload_inventory_locked(task)
                else:
                    self._initialize_local_response_control_locked(task)
                task["status"] = payload_status
            else:
                task["status"] = "assigned_remote"

                if self.active_response_task_id == task_id:
                    self.active_response_task_id = None
                    self._clear_response_guidance_locked()
                    self._promote_next_queued_task_locked()

            return True

        if not self._claim_is_current_and_available_locked(payload, task):
            return False

        self.claims.setdefault(task_id, {})[claimant_drone_id] = dict(payload)
        return True

    def _demote_other_active_assignments_for_owner_locked(self, owner_drone_id: int, active_task_id: str) -> None:
        for task_id, task in self.response_tasks.items():
            if task_id == active_task_id:
                continue

            if task.get("assigned_drone_id") != owner_drone_id:
                continue

            if task.get("status") not in {"transitioning", "orbiting", "assigned_remote"}:
                continue

            task["event"] = "Local invariant check observed another active assignment; no global release was fabricated."


    def receive_claim(self, payload: dict[str, object], payload_byte_count: int = 0) -> bool:
        try:
            with self.lock:
                self.peer_claim_messages_received += 1
                self._record_peer_payload_bytes_locked(payload_byte_count, "received")
                self._advance_locked()
                return self._record_claim_locked(payload)
        except (KeyError, TypeError, ValueError):
            return False

    def pop_outbound_claims(self) -> list[dict[str, object]]:
        with self.lock:
            claims = list(self.outbound_claims)
            self.outbound_claims.clear()
            return claims

    def broadcast_claim(self, claim: dict[str, object]) -> None:
        payload = json.dumps(claim).encode("utf-8")

        for peer in self.peers:
            request = urllib.request.Request(
                f"{peer}/claims",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            started_at = time.monotonic()

            try:
                with urllib.request.urlopen(request, timeout=0.4):
                    pass
            except (OSError, urllib.error.URLError) as error:
                with self.lock:
                    self.peer_send_failures += 1
                    self.peer_connection_state[peer]["state"] = "error"
                    self.peer_connection_state[peer]["last_error"] = str(error)
                    self.peer_connection_state[peer]["latest_error"] = str(error)
                    self.peer_connection_state[peer]["send_failures"] = int(self.peer_connection_state[peer]["send_failures"]) + 1
                continue

            elapsed_ms = (time.monotonic() - started_at) * 1000.0
            with self.lock:
                self.peer_claim_messages_sent += 1
                self._record_peer_payload_bytes_locked(len(payload), "sent")
                self.peer_connection_state[peer]["state"] = "ok"
                self.peer_connection_state[peer]["last_error"] = None
                self.peer_connection_state[peer]["latest_error"] = None
                self.peer_connection_state[peer]["last_success_time"] = self.simulation_time
                self.peer_connection_state[peer]["last_success_wall_time"] = time.monotonic()
                self.peer_connection_state[peer]["last_rtt_ms"] = elapsed_ms
                self.peer_connection_state[peer]["claim_messages_sent"] = int(self.peer_connection_state[peer]["claim_messages_sent"]) + 1
                self.peer_connection_state[peer]["payload_bytes_sent"] = int(self.peer_connection_state[peer]["payload_bytes_sent"]) + len(payload)

    def _resolve_response_tasks_locked(self) -> None:
        task = self.response_tasks.get(self.active_response_task_id or "")

        if not task or task["status"] != "electing" or self.simulation_time < float(task["election_deadline"]):
            return

        claims = [
            claim
            for claim in self.claims.get(str(task["task_id"]), {}).values()
            if self._claim_is_current_and_available_locked(claim, task)
        ]

        if not claims and self._task_is_executable_by_this_node_locked(task):
            claim = self._task_claim_payload_locked(task)
            self.claims.setdefault(str(task["task_id"]), {})[self.drone_id] = claim
            self.outbound_claims.append(claim)
            claims = [claim]

        if not claims:
            task["status"] = "queued"
            task["updated_at"] = self.simulation_time
            if str(task["task_id"]) not in self.response_task_queue:
                self.response_task_queue.append(str(task["task_id"]))
            self.active_response_task_id = None
            self._promote_next_queued_task_locked()
            return

        winning_claim = sorted(
            claims,
            key=lambda claim: (
                float(claim["cost"]),
                int(claim["claimant_drone_id"]),
                str(claim["claimant_drone_name"]),
                str(task["task_id"]),
            ),
        )[0]
        assigned_drone_id = int(winning_claim["claimant_drone_id"])
        if self._drone_has_active_assignment_locked(assigned_drone_id, str(task["task_id"])):
            self.claims.get(str(task["task_id"]), {}).pop(assigned_drone_id, None)
            task["status"] = "queued"
            task["updated_at"] = self.simulation_time
            if str(task["task_id"]) not in self.response_task_queue:
                self.response_task_queue.append(str(task["task_id"]))
            self.active_response_task_id = None
            self._promote_next_queued_task_locked()
            return

        task["assigned_drone_id"] = assigned_drone_id
        task["assigned_drone_name"] = str(winning_claim["claimant_drone_name"])
        task["assignment_epoch"] = int(task.get("assignment_epoch", 0))
        task["updated_at"] = self.simulation_time

        if self._task_type(task) == TASK_TYPE_PAYLOAD_DROP:
            if assigned_drone_id == self.drone_id:
                self._initialize_payload_response_control_locked(task)
                self._reserve_payload_inventory_locked(task)
                task["status"] = PAYLOAD_APPROACHING_STATUS
                task["owner_task_state"] = PAYLOAD_APPROACHING_STATUS
                task["event"] = "Payload drone won deterministic election and is approaching the local shared contact estimate."
                self.outbound_claims.append(self._task_claim_payload_locked(task, PAYLOAD_APPROACHING_STATUS))
            else:
                task["status"] = "assigned_remote"
                task["owner_task_state"] = PAYLOAD_APPROACHING_STATUS
                task["event"] = "Another payload-capable drone won deterministic election."
                self.active_response_task_id = None
                self._promote_next_queued_task_locked()
        elif self.selected_playbook == PLAYBOOK_REPORT_CONTINUE:
            task["status"] = "reported"
            task["playbook_config_version"] = self.playbook_config_version
            task["event"] = "Priority objective reported; all drones continue patrol."
            self.outbound_claims.append(self._task_claim_payload_locked(task, "reported"))
            self.active_response_task_id = None
            self._promote_next_queued_task_locked()
        elif assigned_drone_id == self.drone_id:
            self._initialize_local_response_control_locked(task)
            task["status"] = "transitioning"
            task["playbook_config_version"] = self.playbook_config_version
            task["owner_task_state"] = "transitioning"
            task["event"] = "This drone won deterministic election and is transitioning to orbit."
            self.outbound_claims.append(self._task_claim_payload_locked(task, "transitioning"))
        else:
            task["status"] = "assigned_remote"
            task["playbook_config_version"] = self.playbook_config_version
            task["owner_task_state"] = "transitioning"
            task["event"] = "Another drone won deterministic election; this drone continues patrol."
            self.active_response_task_id = None
            self._promote_next_queued_task_locked()

    def _promote_next_queued_task_locked(self) -> None:
        if self.active_response_task_id is not None:
            return

        queued_task_ids = [
            task_id
            for task_id in self.response_task_queue
            if self.response_tasks.get(task_id, {}).get("status") == "queued"
            and str(self.response_tasks.get(task_id, {}).get("task_id")) not in self.cancelled_task_ids
            and self._task_is_executable_by_this_node_locked(self.response_tasks.get(task_id, {}), task_id)
        ]

        if not queued_task_ids:
            return

        next_task_id = sorted(queued_task_ids)[0]
        self.response_task_queue.remove(next_task_id)
        task = self.response_tasks[next_task_id]
        task["status"] = "electing"
        task["assignment_epoch"] = int(task.get("assignment_epoch", 0)) + 1
        task["election_deadline"] = self.simulation_time + TASK_ELECTION_SECONDS
        task["updated_at"] = self.simulation_time
        self.active_response_task_id = next_task_id
        claim = self._task_claim_payload_locked(task)
        self.claims.setdefault(next_task_id, {})[self.drone_id] = claim
        self.outbound_claims.append(claim)

    def _contact_estimate_for_task_locked(self, task: dict[str, object]) -> dict[str, object] | None:
        contact_ref = self._task_contact_ref(task)

        for object_map in self.objects.values():
            if str(object_map.get("priority_signature") or "") != contact_ref:
                continue

            if object_map.get("tracking_model") == "constant_velocity":
                self._refresh_moving_track_payload_locked(object_map)
                if object_map.get("track_status") == "lost":
                    return None
            else:
                self._refresh_fused_object_locked(object_map)

            if bool(object_map.get("stale", False)):
                return None

            return object_map

        return None

    def _update_payload_guidance_locked(self, elapsed_seconds: float) -> None:
        if not self._has_local_payload_control_locked() or elapsed_seconds <= 0.0:
            return

        task = self.response_tasks[self.active_response_task_id or ""]
        estimate = self._contact_estimate_for_task_locked(task)

        if estimate is None:
            previous_status = str(task.get("status", ""))
            last_broadcast_at = float(task.get("last_state_broadcast_at", -math.inf) or -math.inf)
            should_broadcast = (
                previous_status != PAYLOAD_WAITING_FOR_TRACK_STATUS
                or self.simulation_time - last_broadcast_at >= PAYLOAD_STATE_HEARTBEAT_SECONDS
            )
            task["status"] = PAYLOAD_WAITING_FOR_TRACK_STATUS
            task["owner_task_state"] = PAYLOAD_WAITING_FOR_TRACK_STATUS
            task["event"] = "Payload approach paused: no usable local shared contact estimate is available."
            task["updated_at"] = self.simulation_time

            if should_broadcast:
                self.outbound_claims.append(self._task_claim_payload_locked(task, PAYLOAD_WAITING_FOR_TRACK_STATUS))
                task["last_state_broadcast_at"] = self.simulation_time

            return

        assert self.response_east_m is not None
        assert self.response_north_m is not None
        assert self.response_heading_deg is not None

        target_east_m = clamp(float(estimate["east_m"]), -SCENARIO_BOUND_EAST_M, SCENARIO_BOUND_EAST_M)
        target_north_m = clamp(float(estimate["north_m"]), -SCENARIO_BOUND_NORTH_M, SCENARIO_BOUND_NORTH_M)
        delta_east = target_east_m - self.response_east_m
        delta_north = target_north_m - self.response_north_m
        distance = math.hypot(delta_east, delta_north)
        stand_off_m = float(task.get("payload_standoff_m", self.payload_standoff_m))

        if distance <= stand_off_m:
            self.response_camera_yaw_deg = math.degrees(math.atan2(delta_east, delta_north)) % 360.0 if distance > 0.0 else self.response_heading_deg
            if task["status"] != PAYLOAD_AWAITING_LOCAL_ACQUISITION_STATUS:
                task["status"] = PAYLOAD_AWAITING_LOCAL_ACQUISITION_STATUS
                task["owner_task_state"] = PAYLOAD_AWAITING_LOCAL_ACQUISITION_STATUS
                task["event"] = "Payload drone is holding at stand-off awaiting local acquisition; no release or delivery is claimed."
                self.outbound_claims.append(self._task_claim_payload_locked(task, PAYLOAD_AWAITING_LOCAL_ACQUISITION_STATUS))
                task["last_state_broadcast_at"] = self.simulation_time
            task["updated_at"] = self.simulation_time
            task["estimated_east_m"] = target_east_m
            task["estimated_north_m"] = target_north_m
            task["estimated_latitude"] = float(estimate["latitude"])
            task["estimated_longitude"] = float(estimate["longitude"])
            return

        max_distance = MAX_RESPONSE_SPEED_MPS * elapsed_seconds
        travel_distance = max(0.0, min(max_distance, distance - stand_off_m))
        if travel_distance > 0.0:
            scale = travel_distance / distance
            self.response_east_m += delta_east * scale
            self.response_north_m += delta_north * scale

        desired_heading = math.degrees(math.atan2(delta_east, delta_north)) % 360.0
        self.response_heading_deg = move_angle_toward(
            self.response_heading_deg,
            desired_heading,
            MAX_RESPONSE_TURN_RATE_DEG_PER_SECOND * elapsed_seconds,
        )
        self.response_camera_yaw_deg = desired_heading

        if task["status"] != PAYLOAD_APPROACHING_STATUS:
            task["status"] = PAYLOAD_APPROACHING_STATUS
            task["owner_task_state"] = PAYLOAD_APPROACHING_STATUS
            task["event"] = "Payload approach resumed using a valid local shared contact estimate."
            self.outbound_claims.append(self._task_claim_payload_locked(task, PAYLOAD_APPROACHING_STATUS))
            task["last_state_broadcast_at"] = self.simulation_time

        task["updated_at"] = self.simulation_time
        task["estimated_east_m"] = target_east_m
        task["estimated_north_m"] = target_north_m
        task["estimated_latitude"] = float(estimate["latitude"])
        task["estimated_longitude"] = float(estimate["longitude"])
        task["range_to_contact_estimate_m"] = distance

    def _update_response_guidance_locked(self, elapsed_seconds: float) -> None:
        if not self._has_local_response_control_locked() or elapsed_seconds <= 0.0:
            return

        task = self.response_tasks[self.active_response_task_id or ""]
        estimate_east_m = float(task["estimated_east_m"])
        estimate_north_m = float(task["estimated_north_m"])
        assert self.response_east_m is not None
        assert self.response_north_m is not None
        assert self.response_heading_deg is not None

        self.orbit_phase_deg = (self.orbit_phase_deg + (ORBIT_ANGULAR_RATE_DEG_PER_SECOND * elapsed_seconds)) % 360.0
        phase_rad = math.radians(self.orbit_phase_deg)
        target_east_m = estimate_east_m + (math.cos(phase_rad) * ORBIT_RADIUS_M)
        target_north_m = estimate_north_m + (math.sin(phase_rad) * ORBIT_RADIUS_M)
        max_distance = MAX_RESPONSE_SPEED_MPS * elapsed_seconds
        delta_east = target_east_m - self.response_east_m
        delta_north = target_north_m - self.response_north_m
        distance = math.hypot(delta_east, delta_north)

        if distance <= max_distance or distance == 0.0:
            self.response_east_m = target_east_m
            self.response_north_m = target_north_m
        else:
            scale = max_distance / distance
            self.response_east_m += delta_east * scale
            self.response_north_m += delta_north * scale

        desired_heading = math.degrees(math.atan2(delta_east, delta_north)) % 360.0
        self.response_heading_deg = move_angle_toward(
            self.response_heading_deg,
            desired_heading,
            MAX_RESPONSE_TURN_RATE_DEG_PER_SECOND * elapsed_seconds,
        )
        camera_bearing = math.degrees(
            math.atan2(estimate_east_m - self.response_east_m, estimate_north_m - self.response_north_m),
        ) % 360.0
        self.response_camera_yaw_deg = camera_bearing

        if distance < 6.0 and task["status"] == "transitioning":
            task["status"] = "orbiting"
            task["event"] = "Assigned drone entered standoff orbit around estimated position."
            self.outbound_claims.append(self._task_claim_payload_locked(task, "orbiting"))

        task["updated_at"] = self.simulation_time
        task["camera_yaw_deg"] = self.response_camera_yaw_deg

    def _update_return_to_patrol_locked(self, elapsed_seconds: float) -> None:
        if (
            elapsed_seconds <= 0.0
            or self.return_east_m is None
            or self.return_north_m is None
            or self.return_heading_deg is None
            or self._has_local_execution_control_locked()
        ):
            return

        patrol_latitude, patrol_longitude, patrol_heading = patrol_pose_on_route(self.routes[self.drone_id], self.simulation_time)
        patrol_east_m, patrol_north_m = lat_lon_to_local_m(
            MAP_REFERENCE_LATITUDE,
            MAP_REFERENCE_LONGITUDE,
            patrol_latitude,
            patrol_longitude,
        )
        delta_east = patrol_east_m - self.return_east_m
        delta_north = patrol_north_m - self.return_north_m
        distance = math.hypot(delta_east, delta_north)
        max_distance = MAX_RESPONSE_SPEED_MPS * elapsed_seconds

        if distance <= max_distance or distance < 1.0:
            self.return_east_m = None
            self.return_north_m = None
            self.return_heading_deg = None
            return

        scale = max_distance / distance
        self.return_east_m += delta_east * scale
        self.return_north_m += delta_north * scale
        target_heading = math.degrees(math.atan2(delta_east, delta_north)) % 360.0
        self.return_heading_deg = move_angle_toward(
            self.return_heading_deg,
            target_heading if distance > 2.0 else patrol_heading,
            MAX_RESPONSE_TURN_RATE_DEG_PER_SECOND * elapsed_seconds,
        )

    def cancel_response(self, task_id: str | None = None) -> None:
        with self.lock:
            self._advance_locked()
            target_task_id = task_id or self.active_response_task_id
            task = self.response_tasks.get(target_task_id or "")

            if task is not None:
                signature_id = str(task["signature_id"])
                self.cancelled_task_ids.add(str(task["task_id"]))
                task["previous_assigned_drone_id"] = task.get("assigned_drone_id")
                task["previous_assigned_drone_name"] = task.get("assigned_drone_name")
                task["assigned_drone_id"] = None
                task["assigned_drone_name"] = None
                task["status"] = "cancelled"
                task["owner_task_state"] = "cancelled"
                task["event"] = "Response cancelled; assigned drone returns to patrol."
                task["updated_at"] = self.simulation_time

                if str(task["task_id"]) in self.response_task_queue:
                    self.response_task_queue.remove(str(task["task_id"]))

                self._release_payload_reservation_locked(task)
                self.outbound_claims.append(self._task_claim_payload_locked(task, "cancelled"))

            if self.active_response_task_id == target_task_id:
                self._start_return_to_patrol_locked()
                self.active_response_task_id = None
                self._clear_response_guidance_locked()


class DroneRequestHandler(http.server.BaseHTTPRequestHandler):
    """Serve telemetry, maps, controls, and peer observations for one drone."""

    simulator: DroneSimulator

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._send_common_headers()
        self.end_headers()

    def do_GET(self) -> None:
        if self.path == "/status":
            self._send_json(asdict(self.simulator.status()))
            return

        if self.path == "/objects":
            self._send_json(self.simulator.objects_payload())
            return

        if self.path == "/truth":
            self._send_json(self.simulator.truth_payload())
            return

        if self.path == "/control":
            self._send_json(self.simulator.control_payload())
            return

        if self.path == "/network":
            self._send_json(self.simulator.network_payload())
            return

        if self.path.startswith("/trace?"):
            observation_id = self.path.split("observation_id=", 1)[-1].split("&", 1)[0]
            self._send_json(self.simulator.trace_observation_payload(observation_id))
            return

        self.send_error(404)

    def do_POST(self) -> None:
        try:
            payload, payload_byte_count = self._read_json_payload()
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON")
            return

        if self.path == "/observations":
            accepted = self.simulator.receive_observation(payload, payload_byte_count)
            self._send_json({"accepted": accepted})
            return

        if self.path == "/claims":
            accepted = self.simulator.receive_claim(payload, payload_byte_count)
            self._send_json({"accepted": accepted})
            return

        if self.path == "/control":
            requested_priority_count = (
                int(payload["priority_count"])
                if "priority_count" in payload
                else None
            )
            selected_playbook = (
                str(payload["playbook"])
                if "playbook" in payload
                else None
            )

            scenario_id = str(payload["scenario"]) if "scenario" in payload else None
            initial_observer_id = int(payload["initial_observer_id"]) if "initial_observer_id" in payload else None
            next_observer_id = int(payload["next_observer_id"]) if "next_observer_id" in payload else None

            if "scenario_seed" in payload:
                scenario_seed = int(payload["scenario_seed"])
                self.simulator.reset(
                    scenario_seed,
                    requested_priority_count,
                    selected_playbook,
                    scenario_id,
                    initial_observer_id,
                    next_observer_id,
                    str(payload.get("scenario_run_id", new_run_id(scenario_seed))),
                )
            elif bool(payload.get("new_random_scenario", False)):
                next_seed = time.time_ns() % 1_000_000_000
                self.simulator.reset(
                    next_seed,
                    requested_priority_count,
                    selected_playbook,
                    scenario_id,
                    initial_observer_id,
                    next_observer_id,
                    str(payload.get("scenario_run_id", new_run_id(next_seed))),
                )
            elif bool(payload.get("reset", False)) or bool(payload.get("restart_scenario", False)):
                self.simulator.reset(
                    None,
                    requested_priority_count,
                    selected_playbook,
                    scenario_id,
                    initial_observer_id,
                    next_observer_id,
                    str(payload.get("scenario_run_id", new_run_id(self.simulator.scenario_seed))),
                )

            if "payload_requests_enabled" in payload or "payload_type" in payload or "payload_quantity" in payload or "payload_standoff_m" in payload:
                self.simulator.configure_payload_requests(
                    bool(payload["payload_requests_enabled"]) if "payload_requests_enabled" in payload else None,
                    str(payload["payload_type"]) if "payload_type" in payload else None,
                    int(payload["payload_quantity"]) if "payload_quantity" in payload else None,
                    float(payload["payload_standoff_m"]) if "payload_standoff_m" in payload else None,
                )

            if "paused" in payload:
                self.simulator.set_paused(bool(payload["paused"]))

            if "sharing_enabled" in payload:
                self.simulator.set_sharing_enabled(bool(payload["sharing_enabled"]))

            if "simulation_time" in payload:
                self.simulator.set_simulation_time(float(payload["simulation_time"]))

            if "playbook" in payload:
                self.simulator.set_playbook(
                    str(payload["playbook"]),
                    int(payload.get("playbook_config_version", 0)),
                )

            if bool(payload.get("cancel_response", False)):
                self.simulator.cancel_response(
                    str(payload["cancel_response_task_id"])
                    if "cancel_response_task_id" in payload
                    else None,
                )

            redelivered = False
            if "redeliver_observation_id" in payload:
                redelivered = self.simulator.redeliver_observation(str(payload["redeliver_observation_id"]))
            elif bool(payload.get("redeliver_last", False)):
                redelivered = self.simulator.redeliver_last_observation()

            response = self.simulator.control_payload()
            response["redelivered"] = redelivered
            self._send_json(response)
            return

        self.send_error(404)

    def _read_json_payload(self) -> tuple[dict[str, object], int]:
        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length)
        return json.loads(body.decode("utf-8") or "{}"), len(body)

    def _send_json(self, payload: dict[str, object]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self._send_common_headers()
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_common_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def log_message(self, format: str, *args: object) -> None:
        return


class ThreadingTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one simulated drone process.")
    parser.add_argument("--id", type=int, required=True, dest="drone_id", choices=sorted(DRONE_PROFILES))
    parser.add_argument("--name", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--peers", nargs="*", default=[])
    parser.add_argument("--seed", type=int, default=DEFAULT_SCENARIO_SEED)
    parser.add_argument("--priority-count", type=int, default=RANDOM_PRIORITY_OBJECT_COUNT)
    parser.add_argument("--playbook", choices=sorted(SUPPORTED_PLAYBOOKS), default=DEFAULT_PLAYBOOK)
    parser.add_argument("--scenario", choices=sorted(AVAILABLE_SCENARIOS), default=DEFAULT_SCENARIO_ID)
    parser.add_argument("--initial-observer-id", type=int, choices=sorted(DRONE_PROFILES), default=DEFAULT_INITIAL_OBSERVER_ID)
    parser.add_argument("--next-observer-id", type=int, choices=sorted(DRONE_PROFILES), default=DEFAULT_NEXT_OBSERVER_ID)
    parser.add_argument("--scenario-run-id", default=None)
    parser.add_argument("--team-config", default=None)
    parser.add_argument("--enable-payload-requests", action="store_true")
    parser.add_argument("--payload-type", default=DEFAULT_PAYLOAD_TYPE)
    parser.add_argument("--payload-quantity", type=int, default=DEFAULT_PAYLOAD_QUANTITY)
    parser.add_argument("--payload-standoff-m", type=float, default=DEFAULT_PAYLOAD_STANDOFF_M)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        team_config = load_team_configuration(args.team_config, tuple(sorted(DRONE_PROFILES))).members
    except TeamConfigurationError as error:
        print(f"Invalid team configuration: {error}", file=sys.stderr)
        return 2

    DroneRequestHandler.simulator = DroneSimulator(
        args.drone_id,
        args.name,
        args.peers,
        args.seed,
        args.priority_count,
        args.playbook,
        args.host,
        args.port,
        args.scenario,
        args.initial_observer_id,
        args.next_observer_id,
        args.scenario_run_id,
        dict(team_config),
        args.enable_payload_requests,
        args.payload_type,
        args.payload_quantity,
        args.payload_standoff_m,
    )

    stop_event = threading.Event()

    def request_stop(signum: int, frame: object) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    with ThreadingTCPServer((args.host, args.port), DroneRequestHandler) as server:
        server_thread = threading.Thread(target=server.serve_forever, name="drone-api", daemon=True)
        server_thread.start()

        def observe_and_share() -> None:
            while not stop_event.wait(0.25):
                observations = DroneRequestHandler.simulator.capture_observations()

                for observation in observations:
                    DroneRequestHandler.simulator.broadcast_observation(observation)

                for claim in DroneRequestHandler.simulator.pop_outbound_claims():
                    DroneRequestHandler.simulator.broadcast_claim(claim)

        observation_thread = threading.Thread(target=observe_and_share, name="drone-observer", daemon=True)
        observation_thread.start()

        print(
            f"{args.name} running at http://{args.host}:{args.port}/status "
            f"pid={os.getpid()} build={APPLICATION_BUILD} scenario={args.scenario!r} seed={args.seed} run={DroneRequestHandler.simulator.scenario_run_id}",
            flush=True,
        )

        try:
            while not stop_event.is_set():
                time.sleep(0.2)
        finally:
            server.shutdown()
            server_thread.join(timeout=5)
            stop_event.set()
            observation_thread.join(timeout=5)

    return 0


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        raise SystemExit(main())
