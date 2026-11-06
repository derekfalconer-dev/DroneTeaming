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
import socketserver
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass


SCHEMA_VERSION = "3.0"
APPLICATION_BUILD = "moving-search-noisy-fusion-2026-11-06"
SCENARIO_NAME = "Seeded noisy overlapping moving-search patrol"
DEFAULT_SCENARIO_SEED = 20261106
SCENARIO_OBJECT_COUNT = 24
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
    battery_percent: float
    timestamp: float
    paused: bool
    sharing_enabled: bool
    camera_footprint: list[dict[str, float]]
    flight_trail: list[dict[str, float]]
    diagnostics: dict[str, object]


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
    measurement_type: str = "point"
    object_id: str = ""


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


def generate_ground_truth_objects(seed: int) -> list[dict[str, float | str]]:
    rng = random.Random(seed)
    routes = generate_patrol_routes(seed)
    shared_east_m = routes[1].waypoints_m[3][0]
    shared_north_m = (routes[1].waypoints_m[3][1] + routes[2].waypoints_m[3][1]) / 2.0
    objects: list[dict[str, float | str]] = []

    def add_object(internal_id: str, label: str, east_m: float, north_m: float) -> None:
        latitude, longitude = local_m_to_lat_lon(east_m, north_m)
        objects.append(
            {
                "id": internal_id,
                "label": label,
                "latitude": latitude,
                "longitude": longitude,
            },
        )

    add_object(f"truth-{seed}-shared-rb-00", "vehicle", shared_east_m + 45.0, shared_north_m)
    add_object(f"truth-{seed}-shared-rb-01", "generator", shared_east_m + 82.0, shared_north_m + 8.0)
    add_object(f"truth-{seed}-shared-rbg-02", "antenna", shared_east_m + 8.0, shared_north_m + 88.0)

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


class DroneSimulator:
    """Own one drone's telemetry, observation store, object map, and peer transport."""

    def __init__(
        self,
        drone_id: int,
        name: str,
        peers: list[str],
        scenario_seed: int = DEFAULT_SCENARIO_SEED,
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
        self.routes = generate_patrol_routes(scenario_seed)
        self.truth_objects = generate_ground_truth_objects(scenario_seed)

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
        self.last_local_observation: Observation | None = None
        self.local_observations_generated = 0
        self.peer_observations_sent = 0
        self.peer_observations_received = 0
        self.unique_observations_accepted = 0
        self.duplicate_observations_ignored = 0
        self.validation_rejects = 0
        self.most_recent_received_observation_id: str | None = None
        self.most_recent_received_source: str | None = None
        self.most_recent_sent_observation_id: str | None = None
        self.peer_connection_state: dict[str, dict[str, object]] = {
            peer: {
                "peer": peer,
                "state": "not_contacted",
                "last_error": None,
                "last_success_time": None,
            }
            for peer in peers
        }

    def _advance_locked(self) -> None:
        now = time.monotonic()
        if not self.paused:
            self.simulation_time += now - self.last_wall_time
            self._update_flight_trail_locked()
        self.last_wall_time = now

    def _pose_at_locked(self) -> tuple[float, float, float]:
        return patrol_pose_on_route(self.routes[self.drone_id], self.simulation_time)

    def _pose_at_time_locked(self, simulation_time: float) -> tuple[float, float, float]:
        return patrol_pose_on_route(self.routes[self.drone_id], simulation_time)

    def _update_flight_trail_locked(self) -> None:
        while self.next_trail_sample_time <= self.simulation_time:
            latitude, longitude, _heading = self._pose_at_time_locked(self.next_trail_sample_time)
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

    def _camera_footprint_locked(self) -> list[dict[str, float]]:
        latitude, longitude, heading = self._pose_at_locked()
        points = [{"latitude": latitude, "longitude": longitude}]

        for angle in (heading - CAMERA_HALF_ANGLE_DEG, heading + CAMERA_HALF_ANGLE_DEG):
            angle_rad = math.radians(angle)
            east_m = math.sin(angle_rad) * CAMERA_RANGE_M
            north_m = math.cos(angle_rad) * CAMERA_RANGE_M
            point_latitude, point_longitude = offset_lat_lon(latitude, longitude, east_m, north_m)
            points.append({"latitude": point_latitude, "longitude": point_longitude})

        return points

    def _is_in_camera_footprint_locked(self, latitude: float, longitude: float) -> bool:
        drone_latitude, drone_longitude, heading = self._pose_at_locked()
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
                scenario_name=SCENARIO_NAME,
                scenario_seed=self.scenario_seed,
                id=self.drone_id,
                name=self.name,
                state="online",
                latitude=latitude,
                longitude=longitude,
                altitude_m=altitude,
                speed_mps=speed,
                heading_deg=heading,
                battery_percent=battery,
                timestamp=self.simulation_time,
                paused=self.paused,
                sharing_enabled=self.sharing_enabled,
                camera_footprint=self._camera_footprint_locked(),
                flight_trail=list(self.flight_trail),
                diagnostics=self._diagnostics_locked(),
            )

    def truth_payload(self) -> dict[str, object]:
        with self.lock:
            self._advance_locked()
            return {
                "schema_version": SCHEMA_VERSION,
                "app_build": APPLICATION_BUILD,
                "scenario_name": SCENARIO_NAME,
                "scenario_seed": self.scenario_seed,
                "simulation_time": self.simulation_time,
                "object_count": len(self.truth_objects),
                "objects": list(self.truth_objects),
            }

    def objects_payload(self) -> dict[str, object]:
        with self.lock:
            self._advance_locked()
            objects = []

            for item in self.objects.values():
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
                "scenario_name": SCENARIO_NAME,
                "scenario_seed": self.scenario_seed,
                "drone_id": self.drone_id,
                "drone_name": self.name,
                "simulation_time": self.simulation_time,
                "sharing_enabled": self.sharing_enabled,
                "paused": self.paused,
                "objects": sorted(
                    objects,
                    key=lambda item: str(item["object_id"]),
                ),
                "unresolved_observation_count": len(self.unresolved_observations),
                "unresolved_observations": list(self.unresolved_observations.values())[-25:],
            }

    def control_payload(self) -> dict[str, object]:
        with self.lock:
            return {
                "schema_version": SCHEMA_VERSION,
                "app_build": APPLICATION_BUILD,
                "scenario_name": SCENARIO_NAME,
                "scenario_seed": self.scenario_seed,
                "drone_id": self.drone_id,
                "paused": self.paused,
                "sharing_enabled": self.sharing_enabled,
                "observation_count": len(self.observations),
                "object_count": len(self.objects),
                "unresolved_observation_count": len(self.unresolved_observations),
                "diagnostics": self._diagnostics_locked(),
            }

    def set_paused(self, paused: bool) -> None:
        with self.lock:
            self._advance_locked()
            self.paused = paused
            self.last_wall_time = time.monotonic()

    def set_sharing_enabled(self, sharing_enabled: bool) -> None:
        with self.lock:
            self.sharing_enabled = sharing_enabled

    def reset(self, scenario_seed: int | None = None) -> None:
        with self.lock:
            if scenario_seed is not None:
                self.scenario_seed = scenario_seed

            self.routes = generate_patrol_routes(self.scenario_seed)
            self.truth_objects = generate_ground_truth_objects(self.scenario_seed)
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
            self.last_local_observation = None
            self.local_observations_generated = 0
            self.peer_observations_sent = 0
            self.peer_observations_received = 0
            self.unique_observations_accepted = 0
            self.duplicate_observations_ignored = 0
            self.validation_rejects = 0
            self.most_recent_received_observation_id = None
            self.most_recent_received_source = None
            self.most_recent_sent_observation_id = None

            for peer_state in self.peer_connection_state.values():
                peer_state["state"] = "not_contacted"
                peer_state["last_error"] = None
                peer_state["last_success_time"] = None

    def capture_observations(self) -> list[Observation]:
        new_observations: list[Observation] = []

        with self.lock:
            self._advance_locked()

            if self.paused:
                return []

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
                observation = Observation(
                    id=observation_id,
                    scenario_run_id=f"{self.scenario_seed}",
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
                    object_id=f"{sensor_model.sensor_id}:{source_track_id}",
                )

                if self._apply_observation_locked(observation):
                    self.local_observations_generated += 1
                    self.last_local_observation = observation
                    new_observations.append(observation)

        return new_observations

    def receive_observation(self, payload: dict[str, object]) -> bool:
        with self.lock:
            self.peer_observations_received += 1

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

            try:
                with urllib.request.urlopen(request, timeout=0.4):
                    pass
            except (OSError, urllib.error.URLError) as error:
                with self.lock:
                    self.peer_connection_state[peer]["state"] = "error"
                    self.peer_connection_state[peer]["last_error"] = str(error)
                continue

            with self.lock:
                self.peer_observations_sent += 1
                self.most_recent_sent_observation_id = observation.id
                self.peer_connection_state[peer]["state"] = "ok"
                self.peer_connection_state[peer]["last_error"] = None
                self.peer_connection_state[peer]["last_success_time"] = self.simulation_time

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
            measurement_type=str(payload.get("measurement_type", "point")),
            object_id=str(payload.get("object_id", "")),
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
        }

    def _observation_id(self, source_track_id: str, observation_time_bucket: int) -> str:
        raw_id = f"{self.drone_id}:{SENSOR_MODELS[self.drone_id].sensor_id}:{source_track_id}:{observation_time_bucket}"
        return hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:24]

    def _track_key(self, observation: Observation) -> str:
        return f"{observation.source_drone_id}:{observation.sensor_id}:{observation.source_track_id}"

    def _validate_observation_locked(self, observation: Observation) -> bool:
        if observation.measurement_type != "point":
            return False

        if observation.scenario_run_id != str(self.scenario_seed):
            return False

        if not all(
            math.isfinite(value)
            for value in (observation.east_m, observation.north_m, observation.latitude, observation.longitude)
        ):
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
        if observation.id in self.observations:
            self.duplicate_observations_ignored += 1
            return False

        if not self._validate_observation_locked(observation):
            self.validation_rejects += 1
            return False

        track_key = self._track_key(observation)
        associated_object_id = self._select_association_locked(observation)
        create_new_object = associated_object_id is None and observation.id not in self.unresolved_observations

        if associated_object_id is None and not create_new_object:
            self.observations[observation.id] = observation
            self.unique_observations_accepted += 1
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
            }

        object_map = self.objects[associated_object_id]
        contributions = object_map["contributions"]  # type: ignore[assignment]
        prior_contribution = contributions.get(track_key)  # type: ignore[union-attr]

        if prior_contribution is not None and observation.timestamp <= float(prior_contribution["timestamp"]):
            self.duplicate_observations_ignored += 1
            self.observations[observation.id] = observation
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
        self._refresh_fused_object_locked(object_map)
        return True


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

        self.send_error(404)

    def do_POST(self) -> None:
        try:
            payload = self._read_json_payload()
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON")
            return

        if self.path == "/observations":
            accepted = self.simulator.receive_observation(payload)
            self._send_json({"accepted": accepted})
            return

        if self.path == "/control":
            if "scenario_seed" in payload:
                self.simulator.reset(int(payload["scenario_seed"]))
            elif bool(payload.get("new_random_scenario", False)):
                self.simulator.reset(time.time_ns() % 1_000_000_000)
            elif bool(payload.get("reset", False)) or bool(payload.get("restart_scenario", False)):
                self.simulator.reset()

            if "paused" in payload:
                self.simulator.set_paused(bool(payload["paused"]))

            if "sharing_enabled" in payload:
                self.simulator.set_sharing_enabled(bool(payload["sharing_enabled"]))

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

    def _read_json_payload(self) -> dict[str, object]:
        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length)
        return json.loads(body.decode("utf-8") or "{}")

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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    DroneRequestHandler.simulator = DroneSimulator(args.drone_id, args.name, args.peers, args.seed)

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

        observation_thread = threading.Thread(target=observe_and_share, name="drone-observer", daemon=True)
        observation_thread.start()

        print(
            f"{args.name} running at http://{args.host}:{args.port}/status "
            f"pid={os.getpid()} build={APPLICATION_BUILD} scenario={SCENARIO_NAME!r} seed={args.seed}",
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
