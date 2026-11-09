#!/usr/bin/env python3
"""Tests for deterministic moving-search behavior in the drone simulator."""

from __future__ import annotations

import math
import time
import unittest
from dataclasses import asdict

from simulated_drone import (
    CONTRIBUTION_MAX_AGE_SECONDS,
    DEFAULT_SCENARIO_SEED,
    MEASUREMENT_FRAME,
    MAP_REFERENCE_LATITUDE,
    MAP_REFERENCE_LONGITUDE,
    SCENARIO_OBJECT_COUNT,
    SENSOR_MODELS,
    STALE_AFTER_SECONDS,
    TRAIL_RETENTION_SECONDS,
    DroneSimulator,
    Observation,
    covariance_from_sensor_model,
    covariance_intersection_two,
    is_valid_covariance,
    lat_lon_to_local_m,
    local_m_to_lat_lon,
    matrix_det_2x2,
    noisy_measurement_for_truth,
    source_local_track_id,
)


def set_simulation_time(simulator: DroneSimulator, simulation_time: float) -> None:
    simulator.simulation_time = simulation_time
    simulator.last_wall_time = time.monotonic()


def synthetic_observation(
    *,
    source_drone_id: int,
    source_drone_name: str,
    sensor_id: str,
    source_track_id: str,
    timestamp: float,
    east_m: float,
    north_m: float,
    label: str = "vehicle",
    covariance: list[list[float]] | None = None,
    observation_id: str | None = None,
) -> Observation:
    latitude, longitude = local_m_to_lat_lon(east_m, north_m)
    color = {1: "red", 2: "blue", 3: "green"}.get(source_drone_id, "gray")
    covariance = covariance if covariance is not None else [[25.0, 0.0], [0.0, 25.0]]
    observation_id = observation_id or f"{source_drone_id}-{sensor_id}-{source_track_id}-{timestamp:.1f}"

    return Observation(
        id=observation_id,
        scenario_run_id=str(DEFAULT_SCENARIO_SEED),
        source_drone_id=source_drone_id,
        source_drone_name=source_drone_name,
        source_color=color,
        sensor_id=sensor_id,
        source_track_id=source_track_id,
        timestamp=timestamp,
        measurement_frame=MEASUREMENT_FRAME,
        east_m=east_m,
        north_m=north_m,
        latitude=latitude,
        longitude=longitude,
        label=label,
        classification_score=0.88,
        covariance=covariance,
    )


def apply_observation(simulator: DroneSimulator, observation: Observation, simulation_time: float | None = None) -> bool:
    if simulation_time is not None:
        set_simulation_time(simulator, simulation_time)

    return simulator.receive_observation(asdict(observation))


def find_observation_for_truth(
    simulator: DroneSimulator,
    truth_object: dict[str, object],
    times: list[float],
) -> Observation:
    expected_track_id = source_local_track_id(
        simulator.scenario_seed,
        simulator.drone_id,
        str(truth_object["id"]),
    )

    for simulation_time in times:
        set_simulation_time(simulator, simulation_time)

        for observation in simulator.capture_observations():
            if observation.source_track_id == expected_track_id:
                return observation

    raise AssertionError(f"No observation found for {truth_object['id']} by drone {simulator.drone_id}")


class MovingSearchTest(unittest.TestCase):
    def test_red_search_captures_visible_truth_object_and_populates_map(self) -> None:
        simulator = DroneSimulator(1, "Red", [])

        set_simulation_time(simulator, 1.0)
        observations = simulator.capture_observations()

        self.assertGreaterEqual(len(observations), 1)
        observation = observations[0]
        self.assertTrue(observation.id)
        self.assertTrue(observation.source_track_id)
        self.assertIsInstance(observation.label, str)
        self.assertEqual(1, observation.source_drone_id)
        self.assertEqual("Red", observation.source_drone_name)

        payload = simulator.objects_payload()
        objects = payload["objects"]  # type: ignore[index]
        self.assertGreaterEqual(len(objects), 1)

        object_map = objects[0]
        self.assertTrue(str(object_map["object_id"]).startswith("1-map-"))
        self.assertIsInstance(object_map["label"], str)
        self.assertTrue(object_map["own_sensor"])
        self.assertFalse(object_map["teammate_sourced"])
        self.assertEqual([1], object_map["sources"])
        self.assertEqual(["Red"], object_map["source_names"])
        self.assertEqual([observation.id], object_map["observation_ids"])

    def test_seeded_truth_scenario_contains_expected_object_count(self) -> None:
        simulator = DroneSimulator(1, "Red", [])

        payload = simulator.truth_payload()

        self.assertEqual(SCENARIO_OBJECT_COUNT, payload["object_count"])
        self.assertEqual(SCENARIO_OBJECT_COUNT, len(payload["objects"]))  # type: ignore[arg-type]
        self.assertEqual(simulator.scenario_seed, payload["scenario_seed"])

    def test_restart_reproduces_seed_and_new_seed_changes_objects(self) -> None:
        simulator = DroneSimulator(1, "Red", [], scenario_seed=12345)
        first_objects = simulator.truth_payload()["objects"]

        simulator.reset()
        restarted_objects = simulator.truth_payload()["objects"]
        simulator.reset(67890)
        new_objects = simulator.truth_payload()["objects"]

        self.assertEqual(first_objects, restarted_objects)
        self.assertNotEqual(first_objects, new_objects)

    def test_red_and_blue_detect_common_truth_object_within_30_seconds(self) -> None:
        red = DroneSimulator(1, "Red", [], scenario_seed=DEFAULT_SCENARIO_SEED)
        blue = DroneSimulator(2, "Blue", [], scenario_seed=DEFAULT_SCENARIO_SEED)
        truth = red.truth_payload()["objects"]  # type: ignore[index]
        shared_truth = truth[0]  # type: ignore[index]
        red_track_id = source_local_track_id(DEFAULT_SCENARIO_SEED, 1, str(shared_truth["id"]))
        blue_track_id = source_local_track_id(DEFAULT_SCENARIO_SEED, 2, str(shared_truth["id"]))
        red_detection_times: list[float] = []
        blue_detection_times: list[float] = []

        for simulation_time in [step * 0.5 for step in range(61)]:
            red.simulation_time = simulation_time
            blue.simulation_time = simulation_time

            for observation in red.capture_observations():
                if observation.source_track_id == red_track_id:
                    red_detection_times.append(observation.timestamp)

            for observation in blue.capture_observations():
                if observation.source_track_id == blue_track_id:
                    blue_detection_times.append(observation.timestamp)

        common_detection_times = set(red_detection_times) & set(blue_detection_times)
        self.assertTrue(red_detection_times)
        self.assertTrue(blue_detection_times)
        self.assertTrue(common_detection_times)
        self.assertLessEqual(min(common_detection_times), 30.0)
        self.assertGreaterEqual(len(common_detection_times), 2)

    def test_one_remaining_observer_keeps_shared_object_fresh(self) -> None:
        red = DroneSimulator(1, "Red", [], scenario_seed=DEFAULT_SCENARIO_SEED)
        blue = DroneSimulator(2, "Blue", [], scenario_seed=DEFAULT_SCENARIO_SEED)
        map_node = DroneSimulator(3, "Green", [], scenario_seed=DEFAULT_SCENARIO_SEED)
        shared_truth = red.truth_payload()["objects"][0]  # type: ignore[index]
        sample_times = [step * 0.5 for step in range(61)]

        red_observation = find_observation_for_truth(red, shared_truth, sample_times)
        blue_observation = find_observation_for_truth(blue, shared_truth, sample_times)
        self.assertTrue(apply_observation(map_node, red_observation, red_observation.timestamp))
        self.assertTrue(apply_observation(map_node, blue_observation, blue_observation.timestamp))

        shared_object = next(
            item
            for item in map_node.objects_payload()["objects"]  # type: ignore[index]
            if {1, 2}.issubset({int(source) for source in item["historical_sources"]})
        )
        self.assertFalse(shared_object["stale"])
        self.assertEqual([1, 2], shared_object["sources"])

        later_blue_observations = []
        for simulation_time in sample_times:
            if simulation_time <= blue_observation.timestamp:
                continue

            set_simulation_time(blue, simulation_time)
            later_blue_observations.extend(
                observation
                for observation in blue.capture_observations()
                if observation.source_track_id == blue_observation.source_track_id
            )

        self.assertTrue(later_blue_observations)
        continuing_blue_observation = later_blue_observations[-1]
        self.assertTrue(
            apply_observation(
                map_node,
                continuing_blue_observation,
                continuing_blue_observation.timestamp,
            ),
        )

        fresh_payload = map_node.objects_payload()
        fresh_object = next(
            item
            for item in fresh_payload["objects"]  # type: ignore[index]
            if any(
                measurement["source_track_id"] == continuing_blue_observation.source_track_id
                for measurement in item["measurements"]
            )
        )
        self.assertFalse(fresh_object["stale"])
        self.assertIn(2, fresh_object["sources"])
        self.assertAlmostEqual(
            float(fresh_object["last_seen"]),
            continuing_blue_observation.timestamp,
            delta=0.001,
        )

        set_simulation_time(
            map_node,
            continuing_blue_observation.timestamp + STALE_AFTER_SECONDS + 0.5,
        )
        stale_object = next(
            item
            for item in map_node.objects_payload()["objects"]  # type: ignore[index]
            if str(item["object_id"]) == str(fresh_object["object_id"])
        )
        self.assertTrue(stale_object["stale"])

    def test_independent_sensor_noise_is_reproducible(self) -> None:
        truth = DroneSimulator(1, "Red", [], scenario_seed=DEFAULT_SCENARIO_SEED).truth_payload()["objects"][0]  # type: ignore[index]
        red_measurement = noisy_measurement_for_truth(
            DEFAULT_SCENARIO_SEED,
            1,
            str(truth["id"]),
            12,
            float(truth["latitude"]),
            float(truth["longitude"]),
        )
        repeated_red_measurement = noisy_measurement_for_truth(
            DEFAULT_SCENARIO_SEED,
            1,
            str(truth["id"]),
            12,
            float(truth["latitude"]),
            float(truth["longitude"]),
        )
        blue_measurement = noisy_measurement_for_truth(
            DEFAULT_SCENARIO_SEED,
            2,
            str(truth["id"]),
            12,
            float(truth["latitude"]),
            float(truth["longitude"]),
        )

        self.assertEqual(red_measurement, repeated_red_measurement)
        self.assertNotEqual(red_measurement[:2], blue_measurement[:2])
        self.assertTrue(is_valid_covariance(red_measurement[2]))
        self.assertTrue(is_valid_covariance(blue_measurement[2]))

    def test_empirical_measurement_noise_matches_rotated_anisotropic_covariance(self) -> None:
        truth = DroneSimulator(1, "Red", [], scenario_seed=DEFAULT_SCENARIO_SEED).truth_payload()["objects"][0]  # type: ignore[index]
        expected_covariance = covariance_from_sensor_model(SENSOR_MODELS[1])
        true_east_m, true_north_m = lat_lon_to_local_m(
            MAP_REFERENCE_LATITUDE,
            MAP_REFERENCE_LONGITUDE,
            float(truth["latitude"]),
            float(truth["longitude"]),
        )
        sample_count = 5000
        errors: list[tuple[float, float]] = []

        for timestamp in range(sample_count):
            east_m, north_m, covariance = noisy_measurement_for_truth(
                DEFAULT_SCENARIO_SEED,
                1,
                str(truth["id"]),
                timestamp,
                float(truth["latitude"]),
                float(truth["longitude"]),
            )
            self.assertEqual(expected_covariance, covariance)
            errors.append((east_m - true_east_m, north_m - true_north_m))

        mean_east = sum(error[0] for error in errors) / sample_count
        mean_north = sum(error[1] for error in errors) / sample_count
        empirical_ee = sum((error[0] - mean_east) ** 2 for error in errors) / (sample_count - 1)
        empirical_nn = sum((error[1] - mean_north) ** 2 for error in errors) / (sample_count - 1)
        empirical_en = sum((error[0] - mean_east) * (error[1] - mean_north) for error in errors) / (sample_count - 1)

        self.assertLess(abs(mean_east), 0.45)
        self.assertLess(abs(mean_north), 0.45)
        self.assertAlmostEqual(empirical_ee, expected_covariance[0][0], delta=expected_covariance[0][0] * 0.08)
        self.assertAlmostEqual(empirical_nn, expected_covariance[1][1], delta=expected_covariance[1][1] * 0.08)
        self.assertAlmostEqual(empirical_en, expected_covariance[0][1], delta=max(1.5, abs(expected_covariance[0][1]) * 0.16))
        self.assertNotEqual(0.0, expected_covariance[0][1])

    def test_covariance_intersection_covariance_is_spd(self) -> None:
        _mean, covariance, _objective = covariance_intersection_two(
            (0.0, 0.0),
            [[64.0, 8.0], [8.0, 36.0]],
            (5.0, -2.0),
            [[110.0, -12.0], [-12.0, 49.0]],
        )

        self.assertAlmostEqual(covariance[0][1], covariance[1][0], delta=1e-9)
        self.assertTrue(is_valid_covariance(covariance))

    def test_covariance_intersection_tie_break_favors_second_estimate(self) -> None:
        covariance = [[36.0, 0.0], [0.0, 36.0]]

        mean, fused_covariance, _objective = covariance_intersection_two(
            (0.0, 0.0),
            covariance,
            (10.0, -4.0),
            covariance,
        )
        translated_mean, translated_covariance, _translated_objective = covariance_intersection_two(
            (100.0, 50.0),
            covariance,
            (110.0, 46.0),
            covariance,
        )

        self.assertAlmostEqual(mean[0], 10.0, delta=1e-9)
        self.assertAlmostEqual(mean[1], -4.0, delta=1e-9)
        self.assertEqual(covariance, fused_covariance)
        self.assertAlmostEqual(translated_mean[0] - 100.0, mean[0], delta=1e-9)
        self.assertAlmostEqual(translated_mean[1] - 50.0, mean[1], delta=1e-9)
        self.assertEqual(fused_covariance, translated_covariance)

    def test_nearby_noisy_reports_associate_to_one_isolated_object(self) -> None:
        simulator = DroneSimulator(1, "Red", [], scenario_seed=DEFAULT_SCENARIO_SEED)
        first = synthetic_observation(
            source_drone_id=1,
            source_drone_name="Red",
            sensor_id="eo-red",
            source_track_id="red-track",
            timestamp=1.0,
            east_m=0.0,
            north_m=0.0,
        )
        second = synthetic_observation(
            source_drone_id=2,
            source_drone_name="Blue",
            sensor_id="eo-blue",
            source_track_id="blue-track",
            timestamp=1.0,
            east_m=4.0,
            north_m=3.0,
        )

        self.assertTrue(apply_observation(simulator, first, 1.0))
        self.assertTrue(apply_observation(simulator, second, 1.0))

        payload = simulator.objects_payload()
        self.assertEqual(1, len(payload["objects"]))  # type: ignore[arg-type]
        object_map = payload["objects"][0]  # type: ignore[index]
        self.assertEqual([1, 2], object_map["historical_sources"])

    def test_clearly_separated_objects_remain_separate(self) -> None:
        simulator = DroneSimulator(1, "Red", [], scenario_seed=DEFAULT_SCENARIO_SEED)

        self.assertTrue(apply_observation(simulator, synthetic_observation(
            source_drone_id=1,
            source_drone_name="Red",
            sensor_id="eo-red",
            source_track_id="red-a",
            timestamp=1.0,
            east_m=0.0,
            north_m=0.0,
        ), 1.0))
        self.assertTrue(apply_observation(simulator, synthetic_observation(
            source_drone_id=2,
            source_drone_name="Blue",
            sensor_id="eo-blue",
            source_track_id="blue-b",
            timestamp=1.0,
            east_m=200.0,
            north_m=0.0,
        ), 1.0))

        self.assertEqual(2, len(simulator.objects_payload()["objects"]))  # type: ignore[arg-type]

    def test_ambiguous_candidate_remains_unresolved(self) -> None:
        simulator = DroneSimulator(1, "Red", [], scenario_seed=DEFAULT_SCENARIO_SEED)
        covariance = [[4.0, 0.0], [0.0, 4.0]]

        self.assertTrue(apply_observation(simulator, synthetic_observation(
            source_drone_id=1,
            source_drone_name="Red",
            sensor_id="eo-red",
            source_track_id="left",
            timestamp=1.0,
            east_m=-5.0,
            north_m=0.0,
            covariance=covariance,
        ), 1.0))
        self.assertTrue(apply_observation(simulator, synthetic_observation(
            source_drone_id=2,
            source_drone_name="Blue",
            sensor_id="eo-blue",
            source_track_id="right",
            timestamp=1.0,
            east_m=5.0,
            north_m=0.0,
            covariance=covariance,
        ), 1.0))
        self.assertFalse(apply_observation(simulator, synthetic_observation(
            source_drone_id=3,
            source_drone_name="Green",
            sensor_id="eo-green",
            source_track_id="middle",
            timestamp=2.0,
            east_m=0.0,
            north_m=0.0,
            covariance=covariance,
        ), 2.0))

        payload = simulator.objects_payload()
        self.assertEqual(2, len(payload["objects"]))  # type: ignore[arg-type]
        self.assertEqual(1, payload["unresolved_observation_count"])

    def test_source_track_continuity_rejects_implausible_jump(self) -> None:
        simulator = DroneSimulator(1, "Red", [], scenario_seed=DEFAULT_SCENARIO_SEED)

        self.assertTrue(apply_observation(simulator, synthetic_observation(
            source_drone_id=1,
            source_drone_name="Red",
            sensor_id="eo-red",
            source_track_id="same-track",
            timestamp=1.0,
            east_m=0.0,
            north_m=0.0,
        ), 1.0))
        self.assertFalse(apply_observation(simulator, synthetic_observation(
            source_drone_id=1,
            source_drone_name="Red",
            sensor_id="eo-red",
            source_track_id="same-track",
            timestamp=2.0,
            east_m=500.0,
            north_m=0.0,
        ), 2.0))

        payload = simulator.objects_payload()
        self.assertEqual(1, len(payload["objects"]))  # type: ignore[arg-type]
        self.assertEqual(1, payload["unresolved_observation_count"])

    def test_repeated_same_track_updates_replace_without_covariance_collapse(self) -> None:
        simulator = DroneSimulator(1, "Red", [], scenario_seed=DEFAULT_SCENARIO_SEED)
        covariance = [[25.0, 0.0], [0.0, 25.0]]

        self.assertTrue(apply_observation(simulator, synthetic_observation(
            source_drone_id=1,
            source_drone_name="Red",
            sensor_id="eo-red",
            source_track_id="same-track",
            timestamp=1.0,
            east_m=0.0,
            north_m=0.0,
            covariance=covariance,
        ), 1.0))
        self.assertTrue(apply_observation(simulator, synthetic_observation(
            source_drone_id=1,
            source_drone_name="Red",
            sensor_id="eo-red",
            source_track_id="same-track",
            timestamp=2.0,
            east_m=2.0,
            north_m=0.0,
            covariance=covariance,
        ), 2.0))

        object_map = simulator.objects_payload()["objects"][0]  # type: ignore[index]
        self.assertEqual(1, object_map["active_contribution_count"])
        self.assertEqual(covariance, object_map["covariance"])
        self.assertEqual(2, object_map["observation_count"])

    def test_duplicate_delivery_changes_neither_estimate_nor_evidence(self) -> None:
        simulator = DroneSimulator(1, "Red", [], scenario_seed=DEFAULT_SCENARIO_SEED)
        observation = synthetic_observation(
            source_drone_id=2,
            source_drone_name="Blue",
            sensor_id="eo-blue",
            source_track_id="blue-track",
            timestamp=1.0,
            east_m=0.0,
            north_m=0.0,
            observation_id="duplicate-observation",
        )

        self.assertTrue(apply_observation(simulator, observation, 1.0))
        before = simulator.objects_payload()["objects"][0]  # type: ignore[index]
        self.assertFalse(apply_observation(simulator, observation, 1.0))
        after = simulator.objects_payload()["objects"][0]  # type: ignore[index]

        self.assertEqual(before["east_m"], after["east_m"])
        self.assertEqual(before["north_m"], after["north_m"])
        self.assertEqual(before["covariance"], after["covariance"])
        self.assertEqual(before["observation_ids"], after["observation_ids"])

    def test_out_of_order_report_does_not_replace_newer_contribution(self) -> None:
        simulator = DroneSimulator(1, "Red", [], scenario_seed=DEFAULT_SCENARIO_SEED)

        newer = synthetic_observation(
            source_drone_id=2,
            source_drone_name="Blue",
            sensor_id="eo-blue",
            source_track_id="blue-track",
            timestamp=10.0,
            east_m=10.0,
            north_m=0.0,
            observation_id="newer",
        )
        older = synthetic_observation(
            source_drone_id=2,
            source_drone_name="Blue",
            sensor_id="eo-blue",
            source_track_id="blue-track",
            timestamp=9.0,
            east_m=100.0,
            north_m=0.0,
            observation_id="older",
        )

        self.assertTrue(apply_observation(simulator, newer, 10.0))
        before = simulator.objects_payload()["objects"][0]  # type: ignore[index]
        self.assertFalse(apply_observation(simulator, older, 10.0))
        after = simulator.objects_payload()["objects"][0]  # type: ignore[index]

        self.assertEqual(before["east_m"], after["east_m"])
        self.assertEqual(before["last_seen"], after["last_seen"])
        self.assertEqual(before["observation_ids"], after["observation_ids"])

    def test_expired_contributions_are_excluded_from_current_estimate(self) -> None:
        simulator = DroneSimulator(1, "Red", [], scenario_seed=DEFAULT_SCENARIO_SEED)
        blue_covariance = [[25.0, 0.0], [0.0, 25.0]]

        self.assertTrue(apply_observation(simulator, synthetic_observation(
            source_drone_id=1,
            source_drone_name="Red",
            sensor_id="eo-red",
            source_track_id="old-track",
            timestamp=0.0,
            east_m=0.0,
            north_m=0.0,
        ), 0.0))
        self.assertTrue(apply_observation(simulator, synthetic_observation(
            source_drone_id=2,
            source_drone_name="Blue",
            sensor_id="eo-blue",
            source_track_id="fresh-track",
            timestamp=10.0,
            east_m=3.0,
            north_m=0.0,
            covariance=blue_covariance,
        ), 10.0))

        set_simulation_time(simulator, 10.1)
        active_object_map = simulator.objects_payload()["objects"][0]  # type: ignore[index]
        self.assertEqual([2], active_object_map["sources"])
        self.assertEqual(1, active_object_map["active_contribution_count"])
        self.assertEqual([1, 2], active_object_map["historical_sources"])
        self.assertEqual(["Blue", "Red"], active_object_map["historical_source_names"])
        self.assertEqual(3.0, active_object_map["east_m"])
        self.assertEqual(0.0, active_object_map["north_m"])
        self.assertEqual(blue_covariance, active_object_map["covariance"])
        self.assertFalse(active_object_map["stale"])

        set_simulation_time(simulator, 18.1)
        expired_object_map = simulator.objects_payload()["objects"][0]  # type: ignore[index]
        self.assertEqual([], expired_object_map["sources"])
        self.assertEqual(0, expired_object_map["active_contribution_count"])
        self.assertEqual([1, 2], expired_object_map["historical_sources"])
        self.assertEqual(["Blue", "Red"], expired_object_map["historical_source_names"])
        self.assertEqual(3.0, expired_object_map["east_m"])
        self.assertEqual(0.0, expired_object_map["north_m"])
        self.assertEqual(blue_covariance, expired_object_map["covariance"])
        self.assertTrue(expired_object_map["stale"])

    def test_fused_covariance_is_symmetric_positive_definite(self) -> None:
        simulator = DroneSimulator(1, "Red", [], scenario_seed=DEFAULT_SCENARIO_SEED)

        self.assertTrue(apply_observation(simulator, synthetic_observation(
            source_drone_id=1,
            source_drone_name="Red",
            sensor_id="eo-red",
            source_track_id="red-track",
            timestamp=1.0,
            east_m=0.0,
            north_m=0.0,
            covariance=[[64.0, 8.0], [8.0, 36.0]],
        ), 1.0))
        self.assertTrue(apply_observation(simulator, synthetic_observation(
            source_drone_id=2,
            source_drone_name="Blue",
            sensor_id="eo-blue",
            source_track_id="blue-track",
            timestamp=1.0,
            east_m=4.0,
            north_m=3.0,
            covariance=[[110.0, -12.0], [-12.0, 49.0]],
        ), 1.0))

        covariance = simulator.objects_payload()["objects"][0]["covariance"]  # type: ignore[index]
        self.assertAlmostEqual(covariance[0][1], covariance[1][0], delta=1e-9)
        self.assertTrue(is_valid_covariance(covariance))
        self.assertGreater(matrix_det_2x2(covariance), 0.0)

    def test_optional_bias_is_disabled_by_default_and_not_validated_as_correlated_error(self) -> None:
        for sensor_model in SENSOR_MODELS.values():
            self.assertEqual(0.0, sensor_model.bias_east_m)
            self.assertEqual(0.0, sensor_model.bias_north_m)

    def test_flight_trail_obeys_cutoff_and_freezes_while_paused(self) -> None:
        simulator = DroneSimulator(1, "Red", [])

        for simulation_time in [step * 0.5 for step in range(81)]:
            set_simulation_time(simulator, simulation_time)
            simulator._update_flight_trail_locked()

        status = simulator.status()
        self.assertLessEqual(len(status.flight_trail), int(TRAIL_RETENTION_SECONDS / 0.5) + 2)
        self.assertGreater(min(point["timestamp"] for point in status.flight_trail), 10.0)

        simulator.set_paused(True)
        frozen_trail = list(simulator.status().flight_trail)
        time.sleep(0.05)
        self.assertEqual(frozen_trail, simulator.status().flight_trail)

    def test_camera_footprint_moves_when_simulation_runs(self) -> None:
        simulator = DroneSimulator(1, "Red", [])

        first_status = simulator.status()
        time.sleep(0.05)
        second_status = simulator.status()

        self.assertGreater(second_status.timestamp, first_status.timestamp)
        self.assertNotEqual(first_status.latitude, second_status.latitude)
        self.assertNotEqual(first_status.longitude, second_status.longitude)
        self.assertNotEqual(first_status.camera_footprint, second_status.camera_footprint)
        self.assertEqual(3, len(second_status.camera_footprint))

    def test_paused_search_does_not_advance_or_capture(self) -> None:
        simulator = DroneSimulator(1, "Red", [])
        simulator.set_paused(True)

        before_status = simulator.status()
        time.sleep(0.05)
        observations = simulator.capture_observations()
        after_status = simulator.status()

        self.assertEqual([], observations)
        self.assertAlmostEqual(before_status.timestamp, after_status.timestamp, delta=0.000001)
        self.assertEqual([], simulator.objects_payload()["objects"])


if __name__ == "__main__":
    unittest.main()
