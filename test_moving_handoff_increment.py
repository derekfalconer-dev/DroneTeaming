import math
import time
import unittest
from dataclasses import asdict
from unittest.mock import patch

from moving_tracking import MOVING_PERSON_SPEED_MPS, moving_person_position

from simulated_drone import (
    DEFAULT_SCENARIO_ID,
    DEFAULT_SCENARIO_SEED,
    MOVING_HANDOFF_SCENARIO_ID,
    MOVING_TRACK_LOST_SECONDS,
    MOVING_TRACK_REPORT_TYPE,
    DroneSimulator,
)


def advance_to(simulator: DroneSimulator, simulation_time: float) -> None:
    with simulator.lock:
        if simulation_time < simulator.simulation_time:
            raise AssertionError(
                f"Refusing backward simulation time travel from {simulator.simulation_time:.3f} to {simulation_time:.3f}",
            )

        simulator.simulation_time = simulation_time
        simulator.last_wall_time = time.monotonic()


def visible_moving_report_times(simulator: DroneSimulator, start: int = 0, stop: int = 100) -> list[float]:
    with simulator.lock:
        probe = DroneSimulator(
            simulator.drone_id,
            simulator.name,
            [],
            simulator.scenario_seed,
            simulator.requested_priority_count,
            simulator.selected_playbook,
            scenario_id=simulator.scenario_id,
            initial_observer_id=simulator.initial_observer_id,
            next_observer_id=simulator.next_observer_id,
            scenario_run_id=simulator.scenario_run_id,
        )

    visible_times: list[float] = []

    for timestamp in range(start, stop + 1):
        advance_to(probe, float(timestamp))
        probe._refresh_moving_truth_objects_locked()
        moving_truth = next(item for item in probe.truth_objects if bool(item.get("moving", False)))

        if probe._is_in_camera_footprint_locked(float(moving_truth["latitude"]), float(moving_truth["longitude"])):
            visible_times.append(float(timestamp))

    return visible_times


def commit_due_reports(simulator: DroneSimulator, simulation_time: float) -> None:
    advance_to(simulator, simulation_time)
    simulator.objects_payload()


def moving_objects_payload(simulator: DroneSimulator) -> list[dict[str, object]]:
    return [
        item
        for item in simulator.objects_payload()["objects"]
        if item.get("tracking_model") == "constant_velocity"
    ]


def assert_moving_track_accuracy(
    test_case: unittest.TestCase,
    track: dict[str, object],
    simulation_time: float,
    *,
    max_position_error_m: float,
    max_velocity_error_mps: float,
) -> None:
    truth_east_m, truth_north_m = moving_person_position(simulation_time)
    position_error_m = math.hypot(
        float(track["predicted_east_m"]) - truth_east_m,
        float(track["predicted_north_m"]) - truth_north_m,
    )
    velocity_error_mps = math.hypot(
        float(track["velocity_east_mps"]) - MOVING_PERSON_SPEED_MPS,
        float(track["velocity_north_mps"]),
    )

    test_case.assertLess(
        position_error_m,
        max_position_error_m,
        f"position_error_m={position_error_m:.3f} at t={simulation_time:.1f}; track={track!r}",
    )
    test_case.assertLess(
        velocity_error_mps,
        max_velocity_error_mps,
        f"velocity_error_mps={velocity_error_mps:.3f} at t={simulation_time:.1f}; track={track!r}",
    )


class MovingHandoffIncrementTest(unittest.TestCase):
    def test_moving_person_handoff_keeps_local_object_id_and_infers_velocity(self) -> None:
        simulator = DroneSimulator(
            1,
            "Red",
            [],
            DEFAULT_SCENARIO_SEED,
            scenario_id=MOVING_HANDOFF_SCENARIO_ID,
            initial_observer_id=1,
            next_observer_id=2,
            scenario_run_id="test-moving-run",
        )
        coverage_times = visible_moving_report_times(simulator)
        self.assertGreaterEqual(coverage_times, [10.0], f"unexpected coverage_times={coverage_times!r}")

        accepted_ids: list[str] = []
        for timestamp in coverage_times[:8]:
            advance_to(simulator, timestamp)
            accepted_ids.extend([
                report.id
                for report in simulator.capture_observations()
                if report.measurement_type == MOVING_TRACK_REPORT_TYPE
            ])

        commit_due_reports(simulator, coverage_times[7] + 2.0)
        moving_objects = moving_objects_payload(simulator)

        self.assertGreaterEqual(len(accepted_ids), 2, f"coverage_times={coverage_times!r}")
        self.assertEqual(1, len(moving_objects), f"moving_objects={moving_objects!r}")
        self.assertEqual("tracked", moving_objects[0]["track_status"])
        self.assertGreater(abs(float(moving_objects[0]["velocity_east_mps"])), 0.5)
        assert_moving_track_accuracy(
            self,
            moving_objects[0],
            coverage_times[7] + 2.0,
            max_position_error_m=90.0,
            max_velocity_error_mps=12.0,
        )

    def test_peer_report_from_next_observer_associates_with_existing_track(self) -> None:
        red = DroneSimulator(
            1,
            "Red",
            [],
            DEFAULT_SCENARIO_SEED,
            scenario_id=MOVING_HANDOFF_SCENARIO_ID,
            initial_observer_id=1,
            next_observer_id=2,
            scenario_run_id="test-peer-moving-run",
        )
        blue = DroneSimulator(
            2,
            "Blue",
            [],
            DEFAULT_SCENARIO_SEED,
            scenario_id=MOVING_HANDOFF_SCENARIO_ID,
            initial_observer_id=1,
            next_observer_id=2,
            scenario_run_id="test-peer-moving-run",
        )

        red_coverage_times = visible_moving_report_times(red)
        blue_coverage_times = visible_moving_report_times(blue)
        self.assertTrue(red_coverage_times, "Red never sees the moving person in production geometry.")
        self.assertTrue(blue_coverage_times, "Blue never sees the moving person in production geometry.")
        self.assertGreater(
            max(red_coverage_times),
            min(blue_coverage_times),
            f"Expected overlapping handoff coverage; red={red_coverage_times!r}; blue={blue_coverage_times!r}",
        )

        for timestamp in red_coverage_times[:10]:
            advance_to(red, timestamp)
            red.capture_observations()

        commit_due_reports(red, red_coverage_times[9] + 2.0)

        before = [
            item
            for item in red.objects_payload()["objects"]
            if item.get("tracking_model") == "constant_velocity"
        ][0]
        local_object_id = before["local_object_id"]

        peer_reports = []
        for timestamp in [item for item in blue_coverage_times if item > red_coverage_times[9]][:8]:
            advance_to(blue, timestamp)
            peer_reports.extend([
                report
                for report in blue.capture_observations()
                if report.measurement_type == MOVING_TRACK_REPORT_TYPE
            ])

        self.assertTrue(peer_reports)

        for report in peer_reports:
            red.receive_observation(asdict(report))

        advance_to(red, max(float(report.timestamp) for report in peer_reports) + 2.0)
        red.objects_payload()
        after = [
            item
            for item in red.objects_payload()["objects"]
            if item.get("tracking_model") == "constant_velocity"
        ][0]

        self.assertEqual(local_object_id, after["local_object_id"])
        self.assertIn("Blue", after["historical_source_names"])
        self.assertGreaterEqual(len(after["source_track_ids"]), 2)

    def test_restart_preserves_handoff_configuration_and_clears_previous_state(self) -> None:
        simulator = DroneSimulator(
            1,
            "Red",
            [],
            DEFAULT_SCENARIO_SEED,
            scenario_id=MOVING_HANDOFF_SCENARIO_ID,
            initial_observer_id=3,
            next_observer_id=1,
            scenario_run_id="preserve-before",
        )
        coverage_times = visible_moving_report_times(simulator)
        self.assertTrue(coverage_times, "Configured handoff scenario has no moving-person coverage.")
        advance_to(simulator, coverage_times[0])
        simulator.capture_observations()
        simulator.objects_payload()

        simulator.reset(scenario_run_id="preserve-after")

        self.assertEqual(MOVING_HANDOFF_SCENARIO_ID, simulator.scenario_id)
        self.assertEqual(DEFAULT_SCENARIO_SEED, simulator.scenario_seed)
        self.assertEqual(3, simulator.initial_observer_id)
        self.assertEqual(1, simulator.next_observer_id)
        self.assertEqual("preserve-after", simulator.scenario_run_id)
        self.assertEqual(0, simulator.priority_object_count)
        self.assertEqual(0.0, simulator.simulation_time)
        self.assertEqual([], simulator.flight_trail)
        self.assertEqual({}, simulator.observations)
        self.assertEqual({}, simulator.objects)
        self.assertEqual([], simulator.moving_report_buffer)

    @patch("simulated_drone.time.monotonic", return_value=1000.0)
    def test_handoff_replay_reproduces_motion_and_noise_with_new_run_id_and_rejects_old_reports(self, _mock_monotonic: object) -> None:
        simulator = DroneSimulator(
            1,
            "Red",
            [],
            DEFAULT_SCENARIO_SEED,
            scenario_id=MOVING_HANDOFF_SCENARIO_ID,
            initial_observer_id=1,
            next_observer_id=2,
            scenario_run_id="replay-run-a",
        )
        coverage_times = visible_moving_report_times(simulator)
        self.assertTrue(coverage_times, "Configured handoff scenario has no moving-person coverage.")
        advance_to(simulator, coverage_times[0])
        first_reports = [
            report
            for report in simulator.capture_observations()
            if report.measurement_type == MOVING_TRACK_REPORT_TYPE
        ]
        first_truth = simulator.truth_payload()["objects"]

        self.assertTrue(first_reports)

        old_report = first_reports[0]
        simulator.reset(
            DEFAULT_SCENARIO_SEED,
            None,
            None,
            MOVING_HANDOFF_SCENARIO_ID,
            1,
            2,
            "replay-run-b",
        )

        self.assertFalse(simulator.receive_observation(asdict(old_report)))
        self.assertEqual({}, simulator.objects)

        advance_to(simulator, coverage_times[0])
        replay_reports = [
            report
            for report in simulator.capture_observations()
            if report.measurement_type == MOVING_TRACK_REPORT_TYPE
        ]
        replay_truth = simulator.truth_payload()["objects"]

        self.assertTrue(replay_reports)
        self.assertEqual("replay-run-a", old_report.scenario_run_id)
        self.assertEqual("replay-run-b", replay_reports[0].scenario_run_id)
        self.assertNotEqual(old_report.scenario_run_id, replay_reports[0].scenario_run_id)
        self.assertEqual(old_report.id, replay_reports[0].id)
        self.assertAlmostEqual(old_report.east_m, replay_reports[0].east_m, places=9)
        self.assertAlmostEqual(old_report.north_m, replay_reports[0].north_m, places=9)
        self.assertEqual(first_truth, replay_truth)

    def test_all_directed_observer_pairs_are_selectable_for_handoff(self) -> None:
        pairs = {
            (initial_observer_id, next_observer_id)
            for initial_observer_id in (1, 2, 3)
            for next_observer_id in (1, 2, 3)
            if initial_observer_id != next_observer_id
        }

        for initial_observer_id, next_observer_id in pairs:
            simulator = DroneSimulator(
                1,
                "Red",
                [],
                DEFAULT_SCENARIO_SEED,
                scenario_id=DEFAULT_SCENARIO_ID,
            )
            simulator.reset(
                DEFAULT_SCENARIO_SEED,
                0,
                None,
                MOVING_HANDOFF_SCENARIO_ID,
                initial_observer_id,
                next_observer_id,
                f"pair-{initial_observer_id}-{next_observer_id}",
            )

            self.assertEqual(MOVING_HANDOFF_SCENARIO_ID, simulator.scenario_id)
            self.assertEqual(initial_observer_id, simulator.initial_observer_id)
            self.assertEqual(next_observer_id, simulator.next_observer_id)
            self.assertEqual(0, simulator.priority_object_count)

    def test_uncertainty_grows_during_coasting_and_prediction_error_is_bounded(self) -> None:
        simulator = DroneSimulator(
            1,
            "Red",
            [],
            DEFAULT_SCENARIO_SEED,
            scenario_id=MOVING_HANDOFF_SCENARIO_ID,
            initial_observer_id=1,
            next_observer_id=2,
            scenario_run_id="test-coasting-growth-run",
        )

        coverage_times = visible_moving_report_times(simulator)
        self.assertGreaterEqual(len(coverage_times), 6, f"coverage_times={coverage_times!r}")

        for timestamp in coverage_times[:6]:
            advance_to(simulator, timestamp)
            simulator.capture_observations()

        commit_due_reports(simulator, coverage_times[5] + 2.0)
        tracked = moving_objects_payload(simulator)[0]
        initial_covariance_trace = tracked["state_covariance"][0][0] + tracked["state_covariance"][1][1]
        last_measurement_at = float(tracked["last_measurement_at"])

        coasting_time = last_measurement_at + 5.0
        advance_to(simulator, coasting_time)
        coasting = [
            item
            for item in simulator.objects_payload()["objects"]
            if item.get("tracking_model") == "constant_velocity"
        ][0]
        coasting_covariance_trace = coasting["state_covariance"][0][0] + coasting["state_covariance"][1][1]
        truth_east_m, truth_north_m = moving_person_position(coasting_time)
        position_error_m = math.hypot(float(coasting["predicted_east_m"]) - truth_east_m, float(coasting["predicted_north_m"]) - truth_north_m)
        velocity_error_mps = math.hypot(float(coasting["velocity_east_mps"]) - MOVING_PERSON_SPEED_MPS, float(coasting["velocity_north_mps"]))

        self.assertEqual("coasting", coasting["track_status"])
        self.assertGreater(coasting_covariance_trace, initial_covariance_trace)
        self.assertLess(position_error_m, 90.0, f"position_error_m={position_error_m:.3f}; coasting={coasting!r}")
        self.assertLess(velocity_error_mps, 12.0, f"velocity_error_mps={velocity_error_mps:.3f}; coasting={coasting!r}")

    def test_polling_does_not_mutate_filter_epoch_and_loss_occurs_after_timeout(self) -> None:
        simulator = DroneSimulator(
            1,
            "Red",
            [],
            DEFAULT_SCENARIO_SEED,
            scenario_id=MOVING_HANDOFF_SCENARIO_ID,
            initial_observer_id=1,
            next_observer_id=2,
            scenario_run_id="test-polling-moving-run",
        )

        coverage_times = visible_moving_report_times(simulator)
        self.assertGreaterEqual(len(coverage_times), 6, f"coverage_times={coverage_times!r}")

        for timestamp in coverage_times[:6]:
            advance_to(simulator, timestamp)
            simulator.capture_observations()

        commit_due_reports(simulator, coverage_times[5] + 2.0)
        moving_object = [
            item
            for item in simulator.objects.values()
            if item.get("tracking_model") == "constant_velocity"
        ][0]
        filter_epoch = moving_object["filter_epoch"]

        simulator.objects_payload()
        simulator.objects_payload()

        self.assertEqual(filter_epoch, moving_object["filter_epoch"])

        advance_to(simulator, float(filter_epoch) + MOVING_TRACK_LOST_SECONDS + 2.0)
        payload = simulator.objects_payload()
        displayed = [
            item
            for item in payload["objects"]
            if item.get("tracking_model") == "constant_velocity"
        ][0]

        self.assertEqual("lost", displayed["track_status"])


if __name__ == "__main__":
    unittest.main()
