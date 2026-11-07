import time
import unittest

from simulated_drone import (
    DEFAULT_SCENARIO_SEED,
    MAX_PRIORITY_OBJECT_COUNT,
    MIN_PRIORITY_OBJECT_COUNT,
    PLAYBOOK_OBSERVE_ORBIT,
    RANDOM_PRIORITY_OBJECT_COUNT,
    DroneSimulator,
    Observation,
    covariance_from_sensor_model,
    local_m_to_lat_lon,
    resolve_priority_object_count,
)


def set_simulation_time(simulator: DroneSimulator, simulation_time: float) -> None:
    with simulator.lock:
        simulator.simulation_time = simulation_time
        simulator.last_wall_time = time.monotonic()


class StageAPriorityResponseTest(unittest.TestCase):
    def test_priority_count_is_bounded_and_reproducible(self) -> None:
        first = resolve_priority_object_count(DEFAULT_SCENARIO_SEED, RANDOM_PRIORITY_OBJECT_COUNT)
        second = resolve_priority_object_count(DEFAULT_SCENARIO_SEED, RANDOM_PRIORITY_OBJECT_COUNT)

        self.assertEqual(first, second)
        self.assertGreaterEqual(first, MIN_PRIORITY_OBJECT_COUNT)
        self.assertLessEqual(first, MAX_PRIORITY_OBJECT_COUNT)

    def test_undetected_priority_does_not_leak_into_belief_map(self) -> None:
        simulator = DroneSimulator(1, "Red", [], DEFAULT_SCENARIO_SEED, 5)

        payload = simulator.objects_payload()

        self.assertEqual(payload["objects"], [])

    def test_priority_observation_creates_one_idempotent_task(self) -> None:
        simulator = DroneSimulator(1, "Red", [], DEFAULT_SCENARIO_SEED, 1, PLAYBOOK_OBSERVE_ORBIT)
        latitude, longitude = local_m_to_lat_lon(25.0, 25.0)
        observation = Observation(
            id="priority-observation-1",
            scenario_run_id=str(DEFAULT_SCENARIO_SEED),
            source_drone_id=1,
            source_drone_name="Red",
            source_color="red",
            sensor_id="eo-red",
            source_track_id="track-priority",
            timestamp=1.0,
            measurement_frame="test",
            east_m=25.0,
            north_m=25.0,
            latitude=latitude,
            longitude=longitude,
            label="vehicle",
            classification_score=0.5,
            covariance=covariance_from_sensor_model(simulator=simulator) if False else [[64.0, 0.0], [0.0, 64.0]],
            priority_evidence={
                "signature_id": "mock-search-signature-test",
                "signature_type": "mocked_search_signature_match",
                "match_score": 0.99,
                "model": "stage-a-test",
            },
        )

        self.assertTrue(simulator.receive_observation(observation.__dict__))
        self.assertFalse(simulator.receive_observation(observation.__dict__))
        self.assertEqual(len(simulator.response_tasks), 1)

    def test_guidance_uses_estimated_position_not_truth(self) -> None:
        simulator = DroneSimulator(1, "Red", [], DEFAULT_SCENARIO_SEED, 1, PLAYBOOK_OBSERVE_ORBIT)
        set_simulation_time(simulator, 10.0)
        latitude, longitude = local_m_to_lat_lon(140.0, -20.0)
        observation = Observation(
            id="priority-observation-estimate",
            scenario_run_id=str(DEFAULT_SCENARIO_SEED),
            source_drone_id=1,
            source_drone_name="Red",
            source_color="red",
            sensor_id="eo-red",
            source_track_id="track-priority-estimate",
            timestamp=10.0,
            measurement_frame="test",
            east_m=140.0,
            north_m=-20.0,
            latitude=latitude,
            longitude=longitude,
            label="vehicle",
            classification_score=0.5,
            covariance=[[64.0, 0.0], [0.0, 64.0]],
            priority_evidence={
                "signature_id": "mock-search-signature-estimate",
                "signature_type": "mocked_search_signature_match",
                "match_score": 0.99,
                "model": "stage-a-test",
            },
        )

        self.assertTrue(simulator.receive_observation(observation.__dict__))

        with simulator.lock:
            task = next(iter(simulator.response_tasks.values()))
            task["election_deadline"] = simulator.simulation_time

        status = simulator.status()

        self.assertIsNotNone(status.response_task)
        assert status.response_task is not None
        self.assertEqual(status.response_task["assigned_drone_id"], 1)
        self.assertAlmostEqual(float(status.response_task["estimated_east_m"]), 140.0)
        self.assertAlmostEqual(float(status.response_task["estimated_north_m"]), -20.0)

    def test_cancel_returns_responder_to_patrol(self) -> None:
        simulator = DroneSimulator(1, "Red", [], DEFAULT_SCENARIO_SEED, 1, PLAYBOOK_OBSERVE_ORBIT)
        latitude, longitude = local_m_to_lat_lon(30.0, 0.0)
        observation = Observation(
            id="priority-observation-cancel",
            scenario_run_id=str(DEFAULT_SCENARIO_SEED),
            source_drone_id=1,
            source_drone_name="Red",
            source_color="red",
            sensor_id="eo-red",
            source_track_id="track-priority-cancel",
            timestamp=1.0,
            measurement_frame="test",
            east_m=30.0,
            north_m=0.0,
            latitude=latitude,
            longitude=longitude,
            label="vehicle",
            classification_score=0.5,
            covariance=[[64.0, 0.0], [0.0, 64.0]],
            priority_evidence={
                "signature_id": "mock-search-signature-cancel",
                "signature_type": "mocked_search_signature_match",
                "match_score": 0.99,
                "model": "stage-a-test",
            },
        )

        self.assertTrue(simulator.receive_observation(observation.__dict__))
        simulator.cancel_response()

        self.assertIsNone(simulator.active_response_task_id)
        self.assertFalse(simulator._has_local_response_control_locked())


if __name__ == "__main__":
    unittest.main()
