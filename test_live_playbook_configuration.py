import unittest
from dataclasses import asdict

from simulated_drone import (
    DEFAULT_SCENARIO_SEED,
    Observation,
    DroneSimulator,
    MEASUREMENT_FRAME,
    PLAYBOOK_OBSERVE_ORBIT,
    PLAYBOOK_REPORT_CONTINUE,
    SENSOR_MODELS,
    local_m_to_lat_lon,
    task_id_for_priority_signature,
)


def priority_observation(signature_id: str = "mock-search-signature-live-config") -> Observation:
    latitude, longitude = local_m_to_lat_lon(25.0, 35.0)
    sensor_model = SENSOR_MODELS[1]
    return Observation(
        id="live-config-observation",
        scenario_run_id=str(DEFAULT_SCENARIO_SEED),
        source_drone_id=1,
        source_drone_name="Red",
        source_color="red",
        sensor_id=sensor_model.sensor_id,
        source_track_id="live-config-track",
        timestamp=1.0,
        measurement_frame=MEASUREMENT_FRAME,
        east_m=25.0,
        north_m=35.0,
        latitude=latitude,
        longitude=longitude,
        label="vehicle",
        classification_score=0.95,
        covariance=[[64.0, 0.0], [0.0, 36.0]],
        priority_evidence={
            "signature_id": signature_id,
            "signature_type": "mocked_search_signature_match",
            "match_score": 0.99,
            "model": "test",
        },
        object_id="test-object",
    )


class LivePlaybookConfigurationTest(unittest.TestCase):
    def test_switch_to_orbit_makes_existing_discovery_assignable_without_reset(self) -> None:
        simulator = DroneSimulator(1, "Red", [], selected_playbook=PLAYBOOK_REPORT_CONTINUE)
        observation = priority_observation()
        self.assertTrue(simulator.receive_observation(asdict(observation)))

        with simulator.lock:
            simulator.simulation_time = 12.0
            task_id = task_id_for_priority_signature(DEFAULT_SCENARIO_SEED, "mock-search-signature-live-config")
            simulator.response_tasks[task_id]["status"] = "reported"
            simulator.active_response_task_id = None
            simulator.flight_trail.append({"latitude": observation.latitude, "longitude": observation.longitude, "timestamp": 1.0})
            object_count = len(simulator.objects)
            observation_count = len(simulator.observations)
            trail_count = len(simulator.flight_trail)

        self.assertTrue(simulator.set_playbook(PLAYBOOK_OBSERVE_ORBIT, 7))

        with simulator.lock:
            task = simulator.response_tasks[task_id]
            self.assertEqual(simulator.playbook_config_version, 7)
            self.assertAlmostEqual(simulator.simulation_time, 12.0, places=2)
            self.assertEqual(len(simulator.objects), object_count)
            self.assertEqual(len(simulator.observations), observation_count)
            self.assertEqual(len(simulator.flight_trail), trail_count)
            self.assertEqual(task["status"], "electing")
            self.assertEqual(task["playbook_config_version"], 7)

        self.assertTrue(simulator.set_playbook(PLAYBOOK_OBSERVE_ORBIT, 7))

        with simulator.lock:
            self.assertEqual(len(simulator.outbound_claims), 1)

    def test_delayed_old_assignment_cannot_restore_owner_after_release(self) -> None:
        simulator = DroneSimulator(1, "Red", [], selected_playbook=PLAYBOOK_OBSERVE_ORBIT)
        observation = priority_observation("mock-search-signature-delayed-release")
        self.assertTrue(simulator.receive_observation(asdict(observation)))

        with simulator.lock:
            task_id = task_id_for_priority_signature(DEFAULT_SCENARIO_SEED, "mock-search-signature-delayed-release")
            task = simulator.response_tasks[task_id]
            task["status"] = "assigned_remote"
            task["assigned_drone_id"] = 2
            task["assigned_drone_name"] = "Blue"
            task["revision"] = 2
            task["playbook_config_version"] = 3
            old_assignment = simulator._task_claim_payload_locked(task, "orbiting")
            old_assignment["revision"] = 2
            release = dict(old_assignment)
            release["status"] = "released"
            release["task_state"] = "released"
            release["revision"] = 3

        self.assertTrue(simulator.receive_claim(release))
        self.assertFalse(simulator.receive_claim(old_assignment))

        with simulator.lock:
            task = simulator.response_tasks[task_id]
            self.assertEqual(task["status"], "reported")
            self.assertIsNone(task["assigned_drone_id"])
            self.assertIsNone(task["assigned_drone_name"])
            self.assertEqual(task["previous_assigned_drone_id"], 2)
            self.assertEqual(task["owner_task_state"], "released")
            self.assertEqual(task["revision"], 3)

    def test_switch_to_report_releases_without_explicit_cancellation(self) -> None:
        simulator = DroneSimulator(1, "Red", [], selected_playbook=PLAYBOOK_OBSERVE_ORBIT)
        observation = priority_observation("mock-search-signature-release")
        self.assertTrue(simulator.receive_observation(asdict(observation)))

        with simulator.lock:
            task_id = task_id_for_priority_signature(DEFAULT_SCENARIO_SEED, "mock-search-signature-release")
            task = simulator.response_tasks[task_id]
            task["status"] = "orbiting"
            task["assigned_drone_id"] = 1
            task["assigned_drone_name"] = "Red"
            task["playbook_config_version"] = simulator.playbook_config_version
            simulator.active_response_task_id = task_id
            simulator.response_east_m = 25.0
            simulator.response_north_m = 120.0
            simulator.response_heading_deg = 90.0
            simulator.response_camera_yaw_deg = 180.0
            simulator.outbound_claims.clear()

        self.assertTrue(simulator.set_playbook(PLAYBOOK_REPORT_CONTINUE, 4))

        with simulator.lock:
            self.assertEqual(simulator.response_tasks[task_id]["status"], "reported")
            self.assertNotIn("mock-search-signature-release", simulator.cancelled_priority_signatures)
            self.assertIsNone(simulator.active_response_task_id)
            self.assertIsNotNone(simulator.return_east_m)
            self.assertEqual(len(simulator.outbound_claims), 1)
            self.assertEqual(simulator.outbound_claims[0]["status"], "released")


if __name__ == "__main__":
    unittest.main()
