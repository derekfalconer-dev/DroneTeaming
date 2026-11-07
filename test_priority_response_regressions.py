#!/usr/bin/env python3
"""Regression tests for priority-response ownership and flight-trail behavior."""

from __future__ import annotations

import contextlib
import json
import socket
import subprocess
import sys
import time
import unittest
import urllib.request
from dataclasses import asdict
from pathlib import Path

from simulated_drone import (
    DEFAULT_SCENARIO_SEED,
    MEASUREMENT_FRAME,
    PLAYBOOK_OBSERVE_ORBIT,
    TASK_ELECTION_SECONDS,
    DroneSimulator,
    Observation,
    local_m_to_lat_lon,
)


HOST = "127.0.0.1"


def free_port() -> int:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind((HOST, 0))
        return int(sock.getsockname()[1])


def wait_for_port(port: int, timeout_seconds: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_seconds

    while time.monotonic() < deadline:
        with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            sock.settimeout(0.2)

            if sock.connect_ex((HOST, port)) == 0:
                return True

        time.sleep(0.05)

    return False


def fetch_json(url: str, method: str = "GET", payload: dict[str, object] | None = None) -> dict[str, object]:
    data = None
    headers = {}

    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(url, data=data, headers=headers, method=method)

    with urllib.request.urlopen(request, timeout=3) as response:
        return json.loads(response.read().decode("utf-8"))


def priority_observation(
    *,
    drone_id: int,
    drone_name: str,
    signature_id: str,
    observation_id: str,
    east_m: float,
    north_m: float,
    timestamp: float = 1.0,
) -> Observation:
    latitude, longitude = local_m_to_lat_lon(east_m, north_m)
    color = {1: "red", 2: "blue", 3: "green"}[drone_id]

    return Observation(
        id=observation_id,
        scenario_run_id=str(DEFAULT_SCENARIO_SEED),
        source_drone_id=drone_id,
        source_drone_name=drone_name,
        source_color=color,
        sensor_id=f"eo-{color}",
        source_track_id=f"track-{signature_id}-{drone_id}",
        timestamp=timestamp,
        measurement_frame=MEASUREMENT_FRAME,
        east_m=east_m,
        north_m=north_m,
        latitude=latitude,
        longitude=longitude,
        label="vehicle",
        classification_score=0.9,
        covariance=[[64.0, 0.0], [0.0, 64.0]],
        priority_evidence={
            "signature_id": signature_id,
            "signature_type": "mocked_search_signature_match",
            "match_score": 0.99,
            "model": "stage-a-regression-test",
        },
    )


def force_assignment(simulator: DroneSimulator) -> None:
    with simulator.lock:
        task = next(iter(simulator.response_tasks.values()))
        task["election_deadline"] = simulator.simulation_time
    simulator.status()


class PriorityResponseUnitRegressionTest(unittest.TestCase):
    def test_trail_samples_match_actual_orbit_positions(self) -> None:
        simulator = DroneSimulator(1, "Red", [], DEFAULT_SCENARIO_SEED, 3, PLAYBOOK_OBSERVE_ORBIT)
        observation = priority_observation(
            drone_id=1,
            drone_name="Red",
            signature_id="mock-search-signature-trail",
            observation_id="trail-observation",
            east_m=30.0,
            north_m=0.0,
        )

        self.assertTrue(simulator.receive_observation(asdict(observation)))
        force_assignment(simulator)

        with simulator.lock:
            simulator.last_wall_time = time.monotonic() - 1.0
        status = simulator.status()

        self.assertTrue(status.flight_trail)
        last_point = status.flight_trail[-1]
        self.assertAlmostEqual(last_point["latitude"], status.latitude, delta=0.000001)
        self.assertAlmostEqual(last_point["longitude"], status.longitude, delta=0.000001)

    def test_old_patrol_dots_expire_after_30_seconds_in_orbit(self) -> None:
        simulator = DroneSimulator(1, "Red", [], DEFAULT_SCENARIO_SEED, 3, PLAYBOOK_OBSERVE_ORBIT)
        simulator.status()
        observation = priority_observation(
            drone_id=1,
            drone_name="Red",
            signature_id="mock-search-signature-trail-expiry",
            observation_id="trail-expiry-observation",
            east_m=30.0,
            north_m=0.0,
        )

        self.assertTrue(simulator.receive_observation(asdict(observation)))
        force_assignment(simulator)

        with simulator.lock:
            simulator.last_wall_time = time.monotonic() - 31.0
        status = simulator.status()

        self.assertTrue(status.flight_trail)
        self.assertGreater(min(point["timestamp"] for point in status.flight_trail), status.timestamp - 30.0)

    def test_one_drone_cannot_retain_two_active_assignments(self) -> None:
        simulator = DroneSimulator(1, "Red", [], DEFAULT_SCENARIO_SEED, 3, PLAYBOOK_OBSERVE_ORBIT)

        self.assertTrue(simulator.receive_observation(asdict(priority_observation(
            drone_id=1,
            drone_name="Red",
            signature_id="mock-search-signature-one",
            observation_id="busy-one",
            east_m=20.0,
            north_m=0.0,
        ))))
        force_assignment(simulator)

        self.assertTrue(simulator.receive_observation(asdict(priority_observation(
            drone_id=1,
            drone_name="Red",
            signature_id="mock-search-signature-two",
            observation_id="busy-two",
            east_m=120.0,
            north_m=0.0,
        ))))

        tasks = simulator._visible_response_tasks_locked()
        local_active = [
            task
            for task in tasks
            if task["assigned_drone_id"] == 1 and task["status"] in {"transitioning", "orbiting"}
        ]
        queued = [task for task in tasks if task["status"] == "queued"]

        self.assertEqual(1, len(local_active))
        self.assertEqual(1, len(queued))

    def test_repeated_observations_do_not_duplicate_or_reassign_committed_tasks(self) -> None:
        simulator = DroneSimulator(1, "Red", [], DEFAULT_SCENARIO_SEED, 3, PLAYBOOK_OBSERVE_ORBIT)
        first = priority_observation(
            drone_id=1,
            drone_name="Red",
            signature_id="mock-search-signature-repeat",
            observation_id="repeat-one",
            east_m=20.0,
            north_m=0.0,
        )
        second = priority_observation(
            drone_id=1,
            drone_name="Red",
            signature_id="mock-search-signature-repeat",
            observation_id="repeat-two",
            east_m=22.0,
            north_m=1.0,
            timestamp=2.0,
        )

        self.assertTrue(simulator.receive_observation(asdict(first)))
        force_assignment(simulator)
        before = simulator._visible_response_tasks_locked()
        self.assertTrue(simulator.receive_observation(asdict(second)))
        after = simulator._visible_response_tasks_locked()

        self.assertEqual(1, len(after))
        self.assertEqual(before[0]["assigned_drone_id"], after[0]["assigned_drone_id"])
        self.assertEqual(before[0]["status"], after[0]["status"])

    def test_busy_drone_does_not_abandon_assignment_for_new_discovery(self) -> None:
        simulator = DroneSimulator(1, "Red", [], DEFAULT_SCENARIO_SEED, 3, PLAYBOOK_OBSERVE_ORBIT)

        self.assertTrue(simulator.receive_observation(asdict(priority_observation(
            drone_id=1,
            drone_name="Red",
            signature_id="mock-search-signature-primary",
            observation_id="primary",
            east_m=20.0,
            north_m=0.0,
        ))))
        force_assignment(simulator)
        active_task_id = simulator.active_response_task_id

        self.assertTrue(simulator.receive_observation(asdict(priority_observation(
            drone_id=1,
            drone_name="Red",
            signature_id="mock-search-signature-secondary",
            observation_id="secondary",
            east_m=140.0,
            north_m=0.0,
        ))))

        self.assertEqual(active_task_id, simulator.active_response_task_id)

    def test_cancel_does_not_immediately_reacquire_from_repeated_observation(self) -> None:
        simulator = DroneSimulator(1, "Red", [], DEFAULT_SCENARIO_SEED, 3, PLAYBOOK_OBSERVE_ORBIT)
        first = priority_observation(
            drone_id=1,
            drone_name="Red",
            signature_id="mock-search-signature-cancel-regression",
            observation_id="cancel-regression-one",
            east_m=20.0,
            north_m=0.0,
        )
        repeated = priority_observation(
            drone_id=1,
            drone_name="Red",
            signature_id="mock-search-signature-cancel-regression",
            observation_id="cancel-regression-two",
            east_m=21.0,
            north_m=0.0,
            timestamp=2.0,
        )

        self.assertTrue(simulator.receive_observation(asdict(first)))
        force_assignment(simulator)
        simulator.cancel_response()
        self.assertTrue(simulator.receive_observation(asdict(repeated)))

        tasks = simulator._visible_response_tasks_locked()
        self.assertEqual(1, len(tasks))
        self.assertEqual("cancelled", tasks[0]["status"])
        self.assertIsNone(simulator.active_response_task_id)

    def test_pause_freezes_motion_trails_and_allocation_timers(self) -> None:
        simulator = DroneSimulator(1, "Red", [], DEFAULT_SCENARIO_SEED, 3, PLAYBOOK_OBSERVE_ORBIT)
        self.assertTrue(simulator.receive_observation(asdict(priority_observation(
            drone_id=1,
            drone_name="Red",
            signature_id="mock-search-signature-paused",
            observation_id="paused",
            east_m=20.0,
            north_m=0.0,
        ))))
        simulator.set_paused(True)

        before = simulator.status()
        time.sleep(0.1)
        after = simulator.status()

        self.assertAlmostEqual(before.timestamp, after.timestamp, delta=0.000001)
        self.assertEqual(before.flight_trail, after.flight_trail)
        self.assertEqual("electing", after.response_tasks[0]["status"])

    def test_local_winner_initializes_guidance_before_transitioning_state(self) -> None:
        simulator = DroneSimulator(1, "Red", [], DEFAULT_SCENARIO_SEED, 3, PLAYBOOK_OBSERVE_ORBIT)
        self.assertTrue(simulator.receive_observation(asdict(priority_observation(
            drone_id=1,
            drone_name="Red",
            signature_id="mock-search-signature-local-init",
            observation_id="local-init",
            east_m=20.0,
            north_m=0.0,
        ))))

        force_assignment(simulator)
        status = simulator.status()

        self.assertEqual("transitioning", status.current_mode)
        self.assertIsNotNone(simulator.response_east_m)
        self.assertIsNotNone(simulator.response_north_m)
        self.assertTrue(status.flight_trail)

    def test_peer_delivered_local_ownership_initializes_once_and_rejects_stale_state(self) -> None:
        simulator = DroneSimulator(1, "Red", [], DEFAULT_SCENARIO_SEED, 3, PLAYBOOK_OBSERVE_ORBIT)
        self.assertTrue(simulator.receive_observation(asdict(priority_observation(
            drone_id=1,
            drone_name="Red",
            signature_id="mock-search-signature-peer-local-init",
            observation_id="peer-local-init",
            east_m=25.0,
            north_m=0.0,
        ))))
        simulator.set_paused(True)
        task = simulator._visible_response_tasks_locked()[0]
        transition_claim = {
            "scenario_run_id": str(DEFAULT_SCENARIO_SEED),
            "task_id": task["task_id"],
            "signature_id": task["signature_id"],
            "claimant_drone_id": 2,
            "claimant_drone_name": "Blue",
            "owner_drone_id": 1,
            "owner_drone_name": "Red",
            "status": "transitioning",
            "task_state": "transitioning",
            "revision": 1,
            "estimated_east_m": task["estimated_east_m"],
            "estimated_north_m": task["estimated_north_m"],
            "estimated_latitude": task["estimated_latitude"],
            "estimated_longitude": task["estimated_longitude"],
            "election_deadline": task["election_deadline"],
        }

        self.assertTrue(simulator.receive_claim(transition_claim))
        initial_guidance = (
            simulator.response_east_m,
            simulator.response_north_m,
            simulator.response_heading_deg,
            simulator.response_camera_yaw_deg,
        )
        self.assertEqual("transitioning", simulator.status().current_mode)

        self.assertTrue(simulator.receive_claim(transition_claim))
        self.assertEqual(
            initial_guidance,
            (
                simulator.response_east_m,
                simulator.response_north_m,
                simulator.response_heading_deg,
                simulator.response_camera_yaw_deg,
            ),
        )

        stale_claim = dict(transition_claim)
        stale_claim["task_state"] = "cancelled"
        stale_claim["status"] = "cancelled"
        stale_claim["revision"] = 0

        self.assertFalse(simulator.receive_claim(stale_claim))
        self.assertEqual("transitioning", simulator.status().current_mode)

    def test_cancel_emits_cancelled_claim_and_prevents_reacquisition(self) -> None:
        simulator = DroneSimulator(1, "Red", [], DEFAULT_SCENARIO_SEED, 3, PLAYBOOK_OBSERVE_ORBIT)
        first = priority_observation(
            drone_id=1,
            drone_name="Red",
            signature_id="mock-search-signature-cancel-propagation",
            observation_id="cancel-propagation-one",
            east_m=20.0,
            north_m=0.0,
        )
        repeated = priority_observation(
            drone_id=1,
            drone_name="Red",
            signature_id="mock-search-signature-cancel-propagation",
            observation_id="cancel-propagation-two",
            east_m=21.0,
            north_m=0.0,
            timestamp=2.0,
        )

        self.assertTrue(simulator.receive_observation(asdict(first)))
        force_assignment(simulator)
        simulator.pop_outbound_claims()
        simulator.cancel_response()
        claims = simulator.pop_outbound_claims()
        self.assertTrue(any(claim["task_state"] == "cancelled" for claim in claims))

        self.assertTrue(simulator.receive_observation(asdict(repeated)))
        tasks = simulator._visible_response_tasks_locked()

        self.assertEqual(1, len(tasks))
        self.assertEqual("cancelled", tasks[0]["status"])
        self.assertIsNone(simulator.active_response_task_id)


class PriorityResponsePeerRegressionTest(unittest.TestCase):
    processes: list[subprocess.Popen[str]]
    urls: list[str]

    def setUp(self) -> None:
        self.ports = [free_port(), free_port(), free_port()]
        self.urls = [f"http://{HOST}:{port}" for port in self.ports]
        names = ["Red", "Blue", "Green"]
        script = Path(__file__).with_name("simulated_drone.py")
        self.processes = []
        self.addCleanup(self.cleanup_processes)

        for index, (port, name) in enumerate(zip(self.ports, names, strict=True), start=1):
            peers = [url for url in self.urls if not url.endswith(f":{port}")]
            self.processes.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        str(script),
                        "--id",
                        str(index),
                        "--name",
                        name,
                        "--host",
                        HOST,
                        "--port",
                        str(port),
                        "--peers",
                        *peers,
                        "--priority-count",
                        "3",
                        "--playbook",
                        PLAYBOOK_OBSERVE_ORBIT,
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    text=True,
                ),
            )

        for process, port in zip(self.processes, self.ports, strict=True):
            self.assertTrue(wait_for_port(port), f"Drone on port {port} did not start")
            self.assertIsNone(process.poll(), f"Drone on port {port} exited")

        for url in self.urls:
            fetch_json(f"{url}/control", method="POST", payload={"paused": True})

    def cleanup_processes(self) -> None:
        for process in self.processes:
            if process.poll() is None:
                process.terminate()

        for process in self.processes:
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=5)

        for process in self.processes:
            if process.poll() is None:
                process.kill()

    def post_observation(self, url: str, observation: Observation) -> None:
        response = fetch_json(f"{url}/observations", method="POST", payload=asdict(observation))
        self.assertTrue(response["accepted"])

    def resume_all(self) -> None:
        for url in self.urls:
            fetch_json(f"{url}/control", method="POST", payload={"paused": False})

    def wait_for_assignments(self, expected_task_count: int, timeout_seconds: float = 8.0) -> list[dict[str, object]]:
        deadline = time.monotonic() + timeout_seconds
        last_tasks: list[dict[str, object]] = []

        while time.monotonic() < deadline:
            statuses = [fetch_json(f"{url}/status") for url in self.urls]
            all_tasks = [task for status in statuses for task in status["response_tasks"]]
            by_signature: dict[str, dict[str, object]] = {}

            for task in all_tasks:
                by_signature[str(task["signature_id"])] = task

            last_tasks = list(by_signature.values())
            assigned_tasks = [
                task
                for task in last_tasks
                if task.get("assigned_drone_id") is not None
                and task["status"] in {"transitioning", "orbiting", "assigned_remote"}
            ]

            if len(assigned_tasks) >= expected_task_count:
                return assigned_tasks

            time.sleep(0.2)

        raise AssertionError(f"Assignments did not converge; last_tasks={last_tasks!r}")

    def wait_for_owner_modes(self, owner_ids: set[int], timeout_seconds: float = 8.0) -> list[dict[str, object]]:
        deadline = time.monotonic() + timeout_seconds
        last_statuses: list[dict[str, object]] = []

        while time.monotonic() < deadline:
            last_statuses = [fetch_json(f"{url}/status") for url in self.urls]

            for url in self.urls:
                fetch_json(f"{url}/objects")

            ready_owner_ids = {
                int(status["id"])
                for status in last_statuses
                if int(status["id"]) in owner_ids
                and status["current_mode"] in {"transitioning", "orbiting"}
            }

            if ready_owner_ids == owner_ids:
                return last_statuses

            time.sleep(0.2)

        raise AssertionError(f"Owners did not enter response execution; last_statuses={last_statuses!r}")

    def test_three_connected_drones_service_three_separate_objectives(self) -> None:
        observations = [
            priority_observation(drone_id=1, drone_name="Red", signature_id="mock-search-signature-http-a", observation_id="http-a", east_m=-200.0, north_m=0.0),
            priority_observation(drone_id=2, drone_name="Blue", signature_id="mock-search-signature-http-b", observation_id="http-b", east_m=0.0, north_m=150.0),
            priority_observation(drone_id=3, drone_name="Green", signature_id="mock-search-signature-http-c", observation_id="http-c", east_m=180.0, north_m=-120.0),
        ]

        for url, observation in zip(self.urls, observations, strict=True):
            self.post_observation(url, observation)

        self.resume_all()
        assignments = self.wait_for_assignments(3)
        owners = {int(task["assigned_drone_id"]) for task in assignments}

        self.assertEqual({1, 2, 3}, owners)
        self.wait_for_owner_modes(owners)

    def test_simultaneous_discovery_of_one_objective_produces_one_owner(self) -> None:
        signature_id = "mock-search-signature-http-single-owner"

        for index, (url, name) in enumerate(zip(self.urls, ["Red", "Blue", "Green"], strict=True), start=1):
            self.post_observation(
                url,
                priority_observation(
                    drone_id=index,
                    drone_name=name,
                    signature_id=signature_id,
                    observation_id=f"http-single-owner-{index}",
                    east_m=0.0,
                    north_m=0.0,
                ),
            )

        self.resume_all()
        assignments = self.wait_for_assignments(1)
        owners = {
            int(task["assigned_drone_id"])
            for task in assignments
            if task["signature_id"] == signature_id
        }

        self.assertEqual(1, len(owners))


if __name__ == "__main__":
    unittest.main()
