import contextlib
import json
import math
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.request
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import simulated_drone
from simulated_drone import (
    DEFAULT_SCENARIO_SEED,
    DEFAULT_PAYLOAD_TYPE,
    MAP_REFERENCE_LATITUDE,
    MAP_REFERENCE_LONGITUDE,
    PLAYBOOK_OBSERVE_ORBIT,
    TASK_TYPE_ISR_OBSERVE,
    TASK_TYPE_PAYLOAD_DROP,
    SAR_SUPPLY_SCENARIO_ID,
    DroneSimulator,
    TeamMemberConfiguration,
    Observation,
    lat_lon_to_local_m,
    task_id_for_contact,
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
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if payload is not None else {},
        method=method,
    )

    with urllib.request.urlopen(request, timeout=3) as response:
        return json.loads(response.read().decode("utf-8"))


def process_output(process: subprocess.Popen[str]) -> tuple[str, str]:
    if process.poll() is None:
        return "<still running>", "<still running>"

    try:
        stdout, stderr = process.communicate(timeout=0.2)
    except subprocess.TimeoutExpired:
        return "<stdout unavailable>", "<stderr unavailable>"

    return stdout or "", stderr or ""


def format_process_startup_failure(commands: list[list[str]], processes: list[subprocess.Popen[str]]) -> str:
    lines = ["Drone process startup failed."]

    for command, process in zip(commands, processes, strict=False):
        stdout, stderr = process_output(process)
        lines.extend(
            [
                f"command: {subprocess.list2cmdline(command)}",
                f"exit_code: {process.poll()}",
                f"stdout: {stdout.strip() or '<empty>'}",
                f"stderr: {stderr.strip() or '<empty>'}",
            ],
        )

    return "\n".join(lines)


def close_process_pipes(processes: list[subprocess.Popen[str]]) -> None:
    for process in processes:
        for pipe in (process.stdout, process.stderr):
            if pipe is not None and not pipe.closed:
                pipe.close()


def team_config(
    *,
    red_payload: int = 0,
    blue_payload: int = 1,
    green_payload: int = 0,
    blue_available: bool = True,
    payload_type: str = "rescue_kit",
):
    return {
        1: TeamMemberConfiguration(
            role="isr",
            capabilities=("atr_observe", "isr_observe"),
            available=True,
            camera_mount="steerable",
        ),
        2: TeamMemberConfiguration(
            role="payload",
            capabilities=("payload_drop",),
            available=blue_available,
            payload_inventory={payload_type: blue_payload},
            camera_mount="fixed",
        ),
        3: TeamMemberConfiguration(
            role="payload",
            capabilities=("atr_observe", "isr_observe", "payload_drop"),
            available=True,
            payload_inventory={payload_type: green_payload},
            camera_mount="steerable",
        ),
    }


def sar_assistance_observation(
    *,
    observation_id: str,
    source_drone_id: int = 1,
    source_drone_name: str = "Red",
    signature_id: str = "sar-contact",
    payload_type: str = "water",
    quantity: int = 2,
    east_m: float = 20.0,
    north_m: float = 30.0,
    timestamp: float = 1.0,
    scenario_run_id: str = "payload-run",
) -> Observation:
    observation = priority_observation(
        observation_id=observation_id,
        source_drone_id=source_drone_id,
        source_drone_name=source_drone_name,
        signature_id=signature_id,
        east_m=east_m,
        north_m=north_m,
        timestamp=timestamp,
        scenario_run_id=scenario_run_id,
    )
    evidence = dict(observation.priority_evidence or {})
    evidence["assistance_request"] = {
        "requested_payload_type": payload_type,
        "requested_payload_quantity": quantity,
        "evidence_type": "mocked_visible_supply_need_marker",
        "confidence": 0.93,
    }
    return Observation(**{**asdict(observation), "priority_evidence": evidence})


def priority_observation(
    *,
    observation_id: str,
    source_drone_id: int = 1,
    source_drone_name: str = "Red",
    signature_id: str = "payload-contact",
    east_m: float = 20.0,
    north_m: float = 30.0,
    timestamp: float = 1.0,
    scenario_run_id: str = "payload-run",
) -> Observation:
    return Observation(
        id=observation_id,
        scenario_run_id=scenario_run_id,
        source_drone_id=source_drone_id,
        source_drone_name=source_drone_name,
        source_color="red",
        sensor_id=f"eo-{source_drone_name.lower()}",
        source_track_id=f"track-{signature_id}",
        timestamp=timestamp,
        measurement_frame="test",
        east_m=east_m,
        north_m=north_m,
        latitude=37.7752 + (north_m / 111111),
        longitude=-122.4230 + (east_m / 88000),
        label="vehicle",
        classification_score=0.99,
        covariance=[[25.0, 0.0], [0.0, 25.0]],
        priority_evidence={
            "signature_id": signature_id,
            "signature_type": "mocked_search_signature_match",
            "match_score": 0.99,
            "model": "test",
        },
    )


def distance_to_estimate(simulator: DroneSimulator, task: dict[str, object]) -> float:
    assert simulator.response_east_m is not None
    assert simulator.response_north_m is not None
    return math.hypot(
        simulator.response_east_m - float(task["estimated_east_m"]),
        simulator.response_north_m - float(task["estimated_north_m"]),
    )


def payload_tasks(simulator: DroneSimulator) -> list[dict[str, object]]:
    return [
        task
        for task in simulator.control_payload()["response_tasks"]  # type: ignore[index]
        if task["task_type"] == TASK_TYPE_PAYLOAD_DROP
    ]


def distance_from_status_to_task_estimate(status: dict[str, object], task: dict[str, object]) -> float:
    east_m, north_m = lat_lon_to_local_m(
        MAP_REFERENCE_LATITUDE,
        MAP_REFERENCE_LONGITUDE,
        float(status["latitude"]),
        float(status["longitude"]),
    )
    return math.hypot(east_m - float(task["estimated_east_m"]), north_m - float(task["estimated_north_m"]))


def isr_tasks(simulator: DroneSimulator) -> list[dict[str, object]]:
    return [
        task
        for task in simulator.control_payload()["response_tasks"]  # type: ignore[index]
        if task["task_type"] == TASK_TYPE_ISR_OBSERVE
    ]


def elect_payload_task_with_local_estimate(
    *,
    estimate_east_m: float = 20.0,
    estimate_north_m: float = 30.0,
    payload_standoff_m: float = 30.0,
) -> DroneSimulator:
    publisher = DroneSimulator(
        1,
        "Red",
        [],
        DEFAULT_SCENARIO_SEED,
        1,
        PLAYBOOK_OBSERVE_ORBIT,
        scenario_run_id="payload-run",
        team_configuration=team_config(),
        payload_requests_enabled=True,
    )
    receiver = DroneSimulator(
        2,
        "Blue",
        [],
        DEFAULT_SCENARIO_SEED,
        1,
        PLAYBOOK_OBSERVE_ORBIT,
        scenario_run_id="payload-run",
        team_configuration=team_config(blue_payload=1),
        payload_requests_enabled=True,
        payload_standoff_m=payload_standoff_m,
    )
    observation = priority_observation(
        observation_id="estimate-for-blue",
        east_m=estimate_east_m,
        north_m=estimate_north_m,
    )

    self_accepted = receiver.receive_observation(asdict(observation))
    publisher.receive_observation(asdict(observation))
    payload_message = next(
        message
        for message in publisher.pop_outbound_claims()
        if message["task_type"] == TASK_TYPE_PAYLOAD_DROP
    )

    assert self_accepted
    assert receiver.receive_claim(payload_message)
    receiver.set_simulation_time(2.0)
    receiver.status()
    assert receiver.active_response_task_id is not None
    assert receiver.response_east_m is not None
    assert receiver.response_north_m is not None
    assert receiver.response_heading_deg is not None
    return receiver


class PayloadTeamingIncrement2Test(unittest.TestCase):
    def test_repeated_isr_requests_converge_on_one_payload_task(self) -> None:
        simulator = DroneSimulator(
            1,
            "Red",
            [],
            DEFAULT_SCENARIO_SEED,
            1,
            PLAYBOOK_OBSERVE_ORBIT,
            scenario_run_id="payload-run",
            team_configuration=team_config(),
            payload_requests_enabled=True,
        )

        simulator.receive_observation(asdict(priority_observation(observation_id="obs-1")))
        simulator.receive_observation(asdict(priority_observation(observation_id="obs-2", timestamp=2.0, east_m=21.0)))

        tasks = simulator.control_payload()["response_tasks"]
        payload_tasks = [task for task in tasks if task["task_type"] == TASK_TYPE_PAYLOAD_DROP]
        isr_tasks = [task for task in tasks if task["task_type"] == TASK_TYPE_ISR_OBSERVE]

        self.assertEqual(1, len(payload_tasks))
        self.assertEqual(1, len(isr_tasks))
        self.assertEqual("payload-contact", payload_tasks[0]["contact_ref"])

    def test_payload_eligibility_rejects_unavailable_or_insufficient_inventory(self) -> None:
        insufficient = DroneSimulator(
            2,
            "Blue",
            [],
            DEFAULT_SCENARIO_SEED,
            1,
            PLAYBOOK_OBSERVE_ORBIT,
            scenario_run_id="payload-run",
            team_configuration=team_config(blue_payload=0),
            payload_requests_enabled=True,
        )
        request = {
            "scenario_run_id": str(DEFAULT_SCENARIO_SEED),
            "task_id": task_id_for_contact(DEFAULT_SCENARIO_SEED, "payload-contact", TASK_TYPE_PAYLOAD_DROP),
            "task_type": TASK_TYPE_PAYLOAD_DROP,
            "contact_ref": "payload-contact",
            "signature_id": "payload-contact",
            "object_label": "vehicle",
            "claimant_drone_id": 1,
            "claimant_drone_name": "Red",
            "status": "reported",
            "task_state": "reported",
            "revision": 0,
            "assignment_epoch": 1,
            "playbook_config_version": 0,
            "requested_payload_type": DEFAULT_PAYLOAD_TYPE,
            "requested_payload_quantity": 1,
            "estimated_east_m": 20.0,
            "estimated_north_m": 30.0,
            "estimated_latitude": 37.7752,
            "estimated_longitude": -122.4230,
            "election_deadline": 1.5,
            "created_at": 1.0,
            "updated_at": 1.0,
            "cost": 0.0,
        }

        self.assertTrue(insufficient.receive_claim(request))
        insufficient.set_simulation_time(2.0)
        self.assertFalse(any(task["status"] in {"approaching_payload", "awaiting_local_acquisition"} for task in insufficient.control_payload()["response_tasks"]))

    def test_payload_reservation_is_idempotent_and_released_on_cancel(self) -> None:
        publisher = DroneSimulator(
            1,
            "Red",
            [],
            DEFAULT_SCENARIO_SEED,
            1,
            PLAYBOOK_OBSERVE_ORBIT,
            scenario_run_id="payload-run",
            team_configuration=team_config(),
            payload_requests_enabled=True,
        )
        receiver = DroneSimulator(
            2,
            "Blue",
            [],
            DEFAULT_SCENARIO_SEED,
            1,
            PLAYBOOK_OBSERVE_ORBIT,
            scenario_run_id="payload-run",
            team_configuration=team_config(blue_payload=2),
            payload_requests_enabled=True,
        )
        publisher.receive_observation(asdict(priority_observation(observation_id="reservation-election")))
        payload_message = next(
            message
            for message in publisher.pop_outbound_claims()
            if message["task_type"] == TASK_TYPE_PAYLOAD_DROP
        )

        self.assertTrue(receiver.receive_claim(payload_message))
        receiver.set_simulation_time(2.0)
        receiver.status()

        task = payload_tasks(receiver)[0]
        task_id = str(task["task_id"])
        self.assertIn(task["status"], {"approaching_payload", "waiting_for_track", "awaiting_local_acquisition"})
        self.assertEqual(task_id, receiver.active_response_task_id)
        self.assertIsNotNone(receiver.response_east_m)
        self.assertIsNotNone(receiver.response_north_m)
        self.assertIsNotNone(receiver.response_heading_deg)
        self.assertEqual(1, receiver.control_payload()["payload_reservations"][task_id]["quantity"])

        self.assertFalse(receiver.receive_claim(payload_message))
        receiver._reserve_payload_inventory_locked(receiver.response_tasks[task_id])
        self.assertEqual(1, receiver.control_payload()["payload_reservations"][task_id]["quantity"])

        receiver.cancel_response(task_id)
        self.assertEqual({}, receiver.control_payload()["payload_reservations"])
        self.assertIsNone(receiver.active_response_task_id)
        self.assertIsNone(receiver.response_east_m)
        self.assertIsNone(receiver.response_north_m)
        self.assertIsNone(receiver.response_heading_deg)
        self.assertIsNotNone(receiver.return_east_m)
        self.assertEqual({"rescue_kit": 2}, receiver.team_configuration.payload_inventory)

    def test_peer_serialization_delivers_payload_request(self) -> None:
        publisher = DroneSimulator(
            1,
            "Red",
            [],
            DEFAULT_SCENARIO_SEED,
            1,
            PLAYBOOK_OBSERVE_ORBIT,
            scenario_run_id="payload-run",
            team_configuration=team_config(),
            payload_requests_enabled=True,
        )
        receiver = DroneSimulator(
            2,
            "Blue",
            [],
            DEFAULT_SCENARIO_SEED,
            1,
            PLAYBOOK_OBSERVE_ORBIT,
            scenario_run_id="payload-run",
            team_configuration=team_config(blue_payload=1),
            payload_requests_enabled=True,
        )

        publisher.receive_observation(asdict(priority_observation(observation_id="obs-peer")))
        outbound = publisher.pop_outbound_claims()
        payload_messages = [message for message in outbound if message["task_type"] == TASK_TYPE_PAYLOAD_DROP]

        self.assertEqual(1, len(payload_messages))
        self.assertTrue(receiver.receive_claim(payload_messages[0]))
        self.assertTrue(any(task["task_type"] == TASK_TYPE_PAYLOAD_DROP for task in receiver.control_payload()["response_tasks"]))

    def test_payload_approach_uses_estimated_position_and_holds_at_standoff(self) -> None:
        simulator = elect_payload_task_with_local_estimate(
            estimate_east_m=5.0,
            estimate_north_m=5.0,
            payload_standoff_m=30.0,
        )
        task = payload_tasks(simulator)[0]
        deliberate_truth = {"east_m": 55.0, "north_m": -40.0}

        self.assertNotEqual(deliberate_truth["east_m"], float(task["estimated_east_m"]))
        self.assertNotEqual(deliberate_truth["north_m"], float(task["estimated_north_m"]))

        before = distance_to_estimate(simulator, task)
        simulator._update_payload_guidance_locked(1.0)
        after = distance_to_estimate(simulator, simulator.response_tasks[str(task["task_id"])])
        self.assertLess(after, before)

        simulator.response_east_m = float(task["estimated_east_m"]) - 31.0
        simulator.response_north_m = float(task["estimated_north_m"])
        simulator._update_payload_guidance_locked(10.0)
        held_distance = distance_to_estimate(simulator, simulator.response_tasks[str(task["task_id"])])
        held_position = (simulator.response_east_m, simulator.response_north_m)
        self.assertGreaterEqual(held_distance, 29.99)
        self.assertLessEqual(held_distance, 30.01)

        payload_task = payload_tasks(simulator)[0]
        self.assertEqual("awaiting_local_acquisition", payload_task["status"])
        self.assertNotIn("delivered", payload_task["event"].lower())

        simulator._update_payload_guidance_locked(1.0)
        self.assertEqual(held_position, (simulator.response_east_m, simulator.response_north_m))
        self.assertEqual("awaiting_local_acquisition", payload_tasks(simulator)[0]["status"])

    def test_payload_guidance_tracks_moving_estimate_lost_hold_and_recovery(self) -> None:
        simulator = elect_payload_task_with_local_estimate(
            estimate_east_m=20.0,
            estimate_north_m=30.0,
            payload_standoff_m=25.0,
        )
        task = simulator.response_tasks[simulator.active_response_task_id or ""]
        original_estimate = (float(task["estimated_east_m"]), float(task["estimated_north_m"]))

        self.assertTrue(simulator.receive_observation(asdict(priority_observation(
            observation_id="estimate-update",
            east_m=22.0,
            north_m=31.0,
            timestamp=2.0,
        ))))
        simulator._update_payload_guidance_locked(0.5)
        updated_task = simulator.response_tasks[str(task["task_id"])]
        self.assertNotEqual(original_estimate, (float(updated_task["estimated_east_m"]), float(updated_task["estimated_north_m"])))

        simulator.objects.clear()
        held_position = (simulator.response_east_m, simulator.response_north_m)
        simulator._update_payload_guidance_locked(1.0)
        waiting_task = simulator.response_tasks[str(task["task_id"])]
        self.assertEqual("waiting_for_track", waiting_task["status"])
        self.assertEqual(held_position, (simulator.response_east_m, simulator.response_north_m))

        self.assertTrue(simulator.receive_observation(asdict(priority_observation(
            observation_id="estimate-recovery",
            east_m=23.0,
            north_m=31.5,
            timestamp=3.0,
        ))))
        simulator._update_payload_guidance_locked(1.0)
        self.assertIn(simulator.response_tasks[str(task["task_id"])]["status"], {"approaching_payload", "awaiting_local_acquisition"})

    def test_payload_waiting_for_track_messages_are_bounded(self) -> None:
        simulator = elect_payload_task_with_local_estimate()
        task_id = simulator.active_response_task_id or ""
        simulator.objects.clear()
        simulator.outbound_claims.clear()

        for _index in range(10):
            simulator._update_payload_guidance_locked(0.1)

        waiting_messages = [
            claim
            for claim in simulator.pop_outbound_claims()
            if claim["task_id"] == task_id and claim["task_state"] == "waiting_for_track"
        ]
        self.assertEqual(1, len(waiting_messages))

    def test_pause_freezes_payload_motion_and_isr_tracking_continues_independently(self) -> None:
        simulator = elect_payload_task_with_local_estimate()
        task_id = simulator.active_response_task_id or ""
        original_monotonic = simulated_drone.time.monotonic
        clock = [simulator.last_wall_time + 1.0]

        try:
            simulated_drone.time.monotonic = lambda: clock[0]  # type: ignore[method-assign]
            simulator.set_paused(True)
            before = (simulator.response_east_m, simulator.response_north_m)

            clock[0] += 2.0
            simulator.status()
            self.assertEqual(before, (simulator.response_east_m, simulator.response_north_m))
        finally:
            simulated_drone.time.monotonic = original_monotonic  # type: ignore[method-assign]

        continued_observation = asdict(priority_observation(
            observation_id="isr-continues",
            signature_id="second-contact",
            east_m=80.0,
            north_m=85.0,
            timestamp=5.0,
        ))
        accepted = simulator.receive_observation(continued_observation)
        self.assertTrue(
            accepted,
            {
                "observation": continued_observation,
                "unresolved_observations": dict(simulator.unresolved_observations),
                "validation_rejects": simulator.validation_rejects,
                "duplicate_observations_ignored": simulator.duplicate_observations_ignored,
            },
        )
        self.assertEqual(task_id, simulator.active_response_task_id)
        self.assertGreaterEqual(len(isr_tasks(simulator)), 2)

    def test_conflicting_requests_and_duplicate_payload_messages_do_not_duplicate_commitment(self) -> None:
        simulator = elect_payload_task_with_local_estimate()
        simulator.pop_outbound_claims()
        task = payload_tasks(simulator)[0]
        task_id = str(task["task_id"])
        before_task = dict(simulator.response_tasks[task_id])
        before_position = (simulator.response_east_m, simulator.response_north_m, simulator.response_heading_deg)

        duplicate_claim = simulator._task_claim_payload_locked(simulator.response_tasks[task_id])
        duplicate_claim["claimant_drone_id"] = 2
        duplicate_claim["claimant_drone_name"] = "Blue"
        duplicate_claim["owner_drone_id"] = 2
        duplicate_claim["owner_drone_name"] = "Blue"

        clock = [simulator.last_wall_time]
        with patch("simulated_drone.time.monotonic", lambda: clock[0]):
            simulator.receive_claim(duplicate_claim)
            after_task = simulator.response_tasks[task_id]
            self.assertEqual(1, len(payload_tasks(simulator)))
            self.assertEqual(before_task["task_id"], after_task["task_id"])
            self.assertEqual(before_task["assigned_drone_id"], after_task["assigned_drone_id"])
            self.assertEqual(before_task["assigned_drone_name"], after_task["assigned_drone_name"])
            self.assertEqual(before_task["assignment_epoch"], after_task["assignment_epoch"])
            self.assertEqual(before_task["playbook_config_version"], after_task["playbook_config_version"])
            self.assertEqual(before_task["status"], after_task["status"])
            self.assertEqual(task_id, simulator.active_response_task_id)
            self.assertEqual(before_position, (simulator.response_east_m, simulator.response_north_m, simulator.response_heading_deg))
            reservations = simulator.control_payload()["payload_reservations"]
            self.assertEqual([task_id], sorted(reservations))
            self.assertEqual("rescue_kit", reservations[task_id]["payload_type"])
            self.assertEqual(1, reservations[task_id]["quantity"])
            self.assertEqual([], simulator.pop_outbound_claims())
            self.assertLessEqual(set(simulator.claims.get(task_id, {})), {2})

            conflicting = dict(duplicate_claim)
            conflicting["requested_payload_type"] = "water"
            conflicting["revision"] = int(conflicting["revision"]) + 1
            self.assertFalse(simulator.receive_claim(conflicting))
            self.assertTrue(simulator.response_tasks[task_id].get("metadata_conflict"))

            post_claim_position = (simulator.response_east_m, simulator.response_north_m, simulator.response_heading_deg)
            clock[0] += 1.0
            simulator.status()
            self.assertNotEqual(post_claim_position, (simulator.response_east_m, simulator.response_north_m, simulator.response_heading_deg))

    def test_stale_payload_request_does_not_downgrade_configuration_or_reset_owner(self) -> None:
        simulator = elect_payload_task_with_local_estimate()
        task = payload_tasks(simulator)[0]
        task_id = str(task["task_id"])
        simulator.playbook_config_version = 5
        simulator.response_tasks[task_id]["playbook_config_version"] = 5
        before_task = dict(simulator.response_tasks[task_id])
        before_position = (simulator.response_east_m, simulator.response_north_m, simulator.response_heading_deg)

        stale_request = {
            "scenario_run_id": str(DEFAULT_SCENARIO_SEED),
            "task_id": task_id,
            "task_type": TASK_TYPE_PAYLOAD_DROP,
            "contact_ref": task["contact_ref"],
            "signature_id": task["signature_id"],
            "object_label": task["object_label"],
            "claimant_drone_id": 1,
            "claimant_drone_name": "Red",
            "status": "reported",
            "task_state": "reported",
            "revision": max(0, int(task["revision"]) - 1),
            "assignment_epoch": max(0, int(task["assignment_epoch"]) - 1),
            "playbook_config_version": 1,
            "requested_payload_type": task["requested_payload_type"],
            "requested_payload_quantity": task["requested_payload_quantity"],
            "estimated_east_m": task["estimated_east_m"],
            "estimated_north_m": task["estimated_north_m"],
            "estimated_latitude": task["estimated_latitude"],
            "estimated_longitude": task["estimated_longitude"],
            "election_deadline": task["election_deadline"],
            "created_at": task["created_at"],
            "updated_at": task["updated_at"],
            "cost": 0.0,
        }

        before_active_task_id = simulator.active_response_task_id
        clock = [simulator.last_wall_time]
        with patch("simulated_drone.time.monotonic", lambda: clock[0]):
            self.assertFalse(simulator.receive_claim(stale_request))
            after_task = simulator.response_tasks[task_id]
            self.assertEqual(before_task["assigned_drone_id"], after_task["assigned_drone_id"])
            self.assertEqual(before_task["assigned_drone_name"], after_task["assigned_drone_name"])
            self.assertEqual(before_task["assignment_epoch"], after_task["assignment_epoch"])
            self.assertEqual(before_task["status"], after_task["status"])
            self.assertEqual(before_active_task_id, simulator.active_response_task_id)
            self.assertEqual(5, after_task["playbook_config_version"])
            self.assertEqual(before_position, (simulator.response_east_m, simulator.response_north_m, simulator.response_heading_deg))
            reservations = simulator.control_payload()["payload_reservations"]
            self.assertEqual([task_id], sorted(reservations))
            self.assertEqual("rescue_kit", reservations[task_id]["payload_type"])
            self.assertEqual(1, reservations[task_id]["quantity"])

            post_reject_position = (simulator.response_east_m, simulator.response_north_m, simulator.response_heading_deg)
            clock[0] += 1.0
            simulator.status()
            self.assertNotEqual(post_reject_position, (simulator.response_east_m, simulator.response_north_m, simulator.response_heading_deg))

    def test_cancelled_payload_request_replay_does_not_reopen_task(self) -> None:
        simulator = elect_payload_task_with_local_estimate()
        task = payload_tasks(simulator)[0]
        task_id = str(task["task_id"])
        simulator.cancel_response(task_id)
        simulator.pop_outbound_claims()

        replay_request = {
            "scenario_run_id": str(DEFAULT_SCENARIO_SEED),
            "task_id": task_id,
            "task_type": TASK_TYPE_PAYLOAD_DROP,
            "contact_ref": task["contact_ref"],
            "signature_id": task["signature_id"],
            "object_label": task["object_label"],
            "claimant_drone_id": 1,
            "claimant_drone_name": "Red",
            "status": "reported",
            "task_state": "reported",
            "revision": int(task["revision"]) + 10,
            "assignment_epoch": int(task["assignment_epoch"]) + 10,
            "playbook_config_version": int(task["playbook_config_version"]) + 10,
            "requested_payload_type": task["requested_payload_type"],
            "requested_payload_quantity": task["requested_payload_quantity"],
            "estimated_east_m": task["estimated_east_m"],
            "estimated_north_m": task["estimated_north_m"],
            "estimated_latitude": task["estimated_latitude"],
            "estimated_longitude": task["estimated_longitude"],
            "election_deadline": task["election_deadline"],
            "created_at": task["created_at"],
            "updated_at": task["updated_at"],
            "cost": 0.0,
        }

        self.assertFalse(simulator.receive_claim(replay_request))
        self.assertEqual("cancelled", simulator.response_tasks[task_id]["status"])
        self.assertIsNone(simulator.active_response_task_id)
        self.assertEqual({}, simulator.control_payload()["payload_reservations"])
        self.assertEqual([], simulator.pop_outbound_claims())

    def test_simultaneous_isr_and_payload_tasks_share_one_contact(self) -> None:
        simulator = elect_payload_task_with_local_estimate()
        payload = payload_tasks(simulator)
        isr = isr_tasks(simulator)

        self.assertEqual(1, len(payload))
        self.assertEqual(1, len(isr))
        self.assertEqual(payload[0]["contact_ref"], isr[0]["contact_ref"])
        self.assertNotEqual(payload[0]["task_id"], isr[0]["task_id"])

    def test_sar_payload_request_requires_explicit_assistance_evidence(self) -> None:
        simulator = DroneSimulator(
            1,
            "Red",
            [],
            DEFAULT_SCENARIO_SEED,
            1,
            PLAYBOOK_OBSERVE_ORBIT,
            scenario_id=SAR_SUPPLY_SCENARIO_ID,
            scenario_run_id="payload-run",
            team_configuration=team_config(),
            payload_requests_enabled=True,
        )

        simulator.receive_observation(asdict(priority_observation(observation_id="priority-non-recipient")))
        self.assertEqual([], payload_tasks(simulator))

        simulator.receive_observation(asdict(sar_assistance_observation(observation_id="sar-recipient", payload_type="water", quantity=2)))
        payload = payload_tasks(simulator)
        self.assertEqual(1, len(payload))
        self.assertEqual("water", payload[0]["requested_payload_type"])
        self.assertEqual(2, payload[0]["requested_payload_quantity"])
        self.assertIn("assistance_request_evidence", payload[0])

    def test_legacy_priority_payload_request_behavior_preserved(self) -> None:
        simulator = DroneSimulator(
            1,
            "Red",
            [],
            DEFAULT_SCENARIO_SEED,
            1,
            PLAYBOOK_OBSERVE_ORBIT,
            scenario_run_id="payload-run",
            team_configuration=team_config(),
            payload_requests_enabled=True,
        )

        simulator.receive_observation(asdict(priority_observation(observation_id="legacy-priority")))
        payload = payload_tasks(simulator)
        self.assertEqual(1, len(payload))
        self.assertEqual(DEFAULT_PAYLOAD_TYPE, payload[0]["requested_payload_type"])

    def test_serialized_peer_dispatch_green_wins_moves_holds_blocks_and_cleans_up(self) -> None:
        publisher = DroneSimulator(
            1,
            "Red",
            [],
            DEFAULT_SCENARIO_SEED,
            1,
            PLAYBOOK_OBSERVE_ORBIT,
            scenario_run_id="payload-run",
            team_configuration=team_config(blue_payload=0, green_payload=2, payload_type="water"),
            payload_requests_enabled=True,
        )
        green = DroneSimulator(
            3,
            "Green",
            [],
            DEFAULT_SCENARIO_SEED,
            1,
            PLAYBOOK_OBSERVE_ORBIT,
            scenario_run_id="payload-run",
            team_configuration=team_config(blue_payload=0, green_payload=2, payload_type="water"),
            payload_requests_enabled=True,
            payload_standoff_m=25.0,
        )
        insufficient = DroneSimulator(
            3,
            "Green",
            [],
            DEFAULT_SCENARIO_SEED,
            1,
            PLAYBOOK_OBSERVE_ORBIT,
            scenario_run_id="payload-run",
            team_configuration=team_config(blue_payload=0, green_payload=1, payload_type="water"),
            payload_requests_enabled=True,
        )

        observation = sar_assistance_observation(
            observation_id="serialized-sar-water",
            payload_type="water",
            quantity=2,
            east_m=28.0,
            north_m=36.0,
        )
        self.assertTrue(publisher.receive_observation(asdict(observation)))
        self.assertTrue(green.receive_observation(asdict(observation)))
        self.assertTrue(insufficient.receive_observation(asdict(observation)))

        outbound = publisher.pop_outbound_claims()
        payload_message = next(message for message in outbound if message["task_type"] == TASK_TYPE_PAYLOAD_DROP)
        isr_message = next(message for message in outbound if message["task_type"] == TASK_TYPE_ISR_OBSERVE)

        clock = [green.last_wall_time]
        with patch("simulated_drone.time.monotonic", lambda: clock[0]):
            self.assertTrue(green.receive_claim(payload_message))
            self.assertTrue(insufficient.receive_claim(payload_message))
            self.assertTrue(green.receive_claim(isr_message))

            for simulation_time in (2.0, 3.6, 5.2):
                clock[0] += 0.1
                green.set_simulation_time(simulation_time)
                insufficient.set_simulation_time(simulation_time)
                green.status()
                insufficient.status()

        green_payload = payload_tasks(green)[0]
        task_id = str(green_payload["task_id"])
        self.assertEqual("Green", green_payload["assigned_drone_name"])
        self.assertIn(green_payload["status"], {"approaching_payload", "awaiting_local_acquisition"})
        self.assertEqual(2, green.control_payload()["payload_reservations"][task_id]["quantity"])
        self.assertEqual({"water": 2}, green.team_configuration.payload_inventory)
        self.assertLessEqual(sum(int(item["quantity"]) for item in green.control_payload()["payload_reservations"].values()), 2)
        self.assertEqual(task_id, green.active_response_task_id)

        before = distance_to_estimate(green, green.response_tasks[task_id])
        green._update_payload_guidance_locked(1.0)
        after = distance_to_estimate(green, green.response_tasks[task_id])
        self.assertLess(after, before)

        green.response_east_m = float(green.response_tasks[task_id]["estimated_east_m"]) - 26.0
        green.response_north_m = float(green.response_tasks[task_id]["estimated_north_m"])
        green._update_payload_guidance_locked(10.0)
        self.assertEqual("awaiting_local_acquisition", green.response_tasks[task_id]["status"])
        self.assertAlmostEqual(25.0, distance_to_estimate(green, green.response_tasks[task_id]), places=6)

        insufficient_payload = payload_tasks(insufficient)[0]
        self.assertNotIn(insufficient_payload["status"], {"approaching_payload", "awaiting_local_acquisition"})
        self.assertEqual({}, insufficient.control_payload()["payload_reservations"])

        isr = isr_tasks(publisher)
        self.assertEqual(1, len(isr))
        self.assertIn(isr[0]["status"], {"electing", "transitioning", "orbiting", "assigned_remote"})

        green.cancel_response(task_id)
        self.assertEqual({}, green.control_payload()["payload_reservations"])
        self.assertIsNone(green.active_response_task_id)

        green.receive_claim(payload_message)
        green.set_simulation_time(6.0)
        green.status()
        self.assertLessEqual(sum(int(item["quantity"]) for item in green.control_payload()["payload_reservations"].values()), 2)

        green.reset(scenario_run_id="payload-run-reset")
        self.assertEqual({}, green.control_payload()["payload_reservations"])

    def test_guidance_evidence_age_advances_only_with_accepted_contact_evidence(self) -> None:
        owner = elect_payload_task_with_local_estimate()
        task_id = owner.active_response_task_id or ""
        self.assertTrue(task_id)

        peer = DroneSimulator(
            3,
            "Green",
            [],
            DEFAULT_SCENARIO_SEED,
            1,
            PLAYBOOK_OBSERVE_ORBIT,
            scenario_run_id="payload-run",
            team_configuration=team_config(blue_payload=1),
            payload_requests_enabled=True,
        )

        clock = [owner.last_wall_time]
        with patch("simulated_drone.time.monotonic", lambda: clock[0]):
            owner._update_payload_guidance_locked(0.1)
            first_evidence_at = float(owner.response_tasks[task_id]["guidance_evidence_at"])
            heartbeat = owner._task_claim_payload_locked(
                owner.response_tasks[task_id],
                str(owner.response_tasks[task_id]["status"]),
            )

            self.assertTrue(peer.receive_claim(heartbeat))
            self.assertEqual(first_evidence_at, float(peer.response_tasks[task_id]["guidance_evidence_at"]))

            stale_heartbeat = dict(heartbeat)
            stale_heartbeat["revision"] = max(0, int(stale_heartbeat["revision"]) - 1)
            self.assertFalse(peer.receive_claim(stale_heartbeat))

            conflicting_heartbeat = dict(heartbeat)
            conflicting_heartbeat["revision"] = int(conflicting_heartbeat["revision"]) + 1
            conflicting_heartbeat["requested_payload_type"] = "water"
            self.assertFalse(peer.receive_claim(conflicting_heartbeat))
            self.assertTrue(peer.response_tasks[task_id].get("metadata_conflict"))

            first_evidence_age = owner.simulation_time - first_evidence_at
            clock[0] += 5.0
            owner.status()
            owner._update_payload_guidance_locked(0.1)
            second_evidence_at = float(owner.response_tasks[task_id]["guidance_evidence_at"])
            second_evidence_age = owner.simulation_time - second_evidence_at

            self.assertEqual(first_evidence_at, second_evidence_at)
            self.assertGreater(second_evidence_age, first_evidence_age)

            self.assertTrue(owner.receive_observation(asdict(priority_observation(
                observation_id="fresh-guidance-evidence",
                east_m=24.0,
                north_m=32.0,
                timestamp=first_evidence_at + 10.0,
            ))))
            owner._update_payload_guidance_locked(0.1)
            self.assertGreater(float(owner.response_tasks[task_id]["guidance_evidence_at"]), first_evidence_at)

    def test_real_http_heterogeneous_payload_allocation_and_approach(self) -> None:
        ports = [free_port(), free_port(), free_port()]
        script = Path(__file__).with_name("simulated_drone.py")
        processes: list[subprocess.Popen[str]] = []
        team_payload = {
            str(drone_id): configuration.to_dict()
            for drone_id, configuration in team_config(blue_payload=1).items()
        }

        with tempfile.NamedTemporaryFile("w", suffix=".json") as config_file:
            json.dump(team_payload, config_file)
            config_file.flush()

            commands: list[list[str]] = []

            try:
                for drone_id, name, port in zip((1, 2, 3), ("Red", "Blue", "Green"), ports, strict=True):
                    peers = [f"http://{HOST}:{peer_port}" for peer_port in ports if peer_port != port]
                    command = [
                        sys.executable,
                        str(script),
                        "--id",
                        str(drone_id),
                        "--name",
                        name,
                        "--host",
                        HOST,
                        "--port",
                        str(port),
                        "--peers",
                        *peers,
                        "--playbook",
                        PLAYBOOK_OBSERVE_ORBIT,
                        "--scenario-run-id",
                        str(DEFAULT_SCENARIO_SEED),
                        "--team-config",
                        config_file.name,
                        "--enable-payload-requests",
                        "--payload-type",
                        "rescue_kit",
                    ]
                    commands.append(command)
                    processes.append(
                        subprocess.Popen(
                            command,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                        ),
                    )

                for index, port in enumerate(ports):
                    if not wait_for_port(port) or processes[index].poll() is not None:
                        self.fail(format_process_startup_failure(commands, processes))

                for port in ports:
                    fetch_json(f"http://{HOST}:{port}/control", "POST", {"sharing_enabled": True})

                observation = asdict(priority_observation(
                    observation_id="http-payload-contact",
                    scenario_run_id=str(DEFAULT_SCENARIO_SEED),
                    east_m=25.0,
                    north_m=35.0,
                ))
                self.assertTrue(fetch_json(f"http://{HOST}:{ports[0]}/observations", "POST", observation)["accepted"])
                self.assertTrue(fetch_json(f"http://{HOST}:{ports[1]}/observations", "POST", observation)["accepted"])

                initial_blue_status = fetch_json(f"http://{HOST}:{ports[1]}/status")
                deadline = time.monotonic() + 12.0
                blue_tasks: list[dict[str, object]] = []
                red_isr_tasks: list[dict[str, object]] = []
                final_blue_status = initial_blue_status

                while time.monotonic() < deadline:
                    red_payload = fetch_json(f"http://{HOST}:{ports[0]}/control")
                    blue_payload = fetch_json(f"http://{HOST}:{ports[1]}/control")
                    final_blue_status = fetch_json(f"http://{HOST}:{ports[1]}/status")
                    blue_tasks = [
                        task
                        for task in blue_payload["response_tasks"]  # type: ignore[index]
                        if task["task_type"] == TASK_TYPE_PAYLOAD_DROP
                    ]
                    red_isr_tasks = [
                        task
                        for task in red_payload["response_tasks"]  # type: ignore[index]
                        if task["task_type"] == TASK_TYPE_ISR_OBSERVE
                    ]

                    if (
                        blue_tasks
                        and red_isr_tasks
                        and blue_tasks[0]["status"] in {"approaching_payload", "awaiting_local_acquisition"}
                        and red_isr_tasks[0]["status"] in {"transitioning", "orbiting"}
                        and blue_tasks[0]["contact_ref"] == red_isr_tasks[0]["contact_ref"]
                        and distance_from_status_to_task_estimate(final_blue_status, blue_tasks[0])
                        < distance_from_status_to_task_estimate(initial_blue_status, blue_tasks[0])
                    ):
                        break

                    time.sleep(0.2)

                self.assertEqual(1, len(blue_tasks))
                self.assertEqual(1, len(red_isr_tasks))
                self.assertEqual("Blue", blue_tasks[0]["assigned_drone_name"])
                self.assertEqual("Red", red_isr_tasks[0]["assigned_drone_name"])
                self.assertEqual(blue_tasks[0]["contact_ref"], red_isr_tasks[0]["contact_ref"])
                self.assertIn(blue_tasks[0]["status"], {"approaching_payload", "awaiting_local_acquisition"})
                self.assertIn(red_isr_tasks[0]["status"], {"transitioning", "orbiting"})
                self.assertLess(
                    distance_from_status_to_task_estimate(final_blue_status, blue_tasks[0]),
                    distance_from_status_to_task_estimate(initial_blue_status, blue_tasks[0]),
                )
            finally:
                for process in processes:
                    if process.poll() is None:
                        process.terminate()

                for process in processes:
                    with contextlib.suppress(subprocess.TimeoutExpired):
                        process.wait(timeout=3)

                for process in processes:
                    if process.poll() is None:
                        process.kill()

                close_process_pipes(processes)


if __name__ == "__main__":
    unittest.main()
