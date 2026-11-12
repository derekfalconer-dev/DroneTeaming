#!/usr/bin/env python3
"""Integration tests for real drone-to-drone observation propagation."""

from __future__ import annotations

import contextlib
import json
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.request
from pathlib import Path

from moving_tracking import (
    MOVING_TRACK_LOST_SECONDS,
    MOVING_TRACK_TRACKED_SECONDS,
    moving_person_position,
)

from simulated_drone import (
    APPLICATION_BUILD,
    DEFAULT_SCENARIO_ID,
    DEFAULT_SCENARIO_SEED,
    MOVING_HANDOFF_SCENARIO_ID,
    MOVING_TRACK_REPORT_TYPE,
    PLAYBOOK_REPORT_CONTINUE,
    SCENARIO_OBJECT_COUNT,
)


HOST = "127.0.0.1"
POSITION_TOLERANCE_DEG = 0.00002
HANDOFF_INITIAL_SAMPLE_TIMES = (12.0, 16.0, 20.0, 24.0)
HANDOFF_NEXT_SAMPLE_TIMES = (56.0, 60.0, 64.0, 68.0)


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

    with urllib.request.urlopen(request, timeout=2) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_for_stale_object(
    url: str,
    object_id: str,
    timeout_seconds: float = 20.0,
) -> dict[str, object]:
    deadline = time.monotonic() + timeout_seconds
    last_payload: dict[str, object] | None = None

    while time.monotonic() < deadline:
        try:
            payload = fetch_json(f"{url}/objects")
        except Exception as error:  # pragma: no cover - failure context path
            last_payload = {"error": repr(error)}
        else:
            last_payload = payload

            for item in payload["objects"]:  # type: ignore[index]
                if str(item["object_id"]) == object_id and bool(item.get("stale", False)):
                    return item  # type: ignore[return-value]

        time.sleep(0.2)

    raise AssertionError(f"Object {object_id} did not become stale at {url}; last_payload={last_payload!r}")


class LaunchSimulationPathTest(unittest.TestCase):
    processes: list[subprocess.Popen[str]]

    def setUp(self) -> None:
        self.dashboard_port = free_port()
        self.ports = [free_port(), free_port(), free_port()]
        self.urls = [f"http://{HOST}:{port}" for port in self.ports]
        launcher = Path(__file__).with_name("launch_simulation.py")
        self.processes = [
            subprocess.Popen(
                [
                    sys.executable,
                    str(launcher),
                    "--no-browser",
                    "--dashboard-host",
                    HOST,
                    "--dashboard-port",
                    str(self.dashboard_port),
                    "--drone-host",
                    HOST,
                    "--drone-ports",
                    *(str(port) for port in self.ports),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
            ),
        ]
        self.addCleanup(self.cleanup_processes)

        self.assertTrue(wait_for_port(self.dashboard_port), "Dashboard did not start")
        for port in self.ports:
            self.assertTrue(wait_for_port(port), f"Drone on port {port} did not start")

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

    def control_ack_failures(
        self,
        expected: dict[str, object],
        payloads: list[dict[str, object]],
    ) -> list[str]:
        failures = []

        for index, payload in enumerate(payloads):
            node_failures = []
            if payload["scenario_seed"] != expected["scenario_seed"]:
                node_failures.append(f"seed {payload['scenario_seed']}")
            if payload["scenario_id"] != expected["scenario_id"]:
                node_failures.append(f"scenario {payload['scenario_id']}")
            if payload["scenario_run_id"] != expected["scenario_run_id"]:
                node_failures.append(f"run {payload['scenario_run_id']}")
            if payload["initial_observer_id"] != expected["initial_observer_id"]:
                node_failures.append(f"initial observer {payload['initial_observer_id']}")
            if payload["next_observer_id"] != expected["next_observer_id"]:
                node_failures.append(f"next observer {payload['next_observer_id']}")
            if payload["selected_playbook"] != expected["selected_playbook"]:
                node_failures.append(f"playbook {payload['selected_playbook']}")
            if payload["paused"] != expected["paused"]:
                node_failures.append(f"paused {payload['paused']}")
            if payload["sharing_enabled"] != expected["sharing_enabled"]:
                node_failures.append(f"sharing {payload['sharing_enabled']}")

            if node_failures:
                failures.append(f"{['Red', 'Blue', 'Green'][index]}: {', '.join(node_failures)}")

        return failures

    def wait_for_control_ack(
        self,
        expected: dict[str, object],
        timeout_seconds: float = 8.0,
    ) -> list[dict[str, object]]:
        deadline = time.monotonic() + timeout_seconds
        last_payloads: list[dict[str, object]] = []
        failures: list[str] = []

        while time.monotonic() < deadline:
            last_payloads = [fetch_json(f"{url}/control") for url in self.urls]
            failures = self.control_ack_failures(expected, last_payloads)

            if not failures:
                return last_payloads

            time.sleep(0.2)

        raise AssertionError(
            f"Nodes did not acknowledge expected control state; failures={failures!r}; "
            f"last_payloads={last_payloads!r}",
        )

    def test_team_config_endpoint_and_control_ack_apply_actual_capabilities_and_inventory(self) -> None:
        team_configuration = {
            "1": {
                "role": "isr",
                "capabilities": ["atr_observe", "isr_observe"],
                "available": True,
                "payload_inventory": {},
                "camera_mount": "steerable",
            },
            "2": {
                "role": "payload",
                "capabilities": ["payload_drop"],
                "available": True,
                "payload_inventory": {"water": 4, "medical_supplies": 2},
                "camera_mount": "fixed",
            },
            "3": {
                "role": "isr",
                "capabilities": ["atr_observe", "isr_observe"],
                "available": True,
                "payload_inventory": {},
                "camera_mount": "steerable",
            },
        }
        saved = fetch_json(
            f"http://{HOST}:{self.dashboard_port}/team-config",
            method="POST",
            payload=team_configuration,
        )

        self.assertTrue(saved["saved"])
        self.assertEqual(team_configuration, saved["team_configuration"])

        payload = {
            "restart_scenario": True,
            "scenario_seed": DEFAULT_SCENARIO_SEED,
            "priority_count": 0,
            "playbook": PLAYBOOK_REPORT_CONTINUE,
            "scenario": DEFAULT_SCENARIO_ID,
            "initial_observer_id": 1,
            "next_observer_id": 2,
            "scenario_run_id": "integration-team-config-ack",
            "paused": True,
            "sharing_enabled": True,
            "team_configuration": team_configuration,
        }
        for url in self.urls:
            fetch_json(f"{url}/control", method="POST", payload=payload)

        self.wait_for_control_ack({
            "scenario_seed": DEFAULT_SCENARIO_SEED,
            "scenario_id": DEFAULT_SCENARIO_ID,
            "scenario_run_id": "integration-team-config-ack",
            "initial_observer_id": 1,
            "next_observer_id": 2,
            "selected_playbook": PLAYBOOK_REPORT_CONTINUE,
            "paused": True,
            "sharing_enabled": True,
        })

        blue_status = fetch_json(f"{self.urls[1]}/status")
        self.assertEqual("payload", blue_status["team_configuration"]["role"])
        self.assertEqual(["payload_drop"], blue_status["team_configuration"]["capabilities"])
        self.assertEqual({"water": 4, "medical_supplies": 2}, blue_status["team_configuration"]["payload_inventory"])

    def test_start_restart_seed_contract_and_mismatch_diagnostics(self) -> None:
        initial_control = fetch_json(f"{self.urls[0]}/control")
        applied_seed = initial_control["scenario_seed"]
        self.assertIsInstance(applied_seed, int)

        start_payload = {
            "restart_scenario": True,
            "scenario_seed": applied_seed,
            "priority_count": 0,
            "playbook": PLAYBOOK_REPORT_CONTINUE,
            "scenario": DEFAULT_SCENARIO_ID,
            "initial_observer_id": 1,
            "next_observer_id": 2,
            "scenario_run_id": "integration-seeded-start",
            "paused": False,
            "sharing_enabled": True,
        }
        for url in self.urls:
            fetch_json(f"{url}/control", method="POST", payload=start_payload)

        self.wait_for_control_ack({
            "scenario_seed": applied_seed,
            "scenario_id": DEFAULT_SCENARIO_ID,
            "scenario_run_id": "integration-seeded-start",
            "initial_observer_id": 1,
            "next_observer_id": 2,
            "selected_playbook": PLAYBOOK_REPORT_CONTINUE,
            "paused": False,
            "sharing_enabled": True,
        })

        restart_payload = {
            "restart_scenario": True,
            "scenario_run_id": "integration-seeded-restart",
            "paused": True,
            "sharing_enabled": True,
        }
        for url in self.urls:
            fetch_json(f"{url}/control", method="POST", payload=restart_payload)

        self.wait_for_control_ack({
            "scenario_seed": applied_seed,
            "scenario_id": DEFAULT_SCENARIO_ID,
            "scenario_run_id": "integration-seeded-restart",
            "initial_observer_id": 1,
            "next_observer_id": 2,
            "selected_playbook": PLAYBOOK_REPORT_CONTINUE,
            "paused": True,
            "sharing_enabled": True,
        })

        mismatch_payloads = [fetch_json(f"{url}/control") for url in self.urls]
        failures = self.control_ack_failures(
            {
                "scenario_seed": applied_seed + 1,
                "scenario_id": DEFAULT_SCENARIO_ID,
                "scenario_run_id": "integration-seeded-restart",
                "initial_observer_id": 1,
                "next_observer_id": 2,
                "selected_playbook": PLAYBOOK_REPORT_CONTINUE,
                "paused": True,
                "sharing_enabled": True,
            },
            mismatch_payloads,
        )
        self.assertTrue(failures)
        self.assertTrue(any("Red: seed" in failure for failure in failures), failures)

    def test_handoff_demo_configuration_acknowledges_on_all_processes_and_restart_preserves_it(self) -> None:
        dashboard_request = urllib.request.Request(f"http://{HOST}:{self.dashboard_port}/")
        with urllib.request.urlopen(dashboard_request, timeout=2) as response:
            dashboard_html = response.read().decode("utf-8")

        self.assertIn("handoff-demo-button", dashboard_html)
        self.assertIn("Run handoff demo", dashboard_html)
        self.assertIn("2026110601", dashboard_html)

        demo_payload = {
            "restart_scenario": True,
            "scenario_seed": 2026110601,
            "priority_count": 0,
            "playbook": PLAYBOOK_REPORT_CONTINUE,
            "scenario": MOVING_HANDOFF_SCENARIO_ID,
            "initial_observer_id": 3,
            "next_observer_id": 1,
            "scenario_run_id": "integration-handoff-demo-run-a",
            "paused": True,
            "sharing_enabled": True,
        }
        for url in self.urls:
            fetch_json(f"{url}/control", method="POST", payload=demo_payload)

        acknowledged = self.wait_for_control_ack({
            "scenario_seed": 2026110601,
            "scenario_id": MOVING_HANDOFF_SCENARIO_ID,
            "scenario_run_id": "integration-handoff-demo-run-a",
            "initial_observer_id": 3,
            "next_observer_id": 1,
            "selected_playbook": PLAYBOOK_REPORT_CONTINUE,
            "paused": True,
            "sharing_enabled": True,
        })
        self.assertEqual(3, len(acknowledged))
        self.assertTrue(all(payload["observation_count"] == 0 for payload in acknowledged))
        self.assertTrue(all(payload["object_count"] == 0 for payload in acknowledged))
        self.assertTrue(all(payload["priority_object_count"] == 0 for payload in acknowledged))

        for url in self.urls:
            fetch_json(f"{url}/control", method="POST", payload={"paused": False, "sharing_enabled": True})

        time.sleep(0.8)
        running_status = fetch_json(f"{self.urls[0]}/status")
        self.assertFalse(running_status["paused"])
        self.assertTrue(running_status["sharing_enabled"])
        self.assertGreater(float(running_status["timestamp"]), 0.0)

        restart_payload = {
            "restart_scenario": True,
            "scenario_run_id": "integration-handoff-demo-run-b",
            "paused": True,
            "sharing_enabled": True,
        }
        for url in self.urls:
            fetch_json(f"{url}/control", method="POST", payload=restart_payload)

        restarted = self.wait_for_control_ack({
            "scenario_seed": 2026110601,
            "scenario_id": MOVING_HANDOFF_SCENARIO_ID,
            "scenario_run_id": "integration-handoff-demo-run-b",
            "initial_observer_id": 3,
            "next_observer_id": 1,
            "selected_playbook": PLAYBOOK_REPORT_CONTINUE,
            "paused": True,
            "sharing_enabled": True,
        })
        self.assertTrue(all(payload["observation_count"] == 0 for payload in restarted))
        self.assertTrue(all(payload["object_count"] == 0 for payload in restarted))
        self.assertEqual("integration-handoff-demo-run-b", fetch_json(f"{self.urls[0]}/truth")["scenario_run_id"])

    def test_normal_launcher_path_serves_moving_seeded_scenario(self) -> None:
        dashboard_request = urllib.request.Request(f"http://{HOST}:{self.dashboard_port}/")
        with urllib.request.urlopen(dashboard_request, timeout=2) as response:
            dashboard_html = response.read().decode("utf-8")
            self.assertEqual("no-store", response.headers["Cache-Control"])

        self.assertIn("New random selected scenario", dashboard_html)
        self.assertIn("Random Search of Area", dashboard_html)
        self.assertIn('id="compact-replay-button"', dashboard_html)
        self.assertIn("Restart current scenario", dashboard_html)

        truth = fetch_json(f"{self.urls[0]}/truth")
        self.assertEqual(APPLICATION_BUILD, truth["app_build"])
        self.assertEqual(SCENARIO_OBJECT_COUNT, truth["object_count"])
        first_truth_objects = truth["objects"]

        first_status = fetch_json(f"{self.urls[0]}/status")
        time.sleep(0.5)
        second_status = fetch_json(f"{self.urls[0]}/status")
        self.assertGreater(float(second_status["timestamp"]), float(first_status["timestamp"]))
        self.assertNotEqual(first_status["latitude"], second_status["latitude"])
        self.assertNotEqual(first_status["camera_footprint"], second_status["camera_footprint"])

        fetch_json(f"{self.urls[0]}/control", method="POST", payload={"paused": True})
        paused_status = fetch_json(f"{self.urls[0]}/status")
        time.sleep(0.5)
        still_paused_status = fetch_json(f"{self.urls[0]}/status")
        self.assertAlmostEqual(
            float(paused_status["timestamp"]),
            float(still_paused_status["timestamp"]),
            delta=0.000001,
        )
        self.assertEqual(paused_status["latitude"], still_paused_status["latitude"])
        fetch_json(f"{self.urls[0]}/control", method="POST", payload={"paused": False})

        fetch_json(f"{self.urls[0]}/control", method="POST", payload={"restart_scenario": True})
        restarted_truth = fetch_json(f"{self.urls[0]}/truth")
        self.assertEqual(DEFAULT_SCENARIO_ID, restarted_truth["scenario_id"])
        self.assertEqual(DEFAULT_SCENARIO_SEED, restarted_truth["scenario_seed"])
        self.assertEqual(first_truth_objects, restarted_truth["objects"])

        fetch_json(f"{self.urls[0]}/control", method="POST", payload={"scenario_seed": 987654321})
        new_truth = fetch_json(f"{self.urls[0]}/truth")
        self.assertEqual(987654321, new_truth["scenario_seed"])
        self.assertNotEqual(first_truth_objects, new_truth["objects"])


class PeerPropagationTest(unittest.TestCase):
    processes: list[subprocess.Popen[str]]
    ports: list[int]
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
            fetch_json(
                f"{url}/control",
                method="POST",
                payload={"reset": True, "scenario_run_id": "peer-propagation-shared-run", "sharing_enabled": False},
            )

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

        for process in self.processes:
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=5)

    def wait_for_owned_object(self, url: str, timeout_seconds: float = 5.0) -> dict[str, object]:
        deadline = time.monotonic() + timeout_seconds
        last_payload: dict[str, object] | None = None

        while time.monotonic() < deadline:
            try:
                payload = fetch_json(f"{url}/objects")
            except Exception as error:  # pragma: no cover - failure context path
                last_payload = {"error": repr(error)}
            else:
                last_payload = payload

                for item in payload["objects"]:  # type: ignore[index]
                    if bool(item["own_sensor"]):
                        return item  # type: ignore[return-value]

            time.sleep(0.1)

        self.fail(
            f"No locally observed object appeared in {url}/objects; "
            f"{self.failure_context(url, last_payload)}",
        )

    def assert_observation_absent(self, url: str, observation_id: str) -> None:
        payload = fetch_json(f"{url}/objects")

        for item in payload["objects"]:  # type: ignore[index]
            observation_ids = {
                str(item_observation_id)
                for item_observation_id in item["observation_ids"]  # type: ignore[index]
            }
            self.assertNotIn(observation_id, observation_ids)

    def diagnostics(self, url: str) -> dict[str, object]:
        status = fetch_json(f"{url}/status")
        return status["diagnostics"]  # type: ignore[return-value]

    def failure_context(self, url: str, last_payload: dict[str, object] | None) -> str:
        try:
            diagnostics = self.diagnostics(url)
        except Exception as error:  # pragma: no cover - failure context path
            diagnostics = {"error": repr(error)}

        return f"last_objects_response={last_payload!r}; diagnostics={diagnostics!r}"

    def wait_for_object(
        self,
        url: str,
        object_id: str,
        timeout_seconds: float = 5.0,
    ) -> dict[str, object]:
        deadline = time.monotonic() + timeout_seconds
        last_payload: dict[str, object] | None = None

        while time.monotonic() < deadline:
            try:
                payload = fetch_json(f"{url}/objects")
            except Exception as error:  # pragma: no cover - failure context path
                last_payload = {"error": repr(error)}
            else:
                last_payload = payload

                for item in payload["objects"]:  # type: ignore[index]
                    if str(item["object_id"]) == object_id:
                        return item  # type: ignore[return-value]

            time.sleep(0.1)

        self.fail(
            f"Object {object_id} did not appear in {url}/objects; "
            f"{self.failure_context(url, last_payload)}",
        )

    def wait_for_new_observation_id(
        self,
        url: str,
        object_id: str,
        previous_observation_ids: set[str],
        timeout_seconds: float = 5.0,
    ) -> str:
        deadline = time.monotonic() + timeout_seconds
        last_payload: dict[str, object] | None = None
        last_object: dict[str, object] | None = None

        while time.monotonic() < deadline:
            try:
                payload = fetch_json(f"{url}/objects")
            except Exception as error:  # pragma: no cover - failure context path
                last_payload = {"error": repr(error)}
            else:
                last_payload = payload

                for item in payload["objects"]:  # type: ignore[index]
                    if str(item["object_id"]) != object_id:
                        continue

                    last_object = item  # type: ignore[assignment]
                    observation_ids = {
                        str(observation_id)
                        for observation_id in item["observation_ids"]  # type: ignore[index]
                    }
                    new_observation_ids = observation_ids - previous_observation_ids

                    if new_observation_ids:
                        return sorted(new_observation_ids)[0]

            time.sleep(0.1)

        self.fail(
            f"No new observation id appeared for {object_id} in {url}/objects; "
            f"last_object={last_object!r}; {self.failure_context(url, last_payload)}",
        )

    def wait_for_observation_id(
        self,
        url: str,
        observation_id: str,
        timeout_seconds: float = 5.0,
    ) -> dict[str, object]:
        deadline = time.monotonic() + timeout_seconds
        last_payload: dict[str, object] | None = None

        while time.monotonic() < deadline:
            try:
                payload = fetch_json(f"{url}/objects")
            except Exception as error:  # pragma: no cover - failure context path
                last_payload = {"error": repr(error)}
            else:
                last_payload = payload

                for item in payload["objects"]:  # type: ignore[index]
                    observation_ids = {
                        str(item_observation_id)
                        for item_observation_id in item["observation_ids"]  # type: ignore[index]
                    }

                    if observation_id in observation_ids:
                        return item  # type: ignore[return-value]

            time.sleep(0.1)

        self.fail(
            f"{observation_id} did not appear in {url}/objects; "
            f"{self.failure_context(url, last_payload)}",
        )

    def url_for_drone_id(self, drone_id: int) -> str:
        return self.urls[drone_id - 1]

    def set_all_simulation_times(self, simulation_time: float) -> None:
        for url in self.urls:
            fetch_json(f"{url}/control", method="POST", payload={"simulation_time": simulation_time})

    def run_handoff_samples(self, sample_times: tuple[float, ...]) -> None:
        for sample_time in sample_times:
            self.set_all_simulation_times(sample_time)
            time.sleep(0.35)
            self.set_all_simulation_times(sample_time + 1.25)
            time.sleep(0.25)

    def moving_objects(self, url: str) -> list[dict[str, object]]:
        return [
            item
            for item in fetch_json(f"{url}/objects")["objects"]  # type: ignore[index]
            if item.get("tracking_model") == "constant_velocity"
        ]

    def wait_for_moving_objects(
        self,
        required_sources: set[int] | None = None,
        required_status: str | None = None,
        timeout_seconds: float = 8.0,
    ) -> dict[str, list[dict[str, object]]]:
        deadline = time.monotonic() + timeout_seconds
        last_by_url: dict[str, list[dict[str, object]]] = {}

        while time.monotonic() < deadline:
            last_by_url = {url: self.moving_objects(url) for url in self.urls}
            matches = True

            for objects in last_by_url.values():
                if len(objects) != 1:
                    matches = False
                    break

                item = objects[0]
                sources = {int(source) for source in item["historical_sources"]}  # type: ignore[index]
                if required_sources is not None and not required_sources.issubset(sources):
                    matches = False
                    break

                if required_status is not None and item.get("track_status") != required_status:
                    matches = False
                    break

            if matches:
                return last_by_url

            time.sleep(0.2)

        self.fail(f"Moving objects did not converge; required_sources={required_sources}; required_status={required_status}; last_by_url={last_by_url!r}")

    def assert_moving_accuracy(self, objects_by_url: dict[str, list[dict[str, object]]], simulation_time: float) -> None:
        truth_east_m, truth_north_m = moving_person_position(simulation_time)

        for url, objects in objects_by_url.items():
            item = objects[0]
            position_error_m = ((float(item["predicted_east_m"]) - truth_east_m) ** 2 + (float(item["predicted_north_m"]) - truth_north_m) ** 2) ** 0.5
            velocity_error_mps = ((float(item["velocity_east_mps"]) - 4.0) ** 2 + float(item["velocity_north_mps"]) ** 2) ** 0.5
            self.assertLess(position_error_m, 120.0, f"{url} position_error_m={position_error_m:.3f}; object={item!r}")
            self.assertLess(velocity_error_mps, 14.0, f"{url} velocity_error_mps={velocity_error_mps:.3f}; object={item!r}")

    def wait_for_multisource_object(
        self,
        url: str,
        required_sources: set[int],
        timeout_seconds: float = 15.0,
    ) -> dict[str, object]:
        deadline = time.monotonic() + timeout_seconds
        last_payload: dict[str, object] | None = None

        while time.monotonic() < deadline:
            try:
                payload = fetch_json(f"{url}/objects")
            except Exception as error:  # pragma: no cover - failure context path
                last_payload = {"error": repr(error)}
            else:
                last_payload = payload

                for item in payload["objects"]:  # type: ignore[index]
                    sources = {int(source) for source in item["sources"]}  # type: ignore[index]

                    if required_sources.issubset(sources):
                        matching_objects = [
                            candidate
                            for candidate in payload["objects"]  # type: ignore[index]
                            if str(candidate["object_id"]) == str(item["object_id"])
                        ]
                        self.assertEqual(1, len(matching_objects))
                        return item  # type: ignore[return-value]

            time.sleep(0.1)

        self.fail(
            f"No object with sources {sorted(required_sources)} appeared in {url}/objects; "
            f"{self.failure_context(url, last_payload)}",
        )

    def wait_for_duplicate_count(
        self,
        url: str,
        minimum_duplicate_count: int,
        timeout_seconds: float = 5.0,
    ) -> None:
        deadline = time.monotonic() + timeout_seconds
        last_diagnostics: dict[str, object] | None = None

        while time.monotonic() < deadline:
            try:
                diagnostics = self.diagnostics(url)
            except Exception as error:  # pragma: no cover - failure context path
                last_diagnostics = {"error": repr(error)}
            else:
                last_diagnostics = diagnostics

                if int(diagnostics["duplicates_ignored"]) >= minimum_duplicate_count:
                    return

            time.sleep(0.1)

        self.fail(
            f"{url} did not reach duplicate count {minimum_duplicate_count}; "
            f"last_diagnostics={last_diagnostics!r}",
        )

    def assert_red_provenance(
        self,
        object_map: dict[str, object],
        expected_label: str,
        expected_latitude: float,
        expected_longitude: float,
    ) -> None:
        self.assertTrue(object_map["teammate_sourced"])
        self.assertIn("Red", object_map["historical_source_names"])
        self.assertEqual(object_map["label"], expected_label)
        self.assertLess(abs(float(object_map["latitude"]) - expected_latitude), POSITION_TOLERANCE_DEG * 80)
        self.assertLess(abs(float(object_map["longitude"]) - expected_longitude), POSITION_TOLERANCE_DEG * 80)

    def test_all_directed_observer_pairs_complete_real_http_moving_handoff(self) -> None:
        pairs = [
            (initial_observer_id, next_observer_id)
            for initial_observer_id in (1, 2, 3)
            for next_observer_id in (1, 2, 3)
            if initial_observer_id != next_observer_id
        ]

        for initial_observer_id, next_observer_id in pairs:
            with self.subTest(initial_observer_id=initial_observer_id, next_observer_id=next_observer_id):
                run_id = f"peer-http-handoff-{initial_observer_id}-{next_observer_id}-{time.time_ns()}"
                for url in self.urls:
                    fetch_json(
                        f"{url}/control",
                        method="POST",
                        payload={
                            "restart_scenario": True,
                            "scenario_seed": DEFAULT_SCENARIO_SEED,
                            "priority_count": 0,
                            "playbook": PLAYBOOK_REPORT_CONTINUE,
                            "scenario": MOVING_HANDOFF_SCENARIO_ID,
                            "initial_observer_id": initial_observer_id,
                            "next_observer_id": next_observer_id,
                            "scenario_run_id": run_id,
                            "sharing_enabled": True,
                            "paused": False,
                            "simulation_time": 0.0,
                        },
                    )

                self.run_handoff_samples(HANDOFF_INITIAL_SAMPLE_TIMES)
                after_initial = self.wait_for_moving_objects({initial_observer_id}, "tracked")
                initial_ids = {
                    url: str(objects[0]["local_object_id"])
                    for url, objects in after_initial.items()
                }
                third_drone_id = next(iter({1, 2, 3} - {initial_observer_id, next_observer_id}))

                for drone_id in (1, 2, 3):
                    item = after_initial[self.url_for_drone_id(drone_id)][0]
                    sources = {int(source) for source in item["historical_sources"]}  # type: ignore[index]
                    self.assertEqual({initial_observer_id}, sources, f"drone_id={drone_id}; item={item!r}")

                    if drone_id == initial_observer_id:
                        self.assertTrue(item["own_sensor"], f"initial observer should own the first local reports; item={item!r}")
                    else:
                        self.assertFalse(item["own_sensor"], f"non-initial observer should only have remote initial reports; item={item!r}")
                        self.assertTrue(item["teammate_sourced"], f"non-initial observer should retain teammate-sourced track; item={item!r}")

                self.run_handoff_samples(HANDOFF_NEXT_SAMPLE_TIMES)
                after_handoff = self.wait_for_moving_objects({initial_observer_id, next_observer_id}, "tracked")
                self.assert_moving_accuracy(after_handoff, HANDOFF_NEXT_SAMPLE_TIMES[-1] + 1.25)

                for url, objects in after_handoff.items():
                    item = objects[0]
                    self.assertEqual(initial_ids[url], str(item["local_object_id"]))
                    source_track_ids = list(item["source_track_ids"])  # type: ignore[arg-type]
                    self.assertGreaterEqual(len(source_track_ids), 2, f"{url} source_track_ids={source_track_ids!r}")
                    self.assertEqual(len(source_track_ids), len(set(source_track_ids)))
                    self.assertIn(initial_observer_id, {int(source) for source in item["historical_sources"]})  # type: ignore[index]
                    self.assertIn(next_observer_id, {int(source) for source in item["historical_sources"]})  # type: ignore[index]

                third_url = self.url_for_drone_id(third_drone_id)
                third_object = after_handoff[third_url][0]
                self.assertFalse(third_object["own_sensor"])
                self.assertTrue(third_object["teammate_sourced"])

                coast_time = HANDOFF_NEXT_SAMPLE_TIMES[-1] + MOVING_TRACK_TRACKED_SECONDS + 1.0
                self.set_all_simulation_times(coast_time)
                coasting = self.wait_for_moving_objects({initial_observer_id, next_observer_id}, "coasting")
                self.assertEqual(initial_ids, {url: str(objects[0]["local_object_id"]) for url, objects in coasting.items()})

                lost_time = HANDOFF_NEXT_SAMPLE_TIMES[-1] + MOVING_TRACK_LOST_SECONDS + 3.0
                self.set_all_simulation_times(lost_time)
                lost = self.wait_for_moving_objects({initial_observer_id, next_observer_id}, "lost")
                self.assertEqual(initial_ids, {url: str(objects[0]["local_object_id"]) for url, objects in lost.items()})

                old_report = next(
                    measurement["raw_observation"]
                    for measurement in lost[self.urls[0]][0]["measurements"]  # type: ignore[index]
                    if measurement["measurement_type"] == MOVING_TRACK_REPORT_TYPE
                )
                replay_run_id = f"{run_id}-replay"
                for url in self.urls:
                    fetch_json(
                        f"{url}/control",
                        method="POST",
                        payload={
                            "restart_scenario": True,
                            "scenario_run_id": replay_run_id,
                            "sharing_enabled": True,
                            "paused": False,
                            "simulation_time": 0.0,
                        },
                    )

                rejected = fetch_json(f"{self.urls[0]}/observations", method="POST", payload=old_report)
                self.assertFalse(rejected["accepted"])
                self.assertEqual([], self.moving_objects(self.urls[0]))

                self.run_handoff_samples((HANDOFF_INITIAL_SAMPLE_TIMES[0],))
                replay = self.wait_for_moving_objects({initial_observer_id}, "tracked")
                replay_measurements = replay[self.urls[0]][0]["measurements"]  # type: ignore[index]
                self.assertTrue(any(measurement["scenario_run_id"] == replay_run_id for measurement in replay_measurements))

    def test_red_and_blue_common_object_converges_to_one_multisource_map_object(self) -> None:
        red_url, blue_url, green_url = self.urls

        for url in self.urls:
            fetch_json(f"{url}/control", method="POST", payload={"sharing_enabled": True})

        red_object = self.wait_for_multisource_object(red_url, {1, 2}, timeout_seconds=20.0)
        blue_object = self.wait_for_multisource_object(blue_url, {1, 2}, timeout_seconds=20.0)
        green_object = self.wait_for_multisource_object(green_url, {1, 2}, timeout_seconds=20.0)

        self.assertNotEqual(str(red_object["object_id"]), str(blue_object["object_id"]))
        self.assertNotEqual(str(red_object["object_id"]), str(green_object["object_id"]))
        self.assertIn("Red", red_object["historical_source_names"])
        self.assertIn("Blue", red_object["historical_source_names"])
        self.assertIn("Red", green_object["historical_source_names"])
        self.assertIn("Blue", green_object["historical_source_names"])

        green_observation_count = int(green_object["observation_count"])
        time.sleep(1.5)
        refreshed_green_object = self.wait_for_multisource_object(green_url, {1, 2}, timeout_seconds=5.0)
        self.assertGreaterEqual(int(refreshed_green_object["observation_count"]), green_observation_count)
        matching_green_objects = [
            item
            for item in fetch_json(f"{green_url}/objects")["objects"]  # type: ignore[index]
            if str(item["object_id"]) == str(green_object["object_id"])
        ]
        self.assertEqual(1, len(matching_green_objects))

    def test_red_observation_propagates_after_sharing_and_deduplicates(self) -> None:
        red_url, blue_url, green_url = self.urls

        red_object = self.wait_for_owned_object(red_url)
        self.assertTrue(red_object["own_sensor"])
        red_object_id = str(red_object["object_id"])
        red_label = str(red_object["label"])
        red_latitude = float(red_object["latitude"])
        red_longitude = float(red_object["longitude"])
        initial_red_observation_ids = {
            str(observation_id)
            for observation_id in red_object["observation_ids"]  # type: ignore[index]
        }

        for observation_id in initial_red_observation_ids:
            self.assert_observation_absent(blue_url, observation_id)
            self.assert_observation_absent(green_url, observation_id)

        for url in self.urls:
            fetch_json(f"{url}/control", method="POST", payload={"sharing_enabled": True})

        shared_observation_id = self.wait_for_new_observation_id(
            red_url,
            red_object_id,
            initial_red_observation_ids,
            timeout_seconds=10.0,
        )
        blue_object = self.wait_for_observation_id(blue_url, shared_observation_id)
        green_object = self.wait_for_observation_id(green_url, shared_observation_id)

        self.assert_red_provenance(blue_object, red_label, red_latitude, red_longitude)
        self.assert_red_provenance(green_object, red_label, red_latitude, red_longitude)
        self.assertNotEqual(blue_object["local_object_id"], green_object["local_object_id"])

        blue_diagnostics = self.diagnostics(blue_url)
        green_diagnostics = self.diagnostics(green_url)
        self.assertGreaterEqual(int(blue_diagnostics["peer_observations_received"]), 1)
        self.assertGreaterEqual(int(green_diagnostics["peer_observations_received"]), 1)

        for url in self.urls:
            fetch_json(f"{url}/control", method="POST", payload={"paused": True})

        before_blue = self.wait_for_observation_id(blue_url, shared_observation_id)
        before_green = self.wait_for_observation_id(green_url, shared_observation_id)
        before_blue_count = int(before_blue["observation_count"])
        before_green_count = int(before_green["observation_count"])
        before_blue_evidence = list(before_blue["observation_ids"])  # type: ignore[arg-type]
        before_green_evidence = list(before_green["observation_ids"])  # type: ignore[arg-type]
        before_blue_duplicates = int(self.diagnostics(blue_url)["duplicates_ignored"])
        before_green_duplicates = int(self.diagnostics(green_url)["duplicates_ignored"])

        fetch_json(
            f"{red_url}/control",
            method="POST",
            payload={"redeliver_observation_id": shared_observation_id},
        )
        fetch_json(
            f"{red_url}/control",
            method="POST",
            payload={"redeliver_observation_id": shared_observation_id},
        )

        self.wait_for_duplicate_count(blue_url, before_blue_duplicates + 2)
        self.wait_for_duplicate_count(green_url, before_green_duplicates + 2)

        after_blue = self.wait_for_observation_id(blue_url, shared_observation_id)
        after_green = self.wait_for_observation_id(green_url, shared_observation_id)

        self.assertEqual(before_blue_count, int(after_blue["observation_count"]))
        self.assertEqual(before_green_count, int(after_green["observation_count"]))
        self.assertEqual(before_blue_evidence, list(after_blue["observation_ids"]))  # type: ignore[arg-type]
        self.assertEqual(before_green_evidence, list(after_green["observation_ids"]))  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
