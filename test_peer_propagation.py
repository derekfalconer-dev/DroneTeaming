#!/usr/bin/env python3
"""Integration tests for real drone-to-drone observation propagation."""

from __future__ import annotations

import contextlib
import json
import socket
import subprocess
import sys
import time
import unittest
import urllib.request
from pathlib import Path

from simulated_drone import APPLICATION_BUILD, SCENARIO_OBJECT_COUNT


HOST = "127.0.0.1"
POSITION_TOLERANCE_DEG = 0.00002


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

    def test_normal_launcher_path_serves_moving_seeded_scenario(self) -> None:
        dashboard_request = urllib.request.Request(f"http://{HOST}:{self.dashboard_port}/")
        with urllib.request.urlopen(dashboard_request, timeout=2) as response:
            dashboard_html = response.read().decode("utf-8")
            self.assertEqual("no-store", response.headers["Cache-Control"])

        self.assertIn("New random scenario", dashboard_html)
        self.assertIn("Restart scenario", dashboard_html)

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
            fetch_json(f"{url}/control", method="POST", payload={"reset": True})

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
