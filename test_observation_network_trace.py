#!/usr/bin/env python3
"""Regression tests for observation tracing and peer-network instrumentation."""

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
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if payload is not None else {},
        method=method,
    )

    with urllib.request.urlopen(request, timeout=3) as response:
        return json.loads(response.read().decode("utf-8"))


def synthetic_observation(
    observation_id: str,
    source_drone_id: int = 1,
    source_drone_name: str = "Red",
    timestamp: float = 1.0,
) -> Observation:
    latitude, longitude = local_m_to_lat_lon(25.0, 35.0)
    color = {1: "red", 2: "blue", 3: "green"}[source_drone_id]
    return Observation(
        id=observation_id,
        scenario_run_id=str(DEFAULT_SCENARIO_SEED),
        source_drone_id=source_drone_id,
        source_drone_name=source_drone_name,
        source_color=color,
        sensor_id=f"eo-{color}",
        source_track_id=f"trace-track-{observation_id}",
        timestamp=timestamp,
        measurement_frame=MEASUREMENT_FRAME,
        east_m=25.0,
        north_m=35.0,
        latitude=latitude,
        longitude=longitude,
        label="vehicle",
        classification_score=0.91,
        covariance=[[64.0, 0.0], [0.0, 36.0]],
        object_id=f"test:{observation_id}",
    )


class ObservationNetworkTraceTest(unittest.TestCase):
    def test_inspectable_object_contains_original_observation_json(self) -> None:
        simulator = DroneSimulator(2, "Blue", [])
        observation = synthetic_observation("inspectable-raw-json")
        self.assertTrue(simulator.receive_observation(asdict(observation)))
        objects = simulator.objects_payload()["objects"]
        self.assertEqual(1, len(objects))
        measurement = objects[0]["measurements"][0]  # type: ignore[index]
        self.assertEqual(measurement["observation_id"], observation.id)
        self.assertEqual(measurement["raw_observation"]["id"], observation.id)
        self.assertEqual(measurement["raw_observation"]["sensor_id"], observation.sensor_id)
        self.assertEqual(measurement["raw_observation"]["classification_score"], observation.classification_score)

    def test_real_peer_receipt_and_payload_bytes_are_counted(self) -> None:
        receiver = DroneSimulator(2, "Blue", [])
        observation = synthetic_observation("payload-accounting")
        payload = asdict(observation)
        body = json.dumps(payload).encode("utf-8")

        self.assertTrue(receiver.receive_observation(payload, len(body)))

        network = receiver.network_payload()
        self.assertEqual(network["observation_messages_received"], 1)
        self.assertEqual(network["payload_bytes_received"], len(body))
        self.assertEqual(network["observation_messages_sent"], 0)

    def test_duplicate_delivery_is_distinguishable_from_new_accepted_report(self) -> None:
        receiver = DroneSimulator(2, "Blue", [])
        observation = synthetic_observation("duplicate-trace")
        payload = asdict(observation)
        body = json.dumps(payload).encode("utf-8")

        self.assertTrue(receiver.receive_observation(payload, len(body)))
        first_trace = receiver.trace_observation_payload(observation.id)
        self.assertEqual(first_trace["status"], "accepted")

        self.assertFalse(receiver.receive_observation(payload, len(body)))
        duplicate_trace = receiver.trace_observation_payload(observation.id)
        self.assertEqual(duplicate_trace["status"], "duplicate_ignored")

        network = receiver.network_payload()
        self.assertEqual(network["observation_messages_received"], 2)
        self.assertEqual(network["duplicate_observations_ignored"], 1)
        self.assertEqual(network["payload_bytes_received"], len(body) * 2)

    def test_trace_history_is_bounded_and_reports_eviction(self) -> None:
        simulator = DroneSimulator(1, "Red", [])

        for index in range(300):
            observation = synthetic_observation(f"bounded-{index}", timestamp=float(index))
            simulator.receive_observation(asdict(observation))

        network = simulator.network_payload()
        self.assertLessEqual(network["trace_retained_count"], network["trace_retention_limit"])
        self.assertGreater(network["trace_evictions"], 0)

        evicted_trace = simulator.trace_observation_payload("bounded-0")
        self.assertIsNone(evicted_trace["record"])
        self.assertEqual(evicted_trace["status"], "not_seen_or_no_longer_retained")

    def test_three_real_processes_trace_same_observation_and_dashboard_polling_does_not_inflate_peer_counters(self) -> None:
        ports = [free_port(), free_port(), free_port()]
        script = Path(__file__).with_name("simulated_drone.py")
        processes: list[subprocess.Popen[str]] = []

        try:
            for drone_id, name, port in zip((1, 2, 3), ("Red", "Blue", "Green"), ports, strict=True):
                peers = [f"http://{HOST}:{peer_port}" for peer_port in ports if peer_port != port]
                processes.append(
                    subprocess.Popen(
                        [
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
                        ],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        text=True,
                    ),
                )

            for port in ports:
                self.assertTrue(wait_for_port(port), f"drone on port {port} did not start")

            for port in ports:
                fetch_json(f"http://{HOST}:{port}/control", "POST", {"sharing_enabled": True})

            observation_id = ""
            deadline = time.monotonic() + 10.0

            while time.monotonic() < deadline and not observation_id:
                payload = fetch_json(f"http://{HOST}:{ports[0]}/objects")
                for item in payload["objects"]:  # type: ignore[index]
                    measurements = item.get("measurements", [])
                    if measurements:
                        observation_id = str(measurements[0]["observation_id"])
                        break
                time.sleep(0.2)

            self.assertTrue(observation_id)

            traces: list[dict[str, object]] = []
            deadline = time.monotonic() + 10.0

            while time.monotonic() < deadline:
                traces = [
                    fetch_json(f"http://{HOST}:{port}/trace?observation_id={observation_id}")
                    for port in ports
                ]
                if sum(1 for trace in traces if trace["record"] is not None) == 3:
                    break
                time.sleep(0.2)

            self.assertEqual(3, sum(1 for trace in traces if trace["record"] is not None))
            origin_trace = next(trace for trace in traces if trace["record"] and trace["record"]["originated_locally"])
            self.assertTrue(origin_trace["record"]["originated_locally"])
            origin_port = ports[int(origin_trace["drone_id"]) - 1]

            for port in ports:
                fetch_json(f"http://{HOST}:{port}/control", "POST", {"paused": True})

            before = fetch_json(f"http://{HOST}:{origin_port}/network")
            fetch_json(f"http://{HOST}:{origin_port}/status")
            fetch_json(f"http://{HOST}:{origin_port}/objects")
            fetch_json(f"http://{HOST}:{origin_port}/network")
            after = fetch_json(f"http://{HOST}:{origin_port}/network")

            self.assertGreaterEqual(after["observation_messages_sent"], 1)
            self.assertIn("send_failures", after)
            self.assertEqual(before["payload_bytes_sent"], after["payload_bytes_sent"])
            self.assertEqual(before["payload_bytes_received"], after["payload_bytes_received"])

            time.sleep(0.2)
            paused_network = fetch_json(f"http://{HOST}:{ports[0]}/network")
            self.assertEqual(paused_network["wall_time_source"], "time.monotonic")
            self.assertTrue(
                any(
                    peer["last_rtt_ms"] is not None
                    for peer in paused_network["peer_connection_state"]  # type: ignore[index]
                ),
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


if __name__ == "__main__":
    unittest.main()
