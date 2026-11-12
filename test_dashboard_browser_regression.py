#!/usr/bin/env python3
"""Optional browser regression checks for dashboard editor state across polling."""

from __future__ import annotations

import contextlib
import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from test_peer_propagation import HOST, fetch_json, free_port, wait_for_port


try:
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright
except ImportError:  # pragma: no cover - environment-dependent optional browser check
    sync_playwright = None
    PlaywrightError = Exception


INITIAL_TEAM_CONFIGURATION = {
    "1": {
        "role": "isr",
        "capabilities": ["atr_observe", "isr_observe"],
        "available": True,
        "payload_inventory": {},
        "camera_mount": "steerable",
    },
    "2": {
        "role": "isr",
        "capabilities": ["atr_observe", "isr_observe"],
        "available": True,
        "payload_inventory": {},
        "camera_mount": "steerable",
    },
    "3": {
        "role": "isr",
        "capabilities": ["atr_observe", "isr_observe"],
        "available": True,
        "payload_inventory": {},
        "camera_mount": "steerable",
    },
}


class DashboardBrowserRegressionTest(unittest.TestCase):
    process: subprocess.Popen[str] | None

    def setUp(self) -> None:
        if sync_playwright is None:
            self.skipTest("playwright is not installed; optional browser interaction check skipped")

        self.dashboard_port = free_port()
        self.ports = [free_port(), free_port(), free_port()]
        self.urls = [f"http://{HOST}:{port}" for port in self.ports]
        self.team_config_file = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump(INITIAL_TEAM_CONFIGURATION, self.team_config_file)
        self.team_config_file.close()
        launcher = Path(__file__).with_name("launch_simulation.py")
        self.process = subprocess.Popen(
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
                "--team-config",
                self.team_config_file.name,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        self.addCleanup(self.cleanup_processes)

        self.assertTrue(wait_for_port(self.dashboard_port), "Dashboard did not start")
        for port in self.ports:
            self.assertTrue(wait_for_port(port), f"Drone on port {port} did not start")

    def cleanup_processes(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()

        if self.process:
            with contextlib.suppress(subprocess.TimeoutExpired):
                self.process.wait(timeout=5)

        if self.process and self.process.poll() is None:
            self.process.kill()

        Path(self.team_config_file.name).unlink(missing_ok=True)

    def wait_for_blue_payload_configuration(self) -> dict[str, object]:
        deadline = time.monotonic() + 12.0
        last_status: dict[str, object] | None = None

        while time.monotonic() < deadline:
            last_status = fetch_json(f"{self.urls[1]}/status")
            config = last_status["team_configuration"]  # type: ignore[index]

            if (
                config["role"] == "payload"
                and config["capabilities"] == ["payload_drop"]
                and config["payload_inventory"] == {"water": 12}
            ):
                return last_status

            time.sleep(0.25)

        raise AssertionError(f"Blue did not acknowledge payload configuration; last_status={last_status!r}")

    def test_team_editor_draft_focus_and_apply_survive_polling_refreshes(self) -> None:
        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch()
                try:
                    page = browser.new_page()
                    page.goto(f"http://{HOST}:{self.dashboard_port}/", wait_until="domcontentloaded")
                    page.wait_for_selector("[data-team-role='2']", timeout=10000)

                    role = page.locator("[data-team-role='2']")
                    water = page.locator("[data-team-inventory='2:water']")
                    role.select_option("payload")
                    water.fill("12")
                    water.focus()
                    self.assertEqual("12", water.input_value())
                    self.assertEqual("water", page.evaluate("document.activeElement.getAttribute('data-team-inventory')").split(":")[1])

                    page.wait_for_timeout(2300)
                    self.assertEqual("payload", role.input_value())
                    self.assertEqual("12", water.input_value())
                    self.assertEqual("2:water", page.evaluate("document.activeElement.getAttribute('data-team-inventory')"))
                    self.assertIn("Unsaved changes", page.locator("#team-setup-state").inner_text())

                    page.evaluate("state.scenarioSeed = 'not-a-finite-seed'")
                    page.locator("#apply-team-restart-button").click()
                    page.wait_for_selector("#action-error-banner:not([hidden])", timeout=5000)
                    self.assertIn("Invalid scenario seed", page.locator("#action-error-banner").inner_text())
                    self.assertEqual("payload", role.input_value())
                    self.assertEqual("12", water.input_value())
                    self.assertIn("Unsaved changes", page.locator("#team-setup-state").inner_text())

                    red_status = fetch_json(f"{self.urls[0]}/status")
                    self.assertEqual("isr", red_status["team_configuration"]["role"])
                    self.assertEqual("isr", fetch_json(f"{self.urls[1]}/status")["team_configuration"]["role"])

                    page.evaluate("state.scenarioSeed = 20261106")
                    page.locator("#apply-team-restart-button").click()
                    page.wait_for_function("document.getElementById('team-setup-state').textContent.includes('Applied')", timeout=15000)
                    self.wait_for_blue_payload_configuration()
                finally:
                    browser.close()
        except PlaywrightError as error:
            self.skipTest(f"playwright browser is unavailable: {error}")


if __name__ == "__main__":
    unittest.main()
