import json
import shutil
import subprocess
import textwrap
import unittest

import launch_simulation


class DashboardAssignmentAggregationTest(unittest.TestCase):
    def run_dashboard_script(self, fixture_script: str) -> dict[str, str]:
        node = shutil.which("node")

        if node is None:
            self.skipTest("node is required to execute the embedded dashboard JavaScript regression")

        script = launch_simulation.DASHBOARD_HTML
        script = script.replace("__DRONES_JSON__", json.dumps([
            {"id": 1, "name": "Red", "color": "#ef4444", "url": "http://127.0.0.1:9101"},
            {"id": 2, "name": "Blue", "color": "#3b82f6", "url": "http://127.0.0.1:9102"},
            {"id": 3, "name": "Green", "color": "#22c55e", "url": "http://127.0.0.1:9103"},
        ]))
        javascript = script.split("<script>", 1)[1].split("</script>", 1)[0]
        runner = textwrap.dedent(
            f"""
            const vm = require("vm");
            const elements = new Map();

            function makeElement(id) {{
              return {{
                id,
                innerHTML: "",
                textContent: "",
                checked: true,
                value: "0",
                className: "",
                get selectedOptions() {{
                  const labels = {{
                    "seeded_stationary_patrol": "Seeded stationary patrol",
                    "moving_person_handoff": "Jogging person handoff",
                    "report_continue": "Report and continue",
                    "observe_orbit": "Observe and orbit",
                    "orbit_follow": "Orbit and follow — Stage B",
                    "0": "Random (1–5)",
                    "1": "Red",
                    "2": "Blue",
                    "3": "Green",
                    "4": "4",
                    "5": "5",
                  }};
                  return [{{ value: this.value, textContent: labels[this.value] || this.value || "" }}];
                }},
                width: 520,
                height: 360,
                getBoundingClientRect() {{ return {{ width: 520, height: id.startsWith("belief-map") ? 420 : 360 }}; }},
                classList: {{ add() {{}}, remove() {{}}, toggle() {{}} }},
                style: {{}},
                appendChild() {{}},
                addEventListener() {{}},
                setAttribute(name, value) {{ this[name] = value; }},
                getAttribute(name) {{ return this[name] || ""; }},
                focus() {{}},
                setSelectionRange(start, end) {{ this.selectionStart = start; this.selectionEnd = end; }},
                closest() {{ return null; }},
                getContext() {{
                  return {{
                    clearRect() {{}}, beginPath() {{}}, moveTo() {{}}, lineTo() {{}}, stroke() {{}},
                    fill() {{}}, arc() {{}}, closePath() {{}}, save() {{}}, translate() {{}},
                    rotate() {{}}, restore() {{}}, fillText() {{}}, rect() {{}}, setLineDash() {{}},
                    ellipse() {{}}, measureText(value) {{ return {{ width: String(value).length * 6 }}; }},
                  }};
                }},
                querySelector() {{
                  return makeElement("query");
                }},
                querySelectorAll() {{
                  return [];
                }},
              }};
            }}

            const document = {{
              getElementById(id) {{
                if (!elements.has(id)) {{
                  elements.set(id, makeElement(id));
                }}
                return elements.get(id);
              }},
              createElement(tagName) {{
                return makeElement(tagName);
              }},
              querySelectorAll() {{
                return [];
              }},
              addEventListener() {{}},
              activeElement: null,
            }};

            const context = {{
              console,
              document,
              Date,
              Error,
              JSON,
              Map,
              Math,
              Number,
              Promise,
              Set,
              String,
              fetch: () => new Promise(() => {{}}),
              setInterval() {{}},
              setTimeout(callback) {{ callback(); }},
              window: {{ devicePixelRatio: 1, localStorage: {{ getItem() {{ return null; }}, setItem() {{}} }}, requestAnimationFrame(callback) {{ callback(); }} }},
              navigator: {{ clipboard: {{ writeText() {{}} }} }},
              elements,
              process,
            }};

            vm.createContext(context);
            vm.runInContext({json.dumps(javascript)}, context);
            vm.runInContext({json.dumps(fixture_script)}, context);

            process.stdout.write(JSON.stringify({{
              assignmentDetails: document.getElementById("assignment-details").innerHTML,
              taskState: document.getElementById("task-state").textContent,
              beliefDetails1: document.getElementById("belief-details-1").innerHTML,
              networkDetails: document.getElementById("network-details").innerHTML,
              networkSummary: document.getElementById("network-summary").textContent,
              sharedInspector: document.getElementById("shared-inspector").innerHTML,
            }}));
            """
        )

        try:
            completed = subprocess.run(
                [node, "-e", runner],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as error:
            self.fail(
                "Dashboard JavaScript regression failed.\n"
                f"stdout:\n{error.stdout}\n"
                f"stderr:\n{error.stderr}",
            )
        return json.loads(completed.stdout)

    def test_served_dashboard_contains_collapsed_team_network_section(self) -> None:
        self.assertIn("Network &amp; messages", launch_simulation.DASHBOARD_HTML)
        self.assertIn("Team network", launch_simulation.DASHBOARD_HTML)
        self.assertIn('id="network-summary"', launch_simulation.DASHBOARD_HTML)
        self.assertIn('id="network-details"', launch_simulation.DASHBOARD_HTML)
        self.assertIn('id="tab-network"', launch_simulation.DASHBOARD_HTML)
        self.assertNotIn("<details open>\n      <summary>Peer HTTP diagnostics</summary>", launch_simulation.DASHBOARD_HTML)

    def test_handoff_evidence_preserves_successful_handoff_after_track_loss(self) -> None:
        fixture = """
        state.scenarioId = "moving_person_handoff";
        state.scenarioRunId = "dashboard-handoff-loss";
        state.initialObserverId = 1;
        state.nextObserverId = 2;
        state.statuses.set(1, { id: 1 });
        state.statuses.set(2, { id: 2 });
        state.statuses.set(3, { id: 3 });
        const trackedObject = {
          local_object_id: "moving-a",
          object_id: "moving-a",
          label: "person",
          priority: false,
          tracking_model: "constant_velocity",
          track_status: "tracked",
          source_names: ["Red", "Blue"],
          historical_sources: [1, 2],
          historical_source_names: ["Red", "Blue"],
          own_sensor: true,
          teammate_sourced: true,
          measurements: [
            { observation_id: "red-a", scenario_run_id: "dashboard-handoff-loss", source_drone_id: 1, source_drone_name: "Red", sensor_id: "eo-red", source_track_id: "red-track" },
            { observation_id: "blue-a", scenario_run_id: "dashboard-handoff-loss", source_drone_id: 2, source_drone_name: "Blue", sensor_id: "eo-blue", source_track_id: "blue-track" },
          ],
        };
        const remoteObject = { ...trackedObject, own_sensor: false, teammate_sourced: true };
        state.beliefs.set(1, { objects: [trackedObject] });
        state.beliefs.set(2, { objects: [trackedObject] });
        state.beliefs.set(3, { objects: [remoteObject] });
        let html = handoffEvidenceSummaryHtml();
        state.beliefs.set(1, { objects: [{ ...trackedObject, track_status: "lost", source_names: [] }] });
        state.beliefs.set(2, { objects: [{ ...trackedObject, track_status: "lost", source_names: [] }] });
        state.beliefs.set(3, { objects: [{ ...remoteObject, track_status: "lost", source_names: [] }] });
        html += handoffEvidenceSummaryHtml();
        process.stdout.write(JSON.stringify({ evidenceHtml: html }));
        process.exit(0);
        """
        output = self.run_dashboard_script(fixture)

        self.assertIn("Acquired by initial observer", output["evidenceHtml"])
        self.assertIn("Reports accepted from next observer", output["evidenceHtml"])
        self.assertIn("Mapped-object ID preserved", output["evidenceHtml"])
        self.assertIn("Third node remote track", output["evidenceHtml"])
        self.assertIn("Lost after timeout", output["evidenceHtml"])
        self.assertGreaterEqual(output["evidenceHtml"].count("Verified"), 5)

    def test_selected_object_uses_historical_and_recent_sources_from_latest_polled_record(self) -> None:
        fixture = """
        state.beliefs.set(1, {
          simulation_time: 50,
          unresolved_observation_count: 0,
          objects: [{
            local_object_id: "object-a",
            object_id: "object-a",
            label: "vehicle",
            priority: false,
            stale: true,
            last_seen: 13,
            source_names: [],
            historical_source_names: ["Red", "Blue"],
            own_sensor: true,
            teammate_sourced: true,
            active_contribution_count: 0,
            measurements: [{
              observation_id: "obs-a",
              source_drone_name: "Red",
              source_drone_id: 1,
              sensor_id: "eo-red",
              source_track_id: "track-a",
              timestamp: 13,
              classification_score: 0.91,
              east_m: 25,
              north_m: 35,
              covariance: [[64, 0], [0, 36]],
              scenario_run_id: "20261106",
              raw_observation: { id: "obs-a", sensor_id: "eo-red", classification_score: 0.91 },
            }],
          }],
        });
        state.selectedMapObject = {
          key: "belief:1:object-a",
          kind: "belief",
          droneId: 1,
          object: { label: "vehicle", source_names: [], historical_source_names: [], measurements: [] },
          ageSeconds: 99,
        };
        state.mapHitTargets.set("belief-map-1", [{
          key: "belief:1:object-a",
          kind: "belief",
          droneId: 1,
          object: state.beliefs.get(1).objects[0],
          ageSeconds: 37,
        }]);
        renderBeliefDetails({ id: 1, name: "Red" });
        renderSharedInspector();
        """
        output = self.run_dashboard_script(fixture)

        self.assertIn("stale · 0 unresolved", output["beliefDetails1"])
        self.assertIn("Red belief object", output["sharedInspector"])
        self.assertIn("Observed by", output["sharedInspector"])
        self.assertIn("Red, Blue", output["sharedInspector"])
        self.assertIn("Recent sources", output["sharedInspector"])
        self.assertIn("None — stale", output["sharedInspector"])
        self.assertIn("SIMULATED SENSOR INPUT", output["sharedInspector"])
        self.assertIn("Trace this observation", output["sharedInspector"])
        self.assertIn("classification_score", output["sharedInspector"])

    def test_stale_owner_active_task_contradiction_is_presented_as_historical_not_current(self) -> None:
        fixture = """
        const now = Date.now() / 1000;

        state.statuses.set(3, { current_mode: "orbiting", assigned_objective: "sig-antenna" });
        state.taskSnapshotsByDrone.set(3, {
          droneId: 3,
          droneName: "Green",
          online: true,
          receivedAt: now,
          simulationTime: 100,
          currentMode: "orbiting",
          assignedObjective: "sig-antenna",
          responseTask: {
            task_id: "task-antenna",
            signature_id: "sig-antenna",
            object_label: "antenna",
            assigned_drone_id: 3,
            assigned_drone_name: "Green",
            status: "orbiting",
            revision: 8,
            playbook_config_version: 4,
            last_observation_at: 95,
          },
          responseTasks: [
            {
              task_id: "task-antenna",
              signature_id: "sig-antenna",
              object_label: "antenna",
              assigned_drone_id: 3,
              assigned_drone_name: "Green",
              status: "orbiting",
              revision: 8,
              playbook_config_version: 4,
              last_observation_at: 95,
            },
            {
              task_id: "task-vehicle",
              signature_id: "sig-vehicle",
              object_label: "vehicle",
              assigned_drone_id: 3,
              assigned_drone_name: "Green",
              status: "assigned_remote",
              owner_task_state: "orbiting",
              revision: 6,
              playbook_config_version: 3,
              last_observation_at: 70,
            },
          ],
        });

        renderAssignments();
        renderControls();
        """
        output = self.run_dashboard_script(fixture)

        self.assertIn("PRIORITY Antenna", output["assignmentDetails"])
        self.assertIn("PRIORITY Vehicle", output["assignmentDetails"])
        self.assertIn("<td>Green</td>", output["assignmentDetails"])
        self.assertIn("<td>orbiting</td>", output["assignmentDetails"])
        self.assertIn("<td>Unassigned</td>", output["assignmentDetails"])
        self.assertIn("<td>reported</td>", output["assignmentDetails"])
        self.assertIn("historical owner not displayed as current", output["assignmentDetails"])
        self.assertIn("1 orbiting · 0 transitioning · 0 searching · 0 queued", output["taskState"])

    def test_playbook_selector_initializes_from_acknowledged_backend_state(self) -> None:
        fixture = """
        state.selectedPlaybook = "observe_orbit";
        state.statuses.set(1, { selected_playbook: "observe_orbit", playbook_config_version: 4 });
        state.statuses.set(2, { selected_playbook: "observe_orbit", playbook_config_version: 4 });
        state.statuses.set(3, { selected_playbook: "observe_orbit", playbook_config_version: 4 });
        renderControls();
        process.stdout.write(JSON.stringify({
          playbookValue: elements.get("playbook-select").value,
          playbookState: elements.get("playbook-config-state").textContent,
        }));
        process.exit(0);
        """
        output = self.run_dashboard_script(fixture)

        self.assertEqual("observe_orbit", output["playbookValue"])
        self.assertIn("Applied playbook observe_orbit", output["playbookState"])

    def test_network_summary_uses_backend_metrics(self) -> None:
        fixture = """
        state.networks.set(1, {
          node_id: 1,
          node_name: "Red",
          hostname: "mac",
          pid: 123,
          app_build: "build",
          api_address: "http://127.0.0.1:9101",
          configured_peers: ["http://127.0.0.1:9102"],
          observation_messages_sent: 2,
          observation_messages_received: 3,
          claim_messages_sent: 4,
          claim_messages_received: 5,
          payload_bytes_sent: 100,
          payload_bytes_received: 200,
          payload_bytes_per_second: { sent: 12.5, received: 7.5 },
          send_failures: 1,
          duplicate_observations_ignored: 6,
          validation_rejects: 7,
          trace_retained_count: 8,
          trace_retention_limit: 256,
          trace_evictions: 9,
          peer_connection_state: [{
            peer: "http://127.0.0.1:9102",
            reachability: "recently_reachable",
            last_rtt_ms: 3.25,
            last_success_wall_time: 12345,
            latest_error: null,
            observation_messages_sent: 2,
            claim_messages_sent: 4,
            payload_bytes_sent: 100,
            send_failures: 1,
          }],
        });
        renderNetwork();
        """
        output = self.run_dashboard_script(fixture)

        self.assertIn("1/3 nodes reporting", output["networkSummary"])
        self.assertIn("20.0 B/s JSON payload only", output["networkSummary"])
        self.assertIn("1 send failures", output["networkSummary"])
        self.assertIn("Observation messages", output["networkDetails"])
        self.assertIn("Task/config messages", output["networkDetails"])
        self.assertIn("3.3 ms", output["networkDetails"])

    def test_owner_reports_override_assigned_remote_peer_snapshots_regardless_of_arrival_order(self) -> None:
        fixture = """
        const now = Date.now() / 1000;

        state.statuses.set(1, { current_mode: "orbiting", assigned_objective: "sig-a" });
        state.statuses.set(2, { current_mode: "orbiting", assigned_objective: "sig-b" });
        state.statuses.set(3, { current_mode: "orbiting", assigned_objective: "sig-c" });

        state.taskSnapshotsByDrone.set(2, {
          droneId: 2,
          droneName: "Blue",
          online: true,
          receivedAt: now,
          simulationTime: 42,
          currentMode: "orbiting",
          assignedObjective: "sig-b",
          responseTasks: [
            { task_id: "task-a", signature_id: "sig-a", assigned_drone_id: 1, assigned_drone_name: "Red", status: "assigned_remote", owner_task_state: "orbiting", owner_task_updated_at: 10, last_observation_at: 35 },
            { task_id: "task-b", signature_id: "sig-b", assigned_drone_id: 2, assigned_drone_name: "Blue", status: "orbiting", last_observation_at: 36 },
          ],
        });
        state.taskSnapshotsByDrone.set(3, {
          droneId: 3,
          droneName: "Green",
          online: true,
          receivedAt: now,
          simulationTime: 43,
          currentMode: "orbiting",
          assignedObjective: "sig-c",
          responseTasks: [
            { task_id: "task-b", signature_id: "sig-b", assigned_drone_id: 2, assigned_drone_name: "Blue", status: "assigned_remote", owner_task_state: "orbiting", owner_task_updated_at: 11, last_observation_at: 36 },
            { task_id: "task-c", signature_id: "sig-c", assigned_drone_id: 3, assigned_drone_name: "Green", status: "orbiting", last_observation_at: 37 },
          ],
        });
        state.taskSnapshotsByDrone.set(1, {
          droneId: 1,
          droneName: "Red",
          online: true,
          receivedAt: now,
          simulationTime: 44,
          currentMode: "orbiting",
          assignedObjective: "sig-a",
          responseTasks: [
            { task_id: "task-c", signature_id: "sig-c", assigned_drone_id: 3, assigned_drone_name: "Green", status: "assigned_remote", owner_task_state: "orbiting", owner_task_updated_at: 12, last_observation_at: 37 },
            { task_id: "task-a", signature_id: "sig-a", assigned_drone_id: 1, assigned_drone_name: "Red", status: "orbiting", last_observation_at: 35 },
          ],
        });

        renderAssignments();
        renderControls();
        """
        output = self.run_dashboard_script(fixture)

        self.assertEqual(output["assignmentDetails"].count("<td>orbiting</td>"), 3)
        self.assertNotIn("<td>assigned_remote</td>", output["assignmentDetails"])
        self.assertIn("3 orbiting · 0 transitioning · 0 searching · 0 queued", output["taskState"])

    def test_unavailable_owner_uses_peer_reported_state_or_unknown_without_fabricating_live_status(self) -> None:
        fixture = """
        const now = Date.now() / 1000;

        state.statuses.set(2, { current_mode: "searching", assigned_objective: null });
        state.taskSnapshotsByDrone.set(1, {
          droneId: 1,
          droneName: "Red",
          online: false,
          receivedAt: now - 20,
          unavailableAt: now,
          simulationTime: 20,
          currentMode: "orbiting",
          assignedObjective: "sig-a",
          responseTasks: [
            { task_id: "task-a", signature_id: "sig-a", assigned_drone_id: 1, assigned_drone_name: "Red", status: "orbiting", last_observation_at: 18 },
          ],
        });
        state.taskSnapshotsByDrone.set(2, {
          droneId: 2,
          droneName: "Blue",
          online: true,
          receivedAt: now,
          simulationTime: 30,
          currentMode: "searching",
          assignedObjective: null,
          responseTasks: [
            { task_id: "task-a", signature_id: "sig-a", assigned_drone_id: 1, assigned_drone_name: "Red", status: "assigned_remote", owner_task_state: "transitioning", owner_task_updated_at: 25, last_observation_at: 18 },
            { task_id: "task-b", signature_id: "sig-b", assigned_drone_id: 3, assigned_drone_name: "Green", status: "assigned_remote", last_observation_at: 19 },
          ],
        });

        renderAssignments();
        """
        output = self.run_dashboard_script(fixture)

        self.assertIn("transitioning (peer-reported)", output["assignmentDetails"])
        self.assertIn("Assigned — activity unknown", output["assignmentDetails"])
        self.assertIn("owner unavailable", output["assignmentDetails"])
        self.assertNotIn("<td>orbiting</td>", output["assignmentDetails"])

    def test_stale_owner_does_not_leak_into_newer_queued_record_regardless_of_arrival_order(self) -> None:
        def fixture(order: str) -> str:
            return f"""
            const now = Date.now() / 1000;
            const equipmentReport = {{
              droneId: 2,
              droneName: "Blue",
              online: true,
              receivedAt: now,
              simulationTime: 80,
              currentMode: "orbiting",
              assignedObjective: "sig-equipment",
              responseTask: {{
                task_id: "task-equipment",
                signature_id: "sig-equipment",
                object_label: "equipment case",
                assigned_drone_id: 2,
                assigned_drone_name: "Blue",
                status: "orbiting",
                revision: 6,
                playbook_config_version: 3,
                last_observation_at: 75,
              }},
              responseTasks: [{{
                task_id: "task-equipment",
                signature_id: "sig-equipment",
                object_label: "equipment case",
                assigned_drone_id: 2,
                assigned_drone_name: "Blue",
                status: "orbiting",
                revision: 6,
                playbook_config_version: 3,
                last_observation_at: 75,
              }}],
            }};
            const staleVehicleOwnerReport = {{
              droneId: 1,
              droneName: "Red",
              online: true,
              receivedAt: now,
              simulationTime: 78,
              currentMode: "patrol",
              assignedObjective: null,
              responseTasks: [{{
                task_id: "task-vehicle",
                signature_id: "sig-vehicle",
                object_label: "vehicle",
                assigned_drone_id: 2,
                assigned_drone_name: "Blue",
                status: "assigned_remote",
                owner_task_state: "orbiting",
                revision: 4,
                playbook_config_version: 3,
                last_observation_at: 70,
              }}],
            }};
            const queuedVehicleReport = {{
              droneId: 3,
              droneName: "Green",
              online: true,
              receivedAt: now,
              simulationTime: 81,
              currentMode: "patrol",
              assignedObjective: null,
              responseTasks: [{{
                task_id: "task-vehicle",
                signature_id: "sig-vehicle",
                object_label: "vehicle",
                assigned_drone_id: null,
                assigned_drone_name: null,
                previous_assigned_drone_id: 2,
                previous_assigned_drone_name: "Blue",
                status: "queued",
                revision: 5,
                playbook_config_version: 3,
                last_observation_at: 76,
              }}],
            }};

            if ({json.dumps(order)} === "stale-first") {{
              state.taskSnapshotsByDrone.set(1, staleVehicleOwnerReport);
              state.taskSnapshotsByDrone.set(3, queuedVehicleReport);
            }} else {{
              state.taskSnapshotsByDrone.set(3, queuedVehicleReport);
              state.taskSnapshotsByDrone.set(1, staleVehicleOwnerReport);
            }}

            state.statuses.set(2, {{ current_mode: "orbiting", assigned_objective: "sig-equipment" }});
            state.taskSnapshotsByDrone.set(2, equipmentReport);
            renderAssignments();
            """

        stale_first = self.run_dashboard_script(fixture("stale-first"))["assignmentDetails"]
        queued_first = self.run_dashboard_script(fixture("queued-first"))["assignmentDetails"]

        for output in (stale_first, queued_first):
            self.assertIn("PRIORITY Equipment Case", output)
            self.assertIn("PRIORITY Vehicle", output)
            self.assertIn("<td>Blue</td>", output)
            self.assertIn("<td>orbiting</td>", output)
            self.assertIn("<td>Unassigned</td>", output)
            self.assertIn("<td>queued</td>", output)
            self.assertNotIn("assigned_remote", output)

        self.assertEqual(stale_first, queued_first)

    def test_readable_task_labels_and_only_current_owner_task_executes(self) -> None:
        fixture = """
        const now = Date.now() / 1000;

        state.statuses.set(2, { current_mode: "orbiting", assigned_objective: "sig-active" });
        state.taskSnapshotsByDrone.set(2, {
          droneId: 2,
          droneName: "Blue",
          online: true,
          receivedAt: now,
          simulationTime: 50,
          currentMode: "orbiting",
          assignedObjective: "sig-active",
          responseTask: {
            task_id: "task-active-a7c2",
            signature_id: "sig-active",
            object_label: "antenna",
            display_label: "PRIORITY Antenna",
            assigned_drone_id: 2,
            assigned_drone_name: "Blue",
            status: "orbiting",
            last_observation_at: 46,
          },
          responseTasks: [
            {
              task_id: "task-active-a7c2",
              signature_id: "sig-active",
              object_label: "antenna",
              display_label: "PRIORITY Antenna",
              assigned_drone_id: 2,
              assigned_drone_name: "Blue",
              status: "orbiting",
              last_observation_at: 46,
            },
            {
              task_id: "task-old-b9d1",
              signature_id: "sig-old",
              object_label: "vehicle",
              display_label: "PRIORITY Vehicle",
              assigned_drone_id: 2,
              assigned_drone_name: "Blue",
              status: "orbiting",
              last_observation_at: 20,
            },
          ],
        });

        renderAssignments();
        """
        output = self.run_dashboard_script(fixture)

        self.assertIn("Blue: orbiting PRIORITY Antenna", output["assignmentDetails"])
        self.assertIn("PRIORITY Antenna", output["assignmentDetails"])
        self.assertIn("PRIORITY Vehicle", output["assignmentDetails"])
        self.assertEqual(output["assignmentDetails"].count("<td>orbiting</td>"), 1)
        self.assertIn("not active on owner", output["assignmentDetails"])
        self.assertNotIn("mock-search-signature", output["assignmentDetails"])
