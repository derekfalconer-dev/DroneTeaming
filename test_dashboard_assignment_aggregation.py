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
                width: 520,
                height: 360,
                getBoundingClientRect() {{ return {{ width: 520, height: id.startsWith("belief-map") ? 420 : 360 }}; }},
                classList: {{ add() {{}}, remove() {{}} }},
                style: {{}},
                appendChild() {{}},
                addEventListener() {{}},
                getContext() {{
                  return {{
                    clearRect() {{}}, beginPath() {{}}, moveTo() {{}}, lineTo() {{}}, stroke() {{}},
                    fill() {{}}, arc() {{}}, closePath() {{}}, save() {{}}, translate() {{}},
                    rotate() {{}}, restore() {{}}, fillText() {{}}, rect() {{}}, setLineDash() {{}},
                    ellipse() {{}},
                  }};
                }},
                querySelector() {{
                  return makeElement("query");
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
              window: {{ devicePixelRatio: 1 }},
            }};

            vm.createContext(context);
            vm.runInContext({json.dumps(javascript)}, context);
            vm.runInContext({json.dumps(fixture_script)}, context);

            process.stdout.write(JSON.stringify({{
              assignmentDetails: elements.get("assignment-details").innerHTML,
              taskState: elements.get("task-state") ? elements.get("task-state").textContent : "",
            }}));
            """
        )

        completed = subprocess.run(
            [node, "-e", runner],
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(completed.stdout)

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
