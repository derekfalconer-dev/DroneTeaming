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
        fixture_source = f"(async () => {{\n{fixture_script}\n}})()"
        runner = textwrap.dedent(
            f"""
            const vm = require("vm");
            const dashboardHtml = {json.dumps(script)};
            const elements = new Map();
            const intervals = [];
            const intervalCallbacks = [];
            const scheduledTimeouts = new Map();
            const timeoutMetrics = {{ scheduled: 0, fired: 0, cleared: 0 }};
            const fetchCalls = [];
            let nextTimerId = 1;
            const capturedWrites = [];
            const unexpectedErrors = [];

            function parseDashboardDom(html) {{
              const elementsById = new Map();
              const parentIds = new Map();
              const stack = [];
              const voidTags = new Set(["area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"]);
              const tagPattern = /<\\/?([A-Za-z][A-Za-z0-9:-]*)([^>]*)>/g;
              let match;

              while ((match = tagPattern.exec(html)) !== null) {{
                const fullTag = match[0];
                const tag = match[1].toLowerCase();
                const attrs = match[2] || "";
                const isClosing = fullTag.startsWith("</");

                if (isClosing) {{
                  while (stack.length) {{
                    const node = stack.pop();
                    if (node.tag === tag) {{
                      node.innerHTML = html.slice(node.startInner, match.index);
                      node.outerHTML = html.slice(node.startTagStart, tagPattern.lastIndex);
                      if (node.id) {{
                        elementsById.set(node.id, node);
                      }}
                      break;
                    }}
                  }}
                  continue;
                }}

                const idMatch = attrs.match(/\\sid\\s*=\\s*("([^"]*)"|'([^']*)'|([^\\s>]+))/i);
                const id = idMatch ? (idMatch[2] || idMatch[3] || idMatch[4] || "") : "";
                const parentWithId = stack.slice().reverse().find((node) => node.id);

                if (id && parentWithId) {{
                  parentIds.set(id, parentWithId.id);
                }}

                const node = {{
                  tag,
                  id,
                  startTagStart: match.index,
                  startInner: tagPattern.lastIndex,
                  innerHTML: "",
                  outerHTML: fullTag,
                }};

                if (id) {{
                  elementsById.set(id, node);
                }}

                if (!voidTags.has(tag) && !fullTag.endsWith("/>")) {{
                  stack.push(node);
                }}
              }}

              return {{ elementsById, parentIds }};
            }}

            const parsedDom = parseDashboardDom(dashboardHtml);
            const parsedElements = parsedDom.elementsById;
            const parentIds = parsedDom.parentIds;

            function initialInnerHTMLFor(id) {{
              const parsed = parsedElements.get(String(id));
              return parsed ? parsed.innerHTML : "";
            }}

            function defaultValueFor(id) {{
              const defaults = {{
                "scenario-select": "seeded_stationary_patrol",
                "priority-count-select": "0",
                "playbook-select": "report_continue",
                "initial-observer-select": "1",
                "next-observer-select": "2",
              }};
              return defaults[id] || "0";
            }}

            function optionsFor(id) {{
              const valuesById = {{
                "scenario-select": [
                  ["seeded_stationary_patrol", "Random Search of Area"],
                  ["moving_person_handoff", "Jogging person handoff"],
                  ["sar_supply_demo", "SAR supply recipients"],
                ],
                "playbook-select": [
                  ["report_continue", "Report and continue"],
                  ["observe_orbit", "Observe and orbit"],
                  ["orbit_follow", "Orbit and follow — Stage B"],
                ],
                "priority-count-select": [
                  ["0", "Random (1–5)"],
                  ["1", "1"],
                  ["2", "2"],
                  ["3", "3"],
                  ["4", "4"],
                  ["5", "5"],
                ],
                "initial-observer-select": [
                  ["1", "Red"],
                  ["2", "Blue"],
                  ["3", "Green"],
                ],
                "next-observer-select": [
                  ["1", "Red"],
                  ["2", "Blue"],
                  ["3", "Green"],
                ],
              }};
              return (valuesById[id] || []).map(([value, textContent]) => ({{ value, textContent }}));
            }}

            function implicitParentIdFor(id) {{
              const parents = {{
                "global-scenario-controls": "compact-header",
                "scenario-select": "global-scenario-controls",
                "start-scenario-button": "global-scenario-controls",
                "compact-pause-button": "global-scenario-controls",
                "compact-replay-button": "global-scenario-controls",
                "map-grid-section": "tab-team",
                "assignment-card": "tab-team",
                "team-setup-card": "tab-team",
                "belief-display-card": "tab-team",
                "belief-sensor-details": "tab-team",
                "belief-map-1": "map-grid-section",
                "belief-map-2": "map-grid-section",
                "belief-map-3": "map-grid-section",
                "truth-card": "tab-truth",
                "truth-map": "truth-card",
                "network-card": "tab-network",
              }};
              return parentIds.get(id) || parents[id] || null;
            }}

            function abortError() {{
              const error = new Error("The operation was aborted.");
              error.name = "AbortError";
              return error;
            }}

            class MockAbortSignal {{
              constructor() {{
                this.aborted = false;
                this.reason = undefined;
                this.listeners = new Set();
              }}

              addEventListener(type, listener, options = {{}}) {{
                if (type !== "abort" || typeof listener !== "function") {{
                  return;
                }}

                this.listeners.add({{ listener, once: Boolean(options && options.once) }});
              }}

              removeEventListener(type, listener) {{
                if (type !== "abort") {{
                  return;
                }}

                for (const entry of Array.from(this.listeners)) {{
                  if (entry.listener === listener) {{
                    this.listeners.delete(entry);
                  }}
                }}
              }}

              dispatchAbort() {{
                for (const entry of Array.from(this.listeners)) {{
                  entry.listener.call(this, {{ type: "abort", target: this }});

                  if (entry.once) {{
                    this.listeners.delete(entry);
                  }}
                }}
              }}
            }}

            class MockAbortController {{
              constructor() {{
                this.signal = new MockAbortSignal();
              }}

              abort(reason) {{
                if (this.signal.aborted) {{
                  return;
                }}

                this.signal.aborted = true;
                this.signal.reason = reason || abortError();
                this.signal.dispatchAbort();
              }}
            }}

            function defaultFetch(url, options = {{}}) {{
              fetchCalls.push({{
                url: String(url),
                abortedAtStart: Boolean(options && options.signal && options.signal.aborted),
              }});

              return new Promise((_resolve, reject) => {{
                const signal = options && options.signal;

                if (signal && signal.aborted) {{
                  reject(abortError());
                  return;
                }}

                if (signal && typeof signal.addEventListener === "function") {{
                  signal.addEventListener("abort", () => reject(abortError()), {{ once: true }});
                }}
              }});
            }}

            async function flushMicrotasks() {{
              await Promise.resolve();
              await new Promise((resolve) => setImmediate(resolve));
            }}

            async function runScheduledTimeout(timerId) {{
              if (!scheduledTimeouts.has(timerId)) {{
                return false;
              }}

              const timeout = scheduledTimeouts.get(timerId);
              scheduledTimeouts.delete(timerId);
              timeoutMetrics.fired += 1;
              timeout.callback();
              await flushMicrotasks();
              return true;
            }}

            async function runScheduledTimeouts(limit = scheduledTimeouts.size) {{
              const timerIds = Array.from(scheduledTimeouts.keys()).slice(0, limit);

              for (const timerId of timerIds) {{
                await runScheduledTimeout(timerId);
              }}
            }}

            async function runIntervalCallbacks(cycles = 1) {{
              for (let index = 0; index < cycles; index += 1) {{
                for (const callback of intervalCallbacks.slice()) {{
                  await Promise.resolve(callback());
                  await flushMicrotasks();
                }}
              }}
            }}

            process.on("unhandledRejection", (error) => {{
              unexpectedErrors.push(error && error.stack ? error.stack : String(error));
            }});

            process.on("uncaughtException", (error) => {{
              unexpectedErrors.push(error && error.stack ? error.stack : String(error));
            }});

            function makeElement(id) {{
              let _textContent = "";
              const _drawOps = [];
              let _pendingPath = [];
              const _contextState = {{
                fillStyle: "",
                strokeStyle: "",
                globalAlpha: 1,
                lineWidth: 1,
              }};
              return {{
                id,
                innerHTML: initialInnerHTMLFor(id),
                get textContent() {{ return _textContent; }},
                set textContent(value) {{ _textContent = String(value); }},
                checked: true,
                value: defaultValueFor(id),
                className: "",
                get selectedOptions() {{
                  const labels = {{
                    "seeded_stationary_patrol": "Random Search of Area",
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
                parentNode: null,
                children: [],
                drawOps: _drawOps,
                get outerHTML() {{
                  return `<mock-element id="${{id}}">${{this.innerHTML}}${{this.children.map((child) => child.outerHTML || child.innerHTML || "").join("")}}</mock-element>`;
                }},
                getBoundingClientRect() {{ return {{ width: 520, height: id.startsWith("belief-map") ? 420 : 360 }}; }},
                classList: {{ add() {{}}, remove() {{}}, toggle() {{}} }},
                style: {{}},
                appendChild(child) {{
                  if (!child) {{
                    return;
                  }}

                  if (child.parentNode && child.parentNode.children) {{
                    child.parentNode.children = child.parentNode.children.filter((item) => item !== child);
                  }}

                  child.parentNode = this;
                  this.children.push(child);
                  this.innerHTML += child.outerHTML || child.innerHTML || "";
                }},
                contains(candidate) {{
                  if (!candidate) {{
                    return false;
                  }}

                  if (candidate === this) {{
                    return true;
                  }}

                  let cursor = candidate.parentNode || null;
                  while (cursor) {{
                    if (cursor === this) {{
                      return true;
                    }}
                    cursor = cursor.parentNode || null;
                  }}

                  let parentId = implicitParentIdFor(candidate.id);
                  while (parentId) {{
                    const parent = document.getElementById(parentId);
                    if (parent === this || this.contains(parent)) {{
                      return true;
                    }}
                    parentId = implicitParentIdFor(parentId);
                  }}

                  return false;
                }},
                addEventListener() {{}},
                setAttribute(name, value) {{ this[name] = value; }},
                getAttribute(name) {{ return this[name] || ""; }},
                focus() {{}},
                setSelectionRange(start, end) {{ this.selectionStart = start; this.selectionEnd = end; }},
                closest() {{ return null; }},
                getContext() {{
                  return {{
                    get fillStyle() {{ return _contextState.fillStyle; }},
                    set fillStyle(value) {{ _contextState.fillStyle = String(value); }},
                    get strokeStyle() {{ return _contextState.strokeStyle; }},
                    set strokeStyle(value) {{ _contextState.strokeStyle = String(value); }},
                    get globalAlpha() {{ return _contextState.globalAlpha; }},
                    set globalAlpha(value) {{ _contextState.globalAlpha = Number(value); }},
                    get lineWidth() {{ return _contextState.lineWidth; }},
                    set lineWidth(value) {{ _contextState.lineWidth = Number(value); }},
                    clearRect(...args) {{ _drawOps.length = 0; _pendingPath = []; _drawOps.push({{ op: "clearRect", args }}); }},
                    beginPath() {{ _pendingPath = []; }},
                    moveTo(...args) {{ _pendingPath.push({{ op: "moveTo", args }}); }},
                    lineTo(...args) {{ _pendingPath.push({{ op: "lineTo", args }}); }},
                    stroke() {{ _drawOps.push({{ op: "stroke", strokeStyle: _contextState.strokeStyle, globalAlpha: _contextState.globalAlpha, path: _pendingPath.slice() }}); }},
                    fill() {{ _drawOps.push({{ op: "fill", fillStyle: _contextState.fillStyle, globalAlpha: _contextState.globalAlpha, path: _pendingPath.slice() }}); }},
                    arc(...args) {{
                      const op = {{ op: "arc", args, fillStyle: _contextState.fillStyle, strokeStyle: _contextState.strokeStyle, globalAlpha: _contextState.globalAlpha }};
                      _pendingPath.push(op);
                      _drawOps.push(op);
                    }},
                    closePath() {{ _pendingPath.push({{ op: "closePath" }}); }},
                    save() {{ _drawOps.push({{ op: "save" }}); }},
                    translate(...args) {{ _drawOps.push({{ op: "translate", args }}); }},
                    rotate(...args) {{ _drawOps.push({{ op: "rotate", args }}); }},
                    restore() {{ _drawOps.push({{ op: "restore" }}); }},
                    fillText(...args) {{ _drawOps.push({{ op: "fillText", args, fillStyle: _contextState.fillStyle, globalAlpha: _contextState.globalAlpha }}); }},
                    rect(...args) {{ _pendingPath.push({{ op: "rect", args }}); }},
                    setLineDash(...args) {{ _drawOps.push({{ op: "setLineDash", args }}); }},
                    ellipse(...args) {{ _pendingPath.push({{ op: "ellipse", args }}); }},
                    measureText(value) {{ return {{ width: String(value).length * 6 }}; }},
                  }};
                }},
                querySelector() {{
                  return makeElement("query");
                }},
                querySelectorAll(selector) {{
                  if (selector === "option") {{
                    return optionsFor(id);
                  }}

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
              AbortController: MockAbortController,
              fetch: defaultFetch,
              fetchCalls,
              clearTimeout(timerId) {{
                if (scheduledTimeouts.delete(timerId)) {{
                  timeoutMetrics.cleared += 1;
                }}
              }},
              setInterval(callback, delay) {{
                intervals.push(Number(delay));
                intervalCallbacks.push(callback);
                return nextTimerId++;
              }},
              setTimeout(callback, delay) {{
                const timerId = nextTimerId++;
                scheduledTimeouts.set(timerId, {{ callback, delay: Number(delay) || 0 }});
                timeoutMetrics.scheduled += 1;
                return timerId;
              }},
              window: {{ devicePixelRatio: 1, localStorage: {{ getItem() {{ return null; }}, setItem() {{}} }}, requestAnimationFrame(callback) {{ callback(); }} }},
              flushMicrotasks,
              runIntervalCallbacks,
              runScheduledTimeout,
              runScheduledTimeouts,
              scheduledTimeouts,
              timeoutMetrics,
              navigator: {{ clipboard: {{ writeText() {{}} }} }},
              elements,
              intervals,
              process: {{
                stdout: {{
                  write(value) {{ capturedWrites.push(String(value)); }},
                }},
                exit() {{ this.exited = true; }},
                exited: false,
              }},
            }};

            function dashboardOutput() {{
              return JSON.stringify({{
              assignmentDetails: document.getElementById("assignment-details").innerHTML,
              assignmentRows: vm.runInContext("buildAssignmentRows()", context),
              taskState: document.getElementById("task-state").textContent,
              supplyContacts: document.getElementById("metric-supply-contacts").textContent,
              isrAircraft: document.getElementById("metric-isr-aircraft").textContent,
              payloadApproaching: document.getElementById("metric-payload-approaching").textContent,
              payloadHolding: document.getElementById("metric-payload-holding").textContent,
              aircraftCards: document.getElementById("aircraft-cards").innerHTML,
              beliefDetails1: document.getElementById("belief-details-1").innerHTML,
              networkDetails: document.getElementById("network-details").innerHTML,
              networkSummary: document.getElementById("network-summary").textContent,
              sharedInspector: document.getElementById("shared-inspector").innerHTML,
              teamConfigState: document.getElementById("team-config-state").textContent,
              compactHeader: document.getElementById("compact-header").innerHTML,
              tabTeam: document.getElementById("tab-team").innerHTML,
              tabTruth: document.getElementById("tab-truth").innerHTML,
              scenarioControls: document.getElementById("global-scenario-controls").innerHTML,
              scenarioControlState: document.getElementById("scenario-control-state").textContent,
              actionErrorBanner: document.getElementById("action-error-banner").textContent,
              actionErrorHidden: document.getElementById("action-error-banner").hidden,
              layoutContainment: {{
                belief1InTeam: document.getElementById("tab-team").contains(document.getElementById("belief-map-1")),
                belief2InTeam: document.getElementById("tab-team").contains(document.getElementById("belief-map-2")),
                belief3InTeam: document.getElementById("tab-team").contains(document.getElementById("belief-map-3")),
                truthInTruth: document.getElementById("tab-truth").contains(document.getElementById("truth-map")),
                globalControlsOutsideTabs: !document.getElementById("tab-team").contains(document.getElementById("global-scenario-controls"))
                  && !document.getElementById("tab-truth").contains(document.getElementById("global-scenario-controls"))
                  && !document.getElementById("tab-network").contains(document.getElementById("global-scenario-controls")),
              }},
                compactRunState: document.getElementById("compact-run-state").textContent,
                compactElapsedTime: document.getElementById("compact-elapsed-time").textContent,
                refreshIntervals: intervals,
              }});
            }}

            async function run() {{
              vm.createContext(context);
              vm.runInContext({json.dumps(javascript)}, context);
              const fixtureSource = {json.dumps(fixture_source)};
              const fixturePromise = vm.runInContext(fixtureSource, context);
              await Promise.resolve(fixturePromise);
              await flushMicrotasks();

              if (unexpectedErrors.length) {{
                throw new Error(unexpectedErrors.join("\\n"));
              }}

              process.stdout.write(capturedWrites.length ? capturedWrites[capturedWrites.length - 1] : dashboardOutput());
            }}

            run().catch((error) => {{
              process.stderr.write(error && error.stack ? error.stack : String(error));
              process.exit(1);
            }});
            """
        )

        try:
            completed = subprocess.run(
                [node, "-"],
                input=runner,
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

    def test_dashboard_layout_restores_top_level_map_tabs_and_global_controls(self) -> None:
        output = self.run_dashboard_script("renderControls();")

        self.assertIn("Team view", launch_simulation.DASHBOARD_HTML)
        self.assertIn("Simulation truth", launch_simulation.DASHBOARD_HTML)
        self.assertIn("Network &amp; messages", launch_simulation.DASHBOARD_HTML)
        self.assertIn('aria-selected="true" aria-controls="tab-team"', launch_simulation.DASHBOARD_HTML)
        self.assertIn('id="global-scenario-controls"', launch_simulation.DASHBOARD_HTML)
        self.assertIn('id="start-scenario-button"', launch_simulation.DASHBOARD_HTML)
        self.assertIn('id="compact-pause-button"', launch_simulation.DASHBOARD_HTML)
        self.assertIn('id="compact-replay-button"', launch_simulation.DASHBOARD_HTML)
        self.assertIn('data-tab="truth"', launch_simulation.DASHBOARD_HTML)
        self.assertTrue(output["layoutContainment"]["belief1InTeam"])
        self.assertTrue(output["layoutContainment"]["belief2InTeam"])
        self.assertTrue(output["layoutContainment"]["belief3InTeam"])
        self.assertTrue(output["layoutContainment"]["truthInTruth"])
        self.assertTrue(output["layoutContainment"]["globalControlsOutsideTabs"])
        self.assertIn("assignment-card", output["tabTeam"])
        self.assertIn("team-setup-card", output["tabTeam"])
        self.assertIn("Evaluator-only truth", output["tabTruth"])
        self.assertIn("scenario-select", output["compactHeader"])
        self.assertIn("Start scenario", output["compactHeader"])

    def test_global_scenario_control_state_shows_selected_vs_running_scenario(self) -> None:
        fixture = """
        state.scenarioId = "seeded_stationary_patrol";
        state.scenarioName = "Random Search of Area";
        state.scenarioRunId = "active-run";
        state.scenarioControlStatus = "running";
        state.scenarioControlMessage = "Connected to active backend run.";
        state.paused = false;
        state.scenarioConfigDirty = true; // Mirrors the scenario-select change handler; preserves an unapplied operator selection.
        document.getElementById("scenario-select").value = "moving_person_handoff";
        renderControls();
        const firstMessage = document.getElementById("scenario-control-state").textContent;
        renderControls();
        const repeatedMessage = document.getElementById("scenario-control-state").textContent;
        const preservedSelection = document.getElementById("scenario-select").value;

        postControlNoUpdate = async () => {};
        waitForConfigurationAcknowledgement = async (payload) => {
          state.scenarioId = payload.scenario;
          state.scenarioName = "Jogging person handoff";
          state.scenarioRunId = payload.scenario_run_id;
        };
        updateAll = async () => {};
        await startScenarioWithPayload(selectedRunOptionsForStart(), "Scenario started on all nodes.");
        renderControls();

        process.stdout.write(JSON.stringify({
          firstMessage,
          repeatedMessage,
          preservedSelection,
          appliedDirty: state.scenarioConfigDirty,
          appliedSelection: document.getElementById("scenario-select").value,
          appliedMessage: document.getElementById("scenario-control-state").textContent,
          compactRunState: document.getElementById("compact-run-state").textContent,
        }));
        """
        output = self.run_dashboard_script(fixture)

        self.assertIn("Selected Jogging person handoff", output["firstMessage"])
        self.assertIn("running Random Search of Area", output["firstMessage"])
        self.assertEqual(output["firstMessage"], output["repeatedMessage"])
        self.assertEqual("moving_person_handoff", output["preservedSelection"])
        self.assertFalse(output["appliedDirty"])
        self.assertEqual("moving_person_handoff", output["appliedSelection"])
        self.assertIn("Running Jogging person handoff", output["appliedMessage"])
        self.assertEqual("Running", output["compactRunState"])

    def test_start_scenario_applies_selected_configuration_and_unpauses_every_node(self) -> None:
        fixture = """
        const calls = [];
        state.sharingEnabled = true;
        document.getElementById("scenario-select").value = "sar_supply_demo";
        document.getElementById("priority-count-select").value = "3";
        document.getElementById("playbook-select").value = "observe_orbit";
        postControlNoUpdate = async (payload) => { calls.push(payload); };
        waitForConfigurationAcknowledgement = async (payload) => {
          calls.push({ acknowledgementFor: payload.scenario_run_id, paused: payload.paused, scenario: payload.scenario });
        };
        updateAll = async () => {};
        Date.now = () => 1700000000000;
        Math.random = () => 0.5;
        startScenarioWithPayload(selectedRunOptionsForStart(), "Scenario started on all nodes.").then(() => {
          process.stdout.write(JSON.stringify({ calls, status: state.scenarioControlStatus, message: state.scenarioControlMessage }));
          process.exit(0);
        });
        """
        output = self.run_dashboard_script(fixture)
        payload = output["calls"][0]

        self.assertTrue(payload["restart_scenario"])
        self.assertFalse(payload["paused"])
        self.assertTrue(payload["sharing_enabled"])
        self.assertEqual("sar_supply_demo", payload["scenario"])
        self.assertEqual(3, payload["priority_count"])
        self.assertEqual("observe_orbit", payload["playbook"])
        self.assertEqual("running", output["status"])
        self.assertIn("Scenario started", output["message"])

    def test_pause_preserves_current_run_without_restart(self) -> None:
        fixture = """
        const calls = [];
        state.paused = false;
        postControl = async (payload) => { calls.push(payload); };
        document.getElementById("compact-pause-button").onclick = null;
        postControl({ paused: !state.paused }).then(() => {
          process.stdout.write(JSON.stringify({ calls }));
          process.exit(0);
        });
        """
        output = self.run_dashboard_script(fixture)

        self.assertEqual([{"paused": True}], output["calls"])
        self.assertNotIn("restart_scenario", output["calls"][0])

    def test_restart_replays_applied_scenario_configuration(self) -> None:
        fixture = """
        const calls = [];
        state.sharingEnabled = false;
        state.appliedRunOptions = {
          restart_scenario: true,
          scenario_seed: 123,
          priority_count: 2,
          playbook: "report_continue",
          scenario: "moving_person_handoff",
          initial_observer_id: 1,
          next_observer_id: 2,
          scenario_run_id: "applied-run",
          paused: false,
          sharing_enabled: true,
        };
        postControlNoUpdate = async (payload) => { calls.push(payload); };
        waitForConfigurationAcknowledgement = async () => {};
        updateAll = async () => {};
        const payload = { ...state.appliedRunOptions, restart_scenario: true, paused: false, sharing_enabled: state.sharingEnabled };
        startScenarioWithPayload(payload, "Current scenario restarted on all nodes.").then(() => {
          process.stdout.write(JSON.stringify({ calls, applied: state.appliedRunOptions, message: state.scenarioControlMessage }));
          process.exit(0);
        });
        """
        output = self.run_dashboard_script(fixture)

        self.assertEqual("applied-run", output["calls"][0]["scenario_run_id"])
        self.assertEqual("moving_person_handoff", output["calls"][0]["scenario"])
        self.assertFalse(output["calls"][0]["paused"])
        self.assertIn("restarted", output["message"])

    def test_partial_acknowledgement_is_reported_as_failure(self) -> None:
        fixture = """
        state.scenarioStartPending = false;
        postControlNoUpdate = async () => {};
        waitForConfigurationAcknowledgement = async () => { throw new Error("Blue: run stale"); };
        updateAll = async () => {};
        startScenarioWithPayload(selectedRunOptionsForStart(), "Scenario started on all nodes.").then(async () => {
          renderControls();
          await updateAll();
          renderControls();
          process.stdout.write(JSON.stringify({
            status: state.scenarioControlStatus,
            message: state.scenarioControlMessage,
            banner: document.getElementById("action-error-banner").textContent,
            bannerHidden: document.getElementById("action-error-banner").hidden,
          }));
          process.exit(0);
        });
        """
        output = self.run_dashboard_script(fixture)

        self.assertEqual("error", output["status"])
        self.assertIn("Blue: run stale", output["message"])
        self.assertIn("Scenario start failed", output["banner"])
        self.assertIn("Blue: run stale", output["banner"])
        self.assertFalse(output["bannerHidden"])

    def test_refresh_reconnects_to_active_run_without_resetting(self) -> None:
        fixture = """
        const calls = [];
        state.scenarioControlStatus = "starting";
        state.scenarioStartPending = false;
        state.paused = false;
        updateTruth = async () => {
          state.truth = { simulation_time: 12, scenario_id: "seeded_stationary_patrol" };
          state.scenarioId = "seeded_stationary_patrol";
        };
        updateDrone = async (drone) => {
          state.statuses.set(drone.id, {
            current_mode: "patrol",
            sharing_enabled: true,
            paused: false,
            selected_playbook: "report_continue",
            playbook_config_version: 2,
          });
          state.taskSnapshotsByDrone.set(drone.id, {
            droneId: drone.id,
            droneName: drone.name,
            online: true,
            receivedAt: Date.now() / 1000,
            simulationTime: 12,
            currentMode: "patrol",
            assignedObjective: null,
            responseTask: null,
            responseTasks: [],
            responseQueueDepth: 0,
            playbookConfigVersion: 2,
            teamConfiguration: null,
            scenarioId: "seeded_stationary_patrol",
            scenarioRunId: "passive-active-run",
            scenarioSeed: 20261106,
            selectedPlaybook: "report_continue",
            initialObserverId: 1,
            nextObserverId: 2,
            paused: false,
            sharingEnabled: true,
          });
        };
        postControlNoUpdate = async (payload) => { calls.push(payload); };
        renderControls = () => {};
        renderAssignments = () => {};
        renderMissionSummary = () => {};
        renderNetwork = () => {};
        renderMaps = () => {};
        updateAll().then(() => {
          process.stdout.write(JSON.stringify({ calls, status: state.scenarioControlStatus, message: state.scenarioControlMessage }));
          process.exit(0);
        });
        """
        output = self.run_dashboard_script(fixture)

        self.assertEqual([], output["calls"])
        self.assertEqual("running", output["status"])
        self.assertIn("Reconnected", output["message"])

    def test_initial_connection_reconciles_to_running_when_all_nodes_acknowledge_same_run(self) -> None:
        fixture = """
        state.scenarioControlStatus = "connecting";
        updateTruth = async () => {};
        updateDrone = async (drone) => {
          state.statuses.set(drone.id, { current_mode: "patrol" });
          state.taskSnapshotsByDrone.set(drone.id, {
            droneId: drone.id,
            droneName: drone.name,
            online: true,
            receivedAt: Date.now() / 1000,
            simulationTime: 15,
            currentMode: "patrol",
            assignedObjective: null,
            responseTask: null,
            responseTasks: [],
            responseQueueDepth: 0,
            playbookConfigVersion: 2,
            teamConfiguration: null,
            scenarioId: "seeded_stationary_patrol",
            scenarioRunId: "agreed-run",
            scenarioSeed: 20261106,
            selectedPlaybook: "report_continue",
            initialObserverId: 1,
            nextObserverId: 2,
            paused: false,
            sharingEnabled: true,
          });
        };
        renderControls = () => {};
        renderAssignments = () => {};
        renderMissionSummary = () => {};
        renderNetwork = () => {};
        renderMaps = () => {};
        await updateAll();
        process.stdout.write(JSON.stringify({
          status: state.scenarioControlStatus,
          paused: state.paused,
          runId: state.scenarioRunId,
          message: state.scenarioControlMessage,
        }));
        """
        output = self.run_dashboard_script(fixture)

        self.assertEqual("running", output["status"])
        self.assertFalse(output["paused"])
        self.assertEqual("agreed-run", output["runId"])
        self.assertIn("Connected", output["message"])

    def test_initial_connection_reconciles_to_paused_when_all_nodes_acknowledge_paused_run(self) -> None:
        fixture = """
        state.scenarioControlStatus = "connecting";
        updateTruth = async () => {};
        updateDrone = async (drone) => {
          state.taskSnapshotsByDrone.set(drone.id, {
            droneId: drone.id,
            droneName: drone.name,
            online: true,
            receivedAt: Date.now() / 1000,
            simulationTime: 15,
            currentMode: "patrol",
            assignedObjective: null,
            responseTasks: [],
            scenarioId: "seeded_stationary_patrol",
            scenarioRunId: "paused-run",
            scenarioSeed: 20261106,
            selectedPlaybook: "report_continue",
            initialObserverId: 1,
            nextObserverId: 2,
            paused: true,
            sharingEnabled: true,
          });
        };
        renderControls = () => {};
        renderAssignments = () => {};
        renderMissionSummary = () => {};
        renderNetwork = () => {};
        renderMaps = () => {};
        await updateAll();
        process.stdout.write(JSON.stringify({ status: state.scenarioControlStatus, paused: state.paused, runId: state.scenarioRunId }));
        """
        output = self.run_dashboard_script(fixture)

        self.assertEqual("paused", output["status"])
        self.assertTrue(output["paused"])
        self.assertEqual("paused-run", output["runId"])

    def test_initial_connection_requires_all_nodes_to_match_before_running_while_refresh_continues(self) -> None:
        fixture = """
        let renderNetworkCount = 0;
        let renderMapsCount = 0;
        state.scenarioControlStatus = "connecting";
        updateTruth = async () => {};
        updateDrone = async (drone) => {
          if (drone.id === 3) {
            state.taskSnapshotsByDrone.set(drone.id, {
              droneId: drone.id,
              droneName: drone.name,
              online: true,
              receivedAt: Date.now() / 1000,
              simulationTime: 15,
              currentMode: "patrol",
              assignedObjective: null,
              responseTasks: [],
              scenarioId: "seeded_stationary_patrol",
              scenarioRunId: "different-run",
              scenarioSeed: 20261106,
              selectedPlaybook: "report_continue",
              initialObserverId: 1,
              nextObserverId: 2,
              paused: false,
              sharingEnabled: true,
            });
            return;
          }

          state.taskSnapshotsByDrone.set(drone.id, {
            droneId: drone.id,
            droneName: drone.name,
            online: true,
            receivedAt: Date.now() / 1000,
            simulationTime: 15,
            currentMode: "patrol",
            assignedObjective: null,
            responseTasks: [],
            scenarioId: "seeded_stationary_patrol",
            scenarioRunId: "agreed-run",
            scenarioSeed: 20261106,
            selectedPlaybook: "report_continue",
            initialObserverId: 1,
            nextObserverId: 2,
            paused: false,
            sharingEnabled: true,
          });
        };
        renderControls = () => {};
        renderAssignments = () => {};
        renderMissionSummary = () => {};
        renderNetwork = () => { renderNetworkCount += 1; };
        renderMaps = () => { renderMapsCount += 1; };
        await updateAll();
        process.stdout.write(JSON.stringify({
          status: state.scenarioControlStatus,
          renderNetworkCount,
          renderMapsCount,
        }));
        """
        output = self.run_dashboard_script(fixture)

        self.assertEqual("connecting", output["status"])
        self.assertEqual(1, output["renderNetworkCount"])
        self.assertEqual(1, output["renderMapsCount"])

    def test_render_controls_does_not_break_recurring_refresh_maps_or_network(self) -> None:
        fixture = """
        let cycle = 0;
        const drawnCanvases = [];
        const originalDrawBaseMap = drawBaseMap;

        drawBaseMap = (canvas, bounds) => {
          drawnCanvases.push(`${cycle}:${canvas.id}`);
          return originalDrawBaseMap(canvas, bounds);
        };

        fetchJson = async (url) => {
          if (url.endsWith("/truth")) {
            return {
              objects: [{
                id: "truth-person",
                label: "person",
                priority: true,
                latitude: 37.7753 + (cycle * 0.00001),
                longitude: -122.4229,
              }],
              simulation_time: 10 + cycle,
              scenario_name: "Random Search of Area",
              scenario_seed: 20261106,
              app_build: "test-build",
              selected_playbook: "report_continue",
              scenario_id: "seeded_stationary_patrol",
              scenario_run_id: "active-run",
              initial_observer_id: 1,
              next_observer_id: 2,
              requested_priority_count: 0,
              priority_object_count: 1,
              priority_discovered_count: 1,
            };
          }

          const drone = drones.find((item) => url.startsWith(item.url));
          if (!drone) {
            throw new Error(`unexpected URL ${url}`);
          }

          if (url.endsWith("/status")) {
            return {
              state: "running",
              schema_version: "v1",
              sharing_enabled: true,
              paused: false,
              selected_playbook: "report_continue",
              orbit_radius_m: 60,
              timestamp: 10 + cycle,
              current_mode: drone.id === 1 ? "orbiting" : "patrol",
              assigned_objective: drone.id === 1 ? "sig-person" : null,
              response_task: null,
              response_tasks: [],
              response_queue_depth: 0,
              playbook_config_version: 2,
              scenario_id: "seeded_stationary_patrol",
              scenario_run_id: "active-run",
              scenario_seed: 20261106,
              initial_observer_id: 1,
              next_observer_id: 2,
              team_configuration: {
                role: drone.id === 1 ? "isr" : "relay",
                capabilities: ["atr_observe"],
                available: true,
                payload_inventory: {},
              },
              altitude_m: 100,
              speed_mps: 12,
              heading_deg: 90,
              latitude: 37.7752 + (drone.id * 0.0001),
              longitude: -122.4230 - (cycle * 0.00001),
              battery_percent: 90,
              camera_footprint: [
                { latitude: 37.7751, longitude: -122.4231 },
                { latitude: 37.7751, longitude: -122.4229 },
                { latitude: 37.7753, longitude: -122.4230 },
              ],
              flight_trail: [{ latitude: 37.7752, longitude: -122.4230, timestamp: 10 + cycle }],
              diagnostics: { peer_connection_state: [] },
            };
          }

          if (url.endsWith("/objects")) {
            return {
              simulation_time: 10 + cycle,
              unresolved_observation_count: 0,
              objects: [{
                local_object_id: `person-${drone.id}`,
                object_id: `person-${drone.id}`,
                label: "person",
                priority: true,
                latitude: 37.77525 + (cycle * 0.00001),
                longitude: -122.42295,
                last_seen: 10 + cycle,
                own_sensor: drone.id === 1,
                teammate_sourced: drone.id !== 1,
                stale: false,
                active_contribution_count: 1,
                source_names: [drone.name],
                sources: [drone.id],
                measurements: [],
              }],
            };
          }

          if (url.endsWith("/network")) {
            return {
              node_id: drone.id,
              node_name: drone.name,
              hostname: "test-host",
              pid: 1000 + drone.id,
              app_build: "test-build",
              api_address: drone.url,
              configured_peers: [],
              observation_messages_sent: cycle + drone.id,
              observation_messages_received: cycle + 2,
              claim_messages_sent: cycle,
              claim_messages_received: cycle + 1,
              payload_bytes_sent: 100 + cycle,
              payload_bytes_received: 200 + cycle,
              payload_bytes_per_second: { sent: 5 + cycle, received: 7 + cycle },
              send_failures: cycle === 0 ? 0 : 1,
              duplicate_observations_ignored: 0,
              validation_rejects: 0,
              trace_retained_count: 1,
              trace_retention_limit: 256,
              trace_evictions: 0,
              peer_connection_state: [{
                peer: "http://127.0.0.1:9102",
                reachability: "recently_reachable",
                last_rtt_ms: 2.5 + cycle,
                last_success_wall_time: 12345,
                latest_error: null,
                observation_messages_sent: cycle,
                claim_messages_sent: cycle,
                payload_bytes_sent: 100,
                send_failures: cycle === 0 ? 0 : 1,
              }],
            };
          }

          throw new Error(`unexpected URL ${url}`);
        };

        await updateAll();
        cycle += 1;
        await updateAll();

        process.stdout.write(JSON.stringify({
          runState: document.getElementById("compact-run-state").textContent,
          elapsed: document.getElementById("compact-elapsed-time").textContent,
          networkSummary: document.getElementById("network-summary").textContent,
          scenarioControlState: document.getElementById("scenario-control-state").textContent,
          drawnCanvases,
          refreshIntervals: intervals,
        }));
        """
        output = self.run_dashboard_script(fixture)

        self.assertEqual("Running", output["runState"])
        self.assertEqual("11.0 s", output["elapsed"])
        self.assertIn("3/3 nodes reporting", output["networkSummary"])
        self.assertIn("3 recently reachable peers", output["networkSummary"])
        self.assertIn("0 failing", output["networkSummary"])
        self.assertIn("3 send failures", output["networkSummary"])
        self.assertNotIn("6 send failures", output["networkSummary"])
        self.assertIn("Running Random Search of Area", output["scenarioControlState"])
        self.assertIn("0:belief-map-1", output["drawnCanvases"])
        self.assertIn("1:belief-map-1", output["drawnCanvases"])
        self.assertNotIn("0:operational-map", output["drawnCanvases"])
        self.assertNotIn("1:operational-map", output["drawnCanvases"])
        self.assertTrue(any(delay <= 1000 for delay in output["refreshIntervals"]))

    def test_belief_maps_render_only_own_aircraft_snail_trails(self) -> None:
        fixture = """
        state.scenarioRunId = "trail-run";
        state.scenarioId = "seeded_stationary_patrol";
        state.truth = { objects: [], simulation_time: 40 };

        function statusFor(droneId, timestamp, runId, trail) {
          const baseLatitude = 37.7750 + (droneId * 0.0002);
          const baseLongitude = -122.4230 - (droneId * 0.0002);
          return {
            state: "running",
            schema_version: "v1",
            sharing_enabled: true,
            paused: false,
            selected_playbook: "report_continue",
            playbook_config_version: 2,
            scenario_id: "seeded_stationary_patrol",
            scenario_run_id: runId,
            scenario_seed: 20261106,
            timestamp,
            current_mode: "patrol",
            assigned_objective: null,
            response_task: null,
            response_tasks: [],
            response_queue_depth: 0,
            altitude_m: 100,
            speed_mps: 12,
            heading_deg: 90,
            latitude: baseLatitude,
            longitude: baseLongitude,
            battery_percent: 90,
            camera_footprint: [
              { latitude: baseLatitude + 0.00003, longitude: baseLongitude - 0.00003 },
              { latitude: baseLatitude + 0.00003, longitude: baseLongitude + 0.00003 },
              { latitude: baseLatitude - 0.00003, longitude: baseLongitude },
            ],
            flight_trail: trail,
            diagnostics: { peer_connection_state: [] },
          };
        }

        state.statuses.set(1, statusFor(1, 40, "trail-run", [
          { latitude: 37.77501, longitude: -122.42301, timestamp: 10 },
          { latitude: 37.77502, longitude: -122.42302, timestamp: 11 },
          { latitude: "invalid", longitude: -122.42303, timestamp: 39 },
          { latitude: 37.77504, longitude: -122.42304, timestamp: 39 },
        ]));
        state.statuses.set(2, statusFor(2, 40, "trail-run", [
          { latitude: 37.77541, longitude: -122.42341, timestamp: 12 },
          { latitude: 37.77542, longitude: -122.42342, timestamp: 35 },
        ]));
        state.statuses.set(3, statusFor(3, 40, "trail-run", [
          { latitude: 37.77581, longitude: -122.42381, timestamp: 39 },
        ]));
        state.beliefs.set(1, { simulation_time: 40, unresolved_observation_count: 0, objects: [] });
        state.beliefs.set(2, { simulation_time: 40, unresolved_observation_count: 0, objects: [] });
        state.beliefs.set(3, { simulation_time: 40, unresolved_observation_count: 0, objects: [] });

        function trailOps(canvasId, color) {
          return document.getElementById(canvasId).drawOps.filter((op) => (
            op.op === "arc"
            && Math.abs(Number(op.args[2]) - 3.2) < 0.001
            && String(op.fillStyle).startsWith(color)
          ));
        }

        function alphaValues(ops) {
          return ops.map((op) => parseInt(String(op.fillStyle).slice(-2), 16));
        }

        renderMaps();
        const initial = {
          redOwnTrailCount: trailOps("belief-map-1", "#ef4444").length,
          redBlueTrailCount: trailOps("belief-map-1", "#3b82f6").length,
          redGreenTrailCount: trailOps("belief-map-1", "#22c55e").length,
          blueOwnTrailCount: trailOps("belief-map-2", "#3b82f6").length,
          blueRedTrailCount: trailOps("belief-map-2", "#ef4444").length,
          blueGreenTrailCount: trailOps("belief-map-2", "#22c55e").length,
          greenOwnTrailCount: trailOps("belief-map-3", "#22c55e").length,
          greenRedTrailCount: trailOps("belief-map-3", "#ef4444").length,
          greenBlueTrailCount: trailOps("belief-map-3", "#3b82f6").length,
          redAlphaValues: alphaValues(trailOps("belief-map-1", "#ef4444")),
        };

        state.showFlightTrails = false;
        renderMaps();
        const redWithTruthTrailToggleOff = trailOps("belief-map-1", "#ef4444").length;

        Date.now = () => 999999999999;
        renderMaps();
        const redPausedAlphaValues = alphaValues(trailOps("belief-map-1", "#ef4444"));

        state.scenarioRunId = "new-trail-run";
        renderMaps();
        const redAfterRunReset = trailOps("belief-map-1", "#ef4444").length;

        state.statuses.set(1, {
          ...state.statuses.get(1),
          scenario_run_id: "new-trail-run",
          timestamp: 5,
          flight_trail: [{ latitude: 37.77601, longitude: -122.42401, timestamp: 5 }],
        });
        renderMaps();
        const redNewRunTrailCount = trailOps("belief-map-1", "#ef4444").length;

        process.stdout.write(JSON.stringify({
          ...initial,
          redWithTruthTrailToggleOff,
          redPausedAlphaValues,
          redAfterRunReset,
          redNewRunTrailCount,
        }));
        """
        output = self.run_dashboard_script(fixture)

        self.assertIn('id="belief-trail-toggle" type="checkbox" checked', launch_simulation.DASHBOARD_HTML)
        self.assertEqual(2, output["redOwnTrailCount"])
        self.assertEqual(0, output["redBlueTrailCount"])
        self.assertEqual(0, output["redGreenTrailCount"])
        self.assertEqual(2, output["blueOwnTrailCount"])
        self.assertEqual(0, output["blueRedTrailCount"])
        self.assertEqual(0, output["blueGreenTrailCount"])
        self.assertEqual(1, output["greenOwnTrailCount"])
        self.assertEqual(0, output["greenRedTrailCount"])
        self.assertEqual(0, output["greenBlueTrailCount"])
        self.assertEqual(2, len(output["redAlphaValues"]))
        self.assertLess(output["redAlphaValues"][0], output["redAlphaValues"][1])
        self.assertEqual(2, output["redWithTruthTrailToggleOff"])
        self.assertEqual(output["redAlphaValues"], output["redPausedAlphaValues"])
        self.assertEqual(0, output["redAfterRunReset"])
        self.assertEqual(1, output["redNewRunTrailCount"])

    def test_harness_uses_controlled_timers_abort_controller_and_stdin_safe_fixtures(self) -> None:
        fixture = """
        const literal = `fixture with backticks works`;
        let fired = false;
        const timerId = setTimeout(() => { fired = true; }, 5);
        clearTimeout(timerId);
        await runScheduledTimeout(timerId);

        const controller = new AbortController();
        const abortedFetch = fetch("http://example.invalid/hang", { signal: controller.signal })
          .then(() => "resolved", (error) => error.name);
        controller.abort();
        const abortName = await abortedFetch;

        process.stdout.write(JSON.stringify({
          literal,
          fired,
          abortName,
          timeoutMetrics,
          fetchCalls,
        }));
        """
        output = self.run_dashboard_script(fixture)

        self.assertEqual("fixture with backticks works", output["literal"])
        self.assertFalse(output["fired"])
        self.assertEqual("AbortError", output["abortName"])
        self.assertGreaterEqual(output["timeoutMetrics"]["cleared"], 1)
        self.assertTrue(any(call["url"] == "http://example.invalid/hang" for call in output["fetchCalls"]))

    def test_harness_preserves_automatic_startup_and_recurring_refresh_schedule(self) -> None:
        fixture = """
        await flushMicrotasks();
        process.stdout.write(JSON.stringify({
          initialFetches: fetchCalls.map((call) => call.url),
          pendingTimeoutCount: scheduledTimeouts.size,
          refreshIntervals: intervals,
        }));
        """
        output = self.run_dashboard_script(fixture)

        self.assertGreater(len(output["initialFetches"]), 0)
        self.assertGreater(output["pendingTimeoutCount"], 0)
        self.assertTrue(any(delay <= 1000 for delay in output["refreshIntervals"]))

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
        self.assertIn("<td>Orbiting</td>", output["assignmentDetails"])
        self.assertIn("<td>Unassigned</td>", output["assignmentDetails"])
        self.assertIn("<td>Reported</td>", output["assignmentDetails"])
        self.assertNotIn("<td>Patrolling</td>", output["assignmentDetails"])
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

    def test_per_member_team_configuration_acknowledgement_accepts_matching_members_and_capability_order(self) -> None:
        fixture = """
        const expectedTeam = {
          "1": { role: "isr", capabilities: ["atr_observe", "isr_observe"], available: true, payload_inventory: {}, camera_mount: "steerable" },
          "2": { role: "payload", capabilities: ["payload_drop"], available: true, payload_inventory: { water: 2, rescue_kit: 1 }, camera_mount: "fixed" },
          "3": { role: "payload", capabilities: ["payload_drop", "atr_observe"], available: false, payload_inventory: { water: 2 }, camera_mount: "fixed" },
        };
        const actualMembers = {
          1: { role: "isr", capabilities: ["isr_observe", "atr_observe"], available: true, payload_inventory: {}, camera_mount: "steerable" },
          2: { role: "payload", capabilities: ["payload_drop"], available: true, payload_inventory: { rescue_kit: 1, water: 2 }, camera_mount: "fixed" },
          3: { role: "payload", capabilities: ["atr_observe", "payload_drop"], available: false, payload_inventory: { water: 2 }, camera_mount: "fixed" },
        };
        const controlPayload = {
          scenario_seed: 20261106,
          scenario_id: "seeded_stationary_patrol",
          scenario_run_id: "team-ack-run",
          initial_observer_id: 1,
          next_observer_id: 2,
          selected_playbook: "report_continue",
          sharing_enabled: true,
          paused: false,
        };
        fetchJson = async (url) => {
          const drone = drones.find((item) => url.startsWith(item.url));
          if (!drone) {
            throw new Error(`unexpected URL ${url}`);
          }
          if (url.endsWith("/control")) {
            return { ...controlPayload };
          }
          if (url.endsWith("/status")) {
            return { team_configuration: actualMembers[drone.id] };
          }
          throw new Error(`unexpected URL ${url}`);
        };
        await waitForConfigurationAcknowledgement({
          restart_scenario: true,
          scenario_seed: 20261106,
          scenario: "seeded_stationary_patrol",
          scenario_run_id: "team-ack-run",
          initial_observer_id: 1,
          next_observer_id: 2,
          playbook: "report_continue",
          sharing_enabled: true,
          paused: false,
          team_configuration: expectedTeam,
        }, { timeoutMs: 1000 });
        process.stdout.write(JSON.stringify({
          status: state.scenarioControlStatus,
          message: state.scenarioControlMessage,
        }));
        """
        output = self.run_dashboard_script(fixture)

        self.assertEqual("running", output["status"])
        self.assertIn("All nodes acknowledged", output["message"])

    def test_per_member_team_configuration_acknowledgement_reports_only_changed_green_inventory(self) -> None:
        fixture = """
        let now = 1000;
        Date.now = () => { now += 3; return now; };
        const expectedTeam = {
          "1": { role: "isr", capabilities: ["atr_observe", "isr_observe"], available: true, payload_inventory: {}, camera_mount: "steerable" },
          "2": { role: "payload", capabilities: ["payload_drop"], available: true, payload_inventory: { water: 1 }, camera_mount: "fixed" },
          "3": { role: "payload", capabilities: ["payload_drop"], available: true, payload_inventory: { water: 2 }, camera_mount: "fixed" },
        };
        const actualMembers = {
          1: { role: "isr", capabilities: ["isr_observe", "atr_observe"], available: true, payload_inventory: {}, camera_mount: "steerable" },
          2: { role: "payload", capabilities: ["payload_drop"], available: true, payload_inventory: { water: 1 }, camera_mount: "fixed" },
          3: { role: "payload", capabilities: ["payload_drop"], available: true, payload_inventory: { water: 0 }, camera_mount: "fixed" },
        };
        const controlPayload = {
          scenario_seed: 20261106,
          scenario_id: "seeded_stationary_patrol",
          scenario_run_id: "team-ack-run",
          initial_observer_id: 1,
          next_observer_id: 2,
          selected_playbook: "report_continue",
          sharing_enabled: true,
          paused: false,
        };
        fetchJson = async (url) => {
          const drone = drones.find((item) => url.startsWith(item.url));
          if (!drone) {
            throw new Error(`unexpected URL ${url}`);
          }
          if (url.endsWith("/control")) {
            return { ...controlPayload };
          }
          if (url.endsWith("/status")) {
            return { team_configuration: actualMembers[drone.id] };
          }
          throw new Error(`unexpected URL ${url}`);
        };
        const promise = waitForConfigurationAcknowledgement({
          restart_scenario: true,
          scenario_seed: 20261106,
          scenario: "seeded_stationary_patrol",
          scenario_run_id: "team-ack-run",
          initial_observer_id: 1,
          next_observer_id: 2,
          playbook: "report_continue",
          sharing_enabled: true,
          paused: false,
          team_configuration: expectedTeam,
        }, { timeoutMs: 5 }).then(() => "resolved", (error) => error.message);
        await flushMicrotasks();
        await runScheduledTimeouts();
        const message = await promise;
        process.stdout.write(JSON.stringify({ message }));
        """
        output = self.run_dashboard_script(fixture)

        self.assertIn("Green: payload_inventory.water expected 2, actual 0", output["message"])
        self.assertNotIn("Red:", output["message"])
        self.assertNotIn("Blue:", output["message"])

    def test_per_member_team_configuration_acknowledgement_rejects_field_specific_mismatches(self) -> None:
        fixture = """
        const expected = { role: "payload", capabilities: ["payload_drop", "atr_observe"], available: true, payload_inventory: { water: 2 }, camera_mount: "fixed" };
        const cases = [
          {
            name: "missing capability",
            actual: { role: "payload", capabilities: ["payload_drop"], available: true, payload_inventory: { water: 2 }, camera_mount: "fixed" },
            expectedMessage: 'capabilities expected ["atr_observe","payload_drop"], actual ["payload_drop"]',
          },
          {
            name: "changed availability",
            actual: { role: "payload", capabilities: ["payload_drop", "atr_observe"], available: false, payload_inventory: { water: 2 }, camera_mount: "fixed" },
            expectedMessage: "available expected true, actual false",
          },
          {
            name: "wrong camera",
            actual: { role: "payload", capabilities: ["payload_drop", "atr_observe"], available: true, payload_inventory: { water: 2 }, camera_mount: "steerable" },
            expectedMessage: 'camera_mount expected "fixed", actual "steerable"',
          },
          {
            name: "missing required field",
            actual: { role: "payload", available: true, payload_inventory: { water: 2 }, camera_mount: "fixed" },
            expectedMessage: 'capabilities expected ["payload_drop","atr_observe"], actual undefined',
          },
          {
            name: "same role different capabilities",
            actual: { role: "payload", capabilities: ["payload_drop"], available: true, payload_inventory: { water: 2 }, camera_mount: "fixed" },
            expectedMessage: 'capabilities expected ["atr_observe","payload_drop"], actual ["payload_drop"]',
          },
        ];
        const messages = cases.map((item) => ({
          name: item.name,
          message: memberConfigurationMismatch(item.actual, expected),
          expectedMessage: item.expectedMessage,
        }));
        process.stdout.write(JSON.stringify({ messages }));
        """
        output = self.run_dashboard_script(fixture)

        for item in output["messages"]:
            self.assertEqual(item["expectedMessage"], item["message"], item["name"])

    def test_assignment_details_persist_by_run_and_typed_task_identity_while_content_updates(self) -> None:
        fixture = """
        function snapshotFor(droneId, task) {
          return {
            droneId,
            droneName: droneNameForId(droneId),
            online: true,
            receivedAt: Date.now() / 1000,
            simulationTime: task.simulationTime || 100,
            currentMode: task.currentMode || "patrol",
            assignedObjective: task.assignedObjective || null,
            responseTasks: [task],
          };
        }

        state.scenarioRunId = "run-a";
        state.statuses.set(1, { current_mode: "orbiting", assigned_objective: "sig-a" });
        state.taskSnapshotsByDrone.set(1, snapshotFor(1, {
          task_id: "task-a",
          task_type: "isr_observe",
          signature_id: "sig-a",
          object_label: "antenna",
          assigned_drone_id: 1,
          assigned_drone_name: "Red",
          status: "orbiting",
          last_observation_at: 90,
          simulationTime: 100,
          currentMode: "orbiting",
          assignedObjective: "sig-a",
        }));
        state.taskSnapshotsByDrone.set(2, snapshotFor(2, {
          task_id: "task-b",
          task_type: "payload_drop",
          contact_ref: "sig-b",
          signature_id: "sig-b",
          object_label: "person",
          assigned_drone_id: null,
          assigned_drone_name: null,
          status: "queued",
          requested_payload_type: "water",
          requested_payload_quantity: 1,
          created_at: 80,
          simulationTime: 100,
        }));
        renderAssignments();

        const antennaKey = assignmentRowKey(buildAssignmentRows().find((row) => row.displayName === "Antenna"));
        state.assignmentDetailsOpenByKey.set(antennaKey, true);

        state.taskSnapshotsByDrone.clear();
        state.taskSnapshotsByDrone.set(2, snapshotFor(2, {
          task_id: "task-b",
          task_type: "payload_drop",
          contact_ref: "sig-b",
          signature_id: "sig-b",
          object_label: "person",
          assigned_drone_id: null,
          assigned_drone_name: null,
          status: "queued",
          requested_payload_type: "water",
          requested_payload_quantity: 1,
          created_at: 70,
          simulationTime: 130,
        }));
        state.taskSnapshotsByDrone.set(1, snapshotFor(1, {
          task_id: "task-a",
          task_type: "isr_observe",
          signature_id: "sig-a",
          object_label: "antenna",
          assigned_drone_id: 1,
          assigned_drone_name: "Red",
          status: "orbiting",
          last_observation_at: 110,
          simulationTime: 130,
          currentMode: "orbiting",
          assignedObjective: "sig-a",
        }));
        renderAssignments();
        const openAfterRefresh = document.getElementById("assignment-details").innerHTML;

        state.assignmentDetailsOpenByKey.set(antennaKey, false);
        renderAssignments();
        const closedAfterRefresh = document.getElementById("assignment-details").innerHTML;

        state.scenarioRunId = "run-b";
        renderAssignments();
        const afterRunChange = document.getElementById("assignment-details").innerHTML;

        process.stdout.write(JSON.stringify({
          antennaKey,
          openAfterRefresh,
          closedAfterRefresh,
          afterRunChange,
        }));
        """
        output = self.run_dashboard_script(fixture)

        self.assertIn('data-assignment-row-key="run-a:isr_observe:task-a" open', output["openAfterRefresh"])
        self.assertIn("20.0 s", output["openAfterRefresh"])
        self.assertIn('data-task-identity="isr_observe:task-a"', output["openAfterRefresh"])
        self.assertIn('data-cancel-task-id="task-a"', output["openAfterRefresh"])
        self.assertNotIn('data-assignment-row-key="run-a:isr_observe:task-a" open', output["closedAfterRefresh"])
        self.assertNotIn('data-assignment-row-key="run-b:isr_observe:task-a" open', output["afterRunChange"])

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

        self.assertEqual(output["assignmentDetails"].count("<td>Orbiting</td>"), 3)
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

        self.assertIn("Moving to observation position (peer-reported)", output["assignmentDetails"])
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
            self.assertIn("<td>Orbiting</td>", output)
            self.assertIn("<td>Unassigned</td>", output)
            self.assertIn("<td>Queued</td>", output)
            self.assertNotIn("waiting for eligible aircraft", output)
            self.assertNotIn("assigned_remote", output)

        self.assertEqual(stale_first, queued_first)

    def test_team_configuration_summary_includes_payload_inventory(self) -> None:
        fixture = """
        const now = Date.now() / 1000;

        state.taskSnapshotsByDrone.set(1, {
          droneId: 1,
          droneName: "Red",
          online: true,
          receivedAt: now,
          simulationTime: 10,
          currentMode: "patrol",
          assignedObjective: null,
          responseTasks: [],
          teamConfiguration: {
            role: "payload",
            capabilities: ["atr_observe", "payload_drop"],
            available: true,
            payload_inventory: { water: 2, medkit: 1 },
            camera_mount: "fixed",
          },
        });
        renderControls();
        """
        output = self.run_dashboard_script(fixture)

        self.assertIn("Red: Payload", output["teamConfigState"])
        self.assertIn("atr_observe/payload_drop", output["teamConfigState"])
        self.assertIn("inventory medkit×1/water×2", output["teamConfigState"])

    def test_historical_payload_execution_record_does_not_count_as_current_payload_activity(self) -> None:
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
            task_id: "task-active-isr",
            task_type: "isr_observe",
            signature_id: "sig-active",
            object_label: "antenna",
            assigned_drone_id: 2,
            assigned_drone_name: "Blue",
            status: "orbiting",
            last_observation_at: 46,
          },
          responseTasks: [
            {
              task_id: "task-active-isr",
              task_type: "isr_observe",
              signature_id: "sig-active",
              object_label: "antenna",
              assigned_drone_id: 2,
              assigned_drone_name: "Blue",
              status: "orbiting",
              last_observation_at: 46,
            },
            {
              task_id: "payload-old",
              task_type: "payload_drop",
              contact_ref: "sig-recipient",
              signature_id: "sig-recipient",
              object_label: "person",
              assigned_drone_id: 2,
              assigned_drone_name: "Blue",
              status: "approaching_payload",
              requested_payload_type: "water",
              requested_payload_quantity: 1,
              reservation_state: "reserved",
              created_at: 10,
              guidance_evidence_available: true,
              guidance_evidence_at: 35,
            },
          ],
        });

        renderAssignments();
        renderMissionSummary();
        """
        output = self.run_dashboard_script(fixture)
        payload_row = next(row for row in output["assignmentRows"] if row["taskType"] == "payload_drop")

        self.assertEqual("Person", payload_row["displayName"])
        self.assertEqual("Unassigned", payload_row["assignedDroneName"])
        self.assertIsNone(payload_row["assignedDroneId"])
        self.assertEqual("Historical assignment", payload_row["taskState"])
        self.assertEqual("Supply delivery", payload_row["taskLabel"])
        self.assertIn("<td>Historical assignment</td>", output["assignmentDetails"])
        self.assertNotIn("<td>Approaching recipient</td>", output["assignmentDetails"])
        self.assertNotIn("<td>Holding for visual acquisition</td>", output["assignmentDetails"])
        self.assertEqual("0", output["payloadApproaching"])
        self.assertEqual("0", output["payloadHolding"])
        self.assertEqual("1", output["isrAircraft"])

    def test_payload_states_reflected_in_summary_aircraft_card_and_tasks(self) -> None:
        fixture = """
        const now = Date.now() / 1000;
        state.statuses.set(2, {
          current_mode: "approaching_payload",
          assigned_objective: "sig-recipient",
          diagnostics: { payload_drop: { available_inventory_after_reservations: { water: 1 } } },
        });
        state.taskSnapshotsByDrone.set(2, {
          droneId: 2,
          droneName: "Blue",
          online: true,
          receivedAt: now,
          simulationTime: 40,
          currentMode: "approaching_payload",
          assignedObjective: "sig-recipient",
          responseTask: {
            task_id: "payload-task",
            task_type: "payload_drop",
            contact_ref: "sig-recipient",
            signature_id: "sig-recipient",
            object_label: "person",
            assigned_drone_id: 2,
            assigned_drone_name: "Blue",
            status: "approaching_payload",
            requested_payload_type: "water",
            requested_payload_quantity: 2,
            reserved_payload_quantity: 2,
            reservation_state: "reserved",
            created_at: 10,
            guidance_evidence_available: true,
            guidance_evidence_at: 35,
          },
          responseTasks: [],
          teamConfiguration: {
            role: "payload",
            capabilities: ["payload_drop"],
            available: true,
            payload_inventory: { water: 3 },
            camera_mount: "fixed",
          },
        });
        renderAssignments();
        renderMissionSummary();
        renderAircraftCards();
        """
        output = self.run_dashboard_script(fixture)

        self.assertEqual("1", output["supplyContacts"])
        self.assertEqual("1", output["payloadApproaching"])
        self.assertIn("Approaching recipient", output["assignmentDetails"])
        self.assertIn("Approaching recipient", output["aircraftCards"])
        self.assertIn("water×3", output["aircraftCards"])
        self.assertIn("water×2", output["aircraftCards"])
        self.assertIn("Supply delivery", output["assignmentDetails"])

    def test_blocked_payload_reason_and_aircraft_card_follow_current_owner_state(self) -> None:
        fixture = """
        const now = Date.now() / 1000;

        state.taskSnapshotsByDrone.set(1, {
          droneId: 1,
          droneName: "Red",
          online: true,
          receivedAt: now,
          simulationTime: 20,
          currentMode: "patrol",
          assignedObjective: null,
          responseTasks: [{
            task_id: "payload-water-person",
            task_type: "payload_drop",
            contact_ref: "sig-person",
            signature_id: "sig-person",
            object_label: "person",
            assigned_drone_id: null,
            assigned_drone_name: null,
            status: "queued",
            requested_payload_type: "water",
            requested_payload_quantity: 2,
            reservation_state: "not_reserved",
            created_at: 10,
          }],
          teamConfiguration: {
            role: "isr",
            capabilities: ["atr_observe", "isr_observe"],
            available: true,
            payload_inventory: {},
            camera_mount: "steerable",
          },
        });
        state.taskSnapshotsByDrone.set(3, {
          droneId: 3,
          droneName: "Green",
          online: true,
          receivedAt: now,
          simulationTime: 20,
          currentMode: "patrol",
          assignedObjective: null,
          responseTask: null,
          responseTasks: [{
            task_id: "payload-water-person",
            task_type: "payload_drop",
            contact_ref: "sig-person",
            signature_id: "sig-person",
            object_label: "person",
            assigned_drone_id: null,
            assigned_drone_name: null,
            status: "queued",
            requested_payload_type: "water",
            requested_payload_quantity: 2,
            reservation_state: "not_reserved",
            created_at: 10,
          }],
          teamConfiguration: {
            role: "payload",
            capabilities: ["payload_drop"],
            available: true,
            payload_inventory: { water: 0, rescue_kit: 1 },
            camera_mount: "fixed",
          },
        });
        state.statuses.set(3, {
          current_mode: "patrol",
          assigned_objective: null,
          diagnostics: { payload_drop: { available_inventory_after_reservations: { water: 0, rescue_kit: 1 } } },
        });

        renderAssignments();
        renderAircraftCards();
        renderMissionSummary();
        """
        output = self.run_dashboard_script(fixture)

        self.assertIn("Mission assignments", launch_simulation.DASHBOARD_HTML)
        self.assertIn("Supply delivery", output["assignmentDetails"])
        self.assertIn("Waiting — Green needs 2 Water; 0 available", output["assignmentDetails"])
        self.assertIn("<td>Unassigned</td>", output["assignmentDetails"])
        self.assertNotIn("PAYLOAD", output["assignmentDetails"])
        self.assertNotIn("Assigned contact</dt><dd>Person", output["aircraftCards"])
        self.assertIn("rescue_kit×1", output["aircraftCards"])

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
        renderMissionSummary();
        """
        output = self.run_dashboard_script(fixture)
        antenna_row = next(row for row in output["assignmentRows"] if row["displayName"] == "Antenna")
        vehicle_row = next(row for row in output["assignmentRows"] if row["displayName"] == "Vehicle")

        self.assertIn("Blue: orbiting Antenna", output["assignmentDetails"])
        self.assertIn("Antenna", output["assignmentDetails"])
        self.assertIn("Vehicle", output["assignmentDetails"])
        self.assertEqual("Blue", antenna_row["assignedDroneName"])
        self.assertEqual("Orbiting", antenna_row["taskState"])
        self.assertEqual("Unassigned", vehicle_row["assignedDroneName"])
        self.assertIsNone(vehicle_row["assignedDroneId"])
        self.assertEqual("Historical assignment", vehicle_row["taskState"])
        self.assertEqual(2, vehicle_row["historicalAssignedDroneId"])
        self.assertEqual(output["assignmentDetails"].count("<td>Orbiting</td>"), 1)
        self.assertIn("<td>Historical assignment</td>", output["assignmentDetails"])
        self.assertEqual("1", output["isrAircraft"])
        self.assertIn("not active on owner", output["assignmentDetails"])
        self.assertNotIn("mock-search-signature", output["assignmentDetails"])
