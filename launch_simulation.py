#!/usr/bin/env python3
"""Launch three independent simulated drone processes and a browser dashboard."""

from __future__ import annotations

import argparse
import contextlib
import http.server
import json
import os
import socket
import socketserver
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from pathlib import Path


DEFAULT_DASHBOARD_HOST = "127.0.0.1"
DEFAULT_DASHBOARD_PORT = 8000
DEFAULT_DRONE_HOST = "127.0.0.1"
DEFAULT_DRONE_PORTS = (9101, 9102, 9103)
APPLICATION_BUILD = "priority-response-stage-a-2026-11-06"
SCENARIO_NAME = "Seeded noisy overlapping moving-search patrol"
DEFAULT_SCENARIO_SEED = 20261106
RANDOM_PRIORITY_OBJECT_COUNT = 0
DEFAULT_PLAYBOOK = "report_continue"
DEFAULT_SCENARIO_ID = "seeded_stationary_patrol"
MOVING_HANDOFF_SCENARIO_ID = "moving_person_handoff"
DEFAULT_INITIAL_OBSERVER_ID = 1
DEFAULT_NEXT_OBSERVER_ID = 2
DRONE_SPECS = (
    {"id": 1, "name": "Red", "color": "#ef4444"},
    {"id": 2, "name": "Blue", "color": "#3b82f6"},
    {"id": 3, "name": "Green", "color": "#22c55e"},
)


DASHBOARD_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Drone Simulation Dashboard</title>
  <style>
    :root {
      color-scheme: dark;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #0f172a;
      color: #e2e8f0;
    }

    body {
      margin: 0;
      padding: 2rem;
    }

    h1 {
      margin: 0 0 0.25rem;
    }

    button {
      background: #334155;
      border: 1px solid #64748b;
      border-radius: 8px;
      color: #e2e8f0;
      cursor: pointer;
      font-weight: 700;
      padding: 0.55rem 0.8rem;
    }

    button:hover {
      background: #475569;
    }

    canvas {
      background: #020617;
      border-radius: 10px;
      display: block;
      width: 100%;
    }

    .belief-canvas {
      height: 420px;
    }

    .truth-canvas {
      height: clamp(420px, 65vh, 760px);
    }

    details {
      margin-top: 0.75rem;
    }

    summary {
      color: #cbd5e1;
      cursor: pointer;
      font-weight: 700;
    }

    .controls {
      align-items: center;
      display: flex;
      flex-wrap: wrap;
      gap: 0.75rem;
      margin: 1rem 0;
    }

    .control-panel {
      background: #0f172a;
      border: 1px solid #334155;
      border-radius: 10px;
      margin: 0.75rem 0;
      padding: 0.75rem;
    }

    .control-panel h3 {
      font-size: 0.95rem;
      margin: 0 0 0.5rem;
    }

    .control-note {
      color: #94a3b8;
      display: block;
      font-size: 0.8rem;
      margin-top: 0.25rem;
    }

    .pill {
      background: #1e293b;
      border: 1px solid #334155;
      border-radius: 999px;
      color: #cbd5e1;
      padding: 0.4rem 0.75rem;
    }

    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 1rem;
      margin-bottom: 1rem;
    }

    .map-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 1rem;
      margin-top: 1rem;
      width: 100%;
    }

    .map-card.truth {
      grid-column: 1 / -1;
    }

    @media (max-width: 1150px) {
      .map-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }
    }

    @media (max-width: 760px) {
      body {
        padding: 1rem;
      }

      .map-grid {
        grid-template-columns: minmax(0, 1fr);
      }
    }

    .map-card.belief {
      border-color: var(--drone-color, #334155);
      border-width: 3px;
      min-width: 0;
    }

    .map-card.belief .map-title {
      justify-content: flex-start;
      gap: 0.5rem;
    }

    .map-card.belief .map-title .pill {
      margin-left: auto;
    }

    .diagnostics-grid {
      font-size: 0.85rem;
      grid-template-columns: 1.4fr auto;
      margin-top: 0.5rem;
    }

    .card,
    .map-card {
      background: #1e293b;
      border: 1px solid #334155;
      border-radius: 12px;
      padding: 1rem;
      box-shadow: 0 10px 30px rgb(0 0 0 / 25%);
    }

    .card.offline {
      border-color: #ef4444;
    }

    .name,
    .map-title {
      align-items: center;
      display: flex;
      font-size: 1.15rem;
      font-weight: 700;
      justify-content: space-between;
      margin-bottom: 0.75rem;
    }

    .status {
      border-radius: 999px;
      font-size: 0.75rem;
      padding: 0.25rem 0.6rem;
      text-transform: uppercase;
    }

    .online {
      background: #16a34a;
      color: #f0fdf4;
    }

    .offline .status {
      background: #dc2626;
      color: #fef2f2;
    }

    dl {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 0.5rem;
      margin: 0;
    }

    dt {
      color: #94a3b8;
    }

    dd {
      font-variant-numeric: tabular-nums;
      margin: 0;
    }

    .object-table {
      border-collapse: collapse;
      font-size: 0.82rem;
      margin-top: 0.65rem;
      width: 100%;
    }

    .object-table th,
    .object-table td {
      border-bottom: 1px solid #334155;
      padding: 0.35rem;
      text-align: left;
      vertical-align: top;
    }

    .object-table th {
      color: #94a3b8;
      font-weight: 700;
    }

    .count-row {
      display: grid;
      gap: 0.4rem;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      margin-bottom: 0.5rem;
    }

    .count-box {
      background: #0f172a;
      border: 1px solid #334155;
      border-radius: 8px;
      padding: 0.45rem;
    }

    .count-box strong {
      display: block;
      font-size: 1.1rem;
    }

    .empty-state {
      color: #94a3b8;
      font-style: italic;
      margin-top: 0.5rem;
    }

    .details {
      color: #cbd5e1;
      font-size: 0.9rem;
      line-height: 1.4;
      margin-top: 0.75rem;
      min-height: 4rem;
    }

    .details ul {
      margin: 0.35rem 0 0;
      padding-left: 1.2rem;
    }

    .legend {
      color: #94a3b8;
      font-size: 0.85rem;
      margin-top: 0.5rem;
    }

    .assignment-table {
      border-collapse: collapse;
      font-size: 0.9rem;
      width: 100%;
    }

    .assignment-table th,
    .assignment-table td {
      border-bottom: 1px solid #334155;
      padding: 0.45rem;
      text-align: left;
    }

    .swatch {
      border-radius: 999px;
      display: inline-block;
      height: 0.75rem;
      margin-right: 0.25rem;
      vertical-align: middle;
      width: 0.75rem;
    }

    .sensor-label {
      background: #7c2d12;
      border: 1px solid #fb923c;
      border-radius: 8px;
      color: #fed7aa;
      display: inline-block;
      font-weight: 800;
      letter-spacing: 0.04em;
      margin-bottom: 0.5rem;
      padding: 0.35rem 0.5rem;
    }

    .trace-table,
    .network-table {
      border-collapse: collapse;
      font-size: 0.82rem;
      margin-top: 0.5rem;
      width: 100%;
    }

    .trace-table th,
    .trace-table td,
    .network-table th,
    .network-table td {
      border-bottom: 1px solid #334155;
      padding: 0.35rem;
      text-align: left;
      vertical-align: top;
    }

    pre.raw-json {
      background: #020617;
      border: 1px solid #334155;
      border-radius: 8px;
      overflow: auto;
      padding: 0.5rem;
      white-space: pre-wrap;
    }

    .tabs {
      align-items: flex-end;
      border-bottom: 1px solid #334155;
      display: flex;
      flex-wrap: wrap;
      gap: 0.25rem;
      margin: 1rem 0 0;
    }

    .tab-button {
      background: transparent;
      border: 0;
      border-bottom: 3px solid transparent;
      border-radius: 8px 8px 0 0;
      color: #94a3b8;
      padding: 0.75rem 1rem;
    }

    .tab-button:hover,
    .tab-button:focus-visible {
      background: #0f172a;
      color: #e2e8f0;
      outline: 2px solid #60a5fa;
      outline-offset: -2px;
    }

    .tab-button[aria-selected="true"] {
      background: #1e293b;
      border-bottom-color: #60a5fa;
      color: #f8fafc;
    }

    .tab-panel {
      background: #0f172a;
      border: 1px solid #334155;
      border-top: 0;
      display: none;
      padding: 1rem 0 0;
    }

    .tab-panel.active {
      display: block;
    }

    .shared-inspector {
      margin-top: 1rem;
    }

    .json-toolbar {
      align-items: center;
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem;
      margin: 0.5rem 0;
    }

    .json-area {
      background: #020617;
      border: 1px solid #334155;
      border-radius: 8px;
      color: #e2e8f0;
      display: block;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
      height: 320px;
      overflow: auto;
      padding: 0.5rem;
      resize: vertical;
      white-space: pre;
      width: 100%;
    }

    .inspection-note {
      color: #fbbf24;
      font-size: 0.85rem;
      margin: 0.5rem 0;
    }

    .compact-header {
      align-items: center;
      background: #1e293b;
      border: 1px solid #334155;
      border-radius: 12px;
      display: flex;
      flex-wrap: wrap;
      gap: 0.75rem;
      justify-content: space-between;
      margin: 1rem 0;
      padding: 0.8rem 1rem;
    }

    .compact-actions {
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem;
    }

    .status-banner {
      background: #7f1d1d;
      border: 1px solid #f87171;
      border-radius: 10px;
      color: #fee2e2;
      font-weight: 700;
      margin: 0.75rem 0;
      padding: 0.75rem;
    }

    .evidence-grid {
      display: grid;
      gap: 0.35rem;
      grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
      margin-top: 0.5rem;
    }

    .verification {
      background: #0f172a;
      border: 1px solid #334155;
      border-radius: 8px;
      padding: 0.35rem 0.5rem;
    }
  </style>
</head>
<body>
  <h1>Drone Simulation Dashboard</h1>
  <p><strong>Three independent drone programs. Simulated sensors. Real peer network messages. Each drone maintains its own map.</strong></p>
  <p>Three independent moving drones maintain node-local belief maps, sense only within their moving camera footprints, and exchange original observations over peer TCP transport.</p>

  <section class="compact-header" id="compact-header">
    <div>
      <strong id="compact-scenario-name">Scenario loading…</strong>
      <span id="compact-run-state" class="pill">running</span>
      <span id="compact-elapsed-time" class="pill">0.0 s</span>
    </div>
    <div class="compact-actions">
      <button id="compact-pause-button" type="button">Pause</button>
      <button id="compact-replay-button" type="button">Replay current scenario</button>
    </div>
  </section>
  <div id="action-error-banner" class="status-banner" hidden></div>

  <section class="controls" id="shared-controls">
    <button id="pause-button" type="button">Pause</button>
    <button id="sharing-button" type="button">Enable sharing</button>
    <button id="cancel-response-button" type="button">Cancel response</button>
    <label class="pill">Playbook
      <select id="playbook-select">
        <option value="report_continue" selected>Report and continue</option>
        <option value="observe_orbit">Observe and orbit</option>
        <option value="orbit_follow" disabled>Orbit and follow — Stage B</option>
      </select>
    </label>
    <span id="task-state" class="pill">0 orbiting · 0 transitioning · 0 searching · 0 queued</span>
    <span id="playbook-config-state" class="pill">Playbook config pending status unavailable</span>
    <span id="control-state" class="pill">Sharing off · running</span>
    <span id="scenario-state" class="pill">Scenario loading…</span>
  </section>

  <nav class="tabs" role="tablist" aria-label="Dashboard sections">
    <button id="tab-button-team" class="tab-button" role="tab" type="button" data-tab="team" aria-selected="true" aria-controls="tab-team" tabindex="0">Team picture</button>
    <button id="tab-button-truth" class="tab-button" role="tab" type="button" data-tab="truth" aria-selected="false" aria-controls="tab-truth" tabindex="-1">Simulation truth</button>
    <button id="tab-button-network" class="tab-button" role="tab" type="button" data-tab="network" aria-selected="false" aria-controls="tab-network" tabindex="-1">Network &amp; messages</button>
  </nav>

  <section id="tab-team" class="tab-panel active" role="tabpanel" aria-labelledby="tab-button-team"></section>
  <section id="tab-truth" class="tab-panel" role="tabpanel" aria-labelledby="tab-button-truth"></section>
  <section id="tab-network" class="tab-panel" role="tabpanel" aria-labelledby="tab-button-network"></section>

  <section class="card" id="network-card">
    <div class="map-title"><span>Team network</span><span id="network-summary" class="pill">network diagnostics loading…</span></div>
    <details>
      <summary>Peer HTTP diagnostics</summary>
      <div class="control-note">Peer counters are instrumented in the drone processes' real HTTP send/receive paths. Bandwidth is JSON payload only and excludes dashboard polling.</div>
      <div id="network-details" class="details"></div>
    </details>
  </section>

  <section class="card" id="belief-display-card">
    <div class="map-title"><span>Belief display controls</span><span class="pill">display only</span></div>
    <label class="pill"><input id="measurement-toggle" type="checkbox" checked> Sensor measurements</label>
    <label class="pill"><input id="ellipse-toggle" type="checkbox" checked> Uncertainty ellipses</label>
    <label class="pill"><input id="priority-only-toggle" type="checkbox"> Priority only</label>
    <label class="pill"><input id="belief-label-toggle" type="checkbox"> Belief object labels</label>
    <span class="control-note">These filters affect only dashboard rendering; they do not change sensing, fusion, or allocation.</span>
  </section>

  <section class="card" id="assignment-card">
    <div class="map-title"><span>Priority response assignments</span><span class="pill">connected-peer claim state</span></div>
    <div id="assignment-details" class="details"></div>
  </section>

  <section class="map-grid" id="map-grid-section">
    <article class="map-card belief" style="--drone-color: #ef4444;">
      <div class="map-title"><span><span class="swatch" style="background: #ef4444"></span>Red belief</span><span class="pill">actual local map</span></div>
      <canvas id="belief-map-1" class="belief-canvas" width="780" height="420"></canvas>
      <div id="belief-details-1" class="details"></div>
      <details>
        <summary>Diagnostics</summary>
        <div id="diagnostics-1"></div>
      </details>
    </article>

    <article class="map-card belief" style="--drone-color: #3b82f6;">
      <div class="map-title"><span><span class="swatch" style="background: #3b82f6"></span>Blue belief</span><span class="pill">actual local map</span></div>
      <canvas id="belief-map-2" class="belief-canvas" width="780" height="420"></canvas>
      <div id="belief-details-2" class="details"></div>
      <details>
        <summary>Diagnostics</summary>
        <div id="diagnostics-2"></div>
      </details>
    </article>

    <article class="map-card belief" style="--drone-color: #22c55e;">
      <div class="map-title"><span><span class="swatch" style="background: #22c55e"></span>Green belief</span><span class="pill">actual local map</span></div>
      <canvas id="belief-map-3" class="belief-canvas" width="780" height="420"></canvas>
      <div id="belief-details-3" class="details"></div>
      <details>
        <summary>Diagnostics</summary>
        <div id="diagnostics-3"></div>
      </details>
    </article>

    <article class="map-card truth" id="truth-card">
      <div class="map-title">
        <span>Simulation truth</span>
        <span class="pill">persistent objects</span>
      </div>
      <div class="control-panel">
        <h3>Truth display</h3>
        <label class="pill"><input id="trail-toggle" type="checkbox" checked> Show flight trails</label>
        <label class="pill"><input id="truth-label-toggle" type="checkbox" checked> Truth-object labels</label>
        <label class="pill"><input id="background-truth-toggle" type="checkbox" checked> Background truth objects</label>
      </div>
      <div class="control-panel" id="scenario-control-panel">
        <h3>Scenario configuration</h3>
        <button id="reset-button" type="button">Restart scenario</button>
        <button id="new-scenario-button" type="button">New random selected scenario</button>
        <button id="handoff-demo-button" type="button">Run handoff demo</button>
        <label class="pill">Priority object count
          <select id="priority-count-select">
            <option value="0" selected>Random (1–5)</option>
            <option value="1">1</option>
            <option value="2">2</option>
            <option value="3">3</option>
            <option value="4">4</option>
            <option value="5">5</option>
          </select>
        </label>
        <label class="pill">Scenario
          <select id="scenario-select">
            <option value="seeded_stationary_patrol" selected>Seeded stationary patrol</option>
            <option value="moving_person_handoff">Jogging person handoff</option>
          </select>
        </label>
        <label class="pill">Initial observer
          <select id="initial-observer-select">
            <option value="1" selected>Red</option>
            <option value="2">Blue</option>
            <option value="3">Green</option>
          </select>
        </label>
        <label class="pill">Next observer
          <select id="next-observer-select">
            <option value="1">Red</option>
            <option value="2" selected>Blue</option>
            <option value="3">Green</option>
          </select>
        </label>
        <div id="handoff-demo-feedback" class="control-note">Handoff demo seed: pending dashboard initialization.</div>
        <span class="control-note">Scenario and observer selectors apply on the next Restart scenario or New random selected scenario. They shape coverage only and do not provide estimator identity hints.</span>
      </div>
      <canvas id="truth-map" class="truth-canvas" width="1100" height="520"></canvas>
      <div id="truth-details" class="details"></div>
      <div class="legend">Truth objects are rendered only on this map. Belief maps above are fetched from each drone process and are not populated from truth.</div>
    </article>
  </section>

  <section class="card shared-inspector" id="shared-inspector-card">
    <div class="map-title"><span>Object inspector</span><span class="pill">live object · pinned sensor report</span></div>
    <div id="shared-inspector" class="details">
      <div class="empty-state">Select an object on a drone belief map to inspect its contributing sensor reports.</div>
    </div>
  </section>

  <h2>Flight telemetry</h2>
  <main id="drone-grid" class="grid"></main>

  <script>
    const drones = __DRONES_JSON__;
    const HANDOFF_DEMO_SEED = 2026110601;
    const HANDOFF_DEMO_SECONDS = 95;
    const grid = document.getElementById("drone-grid");
    const state = {
      statuses: new Map(),
      beliefs: new Map(),
      networks: new Map(),
      truth: { objects: [], simulation_time: 0 },
      sharingEnabled: false,
      paused: false,
      scenarioName: "unknown",
      scenarioSeed: null,
      appBuild: "unknown",
      selectedPlaybook: "report_continue",
      requestedPriorityCount: 0,
      priorityObjectCount: 0,
      priorityDiscoveredCount: 0,
      scenarioId: "seeded_stationary_patrol",
      scenarioRunId: null,
      initialObserverId: 1,
      nextObserverId: 2,
      responseTask: null,
      responseTasks: [],
      responseQueueDepth: 0,
      taskSnapshotsByDrone: new Map(),
      droneAvailability: new Map(),
      currentRunKey: null,
      orbitRadiusM: 0,
      playbookConfigVersion: 0,
      pendingPlaybookConfig: null,
      showFlightTrails: true,
      showTruthObjectLabels: true,
      showBeliefObjectLabels: false,
      showBackgroundTruthObjects: true,
      showMeasurements: true,
      showEllipses: true,
      priorityOnly: false,
      beliefObjectTablesOpen: new Map(),
      mapHitTargets: new Map(),
      hoveredMapObject: null,
      selectedMapObject: null,
      selectedObservation: null,
      selectedObservationKey: null,
      selectedObservationSnapshot: null,
      observationSelectionMode: "latest",
      observationTrace: new Map(),
      rawJsonOpen: false,
      rawJsonScrollTop: 0,
      rawJsonSelectionStart: 0,
      rawJsonSelectionEnd: 0,
      rawJsonHadFocus: false,
      selectionMessage: "",
      activeTab: "team",
      playbookSelectDirty: false,
      scenarioConfigDirty: false,
      handoffDemoRunStatus: "idle",
      handoffConfigStatus: "not_configured",
      handoffDemoError: "",
      handoffDemoAckMessage: "",
      handoffDemoFinishPauseRequested: false,
      handoffObjectIdsByDrone: new Map(),
      handoffEvidence: new Map(),
    };

    function createCard(drone) {
      const card = document.createElement("section");
      card.className = "card offline";
      card.id = `drone-${drone.id}`;
      card.innerHTML = `
        <div class="name">
          <span><span class="swatch" style="background: ${drone.color}"></span>${drone.name}</span>
          <span class="status">offline</span>
        </div>
        <details>
          <summary>Detailed flight telemetry</summary>
          <dl>
            <dt>Schema</dt><dd data-field="schema_version">—</dd>
            <dt>Altitude</dt><dd data-field="altitude_m">—</dd>
            <dt>Speed</dt><dd data-field="speed_mps">—</dd>
            <dt>Heading</dt><dd data-field="heading_deg">—</dd>
            <dt>Latitude</dt><dd data-field="latitude">—</dd>
            <dt>Longitude</dt><dd data-field="longitude">—</dd>
            <dt>Battery</dt><dd data-field="battery_percent">—</dd>
            <dt>Sim time</dt><dd data-field="updated_at">—</dd>
          </dl>
        </details>
      `;
      grid.appendChild(card);
    }

    function setField(card, field, value) {
      card.querySelector(`[data-field="${field}"]`).textContent = value;
    }

    async function fetchJson(url, options = {}) {
      const response = await fetch(url, { cache: "no-store", ...options });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      return response.json();
    }

    async function updateDrone(drone) {
      const card = document.getElementById(`drone-${drone.id}`);

      try {
        const [status, belief, network] = await Promise.all([
          fetchJson(`${drone.url}/status`),
          fetchJson(`${drone.url}/objects`),
          fetchJson(`${drone.url}/network`),
        ]);

        state.statuses.set(drone.id, status);
        state.beliefs.set(drone.id, belief);
        state.networks.set(drone.id, network);
        state.sharingEnabled = Boolean(status.sharing_enabled);
        state.paused = Boolean(status.paused);
        state.selectedPlaybook = status.selected_playbook || state.selectedPlaybook;
        state.orbitRadiusM = status.orbit_radius_m || state.orbitRadiusM;
        state.droneAvailability.set(drone.id, {
          online: true,
          receivedAt: Date.now() / 1000,
        });
        state.taskSnapshotsByDrone.set(drone.id, {
          droneId: drone.id,
          droneName: drone.name,
          online: true,
          receivedAt: Date.now() / 1000,
          simulationTime: Number(status.timestamp || 0),
          currentMode: status.current_mode || "searching",
          assignedObjective: status.assigned_objective || null,
          responseTask: status.response_task || null,
          responseTasks: status.response_tasks || [],
          responseQueueDepth: Number(status.response_queue_depth || 0),
          playbookConfigVersion: Number(status.playbook_config_version || 0),
        });

        card.classList.remove("offline");
        card.querySelector(".status").textContent = status.state;
        card.querySelector(".status").className = "status online";

        setField(card, "schema_version", status.schema_version);
        setField(card, "altitude_m", `${status.altitude_m.toFixed(1)} m`);
        setField(card, "speed_mps", `${status.speed_mps.toFixed(1)} m/s`);
        setField(card, "heading_deg", `${status.heading_deg.toFixed(0)}°`);
        setField(card, "latitude", status.latitude.toFixed(6));
        setField(card, "longitude", status.longitude.toFixed(6));
        setField(card, "battery_percent", `${status.battery_percent.toFixed(1)}%`);
        setField(card, "updated_at", `${status.timestamp.toFixed(1)} s`);
      } catch (error) {
        const previousSnapshot = state.taskSnapshotsByDrone.get(drone.id);

        if (previousSnapshot) {
          state.taskSnapshotsByDrone.set(drone.id, {
            ...previousSnapshot,
            online: false,
            unavailableAt: Date.now() / 1000,
          });
        }

        state.statuses.delete(drone.id);
        state.networks.delete(drone.id);
        state.droneAvailability.set(drone.id, {
          online: false,
          receivedAt: Date.now() / 1000,
        });

        card.classList.add("offline");
        card.querySelector(".status").textContent = "offline";
        card.querySelector(".status").className = "status";
      }
    }

    async function updateTruth() {
      state.truth = await fetchJson(`${drones[0].url}/truth`);
      state.scenarioName = state.truth.scenario_name || "unknown";
      state.scenarioSeed = state.truth.scenario_seed;
      state.appBuild = state.truth.app_build || "unknown";
      state.selectedPlaybook = state.truth.selected_playbook || "report_continue";
      state.scenarioId = state.truth.scenario_id || state.scenarioId;
      state.scenarioRunId = state.truth.scenario_run_id || state.scenarioRunId;
      state.initialObserverId = Number(state.truth.initial_observer_id || state.initialObserverId);
      state.nextObserverId = Number(state.truth.next_observer_id || state.nextObserverId);
      state.playbookConfigVersion = Number(state.truth.playbook_config_version || state.playbookConfigVersion || 0);
      state.requestedPriorityCount = state.truth.requested_priority_count || 0;
      state.priorityObjectCount = state.truth.priority_object_count || 0;
      state.priorityDiscoveredCount = state.truth.priority_discovered_count || 0;
      state.orbitRadiusM = state.truth.orbit_radius_m || 0;

      const runKey = [
        state.scenarioId,
        state.scenarioRunId,
        state.scenarioSeed,
        state.selectedPlaybook,
        state.requestedPriorityCount,
        state.initialObserverId,
        state.nextObserverId,
      ].join(":");

      if (state.currentRunKey && state.currentRunKey !== runKey) {
        state.taskSnapshotsByDrone.clear();
        clearInspectionSelection("Scenario changed. Previous selected object and pinned sensor report were cleared.");
      }

      state.currentRunKey = runKey;
    }

    function allMapPoints() {
      const points = [];

      for (const status of state.statuses.values()) {
        points.push(status);
        status.camera_footprint.forEach((point) => points.push(point));
        (status.flight_trail || []).forEach((point) => points.push(point));
      }

      state.truth.objects.forEach((object) => points.push(object));

      for (const belief of state.beliefs.values()) {
        belief.objects.forEach((object) => points.push(object));
      }

      return points;
    }

    function boundsFor(points) {
      const latitudes = points.map((point) => point.latitude);
      const longitudes = points.map((point) => point.longitude);
      const minLatitude = Math.min(...latitudes);
      const maxLatitude = Math.max(...latitudes);
      const minLongitude = Math.min(...longitudes);
      const maxLongitude = Math.max(...longitudes);
      const latitudePadding = Math.max((maxLatitude - minLatitude) * 0.18, 0.0008);
      const longitudePadding = Math.max((maxLongitude - minLongitude) * 0.18, 0.0008);

      return {
        minLatitude: minLatitude - latitudePadding,
        maxLatitude: maxLatitude + latitudePadding,
        minLongitude: minLongitude - longitudePadding,
        maxLongitude: maxLongitude + longitudePadding,
      };
    }

    function project(point, bounds, canvas) {
      const x = ((point.longitude - bounds.minLongitude) / (bounds.maxLongitude - bounds.minLongitude)) * canvas.width;
      const y = canvas.height - (((point.latitude - bounds.minLatitude) / (bounds.maxLatitude - bounds.minLatitude)) * canvas.height);
      return { x, y };
    }

    function priorityClassLabel(object) {
      return `${object.priority ? "PRIORITY " : ""}${titleCaseLabel(object.label || object.object_label || "Objective")}`;
    }

    function registerMapTarget(canvas, target) {
      if (!state.mapHitTargets.has(canvas.id)) {
        state.mapHitTargets.set(canvas.id, []);
      }

      state.mapHitTargets.get(canvas.id).push(target);
    }

    function uncertaintyDimensions(covariance) {
      if (!covariance) {
        return "—";
      }

      const params = ellipseParams(covariance);
      return `${(params.majorM * 2).toFixed(1)} m × ${(params.minorM * 2).toFixed(1)} m`;
    }

    function observationKey(observation) {
      return `${observation.scenario_run_id || "unknown-run"}:${observation.observation_id}`;
    }

    function snapshotObservation(observation, target) {
      return {
        targetKey: target.key,
        droneId: target.droneId,
        droneName: droneNameForId(target.droneId),
        observationKey: observationKey(observation),
        observation: JSON.parse(JSON.stringify(observation)),
        rawJson: JSON.stringify(observation.raw_observation || observation, null, 2),
        savedAt: new Date().toISOString(),
      };
    }

    function currentObservationForTarget(target) {
      if (!target || !target.object || !target.object.measurements || target.object.measurements.length === 0) {
        return null;
      }

      if (state.observationSelectionMode === "pinned" && state.selectedObservationKey) {
        return target.object.measurements.find((item) => observationKey(item) === state.selectedObservationKey) || null;
      }

      return target.object.measurements
        .slice()
        .sort((left, right) => Number(right.timestamp || 0) - Number(left.timestamp || 0))[0];
    }

    function snapshotJsonUiState() {
      const jsonDetails = document.getElementById("observation-json-details");
      const jsonArea = document.getElementById("observation-json-area");

      if (jsonDetails) {
        state.rawJsonOpen = Boolean(jsonDetails.open);
      }

      if (jsonArea) {
        state.rawJsonScrollTop = jsonArea.scrollTop;
        state.rawJsonSelectionStart = Number(jsonArea.selectionStart || 0);
        state.rawJsonSelectionEnd = Number(jsonArea.selectionEnd || 0);
        state.rawJsonHadFocus = document.activeElement === jsonArea;
      }
    }

    function restoreJsonUiState() {
      const jsonArea = document.getElementById("observation-json-area");

      if (!jsonArea) {
        return;
      }

      jsonArea.scrollTop = state.rawJsonScrollTop;

      if (state.rawJsonHadFocus) {
        jsonArea.focus();

        if (typeof jsonArea.setSelectionRange === "function") {
          jsonArea.setSelectionRange(state.rawJsonSelectionStart, state.rawJsonSelectionEnd);
        }
      }
    }

    function selectedObservationSnapshotFor(target) {
      const liveObservation = currentObservationForTarget(target);

      if (state.observationSelectionMode === "latest" && liveObservation) {
        state.selectedObservationKey = observationKey(liveObservation);
        state.selectedObservationSnapshot = snapshotObservation(liveObservation, target);
        return {
          snapshot: state.selectedObservationSnapshot,
          liveAvailable: true,
          pinned: false,
        };
      }

      if (state.selectedObservationSnapshot) {
        return {
          snapshot: state.selectedObservationSnapshot,
          liveAvailable: Boolean(liveObservation),
          pinned: true,
        };
      }

      if (liveObservation) {
        state.selectedObservationKey = observationKey(liveObservation);
        state.selectedObservationSnapshot = snapshotObservation(liveObservation, target);
        return {
          snapshot: state.selectedObservationSnapshot,
          liveAvailable: true,
          pinned: false,
        };
      }

      return null;
    }

    function observationInspectorHtml(target) {
      if (!target || !target.object || !target.object.measurements || target.object.measurements.length === 0) {
        return '<div class="empty-state">No contributing sensor reports are available for this object.</div>';
      }

      const selected = selectedObservationSnapshotFor(target);
      if (!selected) {
        return '<div class="empty-state">No sensor report selected.</div>';
      }

      const observation = selected.snapshot.observation;
      const traceRows = state.observationTrace.get(observation.observation_id) || [];
      const snapshotLabel = selected.pinned
        ? selected.liveAvailable
          ? "Pinned simulated sensor report — exact JSON held stable while live object updates."
          : "Saved inspection snapshot — this exact observation is no longer present in the current node object record."
        : "Latest observation — select a report below to pin its exact JSON.";

      return `
        <div class="control-panel">
          <span class="sensor-label">SIMULATED SENSOR INPUT</span>
          <div class="inspection-note">${escapeHtml(snapshotLabel)}</div>
          <h3>Contributing observations</h3>
          <div class="json-toolbar">
            <button type="button" data-observation-mode="latest" data-target-key="${escapeHtml(target.key)}">Latest observation</button>
            ${target.object.measurements.map((item) => `
              <button type="button" data-observation-id="${escapeHtml(item.observation_id)}" data-observation-run-id="${escapeHtml(item.scenario_run_id || "unknown-run")}" data-target-key="${escapeHtml(target.key)}">
                ${escapeHtml(item.source_drone_name)} · ${escapeHtml(item.sensor_id)} · ${escapeHtml(item.source_track_id)}
              </button>
            `).join(" ")}
          </div>
          <dl class="diagnostics-grid">
            <dt>Original drone</dt><dd>${escapeHtml(observation.source_drone_name)} (#${escapeHtml(observation.source_drone_id)})</dd>
            <dt>Sensor</dt><dd>${escapeHtml(observation.sensor_id)}</dd>
            <dt>Source-local track ID</dt><dd>${escapeHtml(observation.source_track_id)}</dd>
            <dt>Observation ID</dt><dd>${escapeHtml(observation.observation_id)}</dd>
            <dt>Run ID</dt><dd>${escapeHtml(observation.scenario_run_id || "—")}</dd>
            <dt>Capture timestamp</dt><dd>${Number(observation.timestamp || 0).toFixed(1)} s mission time</dd>
            <dt>Object class</dt><dd>${escapeHtml(target.object.label || "—")}</dd>
            <dt>Classification score</dt><dd>${Number(observation.classification_score || 0).toFixed(2)}</dd>
            <dt>Measured position</dt><dd>${Number(observation.east_m || 0).toFixed(1)} m E, ${Number(observation.north_m || 0).toFixed(1)} m N</dd>
            <dt>Covariance</dt><dd>${escapeHtml(JSON.stringify(observation.covariance || []))}</dd>
            <dt>95% horizontal uncertainty</dt><dd>${escapeHtml(uncertaintyDimensions(observation.covariance))}</dd>
          </dl>
          <button type="button" data-trace-observation-id="${escapeHtml(observation.observation_id)}">Trace this observation</button>
          ${traceRows.length ? `
            <table class="trace-table">
              <thead><tr><th>Node</th><th>Trace status</th><th>Associated local map object</th><th>Retention</th></tr></thead>
              <tbody>
                ${traceRows.map((row) => `
                  <tr>
                    <td>${escapeHtml(row.drone_name)} (#${escapeHtml(row.drone_id)})</td>
                    <td>${row.record
                      ? `${row.record.originated_locally ? "Originated locally · " : "Received · "}${escapeHtml(row.record.outcome)}`
                      : "Not seen / trace record no longer retained"}</td>
                    <td>${row.record ? escapeHtml(row.record.associated_local_object_id || "—") : "—"}</td>
                    <td>${escapeHtml(row.retained_trace_count)} retained · ${escapeHtml(row.trace_evictions)} evicted</td>
                  </tr>
                `).join("")}
              </tbody>
            </table>
          ` : ""}
          <details id="observation-json-details"${state.rawJsonOpen ? " open" : ""}>
            <summary>Raw JSON</summary>
            <div class="json-toolbar">
              <button type="button" data-copy-observation-json="true">Copy JSON</button>
              <span class="control-note">Bounded inspector view; live fused-object age can change while this pinned report remains unchanged.</span>
            </div>
            <textarea id="observation-json-area" class="json-area" readonly>${escapeHtml(selected.snapshot.rawJson)}</textarea>
          </details>
        </div>
      `;
    }

    function droneNameForId(droneId) {
      const drone = drones.find((item) => item.id === droneId);
      return drone ? drone.name : "Unknown drone";
    }

    function sourceIdentityList(measurements) {
      const identities = new Set();

      for (const measurement of measurements || []) {
        identities.add(`${measurement.source_drone_name || "Unknown drone"} / ${measurement.sensor_id || "unknown sensor"}`);
      }

      return [...identities].sort();
    }

    function objectDetailHtml(target) {
      if (!target) {
        return "";
      }

      const object = target.object;
      const historicalSources = object.historical_source_names && object.historical_source_names.length
        ? object.historical_source_names
        : object.source_names || [];
      const observedBy = historicalSources.length
        ? historicalSources.join(", ")
        : sourceIdentityList(object.measurements).join(", ") || "Provenance unavailable";
      const recentSources = object.source_names && object.source_names.length
        ? object.source_names.join(", ")
        : object.stale
          ? "None — stale"
          : "None currently usable";
      const age = target.ageSeconds === null || target.ageSeconds === undefined ? "—" : `${target.ageSeconds.toFixed(1)} s`;
      const priority = object.priority ? "yes" : "no";
      const movingDetails = object.tracking_model === "constant_velocity"
        ? `
            <dt>Track state</dt><dd>${escapeHtml(object.track_status || "tracked")}</dd>
            <dt>Prediction age</dt><dd>${Number(object.prediction_age_seconds || 0).toFixed(1)} s</dd>
            <dt>Last measurement age</dt><dd>${Number(object.last_measurement_age_seconds || 0).toFixed(1)} s</dd>
            <dt>Velocity</dt><dd>${Number(object.velocity_east_mps || 0).toFixed(2)} m/s E, ${Number(object.velocity_north_mps || 0).toFixed(2)} m/s N</dd>
            <dt>Local object ID</dt><dd>${escapeHtml(object.local_object_id || object.object_id || "—")}</dd>
            <dt>Source-local tracks</dt><dd>${escapeHtml((object.source_track_ids || []).join(", ") || "—")}</dd>
          `
        : "";

      return `
        <div class="control-panel">
          <h3>Selected object</h3>
          <dl class="diagnostics-grid">
            <dt>Class</dt><dd>${escapeHtml(priorityClassLabel(object))}</dd>
            <dt>Priority</dt><dd>${priority}</dd>
            <dt>Observed by</dt><dd>${escapeHtml(observedBy)}</dd>
            <dt>Recent sources</dt><dd>${escapeHtml(recentSources)}</dd>
            <dt>Age</dt><dd>${escapeHtml(age)}</dd>
            ${movingDetails}
          </dl>
        </div>
        ${observationInspectorHtml(target)}
      `;
    }

    function latestTargetFor(target) {
      if (!target || !target.key) {
        return target;
      }

      for (const targets of state.mapHitTargets.values()) {
        const current = targets.find((candidate) => candidate.key === target.key);

        if (current) {
          return current;
        }
      }

      return target;
    }

    function activeMapDetailFor(kind, droneId = null) {
      const selected = latestTargetFor(state.selectedMapObject);
      const hovered = latestTargetFor(state.hoveredMapObject);
      const matches = (target) => target
        && target.kind === kind
        && (droneId === null || target.droneId === droneId);

      return matches(selected) ? selected : matches(hovered) ? hovered : null;
    }

    function canvasPointFromEvent(canvas, event) {
      const rect = canvas.getBoundingClientRect();
      return {
        x: ((event.clientX - rect.left) / rect.width) * canvas.width,
        y: ((event.clientY - rect.top) / rect.height) * canvas.height,
      };
    }

    function nearestMapTarget(canvas, event) {
      const point = canvasPointFromEvent(canvas, event);
      const targets = state.mapHitTargets.get(canvas.id) || [];
      let nearest = null;

      for (const target of targets) {
        const distance = Math.hypot(point.x - target.x, point.y - target.y);

        if (distance <= target.radius && (!nearest || distance < nearest.distance)) {
          nearest = { target, distance };
        }
      }

      return nearest ? nearest.target : null;
    }

    function refreshMapDetailPanels() {
      renderTruthDetails();

      for (const drone of drones) {
        renderBeliefDetails(drone);
      }

      renderSharedInspector();
    }

    function resizeCanvasBackingStore(canvas) {
      const rect = canvas.getBoundingClientRect ? canvas.getBoundingClientRect() : null;

      if (!rect || rect.width <= 0 || rect.height <= 0) {
        return false;
      }

      const pixelRatio = window.devicePixelRatio || 1;
      const backingWidth = Math.max(1, Math.round(rect.width * pixelRatio));
      const backingHeight = Math.max(1, Math.round(rect.height * pixelRatio));

      if (canvas.width !== backingWidth || canvas.height !== backingHeight) {
        canvas.width = backingWidth;
        canvas.height = backingHeight;
      }

      return true;
    }

    function drawBaseMap(canvas, bounds) {
      if (!resizeCanvasBackingStore(canvas)) {
        return false;
      }

      state.mapHitTargets.set(canvas.id, []);
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.strokeStyle = "#1e293b";
      ctx.lineWidth = 1;

      for (let x = 0; x <= canvas.width; x += 50) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, canvas.height);
        ctx.stroke();
      }

      for (let y = 0; y <= canvas.height; y += 50) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(canvas.width, y);
        ctx.stroke();
      }

      return true;
    }

    function drawFlightTrails(canvas, bounds) {
      if (!state.showFlightTrails) {
        return;
      }

      const ctx = canvas.getContext("2d");

      for (const drone of drones) {
        const status = state.statuses.get(drone.id);

        if (!status || !status.flight_trail) {
          continue;
        }

        for (const point of status.flight_trail) {
          const age = Math.max(0, status.timestamp - point.timestamp);

          if (age >= 30) {
            continue;
          }

          const position = project(point, bounds, canvas);
          const opacity = Math.max(0.08, 1 - (age / 30));
          ctx.fillStyle = `${drone.color}${Math.round(opacity * 180).toString(16).padStart(2, "0")}`;
          ctx.beginPath();
          ctx.arc(position.x, position.y, 3.2, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    }

    function drawDroneMarker(ctx, canvas, bounds, drone, status, alpha) {
      const position = project(status, bounds, canvas);
      const heading = (status.heading_deg - 90) * (Math.PI / 180);

      ctx.save();
      ctx.globalAlpha = alpha;
      ctx.translate(position.x, position.y);
      ctx.rotate(heading);
      ctx.fillStyle = drone.color;
      ctx.beginPath();
      ctx.moveTo(13, 0);
      ctx.lineTo(-9, -7);
      ctx.lineTo(-6, 0);
      ctx.lineTo(-9, 7);
      ctx.closePath();
      ctx.fill();
      ctx.restore();

      ctx.save();
      ctx.globalAlpha = alpha;
      ctx.fillStyle = "#e2e8f0";
      ctx.font = "12px sans-serif";
      ctx.fillText(drone.name, position.x + 10, position.y - 10);
      ctx.restore();
    }

    function drawFootprintsAndDrones(canvas, bounds, emphasizedDroneId = null) {
      const ctx = canvas.getContext("2d");
      const activeDrones = drones
        .map((drone) => ({ drone, status: state.statuses.get(drone.id) }))
        .filter((item) => item.status);
      const orderedDrones = emphasizedDroneId === null
        ? activeDrones
        : [
          ...activeDrones.filter((item) => item.drone.id !== emphasizedDroneId),
          ...activeDrones.filter((item) => item.drone.id === emphasizedDroneId),
        ];

      for (const { drone, status } of activeDrones) {
        const isEmphasized = emphasizedDroneId === null || drone.id === emphasizedDroneId;
        const footprint = status.camera_footprint.map((point) => project(point, bounds, canvas));

        ctx.save();
        ctx.globalAlpha = emphasizedDroneId === null || isEmphasized ? 1 : 0.25;
        ctx.beginPath();
        ctx.moveTo(footprint[0].x, footprint[0].y);
        ctx.lineTo(footprint[1].x, footprint[1].y);
        ctx.lineTo(footprint[2].x, footprint[2].y);
        ctx.closePath();
        ctx.fillStyle = isEmphasized ? `${drone.color}33` : `${drone.color}26`;
        ctx.strokeStyle = isEmphasized ? drone.color : `${drone.color}bb`;
        ctx.lineWidth = isEmphasized ? 3 : 2;
        ctx.fill();
        ctx.stroke();
        ctx.restore();
      }

      for (const { drone, status } of orderedDrones) {
        const alpha = emphasizedDroneId === null || drone.id === emphasizedDroneId ? 1 : 0.4;
        drawDroneMarker(ctx, canvas, bounds, drone, status, alpha);
      }
    }

    function drawTruth(canvas, bounds) {
      const ctx = canvas.getContext("2d");
      const labelRects = [];

      for (const object of state.truth.objects) {
        if (object.moving && object.trajectory) {
          ctx.strokeStyle = "#f472b6";
          ctx.lineWidth = 2;
          ctx.beginPath();
          object.trajectory.forEach((point, index) => {
            const latitude = 37.7752 + (point.north_m / 111111);
            const longitude = -122.4230 + (point.east_m / metersPerDegLon(37.7752));
            const projected = project({ latitude, longitude }, bounds, canvas);
            if (index === 0) {
              ctx.moveTo(projected.x, projected.y);
            } else {
              ctx.lineTo(projected.x, projected.y);
            }
          });
          ctx.stroke();
        }
        if ((!state.showBackgroundTruthObjects && !object.priority) || (state.priorityOnly && !object.priority)) {
          continue;
        }
        const position = project(object, bounds, canvas);
        if (object.moving) {
          ctx.fillStyle = "#f472b6";
          ctx.strokeStyle = "#fdf2f8";
          ctx.lineWidth = 3;
          ctx.beginPath();
          ctx.moveTo(position.x, position.y - 12);
          ctx.lineTo(position.x + 12, position.y);
          ctx.lineTo(position.x, position.y + 12);
          ctx.lineTo(position.x - 12, position.y);
          ctx.closePath();
          ctx.fill();
          ctx.stroke();
        } else if (object.priority) {
          ctx.fillStyle = "#facc15";
          ctx.strokeStyle = "#f97316";
          ctx.lineWidth = 3;
          ctx.beginPath();
          ctx.moveTo(position.x, position.y - 11);
          ctx.lineTo(position.x + 10, position.y + 8);
          ctx.lineTo(position.x - 10, position.y + 8);
          ctx.closePath();
          ctx.fill();
          ctx.stroke();
        } else {
          ctx.fillStyle = "#f8fafc";
          ctx.strokeStyle = "#0f172a";
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.arc(position.x, position.y, 7, 0, Math.PI * 2);
          ctx.fill();
          ctx.stroke();
        }

        registerMapTarget(canvas, {
          key: `truth:${object.id || object.object_id || position.x}:${position.y}`,
          kind: "truth",
          object,
          x: position.x,
          y: position.y,
          radius: object.priority ? 16 : 12,
          ageSeconds: null,
        });

        if (state.showTruthObjectLabels) {
          const label = priorityClassLabel(object);
          ctx.font = "11px sans-serif";
          const width = ctx.measureText(label).width;
          const rect = { x: position.x + 10, y: position.y - 9, width, height: 14 };
          const overlaps = labelRects.some((item) => (
            rect.x < item.x + item.width
            && rect.x + rect.width > item.x
            && rect.y < item.y + item.height
            && rect.y + rect.height > item.y
          ));

          if (!overlaps || object.priority) {
            ctx.fillStyle = object.priority ? "#fde68a" : "#94a3b8";
            ctx.fillText(label, rect.x, position.y + 4);
            labelRects.push(rect);
          }
        }
      }
    }

    function metersPerDegLon(latitude) {
      return 111111 * Math.cos(latitude * Math.PI / 180);
    }

    function localMetricDeltaToCanvas(center, eastM, northM, bounds, canvas) {
      const latitude = center.latitude + (northM / 111111);
      const longitude = center.longitude + (eastM / metersPerDegLon(center.latitude));
      const projectedCenter = project(center, bounds, canvas);
      const projectedPoint = project({ latitude, longitude }, bounds, canvas);
      return {
        x: projectedPoint.x - projectedCenter.x,
        y: projectedPoint.y - projectedCenter.y,
      };
    }

    function ellipseParams(covariance) {
      const a = covariance[0][0];
      const b = covariance[0][1];
      const c = covariance[1][1];
      const trace = a + c;
      const root = Math.sqrt(Math.max(0, ((a - c) * (a - c)) + (4 * b * b)));
      const lambda1 = Math.max(0, (trace + root) / 2);
      const lambda2 = Math.max(0, (trace - root) / 2);
      const angle = Math.atan2(lambda1 - a, b || 1e-9);
      return {
        majorM: Math.sqrt(5.991 * lambda1),
        minorM: Math.sqrt(5.991 * lambda2),
        angle,
      };
    }

    function drawCovarianceEllipse(ctx, canvas, bounds, center, covariance, color) {
      if (!covariance) {
        return;
      }

      const params = ellipseParams(covariance);
      const majorDelta = localMetricDeltaToCanvas(
        center,
        Math.cos(params.angle) * params.majorM,
        Math.sin(params.angle) * params.majorM,
        bounds,
        canvas,
      );
      const minorDelta = localMetricDeltaToCanvas(
        center,
        -Math.sin(params.angle) * params.minorM,
        Math.cos(params.angle) * params.minorM,
        bounds,
        canvas,
      );
      const position = project(center, bounds, canvas);
      const majorPx = Math.hypot(majorDelta.x, majorDelta.y);
      const minorPx = Math.hypot(minorDelta.x, minorDelta.y);
      const anglePx = Math.atan2(majorDelta.y, majorDelta.x);

      ctx.save();
      ctx.translate(position.x, position.y);
      ctx.rotate(anglePx);
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.ellipse(0, 0, majorPx, minorPx, 0, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();
      ctx.setLineDash([]);
    }

    function drawBeliefObjects(canvas, bounds, droneId) {
      const ctx = canvas.getContext("2d");
      const belief = state.beliefs.get(droneId);

      if (!belief) {
        return;
      }

      for (const object of belief.objects) {
        if (state.priorityOnly && !object.priority) {
          continue;
        }
        const position = project(object, bounds, canvas);
        ctx.lineWidth = 3;

        if (state.showMeasurements) {
          for (const measurement of object.measurements || []) {
            const measurementPosition = project(measurement, bounds, canvas);
            ctx.fillStyle = measurement.source_color || "#fbbf24";
            ctx.beginPath();
            ctx.arc(measurementPosition.x, measurementPosition.y, 3.5, 0, Math.PI * 2);
            ctx.fill();

            if (state.showEllipses) {
              drawCovarianceEllipse(ctx, canvas, bounds, measurement, measurement.covariance, `${measurement.source_color || "#fbbf24"}aa`);
            }
          }
        }

        if (state.showEllipses) {
          drawCovarianceEllipse(ctx, canvas, bounds, object, object.covariance, "#f8fafc");
        }

        if (object.tracking_model === "constant_velocity") {
          const statusColors = { tracked: "#f472b6", coasting: "#fbbf24", lost: "#64748b" };
          const color = statusColors[object.track_status] || "#f472b6";
          ctx.strokeStyle = color;
          ctx.fillStyle = `${color}66`;
          ctx.lineWidth = 4;

          for (const point of object.track_trail || []) {
            const latitude = 37.7752 + (point.north_m / 111111);
            const longitude = -122.4230 + (point.east_m / metersPerDegLon(37.7752));
            const trailPosition = project({ latitude, longitude }, bounds, canvas);
            ctx.beginPath();
            ctx.arc(trailPosition.x, trailPosition.y, 2.5, 0, Math.PI * 2);
            ctx.fill();
          }

          ctx.beginPath();
          ctx.moveTo(position.x, position.y - 12);
          ctx.lineTo(position.x + 12, position.y);
          ctx.lineTo(position.x, position.y + 12);
          ctx.lineTo(position.x - 12, position.y);
          ctx.closePath();
          ctx.fill();
          ctx.stroke();

          const velocityScale = 10;
          const velocityEnd = localMetricDeltaToCanvas(
            object,
            Number(object.velocity_east_mps || 0) * velocityScale,
            Number(object.velocity_north_mps || 0) * velocityScale,
            bounds,
            canvas,
          );
          ctx.beginPath();
          ctx.moveTo(position.x, position.y);
          ctx.lineTo(position.x + velocityEnd.x, position.y + velocityEnd.y);
          ctx.stroke();
        }

        if (object.priority) {
          ctx.strokeStyle = "#facc15";
          ctx.lineWidth = 4;
          ctx.beginPath();
          ctx.rect(position.x - 11, position.y - 11, 22, 22);
          ctx.stroke();
        }

        if (object.own_sensor) {
          ctx.fillStyle = "#f8fafc";
          ctx.strokeStyle = "#f59e0b";
          ctx.beginPath();
          ctx.arc(position.x, position.y, 8, 0, Math.PI * 2);
          ctx.fill();
          ctx.stroke();
        }

        if (object.teammate_sourced && !object.own_sensor) {
          ctx.fillStyle = "#38bdf8";
          ctx.strokeStyle = "#e0f2fe";
          ctx.beginPath();
          ctx.moveTo(position.x, position.y - 9);
          ctx.lineTo(position.x - 9, position.y + 8);
          ctx.lineTo(position.x + 9, position.y + 8);
          ctx.closePath();
          ctx.fill();
          ctx.stroke();
        }

        if (object.teammate_sourced && object.own_sensor) {
          ctx.strokeStyle = "#38bdf8";
          ctx.beginPath();
          ctx.arc(position.x, position.y, 13, 0, Math.PI * 2);
          ctx.stroke();
        }

        const age = Math.max(0, Number(belief.simulation_time || 0) - Number(object.last_seen || 0));
        registerMapTarget(canvas, {
          key: `belief:${droneId}:${object.local_object_id || object.object_id}`,
          kind: "belief",
          droneId,
          object,
          x: position.x,
          y: position.y,
          radius: object.priority ? 16 : 12,
          ageSeconds: age,
        });

        if (state.showBeliefObjectLabels) {
          ctx.fillStyle = "#cbd5e1";
          ctx.font = "11px sans-serif";
          ctx.fillText(priorityClassLabel(object), position.x + 12, position.y + 4);
        }
      }
    }

    function renderTruthDetails() {
      const details = document.getElementById("truth-details");
      const activeDetail = activeMapDetailFor("truth");
      details.innerHTML = `
        <strong>${state.truth.objects.length} persistent ground-truth objects · ${state.priorityObjectCount} high-priority · ${state.scenarioName} · seed ${state.scenarioSeed} · build ${state.appBuild}</strong>
        <div>Simulation instrumentation: ${state.priorityDiscoveredCount} priority objectives discovered via accepted priority evidence.</div>
        ${objectDetailHtml(activeDetail)}
        <ul>
          ${state.truth.objects
            .filter((object) => !state.priorityOnly || object.priority)
            .map((object) => `<li>${object.priority ? "⭐ " : ""}${escapeHtml(priorityClassLabel(object))}</li>`)
            .join("")}
        </ul>
      `;
    }

    function renderSharedInspector() {
      snapshotJsonUiState();

      const inspector = document.getElementById("shared-inspector");
      const activeDetail = activeMapDetailFor("belief");

      if (!activeDetail) {
        inspector.innerHTML = state.selectionMessage
          ? `<div class="inspection-note">${escapeHtml(state.selectionMessage)}</div><div class="empty-state">Select an object on a drone belief map to inspect its contributing sensor reports.</div>`
          : '<div class="empty-state">Select an object on a drone belief map to inspect its contributing sensor reports.</div>';
        return;
      }

      const timeline = activeDetail.object.handoff_timeline || [];
      inspector.innerHTML = `
        <div class="control-panel">
          <h3>${escapeHtml(droneNameForId(activeDetail.droneId))} belief object</h3>
          ${objectDetailHtml(activeDetail)}
          ${timeline.length ? `
            <h3>Moving handoff timeline</h3>
            <ul>${timeline.map((event) => `<li><strong>${escapeHtml(event.state)}</strong>: ${escapeHtml(event.detail)}</li>`).join("")}</ul>
          ` : ""}
        </div>
      `;
      restoreJsonUiState();
    }

    function renderBeliefDetails(drone) {
      const details = document.getElementById(`belief-details-${drone.id}`);
      const belief = state.beliefs.get(drone.id);
      const objects = belief ? belief.objects : [];
      const visibleObjects = state.priorityOnly ? objects.filter((object) => object.priority) : objects;
      const localCount = visibleObjects.filter((object) => object.own_sensor).length;
      const teammateOnlyCount = visibleObjects.filter((object) => object.teammate_sourced && !object.own_sensor).length;
      const multiSourceCount = visibleObjects.filter((object) => (object.sources || []).length > 1).length;
      const staleCount = visibleObjects.filter((object) => object.stale).length;
      const unresolvedCount = belief ? belief.unresolved_observation_count || 0 : 0;

      const counts = `
        <div class="count-row">
          <div class="count-box"><strong>${localCount}</strong> local</div>
          <div class="count-box"><strong>${teammateOnlyCount}</strong> teammate-only</div>
          <div class="count-box"><strong>${multiSourceCount}</strong> multi-source</div>
          <div class="count-box"><strong>${staleCount}</strong> stale · ${unresolvedCount} unresolved</div>
        </div>
      `;

      if (!belief || visibleObjects.length === 0) {
        details.innerHTML = `${counts}<div class="empty-state">This node's independently maintained map has no objects.</div>`;
        return;
      }

      const activeDetail = activeMapDetailFor("belief", drone.id);

      details.innerHTML = `
        ${counts}
        <details class="detected-objects"${state.beliefObjectTablesOpen.get(drone.id) ? " open" : ""}>
          <summary>Detected objects (${visibleObjects.length})</summary>
          <table class="object-table">
          <thead>
            <tr>
              <th>Local object ID</th>
              <th>Class</th>
              <th>Recent sources</th>
              <th>Type</th>
              <th>Freshness</th>
            </tr>
          </thead>
          <tbody>
            ${visibleObjects.map((object) => {
              const age = Math.max(0, belief.simulation_time - object.last_seen);
              const sourceType = object.own_sensor && object.teammate_sourced
                ? "multi-source"
                : object.own_sensor
                  ? "own"
                  : "remote";
              return `
                <tr>
                  <td>${object.local_object_id || object.object_id}</td>
                  <td>${object.priority ? "⭐ " : ""}${object.label}</td>
                  <td>${object.source_names.join(", ")}</td>
                  <td>${sourceType}</td>
                  <td>${object.stale ? "stale" : "fresh"} · ${age.toFixed(1)} s · ${object.active_contribution_count || 0} active</td>
                </tr>
              `;
            }).join("")}
          </tbody>
          </table>
        </details>
      `;

      const detectedObjects = details.querySelector(".detected-objects");
      if (detectedObjects) {
        detectedObjects.addEventListener("toggle", () => {
          state.beliefObjectTablesOpen.set(drone.id, detectedObjects.open);
        });
      }
    }

    function renderDiagnostics(drone) {
      const details = document.getElementById(`diagnostics-${drone.id}`);
      const status = state.statuses.get(drone.id);
      const diagnostics = status ? status.diagnostics : null;

      if (!diagnostics) {
        details.innerHTML = '<div class="empty-state">Diagnostics unavailable while node is offline.</div>';
        return;
      }

      const peers = diagnostics.peer_connection_state || [];
      details.innerHTML = `
        <dl class="diagnostics-grid">
          <dt>Local observations generated</dt><dd>${diagnostics.local_observations_generated}</dd>
          <dt>Peer observations sent</dt><dd>${diagnostics.peer_observations_sent}</dd>
          <dt>Peer observations received</dt><dd>${diagnostics.peer_observations_received}</dd>
          <dt>Unique observations accepted</dt><dd>${diagnostics.unique_observations_accepted}</dd>
          <dt>Duplicates ignored</dt><dd>${diagnostics.duplicates_ignored}</dd>
          <dt>Validation rejects</dt><dd>${diagnostics.validation_rejects}</dd>
          <dt>Most recent received ID</dt><dd>${diagnostics.most_recent_received_observation_id || "—"}</dd>
          <dt>Most recent original source</dt><dd>${diagnostics.most_recent_received_source || "—"}</dd>
          <dt>Peer connection state</dt><dd>${peers.map((peer) => `${peer.peer}: ${peer.state}`).join("<br>") || "—"}</dd>
        </dl>
      `;
    }

    function renderMaps() {
      const points = allMapPoints();

      if (points.length === 0) {
        return;
      }

      const bounds = boundsFor(points);
      const truthCanvas = document.getElementById("truth-map");
      if (drawBaseMap(truthCanvas, bounds)) {
        drawFlightTrails(truthCanvas, bounds);
        drawFootprintsAndDrones(truthCanvas, bounds);
        drawTruth(truthCanvas, bounds);
      }
      renderTruthDetails();

      for (const drone of drones) {
        const canvas = document.getElementById(`belief-map-${drone.id}`);
        if (drawBaseMap(canvas, bounds)) {
          drawFootprintsAndDrones(canvas, bounds, drone.id);
          drawBeliefObjects(canvas, bounds, drone.id);
        }
        renderBeliefDetails(drone);
        renderDiagnostics(drone);
      }

      renderSharedInspector();
    }

    const STATUS_REPORT_STALE_SECONDS = 3.5;

    function escapeHtml(value) {
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
    }

    function taskIdFor(task) {
      return task.task_id || task.signature_id || task.object_id || "";
    }

    function assignedDroneIdFor(task) {
      const rawId = task.assigned_drone_id ?? task.owner_drone_id;

      if (rawId === undefined || rawId === null || rawId === "") {
        return null;
      }

      const numericId = Number(rawId);
      return Number.isFinite(numericId) ? numericId : null;
    }

    function assignedDroneNameFor(task) {
      return task.assigned_drone_name || task.owner_drone_name || "queued";
    }

    function stableTaskSuffix(task) {
      const raw = String(taskIdFor(task) || task.signature_id || "");
      const compact = raw.replace(/[^a-zA-Z0-9]/g, "").slice(-4).toUpperCase();
      return compact || "TASK";
    }

    function titleCaseLabel(value) {
      return String(value || "Objective")
        .replace(/^priority\s+/i, "")
        .replaceAll("_", " ")
        .replace(/\s+/g, " ")
        .trim()
        .split(" ")
        .filter(Boolean)
        .map((part) => `${part.slice(0, 1).toUpperCase()}${part.slice(1).toLowerCase()}`)
        .join(" ");
    }

    function taskDisplayName(task, reports = []) {
      const label = task.object_label || task.label || task.classification_label || task.display_label;
      const baseName = `PRIORITY ${titleCaseLabel(label)}`;
      const duplicateCount = reports.filter((report) => {
        const otherLabel = report.task.display_label || report.task.object_label || report.task.label || report.task.classification_label;
        return titleCaseLabel(otherLabel) === titleCaseLabel(label);
      }).length;

      return duplicateCount > 1 ? `${baseName} · ${stableTaskSuffix(task)}` : baseName;
    }

    function taskDebugTitle(task) {
      return [
        `task_id=${task.task_id || "—"}`,
        `signature_id=${task.signature_id || "—"}`,
        `status=${task.status || "—"}`,
      ].join(" · ");
    }

    function normalizedTasksForSnapshot(snapshot) {
      const tasks = [...(snapshot.responseTasks || [])];

      if (snapshot.responseTask) {
        const responseTaskId = taskIdFor(snapshot.responseTask);

        if (!responseTaskId || !tasks.some((task) => taskIdFor(task) === responseTaskId)) {
          tasks.push(snapshot.responseTask);
        }
      }

      return tasks;
    }

    function normalizedExecutionState(value) {
      if (!value) {
        return null;
      }

      const normalized = String(value).toLowerCase().replaceAll("_", " ");

      if (normalized === "assigned remote" || normalized === "assigned local" || normalized === "assigned") {
        return null;
      }

      if (normalized.includes("orbit")) {
        return "orbiting";
      }

      if (normalized.includes("transition") || normalized.includes("enroute") || normalized.includes("respond")) {
        return "transitioning";
      }

      if (normalized.includes("search") || normalized.includes("patrol")) {
        return "searching";
      }

      if (normalized.includes("queue")) {
        return "queued";
      }

      return normalized;
    }

    function peerReportedExecutionState(task) {
      return normalizedExecutionState(
        task.owner_task_state
          || task.owner_task_status
          || task.owner_current_mode
          || task.owner_mode,
      );
    }

    function taskUpdateTime(task) {
      for (const field of ["owner_task_updated_at", "owner_task_state_updated_at", "task_state_updated_at", "updated_at", "last_observation_at"]) {
        const value = Number(task[field]);

        if (Number.isFinite(value)) {
          return value;
        }
      }

      return 0;
    }

    function isFreshSnapshot(snapshot) {
      return Boolean(snapshot && snapshot.online && ((Date.now() / 1000) - snapshot.receivedAt) <= STATUS_REPORT_STALE_SECONDS);
    }

    function snapshotTaskReportAge(snapshot) {
      if (!snapshot || !snapshot.online) {
        return "unavailable";
      }

      const age = Math.max(0, (Date.now() / 1000) - snapshot.receivedAt);
      return age > STATUS_REPORT_STALE_SECONDS ? `stale · ${age.toFixed(1)} s old` : `live · ${age.toFixed(1)} s old`;
    }

    function observationAgeFor(task, snapshot) {
      if (!task || !snapshot || task.last_observation_at === undefined) {
        return "—";
      }

      return `${Math.max(0, Number(snapshot.simulationTime || 0) - Number(task.last_observation_at)).toFixed(1)} s`;
    }

    function snapshotActiveTaskIds(snapshot) {
      const activeIds = new Set();

      if (snapshot.assignedObjective) {
        activeIds.add(String(snapshot.assignedObjective));
      }

      if (snapshot.responseTask) {
        activeIds.add(String(snapshot.responseTask.task_id || ""));
        activeIds.add(String(snapshot.responseTask.signature_id || ""));
      }

      return activeIds;
    }

    function reportMatchesOwnerActiveTask(report) {
      const activeIds = snapshotActiveTaskIds(report.snapshot);

      if (activeIds.size === 0) {
        return false;
      }

      return activeIds.has(String(report.task.task_id || "")) || activeIds.has(String(report.task.signature_id || ""));
    }

    function activeTaskForDrone(droneId) {
      const snapshot = state.taskSnapshotsByDrone.get(droneId);

      if (!snapshot) {
        return null;
      }

      return normalizedTasksForSnapshot(snapshot).find((task) => reportMatchesOwnerActiveTask({ snapshot, task })) || null;
    }

    function numericTaskField(task, fields) {
      for (const field of fields) {
        const value = Number(task[field]);

        if (Number.isFinite(value)) {
          return value;
        }
      }

      return 0;
    }

    function assignmentRevision(task) {
      return numericTaskField(task, ["assignment_revision", "revision", "assignment_epoch"]);
    }

    function assignmentEpoch(task) {
      return numericTaskField(task, ["assignment_epoch", "playbook_config_version"]);
    }

    function assignmentClearsOwner(task) {
      return ["queued", "reported", "released", "cancelled", "infeasible"].includes(String(task.status || "").toLowerCase());
    }

    function executingTaskStatus(task) {
      return ["transitioning", "orbiting"].includes(String(task.status || "").toLowerCase());
    }

    function reportAuthority(report) {
      const assignedDroneId = assignedDroneIdFor(report.task);

      if (assignedDroneId !== null && report.snapshot.droneId === assignedDroneId && reportMatchesOwnerActiveTask(report)) {
        return 4;
      }

      if (assignedDroneId !== null && report.snapshot.droneId === assignedDroneId) {
        return 3;
      }

      if (peerReportedExecutionState(report.task)) {
        return 2;
      }

      return 1;
    }

    function compareAssignmentReports(left, right) {
      const leftKey = [
        assignmentEpoch(left.task),
        assignmentRevision(left.task),
        reportAuthority(left),
        taskUpdateTime(left.task),
      ];
      const rightKey = [
        assignmentEpoch(right.task),
        assignmentRevision(right.task),
        reportAuthority(right),
        taskUpdateTime(right.task),
      ];

      for (let index = 0; index < leftKey.length; index += 1) {
        if (leftKey[index] !== rightKey[index]) {
          return rightKey[index] - leftKey[index];
        }
      }

      return String(taskIdFor(left.task)).localeCompare(String(taskIdFor(right.task)));
    }

    function authoritativeAssignmentReport(reports) {
      return reports.slice().sort(compareAssignmentReports)[0];
    }

    function activeOwnerConflicts(reports, authoritativeReport) {
      const authoritativeRevision = assignmentRevision(authoritativeReport.task);
      const authoritativeEpoch = assignmentEpoch(authoritativeReport.task);
      const owners = new Set();

      for (const report of reports) {
        const ownerId = assignedDroneIdFor(report.task);

        if (
          ownerId !== null
          && executingTaskStatus(report.task)
          && !assignmentClearsOwner(report.task)
          && assignmentRevision(report.task) === authoritativeRevision
          && assignmentEpoch(report.task) === authoritativeEpoch
        ) {
          owners.add(ownerId);
        }
      }

      return owners.size > 1 ? owners : null;
    }

    function buildAssignmentRows() {
      const reportsByTaskId = new Map();

      for (const snapshot of state.taskSnapshotsByDrone.values()) {
        for (const task of normalizedTasksForSnapshot(snapshot)) {
          const taskId = taskIdFor(task);

          if (!taskId) {
            continue;
          }

          if (!reportsByTaskId.has(taskId)) {
            reportsByTaskId.set(taskId, []);
          }

          reportsByTaskId.get(taskId).push({ snapshot, task });
        }
      }

      return [...reportsByTaskId.entries()]
        .sort(([left], [right]) => left.localeCompare(right))
        .map(([taskId, reports]) => {
          const authoritativeReport = authoritativeAssignmentReport(reports);
          const assignedReport = authoritativeReport;
          const ownerCleared = assignmentClearsOwner(assignedReport.task);
          const assignedDroneId = ownerCleared ? null : assignedDroneIdFor(assignedReport.task);
          const ownerReport = assignedDroneId === null
            ? null
            : reports.find((report) => report.snapshot.droneId === assignedDroneId);
          const freshOwnerReport = ownerReport && isFreshSnapshot(ownerReport.snapshot) ? ownerReport : null;
          const freshOwnerActiveReport = freshOwnerReport && reportMatchesOwnerActiveTask(freshOwnerReport) ? freshOwnerReport : null;
          const freshOwnerInactiveReport = freshOwnerReport && !freshOwnerActiveReport ? freshOwnerReport : null;
          const peerReport = assignedDroneId === null
            ? null
            : reports
              .filter((report) => report.snapshot.droneId !== assignedDroneId && peerReportedExecutionState(report.task))
              .sort(compareAssignmentReports)[0];
          const conflicts = activeOwnerConflicts(reports, authoritativeReport);

          let selectedReport = authoritativeReport;
          let taskState = "Assigned — activity unknown";
          let sourceLabel = ownerReport ? `owner ${snapshotTaskReportAge(ownerReport.snapshot)}` : "owner report unavailable";

          if (ownerCleared || assignedDroneId === null) {
            taskState = normalizedExecutionState(assignedReport.task.status) || String(assignedReport.task.status || "queued");
            sourceLabel = `${taskState} objective`;
          } else if (freshOwnerActiveReport) {
            selectedReport = freshOwnerActiveReport;
            taskState = normalizedExecutionState(freshOwnerActiveReport.snapshot.currentMode)
              || normalizedExecutionState(freshOwnerActiveReport.task.status)
              || "Assigned — activity unknown";
            sourceLabel = `owner ${snapshotTaskReportAge(freshOwnerActiveReport.snapshot)}`;
          } else if (freshOwnerInactiveReport) {
            selectedReport = freshOwnerInactiveReport;
            taskState = "reported";
            sourceLabel = `owner ${snapshotTaskReportAge(freshOwnerInactiveReport.snapshot)} · not active on owner · historical owner not displayed as current`;
          } else if (peerReport) {
            selectedReport = peerReport;
            taskState = `${peerReportedExecutionState(peerReport.task)} (peer-reported)`;
            sourceLabel = ownerReport
              ? `owner ${snapshotTaskReportAge(ownerReport.snapshot)} · peer report live`
              : "owner report unavailable · peer report live";
          }

          if (conflicts) {
            sourceLabel = `${sourceLabel} · diagnostic: conflicting current owners ${[...conflicts].join(", ")}`;
          }

          return {
            taskId,
            signatureId: assignedReport.task.signature_id || taskId,
            displayName: taskDisplayName(assignedReport.task, reports),
            debugTitle: taskDebugTitle(assignedReport.task),
            assignedDroneName: (assignedDroneId === null || freshOwnerInactiveReport) ? "Unassigned" : assignedDroneNameFor(assignedReport.task),
            taskState,
            observationAge: observationAgeFor(selectedReport.task, selectedReport.snapshot),
            sourceLabel,
          };
        });
    }

    function buildTeamSummary() {
      const summary = {
        orbiting: 0,
        transitioning: 0,
        searching: 0,
        queued: 0,
      };

      for (const status of state.statuses.values()) {
        const mode = normalizedExecutionState(status.current_mode || "searching");

        if (mode === "orbiting") {
          summary.orbiting += 1;
        } else if (mode === "transitioning") {
          summary.transitioning += 1;
        } else {
          summary.searching += 1;
        }
      }

      for (const row of buildAssignmentRows()) {
        if (row.taskState === "queued" || row.assignedDroneName === "queued") {
          summary.queued += 1;
        }
      }

      return summary;
    }

    function afterLayout(callback) {
      if (typeof window.requestAnimationFrame === "function") {
        window.requestAnimationFrame(() => window.requestAnimationFrame(callback));
      } else {
        setTimeout(callback, 0);
      }
    }

    function arrangeDashboardTabs() {
      document.getElementById("tab-team").appendChild(document.getElementById("map-grid-section"));
      document.getElementById("tab-team").appendChild(document.getElementById("belief-display-card"));
      document.getElementById("tab-team").appendChild(document.getElementById("shared-inspector-card"));
      document.getElementById("tab-team").appendChild(document.getElementById("assignment-card"));
      document.getElementById("tab-truth").appendChild(document.getElementById("truth-card"));
      document.getElementById("tab-network").appendChild(document.getElementById("network-card"));
    }

    function setActiveTab(tabName) {
      state.activeTab = tabName;

      try {
        window.localStorage && window.localStorage.setItem("drone-dashboard-active-tab", tabName);
      } catch (error) {
        // Local storage is optional; tab switching must never affect simulation behavior.
      }

      for (const button of document.querySelectorAll("[data-tab]")) {
        const selected = button.getAttribute("data-tab") === tabName;
        button.setAttribute("aria-selected", selected ? "true" : "false");
        button.setAttribute("tabindex", selected ? "0" : "-1");
      }

      for (const panel of document.querySelectorAll(".tab-panel")) {
        panel.classList.toggle("active", panel.id === `tab-${tabName}`);
      }

      afterLayout(renderMaps);
    }

    function restoreActiveTab() {
      let tabName = "team";

      try {
        tabName = (window.localStorage && window.localStorage.getItem("drone-dashboard-active-tab")) || "team";
      } catch (error) {
        tabName = "team";
      }

      setActiveTab(["team", "truth", "network"].includes(tabName) ? tabName : "team");
    }

    function setupCanvasResizeObserver() {
      if (typeof ResizeObserver !== "function") {
        return;
      }

      const canvases = [
        document.getElementById("truth-map"),
        ...drones.map((drone) => document.getElementById(`belief-map-${drone.id}`)),
      ];
      const observer = new ResizeObserver(() => {
        afterLayout(renderMaps);
      });

      for (const canvas of canvases) {
        observer.observe(canvas);
      }
    }

    function clearInspectionSelection(message) {
      state.selectedMapObject = null;
      state.hoveredMapObject = null;
      state.selectedObservation = null;
      state.selectedObservationKey = null;
      state.selectedObservationSnapshot = null;
      state.observationSelectionMode = "latest";
      state.rawJsonOpen = false;
      state.rawJsonScrollTop = 0;
      state.rawJsonSelectionStart = 0;
      state.rawJsonSelectionEnd = 0;
      state.rawJsonHadFocus = false;
      state.selectionMessage = message;
      state.observationTrace.clear();
    }

    function renderNetwork() {
      const details = document.getElementById("network-details");
      const summary = document.getElementById("network-summary");
      const availableNetworks = drones
        .map((drone) => state.networks.get(drone.id))
        .filter(Boolean);
      const recentlyReachablePeers = availableNetworks.reduce((total, network) => (
        total + (network.peer_connection_state || []).filter((peer) => peer.reachability === "recently_reachable").length
      ), 0);
      const failingPeers = availableNetworks.reduce((total, network) => (
        total + (network.peer_connection_state || []).filter((peer) => peer.reachability === "failing").length
      ), 0);
      const payloadRate = availableNetworks.reduce((total, network) => (
        total
        + Number((network.payload_bytes_per_second || {}).sent || 0)
        + Number((network.payload_bytes_per_second || {}).received || 0)
      ), 0);
      const sendFailures = availableNetworks.reduce((total, network) => total + Number(network.send_failures || 0), 0);

      summary.textContent = `${availableNetworks.length}/${drones.length} nodes reporting · ${recentlyReachablePeers} recently reachable peers · ${failingPeers} failing · ${payloadRate.toFixed(1)} B/s JSON payload only · ${sendFailures} send failures`;

      details.innerHTML = drones.map((drone) => {
        const network = state.networks.get(drone.id);

        if (!network) {
          return `
            <div class="control-panel">
              <h3>${escapeHtml(drone.name)}</h3>
              <div class="empty-state">Node offline or network diagnostics unavailable.</div>
            </div>
          `;
        }

        return `
          <div class="control-panel">
            <h3>${escapeHtml(network.node_name)} · node ${escapeHtml(network.node_id)}</h3>
            <dl class="diagnostics-grid">
              <dt>Hostname</dt><dd>${escapeHtml(network.hostname)}</dd>
              <dt>PID</dt><dd>${escapeHtml(network.pid)}</dd>
              <dt>Application build</dt><dd>${escapeHtml(network.app_build)}</dd>
              <dt>API address</dt><dd>${escapeHtml(network.api_address)}</dd>
              <dt>Configured peers</dt><dd>${(network.configured_peers || []).map(escapeHtml).join("<br>") || "—"}</dd>
              <dt>Observation messages</dt><dd>${network.observation_messages_sent} sent · ${network.observation_messages_received} received</dd>
              <dt>Task/config messages</dt><dd>${network.claim_messages_sent} sent · ${network.claim_messages_received} received</dd>
              <dt>JSON payload bytes</dt><dd>${network.payload_bytes_sent} sent · ${network.payload_bytes_received} received</dd>
              <dt>Rolling JSON payload rate</dt><dd>${Number(network.payload_bytes_per_second.sent || 0).toFixed(1)} B/s sent · ${Number(network.payload_bytes_per_second.received || 0).toFixed(1)} B/s received · JSON payload only</dd>
              <dt>Send failures</dt><dd>${network.send_failures || 0}</dd>
              <dt>Duplicates / rejects</dt><dd>${network.duplicate_observations_ignored} duplicates · ${network.validation_rejects} validation rejects</dd>
              <dt>Trace retention</dt><dd>${network.trace_retained_count}/${network.trace_retention_limit} retained · ${network.trace_evictions} evicted</dd>
            </dl>
            <table class="network-table">
              <thead><tr><th>Peer</th><th>Reachability</th><th>Last RTT</th><th>Last success</th><th>Latest error</th><th>Sent</th></tr></thead>
              <tbody>
                ${(network.peer_connection_state || []).map((peer) => `
                  <tr>
                    <td>${escapeHtml(peer.peer)}</td>
                    <td>${escapeHtml(peer.reachability || "unknown")}</td>
                    <td>${peer.last_rtt_ms === null || peer.last_rtt_ms === undefined ? "—" : `${Number(peer.last_rtt_ms).toFixed(1)} ms`}</td>
                    <td>${peer.last_success_wall_time === null || peer.last_success_wall_time === undefined ? "—" : "recorded with monotonic wall time"}</td>
                    <td>${escapeHtml(peer.latest_error || peer.last_error || "—")}</td>
                    <td>${escapeHtml(peer.observation_messages_sent || 0)} obs · ${escapeHtml(peer.claim_messages_sent || 0)} task/config · ${escapeHtml(peer.payload_bytes_sent || 0)} bytes · ${escapeHtml(peer.send_failures || 0)} failures</td>
                  </tr>
                `).join("")}
              </tbody>
            </table>
          </div>
        `;
      }).join("");
    }

    async function traceObservation(observationId) {
      const rows = await Promise.all(drones.map(async (drone) => {
        try {
          return await fetchJson(`${drone.url}/trace?observation_id=${encodeURIComponent(observationId)}`);
        } catch (error) {
          return {
            drone_id: drone.id,
            drone_name: drone.name,
            observation_id: observationId,
            retained_trace_count: 0,
            trace_evictions: 0,
            record: null,
            status: "node_unavailable",
          };
        }
      }));
      state.observationTrace.set(observationId, rows);
      refreshMapDetailPanels();
    }

    function verificationLabel(status) {
      return status === "verified"
        ? "Verified"
        : status === "failed"
          ? "Failed"
          : status === "pending"
            ? "Pending"
            : "Not evaluated";
    }

    function rememberEvidence(key, status, detail) {
      const previous = state.handoffEvidence.get(key);

      if (previous && previous.status === "verified" && status !== "failed") {
        return previous;
      }

      const evidence = { status, detail };
      state.handoffEvidence.set(key, evidence);
      return evidence;
    }

    function evaluateHandoffEvidence() {
      if (state.scenarioId !== "moving_person_handoff") {
        return new Map();
      }

      const initialName = droneNameForId(state.initialObserverId);
      const nextName = droneNameForId(state.nextObserverId);
      const onlineDroneIds = new Set(state.statuses.keys());
      const movingObjectsByDrone = new Map();

      for (const [droneId, belief] of state.beliefs.entries()) {
        movingObjectsByDrone.set(
          droneId,
          (belief.objects || []).filter((object) => object.tracking_model === "constant_velocity"),
        );
      }

      const allMeasurements = [...movingObjectsByDrone.values()]
        .flat()
        .flatMap((object) => object.measurements || [])
        .filter((measurement) => String(measurement.scenario_run_id || "") === String(state.scenarioRunId || ""));

      const acquired = allMeasurements.some((measurement) => Number(measurement.source_drone_id) === state.initialObserverId);
      const nextAccepted = allMeasurements.some((measurement) => Number(measurement.source_drone_id) === state.nextObserverId);
      const trackedObjects = [...movingObjectsByDrone.values()].flat();
      const lifecycleStates = new Set(trackedObjects.map((object) => String(object.track_status || "")));
      const thirdDroneId = drones
        .map((drone) => drone.id)
        .find((droneId) => droneId !== state.initialObserverId && droneId !== state.nextObserverId);
      const thirdObjects = movingObjectsByDrone.get(thirdDroneId) || [];
      const thirdRemoteTrack = thirdObjects.some((object) => (
        !object.own_sensor
        && Boolean(object.teammate_sourced)
        && (object.historical_sources || []).some((sourceId) => [state.initialObserverId, state.nextObserverId].includes(Number(sourceId)))
      ));

      let idsPreserved = onlineDroneIds.size > 0;
      for (const droneId of onlineDroneIds) {
        const candidate = (movingObjectsByDrone.get(droneId) || []).find((object) => {
          const sources = new Set((object.historical_sources || []).map(Number));
          return sources.has(state.initialObserverId) && sources.has(state.nextObserverId);
        });

        if (!candidate) {
          idsPreserved = false;
          continue;
        }

        const priorId = state.handoffObjectIdsByDrone.get(droneId);
        const currentId = String(candidate.local_object_id || candidate.object_id || "");

        if (priorId && priorId !== currentId) {
          idsPreserved = false;
        } else if (currentId) {
          state.handoffObjectIdsByDrone.set(droneId, currentId);
        }
      }

      rememberEvidence(
        "initial_acquired",
        acquired ? "verified" : "pending",
        acquired ? `Accepted evidence from ${initialName}.` : `Waiting for accepted evidence from ${initialName}.`,
      );
      rememberEvidence(
        "next_accepted",
        nextAccepted ? "verified" : "pending",
        nextAccepted ? `Accepted evidence from ${nextName}.` : `Waiting for accepted evidence from ${nextName}.`,
      );
      rememberEvidence(
        "id_preserved",
        idsPreserved && acquired && nextAccepted ? "verified" : "pending",
        idsPreserved && acquired && nextAccepted
          ? "The same local mapped-object ID is retained after both observer sources contribute."
          : "Waiting for both observer sources on each reporting node without local object-ID churn.",
      );
      rememberEvidence(
        "third_remote",
        thirdRemoteTrack ? "verified" : "pending",
        thirdRemoteTrack
          ? `${droneNameForId(thirdDroneId)} maintains a remotely learned moving track.`
          : `${droneNameForId(thirdDroneId)} has not yet retained a remote moving track for this run.`,
      );
      rememberEvidence(
        "coasting",
        lifecycleStates.has("coasting") || lifecycleStates.has("lost") ? "verified" : "pending",
        lifecycleStates.has("coasting") || lifecycleStates.has("lost")
          ? "Estimator entered coasting after measurements stopped."
          : "Coasting has not been observed yet.",
      );
      rememberEvidence(
        "lost",
        lifecycleStates.has("lost") ? "verified" : "pending",
        lifecycleStates.has("lost")
          ? "Estimator transitioned to lost after the configured timeout."
          : "Lost transition has not been observed yet.",
      );

      return state.handoffEvidence;
    }

    function handoffEvidenceSummaryHtml() {
      if (state.scenarioId !== "moving_person_handoff") {
        return "";
      }

      const evidence = evaluateHandoffEvidence();
      const labels = [
        ["initial_acquired", "Acquired by initial observer"],
        ["next_accepted", "Reports accepted from next observer"],
        ["id_preserved", "Mapped-object ID preserved"],
        ["third_remote", "Third node remote track"],
        ["coasting", "Coasting after measurements stopped"],
        ["lost", "Lost after timeout"],
      ];

      return `
        <div class="evidence-grid">
          ${labels.map(([key, label]) => {
            const item = evidence.get(key) || { status: "not_evaluated", detail: "" };
            return `<div class="verification"><strong>${escapeHtml(label)}:</strong> ${escapeHtml(verificationLabel(item.status))}<br><span class="control-note">${escapeHtml(item.detail)}</span></div>`;
          }).join("")}
        </div>
      `;
    }

    function renderAssignments() {
      const details = document.getElementById("assignment-details");
      const rows = buildAssignmentRows();
      const droneModes = drones.map((drone) => {
        const status = state.statuses.get(drone.id);
        const activeTask = activeTaskForDrone(drone.id);
        const mode = status ? status.current_mode || "patrol" : "offline";

        if (!status) {
          return `${drone.name}: offline`;
        }

        if (activeTask && mode === "transitioning") {
          return `${drone.name}: transitioning to ${taskDisplayName(activeTask)}`;
        }

        if (activeTask && mode !== "patrol") {
          return `${drone.name}: ${mode} ${taskDisplayName(activeTask)}`;
        }

        return `${drone.name}: ${mode}`;
      }).join(" · ");

      if (rows.length === 0) {
        details.innerHTML = `<div>${droneModes}</div><div class="empty-state">No priority objectives discovered yet.</div>`;
        return;
      }

      details.innerHTML = `
        <div>${droneModes}</div>
        <table class="assignment-table">
          <thead><tr><th>Search objective</th><th>Assigned drone</th><th>Task state</th><th>Last observation age</th><th>Status report</th></tr></thead>
          <tbody>
            ${rows.map((row) => `<tr>
              <td title="${escapeHtml(row.debugTitle)}">${escapeHtml(row.displayName)}</td>
              <td>${escapeHtml(row.assignedDroneName)}</td>
              <td>${escapeHtml(row.taskState)}</td>
              <td>${escapeHtml(row.observationAge)}</td>
              <td>${escapeHtml(row.sourceLabel)}</td>
            </tr>`).join("")}
          </tbody>
        </table>
      `;
    }

    function handoffStageFromEvidence() {
      if (state.scenarioId !== "moving_person_handoff") {
        return "Handoff demo is not active.";
      }

      const acceptedMeasurements = [];
      const trackStatuses = new Set();

      for (const belief of state.beliefs.values()) {
        for (const object of belief.objects || []) {
          if (object.tracking_model !== "constant_velocity") {
            continue;
          }

          if (object.track_status) {
            trackStatuses.add(String(object.track_status));
          }

          for (const measurement of object.measurements || []) {
            if (String(measurement.scenario_run_id || "") === String(state.scenarioRunId || "")) {
              acceptedMeasurements.push(measurement);
            }
          }
        }
      }

      if (acceptedMeasurements.length === 0) {
        return "Waiting for accepted moving-track evidence.";
      }

      const sourceIds = new Set(acceptedMeasurements.map((measurement) => Number(measurement.source_drone_id)));
      const initialName = droneNameForId(state.initialObserverId);
      const nextName = droneNameForId(state.nextObserverId);

      if (trackStatuses.has("lost")) {
        return `Accepted tracking evidence retained; latest track lifecycle is lost after ${initialName} → ${nextName}.`;
      }

      if (sourceIds.has(state.initialObserverId) && sourceIds.has(state.nextObserverId)) {
        return `Accepted tracking evidence from ${initialName} and ${nextName}; handoff/overlap confirmed.`;
      }

      if (sourceIds.has(state.nextObserverId)) {
        return `Accepted next-observer tracking evidence from ${nextName}; waiting for continuity evidence.`;
      }

      if (sourceIds.has(state.initialObserverId)) {
        return `Accepted initial-observer tracking evidence from ${initialName}; waiting for ${nextName}.`;
      }

      return `Accepted tracking evidence from ${[...sourceIds].map(droneNameForId).join(", ")}.`;
    }

    function renderHandoffDemoFeedback() {
      const feedback = document.getElementById("handoff-demo-feedback");
      const button = document.getElementById("handoff-demo-button");
      const banner = document.getElementById("action-error-banner");

      if (!feedback || !button) {
        return;
      }

      const observerPair = `${droneNameForId(state.initialObserverId)} → ${droneNameForId(state.nextObserverId)}`;
      const demoTime = Number(state.truth.simulation_time || 0);
      const isHandoffScenario = state.scenarioId === "moving_person_handoff";
      const finished = isHandoffScenario && demoTime >= HANDOFF_DEMO_SECONDS;
      const runStatus = state.handoffConfigStatus === "error" || state.handoffDemoRunStatus === "error"
        ? "error"
        : state.handoffDemoRunStatus === "configuring"
          ? "configuring"
          : state.handoffDemoRunStatus === "finished" || finished
            ? "finished"
            : state.paused
              ? "paused"
              : isHandoffScenario
                ? "running"
                : "not running";

      button.textContent = finished || state.handoffDemoRunStatus === "finished" ? "Replay handoff demo" : "Run handoff demo";
      button.disabled = state.handoffDemoRunStatus === "configuring";

      if (banner) {
        banner.hidden = !state.handoffDemoError;
        banner.textContent = state.handoffDemoError ? `Action error: ${state.handoffDemoError}` : "";
      }

      feedback.innerHTML = [
        `Handoff demo seed: <strong>${HANDOFF_DEMO_SEED}</strong>`,
        `Configuration: <strong>${escapeHtml(state.handoffConfigStatus)}</strong>`,
        `Scenario: <strong>${isHandoffScenario ? "Jogging person handoff" : state.scenarioName}</strong>`,
        `Observer pair: <strong>${escapeHtml(observerPair)}</strong>`,
        `Demo time: <strong>${demoTime.toFixed(1)} s</strong>`,
        `Run status: <strong>${escapeHtml(runStatus)}</strong>`,
        `Stage: <strong>${escapeHtml(handoffStageFromEvidence())}</strong>`,
        state.sharingEnabled ? "<strong>Sharing enabled</strong>" : "<strong>Sharing disabled</strong>",
        state.handoffDemoAckMessage ? escapeHtml(state.handoffDemoAckMessage) : "",
        state.handoffDemoError ? `Error: ${escapeHtml(state.handoffDemoError)}` : "",
        handoffEvidenceSummaryHtml(),
      ].filter(Boolean).join(" · ");
    }

    function renderControls() {
      const teamSummary = buildTeamSummary();
      const acknowledgedVersions = drones.map((drone) => {
        const status = state.statuses.get(drone.id);
        return `${drone.name} v${status ? Number(status.playbook_config_version || 0) : "offline"}`;
      });
      const pendingConfig = state.pendingPlaybookConfig;
      const applied = pendingConfig
        && drones.every((drone) => {
          const status = state.statuses.get(drone.id);
          return status
            && status.selected_playbook === pendingConfig.playbook
            && Number(status.playbook_config_version || 0) >= pendingConfig.version;
        });

      if (applied) {
        state.pendingPlaybookConfig = null;
        state.playbookSelectDirty = false;
      }

      const playbookSelect = document.getElementById("playbook-select");
      if (!state.playbookSelectDirty && playbookSelect.value !== state.selectedPlaybook) {
        playbookSelect.value = state.selectedPlaybook;
      }

      document.getElementById("compact-scenario-name").textContent = state.scenarioName;
      document.getElementById("compact-run-state").textContent = state.handoffDemoRunStatus === "error" || state.handoffConfigStatus === "error"
        ? "Error"
        : state.paused
          ? "Paused"
          : Number(state.truth.simulation_time || 0) >= HANDOFF_DEMO_SECONDS && state.scenarioId === "moving_person_handoff"
            ? "Finished"
            : "Running";
      document.getElementById("compact-elapsed-time").textContent = `${Number(state.truth.simulation_time || 0).toFixed(1)} s`;
      document.getElementById("compact-pause-button").textContent = state.paused ? "Resume" : "Pause";

      if (!state.scenarioConfigDirty) {
        document.getElementById("scenario-select").value = state.scenarioId;
        document.getElementById("initial-observer-select").value = String(state.initialObserverId);
        document.getElementById("next-observer-select").value = String(state.nextObserverId);
        document.getElementById("priority-count-select").value = String(state.requestedPriorityCount || 0);
      }

      const scenarioSelect = document.getElementById("scenario-select");
      const selectedScenarioOption = scenarioSelect.selectedOptions && scenarioSelect.selectedOptions.length
        ? scenarioSelect.selectedOptions[0]
        : null;
      const scenarioLabel = selectedScenarioOption
        ? selectedScenarioOption.textContent
        : state.scenarioName || state.scenarioId || "selected scenario";
      document.getElementById("new-scenario-button").textContent = `New random ${scenarioLabel}`;

      document.getElementById("sharing-button").textContent = state.sharingEnabled ? "Disable sharing" : "Enable sharing";
      document.getElementById("pause-button").textContent = state.paused ? "Resume" : "Pause";
      document.getElementById("control-state").textContent = `${state.sharingEnabled ? "Sharing on" : "Sharing off"} · ${state.paused ? "paused" : "running"}`;
      document.getElementById("scenario-state").textContent = `${state.scenarioName} · seed ${state.scenarioSeed} · run ${state.scenarioRunId || "—"} · priority ${state.priorityObjectCount} · playbook ${state.selectedPlaybook} · build ${state.appBuild}`;
      document.getElementById("task-state").textContent = `${teamSummary.orbiting} orbiting · ${teamSummary.transitioning} transitioning · ${teamSummary.searching} searching · ${teamSummary.queued} queued`;
      document.getElementById("playbook-config-state").textContent = pendingConfig && !applied
        ? `Requested ${pendingConfig.playbook} v${pendingConfig.version} pending · applied ${state.selectedPlaybook} · ${acknowledgedVersions.join(" · ")}`
        : `Applied playbook ${state.selectedPlaybook} · ${acknowledgedVersions.join(" · ")}`;
      renderHandoffDemoFeedback();
    }

    async function postControlNoUpdate(payload, targetDrones = drones) {
      await Promise.all(targetDrones.map((drone) => fetchJson(`${drone.url}/control`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      })));
    }

    async function postControl(payload, targetDrones = drones) {
      await postControlNoUpdate(payload, targetDrones);
      await updateAll();
    }

    async function updateAll() {
      try {
        await updateTruth();
      } catch (error) {
        // Truth is fetched from an actual drone process; leave the prior frame if temporarily unavailable.
      }

      await Promise.all(drones.map(updateDrone));
      renderControls();
      renderAssignments();
      renderNetwork();
      renderMaps();

      if (
        state.scenarioId === "moving_person_handoff"
        && state.handoffDemoRunStatus === "running"
        && Number(state.truth.simulation_time || 0) >= HANDOFF_DEMO_SECONDS
        && !state.handoffDemoFinishPauseRequested
      ) {
        state.handoffDemoFinishPauseRequested = true;
        state.handoffDemoRunStatus = "finished";
        postControlNoUpdate({ paused: true, sharing_enabled: true })
          .then(updateAll)
          .catch((error) => {
            state.handoffDemoError = `Failed to pause completed demo: ${error.message}`;
            renderControls();
          });
      }
    }

    document.getElementById("sharing-button").addEventListener("click", () => {
      postControl({ sharing_enabled: !state.sharingEnabled });
    });

    document.getElementById("pause-button").addEventListener("click", () => {
      postControl({ paused: !state.paused });
    });

    document.getElementById("compact-pause-button").addEventListener("click", () => {
      postControl({ paused: !state.paused });
    });

    document.getElementById("compact-replay-button").addEventListener("click", () => {
      state.handoffDemoError = "";
      state.handoffDemoAckMessage = "";
      state.handoffDemoFinishPauseRequested = false;
      state.handoffEvidence.clear();
      state.handoffObjectIdsByDrone.clear();
      postControl({ restart_scenario: true, ...selectedRunOptions() });
    });

    function selectedRunOptions() {
      const initialObserverId = Number(document.getElementById("initial-observer-select").value);
      let nextObserverId = Number(document.getElementById("next-observer-select").value);

      if (nextObserverId === initialObserverId) {
        nextObserverId = drones.find((drone) => drone.id !== initialObserverId).id;
        document.getElementById("next-observer-select").value = String(nextObserverId);
      }

      return {
        priority_count: Number(document.getElementById("priority-count-select").value),
        playbook: document.getElementById("playbook-select").value,
        scenario: document.getElementById("scenario-select").value,
        initial_observer_id: initialObserverId,
        next_observer_id: nextObserverId,
        scenario_run_id: `dashboard-${Date.now()}-${Math.random().toString(16).slice(2)}`,
      };
    }

    function applyPlaybookConfiguration(playbook) {
      if (playbook === "orbit_follow") {
        return;
      }

      const version = Math.max(
        state.playbookConfigVersion + 1,
        ...[...state.statuses.values()].map((status) => Number(status.playbook_config_version || 0) + 1),
      );
      state.pendingPlaybookConfig = { playbook, version };
      document.getElementById("playbook-config-state").textContent = `Playbook ${playbook} v${version} pending`;
      postControl({ playbook, playbook_config_version: version });
    }

    document.getElementById("reset-button").addEventListener("click", () => {
      state.handoffDemoError = "";
      state.handoffDemoAckMessage = "";
      state.handoffDemoFinishPauseRequested = false;
      postControl({ restart_scenario: true, ...selectedRunOptions() });
    });

    document.getElementById("new-scenario-button").addEventListener("click", () => {
      const scenarioSeed = Math.floor(Date.now() % 1000000000);
      state.handoffDemoRunStatus = "idle";
      state.handoffDemoError = "";
      state.handoffDemoAckMessage = "";
      state.handoffDemoFinishPauseRequested = false;
      postControl({ scenario_seed: scenarioSeed, ...selectedRunOptions() });
    });

    function configurationMismatch(payload, expected) {
      const expectedScenario = expected.scenario_id || expected.scenario;
      const checks = [
        [Number(payload.scenario_seed) === Number(expected.scenario_seed), `seed ${payload.scenario_seed}`],
        [payload.scenario_id === expectedScenario, `scenario ${payload.scenario_id}`],
        [payload.scenario_run_id === expected.scenario_run_id, `run ${payload.scenario_run_id}`],
        [Number(payload.initial_observer_id) === Number(expected.initial_observer_id), `initial observer ${payload.initial_observer_id}`],
        [Number(payload.next_observer_id) === Number(expected.next_observer_id), `next observer ${payload.next_observer_id}`],
        [payload.selected_playbook === expected.playbook, `playbook ${payload.selected_playbook}`],
        [Boolean(payload.sharing_enabled) === Boolean(expected.sharing_enabled), `sharing ${payload.sharing_enabled}`],
        [Boolean(payload.paused) === Boolean(expected.paused), `paused ${payload.paused}`],
      ];
      return checks.filter(([passed]) => !passed).map(([_passed, label]) => label).join(", ");
    }

    async function waitForConfigurationAcknowledgement(expected) {
      const deadline = Date.now() + 8000;
      let lastMessage = "Waiting for node acknowledgements.";

      while (Date.now() < deadline) {
        const results = await Promise.all(drones.map(async (drone) => {
          try {
            const payload = await fetchJson(`${drone.url}/control`);
            const mismatch = configurationMismatch(payload, expected);
            return mismatch ? `${drone.name}: ${mismatch}` : "";
          } catch (error) {
            return `${drone.name}: unavailable`;
          }
        }));
        const failures = results.filter(Boolean);

        if (failures.length === 0) {
          state.handoffConfigStatus = "verified";
          state.handoffDemoAckMessage = "All nodes acknowledged handoff demo configuration.";
          renderControls();
          return;
        }

        lastMessage = `Configuring handoff demo · ${failures.join(" · ")}`;
        state.handoffDemoAckMessage = lastMessage;
        renderControls();
        await new Promise((resolve) => setTimeout(resolve, 250));
      }

      throw new Error(lastMessage);
    }

    async function runHandoffDemo() {
      const initialObserverId = Number(document.getElementById("initial-observer-select").value || 1);
      let nextObserverId = Number(document.getElementById("next-observer-select").value || 2);

      if (nextObserverId === initialObserverId) {
        nextObserverId = drones.find((drone) => drone.id !== initialObserverId).id;
        document.getElementById("next-observer-select").value = String(nextObserverId);
      }

      const version = Math.max(
        state.playbookConfigVersion + 1,
        ...[...state.statuses.values()].map((status) => Number(status.playbook_config_version || 0) + 1),
      );
      const payload = {
        restart_scenario: true,
        scenario_seed: HANDOFF_DEMO_SEED,
        priority_count: 0,
        playbook: "report_continue",
        playbook_config_version: version,
        scenario: "moving_person_handoff",
        initial_observer_id: initialObserverId,
        next_observer_id: nextObserverId,
        scenario_run_id: `handoff-demo-${HANDOFF_DEMO_SEED}-${Date.now()}-${Math.random().toString(16).slice(2)}`,
        paused: true,
        sharing_enabled: true,
      };

      state.handoffDemoRunStatus = "configuring";
      state.handoffConfigStatus = "pending";
      state.handoffDemoError = "";
      state.handoffDemoAckMessage = "Sending handoff demo configuration to all nodes.";
      state.handoffDemoFinishPauseRequested = false;
      state.handoffEvidence.clear();
      state.handoffObjectIdsByDrone.clear();
      state.pendingPlaybookConfig = { playbook: "report_continue", version };
      state.playbookSelectDirty = false;
      state.scenarioConfigDirty = false;
      clearInspectionSelection("Handoff demo replay started. Previous selected object and pinned sensor report were cleared.");
      renderControls();

      try {
        await postControlNoUpdate(payload);
        await waitForConfigurationAcknowledgement(payload);
        await postControlNoUpdate({ paused: false, sharing_enabled: true });
        state.handoffDemoRunStatus = "running";
        state.handoffConfigStatus = "verified";
        state.handoffDemoAckMessage = "All nodes acknowledged configuration; demo running with sharing enabled.";
        await updateAll();
      } catch (error) {
        state.handoffDemoRunStatus = "error";
        state.handoffConfigStatus = "error";
        state.handoffDemoError = error.message;
        state.handoffDemoAckMessage = "";
        await updateAll();
      }
    }

    document.getElementById("handoff-demo-button").addEventListener("click", () => {
      runHandoffDemo();
    });

    document.getElementById("playbook-select").addEventListener("change", (event) => {
      state.playbookSelectDirty = true;
      applyPlaybookConfiguration(event.target.value);
    });

    for (const id of ["priority-count-select", "scenario-select", "initial-observer-select", "next-observer-select"]) {
      document.getElementById(id).addEventListener("change", () => {
        state.scenarioConfigDirty = true;
      });
    }

    document.getElementById("cancel-response-button").addEventListener("click", () => {
      postControl({ cancel_response: true });
    });

    document.getElementById("trail-toggle").addEventListener("change", (event) => {
      state.showFlightTrails = event.target.checked;
      renderMaps();
    });

    document.getElementById("truth-label-toggle").addEventListener("change", (event) => {
      state.showTruthObjectLabels = event.target.checked;
      renderMaps();
    });

    document.getElementById("belief-label-toggle").addEventListener("change", (event) => {
      state.showBeliefObjectLabels = event.target.checked;
      renderMaps();
    });

    document.getElementById("background-truth-toggle").addEventListener("change", (event) => {
      state.showBackgroundTruthObjects = event.target.checked;
      renderMaps();
    });

    document.getElementById("measurement-toggle").addEventListener("change", (event) => {
      state.showMeasurements = event.target.checked;
      renderMaps();
    });

    document.getElementById("ellipse-toggle").addEventListener("change", (event) => {
      state.showEllipses = event.target.checked;
      renderMaps();
    });

    document.getElementById("priority-only-toggle").addEventListener("change", (event) => {
      state.priorityOnly = event.target.checked;
      renderMaps();
    });

    document.addEventListener("toggle", (event) => {
      if (event.target && event.target.id === "observation-json-details") {
        state.rawJsonOpen = Boolean(event.target.open);
      }
    }, true);

    document.addEventListener("scroll", (event) => {
      if (event.target && event.target.id === "observation-json-area") {
        state.rawJsonScrollTop = event.target.scrollTop;
      }
    }, true);

    document.addEventListener("click", (event) => {
      const observationButton = event.target.closest("[data-observation-id]");
      if (observationButton) {
        const target = latestTargetFor(state.selectedMapObject || state.hoveredMapObject);
        const observationId = observationButton.getAttribute("data-observation-id");
        const runId = observationButton.getAttribute("data-observation-run-id");

        if (target && observationId) {
          const observation = (target.object.measurements || []).find((item) => (
            item.observation_id === observationId
            && String(item.scenario_run_id || "unknown-run") === String(runId || "unknown-run")
          ));

          if (observation) {
            state.observationSelectionMode = "pinned";
            state.selectedObservationKey = observationKey(observation);
            state.selectedObservationSnapshot = snapshotObservation(observation, target);
            state.rawJsonOpen = true;
            state.selectionMessage = "";
            refreshMapDetailPanels();
          }
        }
      }

      const traceButton = event.target.closest("[data-trace-observation-id]");
      if (traceButton) {
        traceObservation(traceButton.getAttribute("data-trace-observation-id"));
      }

      const latestButton = event.target.closest("[data-observation-mode='latest']");
      if (latestButton) {
        state.observationSelectionMode = "latest";
        state.selectedObservationKey = null;
        state.selectedObservationSnapshot = null;
        state.rawJsonOpen = true;
        state.selectionMessage = "";
        refreshMapDetailPanels();
      }

      const copyButton = event.target.closest("[data-copy-observation-json]");
      if (copyButton && state.selectedObservationSnapshot) {
        navigator.clipboard.writeText(state.selectedObservationSnapshot.rawJson);
      }
    });

    function focusTabAtOffset(currentButton, offset) {
      const buttons = [...document.querySelectorAll("[role='tab']")];
      const currentIndex = buttons.indexOf(currentButton);
      const nextButton = buttons[(currentIndex + offset + buttons.length) % buttons.length];

      if (nextButton) {
        nextButton.focus();
        setActiveTab(nextButton.getAttribute("data-tab"));
      }
    }

    for (const button of document.querySelectorAll("[role='tab']")) {
      button.addEventListener("click", () => {
        setActiveTab(button.getAttribute("data-tab"));
      });
      button.addEventListener("keydown", (event) => {
        if (event.key === "ArrowRight") {
          event.preventDefault();
          focusTabAtOffset(button, 1);
        } else if (event.key === "ArrowLeft") {
          event.preventDefault();
          focusTabAtOffset(button, -1);
        } else if (event.key === "Home") {
          event.preventDefault();
          focusTabAtOffset(button, -drones.length);
        } else if (event.key === "End") {
          event.preventDefault();
          focusTabAtOffset(button, drones.length);
        }
      });
    }

    function setupMapInteractions() {
      const canvases = [
        document.getElementById("truth-map"),
        ...drones.map((drone) => document.getElementById(`belief-map-${drone.id}`)),
      ];

      for (const canvas of canvases) {
        canvas.addEventListener("mousemove", (event) => {
          const target = nearestMapTarget(canvas, event);
          state.hoveredMapObject = target;
          canvas.style.cursor = target ? "pointer" : "default";
          refreshMapDetailPanels();
        });

        canvas.addEventListener("mouseleave", () => {
          state.hoveredMapObject = null;
          canvas.style.cursor = "default";
          refreshMapDetailPanels();
        });

        canvas.addEventListener("click", (event) => {
          state.selectedMapObject = nearestMapTarget(canvas, event);
          state.observationSelectionMode = "latest";
          state.selectedObservationKey = null;
          state.selectedObservationSnapshot = null;
          state.selectionMessage = "";
          refreshMapDetailPanels();
        });
      }
    }

    arrangeDashboardTabs();
    drones.forEach(createCard);
    setupMapInteractions();
    setupCanvasResizeObserver();
    restoreActiveTab();
    updateAll();
    setInterval(updateAll, 1000);
  </script>
</body>
</html>
"""


class DashboardRequestHandler(http.server.BaseHTTPRequestHandler):
    """Serve the browser dashboard."""

    drones: list[dict[str, str | int]] = []

    def do_GET(self) -> None:
        if self.path not in {"/", "/index.html"}:
            self.send_error(404)
            return

        html = DASHBOARD_HTML.replace("__DRONES_JSON__", json.dumps(self.drones))
        payload = html.encode("utf-8")

        self.send_response(200)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:
        return


class ThreadingTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch drone simulator processes and open a browser dashboard.",
    )
    parser.add_argument("--dashboard-host", default=DEFAULT_DASHBOARD_HOST)
    parser.add_argument("--dashboard-port", type=int, default=DEFAULT_DASHBOARD_PORT)
    parser.add_argument("--drone-host", default=DEFAULT_DRONE_HOST)
    parser.add_argument(
        "--drone-ports",
        type=int,
        nargs=3,
        default=DEFAULT_DRONE_PORTS,
        metavar=("PORT1", "PORT2", "PORT3"),
    )
    parser.add_argument(
        "--priority-count",
        type=int,
        choices=(0, 1, 2, 3, 4, 5),
        default=RANDOM_PRIORITY_OBJECT_COUNT,
        help="Priority objective count: 0 means deterministic random 1-5.",
    )
    parser.add_argument(
        "--playbook",
        choices=("report_continue", "observe_orbit", "orbit_follow"),
        default=DEFAULT_PLAYBOOK,
        help="Simulation-only response playbook. orbit_follow is reserved for Stage B.",
    )
    parser.add_argument(
        "--scenario",
        choices=(DEFAULT_SCENARIO_ID, MOVING_HANDOFF_SCENARIO_ID),
        default=DEFAULT_SCENARIO_ID,
        help="Scenario geometry. Moving handoff is a shared-world tracking scenario, not a playbook.",
    )
    parser.add_argument("--initial-observer-id", type=int, choices=(1, 2, 3), default=DEFAULT_INITIAL_OBSERVER_ID)
    parser.add_argument("--next-observer-id", type=int, choices=(1, 2, 3), default=DEFAULT_NEXT_OBSERVER_ID)
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Start the processes without opening a browser.",
    )
    return parser.parse_args()


def wait_for_port(host: str, port: int, timeout_seconds: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_seconds

    while time.monotonic() < deadline:
        with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            sock.settimeout(0.2)

            if sock.connect_ex((host, port)) == 0:
                return True

        time.sleep(0.1)

    return False


def start_drone_processes(
    drone_host: str,
    drone_ports: tuple[int, int, int],
    scenario_seed: int,
    priority_count: int,
    playbook: str,
    scenario: str,
    initial_observer_id: int,
    next_observer_id: int,
    scenario_run_id: str,
) -> list[subprocess.Popen[str]]:
    drone_script = Path(__file__).with_name("simulated_drone.py")
    processes: list[subprocess.Popen[str]] = []

    for spec, port in zip(DRONE_SPECS, drone_ports, strict=True):
        peer_urls = [
            f"http://{drone_host}:{peer_port}"
            for peer_port in drone_ports
            if peer_port != port
        ]
        process = subprocess.Popen(
            [
                sys.executable,
                str(drone_script),
                "--id",
                str(spec["id"]),
                "--name",
                str(spec["name"]),
                "--host",
                drone_host,
                "--port",
                str(port),
                "--peers",
                *peer_urls,
                "--seed",
                str(scenario_seed),
                "--priority-count",
                str(priority_count),
                "--playbook",
                playbook,
                "--scenario",
                scenario,
                "--initial-observer-id",
                str(initial_observer_id),
                "--next-observer-id",
                str(next_observer_id),
                "--scenario-run-id",
                scenario_run_id,
            ],
            text=True,
        )
        processes.append(process)

    return processes


def terminate_processes(processes: list[subprocess.Popen[str]]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()

    for process in processes:
        if process.poll() is not None:
            continue

        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()

    for process in processes:
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(timeout=1)


def run_dashboard(
    dashboard_host: str,
    dashboard_port: int,
    drones: list[dict[str, str | int]],
) -> ThreadingTCPServer:
    DashboardRequestHandler.drones = drones
    server = ThreadingTCPServer((dashboard_host, dashboard_port), DashboardRequestHandler)
    thread = threading.Thread(target=server.serve_forever, name="dashboard-server", daemon=True)
    thread.start()
    return server


def main() -> int:
    args = parse_args()
    drone_ports = tuple(args.drone_ports)

    drones = [
        {
            "id": spec["id"],
            "name": spec["name"],
            "color": spec["color"],
            "url": f"http://{args.drone_host}:{port}",
        }
        for spec, port in zip(DRONE_SPECS, drone_ports, strict=True)
    ]

    print(f"Launcher path: {Path(__file__).resolve()}")
    print(f"Interpreter path: {sys.executable}")
    print(f"Launcher PID: {os.getpid()}")
    print(f"Application build/version: {APPLICATION_BUILD}")
    print(f"Scenario: {SCENARIO_NAME}")
    print(f"Scenario seed: {DEFAULT_SCENARIO_SEED}")

    playbook = args.playbook if args.playbook != "orbit_follow" else DEFAULT_PLAYBOOK
    initial_observer_id = args.initial_observer_id
    next_observer_id = args.next_observer_id if args.next_observer_id != initial_observer_id else next(
        int(spec["id"])
        for spec in DRONE_SPECS
        if int(spec["id"]) != initial_observer_id
    )
    scenario_run_id = f"launcher-{DEFAULT_SCENARIO_SEED}-{uuid.uuid4().hex[:12]}"
    processes = start_drone_processes(
        args.drone_host,
        drone_ports,
        DEFAULT_SCENARIO_SEED,
        args.priority_count,
        playbook,
        args.scenario,
        initial_observer_id,
        next_observer_id,
        scenario_run_id,
    )

    try:
        for process, port in zip(processes, drone_ports, strict=True):
            if not wait_for_port(args.drone_host, port):
                raise RuntimeError(f"Drone process on port {port} did not start.")

            if process.poll() is not None:
                raise RuntimeError(f"Drone process on port {port} exited unexpectedly.")

        dashboard = run_dashboard(args.dashboard_host, args.dashboard_port, drones)
        dashboard_url = f"http://{args.dashboard_host}:{args.dashboard_port}"

        print(f"Dashboard running at {dashboard_url}")
        print("Press Ctrl-C to stop all processes.")
        print("Acceptance demo: start with sharing off, observe Red-only belief, enable sharing, then watch Blue/Green receive Red's peer observation.")

        if not args.no_browser:
            webbrowser.open(dashboard_url)

        try:
            while all(process.poll() is None for process in processes):
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\nStopping simulation...")
        finally:
            dashboard.shutdown()
            dashboard.server_close()

        return 0
    finally:
        terminate_processes(processes)


if __name__ == "__main__":
    raise SystemExit(main())
