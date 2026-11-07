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
      height: auto;
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
      min-width: 0;
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
  </style>
</head>
<body>
  <h1>Drone Simulation Dashboard</h1>
  <p>Three independent moving drones maintain node-local belief maps, sense only within their moving camera footprints, and exchange original observations over peer TCP transport.</p>

  <section class="controls">
    <button id="pause-button" type="button">Pause</button>
    <button id="reset-button" type="button">Restart scenario</button>
    <button id="new-scenario-button" type="button">New random scenario</button>
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

  <section class="card">
    <div class="map-title"><span>Belief display controls</span><span class="pill">display only</span></div>
    <label class="pill"><input id="measurement-toggle" type="checkbox" checked> Sensor measurements</label>
    <label class="pill"><input id="ellipse-toggle" type="checkbox" checked> Uncertainty ellipses</label>
    <label class="pill"><input id="priority-only-toggle" type="checkbox"> Priority only</label>
    <label class="pill"><input id="belief-label-toggle" type="checkbox"> Belief object labels</label>
    <span class="control-note">These filters affect only dashboard rendering; they do not change sensing, fusion, or allocation.</span>
  </section>

  <section class="card">
    <div class="map-title"><span>Priority response assignments</span><span class="pill">connected-peer claim state</span></div>
    <div id="assignment-details" class="details"></div>
  </section>

  <section class="map-grid">
    <article class="map-card belief">
      <div class="map-title"><span>Red belief</span><span class="pill">actual local map</span></div>
      <canvas id="belief-map-1" class="belief-canvas" width="780" height="420"></canvas>
      <div id="belief-details-1" class="details"></div>
      <details>
        <summary>Diagnostics</summary>
        <div id="diagnostics-1"></div>
      </details>
    </article>

    <article class="map-card belief">
      <div class="map-title"><span>Blue belief</span><span class="pill">actual local map</span></div>
      <canvas id="belief-map-2" class="belief-canvas" width="780" height="420"></canvas>
      <div id="belief-details-2" class="details"></div>
      <details>
        <summary>Diagnostics</summary>
        <div id="diagnostics-2"></div>
      </details>
    </article>

    <article class="map-card belief">
      <div class="map-title"><span>Green belief</span><span class="pill">actual local map</span></div>
      <canvas id="belief-map-3" class="belief-canvas" width="780" height="420"></canvas>
      <div id="belief-details-3" class="details"></div>
      <details>
        <summary>Diagnostics</summary>
        <div id="diagnostics-3"></div>
      </details>
    </article>

    <article class="map-card truth">
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
      <div class="control-panel">
        <h3>Scenario configuration</h3>
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
        <span class="control-note">Applies on the next Restart scenario or New random scenario. Changing this does not reset the current run.</span>
      </div>
      <canvas id="truth-map" class="truth-canvas" width="1100" height="520"></canvas>
      <div id="truth-details" class="details"></div>
      <div class="legend">Truth objects are rendered only on this map. Belief maps above are fetched from each drone process and are not populated from truth.</div>
    </article>
  </section>

  <h2>Flight telemetry</h2>
  <main id="drone-grid" class="grid"></main>

  <script>
    const drones = __DRONES_JSON__;
    const grid = document.getElementById("drone-grid");
    const state = {
      statuses: new Map(),
      beliefs: new Map(),
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
        const [status, belief] = await Promise.all([
          fetchJson(`${drone.url}/status`),
          fetchJson(`${drone.url}/objects`),
        ]);

        state.statuses.set(drone.id, status);
        state.beliefs.set(drone.id, belief);
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
      state.playbookConfigVersion = Number(state.truth.playbook_config_version || state.playbookConfigVersion || 0);
      state.requestedPriorityCount = state.truth.requested_priority_count || 0;
      state.priorityObjectCount = state.truth.priority_object_count || 0;
      state.priorityDiscoveredCount = state.truth.priority_discovered_count || 0;
      state.orbitRadiusM = state.truth.orbit_radius_m || 0;

      const runKey = `${state.scenarioSeed}:${state.selectedPlaybook}:${state.requestedPriorityCount}`;

      if (state.currentRunKey && state.currentRunKey !== runKey) {
        state.taskSnapshotsByDrone.clear();
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

    function objectDetailHtml(target) {
      if (!target) {
        return "";
      }

      const object = target.object;
      const sources = object.source_names && object.source_names.length ? object.source_names.join(", ") : "—";
      const age = target.ageSeconds === null || target.ageSeconds === undefined ? "—" : `${target.ageSeconds.toFixed(1)} s`;
      const priority = object.priority ? "yes" : "no";

      return `
        <div class="control-panel">
          <h3>Selected object</h3>
          <dl class="diagnostics-grid">
            <dt>Class</dt><dd>${escapeHtml(priorityClassLabel(object))}</dd>
            <dt>Priority</dt><dd>${priority}</dd>
            <dt>Observing sources</dt><dd>${escapeHtml(sources)}</dd>
            <dt>Age</dt><dd>${escapeHtml(age)}</dd>
          </dl>
        </div>
      `;
    }

    function activeMapDetailFor(kind, droneId = null) {
      const selected = state.selectedMapObject;
      const hovered = state.hoveredMapObject;
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
    }

    function resizeCanvasBackingStore(canvas) {
      const rect = canvas.getBoundingClientRect ? canvas.getBoundingClientRect() : null;
      const cssWidth = rect && rect.width ? rect.width : canvas.width;
      const cssHeight = rect && rect.height ? rect.height : canvas.height;
      const pixelRatio = window.devicePixelRatio || 1;
      const backingWidth = Math.max(1, Math.round(cssWidth * pixelRatio));
      const backingHeight = Math.max(1, Math.round(cssHeight * pixelRatio));

      if (canvas.width !== backingWidth || canvas.height !== backingHeight) {
        canvas.width = backingWidth;
        canvas.height = backingHeight;
      }
    }

    function drawBaseMap(canvas, bounds) {
      resizeCanvasBackingStore(canvas);
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

    function drawFootprintsAndDrones(canvas, bounds) {
      const ctx = canvas.getContext("2d");

      for (const drone of drones) {
        const status = state.statuses.get(drone.id);

        if (!status) {
          continue;
        }

        const footprint = status.camera_footprint.map((point) => project(point, bounds, canvas));
        ctx.beginPath();
        ctx.moveTo(footprint[0].x, footprint[0].y);
        ctx.lineTo(footprint[1].x, footprint[1].y);
        ctx.lineTo(footprint[2].x, footprint[2].y);
        ctx.closePath();
        ctx.fillStyle = `${drone.color}26`;
        ctx.strokeStyle = `${drone.color}bb`;
        ctx.lineWidth = 2;
        ctx.fill();
        ctx.stroke();

        const position = project(status, bounds, canvas);
        const heading = (status.heading_deg - 90) * (Math.PI / 180);
        ctx.save();
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

        ctx.fillStyle = "#e2e8f0";
        ctx.font = "12px sans-serif";
        ctx.fillText(drone.name, position.x + 10, position.y - 10);
      }
    }

    function drawTruth(canvas, bounds) {
      const ctx = canvas.getContext("2d");
      const labelRects = [];

      for (const object of state.truth.objects) {
        if ((!state.showBackgroundTruthObjects && !object.priority) || (state.priorityOnly && !object.priority)) {
          continue;
        }
        const position = project(object, bounds, canvas);
        if (object.priority) {
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
        ${objectDetailHtml(activeDetail)}
        <details class="detected-objects"${state.beliefObjectTablesOpen.get(drone.id) ? " open" : ""}>
          <summary>Detected objects (${visibleObjects.length})</summary>
          <table class="object-table">
          <thead>
            <tr>
              <th>Local object ID</th>
              <th>Class</th>
              <th>Observing sources</th>
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
      drawBaseMap(truthCanvas, bounds);
      drawFlightTrails(truthCanvas, bounds);
      drawFootprintsAndDrones(truthCanvas, bounds);
      drawTruth(truthCanvas, bounds);
      renderTruthDetails();

      for (const drone of drones) {
        const canvas = document.getElementById(`belief-map-${drone.id}`);
        drawBaseMap(canvas, bounds);
        drawFootprintsAndDrones(canvas, bounds);
        drawBeliefObjects(canvas, bounds, drone.id);
        renderBeliefDetails(drone);
        renderDiagnostics(drone);
      }
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
        assignmentRevision(left.task),
        assignmentEpoch(left.task),
        reportAuthority(left),
        taskUpdateTime(left.task),
      ];
      const rightKey = [
        assignmentRevision(right.task),
        assignmentEpoch(right.task),
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
          } else if (freshOwnerReport) {
            selectedReport = freshOwnerReport;
            sourceLabel = `owner ${snapshotTaskReportAge(freshOwnerReport.snapshot)} · not active on owner`;
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
            assignedDroneName: assignedDroneId === null ? "Unassigned" : assignedDroneNameFor(assignedReport.task),
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
      }

      document.getElementById("sharing-button").textContent = state.sharingEnabled ? "Disable sharing" : "Enable sharing";
      document.getElementById("pause-button").textContent = state.paused ? "Resume" : "Pause";
      document.getElementById("control-state").textContent = `${state.sharingEnabled ? "Sharing on" : "Sharing off"} · ${state.paused ? "paused" : "running"}`;
      document.getElementById("scenario-state").textContent = `${state.scenarioName} · seed ${state.scenarioSeed} · priority ${state.priorityObjectCount} · playbook ${state.selectedPlaybook} · build ${state.appBuild}`;
      document.getElementById("task-state").textContent = `${teamSummary.orbiting} orbiting · ${teamSummary.transitioning} transitioning · ${teamSummary.searching} searching · ${teamSummary.queued} queued`;
      document.getElementById("playbook-config-state").textContent = pendingConfig && !applied
        ? `Playbook ${pendingConfig.playbook} v${pendingConfig.version} pending · ${acknowledgedVersions.join(" · ")}`
        : `Playbook applied · ${acknowledgedVersions.join(" · ")}`;
    }

    async function postControl(payload, targetDrones = drones) {
      await Promise.all(targetDrones.map((drone) => fetchJson(`${drone.url}/control`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      })));
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
      renderMaps();
    }

    document.getElementById("sharing-button").addEventListener("click", () => {
      postControl({ sharing_enabled: !state.sharingEnabled });
    });

    document.getElementById("pause-button").addEventListener("click", () => {
      postControl({ paused: !state.paused });
    });

    function selectedRunOptions() {
      return {
        priority_count: Number(document.getElementById("priority-count-select").value),
        playbook: document.getElementById("playbook-select").value,
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
      postControl({ restart_scenario: true, ...selectedRunOptions() });
    });

    document.getElementById("new-scenario-button").addEventListener("click", () => {
      const scenarioSeed = Math.floor(Date.now() % 1000000000);
      postControl({ scenario_seed: scenarioSeed, ...selectedRunOptions() });
    });

    document.getElementById("playbook-select").addEventListener("change", (event) => {
      applyPlaybookConfiguration(event.target.value);
    });

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
          refreshMapDetailPanels();
        });
      }
    }

    drones.forEach(createCard);
    setupMapInteractions();
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
    processes = start_drone_processes(
        args.drone_host,
        drone_ports,
        DEFAULT_SCENARIO_SEED,
        args.priority_count,
        playbook,
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
