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
APPLICATION_BUILD = "moving-search-noisy-fusion-2026-11-06"
SCENARIO_NAME = "Seeded noisy overlapping moving-search patrol"
DEFAULT_SCENARIO_SEED = 20261106
DRONE_SPECS = (
    {"id": 1, "name": "Red", "color": "#ef4444"},
    {"id": 2, "name": "Blue", "color": "#3b82f6"},
    {"id": 3, "name": "Green", "color": "#22c55e"},
)


DASHBOARD_HTML = """<!doctype html>
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
      height: auto;
      width: 100%;
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
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 1rem;
      margin-top: 1rem;
    }

    .map-card.truth {
      grid-column: 1 / -1;
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
    <button id="sharing-button" type="button">Enable sharing</button>
    <button id="pause-button" type="button">Pause</button>
    <button id="reset-button" type="button">Restart scenario</button>
    <button id="new-scenario-button" type="button">New random scenario</button>
    <label class="pill"><input id="trail-toggle" type="checkbox" checked> Show flight trails</label>
    <label class="pill"><input id="measurement-toggle" type="checkbox" checked> Sensor measurements</label>
    <label class="pill"><input id="ellipse-toggle" type="checkbox" checked> Uncertainty ellipses</label>
    <span id="control-state" class="pill">Sharing off · running</span>
    <span id="scenario-state" class="pill">Scenario loading…</span>
  </section>

  <section class="map-grid">
    <article class="map-card belief">
      <div class="map-title"><span>Red belief</span><span class="pill">actual local map</span></div>
      <canvas id="belief-map-1" width="520" height="360"></canvas>
      <div id="belief-details-1" class="details"></div>
      <details>
        <summary>Diagnostics</summary>
        <div id="diagnostics-1"></div>
      </details>
    </article>

    <article class="map-card belief">
      <div class="map-title"><span>Blue belief</span><span class="pill">actual local map</span></div>
      <canvas id="belief-map-2" width="520" height="360"></canvas>
      <div id="belief-details-2" class="details"></div>
      <details>
        <summary>Diagnostics</summary>
        <div id="diagnostics-2"></div>
      </details>
    </article>

    <article class="map-card belief">
      <div class="map-title"><span>Green belief</span><span class="pill">actual local map</span></div>
      <canvas id="belief-map-3" width="520" height="360"></canvas>
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
      <canvas id="truth-map" width="1100" height="520"></canvas>
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
      showFlightTrails: true,
      showMeasurements: true,
      showEllipses: true,
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

    function drawBaseMap(canvas, bounds) {
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

      for (const object of state.truth.objects) {
        const position = project(object, bounds, canvas);
        ctx.fillStyle = "#f8fafc";
        ctx.strokeStyle = "#0f172a";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(position.x, position.y, 7, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
        ctx.fillStyle = "#f8fafc";
        ctx.font = "12px sans-serif";
        ctx.fillText(object.label, position.x + 10, position.y + 4);
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

        ctx.fillStyle = "#e2e8f0";
        ctx.font = "12px sans-serif";
        ctx.fillText(object.label, position.x + 12, position.y + 4);
      }
    }

    function renderTruthDetails() {
      const details = document.getElementById("truth-details");
      details.innerHTML = `
        <strong>${state.truth.objects.length} persistent ground-truth objects · ${state.scenarioName} · seed ${state.scenarioSeed} · build ${state.appBuild}</strong>
        <ul>
          ${state.truth.objects.map((object) => `<li>${object.label}</li>`).join("")}
        </ul>
      `;
    }

    function renderBeliefDetails(drone) {
      const details = document.getElementById(`belief-details-${drone.id}`);
      const belief = state.beliefs.get(drone.id);
      const objects = belief ? belief.objects : [];
      const localCount = objects.filter((object) => object.own_sensor).length;
      const teammateOnlyCount = objects.filter((object) => object.teammate_sourced && !object.own_sensor).length;
      const multiSourceCount = objects.filter((object) => (object.sources || []).length > 1).length;
      const staleCount = objects.filter((object) => object.stale).length;
      const unresolvedCount = belief ? belief.unresolved_observation_count || 0 : 0;

      const counts = `
        <div class="count-row">
          <div class="count-box"><strong>${localCount}</strong> local</div>
          <div class="count-box"><strong>${teammateOnlyCount}</strong> teammate-only</div>
          <div class="count-box"><strong>${multiSourceCount}</strong> multi-source</div>
          <div class="count-box"><strong>${staleCount}</strong> stale · ${unresolvedCount} unresolved</div>
        </div>
      `;

      if (!belief || objects.length === 0) {
        details.innerHTML = `${counts}<div class="empty-state">This node's independently maintained map has no objects.</div>`;
        return;
      }

      details.innerHTML = `
        ${counts}
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
            ${objects.map((object) => {
              const age = Math.max(0, belief.simulation_time - object.last_seen);
              const sourceType = object.own_sensor && object.teammate_sourced
                ? "multi-source"
                : object.own_sensor
                  ? "own"
                  : "remote";
              return `
                <tr>
                  <td>${object.local_object_id || object.object_id}</td>
                  <td>${object.label}</td>
                  <td>${object.source_names.join(", ")}</td>
                  <td>${sourceType}</td>
                  <td>${object.stale ? "stale" : "fresh"} · ${age.toFixed(1)} s · ${object.active_contribution_count || 0} active</td>
                </tr>
              `;
            }).join("")}
          </tbody>
        </table>
      `;
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

    function renderControls() {
      document.getElementById("sharing-button").textContent = state.sharingEnabled ? "Disable sharing" : "Enable sharing";
      document.getElementById("pause-button").textContent = state.paused ? "Resume" : "Pause";
      document.getElementById("control-state").textContent = `${state.sharingEnabled ? "Sharing on" : "Sharing off"} · ${state.paused ? "paused" : "running"}`;
      document.getElementById("scenario-state").textContent = `${state.scenarioName} · seed ${state.scenarioSeed} · build ${state.appBuild}`;
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
      renderMaps();
    }

    document.getElementById("sharing-button").addEventListener("click", () => {
      postControl({ sharing_enabled: !state.sharingEnabled });
    });

    document.getElementById("pause-button").addEventListener("click", () => {
      postControl({ paused: !state.paused });
    });

    document.getElementById("reset-button").addEventListener("click", () => {
      postControl({ restart_scenario: true });
    });

    document.getElementById("new-scenario-button").addEventListener("click", () => {
      const scenarioSeed = Math.floor(Date.now() % 1000000000);
      postControl({ scenario_seed: scenarioSeed });
    });

    document.getElementById("trail-toggle").addEventListener("change", (event) => {
      state.showFlightTrails = event.target.checked;
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

    drones.forEach(createCard);
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

    processes = start_drone_processes(args.drone_host, drone_ports, DEFAULT_SCENARIO_SEED)

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
