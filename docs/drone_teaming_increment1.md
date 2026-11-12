# DroneTeaming increment 1: capabilities and typed-task foundation

This increment keeps the default three-drone localhost simulation unchanged unless a team configuration file is supplied. The configuration is loaded once at startup, validated before child drone processes are spawned, passed to each child, exposed read-only in `/status`, and preserved across scenario resets/reseeds.

Example launch:

```bash
python launch_simulation.py --team-config examples/team_config_payload_foundation.json
```

The supplied example configures Red and Blue as ISR drones and Green as a payload drone with generic `water` and `medical_supplies` inventory.

Supported fields per drone are:

- `role`: `isr` or `payload`
- `capabilities`: `atr_observe`, `isr_observe`, and/or `payload_drop`
- `available`: boolean configured operational availability
- `payload_inventory`: object of generic payload names to non-negative integer counts
- `camera_mount`: `fixed` or `steerable`

Role labels describe the primary aircraft setup; task eligibility is still driven by capabilities. ISR execution eligibility requires configured availability, the `isr_observe` capability, and an available active-flight slot. Payload eligibility requires configured availability, `payload_drop`, and positive inventory of the requested payload type. Legacy `generalist` configs are normalized only when their capabilities are unambiguous.

Typed tasks now carry `task_type` and `contact_ref`. Existing priority observe/orbit work is `isr_observe`; its `contact_ref` is the mock priority signature evidence reference. Local map IDs remain node-local and are not used as common contact identity.

Limitations intentionally remaining after this increment:

- No delivery workflow, authorization endpoint, inventory reservation/decrement, actuator, release physics, terminal guidance, or intercept planner.
- Payload tasks stay `pending_workflow` and never enter ISR election, orbit guidance, response camera control, or active-flight ownership.
- Camera/compute differences are advertised only. The simulation still uses the existing camera footprint and sensor model; the fixed-camera label does not validate payload-camera guidance.
- There is no heartbeat, service discovery, centralized scheduler, or dashboard allocator.
