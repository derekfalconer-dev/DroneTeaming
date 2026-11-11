import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import launch_simulation
from simulated_drone import (
    DRONE_PROFILES,
    PAYLOAD_PENDING_STATUS,
    PLAYBOOK_OBSERVE_ORBIT,
    TASK_TYPE_ISR_OBSERVE,
    TASK_TYPE_PAYLOAD_DROP,
    DroneSimulator,
    task_id_for_contact,
    task_id_for_priority_signature,
)
from team_configuration import TeamConfigurationError, load_team_configuration


class TeamConfigurationTest(unittest.TestCase):
    def test_defaults_round_trip_and_defensive_inventory_copy(self) -> None:
        config = load_team_configuration(None, tuple(sorted(DRONE_PROFILES)))
        payload = config.to_dict()

        self.assertEqual("generalist", payload["1"]["role"])
        self.assertIn("isr_observe", payload["1"]["capabilities"])
        payload["1"]["payload_inventory"]["water"] = 99
        self.assertEqual({}, config.to_dict()["1"]["payload_inventory"])

    def test_invalid_inputs_rejected(self) -> None:
        cases = [
            '{"4": {"role": "isr", "capabilities": ["isr_observe"], "available": true, "payload_inventory": {}, "camera_mount": "steerable"}}',
            '{"1": {"role": "pilot", "capabilities": ["isr_observe"], "available": true, "payload_inventory": {}, "camera_mount": "steerable"}}',
            '{"1": {"role": "isr", "capabilities": ["warp"], "available": true, "payload_inventory": {}, "camera_mount": "steerable"}}',
            '{"1": {"role": "isr", "capabilities": ["isr_observe"], "available": "yes", "payload_inventory": {}, "camera_mount": "steerable"}}',
            '{"1": {"role": "payload", "capabilities": ["payload_drop"], "available": true, "payload_inventory": {"water": -1}, "camera_mount": "fixed"}}',
            '{"1": {"role": "payload", "capabilities": ["payload_drop"], "available": true, "payload_inventory": {"water": 1.5}, "camera_mount": "fixed"}}',
            '{"1": {"role": "payload", "capabilities": ["payload_drop"], "available": true, "payload_inventory": {"water": true}, "camera_mount": "fixed"}}',
            '{"1": {"role": "payload", "capabilities": ["payload_drop"], "available": true, "payload_inventory": {"water": 1}, "camera_mount": "fixed", "extra": true}}',
            '{"1": {"role": "isr", "role": "payload", "capabilities": ["isr_observe"], "available": true, "payload_inventory": {}, "camera_mount": "steerable"}}',
        ]

        for content in cases:
            with self.subTest(content=content):
                with tempfile.TemporaryDirectory() as temp_dir:
                    path = Path(temp_dir) / "team.json"
                    path.write_text(content, encoding="utf-8")
                    with self.assertRaises(TeamConfigurationError):
                        load_team_configuration(path, tuple(sorted(DRONE_PROFILES)))

    def test_launcher_propagates_resolved_team_config_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "team.json"
            config_path.write_text(json.dumps({}), encoding="utf-8")

            with patch("launch_simulation.subprocess.Popen") as popen:
                launch_simulation.start_drone_processes(
                    "127.0.0.1",
                    (9101, 9102, 9103),
                    1,
                    0,
                    "report_continue",
                    "seeded_stationary_patrol",
                    1,
                    2,
                    "test-run",
                    config_path.resolve(),
                )

            for call in popen.call_args_list:
                command = call.args[0]
                self.assertIn("--team-config", command)
                self.assertIn(str(config_path.resolve()), command)

    def test_reset_preserves_configuration(self) -> None:
        config = load_team_configuration(None, tuple(sorted(DRONE_PROFILES))).members
        simulator = DroneSimulator(1, "Red", [], team_configuration=dict(config))
        before = simulator.status().team_configuration
        simulator.reset(scenario_seed=123)
        after = simulator.status().team_configuration
        self.assertEqual(before, after)


class TypedTaskFoundationTest(unittest.TestCase):
    def test_typed_ids_are_distinct_and_isr_id_is_backward_compatible(self) -> None:
        self.assertEqual(
            task_id_for_priority_signature(7, "contact-a"),
            task_id_for_contact(7, "contact-a", TASK_TYPE_ISR_OBSERVE),
        )
        self.assertNotEqual(
            task_id_for_contact(7, "contact-a", TASK_TYPE_ISR_OBSERVE),
            task_id_for_contact(7, "contact-a", TASK_TYPE_PAYLOAD_DROP),
        )

    def test_two_task_types_share_contact_without_shared_cancellation(self) -> None:
        simulator = DroneSimulator(1, "Red", [])
        payload_task = simulator.create_dormant_payload_task_for_contact("contact-a", "water", 1)
        isr_task_id = task_id_for_contact(simulator.scenario_seed, "contact-a", TASK_TYPE_ISR_OBSERVE)
        simulator.response_tasks[isr_task_id] = {
            **payload_task,
            "task_id": isr_task_id,
            "task_type": TASK_TYPE_ISR_OBSERVE,
            "status": "queued",
            "requested_payload_type": "",
            "requested_payload_quantity": 0,
        }

        simulator.cancel_response(isr_task_id)

        self.assertEqual("cancelled", simulator.response_tasks[isr_task_id]["status"])
        self.assertEqual(PAYLOAD_PENDING_STATUS, simulator.response_tasks[payload_task["task_id"]]["status"])
        self.assertEqual("contact-a", simulator.response_tasks[payload_task["task_id"]]["contact_ref"])

    def test_payload_task_remains_dormant_and_not_active_guidance(self) -> None:
        simulator = DroneSimulator(3, "Green", [])
        task = simulator.create_dormant_payload_task_for_contact("contact-a", "water", 1)

        self.assertEqual(PAYLOAD_PENDING_STATUS, task["status"])
        self.assertIsNone(simulator.active_response_task_id)
        self.assertEqual("patrol", simulator.status().current_mode)

    def test_actual_outgoing_task_messages_include_typed_identity_advertisement_and_payload_metadata(self) -> None:
        simulator = DroneSimulator(3, "Green", [])
        task = simulator.create_dormant_payload_task_for_contact("contact-a", "water", 2)
        simulator.cancel_response(task["task_id"])
        messages = simulator.pop_outbound_claims()

        self.assertEqual(1, len(messages))
        message = messages[0]
        self.assertEqual(TASK_TYPE_PAYLOAD_DROP, message["task_type"])
        self.assertEqual("contact-a", message["contact_ref"])
        self.assertEqual(task["task_id"], message["task_id"])
        self.assertEqual("cancelled", message["task_state"])
        self.assertEqual("water", message["requested_payload_type"])
        self.assertEqual(2, message["requested_payload_quantity"])
        self.assertTrue(message["release_authorization_required"])
        self.assertIn("claimant_team_configuration", message)
        self.assertIn("capabilities", message["claimant_team_configuration"])

    def test_peer_payload_cancellation_preserves_isr_task_for_same_contact(self) -> None:
        owner = DroneSimulator(1, "Red", [])
        peer = DroneSimulator(2, "Blue", [])
        owner_payload_task = owner.create_dormant_payload_task_for_contact("contact-a", "water", 1)
        peer_payload_task = peer.create_dormant_payload_task_for_contact("contact-a", "water", 1)
        peer_isr_task_id = task_id_for_contact(peer.scenario_seed, "contact-a", TASK_TYPE_ISR_OBSERVE)
        peer.response_tasks[peer_isr_task_id] = {
            **peer_payload_task,
            "task_id": peer_isr_task_id,
            "task_type": TASK_TYPE_ISR_OBSERVE,
            "status": "queued",
            "requested_payload_type": "",
            "requested_payload_quantity": 0,
        }

        owner.cancel_response(owner_payload_task["task_id"])
        message = owner.pop_outbound_claims()[0]

        self.assertTrue(peer.receive_claim(message))
        self.assertEqual("cancelled", peer.response_tasks[peer_payload_task["task_id"]]["status"])
        self.assertEqual("queued", peer.response_tasks[peer_isr_task_id]["status"])
        self.assertEqual("contact-a", peer.response_tasks[peer_isr_task_id]["contact_ref"])

    def test_conflicting_task_metadata_rejected_without_mutation(self) -> None:
        simulator = DroneSimulator(1, "Red", [])
        task = simulator.create_dormant_payload_task_for_contact("contact-a", "water", 1)
        before = dict(simulator.response_tasks[task["task_id"]])
        payload = {
            "scenario_run_id": str(simulator.scenario_seed),
            "task_id": task["task_id"],
            "task_type": TASK_TYPE_ISR_OBSERVE,
            "contact_ref": "contact-a",
            "signature_id": "contact-a",
            "claimant_drone_id": 2,
            "claimant_drone_name": "Blue",
            "estimated_east_m": 0.0,
            "estimated_north_m": 0.0,
            "task_state": "claiming",
            "revision": 0,
            "assignment_epoch": 0,
            "playbook_config_version": 0,
            "claimant_team_configuration": simulator.team_configuration.to_dict(),
        }

        self.assertFalse(simulator.receive_claim(payload))
        self.assertEqual(before, simulator.response_tasks[task["task_id"]])


class EligibilityFoundationTest(unittest.TestCase):
    def test_incapable_drone_does_not_self_claim_isr_work(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "team.json"
            path.write_text(json.dumps({
                "1": {
                    "role": "payload",
                    "capabilities": ["atr_observe", "payload_drop"],
                    "available": True,
                    "payload_inventory": {"water": 1},
                    "camera_mount": "fixed",
                },
            }), encoding="utf-8")
            config = load_team_configuration(path, tuple(sorted(DRONE_PROFILES))).members

        simulator = DroneSimulator(1, "Red", [], team_configuration=dict(config))
        task = simulator.create_dormant_payload_task_for_contact("contact-a", "water", 1)
        isr_task_id = task_id_for_contact(simulator.scenario_seed, "contact-b", TASK_TYPE_ISR_OBSERVE)
        simulator.response_tasks[isr_task_id] = {
            **task,
            "task_id": isr_task_id,
            "task_type": TASK_TYPE_ISR_OBSERVE,
            "contact_ref": "contact-b",
            "signature_id": "contact-b",
            "status": "queued",
        }
        simulator._make_task_allocatable_locked(isr_task_id, simulator.response_tasks[isr_task_id])

        self.assertEqual([], simulator.pop_outbound_claims())
        self.assertIsNone(simulator.active_response_task_id)

    def test_payload_only_and_unavailable_drones_do_not_compete_for_peer_isr_claims(self) -> None:
        cases = [
            {
                "role": "payload",
                "capabilities": ["atr_observe", "payload_drop"],
                "available": True,
                "payload_inventory": {"water": 1},
                "camera_mount": "fixed",
            },
            {
                "role": "isr",
                "capabilities": ["isr_observe"],
                "available": False,
                "payload_inventory": {},
                "camera_mount": "steerable",
            },
        ]
        contact_ref = "contact-peer-isr"

        for member_config in cases:
            with self.subTest(member_config=member_config):
                with tempfile.TemporaryDirectory() as temp_dir:
                    path = Path(temp_dir) / "team.json"
                    path.write_text(json.dumps({"1": member_config}), encoding="utf-8")
                    config = load_team_configuration(path, tuple(sorted(DRONE_PROFILES))).members

                simulator = DroneSimulator(1, "Red", [], team_configuration=dict(config))
                simulator.set_playbook(PLAYBOOK_OBSERVE_ORBIT, 1)
                task_id = task_id_for_contact(simulator.scenario_seed, contact_ref, TASK_TYPE_ISR_OBSERVE)
                payload = {
                    "scenario_run_id": str(simulator.scenario_seed),
                    "task_id": task_id,
                    "signature_id": contact_ref,
                    "claimant_drone_id": 2,
                    "claimant_drone_name": "Blue",
                    "estimated_east_m": 0.0,
                    "estimated_north_m": 0.0,
                    "task_state": "claiming",
                    "revision": 0,
                    "assignment_epoch": 0,
                    "playbook_config_version": simulator.playbook_config_version,
                }

                self.assertTrue(simulator.receive_claim(payload))
                self.assertIsNone(simulator.active_response_task_id)
                self.assertEqual([], simulator.pop_outbound_claims())
                self.assertIsNone(simulator.response_tasks[task_id]["assigned_drone_id"])
                self.assertEqual("patrol", simulator.status().current_mode)

    def test_payload_eligibility_requires_inventory_but_does_not_consume(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "team.json"
            path.write_text(json.dumps({
                "3": {
                    "role": "payload",
                    "capabilities": ["payload_drop"],
                    "available": True,
                    "payload_inventory": {"water": 2},
                    "camera_mount": "fixed",
                },
            }), encoding="utf-8")
            config = load_team_configuration(path, tuple(sorted(DRONE_PROFILES))).members

        simulator = DroneSimulator(3, "Green", [], team_configuration=dict(config))
        task = simulator.create_dormant_payload_task_for_contact("contact-a", "water", 2)

        self.assertTrue(simulator._task_is_executable_by_this_node_locked(task))
        self.assertEqual({"water": 2}, simulator.team_configuration.to_dict()["payload_inventory"])
        self.assertEqual(PAYLOAD_PENDING_STATUS, simulator.response_tasks[task["task_id"]]["status"])


if __name__ == "__main__":
    unittest.main()
