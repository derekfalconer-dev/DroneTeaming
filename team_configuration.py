"""Validated immutable team configuration for the drone simulation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any


SUPPORTED_ROLES = frozenset({"isr", "payload", "generalist"})
SUPPORTED_CAPABILITIES = frozenset({"atr_observe", "isr_observe", "payload_drop"})
SUPPORTED_CAMERA_MOUNTS = frozenset({"fixed", "steerable"})
TEAM_CONFIG_FIELDS = frozenset({"role", "capabilities", "available", "payload_inventory", "camera_mount"})


class TeamConfigurationError(ValueError):
    """Raised when a supplied team configuration is invalid."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}

    for key, value in pairs:
        if key in result:
            raise TeamConfigurationError(f"Duplicate JSON key: {key}")

        result[key] = value

    return result


@dataclass(frozen=True)
class TeamMemberConfiguration:
    """One drone's role/capability/inventory advertisement."""

    role: str = "generalist"
    capabilities: tuple[str, ...] = ("atr_observe", "isr_observe")
    available: bool = True
    payload_inventory: MappingProxyType[str, int] = field(default_factory=lambda: MappingProxyType({}))
    camera_mount: str = "steerable"

    @classmethod
    def legacy_default(cls) -> "TeamMemberConfiguration":
        return cls()

    @classmethod
    def from_mapping(cls, value: dict[str, Any], *, drone_id: int | None = None) -> "TeamMemberConfiguration":
        unknown_fields = set(value) - TEAM_CONFIG_FIELDS
        prefix = f"Drone {drone_id}: " if drone_id is not None else ""

        if unknown_fields:
            raise TeamConfigurationError(f"{prefix}unknown configuration field(s): {', '.join(sorted(unknown_fields))}")

        missing_fields = TEAM_CONFIG_FIELDS - set(value)
        if missing_fields:
            raise TeamConfigurationError(f"{prefix}missing required configuration field(s): {', '.join(sorted(missing_fields))}")

        role = value["role"]
        if role not in SUPPORTED_ROLES:
            raise TeamConfigurationError(f"{prefix}invalid role: {role!r}")

        capabilities_value = value["capabilities"]
        if not isinstance(capabilities_value, list):
            raise TeamConfigurationError(f"{prefix}capabilities must be a list")

        capabilities: list[str] = []
        for capability in capabilities_value:
            if capability not in SUPPORTED_CAPABILITIES:
                raise TeamConfigurationError(f"{prefix}invalid capability: {capability!r}")
            if capability not in capabilities:
                capabilities.append(capability)

        available = value["available"]
        if not isinstance(available, bool):
            raise TeamConfigurationError(f"{prefix}available must be a boolean")

        inventory_value = value["payload_inventory"]
        if not isinstance(inventory_value, dict):
            raise TeamConfigurationError(f"{prefix}payload_inventory must be an object")

        inventory: dict[str, int] = {}
        for item_name, count in inventory_value.items():
            if not isinstance(item_name, str) or not item_name:
                raise TeamConfigurationError(f"{prefix}inventory item names must be non-empty strings")
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise TeamConfigurationError(f"{prefix}inventory count for {item_name!r} must be a non-negative integer")
            inventory[item_name] = count

        camera_mount = value["camera_mount"]
        if camera_mount not in SUPPORTED_CAMERA_MOUNTS:
            raise TeamConfigurationError(f"{prefix}invalid camera_mount: {camera_mount!r}")

        return cls(
            role=str(role),
            capabilities=tuple(capabilities),
            available=available,
            payload_inventory=MappingProxyType(dict(inventory)),
            camera_mount=str(camera_mount),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "capabilities": list(self.capabilities),
            "available": self.available,
            "payload_inventory": dict(self.payload_inventory),
            "camera_mount": self.camera_mount,
        }


@dataclass(frozen=True)
class TeamConfiguration:
    """Immutable three-drone team configuration."""

    members: MappingProxyType[int, TeamMemberConfiguration]

    @classmethod
    def legacy_defaults(cls, drone_ids: tuple[int, ...] = (1, 2, 3)) -> "TeamConfiguration":
        return cls(MappingProxyType({drone_id: TeamMemberConfiguration.legacy_default() for drone_id in drone_ids}))

    @classmethod
    def from_file(cls, path: str | Path, drone_ids: tuple[int, ...] = (1, 2, 3)) -> "TeamConfiguration":
        resolved_path = Path(path).expanduser().resolve()

        try:
            payload = json.loads(resolved_path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys)
        except json.JSONDecodeError as error:
            raise TeamConfigurationError(f"Invalid team configuration JSON in {resolved_path}: {error}") from error
        except OSError as error:
            raise TeamConfigurationError(f"Unable to read team configuration {resolved_path}: {error}") from error

        return cls.from_mapping(payload, drone_ids=drone_ids)

    @classmethod
    def from_mapping(cls, payload: dict[str, Any], drone_ids: tuple[int, ...] = (1, 2, 3)) -> "TeamConfiguration":
        if not isinstance(payload, dict):
            raise TeamConfigurationError("Team configuration root must be an object keyed by drone ID")

        valid_ids = set(drone_ids)
        members = {drone_id: TeamMemberConfiguration.legacy_default() for drone_id in drone_ids}

        for raw_drone_id, value in payload.items():
            try:
                drone_id = int(raw_drone_id)
            except (TypeError, ValueError) as error:
                raise TeamConfigurationError(f"Invalid drone ID key: {raw_drone_id!r}") from error

            if str(drone_id) != str(raw_drone_id):
                raise TeamConfigurationError(f"Drone ID keys must be canonical existing IDs: {raw_drone_id!r}")

            if drone_id not in valid_ids:
                raise TeamConfigurationError(f"Unknown drone ID in team configuration: {drone_id}")

            if not isinstance(value, dict):
                raise TeamConfigurationError(f"Drone {drone_id}: configuration must be an object")

            members[drone_id] = TeamMemberConfiguration.from_mapping(value, drone_id=drone_id)

        return cls(MappingProxyType(dict(members)))

    def for_drone(self, drone_id: int) -> TeamMemberConfiguration:
        return self.members[drone_id]

    def to_dict(self) -> dict[str, dict[str, Any]]:
        return {str(drone_id): member.to_dict() for drone_id, member in sorted(self.members.items())}


def load_team_configuration(path: str | Path | None, drone_ids: tuple[int, ...] = (1, 2, 3)) -> TeamConfiguration:
    if path is None:
        return TeamConfiguration.legacy_defaults(drone_ids)

    return TeamConfiguration.from_file(path, drone_ids)


def validate_advertised_team_member(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TeamConfigurationError("Advertised team configuration must be an object")

    return TeamMemberConfiguration.from_mapping(value).to_dict()
