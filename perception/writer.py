"""JSONL writers for perception outputs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import TextIO

from perception.observations import ImageObservation, serialize_observation_jsonl


class JsonlObservationWriter:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._handle: TextIO | None = None

    def open(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8", buffering=1)

    def write_observation(self, observation: ImageObservation) -> None:
        if self._handle is None:
            self.open()

        assert self._handle is not None
        self._handle.write(serialize_observation_jsonl(observation))

    def write_observations(self, observations: list[ImageObservation]) -> None:
        for observation in observations:
            self.write_observation(observation)

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> "JsonlObservationWriter":
        self.open()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


class JsonlContactEventWriter:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._handle: TextIO | None = None

    def open(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8", buffering=1)

    def write_events(self, events: list[Mapping[str, object]]) -> None:
        if self._handle is None:
            self.open()

        assert self._handle is not None
        for event in events:
            self._handle.write(json.dumps(dict(event), sort_keys=True, separators=(",", ":")))
            self._handle.write("\n")

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> "JsonlContactEventWriter":
        self.open()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
