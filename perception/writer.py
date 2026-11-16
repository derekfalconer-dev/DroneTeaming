"""JSONL writer for image observations."""

from __future__ import annotations

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
