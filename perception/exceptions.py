"""Perception-specific exceptions with actionable user-facing messages."""

from __future__ import annotations


class PerceptionError(RuntimeError):
    """Base class for standalone camera perception errors."""


class MissingDependencyError(PerceptionError):
    """Raised when optional perception dependencies are not installed."""


class CameraUnavailableError(PerceptionError):
    """Raised when a configured camera source cannot be opened."""


class CaptureReadError(PerceptionError):
    """Raised when frame capture fails after a camera was opened."""


class ModelLoadError(PerceptionError):
    """Raised when the detector model cannot be loaded."""


class UnlocatedObservationError(PerceptionError):
    """Raised when an image-only observation is sent to a geospatial boundary."""
