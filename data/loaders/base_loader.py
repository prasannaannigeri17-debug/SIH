"""
Base interface for point cloud data providers.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np


@dataclass
class PointCloudFrame:
    """Represents a single LiDAR scan frame."""
    frame_id: int
    timestamp: float
    points: np.ndarray             # (N, 3) float32 [x, y, z] in sensor frame
    intensities: Optional[np.ndarray] = None  # (N,) float32
    semantic_labels: Optional[np.ndarray] = None  # (N,) uint32 / int32
    instance_labels: Optional[np.ndarray] = None  # (N,) uint32 / int32
    ego_pose: Optional[np.ndarray] = None         # (4, 4) transformation matrix
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.points is not None and not isinstance(self.points, np.ndarray):
            self.points = np.asarray(self.points, dtype=np.float32)
        if self.intensities is None and self.points is not None:
            self.intensities = np.ones(len(self.points), dtype=np.float32) * 0.5


class BasePointCloudLoader(ABC):
    """Abstract base class for all point cloud stream providers."""

    @abstractmethod
    def __len__(self) -> int:
        """Total number of frames available."""
        pass

    @abstractmethod
    def get_frame(self, index: int) -> PointCloudFrame:
        """Retrieve a specific frame by index."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset sequence iterator to beginning."""
        pass
