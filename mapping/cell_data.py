"""
2.5D Cell data structures and aggregation strategies.
"""
from dataclasses import dataclass
from typing import Optional, List, Dict
import numpy as np


@dataclass
class AdaptiveCell:
    """
    Represents an adaptive variable-resolution 2.5D spatial cell.
    """
    zone_id: int
    col: int
    row: int
    x_center: float
    y_center: float
    resolution: float
    min_z: float
    max_z: float
    mean_z: float
    rep_z: float               # Representative elevation based on policy
    point_count: int
    semantic_class: int
    confidence: float
    is_drivable: bool
    is_dynamic: bool
    timestamp: float
    roughness: float = 0.0     # Elevation standard deviation

    def to_dict(self) -> Dict:
        return {
            "zone_id": int(self.zone_id),
            "col": int(self.col),
            "row": int(self.row),
            "x": float(round(self.x_center, 3)),
            "y": float(round(self.y_center, 3)),
            "res": float(round(self.resolution, 3)),
            "min_z": float(round(self.min_z, 3)),
            "max_z": float(round(self.max_z, 3)),
            "mean_z": float(round(self.mean_z, 3)),
            "rep_z": float(round(self.rep_z, 3)),
            "count": int(self.point_count),
            "sem": int(self.semantic_class),
            "conf": float(round(self.confidence, 2)),
            "drivable": bool(self.is_drivable),
            "dynamic": bool(self.is_dynamic),
            "time": float(round(self.timestamp, 3)),
            "roughness": float(round(self.roughness, 3)),
        }
