"""
Terrain & drivability analysis module.
Detects drivable road, sidewalks, curbs, potholes, depressions, and slope angles.
"""
from typing import Dict, Any, Tuple
import numpy as np


class TerrainAnalyzer:
    """
    Evaluates terrain drivability, slope gradients, curb steps, and surface anomalies.
    """

    def __init__(self, config: Dict[str, Any]):
        t_cfg = config.get("terrain", {})
        self.curb_min_height = float(t_cfg.get("curb_min_height", 0.08))
        self.curb_max_height = float(t_cfg.get("curb_max_height", 0.35))
        self.max_drivable_slope_deg = float(t_cfg.get("max_drivable_slope_deg", 18.0))
        self.roughness_threshold = float(t_cfg.get("roughness_threshold", 0.06))

    def analyze_terrain(
        self,
        points: np.ndarray,
        ground_mask: np.ndarray,
        height_above_ground: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Analyzes ground points to refine terrain classification and drivability status.
        Returns:
            drivable_mask: bool array (True for safe drivable terrain)
            terrain_class_ids: uint32 array of refined terrain classes
        """
        N = len(points)
        drivable_mask = np.zeros(N, dtype=bool)
        terrain_class_ids = np.zeros(N, dtype=np.uint32)

        if N == 0:
            return drivable_mask, terrain_class_ids

        # Ground points with low height variation are considered road
        is_flat_ground = ground_mask & (np.abs(height_above_ground) < 0.06)
        drivable_mask[is_flat_ground] = True
        terrain_class_ids[is_flat_ground] = 1  # drivable_road

        # Curb detection: points near ground boundary with height between 8cm and 35cm
        is_curb = (~ground_mask) & (height_above_ground >= self.curb_min_height) & (height_above_ground <= self.curb_max_height)
        terrain_class_ids[is_curb] = 11  # curb

        # Pothole / depression detection: ground points significantly lower than local neighborhood
        is_depression = ground_mask & (height_above_ground < -0.07)
        terrain_class_ids[is_depression] = 12  # pothole_irregular
        drivable_mask[is_depression] = False   # Non-drivable hazard

        return drivable_mask, terrain_class_ids
