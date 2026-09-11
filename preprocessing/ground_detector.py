"""
Ground plane and terrain elevation extractor.
Implements multi-sector elevation sorting and iterative plane estimation.
"""
from typing import Tuple, Dict, Any
import numpy as np


class GroundDetector:
    """
    Separates ground points from obstacle points and estimates local terrain elevation.
    """

    def __init__(self, config: Dict[str, Any]):
        terrain_cfg = config.get("terrain", {})
        self.ground_tolerance = float(terrain_cfg.get("ground_elevation_tolerance", 0.15))
        self.num_radial_bins = 20
        self.num_angular_sectors = 16

    def segment_ground(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Splits point cloud into ground points and obstacle points.
        Returns:
            ground_mask: bool array (True for ground points)
            ground_elevation_grid: estimated surface elevation
            height_above_ground: array of delta Z for each point
        """
        N = len(points)
        if N == 0:
            return np.zeros(0, dtype=bool), np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)

        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        r = np.linalg.norm(points[:, :2], axis=1)
        theta = np.arctan2(y, x)

        # Discretize into polar bins
        r_bin = np.clip(np.floor(r / (100.0 / self.num_radial_bins)).astype(np.int32), 0, self.num_radial_bins - 1)
        theta_bin = np.clip(np.floor((theta + np.pi) / (2 * np.pi / self.num_angular_sectors)).astype(np.int32), 0, self.num_angular_sectors - 1)
        bin_ids = r_bin * self.num_angular_sectors + theta_bin

        # Find min-z in each polar bin as initial ground seed
        total_bins = self.num_radial_bins * self.num_angular_sectors
        min_z_per_bin = np.full(total_bins, np.inf, dtype=np.float32)
        np.minimum.at(min_z_per_bin, bin_ids, z)

        # Replace empty bins with global 5th percentile
        global_ground_z = np.percentile(z, 5) if len(z) > 0 else -1.73
        empty_mask = np.isinf(min_z_per_bin)
        min_z_per_bin[empty_mask] = global_ground_z

        # Local ground height for each point
        local_ground_z = min_z_per_bin[bin_ids]
        height_above_ground = z - local_ground_z

        ground_mask = (height_above_ground >= -0.15) & (height_above_ground <= self.ground_tolerance)

        return ground_mask, local_ground_z, height_above_ground
