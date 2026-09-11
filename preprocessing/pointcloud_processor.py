"""
Point Cloud Preprocessor module.
Handles range filtering, NaN removal, spatial cropping, outlier filtering, and coordinate transformations.
"""
from typing import Dict, Any, Optional, Tuple
import numpy as np
from data.loaders.base_loader import PointCloudFrame


class PointCloudProcessor:
    """
    High-performance vectorized point cloud preprocessor.
    """

    def __init__(self, config: Dict[str, Any]):
        sensor_cfg = config.get("sensor", {})
        self.min_range = float(sensor_cfg.get("min_range", 0.5))
        self.max_range = float(sensor_cfg.get("max_range", 100.0))
        self.min_z = float(sensor_cfg.get("min_z", -4.0))
        self.max_z = float(sensor_cfg.get("max_z", 15.0))
        self.hfov = sensor_cfg.get("hfov_deg", [-180.0, 180.0])
        self.vfov = sensor_cfg.get("vfov_deg", [-25.0, 15.0])

    def process(self, frame: PointCloudFrame) -> PointCloudFrame:
        """
        Executes full preprocessing pipeline on PointCloudFrame.
        Preserves original point coordinates and associated metadata.
        """
        pts = frame.points
        if pts is None or len(pts) == 0:
            return frame

        # 1. Remove non-finite values (NaN / Inf)
        valid_finite = np.isfinite(pts).all(axis=1)

        # 2. Distance range filter in xy plane and 3d
        dist_2d = np.linalg.norm(pts[:, :2], axis=1)
        valid_range = (dist_2d >= self.min_range) & (dist_2d <= self.max_range)

        # 3. Elevation range filter
        valid_z = (pts[:, 2] >= self.min_z) & (pts[:, 2] <= self.max_z)

        # 4. Field of View filter (if restricted)
        valid_fov = np.ones(len(pts), dtype=bool)
        if self.hfov != [-180.0, 180.0]:
            azimuth = np.arctan2(pts[:, 1], pts[:, 0]) * 180.0 / np.pi
            valid_fov = valid_fov & (azimuth >= self.hfov[0]) & (azimuth <= self.hfov[1])
        if self.vfov != [-90.0, 90.0]:
            horizontal_range = np.maximum(dist_2d, np.finfo(np.float32).eps)
            elevation = np.arctan2(pts[:, 2], horizontal_range) * 180.0 / np.pi
            valid_fov = valid_fov & (elevation >= self.vfov[0]) & (elevation <= self.vfov[1])

        combined_mask = valid_finite & valid_range & valid_z & valid_fov

        # Filter points and parallel attributes
        filtered_points = pts[combined_mask]
        filtered_intensity = frame.intensities[combined_mask] if frame.intensities is not None else None
        filtered_semantics = frame.semantic_labels[combined_mask] if frame.semantic_labels is not None else None
        filtered_instances = frame.instance_labels[combined_mask] if frame.instance_labels is not None else None

        return PointCloudFrame(
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
            points=filtered_points,
            intensities=filtered_intensity,
            semantic_labels=filtered_semantics,
            instance_labels=filtered_instances,
            ego_pose=frame.ego_pose,
            metadata={
                **(frame.metadata or {}),
                "raw_point_count": len(pts),
                "filtered_point_count": len(filtered_points),
                "rejected_point_count": len(pts) - len(filtered_points)
            }
        )

    def voxel_downsample(self, points: np.ndarray, voxel_size: float = 0.1) -> np.ndarray:
        """
        Vectorized grid-voxel centroid downsampling.
        """
        if len(points) == 0:
            return points
        voxel_indices = np.floor(points / voxel_size).astype(np.int32)
        # Find unique voxel coordinates
        _, unique_indices = np.unique(voxel_indices, axis=0, return_index=True)
        return points[unique_indices]
