"""
Uniform High-Resolution 2.5D Baseline Grid with Vectorized Reduceat.
"""
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from mapping.cell_data import AdaptiveCell


class UniformBaselineGrid:
    """
    Uniform high-resolution grid (5 cm throughout the entire 0-100 m area).
    """

    def __init__(self, resolution: float = 0.05, max_range: float = 100.0):
        self.resolution = resolution
        self.max_range = max_range
        self.cells: Dict[Tuple[int, int], AdaptiveCell] = {}

    def clear(self) -> None:
        self.cells.clear()

    def insert_points(
        self,
        points: np.ndarray,
        semantic_labels: Optional[np.ndarray] = None,
        confidences: Optional[np.ndarray] = None,
        is_dynamic: Optional[np.ndarray] = None,
        timestamp: float = 0.0,
    ) -> int:
        if points is None or len(points) == 0:
            return 0

        pts_x = points[:, 0]
        pts_y = points[:, 1]
        pts_z = points[:, 2]
        dist_2d = np.sqrt(pts_x**2 + pts_y**2)

        valid_mask = dist_2d <= self.max_range
        if not np.any(valid_mask):
            return 0

        pts_x = pts_x[valid_mask]
        pts_y = pts_y[valid_mask]
        pts_z = pts_z[valid_mask]

        sem = semantic_labels[valid_mask] if semantic_labels is not None else np.zeros(len(pts_x), dtype=np.uint32)
        conf = confidences[valid_mask] if confidences is not None else np.ones(len(pts_x), dtype=np.float32) * 0.9
        dyn = is_dynamic[valid_mask] if is_dynamic is not None else np.zeros(len(pts_x), dtype=bool)

        res = float(self.resolution)
        cols = np.floor(pts_x / res).astype(np.int32)
        rows = np.floor(pts_y / res).astype(np.int32)

        OFFSET = 1_000_000
        comp_keys = (cols.astype(np.int64) + OFFSET) * 2_000_000 + (rows.astype(np.int64) + OFFSET)

        sort_order = np.argsort(comp_keys)
        sorted_keys = comp_keys[sort_order]
        sorted_z = pts_z[sort_order]
        sorted_cols = cols[sort_order]
        sorted_rows = rows[sort_order]
        sorted_sem = sem[sort_order]
        sorted_conf = conf[sort_order]
        sorted_dyn = dyn[sort_order]

        _, first_indices, counts = np.unique(sorted_keys, return_index=True, return_counts=True)

        min_zs = np.minimum.reduceat(sorted_z, first_indices)
        max_zs = np.maximum.reduceat(sorted_z, first_indices)
        sum_zs = np.add.reduceat(sorted_z, first_indices)
        mean_zs = sum_zs / counts

        cell_cols = sorted_cols[first_indices]
        cell_rows = sorted_rows[first_indices]
        cell_sems = sorted_sem[first_indices]
        cell_confs = np.add.reduceat(sorted_conf, first_indices) / counts
        cell_dyns = np.logical_or.reduceat(sorted_dyn, first_indices)

        x_centers = (cell_cols + 0.5) * res
        y_centers = (cell_rows + 0.5) * res
        drivables = (cell_sems == 1) & ((max_zs - min_zs) < 0.12)

        for i in range(len(first_indices)):
            c = int(cell_cols[i])
            r = int(cell_rows[i])
            self.cells[(c, r)] = AdaptiveCell(
                zone_id=-1,
                col=c,
                row=r,
                x_center=float(x_centers[i]),
                y_center=float(y_centers[i]),
                resolution=res,
                min_z=float(min_zs[i]),
                max_z=float(max_zs[i]),
                mean_z=float(mean_zs[i]),
                rep_z=float(mean_zs[i]),
                point_count=int(counts[i]),
                semantic_class=int(cell_sems[i]),
                confidence=float(cell_confs[i]),
                is_drivable=bool(drivables[i]),
                is_dynamic=bool(cell_dyns[i]),
                timestamp=timestamp,
                roughness=float(max_zs[i] - min_zs[i])
            )

        return len(self.cells)

    def get_memory_bytes(self) -> int:
        cell_size_bytes = 48
        return len(self.cells) * cell_size_bytes
