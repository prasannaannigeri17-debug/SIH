"""
High-Performance Vectorized Adaptive Variable-Resolution 2.5D Spatial Hash Grid Engine.
Uses numpy.reduceat for ultra-fast C-level cell aggregation (10-20ms per frame).
"""
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from mapping.cell_data import AdaptiveCell


class AdaptiveResolutionGrid:
    """
    Multi-resolution spatial hash grid with vectorized reduceat aggregation.
    """

    def __init__(self, config: Dict[str, Any], class_cfg: Optional[Dict[int, Dict]] = None):
        self.config = config
        self.grid_cfg = config.get("grid", {})
        self.zones = self.grid_cfg.get("zones", [
            {"name": "Zone A", "min_distance": 0.0, "max_distance": 10.0, "resolution": 0.05, "color": "#00E5FF"},
            {"name": "Zone B", "min_distance": 10.0, "max_distance": 30.0, "resolution": 0.15, "color": "#00E676"},
            {"name": "Zone C", "min_distance": 30.0, "max_distance": 60.0, "resolution": 0.30, "color": "#FFEA00"},
            {"name": "Zone D", "min_distance": 60.0, "max_distance": 100.0, "resolution": 0.60, "color": "#FF3D00"},
        ])
        self.default_aggregation = self.grid_cfg.get("default_aggregation", "robust")
        self.class_cfg = class_cfg or {}

        self.zone_min_dists = np.array([z["min_distance"] for z in self.zones], dtype=np.float32)
        self.zone_max_dists = np.array([z["max_distance"] for z in self.zones], dtype=np.float32)
        self.zone_resolutions = np.array([z["resolution"] for z in self.zones], dtype=np.float32)

        self.cells: Dict[Tuple[int, int, int], AdaptiveCell] = {}

    def clear(self) -> None:
        self.cells.clear()

    def get_zone_for_distance(self, distance: float) -> int:
        if distance < self.zone_min_dists[0] or distance > self.zone_max_dists[-1]:
            return -1
        for idx in range(len(self.zones)):
            if idx == len(self.zones) - 1:
                if self.zone_min_dists[idx] <= distance <= self.zone_max_dists[idx]:
                    return idx
            else:
                if self.zone_min_dists[idx] <= distance < self.zone_max_dists[idx]:
                    return idx
        return -1

    def assign_zones_vectorized(self, distances: np.ndarray) -> np.ndarray:
        zone_ids = np.full(len(distances), -1, dtype=np.int32)
        for idx in range(len(self.zones)):
            if idx == len(self.zones) - 1:
                mask = (distances >= self.zone_min_dists[idx]) & (distances <= self.zone_max_dists[idx])
            else:
                mask = (distances >= self.zone_min_dists[idx]) & (distances < self.zone_max_dists[idx])
            zone_ids[mask] = idx
        return zone_ids

    def insert_points(
        self,
        points: np.ndarray,
        semantic_labels: Optional[np.ndarray] = None,
        confidences: Optional[np.ndarray] = None,
        is_dynamic: Optional[np.ndarray] = None,
        timestamp: float = 0.0,
        aggregation_mode: Optional[str] = None
    ) -> int:
        if points is None or len(points) == 0:
            return 0

        mode = aggregation_mode or self.default_aggregation
        pts_x = points[:, 0]
        pts_y = points[:, 1]
        pts_z = points[:, 2]
        dist_2d = np.sqrt(pts_x**2 + pts_y**2)

        zone_ids = self.assign_zones_vectorized(dist_2d)
        valid_mask = zone_ids >= 0
        if not np.any(valid_mask):
            return 0

        pts_x = pts_x[valid_mask]
        pts_y = pts_y[valid_mask]
        pts_z = pts_z[valid_mask]
        zone_ids = zone_ids[valid_mask]

        sem = semantic_labels[valid_mask] if semantic_labels is not None else np.zeros(len(pts_x), dtype=np.uint32)
        conf = confidences[valid_mask] if confidences is not None else np.ones(len(pts_x), dtype=np.float32) * 0.9
        dyn = is_dynamic[valid_mask] if is_dynamic is not None else np.zeros(len(pts_x), dtype=bool)

        for z_id in np.unique(zone_ids):
            z_mask = zone_ids == z_id
            res = float(self.zone_resolutions[z_id])

            zx = pts_x[z_mask]
            zy = pts_y[z_mask]
            zz = pts_z[z_mask]
            z_sem = sem[z_mask]
            z_conf = conf[z_mask]
            z_dyn = dyn[z_mask]

            cols = np.floor(zx / res).astype(np.int32)
            rows = np.floor(zy / res).astype(np.int32)

            OFFSET = 1_000_000
            comp_keys = (cols.astype(np.int64) + OFFSET) * 2_000_000 + (rows.astype(np.int64) + OFFSET)

            sort_order = np.argsort(comp_keys)
            sorted_keys = comp_keys[sort_order]
            sorted_z = zz[sort_order]
            sorted_cols = cols[sort_order]
            sorted_rows = rows[sort_order]
            sorted_sem = z_sem[sort_order]
            sorted_conf = z_conf[sort_order]
            sorted_dyn = z_dyn[sort_order]

            _, first_indices, counts = np.unique(sorted_keys, return_index=True, return_counts=True)

            # Vectorized aggregation using reduceat
            min_zs = np.minimum.reduceat(sorted_z, first_indices)
            max_zs = np.maximum.reduceat(sorted_z, first_indices)
            sum_zs = np.add.reduceat(sorted_z, first_indices)
            mean_zs = sum_zs / counts

            rep_zs = mean_zs
            if mode == "min":
                rep_zs = min_zs
            elif mode == "max":
                rep_zs = max_zs
            elif mode == "robust":
                # Most cells contain few, low-variance samples; keep that path vectorized.
                rep_zs = mean_zs.copy()
                percentile_cells = np.flatnonzero((counts > 4) & ((max_zs - min_zs) > 0.35))
                for cell_index in percentile_cells:
                    start = first_indices[cell_index]
                    end = start + counts[cell_index]
                    cell_z = sorted_z[start:end]
                    rep_zs[cell_index] = (
                        np.percentile(cell_z, 5) + np.percentile(cell_z, 95)
                    ) * 0.5

            cell_cols = sorted_cols[first_indices]
            cell_rows = sorted_rows[first_indices]
            # Count cell/class pairs in sorted order and select the dominant class.
            # This avoids input-order dependence without a Python loop per cell.
            semantic_pair_keys = sorted_keys * 65536 + sorted_sem.astype(np.int64)
            unique_pairs, pair_counts = np.unique(semantic_pair_keys, return_counts=True)
            pair_cells = unique_pairs // 65536
            pair_semantics = (unique_pairs % 65536).astype(np.uint32)
            pair_starts = np.r_[0, 1 + np.flatnonzero(pair_cells[1:] != pair_cells[:-1])]
            pair_group_ends = np.r_[pair_starts[1:], len(pair_counts)]
            max_pair_counts = np.maximum.reduceat(pair_counts, pair_starts)
            repeated_max_counts = np.repeat(max_pair_counts, pair_group_ends - pair_starts)
            winner_mask = pair_counts == repeated_max_counts
            _, winner_indices = np.unique(pair_cells[winner_mask], return_index=True)
            cell_sems = pair_semantics[winner_mask][winner_indices]
            cell_confs = np.add.reduceat(sorted_conf, first_indices) / counts
            cell_dyns = np.logical_or.reduceat(sorted_dyn, first_indices)

            x_centers = (cell_cols + 0.5) * res
            y_centers = (cell_rows + 0.5) * res
            drivables = (cell_sems == 1) & ((max_zs - min_zs) < 0.12)

            # Populate dictionary in batch
            for i in range(len(first_indices)):
                c = int(cell_cols[i])
                r = int(cell_rows[i])
                cell_key = (int(z_id), c, r)

                self.cells[cell_key] = AdaptiveCell(
                    zone_id=int(z_id),
                    col=c,
                    row=r,
                    x_center=float(x_centers[i]),
                    y_center=float(y_centers[i]),
                    resolution=res,
                    min_z=float(min_zs[i]),
                    max_z=float(max_zs[i]),
                    mean_z=float(mean_zs[i]),
                    rep_z=float(rep_zs[i]),
                    point_count=int(counts[i]),
                    semantic_class=int(cell_sems[i]),
                    confidence=float(cell_confs[i]),
                    is_drivable=bool(drivables[i]),
                    is_dynamic=bool(cell_dyns[i]),
                    timestamp=timestamp,
                    roughness=float(max_zs[i] - min_zs[i])
                )

        return len(self.cells)

    def get_cell_count_per_zone(self) -> Dict[str, int]:
        counts = {z["name"]: 0 for z in self.zones}
        for (z_id, _, _) in self.cells.keys():
            if 0 <= z_id < len(self.zones):
                counts[self.zones[z_id]["name"]] += 1
        return counts

    def get_memory_bytes(self) -> int:
        cell_size_bytes = 48
        return len(self.cells) * cell_size_bytes

    def export_cells_for_visualization(self, max_cells: int = 25000) -> List[Dict]:
        exported = []
        for cell in list(self.cells.values())[:max_cells]:
            exported.append(cell.to_dict())
        return exported
