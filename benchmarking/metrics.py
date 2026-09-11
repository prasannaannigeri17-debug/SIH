"""
Benchmarking metrics calculations:
Memory reduction, cell-count reduction, latency differences, and distance-stratified accuracy/IoU.
"""
from typing import Dict, List, Any
import numpy as np


class PerformanceMetrics:
    """
    Computes exact performance metrics comparing Adaptive vs Uniform Baseline representations.
    """

    @staticmethod
    def calculate_memory_reduction(baseline_memory_bytes: int, adaptive_memory_bytes: int) -> float:
        """
        Formula: ((Baseline Memory - Adaptive Memory) / Baseline Memory) * 100
        """
        if baseline_memory_bytes <= 0:
            return 0.0
        reduction = ((baseline_memory_bytes - adaptive_memory_bytes) / baseline_memory_bytes) * 100.0
        return float(round(reduction, 2))

    @staticmethod
    def calculate_cell_reduction(baseline_cells: int, adaptive_cells: int) -> float:
        """
        Formula: ((Baseline Cells - Adaptive Cells) / Baseline Cells) * 100
        """
        if baseline_cells <= 0:
            return 0.0
        reduction = ((baseline_cells - adaptive_cells) / baseline_cells) * 100.0
        return float(round(reduction, 2))

    @staticmethod
    def calculate_iou_per_zone(
        pred_labels: np.ndarray,
        gt_labels: np.ndarray,
        points: np.ndarray,
        zone_ranges: List[tuple[float, float]],
        num_classes: int = 13
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculates mIoU stratified by radial distance zones.
        """
        dist = np.linalg.norm(points[:, :2], axis=1)
        results = {}

        for z_min, z_max in zone_ranges:
            zone_key = f"{int(z_min)}-{int(z_max)}m"
            mask = (dist >= z_min) & (dist < z_max)
            if not np.any(mask):
                results[zone_key] = {"mIoU": 0.0, "accuracy": 0.0, "point_count": 0}
                continue

            z_pred = pred_labels[mask]
            z_gt = gt_labels[mask]

            # Accuracy
            acc = float(np.mean(z_pred == z_gt))

            # Class IoUs
            ious = []
            for c in range(1, num_classes):
                intersection = np.sum((z_pred == c) & (z_gt == c))
                union = np.sum((z_pred == c) | (z_gt == c))
                if union > 0:
                    ious.append(intersection / union)

            miou = float(np.mean(ious)) if ious else 0.0
            results[zone_key] = {
                "mIoU": float(round(miou * 100.0, 2)),
                "accuracy": float(round(acc * 100.0, 2)),
                "point_count": int(np.sum(mask))
            }

        return results
