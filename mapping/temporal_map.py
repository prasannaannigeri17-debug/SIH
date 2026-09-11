"""
Temporal Map Integration & Multi-Frame Maintenance.
Handles moving object eviction, static terrain reinforcement, and confidence updates over time.
"""
from typing import Dict, Any, List
import numpy as np
from mapping.adaptive_grid import AdaptiveResolutionGrid
from mapping.cell_data import AdaptiveCell


class TemporalMapManager:
    """
    Manages continuous temporal updates for the adaptive 2.5D elevation & semantic grid.
    """

    def __init__(self, config: Dict[str, Any], grid: AdaptiveResolutionGrid):
        self.config = config
        self.grid = grid
        t_cfg = config.get("temporal", {})
        self.enable_decay = bool(t_cfg.get("enable_decay", True))
        self.static_decay_rate = float(t_cfg.get("static_decay_rate", 0.02))
        self.dynamic_decay_rate = float(t_cfg.get("dynamic_decay_rate", 0.80))
        self.stale_timeout_sec = float(t_cfg.get("stale_timeout_sec", 1.5))
        self.current_time: float = 0.0

    def update_frame(
        self,
        points: np.ndarray,
        semantic_labels: np.ndarray,
        confidences: np.ndarray,
        is_dynamic: np.ndarray,
        timestamp: float,
        aggregation_mode: str = None
    ) -> int:
        """
        Integrates incoming frame observations, decays stale cells, and evicts dynamic objects.
        """
        self.current_time = timestamp

        if self.enable_decay:
            # 1. Decay and evict stale cells
            stale_keys = []
            for key, cell in self.grid.cells.items():
                age = timestamp - cell.timestamp
                if age > 0:
                    if cell.is_dynamic:
                        # Dynamic objects decay quickly and get evicted when stale
                        cell.confidence *= np.exp(-self.dynamic_decay_rate * age)
                        if age > self.stale_timeout_sec or cell.confidence < 0.2:
                            stale_keys.append(key)
                    else:
                        # Static terrain decays slowly
                        cell.confidence *= np.exp(-self.static_decay_rate * age)

            for key in stale_keys:
                self.grid.cells.pop(key, None)

        # 2. Insert new observations into grid
        self.grid.insert_points(
            points=points,
            semantic_labels=semantic_labels,
            confidences=confidences,
            is_dynamic=is_dynamic,
            timestamp=timestamp,
            aggregation_mode=aggregation_mode
        )

        return len(self.grid.cells)
