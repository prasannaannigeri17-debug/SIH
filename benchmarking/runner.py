"""
Automated Benchmarking Experiment Runner.
Executes systematic comparisons between Uniform Baseline and Adaptive Grid configurations.
"""
import time
import os
import json
from typing import Dict, Any, List
import numpy as np

from data.loaders import create_loader
from preprocessing.pointcloud_processor import PointCloudProcessor
from perception.semantic_segmenter import SemanticSegmenter
from mapping.adaptive_grid import AdaptiveResolutionGrid
from mapping.uniform_grid import UniformBaselineGrid
from benchmarking.metrics import PerformanceMetrics
from benchmarking.profiler import PipelineProfiler


class BenchmarkRunner:
    """
    Executes Experiments A, B, and C to empirically measure memory, cell-count, and latency trade-offs.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def run_full_benchmark(self, num_frames: int = 30, source_path: str = None) -> Dict[str, Any]:
        loader = create_loader(source_path, num_demo_frames=num_frames)
        preprocessor = PointCloudProcessor(self.config)
        segmenter = SemanticSegmenter(self.config)

        # Baseline: Uniform 5cm grid
        baseline_grid = UniformBaselineGrid(resolution=0.05, max_range=100.0)

        # Adaptive Config 1 (Default): 5cm / 15cm / 30cm / 60cm
        adaptive_grid_default = AdaptiveResolutionGrid(self.config)

        # Adaptive Config 2 (Aggressive): 5cm / 20cm / 40cm / 80cm
        cfg_aggressive = dict(self.config)
        cfg_aggressive["grid"] = {
            "zones": [
                {"name": "Zone A", "min_distance": 0.0, "max_distance": 10.0, "resolution": 0.05},
                {"name": "Zone B", "min_distance": 10.0, "max_distance": 30.0, "resolution": 0.20},
                {"name": "Zone C", "min_distance": 30.0, "max_distance": 60.0, "resolution": 0.40},
                {"name": "Zone D", "min_distance": 60.0, "max_distance": 100.0, "resolution": 0.80},
            ]
        }
        adaptive_grid_aggressive = AdaptiveResolutionGrid(cfg_aggressive)

        # Telemetry accumulators
        baseline_cells_list = []
        baseline_time_ms_list = []
        adaptive_cells_list = []
        adaptive_time_ms_list = []
        aggr_cells_list = []
        aggr_time_ms_list = []

        total_points_processed = 0

        for frame_idx in range(min(num_frames, len(loader))):
            raw_frame = loader.get_frame(frame_idx)
            frame = preprocessor.process(raw_frame)
            total_points_processed += len(frame.points)

            sem_labels, confs, is_dyn, _ = segmenter.segment(frame)

            # Benchmark 1: Uniform Baseline
            baseline_grid.clear()
            t0 = time.perf_counter()
            baseline_grid.insert_points(frame.points, sem_labels, confs, is_dyn, frame.timestamp)
            t_baseline = (time.perf_counter() - t0) * 1000.0
            baseline_cells_list.append(len(baseline_grid.cells))
            baseline_time_ms_list.append(t_baseline)

            # Benchmark 2: Default Adaptive Grid
            adaptive_grid_default.clear()
            t0 = time.perf_counter()
            adaptive_grid_default.insert_points(frame.points, sem_labels, confs, is_dyn, frame.timestamp)
            t_adaptive = (time.perf_counter() - t0) * 1000.0
            adaptive_cells_list.append(len(adaptive_grid_default.cells))
            adaptive_time_ms_list.append(t_adaptive)

            # Benchmark 3: Aggressive Adaptive Grid
            adaptive_grid_aggressive.clear()
            t0 = time.perf_counter()
            adaptive_grid_aggressive.insert_points(frame.points, sem_labels, confs, is_dyn, frame.timestamp)
            t_aggr = (time.perf_counter() - t0) * 1000.0
            aggr_cells_list.append(len(adaptive_grid_aggressive.cells))
            aggr_time_ms_list.append(t_aggr)

        # Average statistics
        avg_baseline_cells = int(np.mean(baseline_cells_list))
        avg_adaptive_cells = int(np.mean(adaptive_cells_list))
        avg_aggr_cells = int(np.mean(aggr_cells_list))

        avg_baseline_time = float(np.mean(baseline_time_ms_list))
        avg_adaptive_time = float(np.mean(adaptive_time_ms_list))
        avg_aggr_time = float(np.mean(aggr_time_ms_list))

        baseline_mem_kb = (avg_baseline_cells * 48) / 1024.0
        adaptive_mem_kb = (avg_adaptive_cells * 48) / 1024.0
        aggr_mem_kb = (avg_aggr_cells * 48) / 1024.0

        cell_reduction_default = PerformanceMetrics.calculate_cell_reduction(avg_baseline_cells, avg_adaptive_cells)
        mem_reduction_default = PerformanceMetrics.calculate_memory_reduction(int(baseline_mem_kb * 1024), int(adaptive_mem_kb * 1024))

        cell_reduction_aggr = PerformanceMetrics.calculate_cell_reduction(avg_baseline_cells, avg_aggr_cells)
        mem_reduction_aggr = PerformanceMetrics.calculate_memory_reduction(int(baseline_mem_kb * 1024), int(aggr_mem_kb * 1024))

        report = {
            "summary": {
                "frames_evaluated": len(baseline_cells_list),
                "total_points_evaluated": total_points_processed,
                "avg_points_per_frame": int(total_points_processed / max(1, len(baseline_cells_list))),
            },
            "uniform_baseline_5cm": {
                "avg_cells": avg_baseline_cells,
                "avg_memory_kb": float(round(baseline_mem_kb, 2)),
                "avg_insertion_latency_ms": float(round(avg_baseline_time, 2)),
                "equivalent_fps": float(round(1000.0 / max(0.1, avg_baseline_time), 1))
            },
            "adaptive_foveated_default": {
                "zones": "0-10m: 5cm | 10-30m: 15cm | 30-60m: 30cm | 60-100m: 60cm",
                "avg_cells": avg_adaptive_cells,
                "avg_memory_kb": float(round(adaptive_mem_kb, 2)),
                "cell_reduction_pct": cell_reduction_default,
                "memory_reduction_pct": mem_reduction_default,
                "avg_insertion_latency_ms": float(round(avg_adaptive_time, 2)),
                "equivalent_fps": float(round(1000.0 / max(0.1, avg_adaptive_time), 1)),
                "speedup_factor": float(round(avg_baseline_time / max(0.01, avg_adaptive_time), 2))
            },
            "adaptive_foveated_aggressive": {
                "zones": "0-10m: 5cm | 10-30m: 20cm | 30-60m: 40cm | 60-100m: 80cm",
                "avg_cells": avg_aggr_cells,
                "avg_memory_kb": float(round(aggr_mem_kb, 2)),
                "cell_reduction_pct": cell_reduction_aggr,
                "memory_reduction_pct": mem_reduction_aggr,
                "avg_insertion_latency_ms": float(round(avg_aggr_time, 2)),
                "equivalent_fps": float(round(1000.0 / max(0.1, avg_aggr_time), 1)),
                "speedup_factor": float(round(avg_baseline_time / max(0.01, avg_aggr_time), 2))
            }
        }
        return report
