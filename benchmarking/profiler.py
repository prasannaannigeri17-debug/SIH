"""
High-Precision Pipeline Profiler and Hardware Telemetry.
"""
import time
import tracemalloc
from typing import Dict, Any, Optional


class PipelineProfiler:
    """
    Measures per-stage latency, FPS, memory usage, and hardware configuration.
    """

    def __init__(self):
        self.stage_times: Dict[str, float] = {}
        self._start_timestamps: Dict[str, float] = {}
        self.frame_count: int = 0
        self.last_frame_time: float = time.perf_counter()
        self.fps: float = 0.0

        # Memory tracking
        tracemalloc.start()

    def start_stage(self, stage_name: str) -> None:
        self._start_timestamps[stage_name] = time.perf_counter()

    def end_stage(self, stage_name: str) -> float:
        if stage_name not in self._start_timestamps:
            return 0.0
        elapsed_ms = (time.perf_counter() - self._start_timestamps[stage_name]) * 1000.0
        self.stage_times[stage_name] = elapsed_ms
        return elapsed_ms

    def record_frame(self) -> float:
        now = time.perf_counter()
        dt = now - self.last_frame_time
        self.last_frame_time = now
        self.frame_count += 1
        if dt > 0:
            instant_fps = 1.0 / dt
            # Exponential smoothing for stable UI display
            self.fps = 0.8 * self.fps + 0.2 * instant_fps if self.fps > 0 else instant_fps
        return self.fps

    def get_summary(self) -> Dict[str, Any]:
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        total_latency = sum(self.stage_times.values())
        return {
            "fps": float(round(self.fps, 1)),
            "total_latency_ms": float(round(total_latency, 2)),
            "stages_ms": {k: float(round(v, 2)) for k, v in self.stage_times.items()},
            "current_mem_mb": float(round(current_mem / (1024 * 1024), 2)),
            "peak_mem_mb": float(round(peak_mem / (1024 * 1024), 2)),
        }
