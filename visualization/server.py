"""
FastAPI + WebSocket Streaming Server for Real-Time 3D LiDAR & Adaptive Grid Dashboard.
"""
import os
import sys
import json
import asyncio
import time
import yaml
from typing import Dict, Any, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn
import numpy as np

from data.loaders import create_loader, BasePointCloudLoader
from preprocessing.pointcloud_processor import PointCloudProcessor
from perception.semantic_segmenter import SemanticSegmenter
from mapping.adaptive_grid import AdaptiveResolutionGrid
from mapping.uniform_grid import UniformBaselineGrid
from mapping.temporal_map import TemporalMapManager
from benchmarking.profiler import PipelineProfiler
from benchmarking.metrics import PerformanceMetrics
from benchmarking.runner import BenchmarkRunner


def create_app(
    config_path: str = "config/config.yaml",
    class_cfg_path: str = "config/semantic_classes.yaml",
    source_path: Optional[str] = None
) -> FastAPI:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    with open(class_cfg_path, "r") as f:
        class_cfg = yaml.safe_load(f)

    app = FastAPI(title="Adaptive 2.5D LiDAR Mapping Dashboard")

    # Static web files path
    web_dir = os.path.join(os.path.dirname(__file__), "web")
    if os.path.exists(web_dir):
        app.mount("/static", StaticFiles(directory=web_dir), name="static")

    # State
    pipeline_state = {
        "loader": create_loader(source_path),
        "preprocessor": PointCloudProcessor(config),
        "segmenter": SemanticSegmenter(config),
        "adaptive_grid": AdaptiveResolutionGrid(config, class_cfg.get("classes", {})),
        "uniform_grid": UniformBaselineGrid(resolution=config.get("baseline_grid", {}).get("resolution", 0.05)),
        "temporal_manager": None,
        "profiler": PipelineProfiler(),
        "config": config,
        "class_cfg": class_cfg,
        "current_frame_idx": 0,
        "is_playing": True,
        "fps_target": config.get("server", {}).get("fps_target", 15),
        "aggregation_mode": config.get("grid", {}).get("default_aggregation", "robust")
    }
    pipeline_state["temporal_manager"] = TemporalMapManager(config, pipeline_state["adaptive_grid"])

    @app.get("/")
    async def get_index():
        index_file = os.path.join(web_dir, "index.html")
        if os.path.exists(index_file):
            with open(index_file, "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read())
        return HTMLResponse("<h3>Dashboard UI loading...</h3>")

    @app.get("/api/config")
    async def get_config():
        return JSONResponse(content={
            "config": pipeline_state["config"],
            "classes": pipeline_state["class_cfg"].get("classes", {})
        })

    @app.get("/api/benchmark")
    async def run_benchmark():
        runner = BenchmarkRunner(pipeline_state["config"])
        res = runner.run_full_benchmark(num_frames=15)
        return JSONResponse(content=res)

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        try:
            while True:
                # Check for incoming client commands (play, pause, seek, set_mode)
                try:
                    data = await asyncio.wait_for(websocket.receive_text(), timeout=0.01)
                    msg = json.loads(data)
                    cmd = msg.get("command")
                    if cmd == "play":
                        pipeline_state["is_playing"] = True
                    elif cmd == "pause":
                        pipeline_state["is_playing"] = False
                    elif cmd == "seek":
                        pipeline_state["current_frame_idx"] = int(msg.get("frame_index", 0))
                    elif cmd == "set_aggregation":
                        pipeline_state["aggregation_mode"] = str(msg.get("mode", "robust"))
                    elif cmd == "reset":
                        pipeline_state["current_frame_idx"] = 0
                        pipeline_state["adaptive_grid"].clear()
                        pipeline_state["uniform_grid"].clear()
                        pipeline_state["temporal_manager"].current_time = 0.0
                except asyncio.TimeoutError:
                    pass

                # If playing, process next frame
                idx = pipeline_state["current_frame_idx"]
                loader: BasePointCloudLoader = pipeline_state["loader"]
                total_frames = len(loader)

                if idx >= total_frames:
                    idx = 0
                    pipeline_state["current_frame_idx"] = 0

                profiler: PipelineProfiler = pipeline_state["profiler"]

                # Stage 1: Load raw frame
                profiler.start_stage("load")
                raw_frame = loader.get_frame(idx)
                profiler.end_stage("load")

                # Stage 2: Preprocess
                profiler.start_stage("preprocess")
                frame = pipeline_state["preprocessor"].process(raw_frame)
                profiler.end_stage("preprocess")

                # Stage 3: Semantic Segmentation & Dynamic Perception
                profiler.start_stage("perception")
                sem_labels, confs, is_dyn, bboxes = pipeline_state["segmenter"].segment(frame)
                profiler.end_stage("perception")

                # Stage 4: Adaptive Grid Projection & 2.5D Aggregation
                profiler.start_stage("adaptive_grid")
                adaptive_grid: AdaptiveResolutionGrid = pipeline_state["adaptive_grid"]
                pipeline_state["temporal_manager"].update_frame(
                    points=frame.points,
                    semantic_labels=sem_labels,
                    confidences=confs,
                    is_dynamic=is_dyn,
                    timestamp=frame.timestamp,
                    aggregation_mode=pipeline_state["aggregation_mode"]
                )
                profiler.end_stage("adaptive_grid")

                # Stage 5: Baseline Grid calculation (for real telemetry comparison)
                profiler.start_stage("baseline_grid")
                uniform_grid: UniformBaselineGrid = pipeline_state["uniform_grid"]
                uniform_grid.clear()
                uniform_grid.insert_points(
                    points=frame.points,
                    semantic_labels=sem_labels,
                    confidences=confs,
                    is_dynamic=is_dyn,
                    timestamp=frame.timestamp
                )
                profiler.end_stage("baseline_grid")

                # Snapshot-only adaptive comparison to avoid counting the persistent temporal map
                adaptive_snapshot = AdaptiveResolutionGrid(
                    pipeline_state["config"],
                    pipeline_state["class_cfg"].get("classes", {})
                )
                adaptive_snapshot.insert_points(
                    points=frame.points,
                    semantic_labels=sem_labels,
                    confidences=confs,
                    is_dynamic=is_dyn,
                    timestamp=frame.timestamp,
                    aggregation_mode=pipeline_state["aggregation_mode"]
                )

                profiler.record_frame()
                prof_summary = profiler.get_summary()

                # Calculate live cell and memory savings from the current frame snapshot
                adaptive_cell_count = len(adaptive_snapshot.cells)
                baseline_cell_count = len(uniform_grid.cells)
                cell_reduction_pct = PerformanceMetrics.calculate_cell_reduction(baseline_cell_count, adaptive_cell_count)
                adaptive_mem_kb = adaptive_snapshot.get_memory_bytes() / 1024.0
                baseline_mem_kb = uniform_grid.get_memory_bytes() / 1024.0
                mem_reduction_pct = PerformanceMetrics.calculate_memory_reduction(int(baseline_mem_kb*1024), int(adaptive_mem_kb*1024))

                # Export sampled point cloud for WebGL rendering (downsample for smooth 60fps web streaming)
                pts_xyz = frame.points
                # Keep WebSocket frames small enough for browser and proxy limits.
                step = max(1, len(pts_xyz) // 2500)
                sub_pts = pts_xyz[::step]
                sub_sem = sem_labels[::step]
                sub_int = frame.intensities[::step] if frame.intensities is not None else np.ones(len(sub_pts), dtype=np.float32)

                # Export adaptive cells (up to 4000 cells for real-time mesh rendering)
                cells_export = adaptive_grid.export_cells_for_visualization(max_cells=1200)

                payload = {
                    "frame_id": idx,
                    "total_frames": total_frames,
                    "timestamp": float(round(frame.timestamp, 3)),
                    "telemetry": {
                        "fps": prof_summary["fps"],
                        "total_latency_ms": prof_summary["total_latency_ms"],
                        "stage_latencies": prof_summary["stages_ms"],
                        "point_count": len(frame.points),
                        "adaptive_cells": adaptive_cell_count,
                        "baseline_cells": baseline_cell_count,
                        "cell_reduction_pct": cell_reduction_pct,
                        "adaptive_mem_kb": float(round(adaptive_mem_kb, 1)),
                        "baseline_mem_kb": float(round(baseline_mem_kb, 1)),
                        "mem_reduction_pct": mem_reduction_pct,
                        "zone_distribution": adaptive_grid.get_cell_count_per_zone(),
                        "hardware_device": "CPU / Intel Core" if "torch" not in sys.modules else "CUDA GPU"
                    },
                    "points": {
                        "xyz": sub_pts.tolist(),
                        "sem": sub_sem.tolist(),
                        "intensity": sub_int.tolist()
                    },
                    "cells": cells_export,
                    "bounding_boxes": bboxes,
                    "ego_pose": frame.ego_pose.tolist() if frame.ego_pose is not None else None
                }

                await websocket.send_text(json.dumps(payload))

                if pipeline_state["is_playing"]:
                    pipeline_state["current_frame_idx"] += 1

                # Maintain target streaming frame rate
                await asyncio.sleep(1.0 / pipeline_state["fps_target"])

        except WebSocketDisconnect:
            pass

    return app
