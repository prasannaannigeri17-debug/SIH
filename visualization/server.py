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


def create_app(config_path: str = "config/config.yaml", class_cfg_path: str = "config/semantic_classes.yaml", source_path: Optional[str] = None) -> FastAPI:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    with open(class_cfg_path, "r", encoding="utf-8") as f:
        class_cfg = yaml.safe_load(f)

    app = FastAPI(title="Adaptive 2.5D LiDAR Mapping Dashboard")

    web_dir = os.path.join(os.path.dirname(__file__), "web")
    if os.path.exists(web_dir):
        app.mount("/static", StaticFiles(directory=web_dir), name="static")

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
        "force_update": True, # Always process the very first frame
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

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        try:
            while True:
                # Command handling
                try:
                    data = await asyncio.wait_for(websocket.receive_text(), timeout=0.02)
                    msg = json.loads(data)
                    cmd = msg.get("command")
                    if cmd == "play":
                        pipeline_state["is_playing"] = True
                    elif cmd == "pause":
                        pipeline_state["is_playing"] = False
                    elif cmd == "step":
                        pipeline_state["is_playing"] = False
                        pipeline_state["current_frame_idx"] += 1
                        pipeline_state["force_update"] = True
                    elif cmd == "seek":
                        pipeline_state["current_frame_idx"] = int(msg.get("frame_index", 0))
                        pipeline_state["force_update"] = True
                    elif cmd == "set_aggregation":
                        pipeline_state["aggregation_mode"] = str(msg.get("mode", "robust"))
                        pipeline_state["force_update"] = True
                    elif cmd == "reset":
                        pipeline_state["current_frame_idx"] = 0
                        pipeline_state["adaptive_grid"].clear()
                        pipeline_state["uniform_grid"].clear()
                        pipeline_state["force_update"] = True
                except asyncio.TimeoutError:
                    pass

                if not pipeline_state["is_playing"] and not pipeline_state["force_update"]:
                    await asyncio.sleep(0.05)
                    continue

                # Process frame
                pipeline_state["force_update"] = False
                
                idx = pipeline_state["current_frame_idx"]
                loader: BasePointCloudLoader = pipeline_state["loader"]
                total_frames = len(loader)

                if idx >= total_frames:
                    idx = 0
                    pipeline_state["current_frame_idx"] = 0

                profiler: PipelineProfiler = pipeline_state["profiler"]

                profiler.start_stage("load")
                raw_frame = loader.get_frame(idx)
                profiler.end_stage("load")

                profiler.start_stage("preprocess")
                frame = pipeline_state["preprocessor"].process(raw_frame)
                profiler.end_stage("preprocess")

                profiler.start_stage("perception")
                sem_labels, confs, is_dyn, bboxes = pipeline_state["segmenter"].segment(frame)
                profiler.end_stage("perception")

                profiler.start_stage("adaptive_grid")
                adaptive_grid: AdaptiveResolutionGrid = pipeline_state["adaptive_grid"]
                adaptive_grid.clear()
                adaptive_grid.insert_points(
                    points=frame.points,
                    semantic_labels=sem_labels,
                    confidences=confs,
                    is_dynamic=is_dyn,
                    timestamp=frame.timestamp,
                    aggregation_mode=pipeline_state["aggregation_mode"]
                )
                profiler.end_stage("adaptive_grid")

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

                profiler.record_frame()
                prof_summary = profiler.get_summary()

                adaptive_cell_count = len(adaptive_grid.cells)
                baseline_cell_count = len(uniform_grid.cells)
                cell_reduction_pct = PerformanceMetrics.calculate_cell_reduction(baseline_cell_count, adaptive_cell_count)
                adaptive_mem_kb = adaptive_grid.get_memory_bytes() / 1024.0
                baseline_mem_kb = uniform_grid.get_memory_bytes() / 1024.0
                mem_reduction_pct = PerformanceMetrics.calculate_memory_reduction(int(baseline_mem_kb*1024), int(adaptive_mem_kb*1024))

                lat_load = prof_summary["stages_ms"].get("load", 0)
                lat_prep = prof_summary["stages_ms"].get("preprocess", 0)
                lat_perc = prof_summary["stages_ms"].get("perception", 0)
                lat_adapt = prof_summary["stages_ms"].get("adaptive_grid", 0)
                lat_base = prof_summary["stages_ms"].get("baseline_grid", 0)
                
                total_adapt_ms = lat_load + lat_prep + lat_perc + lat_adapt
                total_base_ms = lat_load + lat_prep + lat_perc + lat_base

                pts_xyz = frame.points
                step = max(1, len(pts_xyz) // 4000)
                sub_pts = pts_xyz[::step]
                sub_sem = sem_labels[::step]
                sub_int = frame.intensities[::step] if frame.intensities is not None else np.ones(len(sub_pts), dtype=np.float32)

                cells_export = adaptive_grid.export_cells_for_visualization(max_cells=4000)

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
                        "hardware_device": "CPU / Intel Core" if "torch" not in sys.modules else "CUDA GPU",
                        "benchmark": {
                            "adaptive_latency_ms": float(round(total_adapt_ms, 2)),
                            "baseline_latency_ms": float(round(total_base_ms, 2)),
                            "speedup": float(round(total_base_ms / max(0.01, total_adapt_ms), 2))
                        }
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
                    await asyncio.sleep(1.0 / pipeline_state["fps_target"])

        except WebSocketDisconnect:
            pass

    return app
