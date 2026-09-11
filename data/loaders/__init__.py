"""
Data loader factory and exports.
"""
import os
from data.loaders.base_loader import BasePointCloudLoader, PointCloudFrame
from data.loaders.kitti_loader import SemanticKITTILoader
from data.loaders.pcd_ply_loader import PCDPLYLoader
from data.loaders.scenario_generator import ProceduralScenarioGenerator


def create_loader(source_path: str = None, num_demo_frames: int = 60) -> BasePointCloudLoader:
    """
    Factory to obtain point cloud loader.
    If source_path is None or doesn't exist, defaults to ProceduralScenarioGenerator.
    """
    if source_path is None or not os.path.exists(source_path):
        return ProceduralScenarioGenerator(num_frames=num_demo_frames)

    if os.path.isdir(source_path):
        if os.path.exists(os.path.join(source_path, "velodyne")):
            return SemanticKITTILoader(source_path)
        # Check if contains .bin files
        bin_files = [f for f in os.listdir(source_path) if f.endswith(".bin")]
        if bin_files:
            return SemanticKITTILoader(source_path)
        return PCDPLYLoader(source_path)

    ext = os.path.splitext(source_path)[1].lower()
    if ext == ".bin":
        return SemanticKITTILoader(os.path.dirname(source_path))
    elif ext in [".pcd", ".ply"]:
        return PCDPLYLoader(source_path)

    return ProceduralScenarioGenerator(num_frames=num_demo_frames)
