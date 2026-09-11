"""
Loader for SemanticKITTI and raw KITTI format (.bin point clouds and .label files).
"""
import os
import glob
from typing import Optional, List
import numpy as np
from data.loaders.base_loader import BasePointCloudLoader, PointCloudFrame


class SemanticKITTILoader(BasePointCloudLoader):
    """
    Loads raw Velodyne .bin files and corresponding SemanticKITTI .label files.
    """

    def __init__(self, sequence_path: str):
        self.sequence_path = os.path.abspath(sequence_path)
        self.velodyne_dir = os.path.join(self.sequence_path, "velodyne")
        self.labels_dir = os.path.join(self.sequence_path, "labels")
        self.poses_file = os.path.join(self.sequence_path, "poses.txt")

        if not os.path.exists(self.velodyne_dir):
            # Fallback if path directly points to directory with .bin files
            if os.path.exists(self.sequence_path):
                self.velodyne_dir = self.sequence_path

        self.bin_files: List[str] = sorted(glob.glob(os.path.join(self.velodyne_dir, "*.bin")))
        self.has_labels = os.path.exists(self.labels_dir)
        self.poses: Optional[np.ndarray] = None

        if os.path.exists(self.poses_file):
            try:
                raw_poses = np.loadtxt(self.poses_file)
                self.poses = raw_poses.reshape(-1, 3, 4)
            except Exception:
                self.poses = None

    def __len__(self) -> int:
        return len(self.bin_files)

    def get_frame(self, index: int) -> PointCloudFrame:
        if index < 0 or index >= len(self.bin_files):
            raise IndexError(f"Frame index {index} out of bounds (total frames: {len(self.bin_files)})")

        bin_path = self.bin_files[index]
        scan = np.fromfile(bin_path, dtype=np.float32)
        points = scan.reshape((-1, 4))
        xyz = points[:, :3]
        intensity = points[:, 3]

        semantic_labels = None
        instance_labels = None

        if self.has_labels:
            base_name = os.path.splitext(os.path.basename(bin_path))[0]
            label_path = os.path.join(self.labels_dir, f"{base_name}.label")
            if os.path.exists(label_path):
                raw_label = np.fromfile(label_path, dtype=np.uint32)
                semantic_labels = raw_label & 0xFFFF  # Lower 16 bits = semantic class
                instance_labels = raw_label >> 16      # Upper 16 bits = instance ID

        ego_pose = None
        if self.poses is not None and index < len(self.poses):
            pose_3x4 = self.poses[index]
            ego_pose = np.eye(4, dtype=np.float32)
            ego_pose[:3, :4] = pose_3x4

        return PointCloudFrame(
            frame_id=index,
            timestamp=index * 0.1,  # 10 Hz standard KITTI frequency
            points=xyz,
            intensities=intensity,
            semantic_labels=semantic_labels,
            instance_labels=instance_labels,
            ego_pose=ego_pose,
            metadata={"source_file": bin_path}
        )

    def reset(self) -> None:
        pass
