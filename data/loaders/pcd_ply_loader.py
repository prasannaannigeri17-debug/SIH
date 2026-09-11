"""
Loader for point cloud files in .pcd and .ply formats.
"""
import os
import glob
from typing import List
import numpy as np
from data.loaders.base_loader import BasePointCloudLoader, PointCloudFrame


class PCDPLYLoader(BasePointCloudLoader):
    """
    Loads .pcd and .ply point cloud files from a directory or single file.
    """

    def __init__(self, file_or_dir_path: str):
        self.path = os.path.abspath(file_or_dir_path)
        self.files: List[str] = []

        if os.path.isfile(self.path):
            self.files = [self.path]
        elif os.path.isdir(self.path):
            pcd_files = glob.glob(os.path.join(self.path, "*.pcd"))
            ply_files = glob.glob(os.path.join(self.path, "*.ply"))
            self.files = sorted(pcd_files + ply_files)

    def __len__(self) -> int:
        return len(self.files)

    def get_frame(self, index: int) -> PointCloudFrame:
        if index < 0 or index >= len(self.files):
            raise IndexError(f"Index {index} out of range (files: {len(self.files)})")

        file_path = self.files[index]
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".pcd":
            xyz, intensity = self._read_pcd(file_path)
        elif ext == ".ply":
            xyz, intensity = self._read_ply(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        return PointCloudFrame(
            frame_id=index,
            timestamp=index * 0.1,
            points=xyz,
            intensities=intensity,
            metadata={"source_file": file_path}
        )

    def reset(self) -> None:
        pass

    def _read_pcd(self, filepath: str) -> tuple[np.ndarray, np.ndarray]:
        with open(filepath, "r", errors="ignore") as f:
            lines = f.readlines()

        header_end = 0
        data_type = "ascii"
        num_points = 0

        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith("DATA"):
                data_type = line.split()[1].lower()
                header_end = i + 1
                break
            elif line.startswith("POINTS"):
                num_points = int(line.split()[1])

        if data_type == "ascii":
            data_lines = lines[header_end:]
            pts = []
            intensities = []
            for l in data_lines:
                tokens = l.strip().split()
                if len(tokens) >= 3:
                    pts.append([float(tokens[0]), float(tokens[1]), float(tokens[2])])
                    intensities.append(float(tokens[3]) if len(tokens) >= 4 else 0.5)
            return np.array(pts, dtype=np.float32), np.array(intensities, dtype=np.float32)
        else:
            # Fallback placeholder for binary PCD
            return np.zeros((0, 3), dtype=np.float32), np.zeros(0, dtype=np.float32)

    def _read_ply(self, filepath: str) -> tuple[np.ndarray, np.ndarray]:
        with open(filepath, "r", errors="ignore") as f:
            lines = f.readlines()

        header_end = 0
        for i, line in enumerate(lines):
            if line.strip() == "end_header":
                header_end = i + 1
                break

        pts = []
        for l in lines[header_end:]:
            parts = l.strip().split()
            if len(parts) >= 3:
                pts.append([float(parts[0]), float(parts[1]), float(parts[2])])

        xyz = np.array(pts, dtype=np.float32) if pts else np.zeros((0, 3), dtype=np.float32)
        return xyz, np.ones(len(xyz), dtype=np.float32) * 0.5
