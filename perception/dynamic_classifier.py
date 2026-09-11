"""
Static vs Dynamic Object Classifier and 3D Bounding Box Clusterer.
"""
from typing import List, Dict, Any, Tuple
import numpy as np
from scipy.spatial import cKDTree


class DynamicClassifier:
    """
    Classifies point semantics into static vs dynamic categories and performs
    spatial Euclidean clustering on dynamic objects.
    """

    # Semantic classes known to be potentially dynamic
    DYNAMIC_CLASSES = {4, 5, 6}  # Vehicle, Pedestrian, Cyclist

    def __init__(self, config: Dict[str, Any]):
        p_cfg = config.get("perception", {})
        self.cluster_eps = float(p_cfg.get("cluster_eps", 0.6))
        self.cluster_min_samples = int(p_cfg.get("cluster_min_samples", 5))

    def classify_dynamics(
        self,
        points: np.ndarray,
        semantic_labels: np.ndarray
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Labels points as static (False) or dynamic (True).
        Extracts 3D bounding boxes for dynamic clusters.
        """
        N = len(points)
        if N == 0:
            return np.zeros(0, dtype=bool), []

        # Vectorized check of dynamic class membership
        is_dynamic = np.isin(semantic_labels, list(self.DYNAMIC_CLASSES))

        # Cluster dynamic obstacle points to extract bounding boxes
        bounding_boxes = []
        dyn_points = points[is_dynamic]
        dyn_sems = semantic_labels[is_dynamic]

        if len(dyn_points) >= self.cluster_min_samples:
            clusters = self._euclidean_clustering(dyn_points, eps=self.cluster_eps, min_pts=self.cluster_min_samples)
            for cluster_indices in clusters:
                c_pts = dyn_points[cluster_indices]
                c_sem = dyn_sems[cluster_indices]

                min_b = np.min(c_pts, axis=0)
                max_b = np.max(c_pts, axis=0)
                center = (min_b + max_b) / 2.0
                size = max_b - min_b

                # Dominant class in cluster
                u_cls, counts = np.unique(c_sem, return_counts=True)
                dominant_cls = int(u_cls[np.argmax(counts)])

                bounding_boxes.append({
                    "center": [float(center[0]), float(center[1]), float(center[2])],
                    "size": [float(max(size[0], 0.3)), float(max(size[1], 0.3)), float(max(size[2], 0.3))],
                    "class_id": dominant_cls,
                    "num_points": len(cluster_indices),
                    "is_dynamic": True
                })

        return is_dynamic, bounding_boxes

    def _euclidean_clustering(self, points: np.ndarray, eps: float = 0.8, min_pts: int = 5) -> List[np.ndarray]:
        """
        Fast KD-Tree spatial Euclidean cluster extraction.
        """
        if len(points) == 0:
            return []

        tree = cKDTree(points[:, :3])
        visited = np.zeros(len(points), dtype=bool)
        clusters = []

        for i in range(len(points)):
            if visited[i]:
                continue
            neighbors = tree.query_ball_point(points[i, :3], r=eps)
            if len(neighbors) < min_pts:
                continue

            current_cluster = []
            queue = list(neighbors)
            visited[i] = True

            while queue:
                idx = queue.pop()
                if not visited[idx]:
                    visited[idx] = True
                    current_cluster.append(idx)
                    sub_neighbors = tree.query_ball_point(points[idx, :3], r=eps)
                    if len(sub_neighbors) >= min_pts:
                        queue.extend([sn for sn in sub_neighbors if not visited[sn]])
                else:
                    if idx not in current_cluster:
                        current_cluster.append(idx)

            if len(current_cluster) >= min_pts:
                clusters.append(np.array(current_cluster, dtype=np.int32))

        return clusters
