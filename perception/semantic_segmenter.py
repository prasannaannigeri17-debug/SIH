"""
Semantic Segmentation Perception Engine.
Supports Hybrid, Geometric, Deep-Learning, and Dataset Replay modes.
"""
from typing import Dict, Any, Tuple
import numpy as np
from data.loaders.base_loader import PointCloudFrame
from preprocessing.ground_detector import GroundDetector
from perception.terrain_analyzer import TerrainAnalyzer
from perception.dynamic_classifier import DynamicClassifier
from perception.pointnet_model import load_pointnet_checkpoint


class SemanticSegmenter:
    """
    Perception engine producing multi-class point annotations and confidences.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        p_cfg = config.get("perception", {})
        self.mode = p_cfg.get("mode", "hybrid")
        self.confidence_threshold = float(p_cfg.get("confidence_threshold", 0.55))
        self.model = None
        self.model_device = None

        self.ground_detector = GroundDetector(config)
        self.terrain_analyzer = TerrainAnalyzer(config)
        self.dynamic_classifier = DynamicClassifier(config)

        if self.mode == "deep_learning":
            model_path = p_cfg.get("model_path")
            if not model_path:
                raise ValueError(
                    "perception.mode='deep_learning' requires perception.model_path"
                )
            self.model, self.model_device = load_pointnet_checkpoint(
                model_path,
                input_features=int(p_cfg.get("input_features", 4)),
                num_classes=int(p_cfg.get("num_classes", 13)),
                device=p_cfg.get("device"),
            )

    def segment(self, frame: PointCloudFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
        """
        Runs semantic segmentation and static/dynamic classification on the frame.
        Returns:
            semantic_labels: (N,) uint32
            confidences: (N,) float32
            is_dynamic: (N,) bool
            bounding_boxes: list of 3D obstacle clusters
        """
        pts = frame.points
        N = len(pts)
        if N == 0:
            return np.zeros(0, dtype=np.uint32), np.zeros(0, dtype=np.float32), np.zeros(0, dtype=bool), []

        # 1. Check if ground-truth labels exist and mode allows using them
        if frame.semantic_labels is not None and len(frame.semantic_labels) == N and self.mode in ["ground_truth", "hybrid"]:
            semantic_labels = frame.semantic_labels.copy()
            confidences = np.full(N, 0.95, dtype=np.float32)
        elif self.mode == "deep_learning":
            semantic_labels, confidences = self._deep_learning_segmentation(pts, frame.intensities)
        else:
            # 2. Geometric / Rule-based Semantic Inference Engine
            semantic_labels, confidences = self._geometric_segmentation(pts, frame.intensities)

        # 3. Dynamic Classification and 3D Bounding Box Extraction
        is_dynamic, bboxes = self.dynamic_classifier.classify_dynamics(pts, semantic_labels)

        return semantic_labels, confidences, is_dynamic, bboxes

    def _deep_learning_segmentation(
        self, points: np.ndarray, intensities: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        import torch

        features = np.column_stack((points, intensities)).astype(np.float32)
        tensor = torch.from_numpy(features.T).unsqueeze(0).to(self.model_device)
        with torch.inference_mode():
            probabilities = torch.softmax(self.model(tensor), dim=1)[0]
        confidences, semantic_labels = probabilities.max(dim=0)
        labels = semantic_labels.cpu().numpy().astype(np.uint32)
        confidence_values = confidences.cpu().numpy().astype(np.float32)
        labels[confidence_values < self.confidence_threshold] = 0
        return labels, confidence_values

    def _geometric_segmentation(self, points: np.ndarray, intensities: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        N = len(points)
        labels = np.zeros(N, dtype=np.uint32)
        confidences = np.full(N, 0.85, dtype=np.float32)

        # Ground detection
        ground_mask, local_gz, height_above_ground = self.ground_detector.segment_ground(points)

        # Terrain analysis
        drivable_mask, terrain_classes = self.terrain_analyzer.analyze_terrain(points, ground_mask, height_above_ground)

        # Assign ground terrain labels
        labels[ground_mask] = terrain_classes[ground_mask]
        # Default flat ground is drivable road (1)
        is_road = ground_mask & (terrain_classes == 1)
        labels[is_road] = 1

        # Non-ground points classification by height and lateral distance
        obs_mask = ~ground_mask
        obs_z = height_above_ground[obs_mask]
        obs_y = np.abs(points[obs_mask, 1])

        obs_labels = np.zeros(np.sum(obs_mask), dtype=np.uint32)

        # Roadside objects vs in-lane obstacles
        in_road_lane = obs_y < 3.6
        tall_object = obs_z > 3.0
        mid_object = (obs_z >= 0.8) & (obs_z <= 2.2)
        low_object = (obs_z >= 0.2) & (obs_z < 0.8)

        # Vehicles: in/near road lane, mid-height
        is_car = in_road_lane & mid_object
        obs_labels[is_car] = 4  # vehicle

        # Pedestrians / Cyclists: sidewalk offset, human height
        is_ped = (obs_y >= 3.6) & (obs_y <= 6.0) & (obs_z >= 0.5) & (obs_z <= 2.0)
        obs_labels[is_ped] = 5  # pedestrian

        # Poles: narrow vertical structures
        is_pole = (obs_y >= 4.0) & (obs_y <= 6.5) & (obs_z > 2.5) & (obs_z <= 6.0)
        obs_labels[is_pole] = 9  # pole_obstacle

        # Buildings / Walls: far lateral distance, tall height
        is_building = (obs_y > 8.0) & tall_object
        obs_labels[is_building] = 7  # building

        # Vegetation: mid/tall outside roadway
        is_veg = (obs_y > 6.0) & ~is_building & ~is_pole
        obs_labels[is_veg] = 8  # vegetation

        labels[obs_mask] = obs_labels

        return labels, confidences
