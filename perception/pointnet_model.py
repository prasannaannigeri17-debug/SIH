"""Small PointNet-style semantic segmentation model for LiDAR points."""
from typing import Any, Dict, Optional

try:
    import torch
    from torch import Tensor, nn
except ImportError:  # Keep geometric/hybrid modes usable without PyTorch.
    torch = None
    Tensor = Any
    nn = None


NUM_CLASSES = 13


if nn is not None:
    class PointNetSemanticModel(nn.Module):
        """Point-wise classifier with a global scene feature."""

        def __init__(self, input_features: int = 4, num_classes: int = NUM_CLASSES):
            super().__init__()
            self.point_encoder = nn.Sequential(
                nn.Conv1d(input_features, 64, 1),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.Conv1d(64, 128, 1),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
            )
            self.classifier = nn.Sequential(
                nn.Conv1d(256, 128, 1),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Conv1d(128, num_classes, 1),
            )

        def forward(self, points: Tensor) -> Tensor:
            encoded = self.point_encoder(points)
            global_feature = encoded.max(dim=2, keepdim=True).values
            global_feature = global_feature.expand(-1, -1, encoded.shape[2])
            return self.classifier(torch.cat((encoded, global_feature), dim=1))
else:
    class PointNetSemanticModel:
        """Placeholder that gives a useful error when PyTorch is unavailable."""

        def __init__(self, *args: Any, **kwargs: Any):
            raise ImportError(
                "PyTorch is required for perception.mode='deep_learning'. "
                "Install dependencies with: pip install -r requirements.txt"
            )


def load_pointnet_checkpoint(
    checkpoint_path: str,
    input_features: int = 4,
    num_classes: int = NUM_CLASSES,
    device: Optional[str] = None,
) -> Any:
    """Load a trained PointNet checkpoint and return an eval-mode model."""
    if torch is None:
        raise ImportError(
            "PyTorch is required for perception.mode='deep_learning'. "
            "Install dependencies with: pip install -r requirements.txt"
        )

    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint: Dict[str, Any] = torch.load(checkpoint_path, map_location=selected_device)
    model = PointNetSemanticModel(
        input_features=int(checkpoint.get("input_features", input_features)),
        num_classes=int(checkpoint.get("num_classes", num_classes)),
    )
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(selected_device)
    model.eval()
    return model, selected_device