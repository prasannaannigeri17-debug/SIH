"""Train the optional PointNet semantic model on labeled project frames."""
import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from data.loaders import create_loader
from perception.pointnet_model import NUM_CLASSES, PointNetSemanticModel


class ProceduralLidarDataset(Dataset):
    def __init__(self, frames: int, points_per_sample: int):
        self.loader = create_loader(num_demo_frames=frames)
        self.points_per_sample = points_per_sample

    def __len__(self):
        return len(self.loader)

    def __getitem__(self, index):
        frame = self.loader.get_frame(index)
        points = frame.points
        labels = frame.semantic_labels
        generator = np.random.default_rng(index)
        if len(points) >= self.points_per_sample:
            selected = generator.choice(len(points), self.points_per_sample, replace=False)
        else:
            selected = generator.choice(len(points), self.points_per_sample, replace=True)
        features = np.column_stack((points[selected], frame.intensities[selected]))
        return torch.from_numpy(features.T.astype(np.float32)), torch.from_numpy(labels[selected].astype(np.int64))


def main():
    parser = argparse.ArgumentParser(description="Train PointNet LiDAR semantic segmentation")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--points", type=int, default=4096)
    parser.add_argument("--output", default="checkpoints/pointnet_semantic.pt")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ProceduralLidarDataset(args.frames, args.points)
    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    model = PointNetSemanticModel(input_features=4, num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"epoch {epoch + 1}/{args.epochs} loss={running_loss / max(1, len(loader)):.4f}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_features": 4,
        "num_classes": NUM_CLASSES,
    }, args.output)
    print(f"saved checkpoint: {args.output}")


if __name__ == "__main__":
    main()