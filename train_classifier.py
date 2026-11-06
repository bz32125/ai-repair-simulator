"""Train the photo classifier with transfer learning (MobileNetV3-Small, pretrained on ImageNet).

Dataset layout (30-50 photos per object, different angles/light/backgrounds):
    dataset/
      door_handle/   *.jpg
      cabinet_door/  *.jpg
      chair_table/   *.jpg
      drawer/        *.jpg

Run (on a laptop CPU or Google Colab):
    pip install torch torchvision
    python train_classifier.py --data dataset --epochs 10

Only the last layer is trained at first (feature extraction); the pretrained layers
already know edges, textures and shapes. Output: models/classifier.pt
NOTE: not yet run, because the dataset does not exist yet (Milestone 2).
"""
import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
EVAL_TF = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                              transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
TRAIN_TF = transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.6, 1.0)), transforms.RandomHorizontalFlip(),
                               transforms.ColorJitter(0.3, 0.3, 0.3), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def build_model(n_classes):
    m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    for p in m.features.parameters():
        p.requires_grad = False  # keep the pretrained features
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, n_classes)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--val", type=float, default=0.2)
    args = ap.parse_args()

    full = datasets.ImageFolder(args.data)
    n_val = max(1, int(len(full) * args.val))
    train_idx, val_idx = random_split(range(len(full)), [len(full) - n_val, n_val], torch.Generator().manual_seed(42))
    train = torch.utils.data.Subset(datasets.ImageFolder(args.data, TRAIN_TF), list(train_idx))
    val = torch.utils.data.Subset(datasets.ImageFolder(args.data, EVAL_TF), list(val_idx))
    tl, vl = DataLoader(train, batch_size=16, shuffle=True), DataLoader(val, batch_size=16)

    model = build_model(len(full.classes))
    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(args.epochs):
        model.train()
        for x, y in tl:
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            opt.step()
        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in vl:
                correct += (model(x).argmax(1) == y).sum().item()
        print(f"epoch {epoch + 1}: validation accuracy {correct / len(val):.0%}")

    Path("models").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "models/classifier.pt")
    Path("models/classes.json").write_text(json.dumps(full.classes))
    print("saved models/classifier.pt")


if __name__ == "__main__":
    main()
