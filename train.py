import time

import lightning as L
from lightning import Fabric

import torch
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader

import torchmetrics

from torchvision import transforms
from torchvision.models import vit_l_16, ViT_L_16_Weights
from watermark import watermark

from torchvision.datasets import CIFAR10

import random
import numpy as np


# Hyperparams
batch_size = 32
num_workers = 16


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(num_epochs, model, optimizer, train_loader, val_loader, device):
    for epoch in range(num_epochs):
        train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(device)
        model.train()
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            preds = model(inputs)
            loss = cross_entropy(preds, targets)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            if not batch_idx % 300:
                print(f"Epoch: {epoch+1:04d}/{num_epochs:04d} | Batch {batch_idx:04d}/{len(train_loader):04d} | Loss: {loss:.4f}")

            model.eval()
            with torch.no_grad():
                predicted_labels = torch.argmax(preds, 1)
                train_acc.update(predicted_labels, targets)
        
        # Log results
        with torch.no_grad():
            val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(device)

            for (features, targets) in val_loader:
                features = features.to(device)
                targets = targets.to(device)
                outputs = model(features)
                predicted_labels = torch.argmax(outputs, 1)
                val_acc.update(predicted_labels, targets)

            print(f"Epoch: {epoch+1:04d}/{num_epochs:04d} | Train acc.: {train_acc.compute()*100:.2f}% | Val acc.: {val_acc.compute()*100:.2f}%")
            train_acc.reset(), val_acc.reset()


if __name__ == "__main__":
    print(watermark(packages="torch, lightning"), python=True)
    print("Torch CUDA available?", torch.cuda.is_available())
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    train_set = CIFAR10(root="./data", train=True, download=True, transform=transform)
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    val_set = CIFAR10(root="./data", train=False, download=True, transform=transform)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)

    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    model = vit_l_16(weights=ViT_L_16_Weights.IMAGENET1K_V1)
    model.heads.head = torch.nn.Linear(in_features=1024, out_features=len(classes))

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

    start = time.time()
    train(
        num_epochs=1,
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device
    )

    end = time.time()
    elapsed = end-start
    print(f"Time elapsed {elapsed/60:.2f} min")
    print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")

    """
    with torch.no_grad():
        model.eval()
        test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(device)

        for (features, targets) in test_loader:
            features = features.to(device)
            targets = targets.to(device)
            outputs = model(features)
            predicted_labels = torch.argmax(outputs, 1)
            test_acc.update(predicted_labels, targets)

    print(f"Test accuracy {test_acc.compute()*100:.2f}%")
    """