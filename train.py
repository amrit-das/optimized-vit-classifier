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
batch_size = 36
num_workers = 16
seed = 1
num_epochs = 15


def train(num_epochs, model, optimizer, train_loader, val_loader):
    for epoch in range(num_epochs):
        train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10)
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            inputs, targets = batch

            preds = model(inputs)
            loss = cross_entropy(preds, targets)

            optimizer.zero_grad()
            fabric.backward(loss)

            optimizer.step()

            if not batch_idx % 50:
                print(
                    f"Epoch: {epoch+1:04d}/{num_epochs:04d} | Batch {batch_idx:04d}/{len(train_loader):04d} | Loss: {loss:.4f}"
                )

            model.eval()
            with torch.no_grad():
                predicted_labels = torch.argmax(preds, 1)
                train_acc.update(predicted_labels, targets)

        # Log results
        with torch.no_grad():
            val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10)

            for batch in val_loader:
                inputs, targets = batch
                outputs = model(inputs)
                predicted_labels = torch.argmax(outputs, 1)
                val_acc.update(predicted_labels, targets)

            print(
                f"Epoch: {epoch+1:04d}/{num_epochs:04d} | Train acc.: {train_acc.compute()*100:.2f}% | Val acc.: {val_acc.compute()*100:.2f}%"
            )
            train_acc.reset(), val_acc.reset()


if __name__ == "__main__":
    print(watermark(packages="torch, lightning", python=True))
    print("Torch CUDA available?", torch.cuda.is_available())
    fabric = Fabric(accelerator="cuda", devices=1, precision="bf16-true")
    fabric.launch()
    fabric.seed_everything(seed + fabric.global_rank)

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_set = CIFAR10(root="./data", train=True, download=True, transform=transform)
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    val_set = CIFAR10(root="./data", train=False, download=True, transform=transform)
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    train_loader = fabric.setup_dataloaders(train_loader)
    val_loader = fabric.setup_dataloaders(val_loader)

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

    with fabric.init_module():
        model = vit_l_16(weights=ViT_L_16_Weights.IMAGENET1K_V1)
        model.heads.head = torch.nn.Linear(in_features=1024, out_features=len(classes))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    num_steps = num_epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)

    model, optimizer = fabric.setup(model, optimizer)

    start = time.time()
    train(
        num_epochs=num_epochs,
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    end = time.time()
    elapsed = end - start
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
