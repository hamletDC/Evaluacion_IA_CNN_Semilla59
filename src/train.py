import argparse
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import CLASSES, get_loaders, set_seed
from model import CIFAR10CNN, validate_shapes

def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    loss_sum, correct, total = 0.0, 0, 0
    predictions, targets = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if training:
            optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                loss.backward()
                optimizer.step()
        loss_sum += loss.item() * len(y)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += len(y)
        if not training:
            predictions.extend(pred.cpu().tolist())
            targets.extend(y.cpu().tolist())
    return loss_sum / total, correct / total, targets, predictions

def main(epochs=10, batch_size=128, lr=0.001):
    set_seed()
    out = Path(__file__).resolve().parents[1] / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader, test_loader = get_loaders(batch_size=batch_size)
    validate_shapes()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Dispositivo:", device)
    model = CIFAR10CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val = float("inf")
    for epoch in range(epochs):
        tl, ta, _, _ = run_epoch(model, train_loader, criterion, device, optimizer)
        vl, va, _, _ = run_epoch(model, val_loader, criterion, device)
        for key, val in zip(history, (tl, vl, ta, va)):
            history[key].append(val)
        print(f"Epoch {epoch+1}/{epochs} | train loss={tl:.4f} acc={ta:.4f} | val loss={vl:.4f} acc={va:.4f}")
        if vl < best_val:
            best_val = vl
            torch.save(model.state_dict(), out / "best_model.pth")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Validación")
    axes[0].set(title="Loss", xlabel="Época")
    axes[0].legend()
    axes[1].plot(history["train_acc"], label="Train")
    axes[1].plot(history["val_acc"], label="Validación")
    axes[1].set(title="Accuracy", xlabel="Época", ylim=(0, 1))
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out / "loss_accuracy.png", dpi=160)
    plt.close(fig)
    model.load_state_dict(torch.load(out / "best_model.pth", map_location=device, weights_only=True))
    test_loss, test_acc, true, pred = run_epoch(model, test_loader, criterion, device)
    print(f"TEST | loss={test_loss:.4f} accuracy={test_acc:.4f}")
    print(classification_report(true, pred, target_names=CLASSES))
    cm = confusion_matrix(true, pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=CLASSES)
    fig, ax = plt.subplots(figsize=(11, 9))
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False, values_format="d")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=160)
    plt.close(fig)
    import json
    (out / "metrics.json").write_text(json.dumps({"seed": 59, "epochs": epochs, "history": history, "test_loss": test_loss, "test_accuracy": test_acc}, indent=2), encoding="utf-8")
    print("Gráficos y métricas guardados en outputs/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()
    main(args.epochs, args.batch_size, args.lr)
