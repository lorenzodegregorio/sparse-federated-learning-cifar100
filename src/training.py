# centralized_baseline.py

import os
import gc
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
gc.collect()
torch.cuda.empty_cache()

def get_dataloaders(train_ds, val_ds, test_ds, batch_size=64, num_workers=2, pin_memory=True):
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader, test_loader

def get_latest_checkpoint(directory, prefix='checkpoint_epoch'):
    files = [f for f in os.listdir(directory) if f.startswith(prefix)]
    if not files:
        return None, 0
    epochs = [int(re.search(r'\d+', f).group()) for f in files]
    last = max(epochs)
    path = os.path.join(directory, f'{prefix}{last}.pth')
    return path, last

def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, train_acc, val_acc, scheduler=None, save_dir='./'):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_acc': train_acc,
        'val_acc': val_acc,
    }
    if scheduler:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    filename = os.path.join(save_dir, f"checkpoint_epoch{epoch}.pth")
    torch.save(checkpoint, filename)
    print(f"💾 Checkpoint saved to: {filename}")

def load_checkpoint(path, model, optimizer, scheduler=None, device='cpu'):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['epoch'], checkpoint['train_loss'], checkpoint['val_loss'], checkpoint['train_acc'], checkpoint['val_acc']


def train_centralized_model(
    model, train_loader, val_loader, test_loader,
    num_epochs=70, lr=0.01,
    save_dir=None, checkpoint_path=None,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    start_epoch = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"✅ Found checkpoint: {checkpoint_path}")
        start_epoch, *_ = load_checkpoint(checkpoint_path, model, optimizer, scheduler, device=device)
        print(f"📦 Resuming from epoch {start_epoch}")
    else:
        print("🚀 Starting training from scratch.")

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(start_epoch, num_epochs):
        print(f"\n🔁 Epoch {epoch + 1}/{num_epochs}")
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        train_loss = total_loss / len(train_loader)
        train_acc = 100. * correct / total

        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

        val_loss /= len(val_loader)
        val_acc = 100. * correct / total

        print(f"📊 Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_checkpoint(model, optimizer, epoch + 1, train_loss, val_loss, train_acc, val_acc, scheduler, save_dir=save_dir)

        scheduler.step()
        gc.collect()
        torch.cuda.empty_cache()

    print("✅ Training finished.")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    test_acc = 100. * correct / total
    print(f"\n🏁 Final Test Accuracy: {test_acc:.2f}%")

    # Plotting
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Acc')
    plt.plot(val_accuracies, label='Val Acc')
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()