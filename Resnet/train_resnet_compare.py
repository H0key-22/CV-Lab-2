import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torchvision import models

from config import (
    NUM_WORKERS,
    WEIGHT_DECAY,
    MOMENTUM,
    STEP_SIZE,
    GAMMA,
    ROOT_DIR,
    SPLIT_JSON
)
from dataset import get_loaders


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        preds = outputs.argmax(dim=1)
        running_loss += loss.item() * x.size(0)
        running_corrects += (preds == y).sum().item()
    return running_loss / len(loader.dataset), running_corrects / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            preds = outputs.argmax(dim=1)
            running_loss += loss.item() * x.size(0)
            running_corrects += (preds == y).sum().item()
    return running_loss / len(loader.dataset), running_corrects / len(loader.dataset)


def run_experiment(pretrained, params, device):
    # Unpack hyperparameters
    epochs = params['NUM_EPOCHS']
    lr_fc = params['LR_FC']
    lr_back = params['LR_BACKBONE']
    batch_size = params['BATCH_SIZE']

    tag = f"pt{pretrained}_fc{lr_fc}_back{lr_back}_bs{batch_size}_ep{epochs}"
    # TensorBoard writer for event file
    writer = SummaryWriter(log_dir=os.path.join(ROOT_DIR, 'runs', tag))

    train_loader, val_loader, _, num_classes = get_loaders(
        batch_size=batch_size
    )

    model = models.resnet18(pretrained=pretrained)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD([
        {'params': model.fc.parameters(), 'lr': lr_fc},
        {'params': [p for n, p in model.named_parameters() if not n.startswith('fc.')], 'lr': lr_back}
    ], momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        # Record metrics
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        # Log to TensorBoard event file
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val',   val_loss,   epoch)
        writer.add_scalar('Acc/train',  train_acc,  epoch)
        writer.add_scalar('Acc/val',    val_acc,    epoch)

        print(f"[pt={pretrained}] Epoch {epoch}/{epochs} "
              f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    writer.close()
    return history


def plot_histories(histories, metric, title):
    plt.figure()
    for label, hist in histories.items():
        plt.plot(hist[metric], label=label)
    plt.xlabel('Epoch')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.title(title)
    plt.legend()
    plt.show()


def main():
    assert os.path.exists(SPLIT_JSON), "Missing data split file"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    fixed_params = {
        'NUM_EPOCHS': 50,
        'LR_FC': 0.01,
        'LR_BACKBONE': 0.001,
        'BATCH_SIZE': 32
    }

    histories = {}
    for pretrained in [False, True]:
        print(f"\nRunning with pretrained={pretrained}")
        hist = run_experiment(pretrained, fixed_params, device)
        histories[f"pretrained={pretrained}"] = hist

    # 保存 JSON 历史数据
    os.makedirs(ROOT_DIR, exist_ok=True)
    history_path = os.path.join(ROOT_DIR, 'histories.json')
    with open(history_path, 'w') as f:
        json.dump(histories, f)
    print(f"Training histories saved to {history_path}")

    # 可视化对比
    plot_histories(histories, 'train_loss', 'Training Loss Comparison')
    plot_histories(histories, 'val_loss', 'Validation Loss Comparison')
    plot_histories(histories, 'train_acc', 'Training Accuracy Comparison')
    plot_histories(histories, 'val_acc', 'Validation Accuracy Comparison')

if __name__ == '__main__':
    main()
