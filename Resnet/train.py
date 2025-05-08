import os
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

from config import (
    HYPERPARAM_GRID,
    NUM_WORKERS,
    WEIGHT_DECAY,
    MOMENTUM,
    STEP_SIZE,
    GAMMA,
    ROOT_DIR,
    SPLIT_JSON,
    CHECKPOINT_DIR
)
from dataset import get_loaders
from model import build_model


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


def run_experiment(params):
    # Unpack hyperparameters
    epochs = params['NUM_EPOCHS']
    lr_fc = params['LR_FC']
    lr_back = params['LR_BACKBONE']
    batch_size = params['BATCH_SIZE']

    # Setup TensorBoard
    tag = f"fc{lr_fc}_back{lr_back}_bs{batch_size}_ep{epochs}"
    writer = SummaryWriter(log_dir=os.path.join('runs', tag))

    # Get data loaders
    train_loader, val_loader, test_loader, num_classes = get_loaders(
        batch_size=batch_size
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()

    # Optimizer and scheduler
    fc_params = list(model.fc.parameters())
    backbone_params = [p for n, p in model.named_parameters() if not n.startswith('fc.')]
    optimizer = optim.SGD([
        {'params': fc_params, 'lr': lr_fc},
        {'params': backbone_params, 'lr': lr_back}
    ], momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    best_val_acc = 0.0
    # Training loop
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"[{tag}] Epoch {epoch}/{epochs} "
              f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        # Log to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Acc/train', train_acc, epoch)
        writer.add_scalar('Acc/val', val_acc, epoch)

        # Every 5 epochs, save checkpoint and update best
        if epoch % 5 == 0:
            ckpt_name = f"{tag}_epoch{epoch}.pth"
            ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt_name)
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint: {ckpt_name}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_path = os.path.join(CHECKPOINT_DIR, 'best_resnet18.pth')
                torch.save(model.state_dict(), best_path)
                print(f"New best for {tag}: {best_val_acc:.4f}, saved to best_resnet18.pth")

    writer.close()
    return best_val_acc


def main():
    # Ensure splits generated
    assert os.path.exists(SPLIT_JSON)

    best_overall = 0.0
    best_params = None

    # Iterate over all hyperparameter combinations
    keys = list(HYPERPARAM_GRID.keys())
    for combo in itertools.product(*HYPERPARAM_GRID.values()):
        params = dict(zip(keys, combo))
        print("\nRunning experiment with params:", params)
        val_acc = run_experiment(params)
        if val_acc > best_overall:
            best_overall = val_acc
            best_params = params.copy()

    print("\n=== Hyperparameter search complete ===")
    print("Best params:", best_params)
    print(f"Best validation accuracy: {best_overall:.4f}")

if __name__ == '__main__':
    main()