import os
import argparse
import torch
import torch.nn as nn

from config import CHECKPOINT_DIR
from dataset import get_loaders
from model import build_model


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
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = running_corrects / len(loader.dataset)
    return epoch_loss, epoch_acc


def main():
    parser = argparse.ArgumentParser(description='Evaluate Caltech-101 model')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=os.path.join(CHECKPOINT_DIR, 'epoch_40_resnet18.pth'),
        help='Path to saved model weights'
    )
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, _, test_loader, num_classes = get_loaders()

    # 构建模型并加载权重
    model = build_model(num_classes)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Test Loss: {test_loss:.4f}  Test Accuracy: {test_acc:.4f}")

if __name__ == '__main__':
    main()