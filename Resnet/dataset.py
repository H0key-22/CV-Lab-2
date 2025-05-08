import os, json
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from config import ROOT_DIR, SPLIT_JSON, BATCH_SIZE, NUM_WORKERS


train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

class Caltech101Dataset(Dataset):
    def __init__(self, paths, cls2idx, transform=None):
        self.paths = paths
        self.cls2idx = cls2idx
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        p = self.paths[i]
        label = self.cls2idx[os.path.basename(os.path.dirname(p))]
        img = Image.open(p).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label
        

def get_loaders(batch_size=None):
    # 如果外部没传，就用 config.py 里的默认
    if batch_size is None:
        batch_size = BATCH_SIZE

    with open(SPLIT_JSON, 'r') as f:
        splits = json.load(f)
    classes = sorted(os.listdir(ROOT_DIR))
    cls2idx = {cls: i for i, cls in enumerate(classes)}

    # 分别用 batch_size 构造 DataLoader
    train_ds = Caltech101Dataset(splits['train'], cls2idx, transform=train_transform)
    val_ds   = Caltech101Dataset(splits['val'],   cls2idx, transform=val_transform)
    test_ds  = Caltech101Dataset(splits['test'],  cls2idx, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
    return train_loader, val_loader, test_loader, len(classes)

