import os, random, json
from config import ROOT_DIR, SPLIT_JSON

NUM_TRAIN = 30
NUM_VAL   = 10

def make_splits(root_dir, num_train, num_val):
    train_paths = []
    val_paths   = []
    test_paths  = []

    for cls in sorted(os.listdir(root_dir)):
        cls_folder = os.path.join(root_dir, cls)
        if not os.path.isdir(cls_folder):
            continue
        imgs = [
            os.path.join(cls_folder, f)
            for f in os.listdir(cls_folder)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        random.shuffle(imgs)
        train_paths += imgs[:num_train]
        val_paths   += imgs[num_train:num_train+num_val]
        test_paths  += imgs[num_train+num_val:]
    return {'train': train_paths, 'val': val_paths, 'test': test_paths}

if __name__ == '__main__':
    splits = make_splits(ROOT_DIR, NUM_TRAIN, NUM_VAL)
    with open(SPLIT_JSON, 'w', encoding='utf-8') as f:
        json.dump(splits, f, ensure_ascii=False, indent=2)
    print(f"train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")
