# Resnet Finetune

This document describes the complete workflow from downloading the Caltech-101 dataset, organizing it under `data/`, processing the data splits, training the model, and evaluating on the test set. It also provides a brief overview of each key script/file in the repository.

------

## 1. Dataset Download & Extraction

1. **Download** the Caltech-101 archive:

   ```bash
   wget http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz -P data/
   ```

2. **Extract** into the `data/` directory:

   ```bash
   cd data
   tar -xzvf 101_ObjectCategories.tar.gz
   cd ..
   ```

3. **Resulting Structure**:

   ```
   data/
   ├── 101_ObjectCategories/       # Raw image folders per class
   ├── 101_ObjectCategories.tar.gz # Downloaded archive
   └── splits/                     # Auto-generated splits (see below)
   ```

------

## 2. Data Splitting & Processing

We stratify the dataset into train/val/test splits to ensure balanced per-class samples.

- **Script**: `data_split.py`

  - Reads all images under `data/101_ObjectCategories/`.
  - For each of the 101 classes:
    - Randomly sample **30** images → **train**
    - Randomly sample **10** images → **validation**
    - Remaining images → **test**
  - Outputs JSON files under `data/splits/`:
    - `train.json`
    - `val.json`
    - `test.json`

- **Usage**:

  ```bash
  python data_split.py --root data/101_ObjectCategories 
  ```

------

## 3. Dataset Loader & Augmentation

- **Script**: `dataset.py`
  - Defines `get_loaders(batch_size, num_workers)`
  - Reads split JSONs and image files
  - Applies transforms:
    - **Train**: RandomResizedCrop(224), RandomHorizontalFlip, Normalize(ImageNet stats)
    - **Val/Test**: Resize(256) → CenterCrop(224) → Normalize
  - Returns PyTorch `DataLoader`s for train, val, and test.

------

## 4. Model Definition

- **Script**: `model.py`
  - Builds a ResNet-18 backbone pretrained on ImageNet.
  - Replaces the final fully connected layer with `nn.Linear(in_features, num_classes)`.
  - Returns the constructed `torch.nn.Module`.

------

## 5. Training Pipeline

- **Script**: `train.py`

  - Loads data loaders and model.
  - Parses hyperparameters (learning rates, batch size, epochs) via `argparse`.
  - Configures optimizer with two parameter groups:
    - Backbone parameters → `lr_backbone`
    - Classifier head parameters → `lr_fc`
  - Uses SGD (momentum=0.9, weight_decay=1e-4).
  - LR scheduler: StepLR (step_size=20, gamma=0.1).
  - Iterates over epochs:
    1. **Train**: forward → loss → backward → optimizer step.
    2. **Validate** at epoch end.
    3. **Log** metrics to TensorBoard (`Loss/train`, `Acc/train`, `Loss/val`, `Acc/val`).
    4. **Checkpoint**: save `epoch_{epoch}.pth` and update `best.pth` if val accuracy improves.

- **Usage**:

  ```bash
  python train.py \
  ```

- **Outputs**:

  ```
  checkpoints/
  ├── epoch_1.pth
  ├── epoch_2.pth
  └── best.pth
  logs/                 # TensorBoard events files
  ```

------

## 6. Evaluation

- **Script**: `test.py`

  - Loads the `resnet18_finetune_best.pth` checkpoint.

  - Constructs the model and loads state dict.

  - Creates the test `DataLoader`.

  - Runs inference in `no_grad` mode to compute average loss & accuracy.

  - Prints results:

    ```
    Test Loss: 0.5217  Test Accuracy: 0.8730
    ```

- **Usage**:

  ```bash
  python test.py
  ```

------

## 7. File Summary

| File            | Description                                          |
| --------------- | ---------------------------------------------------- |
| `config.py`     | Defines constants (paths, default hyperparameters).  |
| `data_split.py` | Generates train/val/test JSON splits.                |
| `dataset.py`    | Implements PyTorch `Dataset` and `DataLoader` logic. |
| `model.py`      | Builds and returns the ResNet-18 based model.        |
| `train.py`      | Main training loop, logging, and checkpointing.      |
| `test.py`       | Evaluation script for computing test metrics.        |

------

End of workflow documentation.