import os

# 数据根目录，修改为你本地路径
ROOT_DIR = './data/caltech-101/101_ObjectCategories'

# 切分结果存放
SPLIT_JSON = 'splits/splits.json'

# 训练参数
BATCH_SIZE    = 32
NUM_EPOCHS    = 60
NUM_WORKERS   = 4
LR_FC         = 1e-3
LR_BACKBONE   = 1e-4
WEIGHT_DECAY  = 1e-4
MOMENTUM      = 0.9
STEP_SIZE     = 20
GAMMA         = 0.1

# 超参数网格：脚本会遍历所有组合
HYPERPARAM_GRID = {
    'NUM_EPOCHS':    [60, 80],
    'LR_FC':         [1e-2, 1e-3],
    'LR_BACKBONE':   [5e-5, 1e-4],
    'BATCH_SIZE':    [16, 32],
}

# 结果存放
CHECKPOINT_DIR = 'checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(SPLIT_JSON), exist_ok=True)