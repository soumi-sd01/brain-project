
import os
import sys
import yaml
import torch

from google.colab import drive

# 1. Mount Google Drive
if not os.path.isdir("/content/drive/MyDrive"):
    drive.mount("/content/drive")

# 2. Restore project paths
PROJECT_DIR = "/content/drive/MyDrive/conformal-mri-missing-modalities"
REPO_DIR = os.path.join(PROJECT_DIR, "repo")
CONFIG_PATH = os.path.join(REPO_DIR, "configs", "config.yaml")

# 3. Add repository to Python path
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

# 4. Load configuration
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

DATA_DIR = os.path.join(REPO_DIR, "data")
RAW_DIR = os.path.join(REPO_DIR, config["data"]["root_dir"])
SPLIT_DIR = os.path.join(DATA_DIR, "splits")
TRAIN_SPLIT = os.path.join(SPLIT_DIR, "train.txt")

# 5. Set reproducibility seed
from src.training.utils import set_seed

SEED = config["project"]["random_seed"]
set_seed(SEED)

# 6. Select device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 7. Verify important paths
assert os.path.isdir(REPO_DIR), "Repository not found"
assert os.path.isdir(RAW_DIR), "Raw dataset not found"
assert os.path.isfile(CONFIG_PATH), "Config not found"
assert os.path.isfile(TRAIN_SPLIT), "Training split not found"

print("Project setup successful!")
print("Repository:", REPO_DIR)
print("Dataset:", RAW_DIR)
print("Seed:", SEED)
print("Device:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
