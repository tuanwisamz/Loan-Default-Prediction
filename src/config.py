"""Settings shared by every file in src/."""
from pathlib import Path

# Folders and files, built from this file's location so they work from any folder
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "dataset" / "raw" / "credit.csv"
RESULTS_DIR = PROJECT_ROOT / "results"

# Data and split settings
TARGET = "default"   # The column to predict (yes / no)
TEST_SIZE = 0.2      # 20% of rows for testing, 80% for training
RANDOM_STATE = 42    # Fixed number so results are the same every run
