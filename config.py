from pathlib import Path

PROJECT_DIR = Path(r"C:\Users\Startklar\Downloads\ML\Sudhanshu")

DATASET_PATH = PROJECT_DIR / "data" / "raw" / "PBC_dataset_normal_DIB"
ROOT = DATASET_PATH
RAW_DATA_DIR = DATASET_PATH

OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTDIR = OUTPUT_DIR

FIGURES_DIR = PROJECT_DIR / "figures"
FIGDIR = FIGURES_DIR

PROCESSED_DATA_DIR = PROJECT_DIR / "data" / "processed"

TRAIN_CSV = PROCESSED_DATA_DIR / "train.csv"
VAL_CSV = PROCESSED_DATA_DIR / "val.csv"
TEST_CSV = PROCESSED_DATA_DIR / "test.csv"

IMAGE_SIZE = (224, 224)
IMG_SIZE = (224, 224)

RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
