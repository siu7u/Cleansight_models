from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model import TCNClassifier
from utils.temporal_main import run_temporal_main


MODEL_NAME = "tcn"
MODEL_CLASS = TCNClassifier


if __name__ == "__main__":
    run_temporal_main(MODEL_NAME, MODEL_CLASS)
