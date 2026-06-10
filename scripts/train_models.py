from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_pipeline import prepare_training_data
from src.modeling import train_and_save_models

if __name__ == "__main__":
    df = prepare_training_data()
    print(f"Training data rows: {len(df)}")
    metrics = train_and_save_models()
    print("\n=== MODEL TRAINING COMPLETE ===")
    for k, v in metrics.items():
        if "report" not in k:
            print(f"{k}: {v}")
