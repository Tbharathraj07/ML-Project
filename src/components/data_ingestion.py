# src/Components/data_ingestion.py

import os
import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# --- Ensure package imports work even when running this file directly ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.exception import CustomException
from src.logger import logging

# ✅ Absolute dataset path (fixed)
DATA_FILE_PATH = Path("/Users/tarumanibharathraj/Desktop/ML-Project/RawData/personal_health_dashboard_extended_dataset.csv")

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # ✅ Check if dataset exists
            if not DATA_FILE_PATH.exists():
                raise FileNotFoundError(f"Input data file not found: {DATA_FILE_PATH}")

            df = pd.read_csv(DATA_FILE_PATH)
            logging.info(f"Dataset loaded successfully with shape {df.shape}")

            # Create artifacts folder if not exists
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save raw dataset
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    print(f"✅ Train data saved to: {train_data}")
    print(f"✅ Test data saved to: {test_data}")
