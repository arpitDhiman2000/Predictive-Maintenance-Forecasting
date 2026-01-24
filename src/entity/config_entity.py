import os
from src.constants import *
from dataclasses import dataclass, field
from datetime import datetime

TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

@dataclass
class TrainingPipelineConfig:
    pipeline_name: str = PIPELINE_NAME
    artifact_dir: str = os.path.join(ARTIFACT_DIR, TIMESTAMP)
    timestamp: str = TIMESTAMP


training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()

@dataclass
class DataIngestionConfig:
    data_ingestion_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_INGESTION_DIR_NAME)

    feature_store_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_FEATURE_STORE_DIR)
    raw_data_file_path: str = os.path.join(feature_store_file_path, RAW_DATA_FILE_NAME)
    raw_rul_file_path: str = os.path.join(feature_store_file_path, RAW_RUL_FILE_NAME)

    training_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TRAIN_FILE_NAME)
    testing_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TEST_FILE_NAME)
    rul_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, RUL_FILE_NAME)

    train_test_split_cloumn: float = DATA_INGESTION_TRAIN_TEST_SPLIT_COLUMN
    data_collection_name:str = DATA_INGESTION_DATA_COLLECTION_NAME
    rul_collection_name:str = DATA_INGESTION_RUL_COLLECTION_NAME


@dataclass
class DataValidationConfig:
    data_validation_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_VALIDATION_DIR_NAME)
    validation_report_file_path: str = os.path.join(data_validation_dir, DATA_VALIDATION_REPORT_FILE_NAME)