import os
import sys

from pandas import DataFrame

from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.exception import MyException
from src.logger import logging
from src.data_access.data import PMFData

class DataIngestion:
    def __init__(self,data_ingestion_config:DataIngestionConfig=DataIngestionConfig()):
        """
        :param data_ingestion_config: configuration for data ingestion
        """
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise MyException(e,sys)
        

    def export_data_into_feature_store(self) -> tuple[DataFrame, DataFrame]:
        """
        Exports:
        1) Data data as-is from MongoDB (with dataset_id + split)
        2) RUL data as-is from MongoDB

        Returns:
            (data_df, rul_df)
        """
        try:
            logging.info("Exporting C-MAPSS data and RUL data from MongoDB (as-is).")

            my_data = PMFData()

            # 1) Data (contains dataset_id, split)
            data_df = my_data.export_collection_as_dataframe(
                collection_name=self.data_ingestion_config.data_collection_name
            )
            logging.info(f"Data DF shape: {data_df.shape}")

            # 2) RUL collection
            rul_df = my_data.export_collection_as_dataframe(
                collection_name=self.data_ingestion_config.rul_collection_name
            )
            logging.info(f"RUL DF shape: {rul_df.shape}")

            # Save raw data DF (as-is)
            raw_data_path = self.data_ingestion_config.raw_data_file_path
            os.makedirs(os.path.dirname(raw_data_path), exist_ok=True)
            data_df.to_csv(raw_data_path, index=False, header=True)
            logging.info(f"Saved raw data data to: {raw_data_path}")

            # Save raw RUL DF (as-is)
            raw_rul_path = self.data_ingestion_config.raw_rul_file_path
            os.makedirs(os.path.dirname(raw_rul_path), exist_ok=True)
            rul_df.to_csv(raw_rul_path, index=False, header=True)
            logging.info(f"Saved raw RUL data to: {raw_rul_path}")

            return data_df, rul_df

        except Exception as e:
            raise MyException(e, sys)

    def split_data_as_train_test(self, data_df: DataFrame, rul_df: DataFrame) -> None:
        """
        Splits based on predefined 'split' column and merges all datasets:
        - train: all rows where split == 'train'
        - test : all rows where split == 'test'

        Drops dataset_id and split from final merged outputs.
        Also saves RUL dataset.
        """
        logging.info("Entered split_data_as_train_test for predefined split merge.")

        try:
            required_cols = {self.data_ingestion_config.train_test_split_cloumn}
            if not required_cols.issubset(data_df.columns):
                raise ValueError(f"Missing required columns in data_df: {required_cols - set(data_df.columns)}")

            # 1) predefined split
            train_set = data_df[data_df[self.data_ingestion_config.train_test_split_cloumn].str.lower() == "train"].copy()
            test_set = data_df[data_df[self.data_ingestion_config.train_test_split_cloumn].str.lower() == "test"].copy()

            logging.info(f"Train rows: {train_set.shape}, Test rows: {test_set.shape}")

            # 2) Drop dataset_id & split as per requirement
            drop_cols = [c for c in self.data_ingestion_config.drop_columns if c in train_set.columns]
            train_set.drop(columns=drop_cols, inplace=True, errors="ignore")
            test_set.drop(columns=drop_cols, inplace=True, errors="ignore")

            logging.info("Performed train test split on the dataframe")
            logging.info(
                "Exited split_data_as_train_test method of Data_Ingestion class"
            )

            # 3) Save merged train/test
            os.makedirs(os.path.dirname(self.data_ingestion_config.training_file_path), exist_ok=True)

            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index=False, header=True)

            logging.info(f"Saved merged train file: {self.data_ingestion_config.training_file_path}")
            logging.info(f"Saved merged test file : {self.data_ingestion_config.testing_file_path}")

            # 4) Save RUL dataset (as separate file)
            os.makedirs(os.path.dirname(self.data_ingestion_config.rul_file_path), exist_ok=True)
            rul_df.to_csv(self.data_ingestion_config.rul_file_path, index=False, header=True)
            logging.info(f"Saved RUL file: {self.data_ingestion_config.rul_file_path}")

        except Exception as e:
            raise MyException(e, sys) from e

    def initiate_data_ingestion(self) ->DataIngestionArtifact:
        """
        Method Name :   initiate_data_ingestion
        Description :   This method initiates the data ingestion components of training pipeline 
        
        Output      :   train set and test set are returned as the artifacts of data ingestion components
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Entered initiate_data_ingestion method of Data_Ingestion class")

        try:
            data_df, rul_df = self.export_data_into_feature_store()

            logging.info("Got the data from mongodb")

            self.split_data_as_train_test(data_df, rul_df)

            logging.info("Performed train test split on the dataset")

            logging.info(
                "Exited initiate_data_ingestion method of Data_Ingestion class"
            )

            artifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path,
                rul_file_path=self.data_ingestion_config.rul_file_path,
            )

            logging.info(f"Data ingestion artifact: {artifact}")
            return artifact

        except Exception as e:
            raise MyException(e, sys) from e