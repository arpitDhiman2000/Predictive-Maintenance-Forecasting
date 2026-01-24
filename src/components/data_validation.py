import json
import sys
import os

import pandas as pd

from pandas import DataFrame

from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import read_yaml_file, expand_columns
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.entity.config_entity import DataValidationConfig
from src.constants import SCHEMA_FILE_PATH


class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_config: DataValidationConfig):
        """
        :param data_ingestion_artifact: Output reference of data ingestion artifact stage
        :param data_validation_config: configuration for data validation
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config =read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e,sys)
        
    def op_sensor_column_list(self) -> list:
        """
        Method Name :   column_list
        Description :   This method generates a list of operational settings and sensors columns
        
        Output      :   Returns list
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            patterns = self._schema_config.get("data_feilds_patterns", {})

            op_settings = patterns.get("op_settings")
            sensor = patterns.get("sensors")

            if not op_settings or not sensor:
                raise KeyError("Missing 'data_feilds_patterns.op_settings' or 'data_feilds_patterns.sensor' in YAML")

            cols = (
                expand_columns(op_settings["prefix"], op_settings["index_range"]) +
                expand_columns(sensor["prefix"], sensor["index_range"])
            )
            return cols

        except Exception as e:
            raise MyException(e, sys)

    def validate_number_of_data_columns(self, dataframe: DataFrame) -> bool:
        """
        Method Name :   validate_number_of_columns
        Description :   This method validates the number of columns
        
        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            columns = (
                self._schema_config["data_fields"] +
                self.op_sensor_column_list()
                )
            
            status = len(dataframe.columns) == len(columns)
            logging.info(f"Is required column present: [{status}]")
            return status
        except Exception as e:
            raise MyException(e, sys)
        
    def validate_number_of_rul_columns(self, dataframe: DataFrame) -> bool:
        """
        Method Name :   validate_number_of_columns
        Description :   This method validates the number of columns
        
        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            columns = (
                self._schema_config["rul_fields"]
                )
            
            status = len(dataframe.columns) == len(columns)
            logging.info(f"Is required column present: [{status}]")
            return status
        except Exception as e:
            raise MyException(e, sys)

    def data_column_exist(self, df: DataFrame) -> bool:
        """
        Method Name :   data_column_exist
        Description :   This method validates the existence of a numerical and categorical columns in data
        
        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            dataframe_columns = df.columns
            numerical_columns = (
                self._schema_config["data_numerical_columns"] +
                self.op_sensor_column_list()
            )

            categorical_columns = (
                self._schema_config["data_categorical_columns"]
            )

            missing_numerical_columns = []
            missing_categorical_columns = []
            for column in numerical_columns:
                if column not in dataframe_columns:
                    missing_numerical_columns.append(column)

            if len(missing_numerical_columns)>0:
                logging.info(f"Missing numerical column in data: {missing_numerical_columns}")


            for column in categorical_columns:
                if column not in dataframe_columns:
                    missing_categorical_columns.append(column)

            if len(missing_categorical_columns)>0:
                logging.info(f"Missing categorical column in data: {missing_categorical_columns}")

            return False if len(missing_categorical_columns)>0 or len(missing_numerical_columns)>0 else True
        except Exception as e:
            raise MyException(e, sys) from e
        

    def rul_column_exist(self, df: DataFrame) -> bool:
        """
        Method Name :   data_column_exist
        Description :   This method validates the existence of a numerical and categorical columns in data
        
        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            dataframe_columns = df.columns
            numerical_columns = (
                self._schema_config["rul_numerical_columns"]
            )

            categorical_columns = (
                self._schema_config["rul_categorical_columns"]
            )

            missing_numerical_columns = []
            missing_categorical_columns = []
            for column in numerical_columns:
                if column not in dataframe_columns:
                    missing_numerical_columns.append(column)

            if len(missing_numerical_columns)>0:
                logging.info(f"Missing numerical column in rul: {missing_numerical_columns}")


            for column in categorical_columns:
                if column not in dataframe_columns:
                    missing_categorical_columns.append(column)

            if len(missing_categorical_columns)>0:
                logging.info(f"Missing categorical column in rul: {missing_categorical_columns}")

            return False if len(missing_categorical_columns)>0 or len(missing_numerical_columns)>0 else True
        except Exception as e:
            raise MyException(e, sys) from e

    @staticmethod
    def read_data(file_path) -> DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e, sys)
        

    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        Method Name :   initiate_data_validation
        Description :   This method initiates the data validation component for the pipeline
        
        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """

        try:
            validation_error_msg = ""
            logging.info("Starting data validation")
            train_df, test_df, rul_df = (DataValidation.read_data(file_path=self.data_ingestion_artifact.trained_file_path),
                                            DataValidation.read_data(file_path=self.data_ingestion_artifact.test_file_path),
                                            DataValidation.read_data(file_path=self.data_ingestion_artifact.rul_file_path))

            # Checking col len of dataframe for train/test df
            status = self.validate_number_of_data_columns(dataframe=train_df)
            if not status:
                validation_error_msg += f"Columns are missing in training dataframe. "
            else:
                logging.info(f"All required columns present in training dataframe: {status}")

            status = self.validate_number_of_data_columns(dataframe=test_df)
            if not status:
                validation_error_msg += f"Columns are missing in test dataframe. "
            else:
                logging.info(f"All required columns present in testing dataframe: {status}")

            status = self.validate_number_of_rul_columns(dataframe=rul_df)
            if not status:
                validation_error_msg += f"Columns are missing in rul dataframe. "
            else:
                logging.info(f"All required columns present in rul dataframe: {status}")

            # Validating col dtype for train/test df
            status = self.data_column_exist(df=train_df)
            if not status:
                validation_error_msg += f"Columns are missing in training dataframe. "
            else:
                logging.info(f"All categorical/int columns present in training dataframe: {status}")

            status = self.data_column_exist(df=test_df)
            if not status:
                validation_error_msg += f"Columns are missing in test dataframe."
            else:
                logging.info(f"All categorical/int columns present in testing dataframe: {status}")

            status = self.rul_column_exist(df=rul_df)
            if not status:
                validation_error_msg += f"Columns are missing in rul dataframe."
            else:
                logging.info(f"All categorical/int columns present in rul dataframe: {status}")

            validation_status = len(validation_error_msg) == 0

            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                message=validation_error_msg,
                validation_report_file_path=self.data_validation_config.validation_report_file_path
            )

            # Ensure the directory for validation_report_file_path exists
            report_dir = os.path.dirname(self.data_validation_config.validation_report_file_path)
            os.makedirs(report_dir, exist_ok=True)

            # Save validation status and message to a JSON file
            validation_report = {
                "validation_status": validation_status,
                "message": validation_error_msg.strip()
            }

            with open(self.data_validation_config.validation_report_file_path, "w") as report_file:
                json.dump(validation_report, report_file, indent=4)

            logging.info("Data validation artifact created and saved to JSON file.")
            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise MyException(e, sys) from e