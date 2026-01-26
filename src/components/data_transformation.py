import sys
import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer

from src.constants import TARGET_COLUMN, SCHEMA_FILE_PATH, RUL_VALUE
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file


class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_transformation_config: DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e, sys)

    def get_data_transformer_object(self) -> Pipeline:
        """
        Creates and returns a data transformer object for the data, 
        including gender mapping, dummy variable creation, column renaming,
        feature scaling, and type adjustments.
        """
        logging.info("Entered get_data_transformer_object method of DataTransformation class")

        try:
            # Initialize transformers
            scaler = RobustScaler()
            logging.info("Transformers Initialized: RobustScaler")

            # Load schema configurations
            scaling_features = self._schema_config['scaling_features']
            logging.info("Cols loaded from schema.")

            # Creating preprocessor pipeline
            preprocessor = ColumnTransformer(
                transformers=[
                    ("RobustScaler", scaler, scaling_features)
                ],
                remainder='passthrough'
            )

            # Wrapping everything in a single pipeline
            final_pipeline = Pipeline(steps=[("Preprocessor", preprocessor)])
            logging.info("Final Pipeline Ready!!")
            logging.info("Exited get_data_transformer_object method of DataTransformation class")
            return final_pipeline

        except Exception as e:
            logging.exception("Exception occurred in get_data_transformer_object method of DataTransformation class")
            raise MyException(e, sys) from e
    
    def remap_engine_id_by_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Makes engine_id unique across dataset_id groups by adding cumulative offsets.

        Expected dataset_id values: FD001, FD002, FD003, FD004 (works for any FDxxx set).
        Example:
        FD001 engine_id 1..100  -> 1..100
        FD002 engine_id 1..100  -> 101..200
        FD003 engine_id 1..100  -> 201..300
        FD004 engine_id 1..100  -> 301..400
        """
        try:
            # Sort datasets like FD001, FD002, ...
            ds_order = sorted(
                df["dataset_id"].dropna().unique(),
                key=lambda x: int(str(x).upper().replace("FD", ""))
            )

            # Compute max engine_id per dataset in that order
            max_per_ds = df.groupby("dataset_id")["engine_id"].max().reindex(ds_order).astype(int)

            # Build cumulative offsets
            offsets = {}
            running = 0
            for ds in ds_order:
                offsets[ds] = running
                running += int(max_per_ds.loc[ds])

            # Apply
            df["engine_id"] = df["engine_id"].astype(int) + df["dataset_id"].map(offsets).astype(int)

            return df
        
        except Exception as e:
            raise MyException(e, sys)

    def _req_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop the columns."""
        logging.info("Dropping columns")
        req_col = self._schema_config['required_features']
        for col in df.columns:
            if col not in req_col:
                df = df.drop(col, axis=1)
        return df

    def calculate_rul(self, df: pd.DataFrame, cap_rul=None) -> pd.DataFrame:
        """
        Input:
        df: DataFrame with at least ['engine_id', 'cycle']
        cap_rul: None (default) or an int

        Behavior:
        - Computes RUL = max(cycle per engine) - cycle
        - If cap_rul is None: returns df + 'rul'
        - If cap_rul is not None: returns df + 'cap_rul' (min(rul, cap_rul)) and drops 'rul'
        """
        try:
            logging.info("Calculating RUL for data")
            max_cycle = df.groupby("engine_id")["cycle"].transform("max")
            df["rul"] = (max_cycle - df["cycle"]).astype(int)

            if cap_rul is None:
                return df

            cap_val = int(cap_rul)
            df["rul"] = df["rul"].clip(upper=cap_val).astype(int)
            return df
        
        except Exception as e:
            raise MyException(e, sys)
    
    def merge_test_and_rul(self, test_df: pd.DataFrame, rul_df: pd.DataFrame) -> pd.DataFrame:
        """
        Input:
        test_df: DataFrame with at least ['engine_id', 'cycle']
        rul_df: DataFrame with at least ['engine_id', 'rul']

        Behavior:
        - Merge last cycle of test data with rul data
        """
        try:
            logging.info("Merging test and rul data")
            test_last = (
                test_df
                .sort_values(["engine_id", "cycle"])
                .groupby("engine_id", as_index=False)
                .tail(1)
            )

            # 2) ensure exact alignment with rul_final_df by merging on engine_id
            test_last = test_last.merge(
                rul_df[["engine_id","rul"]],
                on="engine_id",
                how="inner",
                validate="one_to_one"
            ).sort_values("engine_id")
            
            return test_last
        
        except Exception as e:
            raise MyException(e, sys)


    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Initiates the data transformation component for the pipeline.
        """
        try:
            logging.info("Data Transformation Started !!!")
            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)

            # Load train and test data
            train_df = self.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(file_path=self.data_ingestion_artifact.test_file_path)
            rul_df = self.read_data(file_path=self.data_ingestion_artifact.rul_file_path)
            logging.info("Train-Test data loaded")

            mapped_train_df = self.remap_engine_id_by_dataset(train_df)
            mapped_test_df = self.remap_engine_id_by_dataset(test_df)
            mapped_rul_df = self.remap_engine_id_by_dataset(rul_df)
            logging.info("Train-Test-Rul dataset mapped")

            rul_cal_train_df = self.calculate_rul(mapped_train_df, cap_rul=RUL_VALUE)
            rul_cal_test_df = self.merge_test_and_rul(mapped_test_df, mapped_rul_df)
            logging.info("RUL calculated for train df.")
            
            req_feature_train_df = self._req_column(rul_cal_train_df)
            req_feature_test_df = self._req_column(rul_cal_test_df)
            logging.info("Required input cols defined for both train, test and rul df.")

            input_feature_train_df = rul_cal_train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = rul_cal_train_df[TARGET_COLUMN]

            input_feature_test_df = rul_cal_test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = rul_cal_test_df[TARGET_COLUMN]
            logging.info("Input and Target cols defined for both train and test df.")

            logging.info("Starting data transformation")
            preprocessor = self.get_data_transformer_object()
            logging.info("Got the preprocessor object")

            logging.info("Initializing transformation for Training-data")
            input_feature_train_final = preprocessor.fit_transform(input_feature_train_df)
            logging.info("Initializing transformation for Testing-data")
            input_feature_test_final = preprocessor.transform(input_feature_test_df)
            logging.info("Transformation done end to end to train-test df.")


            train_arr = np.c_[input_feature_train_final, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_final, np.array(target_feature_test_df)]
            logging.info("feature-target concatenation done for train-test df.")

            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)
            logging.info("Saving transformation object and transformed files.")

            logging.info("Data transformation completed successfully")
            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

        except Exception as e:
            raise MyException(e, sys) from e