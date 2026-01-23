# https://github.com/Azure/mlops-project-template/blob/main/classical/python-sdk-v2/data-science/src/prep/prep.py

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Prepares raw data and provides training, validation and test datasets
"""

import argparse
import yaml
from pathlib import Path
import os
import numpy as np
import pandas as pd
from glob import glob
from typing import List, Tuple, Dict
from PIL import Image
import mlflow
from sklearn.model_selection import train_test_split

# AZURE SQL LOGGER LIBRARY
from utils import MLPipelineLogger
import pyodbc

def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser("prep")
    parser.add_argument("--raw_data", type=str, help="Path to raw data")
    parser.add_argument("--config_data", type=str, help="Path to config data")
    parser.add_argument("--output", type=str, help="Path to output dataset")
    
    args = parser.parse_args()

    return args

def main(args):
    print("AZUREML_RUN_ID -----> ",os.getenv("AZUREML_RUN_ID")) 
    AZUREML_RUN_ID = os.getenv("AZUREML_RUN_ID")

    runs = mlflow.search_runs(
        filter_string=f"attributes.runid = '{AZUREML_RUN_ID}'",
        search_all_experiments=True
    )

    print(runs.head())
    run_id = runs['tags.mlflow.rootRunId'][0]

    # List all available ODBC drivers
    available_drivers = pyodbc.drivers()
    print("Available ODBC drivers:")
    for driver in available_drivers:
        print(f"  - {driver}")

    # Initialize AZURE SQL LOGGER
    logger = MLPipelineLogger(
        "DRIVER={ODBC Driver 18 for SQL Server};"
        "SERVER=trackjob.database.windows.net;"
        "DATABASE=devdb;"
        "UID=devadmin;"
        "PWD=Password1$"
    )

    # AZURE SQL LOGGER - MAIN PIPELINE RUN STARTED
    logger.start_pipeline_run(run_id, "Semantic Segmentation Pipeline", "default")
    
    '''Read, split, and save datasets'''

    # ------------ Reading Data ------------ #
    # -------------------------------------- #

    classes = {
        "Unlabeled":            {"rgb": (155, 155, 155), "color": "#9B9B9B"},
        "Water":                {"rgb": (226, 169, 41),  "color": "#E2A929"},
        "Land (unpaved area)":  {"rgb": (132, 41, 246),  "color": "#8429F6"},
        "Road":                 {"rgb": (110, 193, 228), "color": "#6EC1E4"},
        "Building":             {"rgb": (60, 16, 152),   "color": "#3C1098"},
        "Vegetation":           {"rgb": (254, 221, 58),  "color": "#FEDD3A"}
    }
    colors = [v["rgb"] for k, v in classes.items()]

    try:
        print("Path is: "+str(Path(args.raw_data)))
        print(colors)

        files = [f.name for f in Path(args.raw_data).iterdir() if f.is_file()]
        print(files)

        # AZURE SQL LOGGER - START PIPELINE PHASE DATA_PREP
        logger.start_phase(run_id, "data_prep")

        mapping = pd.read_csv(Path(args.raw_data)/'image_mask_mapping.csv')
        
        with open(args.config_data, "r") as f:
            config = yaml.safe_load(f)  # safer than yaml.load
        # Now you can access config like a Python dictionary
        print("Number of classes ---> ",config["model"]["num_classes"])

        # First, split into train and temp (for val/test)
        train_df, temp_df = train_test_split(mapping, test_size=0.3, random_state=42, shuffle=True)

        # Split temp into validation and test
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, shuffle=True)

        print(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")
        
        # Optionally, save to CSV
        train_df.to_csv((Path(args.output) / "image_mask_mapping_train.csv"), index=False)
        val_df.to_csv((Path(args.output) / "image_mask_mapping_val.csv"), index=False)
        test_df.to_csv((Path(args.output) / "image_mask_mapping_test.csv"), index=False)

        # AZURE SQL LOGGER - LOG PIEPELINE METRIC PHASE DATA_PREP ROWS
        logger.log_metric(run_id, "data_prep", "train_rows", len(train_df))
        logger.log_metric(run_id, "data_prep", "val_rows", len(val_df))
        logger.log_metric(run_id, "data_prep", "test_rows", len(test_df))

        # AZURE SQL LOGGER - END PIPELINE PHASE DATA_PREP
        logger.complete_phase(run_id, "data_prep")

        # AZURE SQL LOGGER - MAIN PIPELINE RUNNING
        logger.complete_pipeline_run(run_id, "Running")
    except Exception as e:
        # AZURE SQL LOGGER - MAIN PIPELINE FAILED
        logger.complete_pipeline_run(run_id, "Failed"+str(e))
        raise Exception("Failed")


if __name__ == "__main__":

    mlflow.start_run()

    # ---------- Parse Arguments ----------- #
    # -------------------------------------- #
    

    args = parse_args()

    lines = [
        f"Raw data path: {args.raw_data}",
        f"Config data path: {args.config_data}",
        f"Output path: {args.output}",
    ]

    for line in lines:
        print(line)
    
    main(args)

    mlflow.end_run()
