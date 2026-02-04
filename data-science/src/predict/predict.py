# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Predict component that loads trained model and runs inference
"""

import argparse
import yaml
from pathlib import Path
import os
import torch
import numpy as np
from PIL import Image
import mlflow
from torch.utils.data import DataLoader
from torchgeo.datasets.utils import rgb_to_mask

# Import from train component
import sys
sys.path.append(str(Path(__file__).parent.parent / "train"))
from custom_datamodule import CustomSemanticSegmentationDataModule
from custom_sst_trainer import CustomSemanticSegmentationTask
import lightning.pytorch as pl

# AZURE SQL LOGGER LIBRARY
# from utils import MLPipelineLogger
# import pyodbc

def parse_args():
    '''Parse input arguments'''
    parser = argparse.ArgumentParser("predict")
    parser.add_argument("--model_path", type=str, help="Path to trained model")
    parser.add_argument("--data_dir", type=str, help="Path to data directory")
    parser.add_argument("--prep_input", type=str, help="Path to folder containing split CSV files from prep step")
    parser.add_argument("--config_path", type=str, help="Path to config file")
    parser.add_argument("--predictions_output", type=str, help="Path to save predictions")
    
    args = parser.parse_args()
    return args

def load_model_from_checkpoint(checkpoint_path: str, config: dict):
    """Load trained model from checkpoint"""
    # Create task with same configuration as training
    task = CustomSemanticSegmentationTask(
        model=config['model']['decoder'],
        backbone=config['model']['backbone'],
        weights=config['model']['pretrained'],
        in_channels=3,
        num_classes=config['model']['num_classes'],
        loss='ce',
        lr=config['training']['learning_rate'],
        tmax=50,
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    task.load_state_dict(checkpoint['state_dict'])
    task.eval()
    
    return task

def save_predictions(predictions: np.ndarray, output_path: Path, filename: str):
    """Save predictions as image"""
    # Convert predictions to uint8
    pred_img = Image.fromarray(predictions.astype(np.uint8), mode='L')
    pred_img.save(output_path / filename)

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
    # available_drivers = pyodbc.drivers()
    # print("Available ODBC drivers:")
    # for driver in available_drivers:
    #     print(f"  - {driver}")

    # Initialize AZURE SQL LOGGER
    # logger = MLPipelineLogger(
    #     "DRIVER={ODBC Driver 18 for SQL Server};"
    #     "SERVER=trackjob.database.windows.net;"
    #     "DATABASE=devdb;"
    #     "UID=devadmin;"
    #     "PWD=Password1$"
    # )

    try:
        # logger.start_phase(run_id, "prediction")

        # Load config
        with open(args.config_path, "r") as f:
            config = yaml.safe_load(f)

        # Find model checkpoint
        model_path = Path(args.model_path)
        checkpoint_files = list(model_path.glob("*.ckpt"))
        if not checkpoint_files:
            # Try to find last.ckpt or best checkpoint
            last_checkpoint = model_path / "checkpoints" / "last.ckpt"
            if last_checkpoint.exists():
                checkpoint_path = str(last_checkpoint)
            else:
                # Find best checkpoint
                checkpoint_dir = model_path / "checkpoints"
                if checkpoint_dir.exists():
                    checkpoints = list(checkpoint_dir.glob("*.ckpt"))
                    if checkpoints:
                        checkpoint_path = str(checkpoints[-1])  # Use last one
                    else:
                        raise FileNotFoundError(f"No checkpoint found in {model_path}")
                else:
                    raise FileNotFoundError(f"No checkpoint found in {model_path}")
        else:
            checkpoint_path = str(checkpoint_files[0])

        print(f"Loading model from: {checkpoint_path}")

        # Load model
        model = load_model_from_checkpoint(checkpoint_path, config)
        model.eval()

        # Setup data module (use prep_input so test/predict use test split)
        batch_size = config.get('data', {}).get('batch_size', 8)
        num_workers = config.get('data', {}).get('num_workers', 4)

        dm = CustomSemanticSegmentationDataModule(
            data_dir=args.data_dir,
            splits_dir=args.prep_input,  # folder with image_mask_mapping_test.csv
            config_path=args.config_path,
            batch_size=batch_size,
            num_workers=num_workers
        )
        dm.setup(stage='test')

        # Create a trainer (for test + predict; no training)
        trainer = pl.Trainer(devices=1, accelerator='auto', logger=False)

        # 1) Run test to compute and show metrics
        print("Running evaluation on test set...")
        trainer.test(model, dm)

        # 2) Run prediction and save images (using predict_dataloader)
        output_path = Path(args.predictions_output)
        output_path.mkdir(parents=True, exist_ok=True)
        predictions_dir = output_path / "predictions"
        predictions_dir.mkdir(exist_ok=True)

        predict_loader = dm.predict_dataloader()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(predict_loader):
                images = batch['image']
                masks = batch['mask']  # Ground truth for reference
                
                # Get predictions
                outputs = model(images)
                
                # Get predicted classes
                if isinstance(outputs, dict):
                    preds = outputs['pred']
                else:
                    preds = torch.argmax(outputs, dim=1)
                
                # Save predictions
                for i in range(preds.shape[0]):
                    pred_np = preds[i].cpu().numpy()
                    filename = f"prediction_{batch_idx}_{i}.png"
                    save_predictions(pred_np, predictions_dir, filename)
                    
                    # Optionally save ground truth for comparison
                    mask_np = masks[i].cpu().numpy()
                    gt_filename = f"ground_truth_{batch_idx}_{i}.png"
                    save_predictions(mask_np, predictions_dir, gt_filename)

        print(f"Predictions saved to {predictions_dir}")

        # AZURE SQL LOGGER - END PIPELINE PHASE
        # logger.complete_phase(run_id, "prediction")

        # AZURE SQL LOGGER - MAIN PIPELINE RUN COMPLETED
        # logger.complete_pipeline_run(run_id, "Completed")
        
    except Exception as e:
        # AZURE SQL LOGGER - MAIN PIPELINE FAILED
        # logger.complete_pipeline_run(run_id, "Failed" + str(e))
        raise Exception("Failed")


if __name__ == "__main__":
    mlflow.start_run()

    args = parse_args()

    lines = [
        f"Model path: {args.model_path}",
        f"Data directory: {args.data_dir}",
        f"Config path: {args.config_path}",
        f"Predictions output: {args.predictions_output}",
    ]

    for line in lines:
        print(line)
    
    main(args)

    mlflow.end_run()
