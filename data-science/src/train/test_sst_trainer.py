from custom_datamodule import CustomSemanticSegmentationDataModule
from custom_nongeodataset import CustomSemanticSegmentationDataset
from custom_sst_trainer import CustomSemanticSegmentationTask
import lightning.pytorch as pl
import mlflow
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import yaml
import argparse
import os
from pathlib import Path
import torch

# AZURE SQL LOGGER LIBRARY
# from utils import MLPipelineLogger
# import pyodbc

def parse_args():
    parser = argparse.ArgumentParser("SemanticSegmentationDatamodule")
    parser.add_argument("--data_dir", type=str, help="Data Asset path")
    parser.add_argument("--prep_input", type=str, help="Path to output of prep step")
    parser.add_argument("--config_path", type=str, help="Config Asset path")
    parser.add_argument("--model_output", type=str, help="Model output path")

    args = parser.parse_args()

    return args


def main():
    print("AZUREML_RUN_ID -----> ",os.getenv("AZUREML_RUN_ID")) 
    AZUREML_RUN_ID = os.getenv("AZUREML_RUN_ID")

    runs = mlflow.search_runs(
        filter_string=f"attributes.runid = '{AZUREML_RUN_ID}'",
        search_all_experiments=True
    )

    print(runs.head())
    run_id = runs['tags.mlflow.rootRunId'][0]

    # # List all available ODBC drivers
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
        args = parse_args()
        print(args)

        # logger.start_phase(run_id, "model_train")

        with open(args.config_path, "r") as f:
            config = yaml.safe_load(f)  # safer than yaml.load
        # Now you can access config like a Python dictionary

        # Get batch size from config - try different possible locations
        if 'data' in config and 'batch_size' in config['data']:
            batch_size = config['data']['batch_size']
        elif 'additional' in config and 'batch_size' in config['additional']:
            batch_size = config['additional']['batch_size']
        else:
            batch_size = 8  # default
        
        num_workers = config.get('data', {}).get('num_workers', 4)
        max_epochs = config.get('training', {}).get('num_epochs', 10)
        fast_dev_run = False
        in_channels = 3
        n_classes = config['model']['num_classes']
        backbone = config['model']['backbone']
        decoder = config['model']['decoder']
        pretrained = config['model']['pretrained']
        learning_rate = config['training']['learning_rate']

        dm = CustomSemanticSegmentationDataModule(
                        data_dir = args.data_dir,
                        splits_dir = args.prep_input,
                        config_path = args.config_path,
                        batch_size=batch_size,
                        num_workers=num_workers
                        )
                        
        dm.setup(stage='fit')

        task = CustomSemanticSegmentationTask(
            model=decoder,
            backbone=backbone,
            weights=pretrained,
            in_channels=3,
            num_classes=n_classes,
            loss='ce',
            lr=learning_rate,
            tmax=50,
        )

        # Create output directory for logs and checkpoints
        output_path = Path(args.model_output)
        output_path.mkdir(parents=True, exist_ok=True)
        
        csv_logger = CSVLogger(output_path / "logs", name="unet_experiment")

        # Save model checkpoints
        checkpoint_callback = ModelCheckpoint(
            dirpath=output_path / "checkpoints",
            filename='unet-{epoch:02d}-{val_loss:.2f}',
            monitor='val_loss',
            save_top_k=3,
            mode='min',
            save_last=True
        )

        trainer = pl.Trainer(
            fast_dev_run=fast_dev_run, 
            log_every_n_steps=1, 
            min_epochs=1, 
            max_epochs=max_epochs,
            logger=csv_logger,
            callbacks=[checkpoint_callback]
        )

        trainer.fit(task, dm)
        
        # Save final model
        final_model_path = output_path / "final_model.ckpt"
        trainer.save_checkpoint(final_model_path)
        print(f"Final model saved to {final_model_path}")
        
        # AZURE SQL LOGGER - END PIPELINE PHASE DATA_PREP
        # logger.complete_phase(run_id, "model_train")

        # AZURE SQL LOGGER - MAIN PIPELINE RUN COMPLETED
        # logger.complete_pipeline_run(run_id, "Completed")
    except Exception as e:
        # AZURE SQL LOGGER - MAIN PIPELINE FAILED
        # logger.complete_pipeline_run(run_id, "Failed" + str(e))
        raise Exception("Failed")

if __name__ == "__main__":
    # Uncomment the line below to run the test
    with mlflow.start_run() as run:
        main()
