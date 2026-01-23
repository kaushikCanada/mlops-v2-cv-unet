#!/usr/bin/env python
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Script to submit UNet training pipeline directly from YAML.
For development use - submits pipeline job directly to Azure ML workspace.
"""

import argparse
import yaml
import os
from pathlib import Path
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml import load_job, Input
from azure.ai.ml.constants import InputOutputModes

def parse_args():
    parser = argparse.ArgumentParser("Submit UNet Training Pipeline")
    parser.add_argument("--env", type=str, required=True, choices=["dev", "uat", "prod"],
                        help="Environment: dev, uat, or prod")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Azure ML data path (e.g., azureml://subscriptions/.../paths/...)")
    parser.add_argument("--config_path", type=str, required=True,
                        help="Azure ML config path (e.g., azureml://subscriptions/.../paths/...)")
    parser.add_argument("--experiment_name", type=str, default="unet_train",
                        help="Experiment name (default: unet_train)")
    parser.add_argument("--pipeline_yaml", type=str, 
                        default=os.path.join(os.path.dirname(__file__), "pipeline.yaml"),
                        help="Path to pipeline YAML file")
    
    return parser.parse_args()

def load_config(env: str):
    """Load configuration from config file"""
    config_path = Path(__file__).parent.parent.parent.parent / "config" / f"config-{env}.yml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    return config

def get_ml_client(config: dict):
    """Get ML Client with authentication"""
    try:
        credential = DefaultAzureCredential()
        # Check if given credential can get token successfully
        credential.get_token("https://management.azure.com/.default")
    except Exception as ex:
        print(f"DefaultAzureCredential failed: {ex}")
        print("Falling back to environment variables or interactive login")
        credential = None
    
    ml_client = MLClient(
        credential=credential,
        subscription_id=config["subscription_id"],
        resource_group_name=config["resource_group"],
        workspace_name=config["workspace_name"],
    )
    
    return ml_client

def substitute_variables(pipeline_yaml_path: Path, config: dict):
    """Substitute environment variables in pipeline YAML and component YAMLs"""
    yaml_dir = pipeline_yaml_path.parent
    
    # Substitute in component YAMLs first
    component_yamls = ["split.yaml", "train.yaml", "predict.yaml"]
    for comp_yaml in component_yamls:
        comp_path = yaml_dir / comp_yaml
        if comp_path.exists():
            with open(comp_path, "r") as f:
                content = f.read()
            # Replace environment placeholder
            content = content.replace("azureml:azml-geo-deep-env:47", f"azureml:{config['environment_name']}")
            with open(comp_path, "w") as f:
                f.write(content)
    
    # Substitute in pipeline YAML
    with open(pipeline_yaml_path, "r") as f:
        content = f.read()
    
    # Substitute variables
    content = content.replace("azureml:workspaceblobstore", f"azureml:{config['default_datastore']}")
    content = content.replace("azureml:cpu-cluster", f"azureml:{config['cpu_compute_name']}")
    content = content.replace("azureml:gpu-cluster", f"azureml:{config['gpu_compute_name']}")
    
    # Write to temporary file
    temp_yaml = pipeline_yaml_path.parent / "pipeline_temp.yaml"
    with open(temp_yaml, "w") as f:
        f.write(content)
    
    return temp_yaml

def main():
    args = parse_args()
    
    print(f"Loading configuration for environment: {args.env}")
    config = load_config(args.env)
    
    print(f"Connecting to Azure ML workspace: {config['workspace_name']}")
    ml_client = get_ml_client(config)
    
    # Substitute variables in pipeline YAML
    pipeline_yaml_path = Path(args.pipeline_yaml)
    temp_yaml = substitute_variables(pipeline_yaml_path, config)
    
    try:
        print(f"Loading pipeline from: {temp_yaml}")
        pipeline_job = load_job(temp_yaml)
        
        # Set pipeline inputs using Input class to properly bind them
        pipeline_job.inputs.data_dir = Input(
            path=args.data_dir,
            type="uri_folder",
            mode=InputOutputModes.RO_MOUNT
        )
        pipeline_job.inputs.config_path = Input(
            path=args.config_path,
            type="uri_file",
            mode=InputOutputModes.RO_MOUNT
        )
        
        # Set pipeline settings
        pipeline_job.settings.default_datastore = config["default_datastore"]
        pipeline_job.settings.continue_on_step_failure = False
        
        print(f"Submitting pipeline to experiment: {args.experiment_name}")
        pipeline_job = ml_client.jobs.create_or_update(
            pipeline_job, 
            experiment_name=args.experiment_name
        )
        
        print(f"Pipeline submitted successfully!")
        print(f"Pipeline job name: {pipeline_job.name}")
        print(f"Pipeline job ID: {pipeline_job.id}")
        print(f"View in Azure ML Studio: {ml_client.workspaces.get(ml_client.workspace_name).studio_endpoint}")
        print(f"\nTo stream logs, run:")
        print(f"  ml_client.jobs.stream(pipeline_job.name)")
        
    finally:
        # Clean up temporary file
        if temp_yaml.exists():
            temp_yaml.unlink()

if __name__ == "__main__":
    main()
