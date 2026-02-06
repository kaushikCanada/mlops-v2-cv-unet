#!/usr/bin/env python
"""
Invoke the deployed UNet training pipeline batch endpoint to run a training job.

Usage:
    python invoke_endpoint.py --env dev \
        --data_dir "azureml://subscriptions/.../paths/data" \
        --config_path "azureml://subscriptions/.../paths/config.yml"
"""

import argparse
import yaml
from pathlib import Path
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, Input
from azure.ai.ml.constants import InputOutputModes


def parse_args():
    parser = argparse.ArgumentParser("Invoke UNet Training Pipeline Batch Endpoint")
    parser.add_argument("--env", type=str, required=True, choices=["dev", "uat", "prod"],
                       help="Environment: dev, uat, or prod")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Azure ML data path (e.g., azureml://subscriptions/.../paths/...)")
    parser.add_argument("--config_path", type=str, required=True,
                       help="Azure ML config path (e.g., azureml://subscriptions/.../paths/...)")
    parser.add_argument("--deployment-name", type=str, default=None,
                       help="Specific deployment name (default: use endpoint default)")
    parser.add_argument("--experiment-name", type=str, default="unet_train_endpoint",
                       help="Experiment name (default: unet_train_endpoint)")
    return parser.parse_args()


def load_config(env: str):
    config_path = Path(__file__).parent.parent.parent.parent / "config" / f"config-{env}.yml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_ml_client(config: dict):
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id=config["subscription_id"],
        resource_group_name=config["resource_group"],
        workspace_name=config["workspace_name"],
    )
    return ml_client


def main():
    args = parse_args()
    
    print(f"Invoking UNet Training Pipeline Batch Endpoint")
    print(f"Environment: {args.env}")
    print()
    
    # Load config
    config = load_config(args.env)
    print(f"Connected to workspace: {config['workspace_name']}")
    
    # Get ML Client
    ml_client = get_ml_client(config)
    
    # Define endpoint name (must match deploy_endpoint.py)
    endpoint_name = f"unet-train-endpoint-{args.env}"
    
    # Check if endpoint exists
    try:
        endpoint = ml_client.batch_endpoints.get(endpoint_name)
        print(f"Found endpoint: {endpoint_name}")
        print(f"   Default deployment: {endpoint.defaults.deployment_name}")
    except Exception as e:
        print(f"Error: Endpoint '{endpoint_name}' not found!")
        print(f"   Have you deployed the endpoint? Run: python deploy_endpoint.py --env {args.env}")
        raise
    
    # Prepare inputs using Input class
    print(f"\nPreparing inputs:")
    print(f"   data_dir:    {args.data_dir}")
    print(f"   config_path: {args.config_path}")
    
    inputs = {
        "data_dir": Input(
            type="uri_folder",
            path=args.data_dir,
            mode=InputOutputModes.RO_MOUNT
        ),
        "config_path": Input(
            type="uri_file",
            path=args.config_path,
            mode=InputOutputModes.RO_MOUNT
        )
    }
    
    # Invoke endpoint
    print(f"\nInvoking endpoint...")
    deployment_name_to_use = args.deployment_name if args.deployment_name else endpoint.defaults.deployment_name
    print(f"   Using deployment: {deployment_name_to_use}")
    
    try:
        if args.deployment_name:
            # Invoke specific deployment
            job = ml_client.batch_endpoints.invoke(
                endpoint_name=endpoint_name,
                deployment_name=args.deployment_name,
                inputs=inputs,
                experiment_name=args.experiment_name
            )
        else:
            # Invoke default deployment
            job = ml_client.batch_endpoints.invoke(
                endpoint_name=endpoint_name,
                inputs=inputs,
                experiment_name=args.experiment_name
            )
        
        print(f"\n" + "="*60)
        print("TRAINING JOB SUBMITTED")
        print("="*60)
        print(f"Job Name:       {job.name}")
        print(f"Experiment:     {args.experiment_name}")
        print(f"Status:         {job.status}")
        print()
        print(f"View in Azure ML Studio:")
        print(f"   https://ml.azure.com/runs/{job.name}")
        print()
        print(f"To monitor the job:")
        print(f"   az ml job show --name {job.name}")
        print(f"   az ml job stream --name {job.name}")
        print()
        
        return job
        
    except Exception as e:
        print(f"\nError invoking endpoint: {e}")
        raise


if __name__ == "__main__":
    main()
