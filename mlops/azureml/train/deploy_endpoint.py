#!/usr/bin/env python
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Script to deploy UNet training pipeline as a Pipeline Endpoint.
This creates the endpoint but does NOT run the pipeline.

Usage:
    python deploy_endpoint.py --env dev
    python deploy_endpoint.py --env prod --component_version 2
"""

import argparse
import yaml
from pathlib import Path
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, load_job
from azure.ai.ml.entities import PipelineComponent, PipelineEndpoint


def parse_args():
    parser = argparse.ArgumentParser("Deploy UNet Pipeline Endpoint")
    parser.add_argument("--env", type=str, required=True, 
                       choices=["dev", "uat", "prod"],
                       help="Environment: dev, uat, or prod")
    parser.add_argument("--component_version", type=str, default="1",
                       help="Pipeline component version (default: 1)")
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
    credential = DefaultAzureCredential()
    
    ml_client = MLClient(
        credential=credential,
        subscription_id=config["subscription_id"],
        resource_group_name=config["resource_group"],
        workspace_name=config["workspace_name"],
    )
    
    return ml_client


def substitute_variables(pipeline_yaml_path: Path, config: dict):
    """Substitute environment variables in pipeline YAML"""
    with open(pipeline_yaml_path, "r") as f:
        content = f.read()
    
    # Substitute variables
    content = content.replace("azureml:workspaceblobstore", 
                             f"azureml:{config['default_datastore']}")
    content = content.replace("azureml:cpu-cluster", 
                             f"azureml:{config['cpu_compute_name']}")
    content = content.replace("azureml:gpu-cluster", 
                             f"azureml:{config['gpu_compute_name']}")
    
    # Write to temporary file
    temp_yaml = pipeline_yaml_path.parent / "pipeline_temp.yaml"
    with open(temp_yaml, "w") as f:
        f.write(content)
    
    return temp_yaml


def main():
    args = parse_args()
    
    print(f"🚀 Deploying UNet Pipeline Endpoint for environment: {args.env}")
    
    # Load config
    config = load_config(args.env)
    print(f"✓ Loaded config for workspace: {config['workspace_name']}")
    
    # Get ML Client
    ml_client = get_ml_client(config)
    print(f"✓ Connected to Azure ML workspace")
    
    # Substitute variables in pipeline YAML
    pipeline_yaml_path = Path(__file__).parent / "pipeline.yaml"
    temp_yaml = substitute_variables(pipeline_yaml_path, config)
    
    try:
        # Load pipeline job
        pipeline_job = load_job(temp_yaml)
        print(f"✓ Loaded pipeline definition")
        
        # Create/update pipeline component
        component_name = f"unet-pipeline-component-{args.env}"
        print(f"\n📦 Creating pipeline component: {component_name}")
        
        pipeline_component = PipelineComponent(
            name=component_name,
            version=args.component_version,
            description=f"UNet Semantic Segmentation Pipeline ({args.env})",
            jobs=pipeline_job.jobs,
            inputs=pipeline_job.inputs,
            outputs=pipeline_job.outputs,
        )
        
        component = ml_client.components.create_or_update(pipeline_component)
        print(f"✅ Component '{component_name}:{args.component_version}' created/updated")
        
        # Create/update pipeline endpoint
        endpoint_name = f"unet-pipeline-endpoint-{args.env}"
        print(f"\n🔗 Creating pipeline endpoint: {endpoint_name}")
        
        endpoint = PipelineEndpoint(
            name=endpoint_name,
            description=f"UNet Training Pipeline Endpoint ({args.env.upper()})",
            default_component=f"{component_name}:{args.component_version}"
        )
        
        endpoint = ml_client.pipeline_endpoints.create_or_update(endpoint)
        print(f"✅ Pipeline endpoint '{endpoint_name}' deployed successfully!")
        print(f"\n📌 Endpoint Details:")
        print(f"   Name: {endpoint.name}")
        print(f"   ID: {endpoint.id}")
        print(f"   Default Component: {component_name}:{args.component_version}")
        print(f"\n✨ Endpoint is ready to be invoked via REST API or SDK")
        
    finally:
        # Clean up temporary file
        if temp_yaml.exists():
            temp_yaml.unlink()


if __name__ == "__main__":
    main()
