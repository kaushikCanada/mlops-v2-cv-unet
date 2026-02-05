#!/usr/bin/env python
"""
Deploy UNet training pipeline as a Batch Endpoint with PipelineComponentBatchDeployment.
Reference: https://learn.microsoft.com/en-us/azure/machine-learning/how-to-use-batch-training-pipeline

Usage: python deploy_endpoint.py --env dev
"""

import argparse
import yaml
from pathlib import Path
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, load_job
from azure.ai.ml.entities import BatchEndpoint, PipelineComponentBatchDeployment, PipelineComponent


def parse_args():
    parser = argparse.ArgumentParser("Deploy UNet Pipeline Batch Endpoint")
    parser.add_argument("--env", type=str, required=True, choices=["dev", "uat", "prod"])
    parser.add_argument("--component_version", type=str, default="1")
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


def substitute_variables(pipeline_yaml_path: Path, config: dict):
    with open(pipeline_yaml_path, "r") as f:
        content = f.read()
    content = content.replace("azureml:workspaceblobstore", f"azureml:{config['default_datastore']}")
    content = content.replace("azureml:cpu-cluster", f"azureml:{config['cpu_compute_name']}")
    content = content.replace("azureml:gpu-cluster", f"azureml:{config['gpu_compute_name']}")
    temp_yaml = pipeline_yaml_path.parent / "pipeline_temp.yaml"
    with open(temp_yaml, "w") as f:
        f.write(content)
    return temp_yaml


def main():
    args = parse_args()
    print(f"Deploying UNet Training Pipeline Batch Endpoint - Environment: {args.env}")
    
    config = load_config(args.env)
    print(f"Connected to workspace: {config['workspace_name']}")
    
    ml_client = get_ml_client(config)
    pipeline_yaml_path = Path(__file__).parent / "pipeline.yaml"
    temp_yaml = substitute_variables(pipeline_yaml_path, config)
    
    try:
        pipeline_job = load_job(temp_yaml)
        component_name = f"unet-training-pipeline-{args.env}"
        
        print(f"Creating pipeline component: {component_name}")
        pipeline_component = PipelineComponent(
            name=component_name,
            version=args.component_version,
            description=f"UNet Training Pipeline ({args.env.upper()})",
            jobs=pipeline_job.jobs,
            inputs=pipeline_job.inputs,
            outputs=pipeline_job.outputs,
        )
        component = ml_client.components.create_or_update(pipeline_component)
        print(f"Component created: {component_name}:{args.component_version}")
        
        endpoint_name = f"unet-train-endpoint-{args.env}"
        print(f"Creating batch endpoint: {endpoint_name}")
        
        endpoint = BatchEndpoint(
            name=endpoint_name,
            description=f"UNet Training Pipeline Endpoint ({args.env.upper()})",
            tags={"environment": args.env, "type": "training"},
        )
        
        try:
            endpoint = ml_client.batch_endpoints.begin_create_or_update(endpoint).result()
            print(f"Endpoint created: {endpoint_name}")
        except Exception as e:
            print(f"Endpoint exists, continuing...")
            endpoint = ml_client.batch_endpoints.get(endpoint_name)
        
        deployment_name = f"unet-train-deploy-{args.env}"
        print(f"Creating deployment: {deployment_name}")
        
        deployment = PipelineComponentBatchDeployment(
            name=deployment_name,
            description=f"UNet training deployment ({args.env.upper()})",
            endpoint_name=endpoint_name,
            component=component,
            settings={
                "continue_on_step_failure": False,
                "default_compute": config["cpu_compute_name"],
            },
        )
        
        deployment = ml_client.batch_deployments.begin_create_or_update(deployment).result()
        print(f"Deployment created: {deployment_name}")
        
        print(f"Setting as default deployment...")
        endpoint = ml_client.batch_endpoints.get(endpoint_name)
        endpoint.defaults.deployment_name = deployment_name
        ml_client.batch_endpoints.begin_create_or_update(endpoint).result()
        
        print("\n" + "="*60)
        print("DEPLOYMENT COMPLETE")
        print("="*60)
        print(f"Endpoint:     {endpoint_name}")
        print(f"Deployment:   {deployment_name}")
        print(f"Component:    {component_name}:{args.component_version}")
        print(f"Environment:  {args.env}")
        print("\nEndpoint is ready to be invoked!")
        
    finally:
        if temp_yaml.exists():
            temp_yaml.unlink()


if __name__ == "__main__":
    main()
