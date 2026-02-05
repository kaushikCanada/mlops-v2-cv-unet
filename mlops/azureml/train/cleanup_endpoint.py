#!/usr/bin/env python
"""
Clean up (delete) UNet training pipeline batch endpoint and related resources.

Usage:
    python cleanup_endpoint.py --env dev
    python cleanup_endpoint.py --env prod --delete-component
"""

import argparse
import yaml
from pathlib import Path
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient


def parse_args():
    parser = argparse.ArgumentParser("Cleanup UNet Pipeline Batch Endpoint")
    parser.add_argument("--env", type=str, required=True, choices=["dev", "uat", "prod"])
    parser.add_argument("--delete-component", action="store_true", 
                       help="Also delete the pipeline component (default: False)")
    parser.add_argument("--component-version", type=str, default="1",
                       help="Component version to delete (if --delete-component is set)")
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
    print(f"Cleaning up UNet Training Pipeline Resources - Environment: {args.env}")
    print()
    
    config = load_config(args.env)
    print(f"Connected to workspace: {config['workspace_name']}")
    
    ml_client = get_ml_client(config)
    
    # Define resource names (must match deploy_endpoint.py naming)
    endpoint_name = f"unet-train-endpoint-{args.env}"
    deployment_name = f"unet-train-deploy-{args.env}"
    component_name = f"unet_training_pipeline_{args.env}"
    
    try:
        # Delete endpoint (this automatically deletes all deployments)
        print(f"\nDeleting batch endpoint: {endpoint_name}")
        try:
            ml_client.batch_endpoints.begin_delete(name=endpoint_name).result()
            print(f"Endpoint '{endpoint_name}' deleted successfully")
            print(f"   (All deployments under this endpoint were also deleted)")
        except Exception as e:
            if "ResourceNotFound" in str(e) or "NotFound" in str(e):
                print(f"Endpoint '{endpoint_name}' not found (already deleted or never created)")
            else:
                print(f"Error deleting endpoint: {e}")
                raise
        
        # Optionally delete component
        if args.delete_component:
            print(f"\nDeleting pipeline component: {component_name}:{args.component_version}")
            try:
                ml_client.components.archive(
                    name=component_name,
                    version=args.component_version
                )
                print(f"Component '{component_name}:{args.component_version}' archived")
                print(f"   (Archived components can be restored if needed)")
            except Exception as e:
                if "ResourceNotFound" in str(e) or "NotFound" in str(e):
                    print(f"Component '{component_name}:{args.component_version}' not found")
                else:
                    print(f"Error archiving component: {e}")
                    raise
        else:
            print(f"\nComponent '{component_name}' was NOT deleted")
            print(f"   (Use --delete-component flag to also remove the component)")
        
        print(f"\n" + "="*60)
        print("CLEANUP COMPLETE")
        print("="*60)
        print()
        print("Resources cleaned up:")
        print(f"  Endpoint:    {endpoint_name} - DELETED")
        print(f"  Deployment:  {deployment_name} - DELETED (with endpoint)")
        if args.delete_component:
            print(f"  Component:   {component_name}:{args.component_version} - ARCHIVED")
        else:
            print(f"  Component:   {component_name}:{args.component_version} - KEPT")
        print()
        
    except Exception as e:
        print(f"\nCleanup failed: {e}")
        raise


if __name__ == "__main__":
    main()
