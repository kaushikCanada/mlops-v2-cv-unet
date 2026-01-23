# MLOps v2 UNet Semantic Segmentation Pipeline

This repository contains an MLOps pipeline for training and deploying UNet semantic segmentation models using Azure Machine Learning. The pipeline performs data splitting, model training, and prediction without model registration - models are used on-the-fly from storage paths.

## Project Structure

```
mlops-v2-cv-unet/
├── .github/
│   └── workflows/
│       ├── deploy-pipeline-dev.yml
│       ├── deploy-pipeline-uat.yml
│       └── deploy-pipeline-prod.yml
├── config/
│   ├── config-dev.yml
│   ├── config-uat.yml
│   └── config-prod.yml
├── data-science/
│   └── src/
│       ├── prep/          # Data splitting component
│       ├── train/         # Model training component
│       └── predict/       # Prediction component
├── mlops/
│   └── azureml/
│       └── train/
│           ├── split.yaml
│           ├── train.yaml
│           ├── predict.yaml
│           ├── pipeline.yaml
│           └── submit_pipeline.py
└── README.md
```

## Getting Started

### Initial Setup

1. **Clone or Initialize Repository**:
   ```bash
   # If creating a new repository
   cd /path/to/mlops-v2-cv-unet
   git init
   git add .
   git commit -m "Initial commit: UNet MLOps pipeline"
   
   # If pushing to GitHub
   git remote add origin https://github.com/YOUR_USERNAME/mlops-v2-cv-unet.git
   git branch -M main
   git push -u origin main
   ```

2. **Configure Git** (if not already done):
   ```bash
   git config --global user.name "Your Name"
   git config --global user.email "your.email@example.com"
   ```

3. **Push to GitHub**:
   ```bash
   # Create repository on GitHub first, then:
   git remote add origin https://github.com/YOUR_USERNAME/mlops-v2-cv-unet.git
   git branch -M main
   git push -u origin main
   ```

   Or if using SSH:
   ```bash
   git remote add origin git@github.com:YOUR_USERNAME/mlops-v2-cv-unet.git
   git push -u origin main
   ```

## Prerequisites

1. **Azure ML Workspace**: An existing Azure Machine Learning workspace
2. **Compute Clusters**: 
   - CPU compute cluster (for data splitting)
   - GPU compute cluster (for training and prediction)
3. **Environment**: A registered Azure ML environment (e.g., `azml-geo-deep-env:47`)
4. **Data Structure**: Your data should contain:
   - `images/` folder with image files
   - `masks/` folder with mask files
   - `image_mask_mapping.csv` file mapping images to masks
5. **Git/GitHub**: Repository set up on GitHub (for CI/CD workflows)

## Configuration

Edit the configuration files in `config/` directory for each environment:

### `config/config-dev.yml`, `config/config-uat.yml`, `config/config-prod.yml`

```yaml
subscription_id: "YOUR_SUBSCRIPTION_ID"
resource_group: "YOUR_RESOURCE_GROUP"
workspace_name: "YOUR_WORKSPACE_NAME"
cpu_compute_name: "cpu-cluster"
gpu_compute_name: "gpu-cluster"
default_datastore: "workspaceblobstore"
environment_name: "azml-geo-deep-env:47"  # Format: name:version
```

## Authentication Setup

### Azure CLI Authentication (for Local Development)

1. **Login to Azure**:
   ```bash
   az login
   ```

2. **Set Subscription**:
   ```bash
   az account set --subscription <subscription_id>
   ```

3. **Verify Authentication**:
   ```bash
   az account show
   ```

The `submit_pipeline.py` script uses `DefaultAzureCredential` which automatically uses:
- Azure CLI credentials (if logged in)
- Managed Identity (if running on Azure)
- Environment variables (if set)
- Interactive browser login (as fallback)

### GitHub Actions Authentication

1. **Create Service Principal**:
   ```bash
   az ad sp create-for-rbac --name <service_principal_name> \
     --role contributor \
     --scopes /subscriptions/<subscription_id> \
     --sdk-auth
   ```

2. **Add GitHub Secrets**:
   Go to your GitHub repository → Settings → Secrets → Actions, and add:

   - **AZURE_CREDENTIALS**: Full JSON output from the service principal command (all braces included)
   - **ARM_CLIENT_ID**: Client ID from service principal output
   - **ARM_CLIENT_SECRET**: Client secret from service principal output
   - **ARM_SUBSCRIPTION_ID**: Your Azure subscription ID
   - **ARM_TENANT_ID**: Your Azure tenant ID

### GitHub Secrets for Configuration Values

For GitHub Actions workflows to access Azure ML workspace configuration, you need to add environment-specific secrets. The workflows will dynamically create config files from these secrets at runtime.

**Add GitHub Secrets for each environment** (Dev, UAT, Prod):

Go to your GitHub repository → Settings → Secrets → Actions, and add:

**For Dev Environment:**
- `DEV_SUBSCRIPTION_ID`: Your Azure subscription ID for Dev
- `DEV_RESOURCE_GROUP`: Your Azure resource group for Dev
- `DEV_WORKSPACE_NAME`: Your Azure ML workspace name for Dev
- `DEV_CPU_COMPUTE_NAME`: CPU compute cluster name for Dev
- `DEV_GPU_COMPUTE_NAME`: GPU compute cluster name for Dev
- `DEV_DEFAULT_DATASTORE`: Default datastore name for Dev
- `DEV_ENVIRONMENT_NAME`: Azure ML environment name and version (e.g., `azml-geo-deep-env:47`)

**For UAT Environment:**
- `UAT_SUBSCRIPTION_ID`: Your Azure subscription ID for UAT
- `UAT_RESOURCE_GROUP`: Your Azure resource group for UAT
- `UAT_WORKSPACE_NAME`: Your Azure ML workspace name for UAT
- `UAT_CPU_COMPUTE_NAME`: CPU compute cluster name for UAT
- `UAT_GPU_COMPUTE_NAME`: GPU compute cluster name for UAT
- `UAT_DEFAULT_DATASTORE`: Default datastore name for UAT
- `UAT_ENVIRONMENT_NAME`: Azure ML environment name and version (e.g., `azml-geo-deep-env:47`)

**For Prod Environment:**
- `PROD_SUBSCRIPTION_ID`: Your Azure subscription ID for Prod
- `PROD_RESOURCE_GROUP`: Your Azure resource group for Prod
- `PROD_WORKSPACE_NAME`: Your Azure ML workspace name for Prod
- `PROD_CPU_COMPUTE_NAME`: CPU compute cluster name for Prod
- `PROD_GPU_COMPUTE_NAME`: GPU compute cluster name for Prod
- `PROD_DEFAULT_DATASTORE`: Default datastore name for Prod
- `PROD_ENVIRONMENT_NAME`: Azure ML environment name and version (e.g., `azml-geo-deep-env:47`)

**Note**: The GitHub Actions workflows automatically create the config files from these secrets at runtime, so you don't need to commit actual config files to the repository.

## Usage

### Method 1: Direct YAML Submission (Development)

For development and testing, submit the pipeline directly using the Python script:

```bash
python mlops/azureml/train/submit_pipeline.py \
  --env dev \
  --data_dir azureml://subscriptions/<sub_id>/resourcegroups/<rg>/workspaces/<ws>/datastores/workspaceblobstore/paths/<data_path>/ \
  --config_path azureml://subscriptions/<sub_id>/resourcegroups/<rg>/workspaces/<ws>/datastores/workspaceblobstore/paths/<config_path>/config.yml \
  --experiment_name unet_train
```

**Arguments**:
- `--env`: Environment (dev, uat, or prod)
- `--data_dir`: Azure ML path to data directory containing images, masks, and mapping.csv
- `--config_path`: Azure ML path to config YAML file
- `--experiment_name`: Experiment name in Azure ML (optional, default: unet_train)

### Method 2: GitHub Workflow (Production)

1. **Go to GitHub Actions** in your repository
2. **Select the workflow** for your environment:
   - `Deploy UNet Pipeline - Dev`
   - `Deploy UNet Pipeline - UAT`
   - `Deploy UNet Pipeline - Prod`
3. **Click "Run workflow"**
4. **Provide inputs**:
   - `data_dir`: Azure ML path to your data directory
   - `config_path`: Azure ML path to your config YAML file
5. **Run the workflow**

The workflow will:
- Deploy the pipeline as a component in Azure ML
- Optionally run the pipeline with your provided inputs
- Create/update pipeline endpoints (requires manual setup for REST API)

### Method 3: Pipeline Endpoint (REST API)

After deploying via GitHub workflow, you can create a Pipeline Endpoint for REST API access:

1. **Using Azure ML CLI**:
   ```bash
   az ml pipeline-endpoint create \
     --name unet-pipeline-endpoint-dev \
     --pipeline-id <pipeline_component_id> \
     --resource-group <resource_group> \
     --workspace <workspace_name>
   ```

2. **Using Azure ML Studio**:
   - Navigate to your workspace
   - Go to Endpoints → Pipeline endpoints
   - Create new endpoint and select your pipeline component

3. **Invoke via REST API**:
   ```bash
   curl -X POST \
     -H "Authorization: Bearer <token>" \
     -H "Content-Type: application/json" \
     -d '{
       "InputData": {
         "data_dir": "azureml://...",
         "config_path": "azureml://..."
       }
     }' \
     https://<endpoint_url>/v1.0/subscriptions/<sub>/resourceGroups/<rg>/providers/Microsoft.MachineLearningServices/workspaces/<ws>/pipelineEndpoints/<endpoint_name>/jobs
   ```

## Pipeline Components

### 1. Split Component (`split.yaml`)
- **Purpose**: Splits `image_mask_mapping.csv` into train/val/test splits
- **Inputs**: 
  - `raw_data`: Folder containing data and mapping.csv
  - `config_data`: Config YAML file
- **Outputs**: 
  - `prep_output`: Folder with split CSV files (train, val, test)
- **Compute**: CPU cluster
- **Note**: Does not move actual data files, only creates split CSV files

### 2. Train Component (`train.yaml`)
- **Purpose**: Trains UNet model using PyTorch Lightning
- **Inputs**:
  - `train_data`: Original data directory
  - `prep_input`: Split CSV files from prep step
  - `config_data`: Config YAML file
- **Outputs**:
  - `model_output`: Model checkpoints and artifacts (NOT registered)
- **Compute**: GPU cluster

### 3. Predict Component (`predict.yaml`)
- **Purpose**: Runs predictions using trained model
- **Inputs**:
  - `model_path`: Path to trained model checkpoints
  - `data_dir`: Data directory for prediction
  - `config_data`: Config YAML file
- **Outputs**:
  - `predictions_output`: Prediction results
- **Compute**: GPU cluster

## Data Format

Your data directory should have the following structure:

```
data_dir/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── masks/
│   ├── mask1.png
│   ├── mask2.png
│   └── ...
└── image_mask_mapping.csv
```

The `image_mask_mapping.csv` should have columns:
- `Image` or `image_filename`: Name of image file
- `Mask` or `mask_filename`: Name of mask file

Example:
```csv
image_filename,mask_filename
image1.jpg,mask1.png
image2.jpg,mask2.png
```

## Config YAML Format

Your config YAML should follow this structure (see `config_xyz.yaml` in UnetProject for reference):

```yaml
data:
  batch_size: 8
  num_workers: 4

model:
  backbone: "resnet50"
  decoder: "unet"
  num_classes: 5
  pretrained: true

training:
  num_epochs: 100
  learning_rate: 0.001

# ... other config sections
```

## Environment Setup

The pipeline uses an existing Azure ML environment. Make sure:
1. The environment is already registered in your workspace
2. The environment name and version match your config file (e.g., `azml-geo-deep-env:47`)
3. The environment contains all required dependencies (PyTorch, Lightning, TorchGeo, etc.)

If the environment doesn't exist, create it manually or update the config file with the correct environment name.

## Troubleshooting

### Authentication Issues

**Azure CLI**:
- Ensure you're logged in: `az account show`
- Check subscription: `az account list`

**GitHub Actions**:
- Verify all secrets are set correctly
- Check service principal has contributor role on subscription/resource group

### Pipeline Submission Issues

- Verify compute clusters exist and are running
- Check environment name and version are correct
- Ensure data paths are accessible from the workspace
- Check Azure ML workspace permissions

### Model Loading Issues

- Verify model checkpoint path is correct
- Check model architecture matches config
- Ensure PyTorch Lightning version compatibility

## Git Workflow

### Making Changes

1. **Create a branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** and commit:
   ```bash
   git add .
   git commit -m "Description of your changes"
   ```

3. **Push to GitHub**:
   ```bash
   git push origin feature/your-feature-name
   ```

4. **Create a Pull Request** on GitHub to merge into main branch

### Updating Configuration

**Important**: Before committing, ensure you don't commit sensitive information:
- Review `config/config-*.yml` files - they contain placeholder values
- Never commit actual subscription IDs, secrets, or credentials
- Consider using environment variables or Azure Key Vault for sensitive data

### Branch Strategy

- `main`: Production-ready code
- `dev`: Development branch
- `feature/*`: Feature branches
- `fix/*`: Bug fix branches

## Contributing

This project follows standard MLOps practices. When making changes:
1. Test locally using `submit_pipeline.py`
2. Update config files for your environment
3. Test GitHub workflows in dev environment first
4. Document any new dependencies or requirements
5. Follow the Git workflow above for committing changes

## License

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
