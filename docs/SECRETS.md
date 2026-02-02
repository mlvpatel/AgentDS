# Secrets Management Guide

This guide covers best practices for managing secrets in AgentDS, including integration with enterprise secret management solutions.

---

## Environment Variables (Development)

For local development, AgentDS uses `.env` files:

```bash
# Copy the example file
cp .env.example .env

# Edit with your keys
OPENAI_API_KEY=sk-xxx
ANTHROPIC_API_KEY=sk-ant-xxx
```

> [!CAUTION]
> Never commit `.env` files to version control. They are already in `.gitignore`.

---

## HashiCorp Vault Integration

### Installation

```bash
pip install hvac
```

### Configuration

```python
# config/secrets.py
import hvac
import os

def get_vault_client():
    """Create authenticated Vault client."""
    client = hvac.Client(
        url=os.getenv("VAULT_ADDR", "http://localhost:8200"),
        token=os.getenv("VAULT_TOKEN"),
    )
    
    # Or use AppRole auth for production
    if os.getenv("VAULT_ROLE_ID"):
        client.auth.approle.login(
            role_id=os.getenv("VAULT_ROLE_ID"),
            secret_id=os.getenv("VAULT_SECRET_ID"),
        )
    
    return client

def load_secrets():
    """Load secrets from Vault into environment."""
    client = get_vault_client()
    
    # Read secrets from KV v2 engine
    secrets = client.secrets.kv.v2.read_secret_version(
        path="agentds/api-keys",
        mount_point="secret",
    )
    
    data = secrets["data"]["data"]
    
    os.environ["OPENAI_API_KEY"] = data.get("openai_api_key", "")
    os.environ["ANTHROPIC_API_KEY"] = data.get("anthropic_api_key", "")
    # Add other keys as needed
```

### Usage

```python
# In your application startup
from config.secrets import load_secrets

if os.getenv("USE_VAULT", "false").lower() == "true":
    load_secrets()

# Then initialize settings normally
from agentds.core.config import Settings
settings = Settings()
```

### Storing Secrets in Vault

```bash
# Write secrets to Vault
vault kv put secret/agentds/api-keys \
    openai_api_key="sk-xxx" \
    anthropic_api_key="sk-ant-xxx" \
    gemini_api_key="xxx"
```

---

## AWS Secrets Manager

### Installation

```bash
pip install boto3
```

### Configuration

```python
# config/aws_secrets.py
import json
import boto3
from botocore.exceptions import ClientError

def load_aws_secrets(secret_name: str = "agentds/api-keys", region: str = "us-east-1"):
    """Load secrets from AWS Secrets Manager."""
    client = boto3.client("secretsmanager", region_name=region)
    
    try:
        response = client.get_secret_value(SecretId=secret_name)
        secrets = json.loads(response["SecretString"])
        
        # Set environment variables
        for key, value in secrets.items():
            os.environ[key.upper()] = value
            
    except ClientError as e:
        if e.response["Error"]["Code"] == "ResourceNotFoundException":
            raise ValueError(f"Secret {secret_name} not found")
        raise
```

### Creating Secrets in AWS

```bash
# Using AWS CLI
aws secretsmanager create-secret \
    --name agentds/api-keys \
    --secret-string '{
        "openai_api_key": "sk-xxx",
        "anthropic_api_key": "sk-ant-xxx"
    }'
```

### IAM Policy

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "secretsmanager:GetSecretValue"
            ],
            "Resource": "arn:aws:secretsmanager:*:*:secret:agentds/*"
        }
    ]
}
```

---

## Google Cloud Secret Manager

### Installation

```bash
pip install google-cloud-secret-manager
```

### Configuration

```python
# config/gcp_secrets.py
from google.cloud import secretmanager
import os

def load_gcp_secrets(project_id: str, secret_id: str = "agentds-api-keys"):
    """Load secrets from GCP Secret Manager."""
    client = secretmanager.SecretManagerServiceClient()
    
    # Access the latest version
    name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
    response = client.access_secret_version(request={"name": name})
    
    # Parse JSON payload
    import json
    secrets = json.loads(response.payload.data.decode("UTF-8"))
    
    for key, value in secrets.items():
        os.environ[key.upper()] = value
```

### Creating Secrets in GCP

```bash
# Create secret
gcloud secrets create agentds-api-keys \
    --replication-policy="automatic"

# Add version with data
echo -n '{"openai_api_key":"sk-xxx","anthropic_api_key":"sk-ant-xxx"}' | \
    gcloud secrets versions add agentds-api-keys --data-file=-
```

---

## Azure Key Vault

### Installation

```bash
pip install azure-identity azure-keyvault-secrets
```

### Configuration

```python
# config/azure_secrets.py
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
import os

def load_azure_secrets(vault_url: str):
    """Load secrets from Azure Key Vault."""
    credential = DefaultAzureCredential()
    client = SecretClient(vault_url=vault_url, credential=credential)
    
    # List of secrets to load
    secret_names = [
        ("openai-api-key", "OPENAI_API_KEY"),
        ("anthropic-api-key", "ANTHROPIC_API_KEY"),
    ]
    
    for azure_name, env_name in secret_names:
        try:
            secret = client.get_secret(azure_name)
            os.environ[env_name] = secret.value
        except Exception:
            pass  # Secret not found
```

### Creating Secrets in Azure

```bash
# Create Key Vault
az keyvault create \
    --name agentds-vault \
    --resource-group myResourceGroup \
    --location eastus

# Add secrets
az keyvault secret set \
    --vault-name agentds-vault \
    --name openai-api-key \
    --value "sk-xxx"
```

---

## Secret Rotation

### Recommended Rotation Schedule

| Secret Type | Rotation Frequency | Notes |
|-------------|-------------------|-------|
| LLM API Keys | 90 days | Check provider docs |
| Database Passwords | 30 days | Use automated rotation |
| JWT Signing Keys | 180 days | Support key rollover |
| API Keys (our app) | On demand | Revoke when compromised |

### Implementing Rotation

```python
# Example: AWS Secrets Manager rotation
import boto3

def rotate_secret(secret_id: str, new_value: str):
    """Rotate a secret value."""
    client = boto3.client("secretsmanager")
    
    # Update secret
    client.update_secret(
        SecretId=secret_id,
        SecretString=new_value,
    )
    
    # Invalidate cache
    from agentds.core.cache_layer import get_cache
    get_cache().delete("secrets:api_keys")
```

---

## Kubernetes Secrets

For Kubernetes deployments:

```yaml
# kubernetes/secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: agentds-secrets
type: Opaque
stringData:
  OPENAI_API_KEY: "sk-xxx"
  ANTHROPIC_API_KEY: "sk-ant-xxx"
---
# Use in deployment
apiVersion: apps/v1
kind: Deployment
spec:
  template:
    spec:
      containers:
      - name: agentds
        envFrom:
        - secretRef:
            name: agentds-secrets
```

> [!TIP]
> Use External Secrets Operator to sync from Vault/AWS/GCP/Azure to Kubernetes Secrets.

---

## Best Practices

1. **Never log secrets** - Use structured logging that redacts sensitive fields
2. **Minimal scope** - Only request permissions for secrets you need
3. **Audit access** - Enable audit logging for secret access
4. **Encrypt at rest** - Use encrypted secret stores
5. **Rotate regularly** - Implement automated rotation
6. **Use short-lived tokens** - Prefer time-limited credentials
7. **Separate environments** - Use different secrets per environment
