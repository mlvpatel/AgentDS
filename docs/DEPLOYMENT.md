# AgentDS Deployment Guide

**Production Deployment for All Platforms**

Author: Malav Patel  


---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Docker Deployment](#docker-deployment)
3. [Kubernetes Deployment](#kubernetes-deployment)
4. [AWS Deployment](#aws-deployment)
5. [GCP Deployment](#gcp-deployment)
6. [Azure Deployment](#azure-deployment)
7. [Configuration](#configuration)
8. [Monitoring](#monitoring)

---

## Quick Start

### Local Development

```bash
# Clone and setup
git clone https://github.com/mlvpatel/AgentDS.git
cd AgentDS
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure
cp .env.example .env
nano .env  # Add your API keys

# Run
python -m agentds.web.app
```

Open http://localhost:7860

---

## Docker Deployment

### Single Container

```bash
# Build
docker build -t agentds:latest -f docker/Dockerfile .

# Run
docker run -d \
  --name agentds \
  -p 7860:7860 \
  -e OPENAI_API_KEY=sk-... \
  -v $(pwd)/outputs:/app/outputs \
  agentds:latest
```

### Docker Compose (Full Stack)

```bash
cd docker

# Development (web + redis)
docker-compose up -d web redis

# Full stack (web + api + redis + mlflow + worker)
docker-compose --profile full up -d

# With local LLM (Ollama)
docker-compose --profile local-llm up -d
```

### Environment Variables

Create `.env` file in project root:

```bash
# Required
OPENAI_API_KEY=sk-...

# Optional
ANTHROPIC_API_KEY=sk-ant-...
GROQ_API_KEY=...
REDIS_URL=redis://redis:6379/0
MLFLOW_TRACKING_URI=http://mlflow:5000
```

---

## Kubernetes Deployment

### Prerequisites

- Kubernetes cluster (1.25+)
- kubectl configured
- Helm 3 (optional)

### Deploy with kubectl

```bash
# Create namespace
kubectl create namespace agentds

# Create secrets
kubectl create secret generic agentds-secrets \
  --namespace agentds \
  --from-literal=OPENAI_API_KEY=sk-... \
  --from-literal=ANTHROPIC_API_KEY=sk-ant-...

# Apply manifests
kubectl apply -f docker/k8s/ --namespace agentds

# Check status
kubectl get pods -n agentds
```

### Kubernetes Manifests

**deployment.yaml:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agentds-web
  namespace: agentds
spec:
  replicas: 2
  selector:
    matchLabels:
      app: agentds-web
  template:
    metadata:
      labels:
        app: agentds-web
    spec:
      containers:
        - name: web
          image: ghcr.io/mlvpatel/agentds:latest
          ports:
            - containerPort: 7860
          envFrom:
            - secretRef:
                name: agentds-secrets
          resources:
            requests:
              cpu: "500m"
              memory: "1Gi"
            limits:
              cpu: "2000m"
              memory: "4Gi"
          livenessProbe:
            httpGet:
              path: /
              port: 7860
            initialDelaySeconds: 30
            periodSeconds: 30
          readinessProbe:
            httpGet:
              path: /
              port: 7860
            initialDelaySeconds: 10
            periodSeconds: 10
```

**service.yaml:**
```yaml
apiVersion: v1
kind: Service
metadata:
  name: agentds-web
  namespace: agentds
spec:
  type: LoadBalancer
  ports:
    - port: 80
      targetPort: 7860
  selector:
    app: agentds-web
```

**ingress.yaml:**
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: agentds-ingress
  namespace: agentds
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  ingressClassName: nginx
  tls:
    - hosts:
        - agentds.example.com
      secretName: agentds-tls
  rules:
    - host: agentds.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: agentds-web
                port:
                  number: 80
```

---

## AWS Deployment

### ECS Fargate

```bash
# Create ECR repository
aws ecr create-repository --repository-name agentds

# Build and push
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 123456789.dkr.ecr.us-east-1.amazonaws.com
docker build -t agentds -f docker/Dockerfile .
docker tag agentds:latest 123456789.dkr.ecr.us-east-1.amazonaws.com/agentds:latest
docker push 123456789.dkr.ecr.us-east-1.amazonaws.com/agentds:latest

# Create task definition
aws ecs register-task-definition --cli-input-json file://aws/task-definition.json

# Create service
aws ecs create-service \
  --cluster agentds-cluster \
  --service-name agentds \
  --task-definition agentds:1 \
  --desired-count 2 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx],assignPublicIp=ENABLED}"
```

### AWS CDK

```typescript
import * as cdk from 'aws-cdk-lib';
import * as ecs from 'aws-cdk-lib/aws-ecs';
import * as ecsPatterns from 'aws-cdk-lib/aws-ecs-patterns';

const service = new ecsPatterns.ApplicationLoadBalancedFargateService(this, 'AgentDS', {
  taskImageOptions: {
    image: ecs.ContainerImage.fromRegistry('ghcr.io/mlvpatel/agentds:latest'),
    containerPort: 7860,
    environment: {
      LOG_LEVEL: 'INFO',
    },
    secrets: {
      OPENAI_API_KEY: ecs.Secret.fromSecretsManager(secret, 'OPENAI_API_KEY'),
    },
  },
  cpu: 1024,
  memoryLimitMiB: 2048,
  desiredCount: 2,
});
```

---

## GCP Deployment

### Cloud Run

```bash
# Build with Cloud Build
gcloud builds submit --tag gcr.io/PROJECT_ID/agentds

# Deploy
gcloud run deploy agentds \
  --image gcr.io/PROJECT_ID/agentds \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --min-instances 0 \
  --max-instances 10 \
  --port 7860 \
  --set-env-vars "LOG_LEVEL=INFO" \
  --set-secrets "OPENAI_API_KEY=agentds-openai-key:latest"
```

### GKE

```bash
# Create cluster
gcloud container clusters create agentds-cluster \
  --num-nodes 3 \
  --machine-type e2-standard-4

# Get credentials
gcloud container clusters get-credentials agentds-cluster

# Deploy
kubectl apply -f docker/k8s/
```

---

## Azure Deployment

### Azure Container Instances

```bash
# Create resource group
az group create --name agentds-rg --location eastus

# Create container
az container create \
  --resource-group agentds-rg \
  --name agentds \
  --image ghcr.io/mlvpatel/agentds:latest \
  --cpu 2 \
  --memory 4 \
  --ports 7860 \
  --environment-variables LOG_LEVEL=INFO \
  --secure-environment-variables OPENAI_API_KEY=sk-... \
  --dns-name-label agentds
```

### AKS

```bash
# Create cluster
az aks create \
  --resource-group agentds-rg \
  --name agentds-cluster \
  --node-count 3 \
  --node-vm-size Standard_D4s_v3

# Get credentials
az aks get-credentials --resource-group agentds-rg --name agentds-cluster

# Deploy
kubectl apply -f docker/k8s/
```

---

## Configuration

### Production Settings

```yaml
# config/pipeline_config.yaml (production)
pipeline:
  execution_mode: sequential
  checkpoint:
    enabled: true
    storage: redis

human_in_loop:
  enabled: true
  timeout: 3600  # 1 hour

logging:
  level: INFO
  format: json
  integrations:
    mlflow:
      enabled: true

performance:
  cache:
    enabled: true
    backend: redis
    ttl_seconds: 3600
```

### Resource Recommendations

| Workload | CPU | Memory | Replicas |
|----------|-----|--------|----------|
| Development | 1 | 2GB | 1 |
| Staging | 2 | 4GB | 2 |
| Production | 4 | 8GB | 3+ |
| High Load | 8 | 16GB | 5+ |

---

## Monitoring

### Health Checks

```bash
# HTTP health check
curl http://localhost:7860/

# API health check
curl http://localhost:8000/api/health
```

### Prometheus Metrics

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'agentds'
    static_configs:
      - targets: ['agentds-api:8000']
    metrics_path: /metrics
```

### Grafana Dashboard

Import dashboard ID: `agentds-overview`

Key metrics:
- Pipeline execution time
- Agent success rate
- LLM cost tracking
- Queue depth
- Error rate

### Logging

```bash
# Docker logs
docker logs -f agentds

# Kubernetes logs
kubectl logs -f deployment/agentds-web -n agentds

# Structured query
kubectl logs -f deployment/agentds-web -n agentds | jq 'select(.level=="ERROR")'
```

---

## Security Checklist

- [x] API keys stored in secrets manager
- [x] API key authentication middleware
- [x] Rate limiting (60 req/min default)
- [x] Input validation (path traversal, file size, content-type)
- [ ] HTTPS enabled with valid certificate
- [ ] Network policies configured
- [ ] Non-root container user
- [x] Resource limits set
- [x] Health checks configured
- [x] Logging enabled
- [ ] Backup strategy defined

---

*Author: Malav Patel | malav.patel203@gmail.com*
