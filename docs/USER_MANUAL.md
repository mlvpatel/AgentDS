# AgentDS User Manual

**Complete Guide to Using AgentDS**

Author: Malav Patel  


---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Web Interface Guide](#web-interface-guide)
3. [Command Line Interface](#command-line-interface)
4. [Configuration Guide](#configuration-guide)
5. [Working with Agents](#working-with-agents)
6. [Human-in-the-Loop Controls](#human-in-the-loop-controls)
7. [Output Artifacts](#output-artifacts)
8. [Troubleshooting](#troubleshooting)

---

## Getting Started

### Prerequisites

Before using AgentDS, ensure you have:

1. **Python 3.10+** installed
2. **Redis** server running (for caching and job queue)
3. At least one **LLM API key** (OpenAI, Anthropic, etc.) OR **Ollama** for local inference
4. Sufficient disk space for model artifacts (recommended: 10GB+)

### Quick Installation

```bash
# Clone repository
git clone https://github.com/mlvpatel/agentds-v2.git
cd agentds-v2

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment
cp .env.example .env
nano .env  # Add your API keys
```

### First Run

```bash
# Start Redis (if not running)
redis-server &

# Launch web interface
python -m agentds.web.app

# Open browser to http://localhost:7860
```

---

## Web Interface Guide

### Pipeline Tab

The Pipeline tab is where you'll spend most of your time:

#### 1. Configuration Panel (Left Side)

**Data Source**:
- Upload a file (CSV, Parquet, JSON, Excel)
- OR enter a URL (S3, GCS, HTTP)

**Task Description**:
- Describe your ML task in natural language
- Example: "Predict customer churn based on transaction history and demographics"

**Advanced Options**:
- **Phases**: Select which phases to run (Build, Deploy, Learn)
- **Human-in-Loop**: Enable/disable approval at each step
- **LLM Preset**: Choose cost/quality tradeoff

#### 2. Progress Panel (Right Side)

**Status Display**:
- Shows current pipeline status
- Real-time progress updates

**Progress Bar**:
- Visual indication of completion percentage
- ETA for remaining time

**Current Agent Output**:
- Summary of current agent's work
- Artifacts generated
- Recommendations

**Action Buttons** (when awaiting approval):
- **Approve & Continue**: Accept output and proceed
- **Re-run**: Execute agent again
- **Skip**: Skip this agent
- **Stop Pipeline**: Cancel entire pipeline

### Agents Tab

View status of all 10 agents organized by phase:

**Build Phase**:
- DataLoaderAgent - Data ingestion
- DataCleaningAgent - Data quality
- EDACopilotAgent - Exploratory analysis
- FeatureEngineerAgent - Feature preprocessing
- AutoMLAgent - Model training

**Deploy Phase**:
- APIWrapperAgent - API code generation
- DevOpsAgent - Docker configuration
- CloudDeployAgent - Deployment

**Learn Phase**:
- DriftMonitorAgent - Model monitoring
- OptimizationAgent - Self-improvement

### Configuration Tab

Manage your settings:

**API Keys**:
- Enter keys for OpenAI, Anthropic, Groq, etc.
- Keys are stored securely

**Model Settings**:
- Default model selection
- Temperature configuration

### Jobs Tab

View job history:
- Job ID, name, status
- Creation time, duration
- Click to view details

### Logs Tab

System logs for debugging:
- Real-time log streaming
- Filter by level (DEBUG, INFO, ERROR)

---

## Command Line Interface

### Available Commands

```bash
# Show help
agentds --help

# Check system status
agentds status

# View configuration
agentds config

# List recent jobs
agentds jobs

# Run full pipeline
agentds run data.csv -t "Predict sales" -o ./outputs

# Run single agent
agentds agent DataLoaderAgent data.csv

# Launch web interface
agentds web --port 7860

# Launch API server
agentds api --port 8000
```

### Pipeline Execution

```bash
# Basic usage
agentds run mydata.csv -t "Classification task"

# Specify output directory
agentds run mydata.csv -t "Classification task" -o ./my_outputs

# Select specific phases
agentds run mydata.csv -t "Task" -p build -p deploy

# Disable human-in-the-loop
agentds run mydata.csv -t "Task" --no-hitl
```

### Single Agent Execution

```bash
# Run DataLoaderAgent
agentds agent DataLoaderAgent data.csv -o ./outputs

# Run AutoMLAgent (requires previous outputs)
agentds agent AutoMLAgent ./outputs/clean_data.parquet
```

---

## Configuration Guide

### Environment Variables

Create a `.env` file with your settings:

```bash
# LLM API Keys (at least one required)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GROQ_API_KEY=...

# Optional providers
MISTRAL_API_KEY=...
TOGETHER_API_KEY=...

# Local inference
OLLAMA_API_BASE=http://localhost:11434

# Infrastructure
REDIS_URL=redis://localhost:6379/0
MLFLOW_TRACKING_URI=http://localhost:5000

# Application settings
DEBUG=false
LOG_LEVEL=INFO
HUMAN_IN_LOOP=true
```

### LLM Configuration

Edit `config/llm_config.yaml`:

```yaml
# Default model for all agents
default_model: openai/gpt-4.1-mini

# Agent-specific assignments
agent_llm_mapping:
  DataLoaderAgent:
    model: groq/llama-4-scout
    temperature: 0.0
  AutoMLAgent:
    model: openai/gpt-4.1
    temperature: 0.0
  # ... more agents

# Fallback chain
fallback_chains:
  default:
    - anthropic/claude-sonnet-4-20250514
    - openai/gpt-4.1
    - google/gemini-2.5-pro
    - deepseek/deepseek-v3
    - groq/llama-4-maverick
```

### Presets

Choose a preset based on your needs:

| Preset | Description | Cost | Quality |
|--------|-------------|------|---------|
| `budget` | Lowest cost, uses Groq/Ollama | $ | Good |
| `balanced` | Mix of cloud and local | $$ | Very Good |
| `quality` | Best models only | $$$ | Excellent |
| `local` | Ollama only, no API costs | Free | Good |

---

## Working with Agents

### Agent Execution Order

1. **DataLoaderAgent**: Loads your data from any source
2. **DataCleaningAgent**: Cleans and validates data
3. **EDACopilotAgent**: Generates exploratory analysis
4. **FeatureEngineerAgent**: Creates preprocessing pipeline
5. **AutoMLAgent**: Trains and selects best model
6. **APIWrapperAgent**: Generates API server code
7. **DevOpsAgent**: Creates Docker configuration
8. **CloudDeployAgent**: Deploys to cloud
9. **DriftMonitorAgent**: Monitors production model
10. **OptimizationAgent**: Self-improves agent prompts

### What Each Agent Does

#### DataLoaderAgent
- Supports: CSV, Parquet, JSON, Excel, S3, GCS, databases
- Outputs: `raw_data.parquet`
- Automatically detects encoding and schema

#### DataCleaningAgent
- Removes duplicates
- Handles missing values
- Fixes data types
- Validates against schema
- Outputs: `clean_data.parquet`, `quality_report.json`

#### EDACopilotAgent
- Univariate and bivariate analysis
- Correlation analysis
- Distribution visualization
- AI-generated insights
- Outputs: `eda_dashboard.html`

#### FeatureEngineerAgent
- Encoding (label, one-hot, target)
- Scaling (standard, minmax, robust)
- Imputation strategies
- Feature selection
- Outputs: `preprocessing_pipeline.pkl`

#### AutoMLAgent
- Model selection (XGBoost, LightGBM, sklearn)
- Hyperparameter optimization (Optuna)
- Cross-validation
- SHAP explanations
- Outputs: `best_model.pkl`, `metrics.json`

#### APIWrapperAgent
- Litestar API generation
- Health check endpoint
- Prediction endpoint
- Batch inference
- Outputs: `app.py`, `requirements.txt`

#### DevOpsAgent
- Multi-stage Dockerfile
- docker-compose.yml
- .dockerignore
- GitHub Actions (optional)
- Kubernetes manifests (optional)

#### CloudDeployAgent
- Docker local deployment
- AWS ECS/Fargate
- GCP Cloud Run
- Azure Container Instances
- Kubernetes

#### DriftMonitorAgent
- Data drift detection (PSI)
- Prediction drift monitoring
- Feature-level analysis
- Alert generation
- Outputs: `drift_report.json`

#### OptimizationAgent
- APO (Automatic Prompt Optimization)
- Prompt analysis and critique
- Automatic prompt rewriting
- Performance tracking
- Outputs: `optimized_prompts.json`

---

## Human-in-the-Loop Controls

### Available Actions

At each agent checkpoint, you can:

| Action | Description | When to Use |
|--------|-------------|-------------|
| **Approve & Continue** | Accept output, proceed to next agent | Output looks good |
| **Re-run** | Execute same agent again | Minor issue, try again |
| **Re-run with Feedback** | Re-run with your instructions | Need specific changes |
| **Skip** | Skip this agent | Don't need this step |
| **Stop Pipeline** | Cancel entire pipeline | Major issue found |
| **Download Output** | Download current artifacts | Save intermediate results |
| **Rollback** | Return to previous checkpoint | Undo last step |

### Providing Feedback

When using "Re-run with Feedback":

```
Good feedback examples:
- "Use median imputation instead of mean for the age column"
- "Exclude the 'id' column from feature engineering"
- "Focus on precision over recall for the model"

Poor feedback examples:
- "Do better" (too vague)
- "Fix it" (no specific instructions)
```

### Auto-Approve Mode

For low-risk agents, you can enable auto-approve:

```yaml
# In pipeline_config.yaml
human_in_loop:
  auto_approve:
    enabled: true
    risk_levels:
      - LOW
    agents:
      - DataLoaderAgent
      - DriftMonitorAgent
```

---

## Output Artifacts

### Directory Structure

```
outputs/
└── {job_id}/
    ├── DataLoaderAgent/
    │   └── raw_data.parquet
    ├── DataCleaningAgent/
    │   ├── clean_data.parquet
    │   └── quality_report.json
    ├── EDACopilotAgent/
    │   └── eda_dashboard.html
    ├── FeatureEngineerAgent/
    │   └── preprocessing_pipeline.pkl
    ├── AutoMLAgent/
    │   ├── best_model.pkl
    │   └── metrics.json
    ├── APIWrapperAgent/
    │   ├── app.py
    │   └── requirements.txt
    └── DevOpsAgent/
        ├── Dockerfile
        └── docker-compose.yml
```

### Using the Model

```python
import joblib
import polars as pl

# Load pipeline and model
pipeline = joblib.load("outputs/{job_id}/FeatureEngineerAgent/preprocessing_pipeline.pkl")
model = joblib.load("outputs/{job_id}/AutoMLAgent/best_model.pkl")

# Make prediction
data = pl.read_csv("new_data.csv")
X = pipeline.transform(data)
predictions = model.predict(X)
```

---

## Troubleshooting

### Common Issues

#### "No LLM providers configured"

```bash
# Check your .env file has valid API keys
cat .env | grep API_KEY

# Or use Ollama for local inference
ollama serve
export OLLAMA_API_BASE=http://localhost:11434
```

#### "Redis connection failed"

```bash
# Start Redis server
redis-server

# Or check if running
redis-cli ping
```

#### "Data loading failed"

- Check file path is correct
- Verify file format is supported
- Check file encoding (UTF-8 recommended)
- Ensure file is not corrupted

#### "Agent execution timeout"

- Increase timeout in `pipeline_config.yaml`
- Check LLM API rate limits
- Consider using faster model preset

#### "Out of memory"

- Enable data sampling for large datasets
- Reduce model complexity
- Use streaming data loading

### Getting Help

1. Check the logs: `agentds` logs tab or `logs/` directory
2. Run diagnostics: `agentds status`
3. Check documentation: `docs/` folder
4. Open an issue: https://github.com/mlvpatel/AgentDS/issues

---

## Best Practices

### Data Preparation

1. Clean obvious errors before uploading
2. Use consistent column naming
3. Remove personal/sensitive information
4. Document data dictionary

### Task Description

1. Be specific about the prediction target
2. Mention important constraints
3. Specify evaluation metrics preference
4. Note any domain knowledge

### Resource Management

1. Start with `budget` preset for testing
2. Use `quality` preset for production
3. Monitor LLM costs via the dashboard
4. Clean up old job artifacts regularly

---

*This manual is part of AgentDS*

*Author: Malav Patel | malav.patel203@gmail.com*
