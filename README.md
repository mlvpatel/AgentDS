# 🤖 AgentDS

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type checked: mypy](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](http://mypy-lang.org/)

> **Your autonomous AI-powered data science assistant - from raw data to production-ready ML models**

AgentDS is an intelligent, multi-agent framework that automates the entire data science workflow. Built with modern LLM orchestration frameworks like LangGraph and Pydantic AI, it transforms how data scientists work by providing autonomous agents that handle everything from data cleaning to model deployment.

---

## ✨ Features

### 🎯 Core Capabilities

- **🔄 Automated ML Pipeline**: End-to-end automation from data ingestion to model deployment
- **🤖 Multi-Agent Architecture**: Specialized agents for each phase of the data science lifecycle
- **🧠 LLM-Powered Intelligence**: Leverages multiple LLM providers (OpenAI, Anthropic, Google, etc.)
- **📊 Interactive Dashboard**: Beautiful Gradio web interface for monitoring and control
- **🔌 Extensible Integrations**: n8n workflows, cloud storage, and custom APIs
- **📈 MLOps Ready**: Built-in experiment tracking, model versioning, and deployment tools

### 🛠️ Agent Capabilities

| Agent | Purpose | Key Features |
|-------|---------|--------------|
| **Data Loader** | Intelligent data ingestion | Auto-detection, multi-source support, validation |
| **Data Cleaner** | Data quality improvement | Missing values, outliers, type correction |
| **EDA Copilot** | Exploratory analysis | Statistical insights, visualization, profiling |
| **Feature Engineer** | Feature creation & selection | Automated feature engineering, selection algorithms |
| **AutoML** | Model training & optimization | Algorithm selection, hyperparameter tuning |
| **Drift Monitor** | Production monitoring | Data drift, concept drift, performance tracking |
| **Cloud Deploy** | Model deployment | Multi-cloud support, containerization, scaling |
| **DevOps** | CI/CD automation | GitHub Actions, Docker, Kubernetes configs |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- Redis (for caching and job queue)
- At least one LLM API key (OpenAI, Anthropic, Google, etc.)

### Installation

```bash
# Clone the repository
git clone https://github.com/mlvpatel/AgentDS.git
cd AgentDS

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .

# For development
pip install -e ".[dev]"

# Copy and configure environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Basic Usage

#### CLI Interface

```bash
# Run a complete ML pipeline
agentds run --data data.csv --target target_column --task classification

# Individual agent operations
agentds clean --data data.csv --output cleaned_data.csv
agentds eda --data data.csv --report eda_report.html
agentds train --data data.csv --target target --model xgboost
```

#### Python API

```python
from agentds.workflows.pipeline import DataSciencePipeline

# Initialize pipeline
pipeline = DataSciencePipeline(
    data_path="data.csv",
    target_column="target",
    task_type="classification"
)

# Run complete workflow
results = await pipeline.run()

print(f"Best Model: {results.best_model}")
print(f"Accuracy: {results.metrics['accuracy']:.4f}")
```

#### Web Interface

```bash
# Launch Gradio dashboard
agentds web

# Access at http://localhost:7860
```

---

## 📊 Architecture

### System Overview

```
                            🌐 User Interface Layer
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║   🖥️  Web Dashboard (Gradio)      🔌 REST API (Litestar)    ║
    ║   • Interactive UI                 • Webhooks                ║
    ║   • Real-time Monitoring           • External Integrations   ║
    ║                                                               ║
    ╚═══════════════════════════════╤═══════════════════════════════╝
                                    │
                                    ▼
    ╔═══════════════════════════════════════════════════════════════╗
    ║              🧠 Orchestration & Intelligence Layer            ║
    ║                                                               ║
    ║   ┌──────────────────┐         ┌──────────────────┐         ║
    ║   │   LangGraph      │◄───────►│  Pydantic AI     │         ║
    ║   │   Workflow       │         │  Type-Safe       │         ║
    ║   │   Orchestrator   │         │  Agents          │         ║
    ║   └──────────────────┘         └──────────────────┘         ║
    ║                                                               ║
    ╚═══════════════════════════════╤═══════════════════════════════╝
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
    ╔═══════════════════╗ ╔═══════════════╗ ╔══════════════════╗
    ║   🤖 AI Agents    ║ ║  ⚙️  Core     ║ ║  🔗 Integration  ║
    ║                   ║ ║   Services    ║ ║    Layer         ║
    ╠═══════════════════╣ ╠═══════════════╣ ╠══════════════════╣
    ║ • Data Loader     ║ ║ • LLM Gateway ║ ║ • n8n Workflows  ║
    ║ • Data Cleaner    ║ ║ • Config Mgr  ║ ║ • Cloud Storage  ║
    ║ • EDA Copilot     ║ ║ • Cache Layer ║ ║   - AWS S3       ║
    ║ • Feature Eng.    ║ ║ • Job Queue   ║ ║   - GCS          ║
    ║ • AutoML          ║ ║ • Artifacts   ║ ║   - Azure Blob   ║
    ║ • Drift Monitor   ║ ║ • Logger      ║ ║ • Notifications  ║
    ║ • Cloud Deploy    ║ ║               ║ ║   - Email        ║
    ║ • DevOps          ║ ║               ║ ║   - Slack        ║
    ║                   ║ ║               ║ ║   - Discord      ║
    ╚═══════════════════╝ ╚═══════════════╝ ╚══════════════════╝
                    │               │               │
                    └───────────────┼───────────────┘
                                    ▼
    ╔═══════════════════════════════════════════════════════════════╗
    ║              💾 Data & Infrastructure Layer                    ║
    ║                                                               ║
    ║  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       ║
    ║  │  Redis   │  │ DuckDB   │  │  Polars  │  │  MLflow  │       ║
    ║  │  Cache   │  │ Analytics│  │DataFrames│  │ Tracking │       ║
    ║  └──────────┘  └──────────┘  └──────────┘  └──────────┘       ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
```

### Data Flow Pipeline

```
📊 Raw Data → 🧹 Clean → 🔍 Analyze → ⚙️ Engineer → 🤖 Train → 📈 Monitor → 🚀 Deploy
    │            │          │            │            │           │           │
    ▼            ▼          ▼            ▼            ▼           ▼           ▼
Data Loader  Cleaner   EDA Copilot   Feature    AutoML      Drift       Cloud
  Agent       Agent      Agent        Engineer    Agent      Monitor     Deploy
                                      Agent                   Agent       Agent
```

### Agent Collaboration Flow

```
                              ┌─────────────────┐
                              │  User Request   │
                              └────────┬────────┘
                                       │
                                       ▼
                         ┌─────────────────────────┐
                         │  Workflow Orchestrator  │
                         └─────────────────────────┘
                                       │
                ┌──────────────────────┼──────────────────────┐
                ▼                      ▼                      ▼
    ┏━━━━━━━━━━━━━━━━┓    ┏━━━━━━━━━━━━━━━━┓    ┏━━━━━━━━━━━━━━━━┓
    ┃  Phase 1: Prep ┃───►┃ Phase 2: Build ┃───►┃ Phase 3: Deploy┃
    ┗━━━━━━━━━━━━━━━━┛    ┗━━━━━━━━━━━━━━━━┛    ┗━━━━━━━━━━━━━━━━┛
         │  │  │              │  │  │              │  │  │
         ▼  ▼  ▼              ▼  ▼  ▼              ▼  ▼  ▼
    Loader Clean EDA      Feature Auto Optimize  Deploy Monitor DevOps
                          Engineer ML            
```

---

## 🔧 Configuration

### LLM Providers

Configure your preferred LLM provider in `config/llm_config.yaml`:

```yaml
default_provider: openai
default_model: gpt-4-turbo-preview

providers:
  openai:
    model: gpt-4-turbo-preview
    temperature: 0.1
    max_tokens: 4096
  
  anthropic:
    model: claude-3-5-sonnet-20241022
    temperature: 0.1
    max_tokens: 8192
```

### Pipeline Configuration

Customize pipeline behavior in `config/pipeline_config.yaml`:

```yaml
data_loading:
  auto_detect_types: true
  max_rows: null
  
automl:
  max_trials: 50
  timeout_minutes: 60
  cv_folds: 5
  
deployment:
  container_registry: docker.io
  enable_monitoring: true
```

---

## 📖 Documentation

- **[User Manual](docs/USER_MANUAL.md)** - Complete usage guide
- **[API Reference](docs/API_REFERENCE.md)** - Detailed API documentation
- **[Architecture](docs/ARCHITECTURE.md)** - System design and patterns
- **[LLM Providers](docs/LLM_PROVIDERS.md)** - Supported LLM configurations
- **[Deployment Guide](docs/DEPLOYMENT.md)** - Production deployment
- **[n8n Integration](docs/N8N_GUIDE.md)** - Workflow automation setup
- **[Troubleshooting](docs/TROUBLESHOOTING.md)** - Common issues and solutions

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=agentds --cov-report=html

# Run specific test module
pytest tests/test_agents.py -v
```

---

## 🐳 Docker Deployment

### Quick Start with Docker Compose

```bash
# Start all services
docker-compose -f docker/docker-compose.yml up -d

# View logs
docker-compose -f docker/docker-compose.yml logs -f

# Stop services
docker-compose -f docker/docker-compose.yml down
```

### Kubernetes Deployment

```bash
# Apply Kubernetes manifests
kubectl apply -f docker/k8s/deployment.yaml
kubectl apply -f docker/k8s/service.yaml

# Check status
kubectl get pods
kubectl get services
```

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Install pre-commit hooks
pre-commit install

# Run linting
ruff check .

# Run type checking
mypy agentds/
```

### Code Style

- Follow PEP 8 guidelines
- Use type hints for all functions
- Write docstrings for public APIs
- Maintain test coverage above 80%

---

## 📝 Changelog

See [CHANGELOG.md](docs/CHANGELOG.md) for version history and updates.

---

## 🛡️ License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

Built with amazing open-source tools:

- **[LangGraph](https://github.com/langchain-ai/langgraph)** - Agent orchestration
- **[Pydantic AI](https://github.com/pydantic/pydantic-ai)** - Type-safe AI framework
- **[LiteLLM](https://github.com/BerriAI/litellm)** - Unified LLM API
- **[Polars](https://github.com/pola-rs/polars)** - Lightning-fast dataframes
- **[Litestar](https://github.com/litestar-org/litestar)** - Modern web framework
- **[Gradio](https://github.com/gradio-app/gradio)** - ML web interfaces

---

## 📧 Contact & Support

- **Author**: Malav Patel
- **Email**: malav.patel203@gmail.com
- **GitHub**: [@mlvpatel](https://github.com/mlvpatel)
- **Issues**: [GitHub Issues](https://github.com/mlvpatel/AgentDS/issues)

---

## ⭐ Star History

If you find this project useful, please consider giving it a star! It helps others discover the project.

---

<div align="center">

**Made with ❤️ by data scientists, for data scientists**

[⬆ Back to Top](#-agentds)

</div>
