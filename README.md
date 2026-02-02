<div align="center">

# 🤖 AgentDS

### Your Autonomous AI-Powered Data Science Assistant

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Code style: Ruff](https://img.shields.io/badge/Code%20Style-Ruff-000000?style=for-the-badge)](https://github.com/astral-sh/ruff)
[![Type checked: mypy](https://img.shields.io/badge/Type%20Checked-mypy-blue?style=for-the-badge)](http://mypy-lang.org/)

**From raw data to production-ready ML models — fully automated**

[Quick Start](#-quick-start) • [Features](#-key-features) • [Documentation](#-documentation) • [Contributing](#-contributing)

---

</div>

## 🎯 What is AgentDS?

AgentDS is an intelligent **multi-agent framework** that automates the entire data science workflow. Built with modern LLM orchestration (LangGraph + Pydantic AI), it provides **10 specialized agents** that handle everything from data cleaning to model deployment.

```
📊 Your Data ──► 🤖 AgentDS ──► 🚀 Production ML Model
```

<details>
<summary><b>🔥 Why AgentDS?</b></summary>
<br>

| Traditional ML Pipeline | With AgentDS |
|------------------------|--------------|
| ❌ Manual data cleaning | ✅ Automated quality checks |
| ❌ Write boilerplate code | ✅ Generated pipelines |
| ❌ Trial-and-error modeling | ✅ AutoML with Optuna |
| ❌ Manual Docker setup | ✅ One-click containerization |
| ❌ Complex deployment | ✅ Multi-cloud ready |

</details>

---

## ✨ Key Features

<table>
<tr>
<td width="50%">

### 🤖 10 Specialized Agents

| Agent | Purpose |
|-------|---------|
| 📥 **Data Loader** | Multi-source ingestion |
| 🧹 **Data Cleaner** | Auto quality fixes |
| 📊 **EDA Copilot** | Visual analytics |
| ⚙️ **Feature Engineer** | Smart preprocessing |
| 🧠 **AutoML** | Model training |
| 🔌 **API Wrapper** | FastAPI generation |
| 🐳 **DevOps** | Docker/K8s configs |
| ☁️ **Cloud Deploy** | Multi-cloud deploy |
| 📈 **Drift Monitor** | Production alerts |
| 🔄 **Optimizer** | Self-improvement |

</td>
<td width="50%">

### 🛠️ Enterprise Ready

| Feature | Description |
|---------|-------------|
| 🔐 **Security** | API auth, rate limiting |
| 📝 **Validation** | Input sanitization |
| 🌐 **100+ LLMs** | OpenAI, Anthropic, etc. |
| 💾 **Caching** | Redis-backed |
| 📊 **MLOps** | MLflow integration |
| 🔄 **HITL** | Human-in-the-loop |
| 📦 **Artifacts** | Managed outputs |
| 🔧 **APO** | Auto prompt tuning |

</td>
</tr>
</table>

---

## 🚀 Quick Start

### Installation

```bash
# Clone & install
git clone https://github.com/mlvpatel/AgentDS.git
cd AgentDS
pip install -e .

# Configure
cp .env.example .env
# Add your API keys to .env
```

### Usage Options

<table>
<tr>
<td>

**🖥️ CLI**
```bash
agentds run data.csv \
  -t "Predict customer churn" \
  -o ./outputs
```

</td>
<td>

**🌐 Web UI**
```bash
agentds web
# Open http://localhost:7860
```

</td>
<td>

**🐍 Python**
```python
from agentds import AgentDSPipeline

pipeline = AgentDSPipeline()
results = pipeline.run(
    "data.csv",
    task="Predict churn"
)
```

</td>
</tr>
</table>

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         🌐  USER INTERFACE                              │
│    ┌──────────────────────┐        ┌──────────────────────┐             │
│    │   📱 Web Dashboard    │        │    🔌 REST API        │            │
│    │      (Gradio)        │        │     (Litestar)       │             │
│    └──────────────────────┘        └──────────────────────┘             │
└─────────────────────────────────────┬───────────────────────────────────┘
                                      │
┌─────────────────────────────────────▼───────────────────────────────────┐
│                      🧠  ORCHESTRATION LAYER                            │
│                                                                         │
│         ┌─────────────┐                    ┌─────────────┐              │
│         │  LangGraph  │◄──────────────────►│ Pydantic AI │              │
│         │  Workflows  │                    │   Agents    │              │
│         └─────────────┘                    └─────────────┘              │
└─────────────────────────────────────┬───────────────────────────────────┘
                                      │
          ┌───────────────────────────┼───────────────────────────┐
          ▼                           ▼                           ▼
┌─────────────────────┐   ┌─────────────────────┐   ┌─────────────────────┐
│   🤖 AI AGENTS      │   │   ⚙️ CORE SERVICES   │   │   🔗 INTEGRATIONS   │
├─────────────────────┤   ├─────────────────────┤   ├─────────────────────┤
│ • DataLoaderAgent   │   │ • LLMGateway        │   │ • n8n Workflows     │
│ • DataCleaningAgent │   │ • ConfigManager     │   │ • Cloud Storage     │
│ • EDACopilotAgent   │   │ • CacheLayer        │   │   └─ S3/GCS/Azure   │
│ • FeatureEngineer   │   │ • JobQueue          │   │ • Notifications     │
│ • AutoMLAgent       │   │ • ArtifactStore     │   │   └─ Slack/Email    │
│ • APIWrapperAgent   │   │ • Validation ✨     │   │ • Webhooks          │
│ • DevOpsAgent       │   │ • Exceptions ✨     │   │                     │
│ • CloudDeployAgent  │   │ • APO ✨            │   │                     │
│ • DriftMonitorAgent │   │                     │   │                     │
│ • OptimizationAgent │   │                     │   │                     │
└─────────────────────┘   └─────────────────────┘   └─────────────────────┘
          │                           │                           │
          └───────────────────────────┼───────────────────────────┘
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      💾  DATA & INFRASTRUCTURE                          │
│                                                                         │
│    ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐ │
│    │  Redis  │   │ DuckDB  │   │ Polars  │   │ MLflow  │   │ Docker  │ │
│    │  Cache  │   │Analytics│   │DataFrames   │Tracking │   │  K8s    │ │
│    └─────────┘   └─────────┘   └─────────┘   └─────────┘   └─────────┘ │
└─────────────────────────────────────────────────────────────────────────┘

✨ = New in latest release
```

---

## 📊 Pipeline Flow

```
          ┌─────────────────────────────────────────────────────────────┐
          │                    AGENTDS PIPELINE                        │
          └─────────────────────────────────────────────────────────────┘
                                      │
    ╔═════════════════════════════════╧═════════════════════════════════╗
    ║                         PHASE 1: BUILD                            ║
    ╠═══════════════════════════════════════════════════════════════════╣
    ║                                                                   ║
    ║  📥 Data      🧹 Clean      📊 EDA        ⚙️ Feature    🧠 AutoML  ║
    ║  Loader  ───►  Agent  ───►  Copilot  ───► Engineer ───►  Agent   ║
    ║    │            │            │             │              │       ║
    ║    ▼            ▼            ▼             ▼              ▼       ║
    ║  .parquet    .parquet    dashboard      pipeline       model     ║
    ║                            .html          .pkl          .pkl     ║
    ║                                                                   ║
    ╠═══════════════════════════════════════════════════════════════════╣
    ║                        PHASE 2: DEPLOY                            ║
    ╠═══════════════════════════════════════════════════════════════════╣
    ║                                                                   ║
    ║  🔌 API         🐳 DevOps        ☁️ Cloud                         ║
    ║  Wrapper   ───►  Agent    ───►   Deploy                          ║
    ║    │              │               │                               ║
    ║    ▼              ▼               ▼                               ║
    ║  app.py       Dockerfile      AWS/GCP/Azure                       ║
    ║                                                                   ║
    ╠═══════════════════════════════════════════════════════════════════╣
    ║                        PHASE 3: MONITOR                           ║
    ╠═══════════════════════════════════════════════════════════════════╣
    ║                                                                   ║
    ║  📈 Drift         🔄 Optimization                                 ║
    ║  Monitor    ───►   Agent (APO)                                    ║
    ║    │                 │                                            ║
    ║    ▼                 ▼                                            ║
    ║  alerts          improved prompts                                 ║
    ║                                                                   ║
    ╚═══════════════════════════════════════════════════════════════════╝
```

---

## 🌐 LLM Provider Support

<table>
<tr>
<td><b>☁️ Cloud</b></td>
<td><b>⚡ Fast</b></td>
<td><b>🏠 Local</b></td>
<td><b>🏢 Enterprise</b></td>
</tr>
<tr>
<td>

- OpenAI
- Anthropic
- Google Vertex
- AWS Bedrock
- Azure OpenAI

</td>
<td>

- Groq
- Together AI
- Fireworks
- Cerebras

</td>
<td>

- Ollama
- vLLM
- LlamaCPP

</td>
<td>

- NVIDIA NIM
- Hugging Face
- Replicate

</td>
</tr>
</table>

> 🔧 Configure in `config/llm_config.yaml` — see [LLM Providers Guide](docs/LLM_PROVIDERS.md)

---

## 📦 New in Latest Release

<table>
<tr>
<td width="33%">

### 🔐 Security
- Custom exception hierarchy
- Input validation utilities
- API key authentication
- Rate limiting (60 req/min)

</td>
<td width="33%">

### 🔄 APO Engine
- Auto prompt optimization
- Beam search algorithm
- Prompt version history
- A/B testing support

</td>
<td width="33%">

### 📚 Docs
- [Secrets Management](docs/SECRETS.md)
- [APO Guide](docs/APO_GUIDE.md)
- Updated API Reference
- Enhanced Architecture

</td>
</tr>
</table>

---

## 📖 Documentation

| Guide | Description |
|-------|-------------|
| 📘 [User Manual](docs/USER_MANUAL.md) | Complete usage guide |
| 📗 [API Reference](docs/API_REFERENCE.md) | REST & Python API |
| 📙 [Architecture](docs/ARCHITECTURE.md) | System design |
| 📕 [LLM Providers](docs/LLM_PROVIDERS.md) | 100+ LLM configs |
| 📓 [Deployment](docs/DEPLOYMENT.md) | Production setup |
| 📔 [APO Guide](docs/APO_GUIDE.md) | Prompt optimization |
| 📒 [Secrets](docs/SECRETS.md) | Secrets management |
| 🔧 [Troubleshooting](docs/TROUBLESHOOTING.md) | Common issues |

---

## 🐳 Deployment

<details>
<summary><b>Docker Compose</b></summary>

```bash
docker-compose -f docker/docker-compose.yml up -d
```
</details>

<details>
<summary><b>Kubernetes</b></summary>

```bash
kubectl apply -f docker/k8s/
```
</details>

<details>
<summary><b>Cloud Platforms</b></summary>

| Platform | Guide |
|----------|-------|
| AWS ECS | [docs/DEPLOYMENT.md#aws](docs/DEPLOYMENT.md#aws) |
| GCP Cloud Run | [docs/DEPLOYMENT.md#gcp](docs/DEPLOYMENT.md#gcp) |
| Azure ACI | [docs/DEPLOYMENT.md#azure](docs/DEPLOYMENT.md#azure) |
</details>

---

## 🧪 Testing

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=agentds --cov-report=html

# Specific module
pytest tests/test_apo.py -v
```

---

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Setup
pip install -e ".[dev]"
pre-commit install

# Verify
ruff check .
mypy agentds/
pytest tests/
```

---

## 🙏 Built With

<p align="center">
<a href="https://github.com/langchain-ai/langgraph"><img src="https://img.shields.io/badge/LangGraph-Orchestration-blue?style=flat-square" alt="LangGraph"></a>
<a href="https://github.com/pydantic/pydantic-ai"><img src="https://img.shields.io/badge/Pydantic_AI-Agents-red?style=flat-square" alt="Pydantic AI"></a>
<a href="https://github.com/BerriAI/litellm"><img src="https://img.shields.io/badge/LiteLLM-100+_LLMs-green?style=flat-square" alt="LiteLLM"></a>
<a href="https://github.com/pola-rs/polars"><img src="https://img.shields.io/badge/Polars-DataFrames-orange?style=flat-square" alt="Polars"></a>
<a href="https://github.com/litestar-org/litestar"><img src="https://img.shields.io/badge/Litestar-Web_API-purple?style=flat-square" alt="Litestar"></a>
</p>

---

## 📧 Contact

<p align="center">
<b>Author:</b> Malav Patel<br>
<a href="mailto:malav.patel203@gmail.com">📧 Email</a> •
<a href="https://github.com/mlvpatel">🐙 GitHub</a> •
<a href="https://github.com/mlvpatel/AgentDS/issues">🐛 Issues</a>
</p>

---

<div align="center">

### ⭐ Star us on GitHub — it helps!

**Made with ❤️ by data scientists, for data scientists**

[⬆ Back to Top](#-agentds)

</div>
