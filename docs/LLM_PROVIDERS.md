# AgentDS LLM Providers Guide

**Complete Reference for 100+ LLM Providers via LiteLLM**

Author: Malav Patel  
  
Last Updated: January 2026

---

## Table of Contents

1. [Overview](#overview)
2. [Tier 1: Cloud Providers](#tier-1-cloud-providers)
3. [Tier 2: Specialized Providers](#tier-2-specialized-providers)
4. [Tier 3: Local Inference](#tier-3-local-inference)
5. [Tier 4: Enterprise Providers](#tier-4-enterprise-providers)
6. [Tier 5: Regional Providers](#tier-5-regional-providers)
7. [Embedding Models](#embedding-models)
8. [Configuration](#configuration)
9. [Fallback Chains](#fallback-chains)
10. [Cost Optimization](#cost-optimization)

---

## Overview

Personal Data Scientist uses LiteLLM as a universal gateway to access 100+ LLM providers with a unified API. This allows seamless switching between providers and automatic fallback on failures.

### Key Features

- **Unified API**: Same interface for all providers
- **Automatic Fallback**: Switch providers on errors
- **Cost Tracking**: Monitor spending across providers
- **Caching**: Redis-based response caching
- **Rate Limiting**: Prevent quota exhaustion

---

## Tier 1: Cloud Providers

### OpenAI

**Models Available:**
| Model | Context | Best For |
|-------|---------|----------|
| gpt-4o | 128K | Complex reasoning, code |
| gpt-4o-mini | 128K | Cost-effective tasks |
| gpt-4-turbo | 128K | Legacy compatibility |
| o1-preview | 128K | Advanced reasoning |
| o1-mini | 128K | Fast reasoning |

**Configuration:**
```yaml
# .env
OPENAI_API_KEY=sk-...

# llm_config.yaml
agent_llm_mapping:
  AutoMLAgent:
    model: openai/gpt-4o
    temperature: 0.0
```

**Usage in Personal Data Scientist:**
```python
from agentds.core.llm_gateway import LLMGateway

gateway = LLMGateway()
response = gateway.complete(
    messages=[{"role": "user", "content": "Analyze this data"}],
    model="openai/gpt-4o",
    temperature=0.0
)
```

---

### Anthropic

**Models Available:**
| Model | Context | Best For |
|-------|---------|----------|
| claude-3-5-sonnet-20241022 | 200K | Code, analysis |
| claude-3-5-haiku-20241022 | 200K | Fast, cost-effective |
| claude-3-opus-20240229 | 200K | Complex tasks |

**Configuration:**
```yaml
# .env
ANTHROPIC_API_KEY=sk-ant-...

# llm_config.yaml
agent_llm_mapping:
  APIWrapperAgent:
    model: anthropic/claude-3-5-sonnet-20241022
    temperature: 0.0
```

---

### Google Vertex AI

**Models Available:**
| Model | Context | Best For |
|-------|---------|----------|
| gemini-2.0-flash | 1M | Fast, multimodal |
| gemini-1.5-pro | 2M | Long context |
| gemini-1.5-flash | 1M | Cost-effective |

**Configuration:**
```yaml
# .env
VERTEXAI_PROJECT=your-project-id
VERTEXAI_LOCATION=us-central1
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json

# llm_config.yaml
agent_llm_mapping:
  EDACopilotAgent:
    model: vertex_ai/gemini-1.5-pro
    temperature: 0.1
```

---

### AWS Bedrock

**Models Available:**
| Model | Provider | Best For |
|-------|----------|----------|
| anthropic.claude-3-5-sonnet-20241022-v2:0 | Anthropic | Code |
| meta.llama3-1-70b-instruct-v1:0 | Meta | General |
| amazon.nova-pro-v1:0 | Amazon | Cost-effective |
| mistral.mistral-large-2407-v1:0 | Mistral | Multilingual |

**Configuration:**
```yaml
# .env
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION_NAME=us-east-1

# llm_config.yaml
agent_llm_mapping:
  FeatureEngineerAgent:
    model: bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0
```

---

### Azure OpenAI

**Configuration:**
```yaml
# .env
AZURE_API_KEY=...
AZURE_API_BASE=https://your-resource.openai.azure.com/
AZURE_API_VERSION=2024-02-01

# llm_config.yaml
agent_llm_mapping:
  AutoMLAgent:
    model: azure/gpt-4o
```

---

## Tier 2: Specialized Providers

### Groq (Ultra-Fast Inference)

**Models Available:**
| Model | Speed | Best For |
|-------|-------|----------|
| llama-3.1-70b-versatile | ~500 t/s | Complex tasks |
| llama-3.1-8b-instant | ~1000 t/s | Simple tasks |
| mixtral-8x7b-32768 | ~400 t/s | Multilingual |

**Configuration:**
```yaml
# .env
GROQ_API_KEY=...

# llm_config.yaml
agent_llm_mapping:
  DataLoaderAgent:
    model: groq/llama-3.1-8b-instant
    temperature: 0.0
```

**Best for:** Low-complexity agents requiring fast responses

---

### Mistral AI

**Models Available:**
| Model | Best For |
|-------|----------|
| mistral-large-latest | Complex reasoning |
| mistral-medium-latest | Balanced |
| codestral-latest | Code generation |
| pixtral-large-latest | Vision tasks |

**Configuration:**
```yaml
# .env
MISTRAL_API_KEY=...

# llm_config.yaml
agent_llm_mapping:
  APIWrapperAgent:
    model: mistral/codestral-latest
```

---

### DeepSeek

**Models Available:**
| Model | Best For |
|-------|----------|
| deepseek-chat | General |
| deepseek-coder | Code |
| deepseek-reasoner | Reasoning |

**Configuration:**
```yaml
# .env
DEEPSEEK_API_KEY=...

# llm_config.yaml
agent_llm_mapping:
  AutoMLAgent:
    model: deepseek/deepseek-coder
```

---

### xAI (Grok)

**Models Available:**
| Model | Best For |
|-------|----------|
| grok-2 | General, real-time |
| grok-2-vision | Multimodal |

**Configuration:**
```yaml
# .env
XAI_API_KEY=...

# llm_config.yaml
agent_llm_mapping:
  EDACopilotAgent:
    model: xai/grok-2
```

---

### Together AI

**Models Available:**
- meta-llama/Llama-3.1-70B-Instruct-Turbo
- meta-llama/Llama-3.1-8B-Instruct-Turbo
- Qwen/Qwen2.5-72B-Instruct-Turbo
- deepseek-ai/DeepSeek-V3

**Configuration:**
```yaml
# .env
TOGETHERAI_API_KEY=...

# llm_config.yaml
fallback_chains:
  default:
    - together_ai/meta-llama/Llama-3.1-70B-Instruct-Turbo
```

---

### Fireworks AI

**227+ Models Available**

Optimized for fast inference with extensive model catalog.

**Configuration:**
```yaml
# .env
FIREWORKS_AI_API_KEY=...

# llm_config.yaml
agent_llm_mapping:
  DataCleaningAgent:
    model: fireworks_ai/accounts/fireworks/models/llama-v3p1-70b-instruct
```

---

### Cohere

**Models Available:**
| Model | Best For |
|-------|----------|
| command-r-plus | Complex tasks |
| command-r | General |
| command-light | Fast, simple |

**Configuration:**
```yaml
# .env
COHERE_API_KEY=...

# llm_config.yaml
agent_llm_mapping:
  EDACopilotAgent:
    model: cohere/command-r-plus
```

---

## Tier 3: Local Inference

### Ollama

**Run LLMs locally without API costs.**

**Popular Models:**
| Model | Size | VRAM Required |
|-------|------|---------------|
| llama3.1:70b | 70B | 48GB |
| llama3.1:8b | 8B | 8GB |
| mistral:7b | 7B | 8GB |
| qwen2.5:32b | 32B | 24GB |
| codellama:70b | 70B | 48GB |
| deepseek-r1:70b | 70B | 48GB |

**Setup:**
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull models
ollama pull llama3.1:70b
ollama pull llama3.1:8b

# Start server (default port 11434)
ollama serve
```

**Configuration:**
```yaml
# .env
OLLAMA_API_BASE=http://localhost:11434

# llm_config.yaml
presets:
  local:
    description: "Local inference only"
    default_model: ollama/llama3.1:8b
    agent_overrides:
      AutoMLAgent: ollama/llama3.1:70b
      APIWrapperAgent: ollama/qwen2.5-coder:32b
```

---

### vLLM

**High-throughput serving for production.**

**Configuration:**
```yaml
# .env
VLLM_API_BASE=http://localhost:8000

# llm_config.yaml
agent_llm_mapping:
  AutoMLAgent:
    model: hosted_vllm/meta-llama/Llama-3.1-70B-Instruct
```

---

### LlamaCPP

**CPU-optimized inference.**

**Configuration:**
```yaml
# llm_config.yaml
agent_llm_mapping:
  DataLoaderAgent:
    model: llamacpp/llama-3.1-8b-instruct
```

---

## Tier 4: Enterprise Providers

### Hugging Face

**Configuration:**
```yaml
# .env
HUGGINGFACE_API_KEY=hf_...

# llm_config.yaml
agent_llm_mapping:
  DataCleaningAgent:
    model: huggingface/meta-llama/Llama-3.1-70B-Instruct
```

---

### NVIDIA NIM

**Configuration:**
```yaml
# .env
NVIDIA_NIM_API_KEY=...

# llm_config.yaml
agent_llm_mapping:
  AutoMLAgent:
    model: nvidia_nim/meta/llama-3.1-70b-instruct
```

---

### Replicate

**Configuration:**
```yaml
# .env
REPLICATE_API_TOKEN=...

# llm_config.yaml
agent_llm_mapping:
  EDACopilotAgent:
    model: replicate/meta/llama-3.1-405b-instruct
```

---

## Tier 5: Regional Providers

### Alibaba Qwen

```yaml
# .env
DASHSCOPE_API_KEY=...

# llm_config.yaml
agent_llm_mapping:
  DataLoaderAgent:
    model: qwen/qwen-max
```

### Baidu ERNIE

```yaml
# .env
QIANFAN_ACCESS_KEY=...
QIANFAN_SECRET_KEY=...
```

### Zhipu AI (GLM)

```yaml
# .env
ZHIPUAI_API_KEY=...
```

---

## Embedding Models

### Available Embeddings

| Provider | Model | Dimensions | Best For |
|----------|-------|------------|----------|
| OpenAI | text-embedding-3-large | 3072 | High accuracy |
| OpenAI | text-embedding-3-small | 1536 | Cost-effective |
| Cohere | embed-english-v3.0 | 1024 | English text |
| Voyage | voyage-large-2 | 1536 | General |
| AWS Bedrock | amazon.titan-embed-text-v2:0 | 1024 | AWS native |

**Configuration:**
```yaml
# llm_config.yaml
embeddings:
  default: openai/text-embedding-3-small
  models:
    - openai/text-embedding-3-large
    - cohere/embed-english-v3.0
```

---

## Configuration

### Complete Example

```yaml
# config/llm_config.yaml

litellm_settings:
  drop_params: true
  set_verbose: false
  num_retries: 3
  request_timeout: 120

default_model: openai/gpt-4o-mini

agent_llm_mapping:
  # LOW complexity - fast, cheap
  DataLoaderAgent:
    model: groq/llama-3.1-8b-instant
    temperature: 0.0
    complexity: LOW

  DataCleaningAgent:
    model: groq/llama-3.1-8b-instant
    temperature: 0.0
    complexity: LOW

  # MEDIUM complexity - balanced
  EDACopilotAgent:
    model: openai/gpt-4o-mini
    temperature: 0.1
    complexity: MEDIUM

  FeatureEngineerAgent:
    model: anthropic/claude-3-5-sonnet-20241022
    temperature: 0.0
    complexity: MEDIUM

  # HIGH complexity - best quality
  AutoMLAgent:
    model: openai/gpt-4o
    temperature: 0.0
    complexity: HIGH

  APIWrapperAgent:
    model: anthropic/claude-3-5-sonnet-20241022
    temperature: 0.0
    complexity: HIGH

  # CRITICAL - self-optimization
  OptimizationAgent:
    model: openai/gpt-4o
    temperature: 0.2
    complexity: CRITICAL

fallback_chains:
  default:
    - openai/gpt-4o
    - anthropic/claude-3-5-sonnet-20241022
    - vertex_ai/gemini-1.5-pro
    - ollama/llama3.1:70b

  low_complexity:
    - groq/llama-3.1-8b-instant
    - ollama/llama3.1:8b
    - together_ai/meta-llama/Llama-3.1-8B-Instruct-Turbo

presets:
  budget:
    default_model: groq/llama-3.1-8b-instant
  quality:
    default_model: openai/gpt-4o
  local:
    default_model: ollama/llama3.1:8b
```

---

## Fallback Chains

### How Fallback Works

```
Primary Model (gpt-4o)
    |
    +-- [Error: Rate Limit] --> Fallback 1 (claude-3.5)
                                    |
                                    +-- [Error: API Down] --> Fallback 2 (gemini-1.5)
                                                                  |
                                                                  +-- [Success] --> Return Response
```

### Configuration

```yaml
fallback_chains:
  # For high-complexity agents
  high_complexity:
    - openai/gpt-4o
    - anthropic/claude-3-5-sonnet-20241022
    - vertex_ai/gemini-1.5-pro

  # For code generation
  code_generation:
    - anthropic/claude-3-5-sonnet-20241022
    - openai/gpt-4o
    - deepseek/deepseek-coder

  # Budget fallback
  budget:
    - groq/llama-3.1-8b-instant
    - ollama/llama3.1:8b
```

---

## Cost Optimization

### Cost Comparison (per 1M tokens)

| Provider | Model | Input | Output |
|----------|-------|-------|--------|
| OpenAI | gpt-4o | $2.50 | $10.00 |
| OpenAI | gpt-4o-mini | $0.15 | $0.60 |
| Anthropic | claude-3.5-sonnet | $3.00 | $15.00 |
| Anthropic | claude-3.5-haiku | $0.25 | $1.25 |
| Google | gemini-1.5-pro | $1.25 | $5.00 |
| Groq | llama-3.1-70b | $0.59 | $0.79 |
| Groq | llama-3.1-8b | $0.05 | $0.08 |
| Together | llama-3.1-70b | $0.88 | $0.88 |
| Ollama | Any | Free | Free |

### Cost Tracking

```python
from agentds.core.llm_gateway import LLMGateway

gateway = LLMGateway()

# Run some completions
response = gateway.complete(...)

# Check costs
print(f"Session cost: ${gateway.get_total_cost():.4f}")

# Reset tracking
gateway.reset_cost_tracking()
```

### Budget Alerts

```yaml
# llm_config.yaml
cost_tracking:
  enabled: true
  budget_alert_threshold: 10.0  # USD
  budget_hard_limit: 50.0  # USD
  reset_period: daily
```

---

## Best Practices

### Agent-Model Matching

1. **Low Complexity** (DataLoader, DataCleaning, DriftMonitor):
   - Use fast, cheap models: `groq/llama-3.1-8b-instant`
   - Temperature: 0.0

2. **Medium Complexity** (EDA, FeatureEngineer, DevOps, CloudDeploy):
   - Use balanced models: `openai/gpt-4o-mini`, `anthropic/claude-3.5-haiku`
   - Temperature: 0.0-0.1

3. **High Complexity** (AutoML, APIWrapper):
   - Use best models: `openai/gpt-4o`, `anthropic/claude-3.5-sonnet`
   - Temperature: 0.0

4. **Critical** (Optimization):
   - Use creative models with higher temperature
   - Temperature: 0.2

### Local Development

Use Ollama for development to avoid API costs:

```bash
# Pull models
ollama pull llama3.1:8b
ollama pull llama3.1:70b

# Set preset
export AGENTDS_LLM_PRESET=local
```

---

*This document is maintained as part of AgentDS*

*Author: Malav Patel | malav.patel203@gmail.com*
