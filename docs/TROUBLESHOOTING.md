# AgentDS Troubleshooting Guide

**Common Issues and Solutions**

Author: Malav Patel  


---

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [LLM Connection Issues](#llm-connection-issues)
3. [Pipeline Errors](#pipeline-errors)
4. [Performance Issues](#performance-issues)
5. [Docker Issues](#docker-issues)
6. [FAQ](#faq)

---

## Installation Issues

### Python Version Error

**Error:**
```
ERROR: This package requires Python >=3.10
```

**Solution:**
```bash
# Check Python version
python --version

# Install Python 3.10+
# Ubuntu/Debian
sudo apt install python3.10 python3.10-venv

# macOS
brew install python@3.11

# Use specific version
python3.11 -m venv venv
source venv/bin/activate
```

---

### Dependency Conflicts

**Error:**
```
ERROR: Cannot install package due to conflicting dependencies
```

**Solution:**
```bash
# Create fresh virtual environment
rm -rf venv
python -m venv venv
source venv/bin/activate

# Install with fresh pip
pip install --upgrade pip
pip install -r requirements.txt

# If specific package conflicts
pip install --no-deps package_name
```

---

### Redis Connection Failed

**Error:**
```
redis.exceptions.ConnectionError: Error connecting to localhost:6379
```

**Solution:**
```bash
# Install Redis
# Ubuntu/Debian
sudo apt install redis-server
sudo systemctl start redis

# macOS
brew install redis
brew services start redis

# Docker
docker run -d -p 6379:6379 redis:7-alpine

# Verify connection
redis-cli ping
# Should return: PONG
```

---

## LLM Connection Issues

### OpenAI API Key Invalid

**Error:**
```
openai.AuthenticationError: Invalid API Key
```

**Solution:**
```bash
# Check .env file
cat .env | grep OPENAI

# Verify key format (should start with sk-)
echo $OPENAI_API_KEY

# Test key directly
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

---

### Rate Limit Exceeded

**Error:**
```
litellm.RateLimitError: Rate limit exceeded
```

**Solution:**
```yaml
# config/llm_config.yaml
rate_limiting:
  enabled: true
  requests_per_minute: 30  # Reduce from 60
  tokens_per_minute: 50000

# Or configure fallback
fallback_chains:
  default:
    - openai/gpt-4o
    - anthropic/claude-3.5-sonnet  # Fallback when rate limited
    - groq/llama-3.1-70b
```

---

### Ollama Not Responding

**Error:**
```
httpx.ConnectError: Connection refused to localhost:11434
```

**Solution:**
```bash
# Check if Ollama is running
ps aux | grep ollama

# Start Ollama
ollama serve

# Check available models
ollama list

# Pull required model
ollama pull llama3.1:8b

# Test connection
curl http://localhost:11434/api/tags
```

---

### Model Not Found

**Error:**
```
litellm.NotFoundError: Model not found: openai/gpt-5
```

**Solution:**
```bash
# Check available models
python -c "import litellm; print(litellm.model_list)"

# Use correct model name
# Wrong: openai/gpt-5
# Correct: openai/gpt-4o

# Update config
# config/llm_config.yaml
default_model: openai/gpt-4o  # Valid model name
```

---

## Pipeline Errors

### Data Source Not Found

**Error:**
```
FileNotFoundError: Data source not found: /path/to/data.csv
```

**Solution:**
```bash
# Check file exists
ls -la /path/to/data.csv

# Check file permissions
chmod 644 /path/to/data.csv

# Use absolute path
realpath data.csv
# Use the output in your command
```

---

### Agent Timeout

**Error:**
```
TimeoutError: Agent AutoMLAgent exceeded timeout of 3600s
```

**Solution:**
```yaml
# config/pipeline_config.yaml
agents:
  AutoMLAgent:
    timeout: 7200  # Increase to 2 hours
    
  # Or reduce optimization trials
  optimization:
    trials: 25  # Reduce from 50
```

---

### Memory Error

**Error:**
```
MemoryError: Unable to allocate array
```

**Solution:**
```yaml
# config/pipeline_config.yaml
data:
  sampling:
    enabled: true
    threshold_rows: 500000  # Sample if > 500K rows
    sample_size: 100000

performance:
  memory:
    max_usage_percent: 70  # Reduce from 80
```

---

### Checkpoint Restore Failed

**Error:**
```
ValueError: Checkpoint not found for job_id: xxx
```

**Solution:**
```bash
# Check checkpoints directory
ls -la checkpoints/

# Clear corrupted checkpoints
rm -rf checkpoints/*

# Disable checkpointing temporarily
# config/pipeline_config.yaml
pipeline:
  checkpoint:
    enabled: false
```

---

## Performance Issues

### Slow LLM Responses

**Solution:**
```yaml
# Use faster models for low-complexity agents
agent_llm_mapping:
  DataLoaderAgent:
    model: groq/llama-3.1-8b-instant  # Fast
  DataCleaningAgent:
    model: groq/llama-3.1-8b-instant  # Fast
    
# Enable caching
litellm_settings:
  caching: true
  cache_type: redis
```

---

### High Memory Usage

**Solution:**
```python
# Use Polars lazy evaluation
import polars as pl

# Instead of
df = pl.read_csv("large_file.csv")

# Use lazy loading
df = pl.scan_csv("large_file.csv")
result = df.filter(...).collect()
```

---

### Slow Data Loading

**Solution:**
```bash
# Use Parquet instead of CSV
# Convert CSV to Parquet
python -c "
import polars as pl
pl.read_csv('data.csv').write_parquet('data.parquet')
"

# Use Parquet in pipeline
agentds run data.parquet -t "Your task"
```

---

## Docker Issues

### Container Won't Start

**Error:**
```
docker: Error response from daemon: driver failed programming external connectivity
```

**Solution:**
```bash
# Check port conflicts
lsof -i :7860
lsof -i :8000

# Use different ports
docker run -p 7861:7860 agentds:latest

# Or stop conflicting services
docker stop $(docker ps -q)
```

---

### Volume Mount Permission Denied

**Error:**
```
PermissionError: Permission denied: '/app/outputs'
```

**Solution:**
```bash
# Fix host directory permissions
chmod -R 777 ./outputs

# Or match container user
docker run --user $(id -u):$(id -g) agentds:latest
```

---

### GPU Not Available

**Error:**
```
RuntimeError: CUDA not available
```

**Solution:**
```bash
# Check NVIDIA drivers
nvidia-smi

# Install NVIDIA Container Toolkit
# Ubuntu
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt update && sudo apt install -y nvidia-container-toolkit

# Restart Docker
sudo systemctl restart docker

# Run with GPU
docker run --gpus all agentds:latest
```

---

## FAQ

### Q: How do I switch LLM providers mid-pipeline?

**A:** Update the config and restart:
```yaml
# config/llm_config.yaml
agent_llm_mapping:
  AutoMLAgent:
    model: anthropic/claude-3-5-sonnet  # Changed from OpenAI
```

---

### Q: Can I run without any API keys?

**A:** Yes, use Ollama for local inference:
```bash
# Install and start Ollama
ollama serve

# Pull models
ollama pull llama3.1:8b

# Configure AgentDS
# .env
OLLAMA_API_BASE=http://localhost:11434

# config/llm_config.yaml
presets:
  local:
    default_model: ollama/llama3.1:8b
```

---

### Q: How do I disable specific agents?

**A:** Use feature flags:
```yaml
# config/pipeline_config.yaml
agents:
  CloudDeployAgent:
    enabled: false  # Skip deployment
```

---

### Q: How do I increase logging?

**A:** Set log level:
```bash
# Environment variable
export LOG_LEVEL=DEBUG

# Or in .env
LOG_LEVEL=DEBUG

# Or command line
agentds --debug run data.csv -t "Task"
```

---

### Q: Where are my outputs?

**A:** Default location is `./outputs/{job_id}/`:
```bash
# List outputs
ls -la outputs/

# Find specific job
ls -la outputs/550e8400-e29b-41d4-a716-*/
```

---

## Getting Help

1. **Check logs:**
   ```bash
   cat logs/agentds.log | tail -100
   ```

2. **Enable debug mode:**
   ```bash
   agentds --debug run ...
   ```

3. **Search issues:**
   https://github.com/mlvpatel/AgentDS/issues

4. **Contact:**
   - Email: malav.patel203@gmail.com
   - GitHub: @mlvpatel

---

*Author: Malav Patel | malav.patel203@gmail.com*
