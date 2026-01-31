# Changelog

All notable changes to AgentDS will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2026-01-29

### Added

- **10 Specialized Agents** covering complete data science lifecycle
  - Phase 1 (Build): DataLoaderAgent, DataCleaningAgent, EDACopilotAgent, FeatureEngineerAgent, AutoMLAgent
  - Phase 2 (Deploy): APIWrapperAgent, DevOpsAgent, CloudDeployAgent
  - Phase 3 (Learn): DriftMonitorAgent, OptimizationAgent

- **LangGraph Orchestration** (v1.0.7)
  - State machine-based workflow management
  - Checkpoint and rollback support
  - Conditional routing between agents

- **LiteLLM Universal Gateway** (v1.81.5)
  - 100+ LLM provider support
  - Automatic fallback chains
  - Cost tracking and budget alerts
  - Response caching

- **APO (Automatic Prompt Optimization)**
  - Automatic Prompt Optimization
  - Self-improving agent prompts
  - Performance-based prompt evolution
  - Optional Microsoft Agent Lightning integration

- **Human-in-the-Loop Controls**
  - Approval at every agent step
  - Actions: Approve, Rerun, Feedback, Skip, Stop, Rollback
  - Diff view for data comparisons
  - ETA display

- **Multi-Input Support**
  - Files: CSV, Parquet, JSON, Excel
  - Cloud: S3, GCS, Azure Blob
  - Databases: PostgreSQL, MySQL, MongoDB, Snowflake
  - APIs: REST endpoints

- **Gradio Web Interface** (v5.49.1)
  - Interactive pipeline control
  - Real-time progress updates
  - Configuration panel
  - Job history

- **REST API** (Litestar v2.14.x)
  - n8n webhook integration
  - Full pipeline control
  - OpenAPI documentation

- **Infrastructure**
  - Redis caching and job queue
  - MLflow experiment tracking
  - Docker and Kubernetes configs
  - GitHub Actions CI/CD

### Technical Details

- Python 3.10+ required
- Polars 1.37.0 for data processing
- DuckDB 1.4.4 for SQL queries
- XGBoost 2.1.x, Optuna 4.7.0 for AutoML
- Pydantic 2.10.x for data validation
- Structlog for structured logging

### Configuration

- Temperature settings per agent (0.0-0.2)
- Complexity-based model assignment
- Fallback chain configuration
- Feature flags for optional components

---

## [1.0.0] - 2025-06-15

### Added

- Initial release
- Basic data loading and cleaning
- Simple model training
- Manual deployment

---

## Migration Guide

### Migration Notes

1. **Update requirements:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Update configuration:**
   - Move API keys to `.env` file
   - Create `config/llm_config.yaml`
   - Create `config/pipeline_config.yaml`

3. **Code changes:**
   ```python
   # Old (v1.x)
   from agentds import Pipeline
   pipeline = Pipeline()
   pipeline.run(data="data.csv")

   # New (v1.0)
   from agentds import AgentDSPipeline, PipelineConfig
   config = PipelineConfig(phases=["build", "deploy"])
   pipeline = AgentDSPipeline(config=config)
   result = pipeline.run(
       data_source="data.csv",
       task_description="Your task"
   )
   ```

4. **Docker:**
   ```bash
   # Rebuild images
   docker-compose build --no-cache
   ```

---

## Roadmap

### v1.1.0 (Planned)

- [ ] Multi-agent collaboration
- [ ] Natural language pipeline definition
- [ ] Streaming inference support
- [ ] Additional cloud deployments

### v1.2.0 (Planned)

- [ ] Notebook integration
- [ ] Custom agent creation wizard
- [ ] A/B testing for prompts
- [ ] Advanced drift detection

---

*Maintained by Malav Patel | malav.patel203@gmail.com*
