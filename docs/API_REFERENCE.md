# AgentDS API Reference

**Complete REST API and Python API Documentation**

Author: Malav Patel  
Version: 2.0.0

---

## Table of Contents

1. [REST API](#rest-api)
2. [Python API](#python-api)
3. [WebSocket Events](#websocket-events)
4. [Error Handling](#error-handling)

---

## REST API

Base URL: `http://localhost:8000/api`

### Health Check

```http
GET /api/health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "2.0.0",
  "timestamp": "2026-01-29T12:00:00Z",
  "components": {
    "api": true,
    "job_queue": true,
    "llm": true
  }
}
```

---

### Pipeline Endpoints

#### Start Pipeline

```http
POST /api/pipeline/start
Content-Type: application/json

{
  "data_source": "/path/to/data.csv",
  "task_description": "Predict customer churn",
  "output_destination": "/path/to/outputs",
  "phases": ["build", "deploy"],
  "human_in_loop": true
}
```

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "started",
  "message": "Pipeline started successfully",
  "created_at": "2026-01-29T12:00:00Z"
}
```

#### Get Pipeline Status

```http
GET /api/pipeline/status/{job_id}
```

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "running",
  "current_agent": "DataCleaningAgent",
  "progress_percent": 20.0,
  "started_at": "2026-01-29T12:00:00Z",
  "completed_at": null,
  "error": null,
  "outputs": {}
}
```

#### Cancel Pipeline

```http
POST /api/pipeline/cancel/{job_id}
```

**Response:**
```json
{
  "status": "cancelled",
  "job_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

#### Human-in-the-Loop Action

```http
POST /api/pipeline/action/{job_id}
Content-Type: application/json

{
  "action": "approve_and_continue",
  "feedback": null
}
```

**Available Actions:**
- `approve_and_continue` - Accept output, move to next agent
- `rerun` - Run current agent again
- `rerun_with_feedback` - Rerun with text feedback
- `skip` - Skip current agent
- `stop_pipeline` - Terminate pipeline
- `rollback` - Return to previous checkpoint

**Response:**
```json
{
  "status": "success",
  "action": "approve_and_continue",
  "job_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

---

### Agent Endpoints

#### Run Single Agent

```http
POST /api/agent/run
Content-Type: application/json

{
  "agent_name": "DataLoaderAgent",
  "data_source": "/path/to/data.csv",
  "task_description": "Load sales data",
  "config": {}
}
```

**Response:**
```json
{
  "job_id": "agent-run-001",
  "agent": "DataLoaderAgent",
  "status": "completed",
  "outputs": {
    "data_path": "/outputs/raw_data.parquet",
    "statistics": {
      "row_count": 10000,
      "column_count": 15
    }
  },
  "artifacts": ["agent-run-001/DataLoaderAgent/raw_data.parquet"],
  "duration_seconds": 2.5,
  "error": null
}
```

---

### Job Endpoints

#### List Jobs

```http
GET /api/jobs?status=running&limit=50
```

**Query Parameters:**
- `status` (optional): Filter by status (pending, queued, running, paused, completed, failed, cancelled)
- `limit` (optional): Maximum results (default: 50)

**Response:**
```json
[
  {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "name": "Pipeline: Predict customer churn",
    "status": "running",
    "current_agent": "AutoMLAgent",
    "progress_percent": 60.0,
    "created_at": "2026-01-29T12:00:00Z",
    "started_at": "2026-01-29T12:00:05Z",
    "completed_at": null,
    "duration_seconds": 120.5
  }
]
```

#### Get Job Details

```http
GET /api/jobs/{job_id}
```

#### Delete Job

```http
DELETE /api/jobs/{job_id}
```

---

### Configuration Endpoints

#### Get Configuration

```http
GET /api/config
```

**Response:**
```json
{
  "llm": {
    "default_model": "openai/gpt-4o-mini",
    "default_temperature": 0.0,
    "available_providers": ["openai", "anthropic", "groq", "ollama"]
  },
  "pipeline": {
    "human_in_loop": true
  }
}
```

#### Update Configuration

```http
POST /api/config/update
Content-Type: application/json

{
  "llm_config": {
    "default_model": "anthropic/claude-3-5-sonnet-20241022"
  },
  "feature_flags": {
    "agent_lightning_apo": true
  }
}
```

---

## Python API

### Quick Start

```python
from agentds import AgentDSPipeline, PipelineConfig, PipelinePhase

# Create pipeline
config = PipelineConfig(
    phases=[PipelinePhase.BUILD, PipelinePhase.DEPLOY],
    human_in_loop=False  # Disable for automation
)
pipeline = AgentDSPipeline(config=config)

# Run pipeline
result = pipeline.run(
    data_source="data/sales.csv",
    task_description="Predict customer churn based on transaction history",
    output_destination="outputs/"
)

# Access results
print(f"Job ID: {result['job_id']}")
print(f"Final outputs: {result['state']['final_outputs']}")
```

### Core Classes

#### Personal Data ScientistPipeline

```python
class Personal Data ScientistPipeline:
    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        """Initialize pipeline."""
        
    def run(
        self,
        data_source: str,
        task_description: str,
        output_destination: Optional[str] = None,
        job_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run the complete pipeline."""
        
    def resume(
        self,
        job_id: str,
        user_action: str,
        user_feedback: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Resume pipeline after human review."""
```

#### LLMGateway

```python
class LLMGateway:
    def __init__(
        self,
        settings: Optional[Settings] = None,
    ) -> None:
        """Initialize LLM Gateway."""
        
    def complete(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        agent_name: Optional[str] = None,
        use_fallback: bool = True,
    ) -> LLMResponse:
        """Generate completion from LLM."""
        
    async def acomplete(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> LLMResponse:
        """Async completion."""
        
    def get_total_cost(self) -> float:
        """Get total cost incurred."""
```

#### BaseAgent

```python
class BaseAgent(ABC):
    name: str
    description: str
    phase: str
    complexity: str
    
    def execute(self, context: AgentContext) -> AgentResult:
        """Execute agent task (abstract)."""
        
    def run(self, context: AgentContext) -> AgentResult:
        """Run agent with logging and error handling."""
        
    def call_llm(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        """Call LLM with agent context."""
```

### Using Individual Agents

```python
from agentds.agents import DataLoaderAgent, AutoMLAgent
from agentds.agents.base import AgentContext
from agentds.core.config import get_settings
from agentds.core.llm_gateway import LLMGateway
from agentds.core.artifact_store import ArtifactStore

# Setup
settings = get_settings()
llm_gateway = LLMGateway(settings)
artifact_store = ArtifactStore(settings)

# Create context
context = AgentContext(
    job_id="manual-001",
    settings=settings,
    llm_gateway=llm_gateway,
    artifact_store=artifact_store,
    task_description="Load and analyze sales data",
    extra={"data_source": "data/sales.csv"}
)

# Run DataLoaderAgent
loader = DataLoaderAgent(
    settings=settings,
    llm_gateway=llm_gateway,
    artifact_store=artifact_store
)
result = loader.run(context)

print(f"Status: {result.status}")
print(f"Outputs: {result.outputs}")
```

---

## WebSocket Events

### Real-time Progress (SSE)

```javascript
const eventSource = new EventSource('/api/pipeline/stream/{job_id}');

eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Progress:', data.progress_percent);
  console.log('Current Agent:', data.current_agent);
};

eventSource.addEventListener('agent_complete', (event) => {
  const data = JSON.parse(event.data);
  console.log('Agent completed:', data.agent_name);
});

eventSource.addEventListener('pipeline_complete', (event) => {
  const data = JSON.parse(event.data);
  console.log('Pipeline finished:', data.final_outputs);
  eventSource.close();
});
```

### Event Types

| Event | Description |
|-------|-------------|
| `progress` | Progress update |
| `agent_start` | Agent started |
| `agent_complete` | Agent finished |
| `approval_required` | Waiting for human action |
| `pipeline_complete` | Pipeline finished |
| `error` | Error occurred |

---

## Error Handling

### HTTP Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 201 | Created |
| 400 | Bad Request |
| 404 | Not Found |
| 422 | Validation Error |
| 500 | Internal Server Error |

### Error Response Format

```json
{
  "error": "Job not found",
  "detail": "No job with ID: invalid-id",
  "timestamp": "2026-01-29T12:00:00Z"
}
```

### Python Exceptions

```python
from agentds.core.exceptions import (
    AgentDSError,
    PipelineError,
    AgentError,
    LLMError,
    ValidationError,
)

try:
    result = pipeline.run(...)
except PipelineError as e:
    print(f"Pipeline failed: {e}")
except AgentError as e:
    print(f"Agent {e.agent_name} failed: {e}")
except LLMError as e:
    print(f"LLM call failed: {e}")
```

---

*Author: Malav Patel | malav.patel203@gmail.com*
