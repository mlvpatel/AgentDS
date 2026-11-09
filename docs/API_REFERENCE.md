# AgentDS API Reference

**Complete REST API and Python API Documentation**

Author: Malav Patel  


---

## Table of Contents

1. [REST API](#rest-api)
2. [Python API](#python-api)
3. [WebSocket Events](#websocket-events)
4. [Error Handling](#error-handling)

---

## REST API

Base URL: `http://localhost:8000/api`

### Authentication

All API endpoints (except the `/api/health` endpoints and `/api/docs`) require authentication via API key.

**Header Authentication (Recommended):**
```http
X-API-Key: your-api-key-here
```

**Query Parameter Authentication:**
```http
GET /api/jobs?api_key=your-api-key-here
```

**Error Response (401 Unauthorized):**
```json
{
  "error": "INVALID_API_KEY",
  "message": "Invalid or missing API key",
  "timestamp": "2026-02-02T08:50:00Z"
}
```

> **Note:** In development mode with no API keys configured, all requests are allowed.

### Rate Limiting

API requests are rate-limited to protect service availability.

**Default Limits:**
- 60 requests per minute per API key
- Burst allowance: 120 requests

**Rate Limit Headers:**
```http
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 58
```

**Rate Limited Response (429 Too Many Requests):**
```json
{
  "error": "RATE_LIMIT_EXCEEDED",
  "message": "Rate limit exceeded. Retry after 15 seconds.",
  "retry_after_seconds": 15,
  "timestamp": "2026-02-02T08:50:00Z"
}
```

---

### Health Checks

Three endpoints, none of which needs an API key. All answer immediately: the
dependency checks behind them run in the background, each one at a time, and
the last result is reused until it is refreshed.

#### Liveness

```http
GET /api/health/live
```

Always `200` while the process is serving requests. It checks no dependencies,
so a Redis or LLM outage never gets a working process restarted. Use it for the
Kubernetes `livenessProbe`.

```json
{
  "status": "alive",
  "timestamp": "2026-01-29T12:00:00Z"
}
```

#### Readiness

```http
GET /api/health/ready
```

`200` when this instance can take work, `503` when it can't. Use it for the
Kubernetes `readinessProbe` and the Docker Compose `healthcheck`.

```json
{
  "status": "ready",
  "timestamp": "2026-01-29T12:00:00Z",
  "components": {
    "job_queue": true
  }
}
```

When not ready, `status` is `not_ready`, `job_queue` is `false` and the
response code is `503`.

- `job_queue` is a Redis `PING`, refreshed every 10 seconds, when the Redis job
  queue is in use. If Redis can't be reached when the API starts, the API
  keeps working on an in-memory fallback, but `job_queue` stays `false` (so
  the instance is not ready) and each check tries to reconnect. Once Redis
  answers, the API switches to it, moves over any jobs created in the
  meantime, and becomes ready. With the Redis queue turned off
  (`redis_queue` feature disabled), the in-memory queue is the intended
  backend and `job_queue` is always `true`.
- Before the first check has finished, the endpoint waits up to 0.5 seconds for
  it, so a healthy instance is ready on its first probe. If the check is still
  running after that, the instance reports `503` until the check passes.
- The LLM is not part of readiness. A provider outage affects every instance
  alike, so taking them all out of service would turn it into a full API outage.

#### Health Report

```http
GET /api/health
```

The state of every component, for dashboards and people. Always `200`, even
when `status` is `degraded`; use the endpoints above for probes.

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2026-01-29T12:00:00Z",
  "components": {
    "api": true,
    "job_queue": true,
    "llm": true
  }
}
```

- `job_queue`: the same Redis check as readiness.
- `llm`: a call to the default model, refreshed every 5 minutes. Only this
  endpoint starts it, so the probes don't pay for model calls.

Each is `false` (and `status` is `degraded`) until its first check has passed,
and while the last check failed.

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

**Response when an RQ worker is running** (`202 Accepted`): the run is handed to
the worker (`python -m rq.cli worker`, the `worker` service in docker-compose).
Poll the status endpoint for progress.
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued",
  "message": "Pipeline queued for a worker; poll /api/pipeline/status/550e8400-e29b-41d4-a716-446655440000",
  "created_at": "2026-01-29T12:00:00Z",
  "current_agent": null,
  "error": null
}
```

**Response without a worker** (`201 Created`): the API runs the pipeline itself,
in a background thread, and replies once it completes, fails or pauses for
review. `status` is `completed`, `failed`, `cancelled` or `awaiting_approval`.
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "awaiting_approval",
  "message": "Waiting for review of DataLoaderAgent",
  "created_at": "2026-01-29T12:00:00Z",
  "current_agent": "DataLoaderAgent",
  "error": null
}
```

Runs are handed to a worker only when the Redis job queue is in use and at least
one worker is listening; otherwise they run in the API process.
`PIPELINE_JOB_TIMEOUT` (seconds, default 21600) limits how long a worker may
spend on one run or review action.

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

The job must be waiting for review (status `paused`); otherwise the API returns
`409 Conflict`. A paused run can be resumed by any API instance or worker, because
its state is stored with the job.

**Response when an RQ worker is running** (`202 Accepted`): poll the status endpoint.
```json
{
  "status": "queued",
  "action": "approve_and_continue",
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "message": "Review action queued for a worker; poll /api/pipeline/status/550e8400-e29b-41d4-a716-446655440000"
}
```

**Response without a worker** (`201 Created`):
```json
{
  "status": "success",
  "action": "approve_and_continue",
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "pipeline_status": "awaiting_approval",
  "current_agent": "DataCleaningAgent",
  "message": "Waiting for review of DataCleaningAgent",
  "error": null
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

#### AgentDSPipeline

```python
class AgentDSPipeline:
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
| 401 | Unauthorized (invalid/missing API key) |
| 404 | Not Found |
| 422 | Validation Error |
| 429 | Too Many Requests (rate limit exceeded) |
| 500 | Internal Server Error |

### Error Response Format

All errors return a structured JSON response:

```json
{
  "error": "ERROR_CODE",
  "message": "Human-readable description",
  "details": {
    "field": "additional context"
  },
  "timestamp": "2026-02-02T08:50:00Z"
}
```

### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `INVALID_API_KEY` | 401 | Missing or invalid API key |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests |
| `VALIDATION_ERROR` | 422 | Invalid input data |
| `PATH_TRAVERSAL` | 400 | Directory traversal detected |
| `FILE_SIZE_LIMIT` | 400 | File exceeds size limit |
| `JOB_NOT_FOUND` | 404 | Job ID not found |
| `AGENT_NOT_FOUND` | 404 | Agent name not found |
| `LLM_ERROR` | 500 | LLM provider error |
| `PIPELINE_ERROR` | 500 | Pipeline execution error |

### Python Exceptions

```python
from agentds.core.exceptions import (
    AgentDSError,
    PipelineError,
    AgentError,
    LLMError,
    ValidationError,
    AuthenticationError,
    RateLimitError,
)

try:
    result = pipeline.run(...)
except ValidationError as e:
    print(f"Invalid input: {e.message}")
    print(f"Field: {e.details.get('field')}")
except AuthenticationError as e:
    print(f"Auth failed: {e.code}")
except RateLimitError as e:
    print(f"Rate limited, retry after: {e.retry_after}s")
except PipelineError as e:
    print(f"Pipeline failed: {e}")
except AgentError as e:
    print(f"Agent {e.details.get('agent_name')} failed: {e}")
except LLMError as e:
    print(f"LLM call failed: {e}")
except AgentDSError as e:
    # Catch-all for any AgentDS error
    print(f"Error [{e.code}]: {e.message}")
    print(f"Details: {e.to_dict()}")
```

---

*Author: Malav Patel | malav.patel203@gmail.com*
