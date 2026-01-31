# AgentDS n8n Integration Guide

**Automate ML Pipelines with n8n Workflows**

Author: Malav Patel  


---

## Overview

AgentDS provides REST API endpoints that integrate seamlessly with n8n, enabling you to:

- Trigger ML pipelines from any n8n workflow
- Monitor pipeline progress
- React to pipeline events
- Integrate with 400+ other services in n8n

---

## Prerequisites

1. AgentDS API server running (port 8000)
2. n8n instance (self-hosted or cloud)
3. Network connectivity between n8n and AgentDS

---

## API Endpoints

### Base URL

```
http://your-agentds-server:8000/api
```

### Available Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/pipeline/start` | POST | Start a new pipeline |
| `/pipeline/status/{job_id}` | GET | Get pipeline status |
| `/pipeline/cancel/{job_id}` | POST | Cancel a pipeline |
| `/pipeline/action/{job_id}` | POST | Human-in-the-loop action |
| `/agent/run` | POST | Run single agent |
| `/jobs` | GET | List all jobs |
| `/config` | GET | Get configuration |

---

## n8n Workflow Examples

### Example 1: Simple Pipeline Trigger

Trigger a pipeline when a file is uploaded to Google Drive.

```json
{
  "name": "AgentDS Pipeline Trigger",
  "nodes": [
    {
      "name": "Google Drive Trigger",
      "type": "n8n-nodes-base.googleDriveTrigger",
      "parameters": {
        "event": "fileCreated",
        "folderId": "your-folder-id"
      }
    },
    {
      "name": "Start Pipeline",
      "type": "n8n-nodes-base.httpRequest",
      "parameters": {
        "method": "POST",
        "url": "http://agentds:8000/api/pipeline/start",
        "jsonParameters": true,
        "options": {},
        "bodyParametersJson": {
          "data_source": "={{ $json.webContentLink }}",
          "task_description": "Predict customer churn from uploaded data",
          "phases": ["build", "deploy"]
        }
      }
    },
    {
      "name": "Send Slack Notification",
      "type": "n8n-nodes-base.slack",
      "parameters": {
        "channel": "#ml-pipelines",
        "text": "Pipeline started! Job ID: {{ $json.job_id }}"
      }
    }
  ]
}
```

### Example 2: Pipeline with Status Monitoring

Poll pipeline status and notify on completion.

```json
{
  "name": "Monitor Pipeline",
  "nodes": [
    {
      "name": "Webhook",
      "type": "n8n-nodes-base.webhook",
      "parameters": {
        "path": "start-pipeline",
        "httpMethod": "POST"
      }
    },
    {
      "name": "Start Pipeline",
      "type": "n8n-nodes-base.httpRequest",
      "parameters": {
        "method": "POST",
        "url": "http://agentds:8000/api/pipeline/start",
        "bodyParametersJson": {
          "data_source": "={{ $json.data_url }}",
          "task_description": "={{ $json.task }}",
          "human_in_loop": false
        }
      }
    },
    {
      "name": "Wait 30s",
      "type": "n8n-nodes-base.wait",
      "parameters": {
        "amount": 30
      }
    },
    {
      "name": "Check Status",
      "type": "n8n-nodes-base.httpRequest",
      "parameters": {
        "method": "GET",
        "url": "http://agentds:8000/api/pipeline/status/{{ $node['Start Pipeline'].json.job_id }}"
      }
    },
    {
      "name": "Is Complete?",
      "type": "n8n-nodes-base.if",
      "parameters": {
        "conditions": {
          "string": [
            {
              "value1": "={{ $json.status }}",
              "operation": "equals",
              "value2": "completed"
            }
          ]
        }
      }
    },
    {
      "name": "Loop Back",
      "type": "n8n-nodes-base.noOp"
    },
    {
      "name": "Send Success Email",
      "type": "n8n-nodes-base.emailSend",
      "parameters": {
        "toEmail": "user@example.com",
        "subject": "ML Pipeline Complete",
        "text": "Your pipeline has finished successfully!"
      }
    }
  ]
}
```

### Example 3: Scheduled Drift Monitoring

Run drift monitoring daily and alert on drift detection.

```json
{
  "name": "Daily Drift Check",
  "nodes": [
    {
      "name": "Schedule Trigger",
      "type": "n8n-nodes-base.scheduleTrigger",
      "parameters": {
        "rule": {
          "interval": [{"field": "hours", "hoursInterval": 24}]
        }
      }
    },
    {
      "name": "Run Drift Monitor",
      "type": "n8n-nodes-base.httpRequest",
      "parameters": {
        "method": "POST",
        "url": "http://agentds:8000/api/agent/run",
        "bodyParametersJson": {
          "agent_name": "DriftMonitorAgent",
          "data_source": "s3://bucket/production_data.parquet",
          "config": {
            "reference_data": "s3://bucket/training_data.parquet"
          }
        }
      }
    },
    {
      "name": "Check Drift",
      "type": "n8n-nodes-base.if",
      "parameters": {
        "conditions": {
          "boolean": [
            {
              "value1": "={{ $json.outputs.drift_detected }}",
              "value2": true
            }
          ]
        }
      }
    },
    {
      "name": "Create Jira Ticket",
      "type": "n8n-nodes-base.jira",
      "parameters": {
        "resource": "issue",
        "operation": "create",
        "project": "ML",
        "issueType": "Bug",
        "summary": "Model Drift Detected",
        "description": "Drift score: {{ $json.outputs.overall_drift_score }}"
      }
    }
  ]
}
```

---

## HTTP Request Node Configuration

### Starting a Pipeline

**Node Type**: HTTP Request

**Settings**:
- Method: `POST`
- URL: `http://agentds:8000/api/pipeline/start`
- Body Content Type: `JSON`

**Body Parameters**:
```json
{
  "data_source": "path/to/data.csv",
  "task_description": "Your ML task description",
  "output_destination": "/outputs",
  "phases": ["build", "deploy"],
  "human_in_loop": false
}
```

### Checking Status

**Node Type**: HTTP Request

**Settings**:
- Method: `GET`
- URL: `http://agentds:8000/api/pipeline/status/{{ $json.job_id }}`

### Human-in-the-Loop Action

**Node Type**: HTTP Request

**Settings**:
- Method: `POST`
- URL: `http://agentds:8000/api/pipeline/action/{{ $json.job_id }}`
- Body:
```json
{
  "action": "approve_and_continue",
  "feedback": "Optional feedback text"
}
```

Available actions:
- `approve_and_continue`
- `rerun`
- `rerun_with_feedback`
- `skip`
- `stop_pipeline`

---

## Webhook Configuration

### Receiving Pipeline Events (Coming Soon)

Configure AgentDS to send webhooks to n8n:

```yaml
# In pipeline_config.yaml
webhooks:
  enabled: true
  url: "https://your-n8n-instance.com/webhook/agentds"
  events:
    - pipeline_started
    - agent_completed
    - awaiting_approval
    - pipeline_completed
    - pipeline_failed
```

---

## Authentication

### API Key Authentication

Set up API key authentication:

1. Configure API keys in AgentDS:
```bash
export API_KEYS=your-secret-key-1,your-secret-key-2
```

2. Add header in n8n HTTP Request:
- Header Name: `X-API-Key`
- Header Value: `your-secret-key-1`

---

## Error Handling

### Retry Logic

Configure n8n to retry failed requests:

1. In HTTP Request node settings:
   - Enable "Retry On Fail"
   - Set retry attempts (e.g., 3)
   - Set retry interval (e.g., 5000ms)

### Error Notifications

Add an Error Trigger node to catch failures:

```json
{
  "name": "Error Handler",
  "type": "n8n-nodes-base.errorTrigger"
}
```

---

## Best Practices

### 1. Use Environment Variables

Store Personal Data Scientist URL in n8n credentials:
```
AGENTDS_URL=http://agentds:8000/api
```

### 2. Implement Timeouts

Long-running pipelines should use polling with timeouts:
```javascript
// In Function node
const maxWait = 3600; // 1 hour
const startTime = Date.now();
if ((Date.now() - startTime) / 1000 > maxWait) {
  throw new Error('Pipeline timeout');
}
```

### 3. Log Everything

Add Set nodes to log important data:
```json
{
  "pipeline_id": "{{ $json.job_id }}",
  "status": "{{ $json.status }}",
  "timestamp": "{{ new Date().toISOString() }}"
}
```

### 4. Use Separate Workflows

Split complex automations:
- Pipeline trigger workflow
- Status monitoring workflow
- Alert handling workflow

---

## Troubleshooting

### Connection Refused

```
Error: connect ECONNREFUSED
```

**Solution**: Ensure Personal Data Scientist API is running and accessible from n8n network.

### Timeout Errors

```
Error: ESOCKETTIMEDOUT
```

**Solution**: Increase timeout in HTTP Request node settings.

### Invalid JSON Response

```
Error: Unexpected token
```

**Solution**: Check Personal Data Scientist logs for errors. API may be returning HTML error page.

---

## Support

For issues with n8n integration:
- GitHub Issues: https://github.com/mlvpatel/AgentDS/issues
- Email: malav.patel203@gmail.com

---

*This guide is part of AgentDS*

*Author: Malav Patel*
