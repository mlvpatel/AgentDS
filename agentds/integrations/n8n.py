"""
AgentDS n8n Integration.

Provides client and webhook handling for n8n automation platform.

Author: Malav Patel
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import httpx

from agentds.core.logger import get_logger

logger = get_logger(__name__)


class WebhookEvent(str, Enum):
    """Webhook event types."""

    PIPELINE_STARTED = "pipeline_started"
    AGENT_STARTED = "agent_started"
    AGENT_COMPLETED = "agent_completed"
    AWAITING_APPROVAL = "awaiting_approval"
    PIPELINE_COMPLETED = "pipeline_completed"
    PIPELINE_FAILED = "pipeline_failed"
    DRIFT_DETECTED = "drift_detected"


@dataclass
class WebhookPayload:
    """Webhook payload structure."""

    event: WebhookEvent
    job_id: str
    timestamp: str
    data: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event": self.event.value,
            "job_id": self.job_id,
            "timestamp": self.timestamp,
            "data": self.data,
        }


class N8nClient:
    """
    Client for sending webhooks to n8n.

    Example usage:
        client = N8nClient(webhook_url="https://n8n.example.com/webhook/agentds")
        await client.send_event(
            event=WebhookEvent.PIPELINE_COMPLETED,
            job_id="abc123",
            data={"status": "success"}
        )
    """

    def __init__(
        self,
        webhook_url: str,
        api_key: str | None = None,
        timeout: int = 30,
    ) -> None:
        """
        Initialize n8n client.

        Args:
            webhook_url: n8n webhook URL
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
        """
        self.webhook_url = webhook_url
        self.api_key = api_key
        self.timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout)

    async def send_event(
        self,
        event: WebhookEvent,
        job_id: str,
        data: dict[str, Any] | None = None,
    ) -> bool:
        """
        Send webhook event to n8n.

        Args:
            event: Event type
            job_id: Job identifier
            data: Additional event data

        Returns:
            True if successful
        """
        payload = WebhookPayload(
            event=event,
            job_id=job_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            data=data or {},
        )

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key

        try:
            response = await self._client.post(
                self.webhook_url,
                json=payload.to_dict(),
                headers=headers,
            )
            response.raise_for_status()

            logger.info(
                "Webhook sent successfully",
                event=event.value,
                job_id=job_id,
            )
            return True

        except httpx.HTTPError as e:
            logger.error(
                "Webhook failed",
                event=event.value,
                job_id=job_id,
                error=str(e),
            )
            return False

    async def send_pipeline_started(
        self,
        job_id: str,
        task_description: str,
        phases: list[str],
    ) -> bool:
        """Send pipeline started event."""
        return await self.send_event(
            event=WebhookEvent.PIPELINE_STARTED,
            job_id=job_id,
            data={
                "task_description": task_description,
                "phases": phases,
            },
        )

    async def send_agent_completed(
        self,
        job_id: str,
        agent_name: str,
        status: str,
        outputs: dict[str, Any],
    ) -> bool:
        """Send agent completed event."""
        return await self.send_event(
            event=WebhookEvent.AGENT_COMPLETED,
            job_id=job_id,
            data={
                "agent_name": agent_name,
                "status": status,
                "outputs": outputs,
            },
        )

    async def send_awaiting_approval(
        self,
        job_id: str,
        agent_name: str,
        approval_message: str,
    ) -> bool:
        """Send awaiting approval event."""
        return await self.send_event(
            event=WebhookEvent.AWAITING_APPROVAL,
            job_id=job_id,
            data={
                "agent_name": agent_name,
                "message": approval_message,
            },
        )

    async def send_pipeline_completed(
        self,
        job_id: str,
        outputs: dict[str, Any],
        duration_seconds: float,
    ) -> bool:
        """Send pipeline completed event."""
        return await self.send_event(
            event=WebhookEvent.PIPELINE_COMPLETED,
            job_id=job_id,
            data={
                "outputs": outputs,
                "duration_seconds": duration_seconds,
            },
        )

    async def send_pipeline_failed(
        self,
        job_id: str,
        error: str,
        agent_name: str | None = None,
    ) -> bool:
        """Send pipeline failed event."""
        return await self.send_event(
            event=WebhookEvent.PIPELINE_FAILED,
            job_id=job_id,
            data={
                "error": error,
                "agent_name": agent_name,
            },
        )

    async def send_drift_detected(
        self,
        job_id: str,
        drift_score: float,
        alerts: list[dict[str, Any]],
    ) -> bool:
        """Send drift detected event."""
        return await self.send_event(
            event=WebhookEvent.DRIFT_DETECTED,
            job_id=job_id,
            data={
                "drift_score": drift_score,
                "alerts": alerts,
            },
        )

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    async def __aenter__(self) -> N8nClient:
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        await self.close()


class N8nWebhook:
    """
    Webhook receiver for n8n callbacks.

    Handles incoming webhooks from n8n workflows.
    """

    def __init__(self) -> None:
        """Initialize webhook receiver."""
        self._handlers: dict[str, Any] = {}

    def register_handler(self, action: str, handler: Any) -> None:
        """Register a handler for an action."""
        self._handlers[action] = handler

    async def process_webhook(self, payload: dict[str, Any]) -> dict[str, Any]:
        """
        Process incoming webhook from n8n.

        Args:
            payload: Webhook payload

        Returns:
            Response data
        """
        action = payload.get("action")
        if not action:
            return {"error": "No action specified"}

        handler = self._handlers.get(action)
        if not handler:
            return {"error": f"Unknown action: {action}"}

        try:
            result = await handler(payload)
            return {"status": "success", "result": result}
        except Exception as e:
            logger.error("Webhook handler failed", action=action, error=str(e))
            return {"error": str(e)}
