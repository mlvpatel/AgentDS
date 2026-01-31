"""
AgentDS Notification Services.

Provides notification integrations for Slack, email, and other services.

Author: Malav Patel
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx

from agentds.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Notification:
    """Notification message."""

    title: str
    message: str
    level: str = "info"  # info, warning, error, success
    timestamp: str = ""
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Set defaults after init."""
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()
        if self.metadata is None:
            self.metadata = {}


class NotificationService(ABC):
    """Abstract base class for notification services."""

    @abstractmethod
    async def send(self, notification: Notification) -> bool:
        """Send a notification."""
        pass

    async def send_pipeline_started(self, job_id: str, task: str) -> bool:
        """Send pipeline started notification."""
        return await self.send(
            Notification(
                title="Pipeline Started",
                message=f"Job {job_id[:8]} started: {task[:50]}",
                level="info",
                metadata={"job_id": job_id},
            )
        )

    async def send_pipeline_completed(self, job_id: str, duration: float) -> bool:
        """Send pipeline completed notification."""
        return await self.send(
            Notification(
                title="Pipeline Completed",
                message=f"Job {job_id[:8]} completed in {duration:.1f}s",
                level="success",
                metadata={"job_id": job_id, "duration": duration},
            )
        )

    async def send_pipeline_failed(self, job_id: str, error: str) -> bool:
        """Send pipeline failed notification."""
        return await self.send(
            Notification(
                title="Pipeline Failed",
                message=f"Job {job_id[:8]} failed: {error[:100]}",
                level="error",
                metadata={"job_id": job_id, "error": error},
            )
        )

    async def send_drift_alert(
        self, job_id: str, drift_score: float, features: list[str]
    ) -> bool:
        """Send drift detection alert."""
        return await self.send(
            Notification(
                title="Drift Detected",
                message=f"Drift score: {drift_score:.3f}. Affected: {', '.join(features[:3])}",
                level="warning",
                metadata={"job_id": job_id, "drift_score": drift_score},
            )
        )


class SlackNotifier(NotificationService):
    """Slack notification service."""

    LEVEL_COLORS = {
        "info": "#3498db",
        "warning": "#f39c12",
        "error": "#e74c3c",
        "success": "#2ecc71",
    }

    def __init__(self, webhook_url: str, channel: str | None = None) -> None:
        """
        Initialize Slack notifier.

        Args:
            webhook_url: Slack webhook URL
            channel: Optional channel override
        """
        self.webhook_url = webhook_url
        self.channel = channel
        self._client = httpx.AsyncClient(timeout=10)

    async def send(self, notification: Notification) -> bool:
        """Send notification to Slack."""
        color = self.LEVEL_COLORS.get(notification.level, "#3498db")

        payload = {
            "attachments": [
                {
                    "color": color,
                    "title": notification.title,
                    "text": notification.message,
                    "footer": "AgentDS",
                    "ts": datetime.now(timezone.utc).timestamp(),
                }
            ]
        }

        if self.channel:
            payload["channel"] = self.channel

        try:
            response = await self._client.post(
                self.webhook_url,
                json=payload,
            )
            response.raise_for_status()
            logger.info("Slack notification sent", title=notification.title)
            return True

        except httpx.HTTPError as e:
            logger.error("Slack notification failed", error=str(e))
            return False

    async def close(self) -> None:
        """Close HTTP client."""
        await self._client.aclose()


class EmailNotifier(NotificationService):
    """Email notification service using SMTP."""

    LEVEL_SUBJECTS = {
        "info": "[INFO]",
        "warning": "[WARNING]",
        "error": "[ERROR]",
        "success": "[SUCCESS]",
    }

    def __init__(
        self,
        smtp_host: str,
        smtp_port: int,
        username: str,
        password: str,
        from_email: str,
        to_emails: list[str],
        use_tls: bool = True,
    ) -> None:
        """
        Initialize email notifier.

        Args:
            smtp_host: SMTP server host
            smtp_port: SMTP server port
            username: SMTP username
            password: SMTP password
            from_email: Sender email address
            to_emails: List of recipient email addresses
            use_tls: Whether to use TLS
        """
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_email = from_email
        self.to_emails = to_emails
        self.use_tls = use_tls

    async def send(self, notification: Notification) -> bool:
        """Send notification via email."""
        import smtplib
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText

        prefix = self.LEVEL_SUBJECTS.get(notification.level, "[INFO]")
        subject = f"{prefix} AgentDS: {notification.title}"

        # Create HTML body
        body = f"""
        <html>
        <body>
            <h2>{notification.title}</h2>
            <p>{notification.message}</p>
            <hr>
            <p><small>
                Timestamp: {notification.timestamp}<br>
                Level: {notification.level}
            </small></p>
        </body>
        </html>
        """

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = self.from_email
        msg["To"] = ", ".join(self.to_emails)
        msg.attach(MIMEText(body, "html"))

        try:
            if self.use_tls:
                server = smtplib.SMTP(self.smtp_host, self.smtp_port)
                server.starttls()
            else:
                server = smtplib.SMTP_SSL(self.smtp_host, self.smtp_port)

            server.login(self.username, self.password)
            server.sendmail(self.from_email, self.to_emails, msg.as_string())
            server.quit()

            logger.info("Email notification sent", title=notification.title)
            return True

        except Exception as e:
            logger.error("Email notification failed", error=str(e))
            return False


class CompositeNotifier(NotificationService):
    """Sends notifications to multiple services."""

    def __init__(self, notifiers: list[NotificationService]) -> None:
        """
        Initialize composite notifier.

        Args:
            notifiers: List of notification services
        """
        self.notifiers = notifiers

    async def send(self, notification: Notification) -> bool:
        """Send notification to all services."""
        results = []
        for notifier in self.notifiers:
            try:
                result = await notifier.send(notification)
                results.append(result)
            except Exception as e:
                logger.error(
                    "Notifier failed",
                    notifier=type(notifier).__name__,
                    error=str(e),
                )
                results.append(False)
        return any(results)
