"""
AgentDS Integrations Module.

Provides integrations with external services and automation platforms.

Author: Malav Patel
"""

from agentds.integrations.n8n import N8nClient, N8nWebhook
from agentds.integrations.notifications import (
    NotificationService,
    SlackNotifier,
    EmailNotifier,
)

__all__ = [
    "N8nClient",
    "N8nWebhook",
    "NotificationService",
    "SlackNotifier",
    "EmailNotifier",
]
