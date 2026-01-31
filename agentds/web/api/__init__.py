"""
AgentDS API Module.

REST API endpoints for external integrations.

Author: Malav Patel
"""

from agentds.web.api.webhooks import api_router, create_api

__all__ = ["create_api", "api_router"]
