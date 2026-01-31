"""
AgentDS API Module.

REST API endpoints for external integrations.

Author: Malav Patel
"""

from agentds.web.api.webhooks import create_api, api_router

__all__ = ["create_api", "api_router"]
