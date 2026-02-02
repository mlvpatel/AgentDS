"""
AgentDS API Module.

REST API endpoints for external integrations.

Author: Malav Patel
"""

from agentds.web.api.middleware import (  # noqa: I001
    AuthenticationMiddleware,
    RateLimitMiddleware,
    RateLimiter,
    require_api_key,
)
from agentds.web.api.webhooks import api_router, create_api

__all__ = [
    "create_api",
    "api_router",
    "AuthenticationMiddleware",
    "RateLimitMiddleware",
    "RateLimiter",
    "require_api_key",
]
