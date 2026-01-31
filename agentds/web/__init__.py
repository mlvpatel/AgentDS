"""
AgentDS Web Module.

Provides Gradio web interface and REST API endpoints.

Author: Malav Patel
"""

from agentds.web.app import create_app, launch_app

__all__ = [
    "create_app",
    "launch_app",
]
