"""
AgentDS CloudDeployAgent.

Deploys containerized applications to cloud platforms.

Author: Malav Patel
"""

from __future__ import annotations

import contextlib
from enum import Enum
from pathlib import Path
from typing import Any

from agentds.agents.base import (
    AgentContext,
    AgentResult,
    AgentStatus,
    BaseAgent,
)
from agentds.core.artifact_store import ArtifactType
from agentds.core.logger import get_logger

logger = get_logger(__name__)


class DeploymentPlatform(str, Enum):
    """Supported deployment platforms."""

    DOCKER_LOCAL = "docker_local"
    AWS_ECS = "aws_ecs"
    GCP_CLOUD_RUN = "gcp_cloud_run"
    AZURE_CONTAINER = "azure_container"
    KUBERNETES = "kubernetes"


class CloudDeployAgent(BaseAgent):
    """
    Agent for deploying to cloud platforms.

    Supported platforms:
    - Docker (local)
    - AWS ECS/Fargate
    - GCP Cloud Run
    - Azure Container Instances
    - Kubernetes
    """

    name = "CloudDeployAgent"
    description = "Deploy containerized applications to cloud platforms"
    phase = "deploy"
    complexity = "MEDIUM"
    input_types = ["dockerfile", "docker-compose"]
    output_types = ["deployment_url", "deployment_status"]

    def get_system_prompt(self) -> str:
        """Get system prompt for cloud deployment."""
        return """You are CloudDeployAgent, a specialized agent for cloud deployments.

Your responsibilities:
1. Build Docker images
2. Push images to container registries
3. Deploy to selected cloud platform
4. Configure networking and scaling
5. Verify deployment health

Supported platforms:
- Docker (local): For development and testing
- AWS ECS/Fargate: Serverless containers on AWS
- GCP Cloud Run: Serverless containers on GCP
- Azure Container Instances: Serverless containers on Azure
- Kubernetes: Any K8s cluster

Best practices:
- Always verify Docker daemon is running
- Tag images with version and latest
- Use health checks for deployments
- Configure appropriate resource limits
- Enable auto-scaling where available
"""

    def execute(self, context: AgentContext) -> AgentResult:
        """
        Execute cloud deployment.

        Args:
            context: Execution context with Docker configuration

        Returns:
            AgentResult with deployment status and URL
        """
        result = AgentResult(
            agent_name=self.name,
            status=AgentStatus.RUNNING,
        )

        try:
            # Get DevOps outputs
            devops_result = context.previous_results.get("DevOpsAgent")
            if not devops_result:
                raise ValueError("DevOpsAgent result not found")

            output_dir = devops_result.outputs.get("output_dir")
            if not output_dir:
                raise ValueError("DevOps output directory not found")

            # Determine deployment platform
            platform = context.extra.get("platform", DeploymentPlatform.DOCKER_LOCAL)
            if isinstance(platform, str):
                platform = DeploymentPlatform(platform)

            logger.info("Starting deployment", platform=platform.value)

            # Execute deployment based on platform
            if platform == DeploymentPlatform.DOCKER_LOCAL:
                deployment_info = self._deploy_docker_local(output_dir, context)
            elif platform == DeploymentPlatform.AWS_ECS:
                deployment_info = self._deploy_aws_ecs(output_dir, context)
            elif platform == DeploymentPlatform.GCP_CLOUD_RUN:
                deployment_info = self._deploy_gcp_cloud_run(output_dir, context)
            elif platform == DeploymentPlatform.AZURE_CONTAINER:
                deployment_info = self._deploy_azure_container(output_dir, context)
            elif platform == DeploymentPlatform.KUBERNETES:
                deployment_info = self._deploy_kubernetes(output_dir, context)
            else:
                raise ValueError(f"Unsupported platform: {platform}")

            # Save deployment report
            report = self._generate_deployment_report(deployment_info, platform)
            artifact_id = self.save_artifact(
                job_id=context.job_id,
                name="deployment_report.json",
                data=report,
                artifact_type=ArtifactType.REPORT,
                description=f"Deployment report for {platform.value}",
            )

            # Prepare result
            result.outputs = {
                "platform": platform.value,
                "deployment_url": deployment_info.get("url"),
                "deployment_status": deployment_info.get("status"),
                "container_id": deployment_info.get("container_id"),
                "deployment_info": deployment_info,
            }
            result.artifacts.append(artifact_id)
            result.approval_message = self._format_approval_message(
                deployment_info, platform
            )
            result.mark_completed()

        except Exception as e:
            logger.error("Cloud deployment failed", error=str(e), exc_info=True)
            result.mark_failed(str(e))

        return result

    def _deploy_docker_local(
        self, output_dir: Path, context: AgentContext
    ) -> dict[str, Any]:
        """Deploy using Docker locally."""
        import docker

        try:
            client = docker.from_env()
        except Exception as e:
            raise RuntimeError(f"Docker daemon not available: {e}") from e

        project_name = f"agentds-{context.job_id[:8]}"
        image_name = f"{project_name}:latest"

        # Build image
        logger.info("Building Docker image", image=image_name)
        try:
            image, build_logs = client.images.build(
                path=str(output_dir),
                tag=image_name,
                rm=True,
            )
            for log in build_logs:
                if "stream" in log:
                    logger.debug(log["stream"].strip())
        except Exception as e:
            raise RuntimeError(f"Docker build failed: {e}") from e

        # Run container
        logger.info("Starting container")
        try:
            container = client.containers.run(
                image_name,
                name=project_name,
                ports={"8000/tcp": 8000},
                detach=True,
                remove=False,
            )
        except Exception as e:
            raise RuntimeError(f"Container start failed: {e}") from e

        # Wait for health check
        import time

        max_wait = 30
        for _ in range(max_wait):
            container.reload()
            if container.status == "running":
                # Check if health endpoint responds
                with contextlib.suppress(Exception):
                    import httpx

                    response = httpx.get("http://localhost:8000/health", timeout=2)
                    if response.status_code == 200:
                        break
            time.sleep(1)

        return {
            "status": "running" if container.status == "running" else "unknown",
            "url": "http://localhost:8000",
            "container_id": container.id[:12],
            "image": image_name,
            "platform": "docker_local",
            "logs_command": f"docker logs {project_name}",
            "stop_command": f"docker stop {project_name}",
        }

    def _deploy_aws_ecs(
        self, output_dir: Path, context: AgentContext
    ) -> dict[str, Any]:
        """Deploy to AWS ECS/Fargate."""
        # This is a simplified implementation
        # In production, use boto3 for full ECS deployment

        prompt = """Generate AWS CLI commands for deploying a Docker container to ECS Fargate.

Requirements:
- Use existing VPC and subnets
- Create task definition
- Create or update service
- Configure Application Load Balancer

Output the commands as a shell script."""

        response = self.call_llm(prompt)

        # For now, return deployment instructions
        return {
            "status": "instructions_generated",
            "url": "https://your-alb-url.amazonaws.com",
            "platform": "aws_ecs",
            "instructions": response.content,
            "note": "Manual deployment required. Run the generated commands.",
        }

    def _deploy_gcp_cloud_run(
        self, output_dir: Path, context: AgentContext
    ) -> dict[str, Any]:
        """Deploy to GCP Cloud Run."""
        project_id = context.settings.llm.vertexai_project
        service_name = f"agentds-{context.job_id[:8]}"
        region = context.settings.llm.vertexai_location

        if not project_id:
            return {
                "status": "configuration_required",
                "url": None,
                "platform": "gcp_cloud_run",
                "error": "VERTEXAI_PROJECT not configured",
                "instructions": """
To deploy to Cloud Run:
1. Set VERTEXAI_PROJECT environment variable
2. Authenticate with: gcloud auth login
3. Run: gcloud run deploy SERVICE_NAME --source .
""",
            }

        # Generate deployment commands
        commands = f"""
# Build and push to Artifact Registry
gcloud builds submit --tag gcr.io/{project_id}/{service_name}

# Deploy to Cloud Run
gcloud run deploy {service_name} \\
    --image gcr.io/{project_id}/{service_name} \\
    --platform managed \\
    --region {region} \\
    --allow-unauthenticated \\
    --memory 2Gi \\
    --cpu 2 \\
    --min-instances 0 \\
    --max-instances 10 \\
    --port 8000
"""

        return {
            "status": "instructions_generated",
            "url": f"https://{service_name}-xxxxx-{region[:2]}.a.run.app",
            "platform": "gcp_cloud_run",
            "commands": commands,
            "note": "Run the generated commands to deploy.",
        }

    def _deploy_azure_container(
        self, output_dir: Path, context: AgentContext
    ) -> dict[str, Any]:
        """Deploy to Azure Container Instances."""
        return {
            "status": "instructions_generated",
            "url": "https://your-container.azurecontainer.io",
            "platform": "azure_container",
            "instructions": """
To deploy to Azure Container Instances:
1. Create a resource group: az group create --name agentds-rg --location eastus
2. Create container registry: az acr create --resource-group agentds-rg --name agentdsregistry --sku Basic
3. Build and push: az acr build --registry agentdsregistry --image agentds:latest .
4. Deploy: az container create --resource-group agentds-rg --name agentds-api --image agentdsregistry.azurecr.io/agentds:latest --cpu 2 --memory 2 --ports 8000
""",
        }

    def _deploy_kubernetes(
        self, output_dir: Path, context: AgentContext
    ) -> dict[str, Any]:
        """Deploy to Kubernetes cluster."""
        k8s_dir = output_dir / "k8s"

        if not k8s_dir.exists():
            return {
                "status": "configuration_required",
                "url": None,
                "platform": "kubernetes",
                "error": "Kubernetes manifests not found",
            }

        # Apply manifests
        commands = []
        for manifest in k8s_dir.glob("*.yaml"):
            commands.append(f"kubectl apply -f {manifest}")

        return {
            "status": "instructions_generated",
            "url": "http://your-k8s-ingress",
            "platform": "kubernetes",
            "commands": "\n".join(commands),
            "note": "Ensure kubectl is configured for your cluster.",
        }

    def _generate_deployment_report(
        self, deployment_info: dict[str, Any], platform: DeploymentPlatform
    ) -> str:
        """Generate deployment report as JSON."""
        import json
        from datetime import datetime, timezone

        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "platform": platform.value,
            "deployment": deployment_info,
            "health_check": {
                "endpoint": "/health",
                "expected_status": 200,
            },
        }
        return json.dumps(report, indent=2)

    def _format_approval_message(
        self, deployment_info: dict[str, Any], platform: DeploymentPlatform
    ) -> str:
        """Format approval message."""
        status = deployment_info.get("status", "unknown")
        url = deployment_info.get("url", "N/A")

        return f"""
Cloud Deployment Complete
=========================

Platform: {platform.value}
Status: {status}
URL: {url}

Deployment Details:
{self._format_dict(deployment_info)}

Next Steps:
1. Verify the deployment is healthy
2. Test the API endpoints
3. Configure monitoring and alerts
4. Set up CI/CD for automated deployments

Do you want to proceed?
"""

    def _format_dict(self, d: dict[str, Any], indent: int = 2) -> str:
        """Format dictionary for display."""
        lines = []
        for key, value in d.items():
            if isinstance(value, dict):
                lines.append(f"{' ' * indent}{key}:")
                lines.append(self._format_dict(value, indent + 2))
            else:
                lines.append(f"{' ' * indent}{key}: {value}")
        return "\n".join(lines)
