"""
AgentDS Command Line Interface.

Provides CLI commands for running pipelines and managing the system.

Author: Malav Patel
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional
import uuid

import click

from agentds import __version__
from agentds.core.config import get_settings
from agentds.core.logger import setup_logging


@click.group()
@click.version_option(version=__version__, prog_name="agentds")
@click.option("--debug", is_flag=True, help="Enable debug mode")
@click.pass_context
def cli(ctx: click.Context, debug: bool) -> None:
    """AgentDS - Autonomous Data Science Pipeline.

    A multi-agent system for automating the complete data science lifecycle.
    """
    ctx.ensure_object(dict)
    ctx.obj["debug"] = debug

    # Setup logging
    log_level = "DEBUG" if debug else "INFO"
    setup_logging(level=log_level)


@cli.command()
@click.option("--host", default="0.0.0.0", help="Server host")
@click.option("--port", default=7860, type=int, help="Server port")
@click.option("--share", is_flag=True, help="Create public URL")
@click.pass_context
def web(ctx: click.Context, host: str, port: int, share: bool) -> None:
    """Launch the Gradio web interface."""
    from agentds.web.app import launch_app

    click.echo(f"[INFO] Launching AgentDS web interface on {host}:{port}")
    launch_app(host=host, port=port, share=share)


@cli.command()
@click.option("--host", default="0.0.0.0", help="Server host")
@click.option("--port", default=8000, type=int, help="Server port")
@click.pass_context
def api(ctx: click.Context, host: str, port: int) -> None:
    """Launch the REST API server."""
    import uvicorn

    from agentds.web.api.webhooks import create_api

    click.echo(f"[INFO] Launching AgentDS API on {host}:{port}")
    app = create_api()
    uvicorn.run(app, host=host, port=port)


@cli.command()
@click.argument("data_source", type=click.Path(exists=True))
@click.option("--task", "-t", required=True, help="Task description")
@click.option("--output", "-o", default="./outputs", help="Output directory")
@click.option("--phases", "-p", multiple=True, default=["build", "deploy"],
              help="Phases to run (build, deploy, learn)")
@click.option("--no-hitl", is_flag=True, help="Disable human-in-the-loop")
@click.pass_context
def run(
    ctx: click.Context,
    data_source: str,
    task: str,
    output: str,
    phases: tuple,
    no_hitl: bool,
) -> None:
    """Run the complete pipeline on a dataset.

    Example:
        agentds run data.csv -t "Predict customer churn" -o ./outputs
    """
    from agentds.workflows.pipeline import AgentDSPipeline, PipelineConfig, PipelinePhase

    click.echo("[INFO] Starting AgentDS pipeline")
    click.echo(f"[INFO] Data source: {data_source}")
    click.echo(f"[INFO] Task: {task}")
    click.echo(f"[INFO] Phases: {', '.join(phases)}")

    # Create pipeline config
    phase_enums = [PipelinePhase(p) for p in phases]
    config = PipelineConfig(
        phases=phase_enums,
        human_in_loop=not no_hitl,
    )

    # Create and run pipeline
    pipeline = AgentDSPipeline(config=config)

    try:
        result = pipeline.run(
            data_source=data_source,
            task_description=task,
            output_destination=output,
        )

        click.echo("[OK] Pipeline completed successfully!")
        click.echo(f"[INFO] Job ID: {result['job_id']}")
        click.echo(f"[INFO] Outputs: {output}")

    except Exception as e:
        click.echo(f"[ERROR] Pipeline failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("agent_name")
@click.argument("data_source", type=click.Path(exists=True))
@click.option("--output", "-o", default="./outputs", help="Output directory")
@click.pass_context
def agent(ctx: click.Context, agent_name: str, data_source: str, output: str) -> None:
    """Run a single agent.

    Available agents:
    - DataLoaderAgent
    - DataCleaningAgent
    - EDACopilotAgent
    - FeatureEngineerAgent
    - AutoMLAgent
    - APIWrapperAgent
    - DevOpsAgent
    - CloudDeployAgent
    - DriftMonitorAgent
    - OptimizationAgent

    Example:
        agentds agent DataLoaderAgent data.csv -o ./outputs
    """
    from agentds.agents import AGENT_REGISTRY
    from agentds.agents.base import AgentContext
    from agentds.core.artifact_store import ArtifactStore
    from agentds.core.llm_gateway import LLMGateway

    if agent_name not in AGENT_REGISTRY:
        click.echo(f"[ERROR] Unknown agent: {agent_name}", err=True)
        click.echo(f"[INFO] Available agents: {', '.join(AGENT_REGISTRY.keys())}")
        sys.exit(1)

    click.echo(f"[INFO] Running {agent_name}")
    click.echo(f"[INFO] Data source: {data_source}")

    settings = get_settings()
    agent_class = AGENT_REGISTRY[agent_name]
    agent_instance = agent_class(settings=settings)

    context = AgentContext(
        job_id=str(uuid.uuid4()),
        settings=settings,
        llm_gateway=LLMGateway(settings),
        artifact_store=ArtifactStore(settings),
        task_description="CLI agent run",
        extra={"data_source": data_source},
    )

    try:
        result = agent_instance.run(context)
        click.echo(f"[OK] Agent completed with status: {result.status.value}")
        click.echo(f"[INFO] Duration: {result.duration_seconds:.2f}s")
        click.echo(f"[INFO] LLM calls: {result.llm_calls}")

        if result.error:
            click.echo(f"[ERROR] {result.error}", err=True)
            sys.exit(1)

    except Exception as e:
        click.echo(f"[ERROR] Agent failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Check system status and connections."""
    from agentds.core.llm_gateway import LLMGateway

    click.echo("[INFO] Checking AgentDS system status...")
    click.echo("")

    settings = get_settings()

    # Check LLM providers
    click.echo("LLM Providers:")
    providers = settings.llm.get_available_providers()
    for provider in providers:
        click.echo(f"  [OK] {provider}")
    if not providers:
        click.echo("  [WARN] No LLM providers configured")

    # Check Redis
    click.echo("")
    click.echo("Redis:")
    try:
        import redis
        r = redis.from_url(settings.redis.url)
        if r.ping():
            click.echo(f"  [OK] Connected to {settings.redis.url}")
        else:
            click.echo("  [WARN] Redis not responding")
    except Exception as e:
        click.echo(f"  [WARN] Redis unavailable: {e}")

    # Check directories
    click.echo("")
    click.echo("Directories:")
    for name, path in [
        ("Output", settings.output_dir),
        ("Temp", settings.temp_dir),
        ("Logs", settings.log_dir),
        ("Checkpoints", settings.checkpoint_dir),
    ]:
        if path.exists():
            click.echo(f"  [OK] {name}: {path}")
        else:
            click.echo(f"  [WARN] {name}: {path} (will be created)")


@cli.command()
@click.pass_context
def config(ctx: click.Context) -> None:
    """Display current configuration."""
    import yaml

    settings = get_settings()

    config_data = {
        "app": {
            "name": settings.app_name,
            "version": settings.app_version,
            "debug": settings.debug,
            "environment": settings.environment,
        },
        "server": {
            "host": settings.host,
            "port": settings.port,
        },
        "llm": {
            "default_model": settings.llm.default_model,
            "default_temperature": settings.llm.default_temperature,
            "available_providers": settings.llm.get_available_providers(),
        },
        "paths": {
            "output_dir": str(settings.output_dir),
            "temp_dir": str(settings.temp_dir),
            "log_dir": str(settings.log_dir),
        },
    }

    click.echo(yaml.dump(config_data, default_flow_style=False))


@cli.command()
@click.pass_context
def jobs(ctx: click.Context) -> None:
    """List recent jobs."""
    from agentds.core.job_queue import JobQueue

    settings = get_settings()
    job_queue = JobQueue(settings)

    jobs_list = job_queue.list_jobs(limit=20)

    if not jobs_list:
        click.echo("[INFO] No jobs found")
        return

    click.echo(f"{'ID':<12} {'Name':<30} {'Status':<12} {'Duration':<10}")
    click.echo("-" * 70)

    for job in jobs_list:
        duration = f"{job.duration_seconds:.1f}s" if job.duration_seconds else "-"
        click.echo(
            f"{job.id[:10]:<12} {job.name[:28]:<30} {job.status.value:<12} {duration:<10}"
        )


def main() -> None:
    """Entry point for CLI."""
    cli(obj={})


if __name__ == "__main__":
    main()
