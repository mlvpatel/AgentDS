"""
Personal Data Scientist Gradio Web Interface.

Interactive web UI for the Personal Data Scientist pipeline with human-in-the-loop controls.

Author: Malav Patel
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import gradio as gr

from agentds.agents import AgentAction
from agentds.core.config import Settings, get_settings
from agentds.core.job_queue import JobQueue, JobStatus
from agentds.core.logger import get_logger, setup_logging
from agentds.workflows.pipeline import AgentDSPipeline, PipelineConfig, PipelinePhase

logger = get_logger(__name__)

# Global state for active pipelines
_active_pipelines: Dict[str, Dict[str, Any]] = {}


def create_app(settings: Optional[Settings] = None) -> gr.Blocks:
    """
    Create Gradio application.

    Args:
        settings: Application settings

    Returns:
        Gradio Blocks application
    """
    settings = settings or get_settings()
    job_queue = JobQueue(settings)

    # Custom CSS
    custom_css = """
    .status-running { color: #3b82f6; font-weight: bold; }
    .status-completed { color: #22c55e; font-weight: bold; }
    .status-failed { color: #ef4444; font-weight: bold; }
    .status-paused { color: #f59e0b; font-weight: bold; }
    .agent-card { 
        border: 1px solid #e5e7eb; 
        border-radius: 8px; 
        padding: 16px; 
        margin: 8px 0;
    }
    .progress-bar {
        background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%);
        border-radius: 4px;
        height: 8px;
    }
    """

    with gr.Blocks(
        title="AgentDS",
        theme=gr.themes.Soft(),
        css=custom_css,
    ) as app:
        # State variables
        job_id_state = gr.State(value=None)
        pipeline_state = gr.State(value={})

        # Header
        gr.Markdown(
            """
            # AgentDS
            ### Autonomous Data Science Pipeline with Multi-Agent Orchestration

            Upload your data, describe your task, and let AI handle the rest.
            """
        )

        with gr.Tabs() as tabs:
            # ==================== TAB 1: Pipeline ====================
            with gr.Tab("Pipeline", id="pipeline"):
                with gr.Row():
                    # Left column: Configuration
                    with gr.Column(scale=1):
                        gr.Markdown("### Configuration")

                        # Data source
                        data_file = gr.File(
                            label="Upload Data File",
                            file_types=[".csv", ".parquet", ".json", ".xlsx"],
                        )
                        data_url = gr.Textbox(
                            label="Or Enter Data URL",
                            placeholder="s3://bucket/data.csv or https://...",
                        )

                        # Task description
                        task_desc = gr.Textbox(
                            label="Task Description",
                            placeholder="Predict customer churn based on transaction history...",
                            lines=3,
                        )

                        # Advanced options
                        with gr.Accordion("Advanced Options", open=False):
                            phases = gr.CheckboxGroup(
                                choices=["build", "deploy", "learn"],
                                value=["build", "deploy"],
                                label="Pipeline Phases",
                            )
                            human_loop = gr.Checkbox(
                                value=True,
                                label="Human-in-the-Loop",
                            )
                            llm_preset = gr.Dropdown(
                                choices=["balanced", "budget", "quality", "local"],
                                value="balanced",
                                label="LLM Preset",
                            )

                        # Start button
                        start_btn = gr.Button(
                            "Start Pipeline",
                            variant="primary",
                            size="lg",
                        )

                    # Right column: Progress and outputs
                    with gr.Column(scale=2):
                        gr.Markdown("### Pipeline Progress")

                        # Status display
                        status_md = gr.Markdown("Ready to start...")

                        # Progress bar
                        progress_bar = gr.Slider(
                            minimum=0,
                            maximum=100,
                            value=0,
                            label="Progress",
                            interactive=False,
                        )

                        # Current agent output
                        agent_output = gr.Markdown(
                            label="Current Agent",
                            value="No agent running",
                        )

                        # Human-in-the-loop controls
                        with gr.Row(visible=False) as hitl_controls:
                            approve_btn = gr.Button("Approve & Continue", variant="primary")
                            rerun_btn = gr.Button("Re-run")
                            skip_btn = gr.Button("Skip")
                            stop_btn = gr.Button("Stop Pipeline", variant="stop")

                        # Feedback input
                        feedback_input = gr.Textbox(
                            label="Feedback (optional)",
                            placeholder="Enter feedback for re-run...",
                            visible=False,
                        )
                        rerun_feedback_btn = gr.Button(
                            "Re-run with Feedback",
                            visible=False,
                        )

                        # Output artifacts
                        with gr.Accordion("Output Artifacts", open=False):
                            artifacts_list = gr.JSON(label="Artifacts")
                            download_btn = gr.Button("Download All Outputs")

            # ==================== TAB 2: Agents ====================
            with gr.Tab("Agents", id="agents"):
                gr.Markdown("### Agent Status")

                with gr.Row():
                    for phase_name, agents in [
                        ("Build", ["DataLoaderAgent", "DataCleaningAgent", "EDACopilotAgent", "FeatureEngineerAgent", "AutoMLAgent"]),
                        ("Deploy", ["APIWrapperAgent", "DevOpsAgent", "CloudDeployAgent"]),
                        ("Learn", ["DriftMonitorAgent", "OptimizationAgent"]),
                    ]:
                        with gr.Column():
                            gr.Markdown(f"#### {phase_name} Phase")
                            for agent in agents:
                                with gr.Row(elem_classes="agent-card"):
                                    gr.Markdown(f"**{agent}**")

            # ==================== TAB 3: Configuration ====================
            with gr.Tab("Configuration", id="config"):
                gr.Markdown("### LLM Configuration")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### API Keys")
                        openai_key = gr.Textbox(
                            label="OpenAI API Key",
                            type="password",
                            placeholder="sk-...",
                        )
                        anthropic_key = gr.Textbox(
                            label="Anthropic API Key",
                            type="password",
                            placeholder="sk-ant-...",
                        )
                        groq_key = gr.Textbox(
                            label="Groq API Key",
                            type="password",
                        )

                    with gr.Column():
                        gr.Markdown("#### Model Settings")
                        default_model = gr.Dropdown(
                            choices=[
                                "openai/gpt-4o",
                                "openai/gpt-4o-mini",
                                "anthropic/claude-3-5-sonnet-20241022",
                                "groq/llama-3.1-70b-versatile",
                                "ollama/llama3.1:70b",
                            ],
                            value="openai/gpt-4o-mini",
                            label="Default Model",
                        )
                        temperature = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.0,
                            step=0.1,
                            label="Temperature",
                        )

                save_config_btn = gr.Button("Save Configuration")

            # ==================== TAB 4: Jobs ====================
            with gr.Tab("Jobs", id="jobs"):
                gr.Markdown("### Job History")

                refresh_jobs_btn = gr.Button("Refresh")
                jobs_table = gr.Dataframe(
                    headers=["Job ID", "Name", "Status", "Created", "Duration"],
                    label="Jobs",
                )

            # ==================== TAB 5: Logs ====================
            with gr.Tab("Logs", id="logs"):
                gr.Markdown("### System Logs")

                log_output = gr.Code(
                    label="Logs",
                    language="json",
                    lines=20,
                )
                refresh_logs_btn = gr.Button("Refresh Logs")

        # ==================== Event Handlers ====================

        def start_pipeline(
            data_file: Optional[Any],
            data_url: str,
            task_desc: str,
            phases: List[str],
            human_loop: bool,
            llm_preset: str,
        ) -> Generator[Tuple[str, float, str, Dict], None, None]:
            """Start the pipeline and stream progress."""
            # Validate inputs
            if not data_file and not data_url:
                yield "Error: Please provide data file or URL", 0, "No data source", {}
                return

            if not task_desc:
                yield "Error: Please describe your task", 0, "No task description", {}
                return

            # Determine data source
            if data_file:
                data_source = data_file.name
            else:
                data_source = data_url

            # Create pipeline config
            phase_enums = [PipelinePhase(p) for p in phases]
            config = PipelineConfig(
                phases=phase_enums,
                human_in_loop=human_loop,
            )

            # Create pipeline
            pipeline = AgentDSPipeline(config=config, settings=settings)

            # Generate job ID
            job_id = str(uuid.uuid4())
            _active_pipelines[job_id] = {
                "pipeline": pipeline,
                "status": "running",
                "current_agent": None,
            }

            # Run pipeline
            try:
                result = pipeline.run(
                    data_source=data_source,
                    task_description=task_desc,
                    job_id=job_id,
                )

                # Stream progress updates
                total_agents = sum(len(pipeline.PHASE_AGENTS.get(p, [])) for p in phase_enums)
                completed = 0

                for agent_name in pipeline._agents.keys():
                    _active_pipelines[job_id]["current_agent"] = agent_name

                    status = f"Running: {agent_name}"
                    progress = (completed / total_agents) * 100
                    agent_info = f"### {agent_name}\n\nProcessing..."

                    yield status, progress, agent_info, {}

                    # Simulate agent execution (in real implementation, this would be event-driven)
                    time.sleep(0.5)
                    completed += 1

                # Final result
                final_status = "Pipeline completed successfully!"
                yield final_status, 100, "All agents completed", result

            except Exception as e:
                yield f"Error: {str(e)}", 0, f"Pipeline failed: {e}", {}

        def handle_approve(job_id: str) -> Tuple[str, str]:
            """Handle approve action."""
            if job_id and job_id in _active_pipelines:
                pipeline_data = _active_pipelines[job_id]
                pipeline_data["status"] = "approved"
                return "Approved! Continuing...", ""
            return "No active pipeline", ""

        def handle_rerun(job_id: str) -> Tuple[str, str]:
            """Handle rerun action."""
            if job_id and job_id in _active_pipelines:
                return "Re-running current agent...", ""
            return "No active pipeline", ""

        def handle_skip(job_id: str) -> Tuple[str, str]:
            """Handle skip action."""
            if job_id and job_id in _active_pipelines:
                return "Skipping current agent...", ""
            return "No active pipeline", ""

        def handle_stop(job_id: str) -> Tuple[str, str]:
            """Handle stop action."""
            if job_id and job_id in _active_pipelines:
                _active_pipelines[job_id]["status"] = "stopped"
                return "Pipeline stopped by user", ""
            return "No active pipeline", ""

        def refresh_jobs() -> List[List[str]]:
            """Refresh job list."""
            jobs = job_queue.list_jobs(limit=50)
            return [
                [
                    j.id[:8],
                    j.name[:30],
                    j.status.value,
                    j.created_at.strftime("%Y-%m-%d %H:%M"),
                    f"{j.duration_seconds:.1f}s" if j.duration_seconds else "-",
                ]
                for j in jobs
            ]

        def refresh_logs() -> str:
            """Refresh log output."""
            log_file = settings.log_dir / "agentds.log"
            if log_file.exists():
                with open(log_file, "r") as f:
                    lines = f.readlines()[-100:]  # Last 100 lines
                    return "".join(lines)
            return "No logs available"

        # Connect events
        start_btn.click(
            fn=start_pipeline,
            inputs=[data_file, data_url, task_desc, phases, human_loop, llm_preset],
            outputs=[status_md, progress_bar, agent_output, artifacts_list],
        )

        approve_btn.click(
            fn=handle_approve,
            inputs=[job_id_state],
            outputs=[status_md, feedback_input],
        )

        rerun_btn.click(
            fn=handle_rerun,
            inputs=[job_id_state],
            outputs=[status_md, feedback_input],
        )

        skip_btn.click(
            fn=handle_skip,
            inputs=[job_id_state],
            outputs=[status_md, feedback_input],
        )

        stop_btn.click(
            fn=handle_stop,
            inputs=[job_id_state],
            outputs=[status_md, feedback_input],
        )

        refresh_jobs_btn.click(
            fn=refresh_jobs,
            outputs=[jobs_table],
        )

        refresh_logs_btn.click(
            fn=refresh_logs,
            outputs=[log_output],
        )

    return app


def launch_app(
    host: str = "0.0.0.0",
    port: int = 7860,
    share: bool = False,
    settings: Optional[Settings] = None,
) -> None:
    """
    Launch the Gradio application.

    Args:
        host: Server host
        port: Server port
        share: Create public URL
        settings: Application settings
    """
    settings = settings or get_settings()
    setup_logging(level=settings.log_level, log_format=settings.log_format)

    logger.info(
        "Launching Personal Data Scientist web interface",
        host=host,
        port=port,
    )

    app = create_app(settings)
    app.launch(
        server_name=host,
        server_port=port,
        share=share,
    )


# CLI entry point
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Personal Data Scientist Web Interface")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=7860, help="Server port")
    parser.add_argument("--share", action="store_true", help="Create public URL")

    args = parser.parse_args()
    launch_app(host=args.host, port=args.port, share=args.share)
