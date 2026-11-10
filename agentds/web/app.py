"""
AgentDS Gradio Web Interface.

Interactive web UI for the AgentDS pipeline with human-in-the-loop controls.

Author: Malav Patel
"""

from __future__ import annotations

import uuid
from collections.abc import Generator
from typing import Any

import gradio as gr

from agentds.agents.base import AgentAction
from agentds.core.config import Settings, get_settings
from agentds.core.job_queue import JobQueue
from agentds.core.logger import get_logger, setup_logging
from agentds.workflows.pipeline import AgentDSPipeline, PipelineConfig, PipelinePhase

logger = get_logger(__name__)

# Pipelines paused for human review, keyed by job id. Entries are removed when
# the pipeline completes, fails or is stopped.
_active_pipelines: dict[str, AgentDSPipeline] = {}


def create_app(settings: Settings | None = None) -> gr.Blocks:
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
        gr.State(value={})

        # Header
        gr.Markdown(
            """
            # AgentDS
            ### Autonomous Data Science Pipeline with Multi-Agent Orchestration

            Upload your data, describe your task, and let AI handle the rest.
            """
        )

        with gr.Tabs():
            # ==================== TAB 1: Pipeline ====================
            with gr.Tab("Pipeline", id="pipeline"), gr.Row():
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

                    # Human-in-the-loop controls, shown while a run waits for review
                    with gr.Column(visible=False) as review_panel:
                        with gr.Row():
                            approve_btn = gr.Button("Approve & Continue", variant="primary")
                            rerun_btn = gr.Button("Re-run")
                            skip_btn = gr.Button("Skip")
                            stop_btn = gr.Button("Stop Pipeline", variant="stop")

                        feedback_input = gr.Textbox(
                            label="Feedback (optional)",
                            placeholder="Enter feedback for re-run...",
                        )
                        rerun_feedback_btn = gr.Button("Re-run with Feedback")

                    # Output artifacts
                    with gr.Accordion("Output Artifacts", open=False):
                        artifacts_list = gr.JSON(label="Artifacts")
                        gr.Button("Download All Outputs")

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
                        gr.Textbox(
                            label="OpenAI API Key",
                            type="password",
                            placeholder="sk-...",
                        )
                        gr.Textbox(
                            label="Anthropic API Key",
                            type="password",
                            placeholder="sk-ant-...",
                        )
                        gr.Textbox(
                            label="Groq API Key",
                            type="password",
                        )

                    with gr.Column():
                        gr.Markdown("#### Model Settings")
                        gr.Dropdown(
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
                        gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.0,
                            step=0.1,
                            label="Temperature",
                        )

                gr.Button("Save Configuration")

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
            data_file: Any | None,
            data_url: str,
            task_desc: str,
            phases: list[str],
            human_loop: bool,
            llm_preset: str,
        ) -> Generator[tuple[Any, ...], None, None]:
            """Start the pipeline and report its real outcome."""
            # Validate inputs
            if not data_file and not data_url:
                yield _message("Error: Please provide data file or URL", "No data source")
                return

            if not task_desc:
                yield _message("Error: Please describe your task", "No task description")
                return

            # Determine data source
            data_source = data_file.name if data_file else data_url

            # Create pipeline config
            phase_enums = [PipelinePhase(p) for p in phases]
            config = PipelineConfig(
                phases=phase_enums,
                human_in_loop=human_loop,
            )

            # Share the app's job queue so runs appear in the Jobs tab
            pipeline = AgentDSPipeline(config=config, settings=settings, job_queue=job_queue)
            job_id = str(uuid.uuid4())

            first_agent = next(iter(pipeline.agent_sequence()), "pipeline")
            yield _message("Running...", f"### {first_agent}\n\nRunning...")

            try:
                result = pipeline.run(
                    data_source=data_source,
                    task_description=task_desc,
                    job_id=job_id,
                )
            except Exception as e:
                yield _message(f"Error: {e}", f"Pipeline failed: {e}")
                return

            if result["status"] == "awaiting_approval":
                _active_pipelines[job_id] = pipeline
            yield _render(result)

        def _message(status: str, agent_info: str) -> tuple[Any, ...]:
            """Outputs for a message when no pipeline is awaiting review."""
            return status, 0, agent_info, {}, None, gr.update(visible=False), ""

        def _render(result: dict[str, Any]) -> tuple[Any, ...]:
            """Outputs for a pipeline result from run() or resume()."""
            status = result["status"]
            agent_name = result["current_agent"]
            awaiting = status == "awaiting_approval"

            if awaiting:
                agent_result = result["state"].get("agent_results", {}).get(agent_name, {})
                status_text = f"Waiting for your review of **{agent_name}**"
                agent_info = f"### {agent_name}\n\nStatus: {agent_result.get('status', 'unknown')}"
                if agent_result.get("error"):
                    agent_info += f"\n\nError: {agent_result['error']}"
                if agent_result.get("approval_message"):
                    agent_info += f"\n\n{agent_result['approval_message']}"
            elif status == "completed":
                status_text = "Pipeline completed successfully!"
                agent_info = "All agents completed"
            elif status == "cancelled":
                status_text = "Pipeline stopped by user"
                agent_info = "Stopped"
            else:
                status_text = f"Error: {result['error'] or 'Pipeline failed'}"
                agent_info = f"Pipeline failed: {result['error'] or 'unknown error'}"

            summary = {
                "job_id": result["job_id"],
                "status": status,
                "current_agent": agent_name,
                "outputs": result["state"].get("final_outputs", {}),
            }
            return (
                status_text,
                result["progress_percent"],
                agent_info,
                summary,
                result["job_id"] if awaiting else None,
                gr.update(visible=awaiting),
                "",
            )

        def _review(job_id: str | None, action: str, feedback: str | None = None) -> tuple[Any, ...]:
            """Apply a review action to this session's paused pipeline."""
            pipeline = _active_pipelines.get(job_id) if job_id else None
            if pipeline is None:
                return _message("No pipeline is waiting for review", "No agent running")
            try:
                result = pipeline.resume(job_id, action, feedback or None)
            except Exception as e:
                _active_pipelines.pop(job_id, None)
                return _message(f"Error: {e}", f"Pipeline failed: {e}")
            if result["status"] != "awaiting_approval":
                _active_pipelines.pop(job_id, None)
            return _render(result)

        def handle_approve(job_id: str | None) -> tuple[Any, ...]:
            """Approve the current agent and continue."""
            return _review(job_id, AgentAction.APPROVE_AND_CONTINUE.value)

        def handle_rerun(job_id: str | None) -> tuple[Any, ...]:
            """Re-run the current agent."""
            return _review(job_id, AgentAction.RERUN.value)

        def handle_rerun_with_feedback(job_id: str | None, feedback: str) -> tuple[Any, ...]:
            """Re-run the current agent with the reviewer's feedback."""
            if not feedback or not feedback.strip():
                return _review(job_id, AgentAction.RERUN.value)
            return _review(job_id, AgentAction.RERUN_WITH_FEEDBACK.value, feedback)

        def handle_skip(job_id: str | None) -> tuple[Any, ...]:
            """Skip ahead to the next agent."""
            return _review(job_id, AgentAction.SKIP.value)

        def handle_stop(job_id: str | None) -> tuple[Any, ...]:
            """Stop the pipeline."""
            return _review(job_id, AgentAction.STOP_PIPELINE.value)

        def refresh_jobs() -> list[list[str]]:
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
                with open(log_file) as f:
                    lines = f.readlines()[-100:]  # Last 100 lines
                    return "".join(lines)
            return "No logs available"

        # Connect events. Every pipeline handler updates the same outputs so the
        # review panel and job id stay in sync with the pipeline's state.
        pipeline_outputs = [
            status_md,
            progress_bar,
            agent_output,
            artifacts_list,
            job_id_state,
            review_panel,
            feedback_input,
        ]

        start_btn.click(
            fn=start_pipeline,
            inputs=[data_file, data_url, task_desc, phases, human_loop, llm_preset],
            outputs=pipeline_outputs,
        )

        for button, handler in (
            (approve_btn, handle_approve),
            (rerun_btn, handle_rerun),
            (skip_btn, handle_skip),
            (stop_btn, handle_stop),
        ):
            button.click(fn=handler, inputs=[job_id_state], outputs=pipeline_outputs)

        rerun_feedback_btn.click(
            fn=handle_rerun_with_feedback,
            inputs=[job_id_state, feedback_input],
            outputs=pipeline_outputs,
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
    settings: Settings | None = None,
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
        "Launching AgentDS web interface",
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

    parser = argparse.ArgumentParser(description="AgentDS Web Interface")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=7860, help="Server port")
    parser.add_argument("--share", action="store_true", help="Create public URL")

    args = parser.parse_args()
    launch_app(host=args.host, port=args.port, share=args.share)
