"""
AgentDS DataLoaderAgent.

Loads data from various sources including files, databases, cloud storage, and APIs.

Author: Malav Patel
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import polars as pl

from agentds.agents.base import (
    AgentContext,
    AgentResult,
    AgentStatus,
    BaseAgent,
)
from agentds.core.artifact_store import ArtifactType
from agentds.core.logger import get_logger

logger = get_logger(__name__)


class DataLoaderAgent(BaseAgent):
    """
    Agent for loading data from various sources.

    Supported sources:
    - Files: CSV, Parquet, JSON, Excel, Feather, ORC
    - Cloud: S3, GCS, Azure Blob
    - Databases: PostgreSQL, MySQL, MongoDB, Snowflake
    - APIs: REST endpoints
    """

    name = "DataLoaderAgent"
    description = "Load data from various sources into a unified format"
    phase = "build"
    complexity = "LOW"
    input_types = ["file_path", "url", "sql_query", "api_endpoint"]
    output_types = ["parquet", "dataframe"]

    def get_system_prompt(self) -> str:
        """Get system prompt for data loading."""
        return """You are DataLoaderAgent, a specialized agent for loading data from various sources.

Your responsibilities:
1. Identify the data source type (file, database, cloud storage, API)
2. Determine the correct loading strategy
3. Handle encoding and format detection
4. Validate the loaded data structure
5. Report data statistics and any issues

Supported formats:
- Files: CSV, Parquet, JSON, Excel (.xlsx, .xls), Feather, ORC
- Cloud: S3 (s3://), GCS (gs://), Azure Blob (az://)
- Databases: PostgreSQL, MySQL, MongoDB, Snowflake
- APIs: REST endpoints (HTTP/HTTPS)

Output your analysis in JSON format with:
{
    "source_type": "file|database|cloud|api",
    "format": "detected format",
    "encoding": "detected or suggested encoding",
    "load_strategy": "description of loading approach",
    "potential_issues": ["list of potential issues"],
    "recommendations": ["list of recommendations"]
}
"""

    def execute(self, context: AgentContext) -> AgentResult:
        """
        Execute data loading.

        Args:
            context: Execution context with data source information

        Returns:
            AgentResult with loaded data
        """
        result = AgentResult(
            agent_name=self.name,
            status=AgentStatus.RUNNING,
        )

        try:
            # Get data source from context
            data_source = context.extra.get("data_source")
            if not data_source:
                raise ValueError("No data source provided in context")

            # Analyze the data source
            analysis = self._analyze_source(data_source, context)

            # Load the data
            df = self._load_data(data_source, analysis)

            # Get data statistics
            stats = self._get_data_stats(df)

            # Save as parquet artifact
            output_path = Path(context.settings.temp_dir) / f"{context.job_id}_raw_data.parquet"
            df.write_parquet(output_path)

            artifact_id = self.save_artifact(
                job_id=context.job_id,
                name="raw_data.parquet",
                data=output_path,
                artifact_type=ArtifactType.DATA,
                metadata={
                    "rows": stats["row_count"],
                    "columns": stats["column_count"],
                    "source": str(data_source),
                },
                description="Raw loaded data",
            )

            # Prepare result
            result.outputs = {
                "data_path": output_path,
                "artifact_id": artifact_id,
                "source_analysis": analysis,
                "statistics": stats,
            }
            result.artifacts.append(artifact_id)
            result.approval_message = self._format_approval_message(stats, analysis)
            result.mark_completed()

        except Exception as e:
            logger.error("Data loading failed", error=str(e), exc_info=True)
            result.mark_failed(str(e))

        return result

    def _analyze_source(
        self, data_source: Union[str, Path], context: AgentContext
    ) -> Dict[str, Any]:
        """Analyze the data source to determine loading strategy."""
        source_str = str(data_source)

        # Use LLM to analyze complex sources
        if context.task_description:
            prompt = f"""Analyze this data source and provide loading recommendations:

Data source: {source_str}
Task description: {context.task_description}

Provide your analysis in JSON format."""

            response = self.call_llm(prompt)
            try:
                # Try to parse JSON from response
                import re
                json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
            except Exception:
                pass

        # Default analysis based on source type
        analysis = {
            "source_type": "unknown",
            "format": "unknown",
            "encoding": "utf-8",
            "load_strategy": "auto",
            "potential_issues": [],
            "recommendations": [],
        }

        # Detect source type
        if source_str.startswith(("s3://", "gs://", "az://")):
            analysis["source_type"] = "cloud"
            analysis["load_strategy"] = "smart_open"
        elif source_str.startswith(("http://", "https://")):
            analysis["source_type"] = "api"
            analysis["load_strategy"] = "httpx"
        elif source_str.startswith(("postgresql://", "mysql://", "mongodb://")):
            analysis["source_type"] = "database"
            analysis["load_strategy"] = "connectorx"
        else:
            analysis["source_type"] = "file"
            analysis["load_strategy"] = "polars"

        # Detect format from extension
        path = Path(source_str)
        ext = path.suffix.lower()
        format_map = {
            ".csv": "csv",
            ".parquet": "parquet",
            ".json": "json",
            ".jsonl": "jsonl",
            ".xlsx": "excel",
            ".xls": "excel",
            ".feather": "feather",
            ".orc": "orc",
            ".tsv": "tsv",
        }
        analysis["format"] = format_map.get(ext, "unknown")

        return analysis

    def _load_data(
        self, data_source: Union[str, Path], analysis: Dict[str, Any]
    ) -> pl.DataFrame:
        """Load data based on analysis."""
        source_str = str(data_source)
        source_type = analysis["source_type"]
        fmt = analysis["format"]

        logger.info(
            "Loading data",
            source=source_str,
            source_type=source_type,
            format=fmt,
        )

        # File loading
        if source_type == "file":
            return self._load_file(Path(source_str), fmt, analysis)

        # Cloud storage loading
        elif source_type == "cloud":
            return self._load_cloud(source_str, fmt, analysis)

        # Database loading
        elif source_type == "database":
            return self._load_database(source_str, analysis)

        # API loading
        elif source_type == "api":
            return self._load_api(source_str, analysis)

        else:
            raise ValueError(f"Unsupported source type: {source_type}")

    def _load_file(
        self, path: Path, fmt: str, analysis: Dict[str, Any]
    ) -> pl.DataFrame:
        """Load data from local file."""
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if fmt == "csv":
            return pl.read_csv(
                path,
                encoding=analysis.get("encoding", "utf-8"),
                infer_schema_length=10000,
            )
        elif fmt == "tsv":
            return pl.read_csv(
                path,
                separator="\t",
                encoding=analysis.get("encoding", "utf-8"),
            )
        elif fmt == "parquet":
            return pl.read_parquet(path)
        elif fmt == "json":
            return pl.read_json(path)
        elif fmt == "jsonl":
            return pl.read_ndjson(path)
        elif fmt == "excel":
            return pl.read_excel(path)
        elif fmt == "feather":
            return pl.read_ipc(path)
        else:
            # Try to infer format
            try:
                return pl.read_csv(path)
            except Exception:
                try:
                    return pl.read_parquet(path)
                except Exception:
                    raise ValueError(f"Unable to load file: {path}")

    def _load_cloud(
        self, uri: str, fmt: str, analysis: Dict[str, Any]
    ) -> pl.DataFrame:
        """Load data from cloud storage."""
        from smart_open import open as smart_open_file

        # Read content from cloud
        with smart_open_file(uri, "rb") as f:
            content = f.read()

        # Parse based on format
        import io

        if fmt == "csv":
            return pl.read_csv(io.BytesIO(content))
        elif fmt == "parquet":
            return pl.read_parquet(io.BytesIO(content))
        elif fmt == "json":
            return pl.read_json(io.BytesIO(content))
        else:
            return pl.read_csv(io.BytesIO(content))

    def _load_database(
        self, connection_string: str, analysis: Dict[str, Any]
    ) -> pl.DataFrame:
        """Load data from database."""
        import connectorx as cx

        query = analysis.get("query", "SELECT * FROM data LIMIT 100000")
        return pl.from_pandas(cx.read_sql(connection_string, query))

    def _load_api(self, url: str, analysis: Dict[str, Any]) -> pl.DataFrame:
        """Load data from REST API."""
        import httpx

        response = httpx.get(url, timeout=60)
        response.raise_for_status()

        data = response.json()

        # Handle different response structures
        if isinstance(data, list):
            return pl.DataFrame(data)
        elif isinstance(data, dict):
            # Look for common data keys
            for key in ["data", "results", "items", "records"]:
                if key in data and isinstance(data[key], list):
                    return pl.DataFrame(data[key])
            return pl.DataFrame([data])
        else:
            raise ValueError("Unable to parse API response as tabular data")

    def _get_data_stats(self, df: pl.DataFrame) -> Dict[str, Any]:
        """Get statistics about the loaded data."""
        stats = {
            "row_count": df.height,
            "column_count": df.width,
            "columns": [],
            "memory_bytes": df.estimated_size(),
            "null_counts": {},
            "dtypes": {},
        }

        for col in df.columns:
            col_info = {
                "name": col,
                "dtype": str(df[col].dtype),
                "null_count": df[col].null_count(),
                "null_percent": round(df[col].null_count() / df.height * 100, 2),
            }
            stats["columns"].append(col_info)
            stats["null_counts"][col] = col_info["null_count"]
            stats["dtypes"][col] = col_info["dtype"]

        return stats

    def _format_approval_message(
        self, stats: Dict[str, Any], analysis: Dict[str, Any]
    ) -> str:
        """Format approval message for human review."""
        cols_info = "\n".join(
            f"  - {c['name']}: {c['dtype']} ({c['null_percent']}% null)"
            for c in stats["columns"][:10]  # Show first 10 columns
        )
        if len(stats["columns"]) > 10:
            cols_info += f"\n  ... and {len(stats['columns']) - 10} more columns"

        return f"""
Data Loading Complete
=====================

Source: {analysis.get('source_type', 'unknown')}
Format: {analysis.get('format', 'unknown')}

Statistics:
- Rows: {stats['row_count']:,}
- Columns: {stats['column_count']}
- Memory: {stats['memory_bytes'] / 1024 / 1024:.2f} MB

Columns:
{cols_info}

Potential Issues:
{chr(10).join('- ' + i for i in analysis.get('potential_issues', [])) or '- None detected'}

Do you want to proceed with this data?
"""
