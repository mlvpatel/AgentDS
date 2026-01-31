"""
AgentDS Artifact Store.

Provides artifact management for pipeline outputs and intermediate results.

Author: Malav Patel
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, BinaryIO

from pydantic import BaseModel, Field

from agentds.core.config import Settings, get_settings
from agentds.core.logger import get_logger

logger = get_logger(__name__)


class ArtifactType(str, Enum):
    """Artifact type enumeration."""

    DATA = "data"
    MODEL = "model"
    REPORT = "report"
    CONFIG = "config"
    CODE = "code"
    VISUALIZATION = "visualization"
    CHECKPOINT = "checkpoint"
    OTHER = "other"


class Artifact(BaseModel):
    """Artifact metadata model."""

    id: str = Field(..., description="Artifact identifier")
    job_id: str = Field(..., description="Associated job ID")
    agent: str = Field(..., description="Agent that created the artifact")
    name: str = Field(..., description="Artifact name")
    type: ArtifactType = Field(..., description="Artifact type")

    # File information
    path: Path = Field(..., description="File path")
    size_bytes: int = Field(default=0)
    mime_type: str = Field(default="application/octet-stream")
    checksum: str | None = None

    # Metadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)

    # Description
    description: str | None = None


class ArtifactStore:
    """
    Artifact store for managing pipeline outputs.

    Supports local filesystem storage with optional cloud backends.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """
        Initialize artifact store.

        Args:
            settings: Application settings
        """
        self.settings = settings or get_settings()
        self._base_path = self.settings.output_dir
        self._base_path.mkdir(parents=True, exist_ok=True)
        self._metadata_file = self._base_path / ".artifacts.json"
        self._artifacts: dict[str, Artifact] = self._load_metadata()

    def _load_metadata(self) -> dict[str, Artifact]:
        """Load artifact metadata from disk."""
        if self._metadata_file.exists():
            try:
                with open(self._metadata_file) as f:
                    data = json.load(f)
                return {
                    k: Artifact(**v) for k, v in data.items()
                }
            except Exception as e:
                logger.warning("Failed to load artifact metadata", error=str(e))
        return {}

    def _save_metadata(self) -> None:
        """Save artifact metadata to disk."""
        try:
            data = {k: v.model_dump(mode="json") for k, v in self._artifacts.items()}
            with open(self._metadata_file, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.warning("Failed to save artifact metadata", error=str(e))

    def _get_artifact_path(self, job_id: str, agent: str, name: str) -> Path:
        """Generate artifact file path."""
        job_dir = self._base_path / job_id
        agent_dir = job_dir / agent
        agent_dir.mkdir(parents=True, exist_ok=True)
        return agent_dir / name

    def _calculate_checksum(self, path: Path) -> str:
        """Calculate MD5 checksum of file."""
        import hashlib

        md5 = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                md5.update(chunk)
        return md5.hexdigest()

    def _get_mime_type(self, path: Path) -> str:
        """Determine MIME type from file extension."""
        mime_types = {
            ".csv": "text/csv",
            ".json": "application/json",
            ".parquet": "application/octet-stream",
            ".pkl": "application/octet-stream",
            ".joblib": "application/octet-stream",
            ".html": "text/html",
            ".pdf": "application/pdf",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".py": "text/x-python",
            ".yaml": "text/yaml",
            ".yml": "text/yaml",
            ".txt": "text/plain",
            ".md": "text/markdown",
        }
        return mime_types.get(path.suffix.lower(), "application/octet-stream")

    def save(
        self,
        job_id: str,
        agent: str,
        name: str,
        data: bytes | str | BinaryIO | Path,
        artifact_type: ArtifactType = ArtifactType.OTHER,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        description: str | None = None,
    ) -> Artifact:
        """
        Save an artifact.

        Args:
            job_id: Job identifier
            agent: Agent name
            name: Artifact name (filename)
            data: Artifact data (bytes, string, file-like object, or path)
            artifact_type: Type of artifact
            metadata: Additional metadata
            tags: Tags for categorization
            description: Human-readable description

        Returns:
            Created Artifact instance
        """
        path = self._get_artifact_path(job_id, agent, name)

        # Write data to file
        if isinstance(data, bytes):
            with open(path, "wb") as f:
                f.write(data)
        elif isinstance(data, str):
            with open(path, "w") as f:
                f.write(data)
        elif isinstance(data, Path):
            shutil.copy2(data, path)
        else:
            # File-like object
            with open(path, "wb") as f:
                shutil.copyfileobj(data, f)

        # Create artifact metadata
        artifact_id = f"{job_id}/{agent}/{name}"
        artifact = Artifact(
            id=artifact_id,
            job_id=job_id,
            agent=agent,
            name=name,
            type=artifact_type,
            path=path,
            size_bytes=path.stat().st_size,
            mime_type=self._get_mime_type(path),
            checksum=self._calculate_checksum(path),
            metadata=metadata or {},
            tags=tags or [],
            description=description,
        )

        self._artifacts[artifact_id] = artifact
        self._save_metadata()

        logger.info(
            "Artifact saved",
            artifact_id=artifact_id,
            type=artifact_type,
            size=artifact.size_bytes,
        )

        return artifact

    def get(self, artifact_id: str) -> Artifact | None:
        """Get artifact by ID."""
        return self._artifacts.get(artifact_id)

    def load(self, artifact_id: str) -> bytes | None:
        """Load artifact data."""
        artifact = self.get(artifact_id)
        if artifact and artifact.path.exists():
            with open(artifact.path, "rb") as f:
                return f.read()
        return None

    def load_text(self, artifact_id: str) -> str | None:
        """Load artifact as text."""
        data = self.load(artifact_id)
        if data:
            return data.decode("utf-8")
        return None

    def delete(self, artifact_id: str) -> bool:
        """Delete an artifact."""
        artifact = self._artifacts.pop(artifact_id, None)
        if artifact:
            if artifact.path.exists():
                artifact.path.unlink()
            self._save_metadata()
            logger.info("Artifact deleted", artifact_id=artifact_id)
            return True
        return False

    def list_artifacts(
        self,
        job_id: str | None = None,
        agent: str | None = None,
        artifact_type: ArtifactType | None = None,
        tags: list[str] | None = None,
    ) -> list[Artifact]:
        """
        List artifacts with optional filters.

        Args:
            job_id: Filter by job ID
            agent: Filter by agent name
            artifact_type: Filter by artifact type
            tags: Filter by tags (any match)

        Returns:
            List of matching artifacts
        """
        results = []
        for artifact in self._artifacts.values():
            if job_id and artifact.job_id != job_id:
                continue
            if agent and artifact.agent != agent:
                continue
            if artifact_type and artifact.type != artifact_type:
                continue
            if tags and not any(t in artifact.tags for t in tags):
                continue
            results.append(artifact)
        return results

    def get_job_artifacts(self, job_id: str) -> list[Artifact]:
        """Get all artifacts for a job."""
        return self.list_artifacts(job_id=job_id)

    def get_artifact_path(self, artifact_id: str) -> Path | None:
        """Get file path for an artifact."""
        artifact = self.get(artifact_id)
        if artifact:
            return artifact.path
        return None

    def cleanup_job(self, job_id: str) -> int:
        """
        Clean up all artifacts for a job.

        Args:
            job_id: Job identifier

        Returns:
            Number of artifacts deleted
        """
        artifacts = self.get_job_artifacts(job_id)
        count = 0
        for artifact in artifacts:
            if self.delete(artifact.id):
                count += 1

        # Remove job directory
        job_dir = self._base_path / job_id
        if job_dir.exists():
            shutil.rmtree(job_dir)

        logger.info("Job artifacts cleaned up", job_id=job_id, count=count)
        return count

    def get_total_size(self, job_id: str | None = None) -> int:
        """
        Get total size of artifacts.

        Args:
            job_id: Optional job ID to filter

        Returns:
            Total size in bytes
        """
        artifacts = self.list_artifacts(job_id=job_id)
        return sum(a.size_bytes for a in artifacts)
