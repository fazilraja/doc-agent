"""Data models for the document crawler application."""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum


@dataclass
class CrawlResult:
    """Result of crawling a URL."""
    url: str
    success: bool
    markdown: str = ""
    error_message: str = ""
    dispatch_result: Optional[Any] = None


@dataclass
class ProcessedChunk:
    """A processed chunk of text with metadata and embeddings."""
    url: str
    chunk_number: int
    title: str
    summary: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]


class ChunkMetadata(BaseModel):
    """Metadata for a chunk of text, including title and summary."""
    title: str = Field(
        ...,
        max_length=80,
        description="Document title if present, otherwise a descriptive topic-based title. Avoid generic titles like 'Introduction'."
    )
    summary: str = Field(
        ...,
        max_length=200,
        description="Concise overview focusing on key technical concepts and main points"
    )


class SourceType(str, Enum):
    """Type of content source."""
    MARKDOWN = "markdown"
    HTML = "html"
    PDF = "pdf"
    PLAIN_TEXT = "plain_text"


class DocumentMetadata(BaseModel):
    """Metadata for a document."""
    source: str
    url_path: str
    crawled_at: datetime
    source_type: SourceType = SourceType.MARKDOWN
    additional_info: Dict[str, Any] = Field(default_factory=dict) 