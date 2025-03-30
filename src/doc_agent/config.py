"""Configuration module for the document crawler application."""

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional
from supabase import create_client, Client

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Logging configuration
def setup_logging(log_file: Optional[str] = "crawler.log") -> logging.Logger:
    """Set up logging for the application."""
    logger = logging.getLogger("doc_agent")
    logger.setLevel(logging.INFO)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger

# Create logger instance
logger = setup_logging()

# Supabase configuration
def get_supabase_client() -> Client:
    """Get Supabase client."""
    url: str = os.environ.get("SUPABASE_URL", "")
    key: str = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
    
    if not url or not key:
        logger.error("SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY environment variables not set")
        sys.exit(1)
    
    return create_client(url, key)

# OpenAI configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY environment variable not set")

# Crawler configuration
class CrawlerConfig:
    """Configuration for the crawler."""
    CHUNK_SIZE = 1000
    EMBEDDING_MODEL = "text-embedding-3-small"
    SUMMARY_MODEL = "gpt-4o-mini"
    MAX_CONCURRENT_REQUESTS = 10
    REQUEST_DELAY_MIN = 1.0
    REQUEST_DELAY_MAX = 2.0
    MEMORY_THRESHOLD_PERCENT = 95.0 