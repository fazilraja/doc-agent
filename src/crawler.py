import argparse
import asyncio
import os
import re
import logging
import sys
from typing import List, Tuple
from xml.etree import ElementTree
from dotenv import load_dotenv
import requests
from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CacheMode,
    CrawlerMonitor,
    CrawlerRunConfig,
    DisplayMode,
    MemoryAdaptiveDispatcher,
    RateLimiter,
)
from supabase import Client, create_client
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
from mirascope import llm, prompt_template
from pydantic import BaseModel, Field
from datetime import datetime, timezone
from urllib.parse import urlparse
from openai import AsyncOpenAI
import lilypad
import chamois
import random
load_dotenv()
lilypad.configure()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("crawler.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Log environment setup
logger.info("Setting up Supabase connection...")
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

if not url or not key:
    logger.error("SUPABASE_URL or SUPABASE_KEY environment variables not set")
    sys.exit(1)
    raise ValueError("Supabase credentials missing")

supabase: Client = create_client(url, key)
logger.info("Supabase client created")

# Get the absolute path of the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
logger.info(f"Output directory created at {OUTPUT_DIR}")

# Initialize OpenAI client
client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

@dataclass
class ProcessedChunk:
    url: str
    chunk_number: int
    title: str
    summary: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]
        
async def process_chunk(chunk: str, chunk_number: int, url: str) -> ProcessedChunk:
    """Process a single chunk of text."""
    logger.info(f"Processing chunk {chunk_number} for {url}")
    
    # Get title and summary
    logger.debug(f"Summarizing chunk {chunk_number} (size: {len(chunk)})")
    extracted = await summarize_chunk(chunk)
    logger.debug(f"Got title: '{extracted['title']}' and summary for chunk {chunk_number}")
    
    # Get embedding
    logger.debug(f"Getting embedding for chunk {chunk_number}")
    embedding = await get_embedding(chunk)
    logger.debug(f"Received embedding vector of length {len(embedding)}")
    
    # Parse URL to get domain for source name
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    
    # Extract the main domain name
    # Handle various patterns like www.example.com, docs.example.com, etc.
    domain_parts = domain.split('.')
    
    if len(domain_parts) >= 2:
        # For common TLDs like .com, .org, .io - use the second-to-last part
        if domain_parts[-1] in ['com', 'org', 'net', 'io', 'ai', 'dev']:
            main_domain = domain_parts[-2]
        # For country-specific TLDs like .co.uk - use the third-to-last part
        elif len(domain_parts) >= 3 and domain_parts[-2] in ['co', 'com', 'org', 'net']:
            main_domain = domain_parts[-3]
        else:
            main_domain = domain_parts[-2]
    else:
        # Fallback for unusual domains
        main_domain = domain
    
    # Clean up and format the source name
    source_name = main_domain.replace('-', '_') + "_docs"
    
    # Create metadata
    metadata = {
        "source": source_name,
        "chunk_size": len(chunk),
        "crawled_at": datetime.now(timezone.utc).isoformat(),
        "url_path": parsed_url.path
    }
    
    logger.info(f"Chunk {chunk_number} processed successfully for {url}")
    return ProcessedChunk(
        url=url,
        chunk_number=chunk_number,
        title=extracted['title'],
        summary=extracted['summary'],
        content=chunk,
        metadata=metadata,
        embedding=embedding
    )

async def insert_chunk(chunk: ProcessedChunk):
    """Insert a processed chunk into Supabase."""
    try:
        logger.info(f"Inserting chunk {chunk.chunk_number} for {chunk.url} into Supabase")
        
        data = {
            "url": chunk.url,
            "chunk_number": chunk.chunk_number,
            "title": chunk.title,
            "summary": chunk.summary,
            "content": chunk.content,
            "metadata": chunk.metadata,
            "embedding": chunk.embedding
        }
        
        # Log detailed info about the insertion attempt
        logger.debug(f"Inserting data: URL={chunk.url}, chunk={chunk.chunk_number}, " 
                    f"title={chunk.title[:30]}..., content_length={len(chunk.content)}, "
                    f"embedding_length={len(chunk.embedding)}")
        
        # Execute the insert and capture the response
        result = supabase.table("site_pages").insert(data).execute()
        
        # Log detailed response info
        logger.debug(f"Supabase response: {result}")
        logger.info(f"Successfully inserted chunk {chunk.chunk_number} for {chunk.url}")
        
        # Write to a success log file as a backup record
        with open("successful_inserts.txt", "a") as f:
            f.write(f"{datetime.now().isoformat()} - Inserted: {chunk.url} - Chunk: {chunk.chunk_number} - Title: {chunk.title[:50]}\n")
            
        return result
    except Exception as e:
        logger.error(f"Error inserting chunk {chunk.chunk_number} for {chunk.url}: {e}")
        sys.exit(1)
        
        # Write to a failure log file
        with open("failed_inserts.txt", "a") as f:
            f.write(f"{datetime.now().isoformat()} - Failed: {chunk.url} - Chunk: {chunk.chunk_number} - Error: {str(e)}\n")
            
        return None

    
def chunk_text(text: str, chunk_size: int = 1000) -> List[str]:
    """
    Chunk text into smaller pieces of a specified size. Respect code blocks and paragraphs.
    
    Args:
        text: The text to chunk
        chunk_size: The size of each chunk (default is 1000 characters)
        
    Returns:
        List[str]: List of text chunks, with code blocks preserved as single chunks
    """
    logger.info(f"Chunking text of size {len(text)} characters with chunk size {chunk_size}")
    chunks = []
    chunk_start = 0
    text_length = len(text)
    in_code_block = False
    
    while chunk_start < text_length:
        # calc position
        chunk_end = chunk_start + chunk_size
        
        # if at the end of the text, grab the remaining text
        if chunk_end >= text_length:
            chunks.append(text[chunk_start:].strip())
            break
        
        chunk_block = text[chunk_start:chunk_end]
        
        # Handle code blocks
        code_block_start = chunk_block.find("```")
        if code_block_start != -1:
            if not in_code_block:  # Found start of code block
                # Look for the end of this code block
                code_block_end = text.find("```", chunk_start + code_block_start + 3)
                if code_block_end != -1:
                    # Include the entire code block as one chunk
                    chunk = text[chunk_start:code_block_end + 3].strip()
                    if chunk:
                        chunks.append(chunk)
                    chunk_start = code_block_end + 3
                    continue
                else:
                    # Code block continues beyond what we can see
                    in_code_block = True
                    chunk_end = chunk_start + code_block_start
            else:  # Found end of code block
                in_code_block = False
                chunk_end = chunk_start + code_block_start + 3
        
        # If we're in a code block, continue to next iteration to find its end
        if in_code_block:
            chunk_end = min(chunk_end, text_length)
            chunks.append(text[chunk_start:chunk_end].strip())
            chunk_start = chunk_end
            continue
            
        # if no code block try to find a paragraph
        if '\n\n' in chunk_block:
            # find the end of the paragraph
            last_break = chunk_block.find('\n\n')
            if last_break > chunk_size * 0.3:  # break past 30% of chunk
                chunk_end = chunk_start + last_break
                
        # break at sentence if no code block or paragraph
        elif '. ' in chunk_block:
            last_period = chunk_block.rfind('. ')
            if last_period > chunk_size * 0.3:  # break past 30% of chunk
                chunk_end = chunk_start + last_period + 1

        # clean up the chunk
        chunk = text[chunk_start:chunk_end].strip()
        if chunk:
            chunks.append(chunk)
        
        # update the chunk start
        chunk_start = max(chunk_start + 1, chunk_end)
    
    logger.info(f"Text successfully chunked into {len(chunks)} chunks")
    return chunks
            
class ChunkMetadata(BaseModel):
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

@lilypad.generation()                
@llm.call(provider='google', model="gemini-2.0-flash-lite", response_model=ChunkMetadata)
@prompt_template(
    """
    SYSTEM: You are a documentation analyzer that creates clear, informative titles and summaries.
    USER: Extract a title and summary from this text chunk: {chunk}. Keep it short and concise.
    """
)
async def summarize_chunk(chunk: str) -> str: ...

async def get_embedding(text: str) -> List[float]:
    """
    Get embeddings for a text using OpenAI's text-embedding-3-small model.
    
    Args:
        text: The text to get embeddings for

    Returns:
        List[float]: The embeddings for the text
    """
    try: 
        logger.debug(f"Getting embedding for text of length {len(text)}")
        @chamois.embed("openai:text-embedding-3-small", dims=1536)
        async def split_text_async(text: str) -> list[str]:
            return [text]
        result = await split_text_async(text)
        logger.debug("Successfully retrieved embedding")
        return result[0].embedding
    except Exception as e:
        logger.error(f"Error getting embeddings: {e}")
        sys.exit(1)
        return [0] * 1536  # Return zero vector as fallback

            
def get_llms_urls(url: str) -> Tuple[List[str], bool]:
    """
    Fetches all URLs from the llms.txt file.
    
    Args:
        url: Base URL to fetch llms.txt from
        
    Returns:
        Tuple[List[str], bool]: List of URLs found in llms.txt and a boolean indicating success
    """
    logger.info(f"Attempting to fetch URLs from llms.txt at {url}")
    llms_url = url.rstrip('/') + "/llms.txt"
    
    try:
        logger.debug(f"Requesting {llms_url}")
        response = requests.get(llms_url)
        response.raise_for_status()
        
        logger.debug(f"Successfully fetched llms.txt, size: {len(response.text)} bytes")
        
        # Use regex to find all markdown links
        # This pattern matches [text](url) where url doesn't contain parentheses
        pattern = r'\[([^\]]+)\]\(([^)]+)\)'
        matches = re.findall(pattern, response.text)
        
        # Extract just the URLs from the matches
        urls = [url for _, url in matches]
        
        if urls:
            logger.info(f"Successfully parsed llms.txt and found {len(urls)} URLs")
            return urls, True
        else:
            logger.warning("No URLs found in llms.txt")
            return [], False
            
    except Exception as e:
        logger.error(f"Error fetching llms.txt: {e}")
        return [], False

def get_sitemap_urls(url: str) -> List[str]:
    """
    Fetches all URLs from the sitemap.
    
    Args:
        url: Base URL to fetch sitemap from
        
    Returns:
        List[str]: List of URLs found in the sitemap
    """
    logger.info(f"Attempting to fetch URLs for {url}")
    
    # First try llms.txt
    urls, success = get_llms_urls(url)
    if success:
        return urls
        
    # Fall back to sitemap.xml if llms.txt fails
    logger.info("Falling back to sitemap.xml...")
    sitemap_url = url.rstrip('/') + "/sitemap.xml"
                
    try:
        logger.debug(f"Requesting {sitemap_url}")
        response = requests.get(sitemap_url)
        response.raise_for_status()
        
        logger.debug(f"Successfully fetched sitemap.xml, size: {len(response.content)} bytes")
        
        # Parse the XML
        root = ElementTree.fromstring(response.content)
        
        # Extract all URLs from the sitemap
        # The namespace is usually defined in the root element
        namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        urls = [loc.text for loc in root.findall('.//ns:loc', namespace)]
        
        logger.info(f"Successfully parsed sitemap.xml and found {len(urls)} URLs")
        return urls
    except Exception as e:
        logger.error(f"Error fetching sitemap: {e}")
        return []

async def crawl_parallel(urls: List[str]):
    """
    Crawl URLs in parallel using Crawl4AI's built-in dispatcher.
    
    Args:
        urls: List of URLs to crawl
    """
    logger.info(f"\n=== Starting parallel crawling of {len(urls)} URLs ===")
    
    # Configure the browser
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=["--disable-dev-shm-usage", "--no-sandbox"],
    )
    
    # Configure the crawler
    crawl_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        stream=True  # Enable streaming for real-time processing
    )
    
    # Configure the dispatcher with rate limiting and monitoring
    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=95.0,  # Pause if memory exceeds 95%
        check_interval=1.0,
        max_session_permit=10,
        rate_limiter=RateLimiter(
            base_delay=(1.0, 2.0),  # Random delay between requests
            max_delay=30.0,
            max_retries=3
        ),
        monitor=CrawlerMonitor(
            max_visible_rows=15,
            display_mode=DisplayMode.DETAILED
        )
    )

    success_count = 0
    fail_count = 0

    logger.info("Initializing web crawler")
    async with AsyncWebCrawler(config=browser_config) as crawler:
        logger.info("Starting crawler task execution")
        async for result in await crawler.arun_many(
            urls=urls,
            config=crawl_config,
            dispatcher=dispatcher
        ):
            # Process results as they come in
            logger.info(f"Received crawl result for {result.url}")
            batch_output_dir = OUTPUT_DIR
            os.makedirs(batch_output_dir, exist_ok=True)
            
            # Create a safe filename from the URL
            safe_filename = "".join(c if c.isalnum() else "_" for c in result.url.split("://")[1])
            output_file = os.path.join(batch_output_dir, f"{safe_filename}.md")
            
            try:
                logger.debug(f"Saving crawl result to {output_file}")
                with open(output_file, "w", encoding='utf-8') as f:
                    f.write(f"# Crawl Result for {result.url}\n\n")
                    if result.success:
                        logger.info(f"Crawl succeeded for {result.url}, processing document")
                        f.write(result.markdown)  # Save the markdown for debugging
                        await process_and_store_document(result.url, result.markdown)
                        success_count += 1
                    else:
                        logger.error(f"Crawl failed for {result.url}: {result.error_message}")
                        sys.exit(1)
                        f.write(f"## Error\n\n```\n{result.error_message}\n```\n")
                        fail_count += 1
            except Exception as e:
                logger.error(f"Error saving results for {result.url}: {e}")
                sys.exit(1)
                fail_count += 1

            # Print dispatch metrics
            if result.dispatch_result:
                dr = result.dispatch_result
                metrics_info = (
                    f"\nMetrics for {result.url}:\n"
                    f"Memory Usage: {dr.memory_usage:.1f}MB\n"
                    f"Duration: {dr.end_time - dr.start_time}"
                )
                logger.info(metrics_info)

    summary = (
        f"\nCrawl Summary:\n"
        f"  - Successfully crawled: {success_count}\n"
        f"  - Failed: {fail_count}"
    )
    logger.info(summary)
    
    # Write summary to a file
    with open("crawl_summary.txt", "a") as f:
        f.write(f"\n--- Crawl run at {datetime.now().isoformat()} ---\n")
        f.write(f"URLs attempted: {len(urls)}\n")
        f.write(f"Success: {success_count}\n")
        f.write(f"Failed: {fail_count}\n\n")
    
async def process_and_store_document(url: str, markdown: str):
    """Process a document and store its chunks in parallel."""
    logger.info(f"Processing document from {url} (size: {len(markdown)} bytes)")
    
    # Split into chunks
    chunks = chunk_text(markdown)
    logger.info(f"Document split into {len(chunks)} chunks")
    
    # Process chunks in parallel
    logger.info(f"Processing {len(chunks)} chunks in parallel")
    tasks = [
        process_chunk(chunk, i, url) 
        for i, chunk in enumerate(chunks)
    ]
    processed_chunks = await asyncio.gather(*tasks)
    logger.info(f"Successfully processed {len(processed_chunks)} chunks")
    
    # Store chunks in parallel
    logger.info(f"Storing {len(processed_chunks)} chunks in Supabase")
    insert_tasks = [
        insert_chunk(chunk) 
        for chunk in processed_chunks
    ]
    insert_results = await asyncio.gather(*insert_tasks)
    
    # Count successful insertions
    successful_inserts = sum(1 for result in insert_results if result is not None)
    logger.info(f"Database insertion complete: {successful_inserts}/{len(processed_chunks)} chunks successfully inserted")
    
    # Record document processing in log file
    with open("documents_processed.txt", "a") as f:
        f.write(f"{datetime.now().isoformat()} - URL: {url} - Chunks: {len(chunks)} - Inserted: {successful_inserts}\n")


async def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Web crawler with sitemap support')
    parser.add_argument('--url', type=str, required=True,
                       help='Base URL to crawl (e.g., https://example.com)')
    
    args = parser.parse_args()
    logger.info(f"Starting crawler for URL: {args.url}")
    
    # Test Supabase connection before starting
    try:
        logger.info("Testing Supabase connection...")
        test_result = supabase.table("site_pages").select("*", count="exact").limit(0).execute()
        row_count = test_result.count if hasattr(test_result, 'count') else 0
        logger.info(f"Supabase connection test successful. Row count: {row_count}")
        
        # Use random chunk number to avoid conflicts
        random_chunk = random.randint(10000, 99999)
        
        # test a insert query
        result = supabase.table("site_pages").insert({
            "url": "https://example.com",
            "chunk_number": random_chunk,  # Use random chunk number
            "title": "Example",
            "summary": "Example summary",
            "content": "Example content",
            "metadata": {"test": "test"},  # Required field
        }).execute()    
    except Exception as e:
        logger.error(f"Supabase connection test failed: {e}")
        sys.exit(1)
        logger.error("Crawling will continue but database insertions may fail")
    
    urls = get_sitemap_urls(args.url)
    if urls:
        logger.info(f"Found {len(urls)} URLs to crawl")
        await crawl_parallel(urls)
    else:
        logger.error("No URLs found to crawl")    
        sys.exit(1)

if __name__ == "__main__":
    try:
        logger.info("Starting crawler script")
        asyncio.run(main())
        logger.info("Crawler script completed successfully")
    except Exception as e:
        logger.critical(f"Unhandled exception in main: {e}", exc_info=True)