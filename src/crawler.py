import argparse
import asyncio
import os
import re
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
from google import genai
from datetime import datetime, timezone
from urllib.parse import urlparse

load_dotenv()

client = genai.Client(api_key="GEMINI_API_KEY")
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

# Get the absolute path of the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    # Get title and summary
    extracted = await summarize_chunk(chunk)
    
    # Get embedding
    embedding = await get_embedding(chunk)
    
    # Create metadata
    metadata = {
        "source": "pydantic_ai_docs",
        "chunk_size": len(chunk),
        "crawled_at": datetime.now(timezone.utc).isoformat(),
        "url_path": urlparse(url).path
    }
    
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
        data = {
            "url": chunk.url,
            "chunk_number": chunk.chunk_number,
            "title": chunk.title,
            "summary": chunk.summary,
            "content": chunk.content,
            "metadata": chunk.metadata,
            "embedding": chunk.embedding
        }
        
        result = supabase.table("site_pages").insert(data).execute()
        print(f"Inserted chunk {chunk.chunk_number} for {chunk.url}")
        return result
    except Exception as e:
        print(f"Error inserting chunk: {e}")
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
                
@llm.call(provider='google', model="gemini-2.0-flash-lite", response_model=ChunkMetadata)
@prompt_template(
    """
    SYSTEM: You are a documentation analyzer that creates clear, informative titles and summaries.
    USER: Extract a title and summary from this text chunk: {chunk}
    """
)
async def summarize_chunk(chunk: str) -> str: ...

async def get_embedding(text: str) -> List[float]:
    """
    Get embeddings for a text using Google's embedding API.
    
    Args:
        text: The text to get embeddings for

    Returns:
        List[float]: The embeddings for the text
    """
    try: 
        result = client.models.embed_content(
            model="gemini-embedding-exp-03-07",
            contents=text,
            task_type="retrieval_document")
        return result.embeddings[1]
    except Exception as e:
        print(f"Error getting embeddings: {e}")
        return [0] * 3072

            
def get_llms_urls(url: str) -> Tuple[List[str], bool]:
    """
    Fetches all URLs from the llms.txt file.
    
    Args:
        url: Base URL to fetch llms.txt from
        
    Returns:
        Tuple[List[str], bool]: List of URLs found in llms.txt and a boolean indicating success
    """
    llms_url = url.rstrip('/') + "/llms.txt"
    
    try:
        response = requests.get(llms_url)
        response.raise_for_status()
        
        # Use regex to find all markdown links
        # This pattern matches [text](url) where url doesn't contain parentheses
        pattern = r'\[([^\]]+)\]\(([^)]+)\)'
        matches = re.findall(pattern, response.text)
        
        # Extract just the URLs from the matches
        urls = [url for _, url in matches]
        
        if urls:
            print(f"Successfully parsed llms.txt and found {len(urls)} URLs")
            return urls, True
        else:
            print("No URLs found in llms.txt")
            return [], False
            
    except Exception as e:
        print(f"Error fetching llms.txt: {e}")
        return [], False

def get_sitemap_urls(url: str) -> List[str]:
    """
    Fetches all URLs from the sitemap.
    
    Args:
        url: Base URL to fetch sitemap from
        
    Returns:
        List[str]: List of URLs found in the sitemap
    """
    # First try llms.txt
    urls, success = get_llms_urls(url)
    if success:
        return urls
        
    # Fall back to sitemap.xml if llms.txt fails
    print("Falling back to sitemap.xml...")
    sitemap_url = url.rstrip('/') + "/sitemap.xml"
                
    try:
        response = requests.get(sitemap_url)
        response.raise_for_status()
        
        # Parse the XML
        root = ElementTree.fromstring(response.content)
        
        # Extract all URLs from the sitemap
        # The namespace is usually defined in the root element
        namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        urls = [loc.text for loc in root.findall('.//ns:loc', namespace)]
        
        return urls
    except Exception as e:
        print(f"Error fetching sitemap: {e}")
        return []

async def crawl_parallel(urls: List[str]):
    """
    Crawl URLs in parallel using Crawl4AI's built-in dispatcher.
    
    Args:
        urls: List of URLs to crawl
    """
    print("\n=== Parallel Crawling with Memory-Adaptive Dispatcher ===")
    
    # Configure the browser
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
    )
    
    # Configure the crawler
    crawl_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        stream=True  # Enable streaming for real-time processing
    )
    
    # Configure the dispatcher with rate limiting and monitoring
    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=70.0,  # Pause if memory exceeds 70%
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

    async with AsyncWebCrawler(config=browser_config) as crawler:
        async for result in await crawler.arun_many(
            urls=urls,
            config=crawl_config,
            dispatcher=dispatcher
        ):
            # Process results as they come in
            batch_output_dir = OUTPUT_DIR
            os.makedirs(batch_output_dir, exist_ok=True)
            
            # Create a safe filename from the URL
            safe_filename = "".join(c if c.isalnum() else "_" for c in result.url.split("://")[1])
            output_file = os.path.join(batch_output_dir, f"{safe_filename}.md")
            
            try:
                with open(output_file, "w", encoding='utf-8') as f:
                    f.write(f"# Crawl Result for {result.url}\n\n")
                    if result.success:
                        f.write(f"## Content\n\n{result.markdown}\n")
                        success_count += 1
                    else:
                        f.write(f"## Error\n\n```\n{result.error_message}\n```\n")
                        fail_count += 1
            except Exception as e:
                print(f"Error saving results for {result.url}: {e}")
                fail_count += 1

            # Print dispatch metrics
            if result.dispatch_result:
                dr = result.dispatch_result
                print(f"\nMetrics for {result.url}:")
                print(f"Memory Usage: {dr.memory_usage:.1f}MB")
                print(f"Duration: {dr.end_time - dr.start_time}")

    print("\nSummary:")
    print(f"  - Successfully crawled: {success_count}")
    print(f"  - Failed: {fail_count}")

async def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Web crawler with sitemap support')
    parser.add_argument('--url', type=str, required=True,
                       help='Base URL to crawl (e.g., https://example.com)')
    
    args = parser.parse_args()
    
    urls = get_sitemap_urls(args.url)
    if urls:
        print(f"Found {len(urls)} URLs to crawl")
        await crawl_parallel(urls)
    else:
        print("No URLs found to crawl")    

if __name__ == "__main__":
    asyncio.run(main())