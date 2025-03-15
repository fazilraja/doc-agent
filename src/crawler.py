import os
import sys
import argparse
import asyncio
import requests
from xml.etree import ElementTree
from typing import List
from crawl4ai import (
    AsyncWebCrawler, 
    BrowserConfig, 
    CrawlerRunConfig, 
    CacheMode,
    MemoryAdaptiveDispatcher,
    RateLimiter,
    CrawlerMonitor,
    DisplayMode
)

# Get the absolute path of the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

    print(f"\nSummary:")
    print(f"  - Successfully crawled: {success_count}")
    print(f"  - Failed: {fail_count}")

def get_sitemap_urls(url: str) -> List[str]:
    """
    Fetches all URLs from the sitemap.
    
    Args:
        url: Base URL to fetch sitemap from
        
    Returns:
        List[str]: List of URLs found in the sitemap
    """
    # Get the sitemap url from the url
    sitemap_url = url + "/sitemap.xml"
                
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