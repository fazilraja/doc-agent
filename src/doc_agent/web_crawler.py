"""Web crawler module for the document crawler application."""

import re
import asyncio
from typing import List, Tuple, AsyncGenerator, Optional, Dict, Any
from xml.etree import ElementTree
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

from .models import CrawlResult
from .config import logger, CrawlerConfig

class CrawlerService:
    """Service for crawling websites and discovering URLs."""
    
    def __init__(self):
        """Initialize the crawler service."""
        self.browser_config = BrowserConfig(
            headless=True,
            verbose=False,
            extra_args=["--disable-dev-shm-usage", "--no-sandbox"],
        )
        
        self.crawl_config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            stream=True  # Enable streaming for real-time processing
        )
        
        self.dispatcher = MemoryAdaptiveDispatcher(
            memory_threshold_percent=CrawlerConfig.MEMORY_THRESHOLD_PERCENT,
            check_interval=1.0,
            max_session_permit=CrawlerConfig.MAX_CONCURRENT_REQUESTS,
            rate_limiter=RateLimiter(
                base_delay=(CrawlerConfig.REQUEST_DELAY_MIN, CrawlerConfig.REQUEST_DELAY_MAX),
                max_delay=30.0,
                max_retries=3
            ),
            monitor=CrawlerMonitor(
                max_visible_rows=15,
                display_mode=DisplayMode.DETAILED
            )
        )
    
    async def discover_urls(self, base_url: str) -> List[str]:
        """
        Discover URLs from a base URL using sitemap or llms.txt.
        
        Args:
            base_url: The base URL to discover URLs from
            
        Returns:
            List[str]: A list of discovered URLs
        """
        logger.info(f"Discovering URLs for {base_url}")
        
        # First try llms.txt
        urls, success = self._get_llms_urls(base_url)
        if success:
            logger.info(f"Found {len(urls)} URLs in llms.txt")
            return urls
            
        # Fall back to sitemap.xml
        logger.info("Falling back to sitemap.xml...")
        urls = self._get_sitemap_urls(base_url)
        logger.info(f"Found {len(urls)} URLs in sitemap.xml")
        return urls
    
    def _get_llms_urls(self, url: str) -> Tuple[List[str], bool]:
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
    
    def _get_sitemap_urls(self, url: str) -> List[str]:
        """
        Fetches all URLs from the sitemap.
        
        Args:
            url: Base URL to fetch sitemap from
            
        Returns:
            List[str]: List of URLs found in the sitemap
        """
        logger.info(f"Attempting to fetch URLs from sitemap.xml for {url}")
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
    
    async def crawl_url(self, url: str) -> CrawlResult:
        """
        Crawl a single URL and return the result.
        
        Args:
            url: The URL to crawl
            
        Returns:
            CrawlResult: The result of the crawl
        """
        logger.info(f"Crawling URL: {url}")
        
        async with AsyncWebCrawler(config=self.browser_config) as crawler:
            try:
                result = await crawler.arun(url=url, config=self.crawl_config)
                
                return CrawlResult(
                    url=url,
                    success=result.success,
                    markdown=result.markdown,
                    error_message=result.error_message if not result.success else "",
                    dispatch_result=result.dispatch_result
                )
            except Exception as e:
                logger.error(f"Error crawling {url}: {e}")
                return CrawlResult(
                    url=url,
                    success=False,
                    error_message=str(e)
                )
    
    async def crawl_urls(self, urls: List[str]) -> AsyncGenerator[CrawlResult, None]:
        """
        Crawl multiple URLs in parallel and yield results as they complete.
        
        Args:
            urls: The URLs to crawl
            
        Yields:
            CrawlResult: The results of the crawl as they complete
        """
        logger.info(f"Starting parallel crawling of {len(urls)} URLs")
        
        async with AsyncWebCrawler(config=self.browser_config) as crawler:
            async for result in await crawler.arun_many(
                urls=urls,
                config=self.crawl_config,
                dispatcher=self.dispatcher
            ):
                logger.info(f"Received crawl result for {result.url}")
                
                yield CrawlResult(
                    url=result.url,
                    success=result.success,
                    markdown=result.markdown if result.success else "",
                    error_message=result.error_message if not result.success else "",
                    dispatch_result=result.dispatch_result
                )
                
                # Log metrics
                if result.dispatch_result:
                    dr = result.dispatch_result
                    metrics_info = (
                        f"Metrics for {result.url}:\n"
                        f"Memory Usage: {dr.memory_usage:.1f}MB\n"
                        f"Duration: {dr.end_time - dr.start_time}"
                    )
                    logger.info(metrics_info) 