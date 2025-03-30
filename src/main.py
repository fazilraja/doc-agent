#!/usr/bin/env python3
"""Main entry point for the document crawler application."""

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

from doc_agent import (
    logger,
    CrawlerService,
    TextProcessor,
    EmbeddingService,
    DataStore,
    CrawlResult
)

OUTPUT_DIR = Path(__file__).parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

async def process_crawl_result(
    result: CrawlResult,
    text_processor: TextProcessor,
    data_store: DataStore
) -> bool:
    """
    Process a crawl result by chunking, embedding, and storing it.
    
    Args:
        result: The crawl result to process
        text_processor: The text processor to use
        data_store: The data store to use
        
    Returns:
        bool: True if processing was successful, False otherwise
    """
    if not result.success:
        logger.error(f"Crawl failed for {result.url}: {result.error_message}")
        return False
        
    # Save the markdown for debugging purposes
    safe_filename = "".join(c if c.isalnum() else "_" for c in result.url.split("://")[1])
    output_file = OUTPUT_DIR / f"{safe_filename}.md"
    
    try:
        with open(output_file, "w", encoding='utf-8') as f:
            f.write(f"# Crawl Result for {result.url}\n\n")
            f.write(result.markdown)
    except Exception as e:
        logger.error(f"Error saving crawl result to file: {e}")
    
    try:
        # Process the document
        logger.info(f"Processing document from {result.url}")
        processed_chunks = await text_processor.process_document(result.url, result.markdown)
        
        # Store the chunks
        logger.info(f"Storing {len(processed_chunks)} chunks for {result.url}")
        insert_results = await data_store.insert_chunks(processed_chunks)
        
        # Count successful insertions
        successful = sum(1 for r in insert_results if r)
        logger.info(f"Successfully stored {successful}/{len(processed_chunks)} chunks for {result.url}")
        
        # Record document processing in log file
        with open("documents_processed.txt", "a") as f:
            f.write(f"{result.url} - Chunks: {len(processed_chunks)} - Inserted: {successful}\n")
            
        return successful > 0
        
    except Exception as e:
        logger.error(f"Error processing document from {result.url}: {e}")
        return False

async def main():
    """Main entry point for the application."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Document crawler with sitemap support')
    parser.add_argument('--url', type=str, required=True,
                       help='Base URL to crawl (e.g., https://example.com)')
    
    args = parser.parse_args()
    logger.info(f"Starting crawler for URL: {args.url}")
    
    # Create services
    embedding_service = EmbeddingService()
    text_processor = TextProcessor(embedding_service)
    crawler_service = CrawlerService()
    data_store = DataStore()
    
    # Test database connection
    if not await data_store.test_connection():
        logger.error("Database connection test failed, exiting")
        sys.exit(1)
    
    # Discover URLs
    urls = await crawler_service.discover_urls(args.url)
    if not urls:
        logger.error("No URLs found to crawl")
        sys.exit(1)
    
    logger.info(f"Found {len(urls)} URLs to crawl")
    
    # Crawl URLs and process results
    success_count = 0
    fail_count = 0
    
    async for result in crawler_service.crawl_urls(urls):
        if await process_crawl_result(result, text_processor, data_store):
            success_count += 1
        else:
            fail_count += 1
            
        # Print dispatch metrics if available
        if result.dispatch_result:
            dr = result.dispatch_result
            metrics_info = (
                f"\nMetrics for {result.url}:\n"
                f"Memory Usage: {dr.memory_usage:.1f}MB\n"
                f"Duration: {dr.end_time - dr.start_time}"
            )
            logger.info(metrics_info)
    
    # Print summary
    summary = (
        f"\nCrawl Summary:\n"
        f"  - Successfully crawled and processed: {success_count}\n"
        f"  - Failed: {fail_count}"
    )
    logger.info(summary)
    
    # Write summary to a file
    with open(OUTPUT_DIR / "crawl_summary.txt", "a") as f:
        f.write(f"--- Crawl run for {args.url} ---\n")
        f.write(f"URLs attempted: {len(urls)}\n")
        f.write(f"Success: {success_count}\n")
        f.write(f"Failed: {fail_count}\n\n")

if __name__ == "__main__":
    try:
        logger.info("Starting document crawler")
        asyncio.run(main())
        logger.info("Document crawler completed successfully")
    except Exception as e:
        logger.critical(f"Unhandled exception in main: {e}", exc_info=True)
        sys.exit(1) 