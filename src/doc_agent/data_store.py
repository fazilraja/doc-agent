"""Data store module for storing processed documents."""

import random
from typing import List, Dict, Any, Optional
from datetime import datetime
from supabase import Client

from .models import ProcessedChunk
from .config import logger, get_supabase_client

class DataStore:
    """Service for storing processed data in a database."""
    
    def __init__(self, client: Optional[Client] = None):
        """
        Initialize the data store.
        
        Args:
            client: The Supabase client to use. If None, creates a new client.
        """
        self.client = client or get_supabase_client()
        self.table_name = "site_pages"
        logger.info("Data store initialized")
    
    async def test_connection(self) -> bool:
        """
        Test the database connection.
        
        Returns:
            bool: True if the connection is successful, False otherwise
        """
        try:
            logger.info("Testing database connection...")
            
            # Query to get row count
            test_result = self.client.table(self.table_name).select("*", count="exact").limit(0).execute()
            row_count = test_result.count if hasattr(test_result, 'count') else 0
            logger.info(f"Database connection test successful. Row count: {row_count}")
            
            # Test insert with a random entry
            random_chunk = random.randint(10000, 99999)
            
            test_insert = self.client.table(self.table_name).insert({
                "url": "https://test.example.com",
                "chunk_number": random_chunk,
                "title": "Test Connection",
                "summary": "This is a test entry to verify database connection",
                "content": "Test content",
                "metadata": {"test": "test", "timestamp": datetime.now().isoformat()},
            }).execute()
            
            logger.info(f"Test insert successful: {test_insert}")
            return True
            
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False
    
    async def insert_chunk(self, chunk: ProcessedChunk) -> Dict[str, Any]:
        """
        Insert a processed chunk into the database.
        
        Args:
            chunk: The processed chunk to insert
            
        Returns:
            Dict[str, Any]: The result of the insertion operation
        """
        try:
            logger.info(f"Inserting chunk {chunk.chunk_number} for {chunk.url} into database")
            
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
            result = self.client.table(self.table_name).insert(data).execute()
            
            # Log detailed response info
            logger.debug(f"Database response: {result}")
            logger.info(f"Successfully inserted chunk {chunk.chunk_number} for {chunk.url}")
            
            # Write to a success log file as a backup record
            with open("successful_inserts.txt", "a") as f:
                f.write(f"{datetime.now().isoformat()} - Inserted: {chunk.url} - Chunk: {chunk.chunk_number} - Title: {chunk.title[:50]}\n")
                
            return result.data[0] if result.data else {}
            
        except Exception as e:
            logger.error(f"Error inserting chunk {chunk.chunk_number} for {chunk.url}: {e}")
            
            # Write to a failure log file
            with open("failed_inserts.txt", "a") as f:
                f.write(f"{datetime.now().isoformat()} - Failed: {chunk.url} - Chunk: {chunk.chunk_number} - Error: {str(e)}\n")
                
            return {}
    
    async def insert_chunks(self, chunks: List[ProcessedChunk]) -> List[Dict[str, Any]]:
        """
        Insert multiple chunks into the database.
        
        Args:
            chunks: The processed chunks to insert
            
        Returns:
            List[Dict[str, Any]]: The results of the insertion operations
        """
        logger.info(f"Inserting {len(chunks)} chunks into database")
        
        results = []
        for chunk in chunks:
            result = await self.insert_chunk(chunk)
            results.append(result)
            
        # Count successful insertions
        successful = sum(1 for r in results if r)
        logger.info(f"Database insertion complete: {successful}/{len(chunks)} chunks successfully inserted")
        
        return results
    
    async def get_chunks_by_url(self, url: str) -> List[Dict[str, Any]]:
        """
        Get all chunks for a URL.
        
        Args:
            url: The URL to get chunks for
            
        Returns:
            List[Dict[str, Any]]: The chunks for the URL
        """
        try:
            logger.info(f"Getting chunks for URL {url}")
            
            result = self.client.table(self.table_name).select("*").eq("url", url).execute()
            
            chunks = result.data
            logger.info(f"Found {len(chunks)} chunks for URL {url}")
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error getting chunks for URL {url}: {e}")
            return [] 