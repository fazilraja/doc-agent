"""Embedding service for generating embeddings from text."""

from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI

from .config import logger, OPENAI_API_KEY, CrawlerConfig

class EmbeddingService:
    """Service for generating embeddings from text."""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize the embedding service.
        
        Args:
            api_key: OpenAI API key. If None, uses the key from the environment
            model: The embedding model to use. If None, uses the model from config
        """
        self.api_key = api_key or OPENAI_API_KEY
        if not self.api_key:
            logger.error("No OpenAI API key provided")
            raise ValueError("OpenAI API key is required")
            
        self.client = AsyncOpenAI(api_key=self.api_key)
        self.model = model or CrawlerConfig.EMBEDDING_MODEL
        
        # Size of embedding vector (for fallbacks)
        self.embedding_size = 1536  # Default for text-embedding-3-small
        
        logger.info(f"Embedding service initialized with model {self.model}")
    
    async def get_embedding(self, text: str) -> List[float]:
        """
        Get embeddings for a text using OpenAI's embedding model.
        
        Args:
            text: The text to get embeddings for

        Returns:
            List[float]: The embeddings for the text
        """
        try:
            logger.debug(f"Getting embedding for text of length {len(text)}")
            
            # Truncate text if too long (OpenAI has token limits)
            # A rough estimate is that 1 token is about 4 characters for English text
            max_chars = 8000  # Safe limit for most embedding models
            if len(text) > max_chars:
                logger.warning(f"Text too long ({len(text)} chars), truncating to {max_chars} chars")
                text = text[:max_chars]
                
            response = await self.client.embeddings.create(
                input=text,
                model=self.model
            )
            
            embedding = response.data[0].embedding
            self.embedding_size = len(embedding)  # Update size for fallbacks
            
            logger.debug(f"Successfully generated embedding of dimension {len(embedding)}")
            return embedding
            
        except Exception as e:
            logger.error(f"Error getting embeddings: {e}")
            # Return a zero vector as fallback
            logger.warning(f"Returning zero vector of dimension {self.embedding_size} as fallback")
            return [0] * self.embedding_size
    
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for multiple texts in parallel.
        
        Args:
            texts: The texts to get embeddings for

        Returns:
            List[List[float]]: The embeddings for the texts
        """
        try:
            logger.debug(f"Getting embeddings for {len(texts)} texts")
            
            # Truncate texts if too long
            max_chars = 8000
            processed_texts = []
            for text in texts:
                if len(text) > max_chars:
                    logger.warning(f"Text too long ({len(text)} chars), truncating to {max_chars} chars")
                    processed_texts.append(text[:max_chars])
                else:
                    processed_texts.append(text)
            
            response = await self.client.embeddings.create(
                input=processed_texts,
                model=self.model
            )
            
            embeddings = [item.embedding for item in response.data]
            
            # Update embedding size if necessary
            if embeddings:
                self.embedding_size = len(embeddings[0])
                
            logger.debug(f"Successfully generated {len(embeddings)} embeddings")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error getting batch embeddings: {e}")
            # Return zero vectors as fallback
            logger.warning(f"Returning zero vectors of dimension {self.embedding_size} as fallback")
            return [[0] * self.embedding_size for _ in range(len(texts))] 