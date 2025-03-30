"""Text processor module for chunking and summarizing text."""

from typing import List
from urllib.parse import urlparse
from datetime import datetime, timezone
from mirascope import llm, prompt_template

from .models import ProcessedChunk, ChunkMetadata, SourceType, DocumentMetadata
from .config import logger, CrawlerConfig
from .embedding_service import EmbeddingService

class TextProcessor:
    """Service for processing text documents."""
    
    def __init__(self, embedding_service: EmbeddingService):
        """
        Initialize the text processor.
        
        Args:
            embedding_service: The embedding service to use for generating embeddings
        """
        self.embedding_service = embedding_service
    
    def chunk_text(self, text: str, chunk_size: int = CrawlerConfig.CHUNK_SIZE) -> List[str]:
        """
        Chunk text into smaller pieces of a specified size. Respect code blocks and paragraphs.
        
        Args:
            text: The text to chunk
            chunk_size: The size of each chunk
            
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
                last_break = chunk_block.rfind('\n\n')
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
    
    def extract_metadata(self, url: str) -> DocumentMetadata:
        """
        Extract metadata from a URL.
        
        Args:
            url: The URL to extract metadata from
            
        Returns:
            DocumentMetadata: The extracted metadata
        """
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
        
        return DocumentMetadata(
            source=source_name,
            url_path=parsed_url.path,
            crawled_at=datetime.now(timezone.utc),
            source_type=SourceType.MARKDOWN
        )
        
    @llm.call(provider='openai', model=CrawlerConfig.SUMMARY_MODEL, response_model=ChunkMetadata)
    @prompt_template(
        """
        SYSTEM: You are an AI specializing in extracting precise, concise titles and summaries from documentation chunks. Your PRIMARY RESPONSIBILITY is to NEVER exceed character limits.

        STRICT CHARACTER LIMITS:
        - TITLE: MAXIMUM 80 CHARACTERS (no exceptions)
        - SUMMARY: MAXIMUM 200 CHARACTERS (no exceptions)

        If your response exceeds these limits, it will be rejected. Count characters carefully.

        Content guidelines:
        - For TITLE: If this is the start of a document, extract its actual title. For a middle chunk, create a specific, descriptive title related to the content.
        - For SUMMARY: Create a concise summary focusing on key technical concepts and main points.
        - Avoid generic titles like "Introduction" or "Overview" unless that's the actual document title.
        - Prioritize technical accuracy and informativeness within the character constraints.
        
        USER: URL: {url}\n\nChunk: {chunk}
        """
    )
    async def summarize_chunk(self, chunk: str, url: str) -> ChunkMetadata:
        """This will be implemented by the LLM decorator."""
        pass  # This method is implemented by the decorator
    
    async def process_chunk(self, chunk: str, chunk_number: int, url: str) -> ProcessedChunk:
        """
        Process a single chunk of text.
        
        Args:
            chunk: The chunk to process
            chunk_number: The chunk number
            url: The URL the chunk is from
            
        Returns:
            ProcessedChunk: The processed chunk with metadata and embeddings
        """
        logger.info(f"Processing chunk {chunk_number} for {url}")
        
        # Get title and summary
        logger.debug(f"Summarizing chunk {chunk_number} (size: {len(chunk)})")
        try:
            extracted = await self.summarize_chunk(chunk, url)
            logger.debug(f"Got title: '{extracted.title}' and summary for chunk {chunk_number}")
        except Exception as e:
            logger.error(f"Error summarizing chunk {chunk_number}: {e}")
            # Provide default values in case of error
            extracted = ChunkMetadata(
                title=f"Chunk {chunk_number} from {urlparse(url).netloc}",
                summary=f"Content from {url}"
            )
            
        # Get embedding
        logger.debug(f"Getting embedding for chunk {chunk_number}")
        embedding = await self.embedding_service.get_embedding(chunk)
        logger.debug(f"Received embedding vector of length {len(embedding)}")
        
        # Get metadata
        doc_metadata = self.extract_metadata(url)
        
        # Create metadata dictionary
        metadata = {
            "source": doc_metadata.source,
            "chunk_size": len(chunk),
            "crawled_at": doc_metadata.crawled_at.isoformat(),
            "url_path": doc_metadata.url_path,
            "source_type": doc_metadata.source_type
        }
        
        logger.info(f"Chunk {chunk_number} processed successfully for {url}")
        return ProcessedChunk(
            url=url,
            chunk_number=chunk_number,
            title=extracted.title,
            summary=extracted.summary,
            content=chunk,
            metadata=metadata,
            embedding=embedding
        )
        
    async def process_document(self, url: str, text: str) -> List[ProcessedChunk]:
        """
        Process a document by chunking it and processing each chunk.
        
        Args:
            url: The URL the document is from
            text: The document text
            
        Returns:
            List[ProcessedChunk]: The processed chunks
        """
        logger.info(f"Processing document from {url} (size: {len(text)} bytes)")
        
        # Split into chunks
        chunks = self.chunk_text(text)
        logger.info(f"Document split into {len(chunks)} chunks")
        
        # Process each chunk
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            processed_chunk = await self.process_chunk(chunk, i, url)
            processed_chunks.append(processed_chunk)
            
        logger.info(f"Document processing complete: {len(processed_chunks)} chunks processed")
        return processed_chunks 