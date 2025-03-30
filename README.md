# Document Crawler Agent

A modular document crawler designed to index documentation from websites, generate embeddings, and store content in a database for future use in documentation assistants.

## Features

- **URL Discovery**: Find URLs to crawl from sitemaps or custom formats
- **Web Crawling**: Crawl web pages to extract Markdown content
- **Text Processing**: Chunk text intelligently and generate summaries
- **Embedding Generation**: Create vector embeddings for document chunks
- **Data Storage**: Store processed documents in a Supabase database

## Architecture

The system consists of several modular components:

- `web_crawler.py`: URL discovery and web crawling
- `text_processor.py`: Text chunking and summarization
- `embedding_service.py`: Vector embedding generation
- `data_store.py`: Database operations
- `models.py`: Data models
- `config.py`: Configuration management

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/doc-agent.git
   cd doc-agent
   ```

2. Install dependencies:
   ```
   pip install -e .
   ```

3. Create a `.env` file with the following environment variables:
   ```
   OPENAI_API_KEY=your_openai_api_key
   SUPABASE_URL=your_supabase_url
   SUPABASE_SERVICE_ROLE_KEY=your_supabase_key
   ```

## Usage

```bash
python src/main.py --url https://example.com
```

The crawler will:
1. Discover URLs to crawl from the sitemap or llms.txt
2. Crawl each URL to extract content
3. Process the content by chunking and summarizing
4. Generate embeddings for each chunk
5. Store the chunks in the database

## Database Schema

The system requires a Supabase database with a `site_pages` table that has the following schema:

- `url` (text): The URL of the page
- `chunk_number` (integer): The chunk number
- `title` (text): The chunk title
- `summary` (text): A summary of the chunk
- `content` (text): The chunk content
- `metadata` (json): Additional metadata
- `embedding` (vector): The embedding vector

## Development

- Run tests: `pytest tests/`
- Format code: `black src/`
- Check types: `mypy src/`

## License

MIT