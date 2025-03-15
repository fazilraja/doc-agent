# Documentation Agent

A powerful documentation crawler and processor that allows you to chat with documentation.

## Features

- Parallel web crawling with memory-adaptive dispatching
- Automatic sitemap detection and processing
- Rate limiting and retry mechanisms
- Real-time progress monitoring
- Markdown output generation

## Installation

```bash
# Clone the repository
git clone https://github.com/fazilraja/doc-agent.git
cd doc-agent

# Install dependencies using uv
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

## Usage

```bash
python src/crawler.py --url https://example.com
```

The crawler will:
1. Fetch the sitemap from the provided URL
2. Crawl all discovered URLs in parallel
3. Save the results as markdown files in the `output` directory

## Configuration

The crawler uses several configuration options:
- Memory threshold: 70% (adjustable)
- Concurrent requests: 10 (adjustable)
- Rate limiting: 1-2 second delay between requests
- Retry attempts: 3 with exponential backoff

## Output

Results are saved in the `output` directory as markdown files, with:
- Content from successful crawls
- Error messages from failed attempts
- Performance metrics for each URL

## License

MIT License - see [LICENSE](LICENSE) file for details