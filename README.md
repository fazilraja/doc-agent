# Documentation Agent

A powerful documentation crawler and processor that allows you to chat with documentation.

## Features

- Parallel web crawling with memory-adaptive dispatching
- Automatic sitemap detection and processing
- Rate limiting and retry mechanisms
- Real-time progress monitoring
- Markdown output generation

## Installation

This project uses `uv` for dependency management. First, make sure you have `uv` installed:

```bash
pip install uv
```

Then:

```bash
# Clone the repository
git clone https://github.com/fazilraja/doc-agent.git
cd doc-agent

# Create and activate virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies and create lock file
uv pip install .
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

## Development

For development, install dev dependencies:

```bash
uv pip install ".[dev]"
```

Run tests:
```bash
uv run pytest
```

Format and lint code:
```bash
# Format with black
uv run black src tests

# Sort imports
uv run isort src tests

# Lint with Ruff
uv run ruff check src tests

# Auto-fix Ruff violations
uv run ruff check --fix src tests
```

Type checking:
```bash
uv run mypy src
```

## Output

Results are saved in the `output` directory as markdown files, with:
- Content from successful crawls
- Error messages from failed attempts
- Performance metrics for each URL

## License

MIT License - see [LICENSE](LICENSE) file for details