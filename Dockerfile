FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY setup.py /app/
COPY README.md /app/

# Install dependencies - make sure it includes both OpenAI and Google AI
RUN pip install --no-cache-dir -e .

# Copy application code
COPY src/ /app/src/

# Create output directory
RUN mkdir -p /app/output

# Create volume for outputs and logs
VOLUME ["/app/output"]

# Command to run when the container starts
# This can be overridden with docker run command
ENTRYPOINT ["python", "src/main.py"]

# Default arguments (can be overridden)
CMD ["--url", "https://example.com"] 