"""Setup script for doc-agent package."""

from setuptools import setup, find_packages

setup(
    name="doc-agent",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "crawl4ai",
        "python-dotenv",
        "requests",
        "supabase",
        "openai>=1.6.0",
        "google-generativeai>=0.3.0",
        "mirascope[openai,gemini]",
        "pydantic",
        "pydantic-ai>=0.0.46",
        "nest_asyncio>=1.5.6",
        "streamlit>=1.22.0",
    ],
    entry_points={
        "console_scripts": [
            "doc-agent=main:main",
        ],
    },
    python_requires=">=3.11",
) 