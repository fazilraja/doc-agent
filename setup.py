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
        "openai",
        "mirascope",
        "pydantic",
    ],
    entry_points={
        "console_scripts": [
            "doc-agent=main:main",
        ],
    },
    python_requires=">=3.8",
) 