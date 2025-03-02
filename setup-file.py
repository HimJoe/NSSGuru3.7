"""
Setup script for the Agentic AI framework.
"""

from setuptools import setup, find_packages

# Read the requirements
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# Read the long description
with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name="agentic-ai",
    version="0.1.0",
    description="A framework for building autonomous AI agents with reasoning, tool use, and memory",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/agentic-ai-solution",
    packages=find_packages("src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=requirements,
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="ai, agent, llm, autonomous, tools, reasoning",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/agentic-ai-solution/issues",
        "Documentation": "https://github.com/yourusername/agentic-ai-solution/docs",
        "Source Code": "https://github.com/yourusername/agentic-ai-solution",
    },
)
