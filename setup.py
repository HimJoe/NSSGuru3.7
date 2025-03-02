from setuptools import setup, find_packages

setup(
    name="agentic_ai",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "streamlit>=1.20.0",
        "anthropic>=0.18.0",
        "pyyaml>=6.0",
    ],
)
