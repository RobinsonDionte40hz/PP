"""
Prediction Engine Package Setup

This setup.py makes the prediction engine installable as a proper Python package.
External code can then import from `ubf_protein.api` without sys.path hacks.

Installation:
    # Core only (PyPy compatible)
    pip install -e .
    
    # With QCPP support
    pip install -e .[qcpp]
    
    # Full worker installation
    pip install -e .[worker]
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text() if readme_path.exists() else ""

setup(
    name="ubf-protein",
    version="1.0.0",
    description="EmergentFolds Protein Structure Prediction Engine",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="EmergentFolds Team",
    author_email="dionterobinson.biorxiv@gmail.com",
    url="https://emergentfolds.com",
    
    # Package discovery - find ubf_protein and all subpackages
    packages=find_packages(exclude=["tests", "tests.*", "examples", "experiments"]),
    
    # Python version
    python_requires=">=3.8",
    
    # Core dependencies (PyPy compatible)
    install_requires=[
        "typing-extensions>=4.0.0",
    ],
    
    # Optional dependencies
    extras_require={
        # QCPP support (CPython only)
        "qcpp": [
            "numpy>=1.20.0",
            "scipy>=1.7.0",
            "biopython>=1.79",
            "pandas>=1.2.0",
            "scikit-learn>=0.24.0",
            "matplotlib>=3.5.0",
        ],
        # Worker mode (with Celery)
        "worker": [
            "numpy>=1.20.0",
            "scipy>=1.7.0",
            "biopython>=1.79",
            "pandas>=1.2.0",
            "scikit-learn>=0.24.0",
            "celery>=5.4.0",
            "redis>=5.2.0",
        ],
        # Development
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=24.0.0",
            "mypy>=1.0.0",
        ],
    },
    
    # Entry points for CLI tools
    entry_points={
        "console_scripts": [
            "predict-protein=ubf_protein.cli:main",
        ],
    },
    
    # Classifiers
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
    
    # Package data
    include_package_data=True,
    package_data={
        "ubf_protein": ["*.json", "*.md"],
    },
)
