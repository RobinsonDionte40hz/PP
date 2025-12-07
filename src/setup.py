from setuptools import setup, find_packages

setup(
    name="quantum_protein_predictor",
    version="0.2.0",  # Updated for Quantum Refinement Engine integration
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "biopython>=1.79",
        "pandas>=1.2.0",
        "scikit-learn>=0.24.0",
    ],
    description="Dual-system protein structure prediction: QCPP + UBF with Quantum Refinement Engine",
    long_description="""
    PRIMARY MODULES:
    - test_protein.py: Universal protein testing with Quantum Refinement Engine
    - systematic_protein_testing.py: Systematic testing campaign (100+ proteins)
    
    SYSTEMS:
    - UBF Protein System (ubf_protein/): Multi-agent exploration with quantum refinement
    - QCPP System (src/): Quantum coherence stability prediction
    
    FEATURES:
    - Real RMSD calculations with CA-only native structure alignment (FIXED)
    - Quantum Refinement Engine for two-stage optimization
    - Multi-agent coordination with consciousness-inspired parameters
    - Geometric attractor analysis (golden ratio patterns)
    - Mediator agents for pattern detection
    - Production-ready validation suite (999/1016 tests passing)
    """,
    author="Quantum Protein Structure Prediction Team",
    author_email="dionterobinson.biorxiv@gmail.com",
    url="https://github.com/RobinsonDionte40hz/quantum_protein_predictorII",
    classifiers=[
        "Development Status :: 4 - Beta",  # Updated from Alpha
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    entry_points={
        'console_scripts': [
            'test-protein=test_protein:main',
            'systematic-testing=systematic_protein_testing:main',
            'qcpp-analysis=run_analysis:main',  # Legacy
        ],
    },
)