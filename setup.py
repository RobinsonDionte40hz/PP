from setuptools import setup, find_packages

setup(
    name="quantum_protein_predictor",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "biopython>=1.79",
        "pandas>=1.2.0",
        "scikit-learn>=0.24.0",
    ],
    description="Protein structure prediction using quantum resonance principles",
    author="Quantum Protein Structure Prediction Team",
    author_email="dionterobinson.biorxiv@gmail.com",
    url="https://github.com/RobinsonDionte40hz/quantum_protein_predictorII",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.8",
)