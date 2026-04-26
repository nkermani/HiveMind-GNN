from setuptools import setup, find_packages

setup(
    name="hivemind-gnn",
    version="0.1.0",
    description="Neural Combinatorial Optimization for Autonomous Bee-Worker Routing",
    author="Nathan Kermani",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torch-geometric>=2.3.0",
        "networkx>=3.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pytest>=7.0.0",
        "tqdm>=4.65.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)