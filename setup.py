from setuptools import setup, find_packages

setup(
    name="neural-compression",
    version="0.1.0",
    author="Neural Compression Team",
    description="Advanced Deep Learning Compression System",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "pytorch-lightning>=2.0.0",
        "hydra-core>=1.3.0",
        "wandb>=0.15.0",
        "opencv-python>=4.8.0",
        "albumentations>=1.3.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
        "omegaconf>=2.3.0",
        "rich>=13.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
        "optimization": [
            "onnx>=1.14.0",
            "tensorrt>=8.6.0",
            "torch-tensorrt>=1.4.0",
        ],
        "analysis": [
            "jupyter>=1.0.0",
            "ipywidgets>=8.0.0",
            "plotly>=5.15.0",
            "seaborn>=0.12.0",
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    entry_points={
        "console_scripts": [
            "neural-compress=neural_compression.cli:main",
        ],
    },
)

import os