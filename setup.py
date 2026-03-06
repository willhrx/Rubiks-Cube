"""Setup script for the Rubik's Cube ML project."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="rubiks_cube_ml",
    version="0.1.0",
    description="Machine learning project for solving Rubik's Cube",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Will",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "torch>=1.10.0",
        "gymnasium>=0.28.0",
        "stable-baselines3>=2.0.0",
        "tensorboard>=2.7.0",
        "tqdm>=4.62.0",
        "matplotlib-inline>=0.1.3",
        "seaborn>=0.11.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.5",
            "black>=21.10b0",
            "mypy>=0.910",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
