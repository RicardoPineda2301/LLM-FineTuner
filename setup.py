#!/usr/bin/env python3
"""
Setup script para LLM-FineTuner.

Instala el paquete en modo desarrollo.
"""

from setuptools import setup, find_packages

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [req.strip() for req in f.readlines() if req.strip() and not req.startswith("#")]

setup(
    name="llm-finetuner",
    version="0.1.0",
    description="Fine-tuner local para modelos de lenguaje usando QLoRA",
    author="LLM-FineTuner",
    author_email="",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "llm-ingest = scripts.ingest:main",
            "llm-train = scripts.train:main",
            "llm-merge = scripts.merge:main",
            "llm-export = scripts.export:main",
            "llm-package = scripts.package_ollama:main",
            "llm-create-dataset = scripts.create_dataset:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Operating System :: OS Independent",
    ],
)
