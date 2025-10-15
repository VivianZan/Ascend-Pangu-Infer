from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ascend-pangu-infer",
    version="0.1.0",
    author="Ascend Pangu Inference Team",
    description="Inference framework for Pangu models on Ascend hardware",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/VivianZan/Ascend-Pangu-Infer",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "torch>=2.0.0",
    ],
    extras_require={
        "full": [
            "transformers>=4.30.0",
            "accelerate>=0.20.0",
            "sentencepiece>=0.1.99",
            "protobuf>=3.20.0",
        ],
        "mindspore": [
            "mindspore>=2.0.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
    },
)
