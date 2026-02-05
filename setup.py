from setuptools import setup, find_packages
import os

# Читаем README для длинного описания
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ruaccent-predictor",
    version="1.2.0",
    author="Eduard Emkuzhev",
    author_email="info@copperline.info",
    description="Russian stress accent prediction using Transformer model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kubataba/Russian-Stress-Accent-Predictor",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Text Processing :: Markup",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "tqdm>=4.65.0",
    ],
    package_data={
        "ruaccent": ["model/*.pt", "model/*.json"],
    },
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "ruaccent=ruaccent.cli:main",
        ],
    },
    keywords=[
        "nlp", "russian", "accent", "stress", 
        "transformer", "text-processing", "linguistics"
    ],
)
