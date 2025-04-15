from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh.readlines()]

setup(
    name="yfin",
    version="0.1.0",
    author="YFin Team",
    author_email="example@example.com",
    description="Advanced Stock Tracker using Yahoo Finance data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/yfin",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "yfin=run:main",
        ],
    },
) 