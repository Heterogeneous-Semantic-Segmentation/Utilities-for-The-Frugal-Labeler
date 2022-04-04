from setuptools import setup, find_packages

setup(
    name="Utilities for the frugal labeler",
    version="1.1.0",
    url="https://github.com/Heterogeneous-Semantic-Segmentation/Utilities-for-The-Frugal-Labeler",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "pillow>=9.0.0",
        "numpy>=1.21.2",
        "segmentation-models>=1.0.1",
        "matplotlib>=2.8.2",
        "opencv-python>=4.5.5.62",
        "tensorflow>=2.8.0"
    ],
)
