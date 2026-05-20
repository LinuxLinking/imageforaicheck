from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="imageforai",
    version="1.0.0",
    author="AI Image Detection Team",
    author_email="team@example.com",
    description="AI Image Detection Toolkit - Detect AI-generated images, extract metadata, and analyze image content",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/imageforai",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "Pillow>=9.0.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "opencv-python>=4.5.0",
        "ultralytics>=8.0.0",
        "exifread>=2.3.0",
        "piexif>=1.1.3",
    ],
    entry_points={
        "console_scripts": [
            "imageforai=imageforai.cli:main",
        ],
    },
)