from setuptools import setup, find_packages

setup(
    name="massive-serve",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # Add your package dependencies here
    ],
    author="Rulin Shao",
    author_email="rulins@cs.washington.edu",
    description="A package for massive serving",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/RulinShao/massive-serve",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
) 