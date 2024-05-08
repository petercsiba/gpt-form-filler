from setuptools import find_packages, setup


# Function to read the requirements file
def parse_requirements(filename):
    """Load requirements from a pip requirements file."""
    with open(filename) as req_file:
        lines = req_file.readlines()
    # Filter out comments and empty lines, and ensure we have no leading/trailing whitespace
    requirements = [
        line.strip() for line in lines if line.strip() and not line.startswith("#")
    ]
    return requirements


setup(
    name="gpt-form-filler",
    version="0.1.0",
    author="Peter Csiba",
    author_email="me@petercsiba.com",
    description="Programmatically fill in annoying intake forms from data dumps or knowledge base",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/petercsiba/gpt-form-filler",
    packages=find_packages(),
    install_requires=parse_requirements("requirements/common.txt"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.7",  # Minimum Python version requirement
)
