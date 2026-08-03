from setuptools import setup, find_packages

setup(
    name="krish_assignment",
    version="0.1.0",
    packages=find_packages(),
    # Using flexible version constraints to avoid conflicts with cluster libraries
    # If libraries are pre-installed on cluster, these won't reinstall
    install_requires=[
        "transformers>=4.30.0",  # Flexible: accepts any version >= 4.30.0
        "torch>=2.0.0",
        "mlflow>=2.0.0",
        "databricks-sdk>=0.1.0",
    ],
)