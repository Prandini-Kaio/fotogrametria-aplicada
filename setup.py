from setuptools import setup, find_packages

setup(
    name="fotogrametria_3d",
    version="0.1",
    packages=find_packages(include=['src', 'src.*']),
    package_dir={'': '.'}
)