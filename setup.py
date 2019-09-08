from setuptools import setup
from setuptools import find_packages

setup(
    name='supressim',
    version='0.0',
    description='Super Resolutuion nbody Simulations',
    author='Yin Li',
    author_email='eelregit@gmail.com',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'bigfile', # only for training data preprocessing
    ], 
    scripts=['scripts/srgan.py']
)
