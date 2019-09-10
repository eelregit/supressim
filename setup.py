from setuptools import setup
from setuptools import find_packages

setup(
    name='supressim',
    version='0.0',
    description='Simulation enhancers and transformers',
    author='Yin Li, Yu Feng, et al.',
    author_email='eelregit@gmail.com',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'bigfile',  # only for training data preprocessing
    ],
    scripts=['scripts/srgan.py']
)
