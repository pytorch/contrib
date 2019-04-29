#!/usr/bin/env python
import os
import shutil
import sys
from setuptools import setup, find_packages


readme = open('README.rst').read()

VERSION = '0.0.2'

setup(
    # Metadata
    name='torchcontrib',
    version=VERSION,
    author='PyTorch Core Team and Contributors',
    author_email='soumith@pytorch.org',
    url='https://github.com/pytorch/contrib',
    description='implementations of ideas from recent papers',
    long_description=readme,
    license='BSD',

    # Package info
    packages=find_packages(exclude=('test',)),

    zip_safe=True,
)
