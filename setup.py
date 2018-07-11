#!/usr/bin/env python
import os
import shutil
import sys
from glob import glob
from setuptools import setup, find_packages
from torch.utils.cpp_extension import CppExtension, BuildExtension

readme = open('README.rst').read()

VERSION = '0.0.1'

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
    ext_modules=[
        CppExtension('torchcontrib._C', glob('torchcontrib/csrc/*.cpp'))
    ],
    cmdclass={'build_ext': BuildExtension},
)
