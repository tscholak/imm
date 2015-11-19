#!/usr/bin/env python
# -*- coding: utf-8 -*-

import codecs
import os

try:
    from setuptools import setup, find_packages
    from setuptools import Extension
except ImportError:
    from distutils.core import setup, find_packages
    from distutils.extension import Extension

here = os.path.abspath(os.path.dirname(__file__))

def read(*parts):
    return codecs.open(os.path.join(here, *parts), 'r').read()

from imm import __version__

install_requires = ['numpy', 'scipy']

ext_modules = []

if __name__ == '__main__':
    setup(name='imm',
          version=__version__,
          description='Infinite mixture model library for Python.',
          long_description=read('README.rst'),
          classifiers=['Programming Language :: Python',
                       'Development Status :: 3 - Alpha',
                       'Natural Language :: English',
                       'Environment :: Console',
                       'Intended Audience :: Science/Research',
                       'License :: OSI Approved :: Apache Software License',
                       'Operating System :: OS Independent',
                       'Programming Language :: Python :: 2.7',
                       'Topic :: Scientific/Engineering :: Information Analysis',
                       'Topic :: Scientific/Engineering :: Mathematics',
                       'Topic :: Utilities',],
          keywords='Bayesian inference statistics',
          license='Apache License 2.0',
          url='https://github.com/tscholak/imm',
          author='Torsten Scholak',
          author_email='torsten.scholak@googlemail.com',
          packages=find_packages(),
          platforms=['Any'],
          install_requires=install_requires,
          ext_modules=ext_modules,)
