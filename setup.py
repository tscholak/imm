#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np

try:
    from setuptools import setup, find_packages
    from setuptools import Extension
except ImportError:
    from distutils.core import setup, find_packages
    from distutils.extension import Extension

try:
    from Cython.Distutils import build_ext
    from Cython.Build import cythonize
except ImportError:
    raise RuntimeError("Cython is required to build extension modules.")


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

def get_version():
    dummy = {}
    exec(read('imm/__version__.py'), dummy)
    return dummy['__version__']
__version__ = get_version()

def read_to_rst(fname):
    try:
        import pypandoc
        with open('README.rst', 'w') as f:
            f.write(pypandoc.convert('README.md', 'rst'))
    except:
        return read(fname)

compile_flags = ["-O3", "-Wall", "-std=c++11"]

def make_extension(ext_name, ext_libraries=(), is_directory=False):
    ext_path = ext_name
    if is_directory:
        ext_path += ".__init__"
    return Extension(
            ext_name,
            [ext_path.replace(".", os.path.sep) + ".pyx"],
            include_dirs=([np.get_include(), "."]),
            language="c++",
            libraries=ext_libraries,
            extra_compile_args=compile_flags,
        )

extensions = [
        make_extension("imm.utils"),
    ]

nthreads = int(os.environ.get('COMPILE_NTHREADS', 0))

ext_modules = cythonize(extensions, nthreads=nthreads)

setup(name='imm',
      version=__version__,
      author='Torsten Scholak',
      author_email='torsten.scholak@googlemail.com',
      description='Bayesian inference for infinite mixture models',
      long_description=read_to_rst('README.md'),
      license='Apache License 2.0',
      keywords=['Bayesian', 'inference', 'statistics', 'mcmc'],
      url='https://github.com/tscholak/imm',
      ext_modules=ext_modules,
      include_dirs=[np.get_include()],
      packages=find_packages(),
      platforms=['Any'],
      install_requires=['numpy>=1.7', 'scipy>=0.16', 'cython'],
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
                   'Topic :: Utilities',],)
