# https://docs.python.org/3/distutils/apiref.html
from distutils.core import setup
from distutils.extension import Extension

import os


random_cpp = Extension(
    'random_cpp',
    sources=['random_cpp.cpp'],
    libraries=['boost_python39', 'boost_numpy39'],
    extra_compile_args=['-std=c++2a']  # lambda support required
)

setup(
    name='random_cpp',
    version='0.1',
    ext_modules=[random_cpp])

# call with: python setup.py build_ext --inplace
