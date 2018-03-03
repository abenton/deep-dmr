#!/usr/bin/env python

import setuptools
from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules=[
    Extension("deepdmr._sample",
              ["deepdmr/_sample.pyx", 'deepdmr/gamma.c'],
              libraries=["m"],
              extra_compile_args = ["-O3", "-ffast-math", "-march=native" ],
              extra_link_args=[]
              )
]

setup( 
  name = "deepdmr",
  cmdclass = {"build_ext": build_ext},
  ext_modules = ext_modules
)
