from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension

import numpy

extension = Extension('cutils',
                        sources=['cutils.pyx', 
                                ],
                        include_dirs = [
                                        numpy.get_include(),
                                        # "/Users/holly/Programs/eigen"
                                        ],

                       language="c++",
                       extra_compile_args=['-std=c++11'],

                      )

setup(
    ext_modules = cythonize(extension)
)