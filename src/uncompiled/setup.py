from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np

setup( ext_modules=cythonize( Extension( name="traversal",
                                         sources=[ 'traversal.pyx' ],
                                         include_dirs=[ np.get_include() ],
                                         library_dirs=[],
                                         extra_compile_args=[ '-fopenmp' ],
                                         extra_link_args=[ '-fopenmp' ],
                                         language='c++' ) ) )

# THIS IS INTENDED TO BE BUILD INSIDE A DOCKER CONTAINER!!!
# Just run run.sh