from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np

def defaultExtension( name ):
    return Extension( name=name,
                      sources=[ '%s.pyx'%name ],
                      include_dirs=[ np.get_include(), '.' ],
                      library_dirs=[],
                      extra_compile_args=[ '-fopenmp' ],
                      extra_link_args=[ '-fopenmp' ],
                      language='c++' )


setup( ext_modules=cythonize( [ defaultExtension( 'traversal' ),
                                defaultExtension( 'inference' ) ] ) )

# THIS IS INTENDED TO BE BUILD INSIDE A DOCKER CONTAINER!!!
# Just run "./run.sh" and then within the terminal "python -m host.main"