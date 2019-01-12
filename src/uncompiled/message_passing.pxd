import cython
import numpy as np
cimport numpy as np

cdef extern from 'constants.h' namespace 'cython_accessors':
    cdef int NODE
    cdef int EDGE

    cdef int NODE_INDEX
    cdef int N_EDGES
