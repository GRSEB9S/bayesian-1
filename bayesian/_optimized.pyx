import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def table_mul(np.ndarray[np.float_t, ndim=1] left, np.ndarray[np.float_t, ndim=1] right, np.ndarray[np.float_t, ndim=1] result, int left_divide, int right_mod, int nb_elements):
    cdef int i
    for i in range(nb_elements):
        result[i] = \
            left[i // left_divide] * right[i % right_mod]
