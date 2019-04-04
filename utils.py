import numpy as np
from numba import njit


@njit(error_model='numpy', parallel=True, fastmath=True)
def get_run_length_enc(array):
    """given a 2D float32 image of 0's and 1's, returns the run-length encoding of that array
    for submission (a list of integers)"""
    flattened_array = np.transpose(array).flatten()
    run_length = []
    # handles the first element of the array
    if flattened_array[0] == 1.:
        run_length.append(1)
    # body of the array
    for i in range(1, flattened_array.shape[0]):
        if flattened_array[i] == 1. and flattened_array[i-1] == 0.:
            run_length.append(i + 1)
        elif flattened_array[i] == 0. and flattened_array[i-1] == 1.:
            run_length.append(i - run_length[-1] + 1)
    # last element
    if flattened_array[-1] == 1.:
        run_length.append(flattened_array.shape[0] - run_length[-1] + 1)

    return run_length
