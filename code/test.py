import numpy as np
from scipy.stats import pearsonr
from memory_profiler import profile
import time

from joblib import Parallel, delayed

@profile
def is_symmetric(matrix, tol=1e-8):
    # Check if the matrix is square
    if matrix.shape[0] != matrix.shape[1]:
        return False
    
    # Check if the matrix is equal to its transpose
    return np.allclose(matrix, matrix.T, atol=tol)

@profile
def make_sym(data):
    transposed = np.transpose(data, [1, 0, 2])
    sym_data = (data + transposed) / 2
    assert is_symmetric(sym_data[:, :, 0])
    return sym_data


if __name__ == "__main__":
    # Generate random data for testing
    group1 = np.random.rand(19, 400, 400, 475)
    group2 = np.random.rand(19, 400, 400, 475)
    print("Shape of group1:", group1.shape)
    print("Shape of group2:", group2.shape)
 
    # Benchmark scipy euclidean batch method
    start_time = time.time()    
    distance_matrix = make_sym(group1)
    print("method took: {:.2f} seconds".format(time.time() - start_time))
    
    # Output the shape to ensure it is correct
    print("Shape of distance matrix:", distance_matrix.shape)
