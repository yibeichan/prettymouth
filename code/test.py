import numpy as np
from scipy.stats import pearsonr
from memory_profiler import profile
import time

from joblib import Parallel, delayed

def compute_correlation(i, j, coflt_data, button_press_data):
    x = coflt_data[i, j, :] - np.mean(coflt_data[i, j, :])
    y = button_press_data
    corr = np.corrcoef(x, y)[0, 1]
    return corr

@profile
def get_correlation(coflt_data, button_press_data, n_jobs=4):
    n_parcel = coflt_data.shape[0]
    cor_matrix = np.zeros((n_parcel, n_parcel), dtype=np.float32)  # Use float32 for memory efficiency

    # Subtract the mean of the button_press_data from itself
    button_press_data = button_press_data - np.mean(button_press_data)

    # Parallel computation of correlations
    # results = Parallel(n_jobs=n_jobs)(delayed(compute_correlation)(i, j, coflt_data, button_press_data)
    #                                   for i in range(n_parcel) for j in range(i, n_parcel))
    for i in range(n_parcel):
        for j in range(i, n_parcel):
            corr = compute_correlation(i, j, coflt_data, button_press_data)
            cor_matrix[i, j] = corr

            if i != j:
                cor_matrix[j, i] = corr
    
    return cor_matrix

if __name__ == "__main__":
    # Generate random data for testing
    coflt_data = np.random.rand(400, 400, 475)
    button_press_data = np.random.rand(475)
    print("Shape of cofluctuation matrix:", coflt_data.shape)
    print("Shape of button press data:", button_press_data.shape)
 
    # Benchmark scipy euclidean batch method
    start_time = time.time()    
    cor_matrix = get_correlation(coflt_data, button_press_data)
    print("pearson r method took: {:.2f} seconds".format(time.time() - start_time))
    
    # Output the shape to ensure it is correct
    print("Shape of distance matrix:", cor_matrix.shape)
