import os
import sys
import glob
import gc
from dotenv import load_dotenv
import numpy as np
from skbio.stats.distance import mantel

def compute_mantel_test(matrix_i, matrix_j):
    r, p, _ = mantel(matrix_i, matrix_j, method="pearson", permutations=999)
    print(f"Completed Mantel test computation. r = {r}, p = {p}")
    return r, p

def main(n_parcel, group, eFC_output_dir):
    print("Starting main function.")
    matrix_files = sorted(glob.glob(os.path.join(eFC_output_dir, f"{n_parcel}parcel", f"{group}_eFC_event*.npy")))
    num_matrices = len(matrix_files)
    similarity_matrix = np.zeros((num_matrices, num_matrices), dtype=float)
    p_values = np.zeros((num_matrices, num_matrices), dtype=float)

    for i in range(num_matrices):
        matrix_i = 1 - np.load(matrix_files[i])
        for j in range(i, num_matrices):
            print(f"Computing similarity for matrices {i} and {j}.")
            matrix_j = 1 - np.load(matrix_files[j])
            sim_value, p_value = compute_mantel_test(matrix_i, matrix_j)
            similarity_matrix[i, j] = sim_value
            similarity_matrix[j, i] = sim_value
            p_values[i, j] = p_value
            p_values[j, i] = p_value
            gc.collect()
            print(f"Completed similarity for matrices {i} and {j}.")

    print("Saving results to disk.")
    np.save(os.path.join(eFC_output_dir, f"{n_parcel}parcel", f"{group}_eFC_similarity_matrix.npy"), similarity_matrix)
    np.save(os.path.join(eFC_output_dir, f"{n_parcel}parcel", f"{group}_eFC_p_values.npy"), p_values)
    print(f"Saved similarity matrix and p-values for {group}.")
    
if __name__ == "__main__":
    load_dotenv()
    scratch_dir = os.getenv("SCRATCH_DIR")
    data_dir = os.path.join(scratch_dir, "data")
    n_parcel = int(sys.argv[1])
    eFC_type = sys.argv[2]
    group = sys.argv[3]
    eFC_output_dir = os.path.join(scratch_dir, "output", f"eFC_{eFC_type}")
    main(n_parcel, group, eFC_output_dir)