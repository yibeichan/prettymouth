import os
import sys
from dotenv import load_dotenv
import gc
import pickle
from typing import Tuple
import numpy as np
import multiprocessing as mp
from utils.coflt import read_coflt_data, fdr_correction

import logging

import numpy as np

def calculate_distance_matrix(group1: np.ndarray, group2: np.ndarray, indices: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """
    Calculate distance matrix between two groups using 1 - Pearson's r, handling NaNs appropriately.
    
    Args:
        group1 (np.ndarray): First group's data matrix.
        group2 (np.ndarray): Second group's data matrix.
        indices (Tuple[np.ndarray, np.ndarray]): Upper triangle indices.
    
    Returns:
        np.ndarray: Distance matrix.
    """
    # Initialize the distance matrix with the correct shape
    n_node = group1.shape[1]
    n_TR = group1.shape[3]
    distance_matrix = np.zeros((n_node, n_node, n_TR), dtype=np.float32)
    
    # Iterate through each pair of nodes
    for i, j in zip(*indices):
        # Extract the data for the two groups for the given nodes across all TRs
        data1 = group1[:, i, j, :]
        data2 = group2[:, i, j, :]
        
        # Stack the data from both groups along a new axis
        stacked_data = np.stack((data1, data2), axis=0)
        
        # Calculate the Pearson correlation for each time point
        # Result will be a 2x2 matrix for each time point
        correlations = np.array([np.corrcoef(stacked_data[:, :, n])[0, 1] for n in range(n_TR)])
        
        # Handle potential NaN values resulting from correlation calculation
        correlations = np.where(np.isnan(correlations), 0, correlations)
        
        # Calculate distance as 1 - correlation
        distances = 1 - correlations
        
        distance_matrix[i, j, :] = distances
        if i != j:
            distance_matrix[j, i, :] = distances

    return distance_matrix

def permuted_distance_matrix(permuted_labels: np.ndarray, all_tsISC: np.ndarray, group_size: int, indices: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """
    Calculate the permuted distance matrix for given permuted labels.

    Args:
        permuted_labels (np.ndarray): Permuted labels for the subjects.
        all_tsISC (np.ndarray): Combined data from both groups.
        group_size (int): Size of one of the groups.
        indices (Tuple[np.ndarray, np.ndarray]): Indices of the upper triangle of the matrix.

    Returns:
        np.ndarray: The calculated permuted distance matrix.
    """
    perm_group1 = all_tsISC[permuted_labels[:group_size]]
    perm_group2 = all_tsISC[permuted_labels[group_size:]]

    distance_matrix = calculate_distance_matrix(perm_group1, perm_group2, indices)
    print(f"perm distance shape: {distance_matrix.shape}")
    
    del perm_group1, perm_group2
    gc.collect()  # Force garbage collection after large arrays are no longer needed
    return distance_matrix

def compute_worker_permutations(args):
    seeds, all_tsISC, group_size, indices, total_subjects = args
    null_dist_local = []
    for i, seed in enumerate(seeds):
        np.random.seed(seed)
        try:
            print(f"Processing seed {seed}")
            permuted_labels = np.random.permutation(total_subjects)
            permuted_distance = permuted_distance_matrix(permuted_labels, all_tsISC, group_size, indices)
            null_dist_local.append(permuted_distance)
        except Exception as e:
            logging.error(f"Error in permutation with seed {seed}: {e}")

        finally:
            del permuted_labels
            gc.collect()  # Force garbage collection after permutation is done
    
    return np.array(null_dist_local)

def permutation_test(matrix1: np.ndarray, matrix2: np.ndarray, n_permutations: int = 1000, n_jobs: int = 4) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform a permutation test to compare two matrices.

    Args:
        matrix1 (np.ndarray): The first matrix.
        matrix2 (np.ndarray): The second matrix.
        n_permutations (int): The number of permutations to perform. Default is 10000.
        n_jobs (int): The number of parallel processes to use. Default is 4.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing the observed distance matrix, the null distribution, and the p-value matrix.
    """
    n_node = matrix1.shape[1]
    indices = np.triu_indices(n_node)
    print(f"indices: {indices}")
    observed = calculate_distance_matrix(matrix1, matrix2, indices)
    print(f"observed shape: {observed.shape}")

    total_subjs = matrix1.shape[0] + matrix2.shape[0]
    group_size = matrix1.shape[0]
    all_tsISC = np.concatenate((matrix1, matrix2), axis=0, dtype=np.float32)

    del matrix1, matrix2
    gc.collect()

    seeds = np.random.randint(0, int(1e6), size=n_permutations)
    chunks = np.array_split(seeds, n_jobs)

    print("Start computing permutations")

    with mp.Pool(processes=n_jobs) as pool:
        results = pool.map(compute_worker_permutations, 
                           [(chunk, all_tsISC, group_size, indices, total_subjs) for chunk in chunks])

    # Concatenate the results
    null_distribution = np.concatenate(results, axis=0)
    print("Final null_distribution:", null_distribution)
    # Compute p-values
    eps = 1e-14
    gamma = np.maximum(eps, np.abs(eps * observed))
    adjustment = 1
    cmps = null_distribution >= observed - gamma
    p_values = (cmps.sum(axis=0) + adjustment) / (n_permutations + adjustment)
    p_values = np.clip(p_values, 0, 1)

    print(f"Number of p < 0.05: {np.sum(p_values < 0.05)}")

    assert p_values.shape == observed.shape, "The shape of the p_values is not correct."

    return observed, null_distribution, p_values

def main(n_parcel: int, output_dir: str) -> None:
    """
    Read data, perform calculations, and save results.

    Args:
        n_parcel (int): Number of parcels.
        output_dir (str): Directory to save the results.

    Returns:
        None
    """
    # Read the data
    coflt_dir = os.path.join(output_dir, "cofluctuation_LOO")
    affair_coflt = read_coflt_data(n_parcel, "affair", coflt_dir)
    paranoia_coflt = read_coflt_data(n_parcel, "paranoia", coflt_dir)

    # Perform permutation test
    observed, null_distribution, p_values = permutation_test(affair_coflt, paranoia_coflt, n_permutations=1000, n_jobs=4)
    corrected_p_values = fdr_correction(p_values[:,:, 17:468])
    
    # Save results to .npz file
    distance_output_dir = os.path.join(output_dir, "group_distance", f"{n_parcel}parcel")
    # output_file = os.path.join(distance_output_dir, f"event_group_distance_perm_ntw.npz")
    output_file = os.path.join(distance_output_dir, f"TR_group_distance_perm_parcel_pr.npz")
    np.savez(output_file, observed=observed, null_distribution=null_distribution, p_values=p_values, corrected_p_values=corrected_p_values)
    print(f"Saved permutation test results for {n_parcel} parcels")

if __name__ == "__main__":
    load_dotenv()
    # n_parcel = int(sys.argv[1])
    n_parcel = 400
    scratch_dir = os.getenv("SCRATCH_DIR")
    behav_dir = os.path.join(scratch_dir, "output", "behav_results")
    parcellation_dir = os.path.join(scratch_dir, "data", "fmriprep", "yan_parcellations")
    output_dir = os.path.join(scratch_dir, "output")

    main(n_parcel, output_dir)