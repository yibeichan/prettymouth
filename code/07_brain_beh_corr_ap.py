import os
import sys
import pickle
import numpy as np
import pandas as pd

from scipy.special import factorial
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests

import multiprocessing as mp
from memory_profiler import profile

from dotenv import load_dotenv

def boynton_hrf(t, T0=0, n=4, lambd=2):
    h = np.zeros_like(t)
    mask = t > T0
    h[mask] = ((t[mask] - T0)**(n-1) / (lambd**n * factorial(n-1))) * np.exp(-(t[mask] - T0) / lambd)
    return h

def convolution(button_presses, hrf):
    n_participants = button_presses.shape[0]
    n_trs = button_presses.shape[1]

    convolved_data = np.zeros_like(button_presses, dtype=float)

    for i in range(n_participants):
        convolved_data[i, :] = np.convolve(button_presses[i, :], hrf)[:n_trs]

    aggregated_convolved_data = np.mean(convolved_data, axis=0)

    return aggregated_convolved_data

def compute_correlation(i, j, coflt_data, button_press_data):
    x = coflt_data[i, j, :] - np.mean(coflt_data[i, j, :])
    y = button_press_data
    corr = np.sum(x * y) / np.sqrt(np.sum(x ** 2) * np.sum(y ** 2))
    return corr

def get_correlation(coflt_data, button_press_data):
    n_parcel = coflt_data.shape[0]
    cor_matrix = np.zeros((n_parcel, n_parcel), dtype=np.float32)  # Use float32 for memory efficiency    

    for i in range(n_parcel):
        for j in range(i, n_parcel):
            corr = compute_correlation(i, j, coflt_data, button_press_data)
            cor_matrix[i, j] = corr

            if i != j:
                cor_matrix[j, i] = corr
    
    return cor_matrix

def worker(coflt_data, convolved_data, chunk):
    n_parcel = coflt_data.shape[0]
    cor_matrices = np.zeros((len(chunk), n_parcel, n_parcel), dtype=np.float32)
    for i, seed in enumerate(chunk):
        np.random.seed(seed)
        shuffled_convolved_data = np.random.permutation(convolved_data)
        cor_matrix = get_correlation(coflt_data, shuffled_convolved_data)
        cor_matrices[i, :, :] = cor_matrix
    return cor_matrices

@profile
def permutation_test(coflt_data, convolved_data, n_permutations=10000, n_jobs=4):
    n_parcel = coflt_data.shape[0]
    seeds = np.random.randint(0, int(1e6), size=n_permutations)
    chunks = np.array_split(seeds, n_jobs)
    convolved_data = convolved_data - np.mean(convolved_data)
    
    observed_cor_matrix = get_correlation(coflt_data, convolved_data)

    with mp.Pool(processes=n_jobs) as pool:
        results = pool.starmap(worker, [(coflt_data, convolved_data.copy(), chunk) for chunk in chunks])

    permuted_cor_matrices = np.vstack(results)

    eps = 1e-14
    adjustment = 1
    pvalue_matrix = np.zeros((n_parcel, n_parcel), dtype=np.float32)

    for i in range(n_parcel):
        for j in range(i, n_parcel):  # Only upper triangle and diagonal
            observed = np.abs(observed_cor_matrix[i, j])
            gamma = np.maximum(eps, np.abs(eps * observed))
            cmps = permuted_cor_matrices[:, i, j] >= observed - gamma
            pvalue_matrix[i, j] = (cmps.sum() + adjustment) / (n_permutations + adjustment)
            if i != j:
                pvalue_matrix[j, i] = pvalue_matrix[i, j]  # Mirror the upper triangle to the lower

    # Ensure p-values are within [0, 1]
    pvalue_matrix = np.clip(pvalue_matrix, 0, 1)

    return observed_cor_matrix, permuted_cor_matrices, pvalue_matrix

def fdr_correction(pvalue_matrix):

    n_parcel = pvalue_matrix.shape[0]

    upper_triangle_indices = np.triu_indices(n_parcel)
    upper_triangle_pvalues = pvalue_matrix[upper_triangle_indices]

    _, corrected_pvalues, _, _ = multipletests(upper_triangle_pvalues, method='fdr_bh')

    corrected_pvalue_matrix = np.zeros_like(pvalue_matrix, dtype=np.float32)
    corrected_pvalue_matrix[upper_triangle_indices] = corrected_pvalues
    corrected_pvalue_matrix = np.triu(corrected_pvalue_matrix) + np.triu(corrected_pvalue_matrix, 1).T

    return corrected_pvalue_matrix


def main(n_parcel, brain_group, beh_group, output_dir):
    evidence_file = os.path.join(output_dir, "behav_results", f"individual_response_evidence_{beh_group}.pkl")

    with open(evidence_file, 'rb') as f:
        evidence = pickle.load(f)

    botton_press = np.vstack(evidence["binary_response"].values.tolist())[:, 14:]

    tr = 1.5
    # Covering a range of 0 to 30 seconds with steps of tr
    times = np.arange(0, 30, tr)

    hrf = boynton_hrf(times)

    convolved_button = convolution(botton_press, hrf)

    group_coflt_file = os.path.join(output_dir, "group_distance", f"{n_parcel}parcel", f"{brain_group}_coflt_avg.npy")

    group_coflt = np.load(group_coflt_file)[:, :, 14:465]

    cor_mtx, permuted_cor_matrices, pvalue_mtx = permutation_test(group_coflt, convolved_button, n_permutations=10000, n_jobs=12)

    corrected_pvalue_mtx = fdr_correction(pvalue_mtx)
    save_dir = os.path.join(output_dir, "brain_beh_corr", f"{n_parcel}parcel")
    os.makedirs(save_dir, exist_ok=True)
    # save results to npz file
    output_file = os.path.join(save_dir, f"brain_beh_corr_{brain_group}-{beh_group}.npz")
    np.savez(output_file, cor_mtx=cor_mtx, permuted_cor_matrices=permuted_cor_matrices, pvalue_mtx=pvalue_mtx, corrected_pvalue_mtx=corrected_pvalue_mtx)

if __name__ == "__main__":
    load_dotenv()
    n_parcel = int(sys.argv[1])
    scratch_dir = os.getenv("SCRATCH_DIR")
    output_dir = os.path.join(scratch_dir, "output")
    brain_group = "affair"
    beh_group = "paranoia"

    main(n_parcel, brain_group, beh_group, output_dir)
