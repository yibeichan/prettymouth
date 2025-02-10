import os
import glob
import numpy as np
from typing import Tuple, Dict, List
import numba as nb
from scipy.stats import zscore
from scipy.linalg import norm
from scipy.spatial.distance import euclidean
from statsmodels.stats.multitest import multipletests
import multiprocessing as mp
import logging
import gc
import psutil

def print_memory_usage(step: str):
    process = psutil.Process()
    mem_info = process.memory_info()
    print(f"{step} - Memory Usage: {mem_info.rss / (1024 * 1024):.2f} MB")

def cofluctuation(x, y):
    return zscore(x, nan_policy='omit') * zscore(y, nan_policy='omit')

def is_symmetric(matrix, tol=1e-8):
    return np.allclose(matrix, matrix.T, atol=tol, equal_nan=True)

def make_sym(data):
    transposed = np.transpose(data, (1, 0, 2))
    np.add(data, transposed, out=data)
    data /= 2
    return data

def process_coflt_file(file: str) -> np.ndarray:
    """
    Reads a numpy file and converts its data type to float32.

    Args:
        file (str): The path to the numpy file.

    Returns:
        np.ndarray: The processed data with float32 data type.
    """
    data = np.load(file, mmap_mode='r')
    data = data.astype(np.float32, copy=False)
    return make_sym(data)

def read_coflt_data(n_parcel: int, group_id: str, output_dir: str) -> np.ndarray:
    """
    Read and process coflt data files.

    Args:
        n_parcel (int): Number of parcels.
        group_id (str): Group ID.
        output_dir (str): Output directory.

    Returns:
        np.ndarray: Processed coflt data.

    Raises:
        AssertionError: If the coflt data is not 4D.
    """

    file_pattern = os.path.join(output_dir, f"{n_parcel}parcel", group_id, "*.npy")
    files = sorted(glob.glob(file_pattern))

    coflt = np.empty((len(files), *np.load(files[0], mmap_mode='r').shape), dtype=np.float32)

    for i, file in enumerate(files):
        coflt[i] = process_coflt_file(file)
    
    print("coflt shape:", coflt.shape)
    assert coflt.ndim == 4, "The coflt should be 4D."
    print_memory_usage("After reading coflt data")
    return coflt

def get_seg_coflt(boundaries: np.ndarray, coflt_mtx: np.ndarray, delta: int = 3) -> np.ndarray:
    """
    Calculate segmented co-fluctuation matrix.

    Args:
        boundaries (np.ndarray): List of boundary indices.
        coflt_mtx (np.ndarray): Co-fluctuation matrix.
        delta (int, optional): Delta value. Defaults to 3.

    Returns:
        list: List of segmented co-fluctuation matrices.
    """
    seg_coflt = []

    max_len = 0
    for i, r in enumerate(boundaries):

        end = r + delta

        if i == 0:
            start = 14 + delta
        else:
            start = boundaries[i - 1]  + delta
            
        # Ensure start and end are within bounds
        start = max(0, min(start, coflt_mtx.shape[3]))
        end = max(0, min(end, coflt_mtx.shape[3]))
        
        segment = coflt_mtx[:, :, :, start:end]
        seg_coflt.append(segment)
        max_len = max(max_len, segment.shape[3])
        print(f"Processed segment {i}: start {start}, end {end}, segment length {segment.shape[3]}")
    
    seg_coflt.append(coflt_mtx[:, :, :, end:465+delta])
    print(f"Number of segments: {len(seg_coflt)}")
    
    # Pad each segment to the maximum length with NaN
    padded_segments = []
    for segment in seg_coflt:
        pad_width = ((0, 0), (0, 0), (0, 0), (0, max_len - segment.shape[3]))
        padded_segment = np.pad(segment, pad_width, mode='constant', constant_values=np.nan)
        padded_segments.append(padded_segment)

    # Create output matrix
    output_mtx = np.stack(padded_segments, axis=0)
    # Transpose to make n_seg the last dimension
    print(f"before output_mtx shape: {output_mtx.shape}")
    output_mtx = np.transpose(output_mtx, [1, 2, 3, 4, 0])
    print(f"output mtx shape: {output_mtx.shape}")
    print_memory_usage("After processing seg_coflt")
    return output_mtx

def flatten_3D(data):
    n_row, _, n_TR = data.shape
    # Handle the case where the input array has a dimension of size 1
    if n_row == 1:
        return data.reshape((1, n_TR))
    # Extract upper triangle indices
    i_upper, j_upper = np.triu_indices(n_row, k=1)
    # Convert the 3D array into a 2D array with shape (n_edge*(n_edge-1)/2, n_TR)
    flattened = data[i_upper, j_upper]
    print(f"The shape of flattened is {flattened.shape}")
    assert flattened.shape[0] == n_row * (n_row - 1) / 2
    assert flattened.shape[1] == n_TR
    return flattened

def get_reconstructed_rms(matrix: np.ndarray, name_idx_dict: dict) -> np.ndarray:
    """
    Calculate the root mean square (RMS) of a matrix based on the indices specified in the name_idx_dict.

    Args:
        matrix (np.ndarray): The input matrix.
        name_idx_dict (dict): A dictionary mapping names to indices.

    Returns:
        np.ndarray: The RMS matrix.
    """
    _, _, n_TRs = matrix.shape
    if name_idx_dict != None:
        n_keys = len(name_idx_dict)
        # initialize an empty 2D matrix to save RMS
        rms_matrix = np.zeros((n_keys, n_TRs))
        for i, (_, idx) in enumerate(name_idx_dict.items()):
            # Extract only the relevant indices for each name
            relevant_data = matrix[idx, :, :][:, idx, :]
            # Flatten the upper triangle of the 3D matrix
            flattened_data = flatten_3D(relevant_data)
            # Calculate the RMS
            rms = norm(flattened_data, axis=0) / np.sqrt(flattened_data.shape[0])
            rms_matrix[i, :] = rms
    else:
        # calculating global rms
        n_keys = 1
        rms_matrix = np.zeros((n_keys, n_TRs))
        flattened_matrix = flatten_3D(matrix)
        flattened_matrix = np.nan_to_num(flattened_matrix, nan=0.0)
        rms = norm(flattened_matrix, axis=0) / np.sqrt(flattened_matrix.shape[0])
        rms_matrix[0, :] = rms

    return rms_matrix

def reconstruct_matrix(matrix: np.ndarray, name_idx_dict: Dict[str, List[int]]) -> Tuple[np.ndarray, Dict[Tuple[int, int], Tuple[str, str]]]:
    """
    Reconstructs a matrix by calculating the mean of specific elements from the original matrix based on the name-index mappings.
    Also returns a map of the new indices to the original keys.

    Args:
        matrix (np.ndarray): The original matrix with shape (n_subjs, n_ROIs, n_ROIs, n_TRs).
        name_idx_dict (dict): A dictionary that maps names to indices.

    Returns:
        Tuple[np.ndarray, Dict[Tuple[int, int], Tuple[str, str]]]: 
            - The reconstructed matrix with shape (n_subjs, n_keys, n_keys, n_TRs).
            - A dictionary mapping new matrix indices to the corresponding keys.
    """
    print("Reconstructing matrix...", matrix.shape)
    n_subjs, _, _, n_TRs = matrix.shape
    n_keys = len(name_idx_dict)

    new_matrix = np.zeros((n_subjs, n_keys, n_keys, n_TRs))
    index_key_map = {}

    for i, k1 in enumerate(name_idx_dict.keys()):
        for j, k2 in enumerate(name_idx_dict.keys()):
            idx1 = name_idx_dict[k1]
            idx2 = name_idx_dict[k2]
            idx1_grid, idx2_grid = np.meshgrid(idx1, idx2, indexing='ij')
            # print("Index grid shape:", idx1_grid.shape, idx2_grid.shape)
            sliced_matrix = matrix[:, idx1_grid, idx2_grid, :]
            # print("Sliced matrix shape:", sliced_matrix.shape) 
            mean_values = np.nanmean(sliced_matrix, axis=(1, 2))
            # print("Mean values shape:", mean_values.shape)
            new_matrix[:, i, j, :] = mean_values

            # Add to the index-key map
            index_key_map[(i, j)] = (k1, k2)

    print("Reconstructed matrix shape:", new_matrix.shape)
    return new_matrix, index_key_map
    
def extract_upper_triangle(matrix: np.ndarray) -> np.ndarray:
    """
    Extracts the upper triangle values from a square matrix along the 2nd and 3rd dimensions.

    Parameters:
    matrix (np.ndarray): The input matrix of shape (n_samples, n_dim1, n_dim2, n_timepoints).

    Returns:
    np.ndarray: The upper triangle values of the input matrix for each timepoint and sample.
    """
    _, n_dim1, n_dim2, _ = matrix.shape
    assert n_dim1 == n_dim2, "The matrix should be square along the 2nd and 3rd dimensions."

    # Get the indices of the upper triangle, including the diagonal.
    upper_tri_indices = np.triu_indices(n_dim1)
    # Extract the upper triangle values for each timepoint and sample.
    upper_tri_values = matrix[:, upper_tri_indices[0], upper_tri_indices[1], :]
    # This will give you a matrix of shape (n_samples, n_pairs, n_timepoints)
    return upper_tri_values


def fdr_correction(p_values: np.ndarray) -> np.ndarray:
    """
    Perform FDR correction on a symmetric 3D matrix of p-values.

    Args:
        p_values (np.ndarray): The symmetric 3D matrix of p-values (n_node, n_node, n_seg).

    Returns:
        np.ndarray: The FDR-corrected p-values with the same shape as input.
    """
    n_node, _, n_seg = p_values.shape
    indices = np.triu_indices(n_node, k=0)  # Include diagonal with k=0

    # Extract upper triangular elements including the diagonal for each segment
    upper_triangular_pvals = p_values[indices[0], indices[1], :].reshape(-1)

    # Perform FDR correction
    corrected_pvals = np.zeros_like(upper_triangular_pvals)
    rejected, corrected_pvals, _, _ = multipletests(upper_triangular_pvals, method='fdr_bh')
    print(f"Number of rejected hypotheses: {np.sum(rejected)}")

    # Reshape corrected p-values back to the shape for each segment
    corrected_pvals_reshaped = corrected_pvals.reshape(len(indices[0]), n_seg)

    # Create a full matrix of corrected p-values
    corrected_p_values = np.zeros_like(p_values)
    corrected_p_values[indices[0], indices[1], :] = corrected_pvals_reshaped
    corrected_p_values[indices[1], indices[0], :] = corrected_pvals_reshaped  # Symmetrize

    return corrected_p_values

def to_matrix(array: np.ndarray, n_dim: int) -> np.ndarray:
    """
    Convert a 2D array to a 3D matrix.

    Args:
        array (np.ndarray): 2D array.
        n_dim (int): Dimension of the square matrix.

    Returns:
        np.ndarray: 3D matrix where each slice represents a value.

    Raises:
        AssertionError: If array is not 2D or its size does not match n_dim.

    """
    assert len(array.shape) == 2, "array should be 2D"
    assert array.shape[0] == n_dim * (n_dim + 1) // 2, "Size of array does not match n_dim"

    matrix = np.zeros((n_dim, n_dim, array.shape[1]))  # initialize empty matrix

    for h in range(array.shape[1]):
        triu_idx = np.triu_indices(n_dim, k=0)
        matrix[triu_idx[0], triu_idx[1], h] = array[:, h]  # fill upper triangle

    for i in range(n_dim):
        for j in range(i+1, n_dim):  # don't include diagonal
            matrix[j, i, :] = matrix[i, j, :]  # fill lower triangle with upper triangle values

    return matrix

def nan_euclidean(u: np.ndarray, v: np.ndarray) -> float:
    """
    Compute Euclidean distance between two arrays, ignoring NaN values.
    
    Args:
        u (np.ndarray): First array.
        v (np.ndarray): Second array.
    
    Returns:
        float: Euclidean distance between u and v, ignoring NaN values.
    """
    mask = ~np.isnan(u) & ~np.isnan(v)
    if np.any(mask):
        return euclidean(u[mask], v[mask])
    else:
        return np.nan  # If there are no valid points to compare, return NaN

def calculate_distance_matrix(group1_avg: np.ndarray, group2_avg: np.ndarray, indices: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """
    Calculate distance matrix between two groups, handling NaNs appropriately.
    
    Args:
        group1_avg (np.ndarray): First group's average matrix.
        group2_avg (np.ndarray): Second group's average matrix.
        indices (Tuple[np.ndarray, np.ndarray]): Upper triangle indices.
    
    Returns:
        np.ndarray: Distance matrix.
    """
    distance_matrix = np.zeros((group1_avg.shape[0], group1_avg.shape[1], group1_avg.shape[3]), dtype=np.float32)
    for n in range(group1_avg.shape[3]):
        for i, j in zip(*indices):
            distance_matrix[i, j, n] = nan_euclidean(group1_avg[i, j, :, n], group2_avg[i, j, :, n])
            if i != j:
                distance_matrix[j, i, n] = distance_matrix[i, j, n]
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
    perm_group1 = np.mean(all_tsISC[permuted_labels[:group_size]], axis=0, dtype=np.float32)
    perm_group2 = np.mean(all_tsISC[permuted_labels[group_size:]], axis=0, dtype=np.float32)

    distance_matrix = calculate_distance_matrix(perm_group1, perm_group2, indices)
    print(f"perm distance shape: {distance_matrix.shape}")
    
    del perm_group1, perm_group2
    gc.collect()  # Force garbage collection after large arrays are no longer needed
    print_memory_usage("After GC in permuted_distance_matrix")
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

def permutation_test(matrix1: np.ndarray, matrix2: np.ndarray, n_permutations: int = 10000, n_jobs: int = 4) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    mtx1_avg = np.nanmean(matrix1, axis=0)
    mtx2_avg = np.nanmean(matrix2, axis=0)
    observed = calculate_distance_matrix(mtx1_avg, mtx2_avg, indices)
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

def get_eFC_chunked(data, chunk_size=1000):
    n_edges = data.shape[0]
    data -= data.mean(axis=1, keepdims=True)
    print("the data shape is zero centered")
    std_devs = np.std(data, axis=1)
    # Initialize the eFC matrix
    eFC_matrix = np.zeros((n_edges, n_edges))
    # Chunk-wise computation
    for i in range(0, n_edges, chunk_size):
        end_i = min(i + chunk_size, n_edges)
        for j in range(0, n_edges, chunk_size):
            end_j = min(j + chunk_size, n_edges)
            
            # Directly compute the sum of products without storing intermediate chunk_product
            chunk_sum = np.sum(data[i:end_i, None, :] * data[None, j:end_j, :], axis=2)
            
            norm_factors = np.sqrt(std_devs[i:end_i, None] * std_devs[None, j:end_j])
            eFC_matrix[i:end_i, j:end_j] = chunk_sum / norm_factors
            
            # Explicitly delete temporary variables to free memory
            del chunk_sum, norm_factors
    
    # Ensure values are bounded within [-1, 1] using in-place operation
    np.clip(eFC_matrix, -1, 1, out=eFC_matrix)
    print(f"The shape of eFC is {eFC_matrix.shape}")
    return eFC_matrix

def get_eFC(data):
    # Zero-center the data
    # data -= data.mean(axis=1, keepdims=True)
    # Compute standard deviations for each product time series
    std_devs = np.std(data, axis=1)

    # Element-wise multiplication between all pairs of time series
    elementwise_products = data[:, None, :] * data[None, :, :]
    
    # Sum over the time axis
    summed_products = np.sum(elementwise_products, axis=2)
    
    # Normalize by the square root of the product of standard deviations
    norm_factors = np.sqrt(std_devs[:, None] * std_devs[None, :])
    eFC_matrix = summed_products / norm_factors
    
    # Ensure values are bounded within [-1, 1]
    eFC_matrix = np.clip(eFC_matrix, -1, 1)
    
    print(f"The shape of eFC is {eFC_matrix.shape}")
    return eFC_matrix