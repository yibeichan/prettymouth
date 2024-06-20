from typing import Tuple, Dict, List
from datetime import datetime
import numpy as np
from tqdm import tqdm
from scipy.stats import zscore
from scipy.linalg import norm
from statsmodels.stats.multitest import multipletests

def cofluctuation(x, y):
    return zscore(x, nan_policy='omit') * zscore(y, nan_policy='omit')

def process_coflt_file(file: str) -> np.ndarray:
    """
    Reads a numpy file and converts its data type to float32.

    Args:
        file (str): The path to the numpy file.

    Returns:
        np.ndarray: The processed data with float32 data type.
    """
    data = np.load(file, mmap_mode='r')
    data = data.astype(np.float32)
    return data

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
    import os
    import glob

    # Create file pattern
    file_pattern = os.path.join(output_dir, f"{n_parcel}parcel", group_id, "*.npy")

    # Get list of files matching the pattern
    files = glob.glob(file_pattern)
    files.sort()

    # Initialize an empty list to hold the data arrays
    coflt = []

    # Process each file individually to reduce memory usage
    for file in files:
        data = process_coflt_file(file)
        coflt.append(data)

    # Convert to a NumPy array
    coflt = np.array(coflt)

    # Print the shape of the coflt array
    print("coflt shape:", coflt.shape)

    # Assert whether coflt is 4D
    assert len(coflt.shape) == 4, "The coflt should be 4D."

    return coflt

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

def reconstruct_matrix(matrix: np.ndarray, name_idx_dict: Dict[str, List[int]]) -> np.ndarray:
    """
    Reconstructs a matrix by calculating the mean of specific elements from the original matrix based on the name-index mappings.

    Args:
        matrix (np.ndarray): The original matrix with shape (n_subjs, n_ROIs, n_ROIs, n_TRs).
        name_idx_dict (dict): A dictionary that maps names to indices.

    Returns:
        np.ndarray: The reconstructed matrix with shape (n_subjs, n_keys, n_keys, n_TRs).
    """
    print("Reconstructing matrix...", matrix.shape)
    n_subjs, _, _, n_TRs = matrix.shape
    n_keys = len(name_idx_dict)

    new_matrix = np.zeros((n_subjs, n_keys, n_keys, n_TRs))

    for i, k1 in enumerate(name_idx_dict.keys()):
        for j, k2 in enumerate(name_idx_dict.keys()):
            idx1 = name_idx_dict[k1]
            # print(k1, idx1)
            idx2 = name_idx_dict[k2]
            # print(k2, idx2)
            idx1_grid, idx2_grid = np.meshgrid(idx1, idx2, indexing='ij')
            print("Index grid shape:", idx1_grid.shape, idx2_grid.shape)
            sliced_matrix = matrix[:, idx1_grid, idx2_grid, :]
            print("Sliced matrix shape:", sliced_matrix.shape) 
            mean_values = np.nanmean(sliced_matrix, axis=(1, 2))
            print("Mean values shape:", mean_values.shape)
            new_matrix[:, i, j, :] = mean_values

    print("Reconstructed matrix shape:", new_matrix.shape)
    return new_matrix
    
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

def apply_fdr_correction(p_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply the False Discovery Rate (FDR) correction to a given array of p-values.

    Args:
        p_values: An array of p-values.

    Returns:
        A tuple containing the rejected hypotheses mask and the corrected p-values.
    """
    original_shape = p_values.shape  # Save the original shape
    p_values = p_values.flatten()  # Flatten the array to 1D

    # Apply the FDR correction
    rejected, p_values_corrected, _, _ = multipletests(p_values, method='fdr_bh')
    print(f"Number of rejected hypotheses: {np.sum(rejected)}")

    # Reshape the corrected p-values and the "rejected" mask back to the original shape
    p_values_corrected = p_values_corrected.reshape(original_shape)
    rejected = rejected.reshape(original_shape)

    return rejected, p_values_corrected

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

def permutation_test(matrix1: np.ndarray, matrix2: np.ndarray, n_permutations: int = 10000, alternative: str = "two-sided") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform a permutation test to compare two matrices.

    Args:
        matrix1 (np.ndarray): The first matrix.
        matrix2 (np.ndarray): The second matrix.
        n_permutations (int): The number of permutations to perform. Default is 10000.
        alternative (str): The alternative hypothesis. Can be "less", "greater", or "two-sided". Default is "two-sided".

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing the difference matrix, the corrected p-values, and the p-value matrix.
    """
    matrix1_upper = extract_upper_triangle(matrix1)
    matrix2_upper = extract_upper_triangle(matrix2)

    # get the absolute difference between the two upper triangles
    diff = matrix1_upper - matrix2_upper
    observed = np.nanmean(diff, axis=0)

    total_subjs = matrix1_upper.shape[0] + matrix2_upper.shape[0]
    
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    dat_filename = f"../logs/null_distribution_{timestamp}.dat"

    null_distribution = np.memmap(dat_filename, dtype='float32', mode='w+', shape=(n_permutations, *matrix1_upper.shape[1:]))

    # iterate over permutations with progress bar
    for i in tqdm(range(n_permutations)):
        permutation = np.random.permutation(total_subjs)
        total_matrix = np.vstack((matrix1_upper, matrix2_upper))
        permuted_data = np.split(total_matrix[permutation], 2)
        null_diff = permuted_data[0] - permuted_data[1]
        mean_null_diff = np.nanmean(null_diff, axis=0)
        null_distribution[i] = mean_null_diff

    # relative tolerance for detecting numerically distinct but theoretically equal values in the null distribution
    eps = 1e-14
    gamma = np.maximum(eps, np.abs(eps * observed))
    adjustment = 1

    def less(null_distribution, observed):
        cmps = null_distribution <= observed + gamma
        pvalues = (cmps.sum(axis=0) + adjustment) / (n_permutations + adjustment)
        return pvalues

    def greater(null_distribution, observed):
        cmps = null_distribution >= observed - gamma
        pvalues = (cmps.sum(axis=0) + adjustment) / (n_permutations + adjustment)
        return pvalues

    def two_sided(null_distribution, observed):
        pvalues_less = less(null_distribution, observed)
        pvalues_greater = greater(null_distribution, observed)
        pvalues = np.minimum(pvalues_less, pvalues_greater) * 2
        return pvalues

    compare = {"less": less,
            "greater": greater,
            "two-sided": two_sided}

    pvalues = compare[alternative](null_distribution, observed)
    pvalues = np.clip(pvalues, 0, 1)

    print(f"Number of p < 0.05: {np.sum(pvalues < 0.05)}")

    assert pvalues.shape == observed.shape, "The shape of the p_value is not correct."

    _, p_value_corrected = apply_fdr_correction(pvalues)

    p_value_mtx = to_matrix(p_value_corrected, n_dim=matrix1.shape[1])
    
    return diff, p_value_corrected, p_value_mtx

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