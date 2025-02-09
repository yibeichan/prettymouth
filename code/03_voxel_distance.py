import os
import numpy as np
import nibabel as nib
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm
from multiprocessing import Pool
from dotenv import load_dotenv
import argparse

# Add constants at the top
SHIFT_START_IDX = 17
SHIFT_END_IDX = 468

def calculate_euclidean_distance(group1: np.ndarray, group2: np.ndarray) -> np.ndarray:
    """Calculate voxel-wise Euclidean distance between two groups.
    
    Args:
        group1: Array of shape (n_subjects, n_voxels, n_timepoints)
        group2: Array of shape (m_subjects, n_voxels, n_timepoints)
        
    Returns:
        distances: Array of shape (n_voxels,) containing Euclidean distances
    """
    if not isinstance(group1, np.ndarray) or not isinstance(group2, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
    if group1.shape[1:] != group2.shape[1:]:
        raise ValueError(f"Incompatible shapes: {group1.shape} vs {group2.shape}")
    
    # Calculate mean time series for each voxel across subjects
    # Shape: (n_voxels, n_timepoints)
    mean_group1 = np.nanmean(group1, axis=0)  # Average across subjects
    mean_group2 = np.nanmean(group2, axis=0)
    
    # Calculate Euclidean distance for each voxel's time series
    # For each voxel, we want sqrt(sum((ts1 - ts2)^2)) where ts1, ts2 are time series
    diff = mean_group1 - mean_group2  # Shape: (n_voxels, n_timepoints)
    distances = np.sqrt(np.sum(diff * diff, axis=1))  # Sum across timepoints
    
    return np.nan_to_num(distances, nan=0.0, posinf=0.0, neginf=0.0)

def run_permutation_multiprocessing(args):
    combined_groups, n_subjects, permuted_idx = args
    permuted_groups = combined_groups[permuted_idx]
    perm_group1 = permuted_groups[:n_subjects // 2]
    perm_group2 = permuted_groups[n_subjects // 2:]
    return calculate_euclidean_distance(perm_group1, perm_group2)

def batch_permutation_test_multiprocessing(group1, group2, save_dir, num_permutations=10000, batch_size=1000, n_jobs=4):
    """Optimized permutation test implementation."""
    n_subjects = len(group1) + len(group2)
    rng = np.random.default_rng(seed=42)  # Create RNG once at the start
    
    chunk_size = batch_size * n_jobs
    n_chunks = (num_permutations + chunk_size - 1) // chunk_size
    
    combined_groups = np.concatenate([group1, group2], axis=0)
    original_distances = calculate_euclidean_distance(group1, group2)
    
    with Pool(n_jobs) as pool:
        for chunk_idx in tqdm(range(n_chunks)):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, num_permutations)
            
            # Each call to permutation will generate different random permutations
            # even with the same RNG because it's a new call each time
            chunk_permutations = np.array([
                rng.permutation(n_subjects) 
                for _ in range(end_idx - start_idx)
            ])
            
            # Process chunk in parallel
            chunk_results = pool.map(
                run_permutation_multiprocessing,
                [(combined_groups, n_subjects, perm) 
                 for perm in chunk_permutations]
            )
            
            # Save results immediately
            chunk_results = np.array(chunk_results)
            np.save(
                os.path.join(save_dir, f"permuted_distances_{start_idx}_{end_idx}.npy"),
                chunk_results
            )
            
            del chunk_results
    return original_distances

def load_permuted_distances(save_dir, num_permutations, batch_size, n_voxels):
    """Load permuted distances from saved batch files.
    
    Args:
        save_dir: Directory containing saved batch files
        num_permutations: Total number of permutations
        batch_size: Size of each batch
        n_voxels: Number of voxels
        
    Returns:
        np.ndarray: Array of permuted distances
    """
    permuted_distances = np.zeros((num_permutations, n_voxels))
    n_batches = (num_permutations + batch_size - 1) // batch_size
    
    for batch in range(n_batches):
        start = batch * batch_size
        end = min(start + batch_size, num_permutations)
        batch_file = os.path.join(save_dir, f"permuted_distances_{start}_{end}.npy")
        batch_data = np.load(batch_file)
        permuted_distances[start:end] = batch_data

    return permuted_distances

def fdr_correction(original_distances, permuted_distances, alpha=0.05):
    """
    Apply FDR correction and return significant voxels.
    """
    p_values = np.mean(permuted_distances >= original_distances, axis=0)
    _, p_values_fdr, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')
    significant_voxels = p_values_fdr < alpha
    return significant_voxels, p_values, p_values_fdr

def load_data(output_dir, shift=True):
    """
    Load data for both groups from files.
    """
    group1 = np.load(os.path.join(output_dir, "group_data_native", "affair_data_postproc.npy"))
    group2 = np.load(os.path.join(output_dir, "group_data_native", "paranoia_data_postproc.npy"))
    
    if shift:
        return group1[:, :, SHIFT_START_IDX:SHIFT_END_IDX], group2[:, :, SHIFT_START_IDX:SHIFT_END_IDX]
    else:
        return group1, group2

def main(output_dir, save_dir, shift=True, num_permutations=10000, batch_size=1000, n_jobs=4):
    group1, group2 = load_data(output_dir)
    print("Data loaded.")
    
    # Perform the permutation test and save the permuted distances in batches
    original_distances = batch_permutation_test_multiprocessing(group1, group2, save_dir, num_permutations, batch_size, n_jobs)
    
    # Load all the saved permuted distances for FDR correction
    n_voxels = group1.shape[1]
    permuted_distances = load_permuted_distances(save_dir, num_permutations, batch_size, n_voxels)
    
    # Perform FDR correction
    significant_voxels, p_values, p_values_fdr = fdr_correction(original_distances, permuted_distances)
    
    # Save the results
    if shift:
        outfile = os.path.join(save_dir, "masked_wholebrain_distance_perm_3TR_shifted.npz")
    else:
        outfile = os.path.join(save_dir, "masked_wholebrain_distance_perm.npz")
        
    np.savez_compressed(outfile, original_distances=original_distances, 
                        permuted_distances=permuted_distances, 
                        significant_voxels=significant_voxels, 
                        p_values=p_values, p_values_fdr=p_values_fdr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate voxel-wise distances between groups')
    parser.add_argument('--no-shift', action='store_false', dest='shift',
                       help='Disable time series shifting (default: True)')
    parser.add_argument('--njobs', type=int, default=6,
                       help='Number of parallel jobs (default: 32)')
    args = parser.parse_args()
    
    load_dotenv()
    
    base_dir = os.getenv("BASE_DIR")
    scratch_dir = os.getenv("SCRATCH_DIR") 
    output_dir = os.path.join(scratch_dir, "output")
    save_dir = os.path.join(output_dir, "group_distance_perm")
    os.makedirs(save_dir, exist_ok=True)
    
    num_permutations = 10000
    batch_size = 1000  
            
    main(output_dir, save_dir, shift=args.shift, 
         num_permutations=num_permutations, 
         batch_size=batch_size, 
         n_jobs=args.njobs)
