import os
import numpy as np
import nibabel as nib
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm
from multiprocessing import Pool
from dotenv import load_dotenv

def calculate_euclidean_distance(group1, group2):
    assert group1.shape == group2.shape, "Group shapes must match"
    mean_group1 = np.mean(group1, axis=0)
    mean_group2 = np.mean(group2, axis=0)
    distances = np.linalg.norm(mean_group1 - mean_group2, axis=0)
    return distances

def run_permutation_multiprocessing(args):
    combined_groups, n_subjects, permuted_idx = args
    permuted_groups = combined_groups[permuted_idx]
    perm_group1 = permuted_groups[:n_subjects // 2]
    perm_group2 = permuted_groups[n_subjects // 2:]
    return calculate_euclidean_distance(perm_group1, perm_group2)

def batch_permutation_test_multiprocessing(group1, group2, save_dir, num_permutations=10000, batch_size=1000, n_jobs=4):
    n_subjects = group1.shape[0] + group2.shape[0]
    
    combined_groups = np.concatenate([group1, group2], axis=0)
    original_distances = calculate_euclidean_distance(group1, group2)
    
    with Pool(n_jobs) as pool:
        for batch in tqdm(range(num_permutations // batch_size)):
            start = batch * batch_size
            end = start + batch_size
            
            batch_permuted_indices = [np.random.permutation(combined_groups.shape[0]) for _ in range(batch_size)]
            batch_results = pool.map(run_permutation_multiprocessing, [(combined_groups, n_subjects, perm_idx) for perm_idx in batch_permuted_indices])
            batch_results = np.array(batch_results)
            
            # Save batch to disk to avoid memory overuse
            np.save(os.path.join(save_dir, f"permuted_distances_{start}_{end}.npy"), batch_results)
            print(f"Saved batch {start} to {end}")
            
            # Optionally delete batch_results to free up memory
            del batch_results
    
    return original_distances

def load_permuted_distances(save_dir, num_permutations, batch_size, n_voxels):
    """
    Load permuted distances that were saved in batches to disk.
    """
    permuted_distances = np.zeros((num_permutations, n_voxels))
    
    for batch in range(num_permutations // batch_size):
        start = batch * batch_size
        end = start + batch_size
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
    group1 = np.load(os.path.join(output_dir, "group_data", "affair_data_masked.npy"))
    group2 = np.load(os.path.join(output_dir, "group_data", "paranoia_data_masked.npy"))
    
    if shift:
        return group1[:, 17:468, :], group2[:, 17:468, :]
    else:
        return group1, group2

def main(output_dir, save_dir, shift=True, num_permutations=10000, batch_size=1000, n_jobs=4):
    group1, group2 = load_data(output_dir)
    print("Data loaded.")
    
    # Perform the permutation test and save the permuted distances in batches
    original_distances = batch_permutation_test_multiprocessing(group1, group2, save_dir, num_permutations, batch_size, n_jobs)
    
    # Load all the saved permuted distances for FDR correction
    n_voxels = group1.shape[2]
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
    load_dotenv()
    
    base_dir = os.getenv("BASE_DIR")
    scratch_dir = os.getenv("SCRATCH_DIR")
    nese_dir = os.getenv("NESE_DIR")
    output_dir = os.path.join(nese_dir, "output")
    save_dir = os.path.join(output_dir, "group_distance_perm")
    os.makedirs(save_dir, exist_ok=True)
    
    num_permutations = int(os.getenv("NUM_PERMUTATIONS", 10000)) 
    batch_size = int(os.getenv("BATCH_SIZE", 1000))  
    n_jobs = int(os.getenv("N_JOBS", 4))  
            
    main(output_dir, save_dir, shift=True, num_permutations=num_permutations, batch_size=batch_size, n_jobs=n_jobs)