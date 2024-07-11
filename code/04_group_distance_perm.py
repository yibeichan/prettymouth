from memory_profiler import profile
import os
import sys
import numpy as np
import logging
import multiprocessing
from scipy.spatial.distance import euclidean
from statsmodels.stats.multitest import fdrcorrection
import gc
import psutil
from concurrent.futures import ProcessPoolExecutor, as_completed

def monitor_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 ** 3)  # Convert to GB

@profile
def calculate_distance_matrix(group1_avg, group2_avg, indices):
    distance_matrix = np.zeros((group1_avg.shape[0], group1_avg.shape[1]), dtype=np.float32)
    for i, j in zip(*indices):
        distance_matrix[i, j] = euclidean(group1_avg[:, i, j], group2_avg[:, i, j])
        if i != j:
            distance_matrix[j, i] = distance_matrix[i, j]
    return distance_matrix

@profile
def permuted_distance_matrix(permuted_labels, all_tsISC, group_size, indices):
    perm_group1 = np.mean(all_tsISC[permuted_labels[:group_size]], axis=0, dtype=np.float32)
    perm_group2 = np.mean(all_tsISC[permuted_labels[group_size:]], axis=0, dtype=np.float32)
    
    distance_matrix = calculate_distance_matrix(perm_group1, perm_group2, indices)
    
    del perm_group1, perm_group2
    gc.collect()  # Force garbage collection after large arrays are no longer needed
    
    return distance_matrix

@profile
def compute_and_update_p_values(seed, observed_distance_matrix, all_tsISC, group_size, indices, p_values, total_subjects):
    try:
        np.random.seed(seed)
        permuted_labels = np.random.permutation(total_subjects)
        permuted_distance = permuted_distance_matrix(permuted_labels, all_tsISC, group_size, indices)
        
        for i, j in zip(*indices):
            observed_dist = observed_distance_matrix[i, j]
            if permuted_distance[i, j] >= observed_dist:
                p_values[i, j] += 1
                if i != j:
                    p_values[j, i] += 1
    except Exception as e:
        logging.error(f"Error in permutation with seed {seed}: {e}")
    finally:
        del permuted_labels, permuted_distance
        gc.collect()  # Force garbage collection after permutation is done

@profile
def permute_and_compute_distances(group1_tsISC, group2_tsISC, n_permutations=1000, n_jobs=1):
    total_subjects = group1_tsISC.shape[0] + group2_tsISC.shape[0]
    group_size = group1_tsISC.shape[0]
    all_tsISC = np.concatenate((group1_tsISC, group2_tsISC), axis=0, dtype=np.float32)
    n_parcel = group1_tsISC.shape[1]
    
    group1_avg = np.mean(group1_tsISC, axis=0, dtype=np.float32)
    group2_avg = np.mean(group2_tsISC, axis=0, dtype=np.float32)

    del group1_tsISC, group2_tsISC
    gc.collect()
    
    indices = np.triu_indices(n_parcel)
    
    observed_distance_matrix = calculate_distance_matrix(group1_avg, group2_avg, indices)
    print(f"Observed distance matrix calculated. Current memory usage: {monitor_memory():.2f} GB")

    p_values = np.zeros((n_parcel, n_parcel), dtype=np.float32)

    seeds = np.random.randint(0, 1e6, size=n_permutations)
    
    print("Start parallel computing")
    pool_args = [(seed, observed_distance_matrix, all_tsISC, group_size, indices, p_values, total_subjects) for seed in seeds]
    
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = [executor.submit(compute_and_update_p_values, *args) for args in pool_args]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logging.error(f"Error in permutation: {e}")
            finally:
                gc.collect()
            print(f"Permutation done. Current memory usage: {monitor_memory():.2f} GB")

    p_values /= n_permutations

    p_values_flat = p_values[indices]
    _, p_values_corrected_flat = fdrcorrection(p_values_flat)
    
    p_values_corrected = np.zeros((n_parcel, n_parcel), dtype=np.float32)
    for idx, (i, j) in enumerate(zip(*indices)):
        p_values_corrected[i, j] = p_values_corrected_flat[idx]
        if i != j:
            p_values_corrected[j, i] = p_values_corrected_flat[idx]
    
    return observed_distance_matrix, p_values, p_values_corrected

@profile
def save_results_as_npz(observed_distance_matrix, p_values, p_values_corrected, distance_output_dir):
    results_path = os.path.join(distance_output_dir, "distance_per_parcel_permutated.npz")
    print(f"Saving results to {results_path}...")
    np.savez(results_path, observed_distance_matrix=observed_distance_matrix, 
             p_values=p_values, p_values_corrected=p_values_corrected)
    print(f"Results saved. Current memory usage: {monitor_memory():.2f} GB")

@profile
def main(affair_coflt, paranoia_coflt, distance_output_dir, n_jobs):
    start, end = 14, 465
    group1_tsISC = affair_coflt[:, :, :, start:end]
    group2_tsISC = paranoia_coflt[:, :, :, start:end]

    observed_distance_matrix, p_values, p_values_corrected = permute_and_compute_distances(
        group1_tsISC, group2_tsISC, n_permutations=1000, n_jobs=n_jobs
    )
    
    save_results_as_npz(observed_distance_matrix, p_values, p_values_corrected, distance_output_dir)

if __name__ == '__main__':
    from dotenv import load_dotenv
    from utils import setup_logging
    from utils.coflt import read_coflt_data

    load_dotenv()
    scratch_dir = os.getenv("SCRATCH_DIR")
    base_dir = os.getenv("BASE_DIR")
    output_dir = os.path.join(scratch_dir, "output")
    coflt_LOO = os.path.join(output_dir, "cofluctuation_LOO")
    
    try:
        n_parcel = int(sys.argv[1])
    except (IndexError, ValueError):
        logging.error("Please provide the number of parcels as a command line argument.")
        sys.exit(1)

    affair_coflt_LOO = os.path.join(coflt_LOO, f"{n_parcel}parcel", "affair")
    paranoia_coflt_LOO = os.path.join(coflt_LOO, f"{n_parcel}parcel", "paranoia")

    affair_coflt = read_coflt_data(n_parcel, "affair", coflt_LOO)
    paranoia_coflt = read_coflt_data(n_parcel, "paranoia", coflt_LOO)

    distance_output_dir = os.path.join(output_dir, "group_distance", f"{n_parcel}parcel")
    os.makedirs(distance_output_dir, exist_ok=True)

    log_dir = os.path.join(output_dir, "logs")
    setup_logging(base_dir=base_dir, task="group_distance_perm", task_id=f"{n_parcel}parcel")

    n_jobs = 4
    print(f"Running {n_jobs} jobs. Initial memory usage: {monitor_memory():.2f} GB")
    
    main(affair_coflt, paranoia_coflt, distance_output_dir, n_jobs)
