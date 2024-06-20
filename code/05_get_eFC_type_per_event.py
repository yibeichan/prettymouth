import os
import sys
from dotenv import load_dotenv
import gc
import numpy as np
from utils.coflt import get_eFC

def bootstrap_eFC(data, n_iterations=5000):
    """
    Compute bootstrapped eFC.
    Args:
        data (numpy.ndarray): The flattened co-fluctuation matrix.
        n_iterations (int): Number of bootstrap iterations.
    Returns:
        mean_eFC (numpy.ndarray): Mean eFC across bootstrap iterations.
    """
    eFC_samples = []
    
    for _ in range(n_iterations):
        # Sample with replacement along the time axis
        bootstrap_sample = data[:, np.random.choice(data.shape[1], size=data.shape[1], replace=True)]
        eFC_samples.append(get_eFC(bootstrap_sample))
        
    mean_eFC = np.mean(eFC_samples, axis=0)
    return mean_eFC

def get_seg_eFC(boundaries, flattened_coflt, group, data_dir):
    delta = 3
    for i, r in enumerate(boundaries):
        if i == 0:
            start = 14 + delta
        else:
            start = r + delta
        if i != len(boundaries) - 1:
            end = boundaries[i+1] + delta
        else:
            end = 465 + delta
        seg_coflt = flattened_coflt[:, start:end]
        seg_coflt -= seg_coflt.mean(axis=1, keepdims=True)
        
        # Use bootstrapping to compute eFC
        seg_eFC = bootstrap_eFC(seg_coflt)
        
        np.save(os.path.join(data_dir, f"{group}_eFC_event{i+1:02d}.npy"), seg_eFC)
        print(f"Saved {group}_eFC_event{i+1:02d}.npy")
        del seg_coflt, seg_eFC
        gc.collect()

def main(n_parcel, group, boundary_dir, eFC_output_dir):
    data_dir = os.path.join(eFC_output_dir, f"{n_parcel}parcel")
    
    boundary_file = os.path.join(boundary_dir, "prominent_peaks_010.txt")
    with open(boundary_file, 'r') as f:
        lines = f.readlines()
        boundaries = np.array([int(float(line.strip())) for line in lines], dtype=int)
    
    flattened_data = np.load(os.path.join(data_dir, f"flattened_{group}_coflt.npy"))
    get_seg_eFC(boundaries, flattened_data, group, data_dir)

if __name__ == "__main__":
    load_dotenv()
    scratch_dir = os.getenv("SCRATCH_DIR")
    boundary_dir = os.path.join(scratch_dir, "output", "behav_results")
    n_parcel = int(sys.argv[1])
    eFC_type = sys.argv[2]
    group = sys.argv[3]
    eFC_output_dir = os.path.join(scratch_dir, "output", f"eFC_{eFC_type}")
    main(n_parcel, group, boundary_dir, eFC_output_dir)
