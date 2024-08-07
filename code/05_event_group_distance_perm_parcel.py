import os
import sys
from dotenv import load_dotenv
import gc
import pickle
import numpy as np
from utils.preproc import get_roi_and_network_ids
from utils.coflt import read_coflt_data, reconstruct_matrix, get_seg_coflt, permutation_test, fdr_correction

def main(n_parcel: int, output_dir: str, behav_dir: str, parcellation_dir: str) -> None:
    """
    Read data, perform calculations, and save results.

    Args:
        n_parcel (int): Number of parcels.
        output_dir (str): Directory to save the results.
        behav_dir (str): Directory containing behavioral results.
        parcellation_dir (str): Directory containing parcellation data.

    Returns:
        None
    """
    # Read the data
    coflt_dir = os.path.join(output_dir, "cofluctuation_LOO")
    affair_coflt = read_coflt_data(n_parcel, "affair", coflt_dir)
    paranoia_coflt = read_coflt_data(n_parcel, "paranoia", coflt_dir)

    # Read event boundaries
    boundary_file = os.path.join(behav_dir, "prominent_peaks_filtered_024.txt")
    with open(boundary_file, 'r') as f:
        lines = f.readlines()
        boundaries = np.array([int(float(line.strip())) for line in lines], dtype=int)

    # Load parcellation data
    parcellation = np.load(os.path.join(parcellation_dir, f"yan_kong17_{n_parcel}parcels.npz"))
    _, _, node_idx_dict, _ = get_roi_and_network_ids(parcellation, n_parcel)

    # Reconstruct matrices
    node_mtx_affair, _ = reconstruct_matrix(affair_coflt, node_idx_dict)
    node_mtx_paranoia, _ = reconstruct_matrix(paranoia_coflt, node_idx_dict)

    # Clean up memory
    del affair_coflt, paranoia_coflt
    gc.collect()

    # Segment matrices based on event boundaries
    affair_seg = get_seg_coflt(boundaries, node_mtx_affair)
    paranoia_seg = get_seg_coflt(boundaries, node_mtx_paranoia)

    # Perform permutation test
    observed, null_distribution, p_values = permutation_test(affair_seg, paranoia_seg, n_permutations=5000, n_jobs=8)
    corrected_p_values = fdr_correction(p_values)
    
    # Save results to .npz file
    distance_output_dir = os.path.join(output_dir, "group_distance", f"{n_parcel}parcel")
    output_file = os.path.join(distance_output_dir, f"event_group_distance_perm_roi.npz")
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

    main(n_parcel, output_dir, behav_dir, parcellation_dir)