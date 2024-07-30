import os
import sys
from dotenv import load_dotenv
import gc
import pickle
import numpy as np
from utils.coflt import read_coflt_data, get_seg_coflt, permutation_test, fdr_correction

def main(n_parcel: int, output_dir: str, behav_dir: str) -> None:
    """
    Read data, perform calculations, and save results.

    Args:
        n_parcel (int): Number of parcels.
        output_dir (str): Directory to save the results.
        behav_dir (str): Directory containing behavioral results.

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

    # Segment matrices based on event boundaries
    affair_seg = get_seg_coflt(boundaries, affair_coflt)
    paranoia_seg = get_seg_coflt(boundaries, paranoia_coflt)

    del affair_coflt, paranoia_coflt
    gc.collect()

    # Perform permutation test
    observed, null_distribution, p_values = permutation_test(affair_seg, paranoia_seg, n_permutations=10000, n_jobs=4)
    corrected_p_values = fdr_correction(p_values)
    
     # Save results to .npz file
    distance_output_dir = os.path.join(output_dir, "group_distance", f"{n_parcel}parcel")
    output_file = os.path.join(distance_output_dir, f"event_group_distance_perm_parcel.npz")
    np.savez(output_file, observed=observed, null_distribution=null_distribution, p_values=p_values, corrected_p_values=corrected_p_values)
    print(f"Saved permutation test results for {n_parcel} parcels")

if __name__ == "__main__":
    load_dotenv()
    n_parcel = int(sys.argv[1])
    scratch_dir = os.getenv("SCRATCH_DIR")
    behav_dir = os.path.join(scratch_dir, "output", "behav_results")
    output_dir = os.path.join(output_dir, "output", "group_distance", f"{n_parcel}parcel")

    main(n_parcel, output_dir, behav_dir)