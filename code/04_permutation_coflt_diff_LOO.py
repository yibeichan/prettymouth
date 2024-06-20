from typing import Dict, List
import os
import sys
from dotenv import load_dotenv
import glob
import gc
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils.preproc import get_roi_and_network_ids
from utils.coflt import read_coflt_data, reconstruct_matrix, permutation_test

def get_seg_coflt(boundaries: np.ndarray, coflt_mtx: np.ndarray, delta: int = 3) -> np.ndarray:
    """
    Calculate segmented co-fluctuation matrix.

    Args:
        boundaries (np.ndarray): List of boundary indices.
        coflt_mtx (np.ndarray): Co-fluctuation matrix.
        delta (int, optional): Delta value. Defaults to 3.

    Returns:
        np.ndarray: Segmented co-fluctuation matrix.
    """
    num_boundaries = len(boundaries)
    seg_coflt = np.zeros((coflt_mtx.shape[0], coflt_mtx.shape[1], coflt_mtx.shape[2], num_boundaries))
    
    for i, r in enumerate(boundaries):
        if i==0:
            start = 14 + delta
        else:
            start = r + delta
        if i != len(boundaries) - 1:
            end = boundaries[i+1] + delta
        else:
            end = 465 + delta
        seg_coflt[:, :, :, i] = np.nanmean(coflt_mtx[:, :, :, start:end], axis=3)
    print(seg_coflt.shape)
    return seg_coflt

def plot_group_diff(diff, p_value_ntw, network_pairs, n_parcel, figure_dir):
    """
    Plot difference as a heatmap with p-values as the mask.

    Args:
        diff (np.ndarray): Difference matrix.
        p_value_ntw (np.ndarray): Array of p-values.
        network_pairs (list): List of network pairs.
        n_parcel (int): Number of parcels.
        figure_dir (str): Directory to save the plot.

    Returns:
        None
    """
    plt.figure(figsize=(12, 30))
    n_seg = p_value_ntw.shape[1]
    ax = sns.heatmap(np.nanmean(diff, axis=0), cmap="coolwarm", mask=p_value_ntw >= 0.05, yticklabels=network_pairs)
    plt.xticks(np.arange(0.5, n_seg+0.5, 1), np.arange(1, n_seg+1, 1))
    plt.xlabel("Story segment", fontsize=16)
    plt.ylabel("Network pairs", fontsize=16)

    # Set axis font size
    plt.tick_params(axis='both', which='major', labelsize=12)
    cbar = plt.gcf().axes[-1]
    cbar.tick_params(labelsize=12)
    # colorbar text size
    cbar.set_ylabel('Group difference', rotation=270, fontsize=16, labelpad=20)

    # X-axis ticks at the top and the bottom
    ax.xaxis.set_ticks_position('both')
    ax.xaxis.set_tick_params(labeltop=True, labelbottom=True)

    # Add vertical lines at each x-tick
    for x in np.arange(0.5, n_seg+0.5, 1):
        plt.axvline(x, color='darkgreen', alpha=0.1)

    # Save figure
    plt.savefig(os.path.join(figure_dir, f"sig_group_diff_{n_parcel}parcel.svg"), dpi=600, bbox_inches='tight')
    plt.close()
    
# we need to run the permutation test for "affair", "paranoia"
def main(n_parcel: int, parcellation_dir: str, output_dir: str, figure_dir: str) -> None:
    """
    Read data, perform calculations, and save results.

    Args:
        n_parcel (int): Number of parcels.
        parcellation_dir (str): Directory containing the parcellation data.
        output_dir (str): Directory to save the results.
        figure_dir (str): Directory to save the figures.

    Returns:
        None
    """
    # Read the data
    affair_coflt = read_coflt_data(n_parcel, "affair", output_dir)
    paranoia_coflt = read_coflt_data(n_parcel, "paranoia", output_dir)

    # Load parcellation data
    parcellation = np.load(os.path.join(parcellation_dir, f"yan_kong17_{n_parcel}parcels.npz"))
    _, _, _, networknames_idx_dict = get_roi_and_network_ids(parcellation, n_parcel)

    # Reconstruct matrices
    networks_mtx_affair = reconstruct_matrix(affair_coflt, networknames_idx_dict)
    networks_mtx_paranoia = reconstruct_matrix(paranoia_coflt, networknames_idx_dict)

    # Clean up memory
    del affair_coflt, paranoia_coflt
    gc.collect()

    # Read event boundaries
    boundary_file = os.path.join(boundary_dir, "prominent_peaks_010.txt")
    with open(boundary_file, 'r') as f:
        lines = f.readlines()
        boundaries = np.array([int(float(line.strip())) for line in lines], dtype=int)

    # Segment matrices based on event boundaries
    network_mtx_affair_seg = get_seg_coflt(boundaries, networks_mtx_affair)
    network_mtx_paranoia_seg = get_seg_coflt(boundaries, networks_mtx_paranoia)

    # Perform permutation test
    diff, p_value_ntw, p_value_mtx_ntw = permutation_test(network_mtx_affair_seg, network_mtx_paranoia_seg)

    # Save results to pickle file
    with open(os.path.join(output_dir, f"permutation_affair-paranoia_{n_parcel}parcel_p.pkl"), "wb") as f:
        pickle.dump((diff, p_value_ntw, p_value_mtx_ntw), f)
    print(f"Saved permutation test results for {n_parcel} parcels")

    # Generate network pairs for plotting
    networknames = list(networknames_idx_dict.keys())
    network_pairs = [f"{networknames[i]} x {networknames[j]}" for i in range(len(networknames)) for j in range(i, len(networknames))]

    # Plot p-values
    plot_group_diff(diff, p_value_ntw, network_pairs, n_parcel, figure_dir)

if __name__ == "__main__":
    load_dotenv()
    scratch_dir = os.getenv("SCRATCH_DIR")
    data_dir = os.path.join(scratch_dir, "data") # type: ignore
    parcellation_dir = os.path.join(data_dir, "fmriprep", "yan_parcellations")
    boundary_dir = os.path.join(scratch_dir, "output", "behav_results")
    output_dir = os.path.join(scratch_dir, "output", "cofluctuation_LOO") # type: ignore
    figure_dir = os.path.join(output_dir, "figures")
    os.makedirs(figure_dir, exist_ok=True)
    n_parcel = int(sys.argv[1])
    main(n_parcel, parcellation_dir, output_dir, figure_dir)