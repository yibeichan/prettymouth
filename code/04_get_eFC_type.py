import os
import sys
from dotenv import load_dotenv
import gc
import numpy as np
from utils.preproc import get_roi_and_network_ids
from utils.coflt import reconstruct_matrix, read_coflt_data, flatten_3D, get_eFC

def process_group(n_parcel, group, coft_output_dir, idx_dict, eFC_output_dir):
    coflt_data = read_coflt_data(n_parcel, group, coft_output_dir)
    reconstructed_coflt = reconstruct_matrix(coflt_data, idx_dict)
    # Average across subjects
    mean_coflt = np.nanmean(reconstructed_coflt, axis=0)
    del coflt_data, reconstructed_coflt
    gc.collect()
    
    save_dir = os.path.join(eFC_output_dir, f"{n_parcel}parcel")
    os.makedirs(save_dir, exist_ok=True)

    flattened_data = flatten_3D(mean_coflt)
    np.save(os.path.join(save_dir, f"flattened_{group}_coflt.npy"), flattened_data)
    # flattened_data = np.load(os.path.join(save_dir, f"flattened_{group_name}_coflt.npy"))
    delta = 3
    start = 14 + delta
    end = start + 451
    eFC_data = get_eFC(flattened_data[:, start:end])
    np.save(os.path.join(save_dir, f"{group}_eFC.npy"), eFC_data)
    del flattened_data, eFC_data
    gc.collect()

def main(n_parcel, eFC_type, group, coft_output_dir, eFC_output_dir):
    # Load parcellation data
    parcellation = np.load(os.path.join(parcellation_dir, f"yan_kong17_{n_parcel}parcels.npz"))
    _, roinames_ntw_idx_dict, _, networknames_idx_dict = get_roi_and_network_ids(parcellation, n_parcel)
    if eFC_type == "roi_network":
        idx_dict = roinames_ntw_idx_dict
    elif eFC_type == "network":
        idx_dict = networknames_idx_dict
    else:
        raise ValueError("eFC type must be one of 'roi_network' or 'network'.")
    # Process individual groups
    process_group(n_parcel, group, coft_output_dir, idx_dict, eFC_output_dir)

if __name__ == "__main__":
    load_dotenv()
    scratch_dir = os.getenv("SCRATCH_DIR")
    data_dir = os.path.join(scratch_dir, "data")
    parcellation_dir = os.path.join(data_dir, "fmriprep", "yan_parcellations")
    coft_output_dir = os.path.join(scratch_dir, "output", "cofluctuation_LOO")
    n_parcel = int(sys.argv[1])
    eFC_type = sys.argv[2]
    group = sys.argv[3]
    eFC_output_dir = os.path.join(scratch_dir, "output", f"eFC_{eFC_type}")
    os.makedirs(eFC_output_dir, exist_ok=True)
    main(n_parcel, eFC_type, group, coft_output_dir, eFC_output_dir)
