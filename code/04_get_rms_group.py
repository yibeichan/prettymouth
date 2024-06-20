import os
import sys
from dotenv import load_dotenv
import numpy as np
from utils.preproc import get_roi_and_network_ids
from utils.coflt import read_coflt_data, get_reconstructed_rms

def calculate_and_save_rms(data, index_dicts, scenario, RMS_output_dir):
    """
    Calculate and save the root mean squares (RMS) for the given data.

    Parameters:
    data (numpy.ndarray): Input data.
    index_dicts (dict): Dictionary containing different index dictionaries.
    scenario (str): The current scenario being processed.
    RMS_output_dir (str): Directory to save the RSS output files.
    """
    for key, index_dict in index_dicts.items():
        print(f"Calculating RMS for {key}, {index_dict}")
        rss = get_reconstructed_rms(data, index_dict)
        print(f"RMS shape: {rss.shape}")
        np.save(os.path.join(RMS_output_dir, f"rms_{scenario}_{key}.npy"), rss)

def process_scenario(scenario, n_parcel, coft_output_dir, RMS_output_dir, index_dicts):
    data = read_coflt_data(n_parcel, scenario, coft_output_dir)
    mean_data = np.nanmean(data, axis=0)
    calculate_and_save_rms(mean_data, index_dicts, scenario, RMS_output_dir)

def main(n_parcel, parcellation_dir, coft_output_dir, RMS_output_dir):
    # Load parcellation data
    parcellation = np.load(os.path.join(parcellation_dir, f"yan_kong17_{n_parcel}parcels.npz"))
    roinames_idx_dict, roinames_ntw_idx_dict, roinames_hem_idx_dict, networknames_idx_dict = get_roi_and_network_ids(parcellation, n_parcel)

    index_dicts = {
        "ntw": networknames_idx_dict,
        "roi": roinames_idx_dict,
        "roi_ntw": roinames_ntw_idx_dict,
        "roi_hem": roinames_hem_idx_dict,
        "global": None
    }
    
    # Process the scenarios
    scenarios = ["affair", "paranoia"]
    for scenario in scenarios:
        print(f"Processing scenario: {scenario}")
        process_scenario(scenario, n_parcel, coft_output_dir, RMS_output_dir, index_dicts)

    # Process mix and diff
    mean_affair = np.nanmean(read_coflt_data(n_parcel, "affair", coft_output_dir), axis=0)
    mean_paranoia = np.nanmean(read_coflt_data(n_parcel, "paranoia", coft_output_dir), axis=0)

    mix_data = (mean_affair + mean_paranoia) / 2
    diff_data = np.abs(mean_affair - mean_paranoia)

    composite_data_dict = {"mix": mix_data, "diff": diff_data}
    for scenario, data in composite_data_dict.items():
        print(f"Processing scenario: {scenario}")
        calculate_and_save_rms(data, index_dicts, scenario, RMS_output_dir)

if __name__ == "__main__":
    load_dotenv()
    scratch_dir = os.getenv("SCRATCH_DIR")
    data_dir = os.path.join(scratch_dir, "data")
    parcellation_dir = os.path.join(data_dir, "fmriprep", "yan_parcellations")
    n_parcel = int(sys.argv[1])
    cal_type = sys.argv[2]
    if cal_type == "inter_coflt":
        coft_output_dir = os.path.join(scratch_dir, "output", "cofluctuation_LOO")
    elif cal_type == "intra_coflt":
        coft_output_dir = os.path.join(scratch_dir, "output", "coflt_per_subject")
    else:
        raise ValueError("cal_type must be 'inter_coflt' or 'intra_coflt'")
    RMS_output_dir = os.path.join(scratch_dir, "output", f"RMS_group_{cal_type}", f"{n_parcel}parcel")
    os.makedirs(RMS_output_dir, exist_ok=True)
    main(n_parcel, parcellation_dir, coft_output_dir, RMS_output_dir)