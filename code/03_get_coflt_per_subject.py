import os
import glob
import sys
import numpy as np
from dotenv import load_dotenv
from utils.coflt import cofluctuation

def pairwise_cofluctuations(data, n_parcel):
    """
    Calculate pairwise co-fluctuations between time series data from multiple parcels.

    Parameters:
        data (ndarray): A 2D array of shape (n_TR, n_p) representing time series data from n_p parcels.
        n_parcel (int): The number of parcels in the data.

    Returns:
        ndarray: A 3D array of shape (n_parcel, n_parcel, n_TR) containing the pairwise co-fluctuations between parcels.
    """
    assert len(data.shape) == 2, "Data must be 2D"
    n_TR, n_p = data.shape
    print("Shape of data:", data.shape)
    assert n_p == n_parcel, "Data must have the same number of parcels as specified"
    result = np.zeros((n_parcel, n_parcel, n_TR))
    
    for i in range(n_parcel):
        for j in range(i, n_parcel): 
            result[i, j] = cofluctuation(data[:, i], data[:, j])
            if i != j:  
                result[j, i] = result[i, j]
    return result

def read_parcel_ts(group_id, n_parcel, output_dir):
    """
    Read parcel time series data for a given group ID and number of parcels.

    Args:
        group_id: The ID of the group.
        n_parcel (int): The number of parcels.
        output_dir (str): The path to the output directory.

    Returns:
        subject_ts (dict): A dictionary containing the subject IDs as keys and their corresponding time series data as values.
            The time series data is a 2D numpy array of shape (n_timepoints, n_parcel), where n_timepoints is the number
            of time points and n_parcel is the number of parcels.
    """
    ts_files = glob.glob(
        os.path.join(output_dir, "parcel_ts", f"{n_parcel}parcel", f"{group_id}", "*.npy")
    )
    ts_files.sort()

    subject_ts = {}

    for f in ts_files:
        filename = os.path.basename(f)
        sub_id, p_id = filename.split('_')
        p_id = int(p_id.split('.')[0][1:])

        parcel_ts = np.load(f)
        mean_parcel_ts = np.nanmean(parcel_ts, axis=1)

        subject_ts.setdefault(sub_id, np.empty((len(mean_parcel_ts), n_parcel)) * np.nan)

        if len(mean_parcel_ts) != subject_ts[sub_id].shape[0]:
            raise ValueError(f"Inconsistent number of timepoints in parcel data for subject {sub_id}")

        subject_ts[sub_id][:, p_id - 1] = mean_parcel_ts
    print(f"Read {len(subject_ts)} subjects' parcel time series data.")
    return subject_ts

def process_and_save_subject_data(subject_ts, n_parcel, save_dir):
    """
    Process and save subject data.

    Args:
        subject_ts (dict): A dictionary containing subject timestamps.
        n_parcel (int): The number of parcels.
        save_dir (str): The directory where the data will be saved.

    Returns:
        None
    """
    for sub_id, data in subject_ts.items():
        print(f"Processing data for subject {sub_id}")
        result = pairwise_cofluctuations(data, n_parcel)
        print(f"Shape of result: {result.shape}")
        output_filename = f"{sub_id}.npy"
        output_path = os.path.join(save_dir, output_filename)
        np.save(output_path, result)
        print(f"Saved cofluctuation data for subject {sub_id} to {output_path}")

def main(group_id, n_parcel, output_dir):
    """
    Generates a main function comment.

    Args:
        group_id (int): The group id.
        n_parcel (int): The number of parcels.
        output_dir (str): The output directory.

    Returns:
        None
    """
    save_dir = os.path.join(output_dir, "coflt_per_subject", f"{n_parcel}parcel", f"{group_id}")
    os.makedirs(save_dir, exist_ok=True)
    process_and_save_subject_data(read_parcel_ts(group_id, n_parcel, output_dir), n_parcel, save_dir)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: script.py <group_id> <n_parcel>")
        sys.exit(1)

    group_id = sys.argv[1]
    n_parcel = int(sys.argv[2])
    load_dotenv()
    scratch_dir = os.getenv("SCRATCH_DIR")
    output_dir = os.path.join(scratch_dir, "output")
    main(group_id, n_parcel, output_dir)