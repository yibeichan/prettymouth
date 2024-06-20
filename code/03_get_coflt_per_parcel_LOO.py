import os
import glob
import sys
import numpy as np
from dotenv import load_dotenv
from utils.coflt import cofluctuation

def read_parcel_ts(group_id, n_parcel, p_id, output_dir):
    ts_files = glob.glob(
        os.path.join(output_dir, "parcel_ts", f"{n_parcel}parcel", f"{group_id}", f"*p{int(p_id):03d}.npy")
    )
    ts_files.sort()
    mean_parcel_ts_list = []

    for f in ts_files:
        parcel_ts = np.load(f)
        mean_parcel_ts = np.nanmean(parcel_ts, axis=1)
        mean_parcel_ts_list.append(mean_parcel_ts)
    
    return np.array(mean_parcel_ts_list)

def get_all_parcel_ts(group_id, n_parcel, output_dir):
    all_parcel_ts = [read_parcel_ts(group_id, n_parcel, p_id, output_dir) for p_id in range(1, n_parcel + 1)]
    return np.array(all_parcel_ts)

def intersubj_cofluctuation(ts_array, group_id, n_parcel, output_dir):
    # Validate input dimensions
    if ts_array.ndim != 3 or ts_array.shape[0] != n_parcel:
        raise ValueError("ts_array must be a 3D array with the shape (n_parcel, n_subjects, n_timepoints)")
    
    n_subj, n_tr = ts_array.shape[1], ts_array.shape[2]
    for s in range(n_subj):
        subj_corr = np.zeros((n_parcel, n_parcel, n_tr))
        for i, parcel_i in enumerate(ts_array):
            subj_ts = parcel_i[s]
            for j, parcel_j in enumerate(ts_array):
                other_subj_ts = np.delete(parcel_j, s, axis=0)
                other_ts = np.nanmean(other_subj_ts, axis=0)
                subj_corr[i, j, :] = cofluctuation(subj_ts, other_ts)
        save_dir = os.path.join(output_dir, "cofluctuation_LOO", f"{n_parcel}parcel", f"{group_id}")
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, f"subj{s+1:02d}_intersubj_coflt.npy"), subj_corr)
        print(f"{n_parcel}_{group_id}_subj{s+1} done.")

if __name__ == "__main__":
    group_id = sys.argv[1]
    n_parcel = int(sys.argv[2])
    load_dotenv()
    scratch_dir = os.getenv("SCRATCH_DIR")
    output_dir = os.path.join(scratch_dir, "output")
    
    all_parcel_ts = get_all_parcel_ts(group_id, n_parcel, output_dir)
    intersubj_cofluctuation(all_parcel_ts, group_id, n_parcel, output_dir)