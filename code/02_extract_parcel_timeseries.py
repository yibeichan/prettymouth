import os
import numpy as np
import pandas as pd
import nibabel as nib
from dotenv import load_dotenv
import argparse

def create_directories(output_dir, n_parcel, group_info):
    parcel_save_dir = os.path.join(output_dir, "parcel_ts", f"{n_parcel}parcel")
    os.makedirs(parcel_save_dir, exist_ok=True)

    for group_id in np.unique(group_info["group"]):
        group_dir = os.path.join(parcel_save_dir, group_id)
        os.makedirs(group_dir, exist_ok=True)
    
    return parcel_save_dir

def process_subject(row, parcellation, n_parcel, parcel_save_dir):
    subj_id = row["subj_id"]
    group_id = row["group"]
    ppp_file = row["post_preprocessed_files"]

    img = nib.load(ppp_file)
    img_data = np.array(img.dataobj)

    for i in range(1, n_parcel+1):
        p_idx = np.where(parcellation["map_all"] == i)
        parcel_ts = img_data[:, p_idx][:, 0]
        np.save(
            os.path.join(
                parcel_save_dir,
                f"{group_id}",
                f"{subj_id}_p{int(i):03d}.npy"
            ),
            parcel_ts
        )

def main(n_parcel):
    load_dotenv()
    scratch_dir = os.getenv("SCRATCH_DIR")
    data_dir = os.path.join(scratch_dir, 'data')
    fmriprep_dir = os.path.join(data_dir, "fmriprep")
    parcellation_dir = os.path.join(fmriprep_dir, "yan_parcellations")
    output_dir = os.path.join(scratch_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    group_info = pd.read_csv(os.path.join(data_dir, "group_info.csv"))
    parcellation = np.load(os.path.join(parcellation_dir, f"yan_kong17_{n_parcel}parcels.npz"))
    parcel_save_dir = create_directories(output_dir, n_parcel, group_info)

    for _, row in group_info.iterrows():
        process_subject(row, parcellation, n_parcel, parcel_save_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process parcel timeseries for subjects.')
    parser.add_argument('--n_parcel', type=int, required=True, help='Number of parcels')
    args = parser.parse_args()
    main(args.n_parcel)
