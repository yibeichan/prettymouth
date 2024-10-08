# this script is to run hyperalign on the roi level (hippocampus + mPFC)

import os
import numpy as np
import pandas as pd
import nibabel as nib
from dotenv import load_dotenv
from argparse import ArgumentParser

def get_parcel_data(atlas_data, labels):
    flat_atlas = atlas_data.flatten()
    return {label: np.unravel_index(np.where(flat_atlas == label)[0], atlas_data.shape) for label in labels}

def get_region_data(atlas_data, labels, subject_data):
    parcel_data = get_parcel_data(atlas_data, labels)
    return np.array([np.mean(subject_data[indices], axis=0) for indices in parcel_data.values()])

def get_group_data(atlas_data, labels, group_files):
    return np.array([get_region_data(atlas_data, labels, nib.load(file).get_fdata()) for file in group_files])

def main(atlas_data, cortical_labels, subcortical_labels, group, output_dir):
    group_ids = {
        "affair": ['sub-023', 'sub-032', 'sub-034', 'sub-050', 'sub-083', 'sub-084', 'sub-085', 'sub-086', 'sub-087', 'sub-088', 'sub-089', 'sub-090', 'sub-091', 'sub-092', 'sub-093', 'sub-094', 'sub-095', 'sub-096', 'sub-097'],
        "paranoia": ['sub-030', 'sub-052', 'sub-065', 'sub-066', 'sub-079', 'sub-081', 'sub-098', 'sub-099', 'sub-100', 'sub-101', 'sub-102', 'sub-103', 'sub-104', 'sub-106', 'sub-107', 'sub-108', 'sub-109', 'sub-110', 'sub-111']
    }

    if group not in group_ids:
        raise ValueError(f"Invalid group: {group}")
    
    group_files = [os.path.join(nese_dir, "output", "postproc_native", f"{sub_id}_cleaned_smoothed_masked_bold.nii.gz") for sub_id in group_ids[group]]
    mPFC_labels = cortical_labels[cortical_labels[1].str.contains("PFCm")].index.values + 1
    hipp_labels = subcortical_labels[subcortical_labels[0].str.contains("HIP")].index.values + 1 + np.max(cortical_labels.index.values)

    mPFC_group_data = get_group_data(atlas_data, mPFC_labels, group_files)
    hipp_group_data = get_group_data(atlas_data, hipp_labels, group_files)

    np.save(os.path.join(output_dir, f"{group}_mPFC_1000Parcels.npy"), mPFC_group_data)
    np.save(os.path.join(output_dir, f"{group}_hipp_1000Parcels.npy"), hipp_group_data)

if __name__ == "__main__":
    load_dotenv()

    parser = ArgumentParser()
    parser.add_argument("--group", type=str, required=True)
    args = parser.parse_args()

    nese_dir = os.getenv("NESE_DIR")
    output_dir = os.path.join(nese_dir, "output", "group_data")
    os.makedirs(output_dir, exist_ok=True)

    atlas_file = os.path.join(nese_dir, "data", "combined_parcellations", "combined_Schaefer2018_1000Parcels_Kong2022_17Networks_Tian_Subcortex_S4_3T_2009cAsym_native.nii.gz")
    atlas_data = nib.load(atlas_file).get_fdata()
    
    cortical_labels = pd.read_csv(os.path.join(nese_dir, "data", "combined_parcellations", "Schaefer2018_1000Parcels_Kong2022_17Networks_order.txt"), header=None, sep="\t")
    subcortical_labels = pd.read_csv(os.path.join(nese_dir, "data", "combined_parcellations", "Tian_Subcortex_S4_3T_label.txt"), header=None, sep="\t")

    main(atlas_data, cortical_labels, subcortical_labels, args.group, output_dir)