from typing import Dict, List, Tuple
import os
import glob
import numpy as np
import pandas as pd
from sklearn.utils import Bunch

def load_parcellation(np_path):
    yan_2023 = np.load(np_path)
    parcellation = Bunch()
    parcellation.ids = yan_2023['ids']
    parcellation.map_all = yan_2023['map_all']

    labels = yan_2023['labels']
    labelsdict = dict()
    rgba = yan_2023['rgba']
    rgbadict = dict()
    for i, k in enumerate(parcellation.ids):
        labelsdict[k] = labels[i]
        rgbadict[k] = rgba[i]

    parcellation.labels = labelsdict
    parcellation.rgba = rgbadict

    i = 0
    nontrivial_ids = []
    for k in parcellation.ids:
        if k!=0:
            nontrivial_ids.append(k)
            i += 1
    parcellation.nontrivial_ids = np.array(nontrivial_ids)
    return parcellation

def get_single_parcel(output_dir: str, n_parcel: int, group_id: str, p_id: int) -> np.ndarray:
    """
    Get a single parcel data.

    Args:
        output_dir (str): The directory where the data is stored.
        n_parcel (int): The number of the parcel.
        group_id (str): The group ID.
        p_id (int): The parcel ID.

    Returns:
        np.ndarray: The data array with shape (n_subject, n_timepoints, n_voxels).
    """
    # Create the file pattern
    file_pattern = os.path.join(output_dir, 'masked_parcels', f'{n_parcel}parcel', f'{group_id}', f'*_p{p_id+1:03d}.npy')

    # Get the list of files matching the pattern
    files = glob.glob(file_pattern)
    files.sort()

    # Load the data from the files and stack them along a new dimension
    data = np.stack([np.load(file) for file in files])

    return data

def select_parcels(parcel_indices_dict: Dict[str, List[int]], group_id: str, output_dir: str, n_parcel: int) -> None:
    """
    Selects and saves specific parcels from a masked parcels directory.

    Args:
    - parcel_indices_dict (Dict[str, List[int]]): A dictionary containing the group IDs as keys and a list of parcel indices as values.
    - group_id (str): The ID of the group for which parcels need to be selected.
    - output_dir (str): The directory where the selected parcels will be saved.
    - n_parcel (int): The total number of parcels.

    Returns:
    None. The function saves the selected parcels as numpy arrays in the specified directory.
    """
    for k, v in parcel_indices_dict.items():
        print(f'Processing {k} parcels {v}')
        parcel_data = [get_single_parcel(output_dir, n_parcel, group_id, i) for i in v]
        parcels = np.concatenate(parcel_data, axis=2)
        print(f'Parcels {k} shape: {parcels.shape}')
        save_dir = os.path.join(output_dir, 'selected_parcels', f'{n_parcel}parcel', f'{group_id}')
        os.makedirs(save_dir, exist_ok=True)
        file = os.path.join(save_dir, f'{k}_{len(v)}.npy')
        np.save(file, parcels)
        print(f'Saved {k} parcels for {group_id} group')

def get_roi_and_network_ids(parcellation: Dict, n_parcel: int) -> Tuple[Dict, Dict, Dict, Dict]:
    # Extract labels from parcellation data
    labels = parcellation["labels"]

    # Create a DataFrame with labels excluding 0 and labels > n_parcel
    parcel_df = pd.DataFrame(labels[1:n_parcel + 1])

    def safe_split(label, idx, default="Unknown"):
        parts = str(label).split("_")
        return parts[idx] if idx < len(parts) else default

    # Get unique ROI names, handling cases with different label lengths
    roinames = np.unique([safe_split(i, 2) for i in parcel_df[0].values])

    # Get unique ROI names separated by networks
    roinames_ntw = np.unique(["_".join([safe_split(i, 1), safe_split(i, 2)]) for i in parcel_df[0].values])

    # Get unique ROI names separated by hemispheres
    roinames_hem = np.unique(["_".join([safe_split(i, 0), safe_split(i, 1), safe_split(i, 2)]) for i in parcel_df[0].values])

    # Get unique network names
    networknames = np.unique([safe_split(i, 1) for i in parcel_df[0].values])

    # Create dictionaries mapping ROI and network names to their indices
    roinames_idx_dict = {roiname: np.where([roiname in i for i in parcel_df[0].values])[0] for roiname in roinames}
    roinames_ntw_idx_dict = {roiname_ntw: np.where([roiname_ntw in i for i in parcel_df[0].values])[0] for roiname_ntw in roinames_ntw}
    roinames_hem_idx_dict = {roiname_hem: np.where([roiname_hem in i for i in parcel_df[0].values])[0] for roiname_hem in roinames_hem}
    networknames_idx_dict = {networkname: np.where([networkname in i for i in parcel_df[0].values])[0] for networkname in networknames}

    # Print the number of ROIs, ROIs separated by networks, ROIs separated by hemispheres, and networks
    print(f"Number of ROIs: {len(roinames_idx_dict)}")
    print(f"Number of ROIs separated by networks: {len(roinames_ntw_idx_dict)}")
    print(f"Number of ROIs separated by hemispheres: {len(roinames_hem_idx_dict)}")
    print(f"Number of networks: {len(networknames_idx_dict)}")

    return roinames_idx_dict, roinames_ntw_idx_dict, roinames_hem_idx_dict, networknames_idx_dict