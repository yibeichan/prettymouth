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


def match_grayordinates(parcellation, substrings):
    """
    Returns the grayordinate indices in 'map_all' that correspond to the labels matching the given patterns.
    
    - If 'substrings' is a single string or a list with one element, performs an exact match.
    - If 'substrings' is a list with multiple elements, matches labels containing all substrings.
    
    Parameters:
    - parcellation: dict
        A dictionary with the following keys:
            - 'labels': list or array of label strings.
            - 'map_all': array of label indices corresponding to each grayordinate.
    - substrings: str or list of str
        A string or list of strings to search for within the labels.
    
    Returns:
    - matching_grayordinates: numpy.ndarray
        Indices of the grayordinates that correspond to the matching labels.
    """
    
    labels = parcellation['labels']
    map_all = parcellation['map_all']
    
    # Ensure 'substrings' is a list for uniform processing
    if isinstance(substrings, str):
        substrings = [substrings]
    elif isinstance(substrings, (list, tuple, np.ndarray)):
        substrings = list(substrings)
    else:
        raise ValueError("substrings must be a string or a list of strings.")
    
    if len(substrings) == 1:
        # Exact match when only one substring is provided
        target_label = substrings[0]
        mask = np.array(labels) == target_label
    else:
        # Substring match: label must contain all substrings
        mask = np.ones(len(labels), dtype=bool)
        for substring in substrings:
            # Update mask to retain labels containing the current substring
            mask &= np.char.find(labels, substring) != -1
    
    # Get the indices of labels that match the criteria
    matched_label_indices = np.where(mask)[0]
    
    # Find grayordinates in 'map_all' that correspond to the matched label indices
    # 'map_all' contains label indices for each grayordinate
    matching_grayordinates = np.where(np.isin(map_all, matched_label_indices))[0]
    
    return matching_grayordinates

def get_roi(group_data, parcellation, strings):
    roi_index = match_grayordinates(parcellation, strings)
    return group_data[:, :, roi_index], roi_index