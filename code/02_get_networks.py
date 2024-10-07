import os
import numpy as np
import pandas as pd
import nibabel as nib
from dotenv import load_dotenv

def get_network_indices(atlas_data, network_indices):
    # Flatten the 3D atlas data
    flat_atlas = atlas_data.flatten()
    # Use np.isin to create a mask of voxels belonging to the network
    network_mask = np.isin(flat_atlas, network_indices)
    # Use np.where to get the indices of True values in the mask
    network_indices = np.where(network_mask)[0]
    # Convert flat indices back to 3D coordinates
    return np.unravel_index(network_indices, atlas_data.shape)

def extract_network_data(subject_data, atlas_data, network_coords):
    network_data = {}
    
    for network, coords in network_coords.items():
        # Extract data for this network
        network_voxels = subject_data[coords]
        
        # Get the unique parcels in this network
        unique_parcels = np.unique(atlas_data[coords])
        unique_parcels = unique_parcels[unique_parcels != 0]  # Remove 0 if it's there
        
        # Initialize array to store parcel data
        parcel_data = np.zeros((len(unique_parcels), subject_data.shape[-1]))
        
        # Extract data for each parcel
        for i, parcel in enumerate(unique_parcels):
            parcel_mask = atlas_data[coords] == parcel
            parcel_data[i] = np.mean(network_voxels[parcel_mask], axis=0)
        
        network_data[network] = parcel_data
    
    return network_data

def main(subj_id, n_parcels, combined_atlas_file, cortical_labels_file, output_dir):
    subject_file = os.path.join(output_dir, "postproc_native", f"{subj_id}_cleaned_smoothed_masked_bold.nii.gz")
    # Load the subject data
    subject_img = nib.load(subject_file)
    subject_data = subject_img.get_fdata()
    
    # Load the combined atlas data
    atlas_img = nib.load(combined_atlas_file)
    atlas_data = atlas_img.get_fdata()
    
    # Load the cortical labels
    cortical_labels = pd.read_csv(cortical_labels_file, header=None, sep="\t")
    cortical_labels["network"] = cortical_labels[1].apply(lambda x: x.split("_")[2])

    # get network indices
    unique_networks = np.unique(cortical_labels["network"])
    network_indices = {network: np.where(cortical_labels["network"] == network)[0]+1 for network in unique_networks}
    network_indices["Subcortical"] = np.arange(n_parcels+1, int(np.max(atlas_data))+1, dtype=int)

    network_coords = {}
    for network, indices in network_indices.items():
        network_coords[network] = get_network_indices(atlas_data, indices)

    network_data = extract_network_data(subject_data, atlas_data, network_coords)

    save_dir = os.path.join(output_dir, "network_data")
    os.makedirs(save_dir, exist_ok=True)

    for network, data in network_data.items():
        np.save(os.path.join(save_dir, f"{subj_id}_{network}_{n_parcels}Parcels.npy"), data)

if __name__ == "__main__":
    load_dotenv()

    nese_dir = os.getenv("NESE_DIR")
    if nese_dir is None:
        raise EnvironmentError("NESE_DIR environment variable is not set.")

    output_dir = os.path.join(nese_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    subject_ids = ["sub-023", "sub-030", "sub-032", "sub-034", "sub-038", "sub-050", "sub-052", "sub-065", "sub-066", "sub-079", "sub-081", "sub-083", "sub-084", "sub-085", "sub-086", "sub-087", "sub-088", "sub-089", "sub-090", "sub-091", "sub-092", "sub-093", "sub-094", "sub-095", "sub-096", "sub-097", "sub-098", "sub-099", "sub-100", "sub-101", "sub-102", "sub-103", "sub-104", "sub-105", "sub-106", "sub-107", "sub-108", "sub-109", "sub-110", "sub-111"]
    # remove sub-038, sub-105
    for subj_id in subject_ids:
        print(f"Processing {subj_id}")
        if subj_id == "sub-038" or subj_id == "sub-105":
            continue
        n_parcels = 1000
        combined_atlas_file = os.path.join(nese_dir, "data", "combined_parcellations", f"combined_Schaefer2018_{n_parcels}Parcels_Kong2022_17Networks_Tian_Subcortex_S4_3T_2009cAsym_native.nii.gz")
        cortical_labels_file = os.path.join(nese_dir, "data", "combined_parcellations", f"Schaefer2018_{n_parcels}Parcels_Kong2022_17Networks_order.txt")

        main(subj_id, n_parcels, combined_atlas_file, cortical_labels_file, output_dir)