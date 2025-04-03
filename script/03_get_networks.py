import os
import numpy as np
import pandas as pd
import nibabel as nib
from dotenv import load_dotenv
import glob
from typing import Dict, List, Tuple

def load_parcel_data(subject_id: str, parcels_dir: str) -> Dict[int, np.ndarray]:
    """
    Load all parcel time series for a subject from the extracted parcels directory.
    
    Args:
        subject_id: Subject identifier
        parcels_dir: Directory containing extracted parcel data
        
    Returns:
        Dictionary mapping parcel_id to time series data
    """
    subject_dir = os.path.join(parcels_dir, subject_id)
    if not os.path.exists(subject_dir):
        raise FileNotFoundError(f"Subject directory not found: {subject_dir}")
    
    parcel_files = glob.glob(os.path.join(subject_dir, "parcel_*_timeseries.npy"))
    if not parcel_files:
        raise ValueError(f"No parcel data found for subject {subject_id}")
    
    parcel_data = {}
    timepoints = None
    
    for parcel_file in parcel_files:
        # Extract parcel ID from filename
        try:
            parcel_id = int(os.path.basename(parcel_file).split("_")[1])
        except (IndexError, ValueError):
            print(f"  Warning: Could not parse parcel ID from filename: {parcel_file}")
            continue
            
        # Skip parcel 0 (background) if it exists
        if parcel_id == 0:
            print(f"  Skipping background parcel (ID=0)")
            continue
        
        # Load the time series data
        time_series = np.load(parcel_file)
        
        # If the parcel contains multiple voxels, average them
        if time_series.ndim > 1 and time_series.shape[0] > 1:
            time_series = np.mean(time_series, axis=0)
        elif time_series.ndim > 1:
            time_series = time_series[0]
        
        # Check for consistent timepoints
        if timepoints is None:
            timepoints = len(time_series)
        elif len(time_series) != timepoints:
            print(f"  Warning: Inconsistent timepoints for parcel {parcel_id} ({len(time_series)} vs {timepoints})")
            
        # Skip empty or invalid timeseries
        if len(time_series) == 0:
            print(f"  Warning: Empty timeseries for parcel {parcel_id}")
            continue
            
        parcel_data[parcel_id] = time_series
    
    print(f"Loaded {len(parcel_data)} parcels for subject {subject_id}")
    return parcel_data

def get_network_parcels(cortical_labels_file: str) -> Dict[str, List[int]]:
    """
    Get the parcels that belong to each network.
    
    Args:
        cortical_labels_file: Path to the cortical labels file
        
    Returns:
        Dictionary mapping network names to lists of parcel IDs
    """
    # Load the cortical labels
    cortical_labels = pd.read_csv(cortical_labels_file, header=None, sep="\t")
    cortical_labels["network"] = cortical_labels[1].apply(lambda x: x.split("_")[2])
    
    # Get unique networks
    unique_networks = np.unique(cortical_labels["network"])
    
    # Map each network to its parcel IDs
    network_parcels = {}
    for network in unique_networks:
        network_indices = np.where(cortical_labels["network"] == network)[0]
        # Add 1 because parcels are 1-indexed in the atlas
        network_parcels[network] = list(network_indices + 1)
    
    return network_parcels

def extract_network_data(parcel_data: Dict[int, np.ndarray], 
                         network_parcels: Dict[str, List[int]]) -> Dict[str, np.ndarray]:
    """
    Organize parcel data by network.
    
    Args:
        parcel_data: Dictionary mapping parcel_id to time series
        network_parcels: Dictionary mapping network names to parcel IDs
        
    Returns:
        Dictionary mapping network names to arrays of time series
    """
    network_data = {}
    
    for network, parcel_ids in network_parcels.items():
        # Find which parcels in this network have data
        available_parcels = [p for p in parcel_ids if p in parcel_data]
        
        if not available_parcels:
            print(f"  Warning: No data found for network {network}")
            continue
            
        # Stack time series for all parcels in this network
        network_series = np.vstack([parcel_data[p] for p in available_parcels])
        network_data[network] = network_series
        
        print(f"  Network {network}: {network_series.shape[0]} parcels, {network_series.shape[1]} timepoints")
    
    return network_data

def main(subj_id: str, cortical_labels_file: str, 
         extract_parcels_dir: str, output_dir: str) -> None:
    """
    Process networks for a subject using extracted parcel data.
    
    Args:
        subj_id: Subject identifier
        cortical_labels_file: Path to cortical labels file
        extract_parcels_dir: Directory with extracted parcel data
        output_dir: Output directory
    """
    try:
        print(f"Processing {subj_id}")
        
        # Load all parcel data for this subject
        parcel_data = load_parcel_data(subj_id, extract_parcels_dir)
        
        # Get parcels belonging to each network
        network_parcels = get_network_parcels(cortical_labels_file)
        
        # Organize parcel data by network
        network_data = extract_network_data(parcel_data, network_parcels)
        
        
        for network, data in network_data.items():
            output_path = os.path.join(output_dir, f"{subj_id}_{network}_network_data.npy")
            np.save(output_path, data)
            print(f"  Saved network data to {output_path}")
            
    except Exception as e:
        print(f"Error processing subject {subj_id}: {str(e)}")
        raise

if __name__ == "__main__":
    load_dotenv()
    
    scratch_dir = os.getenv("SCRATCH_DIR")
    if scratch_dir is None:
        raise EnvironmentError("SCRATCH_DIR environment variable is not set.")
    
    affair_ids = os.getenv("AFFAIR_SUBJECTS").split(",")
    paranoia_ids = os.getenv("PARANOIA_SUBJECTS").split(",")
    all_subjects = affair_ids + paranoia_ids
    
    res = "native"  # or other resolution as needed
    
    # Set up paths
    parcellation_dir = os.path.join(scratch_dir, "data", "combined_parcellations")
    extract_parcels_dir = os.path.join(scratch_dir, "output", f"02_extract_parcels_{res}")
    output_dir = os.path.join(scratch_dir, "output", f"03_network_data_{res}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Path to cortical labels file
    cortical_labels_file = os.path.join(
        parcellation_dir, 
        f"Schaefer2018_1000Parcels_Kong2022_17Networks_order.txt"
    )
    
    # Process all subjects
    for subj_id in all_subjects:
        main(
            subj_id=subj_id,
            cortical_labels_file=cortical_labels_file,
            extract_parcels_dir=extract_parcels_dir,
            output_dir=output_dir
        )