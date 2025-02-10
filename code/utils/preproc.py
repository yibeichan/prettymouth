from typing import Dict, List, Tuple
import os
import glob
import numpy as np
import pandas as pd
from sklearn.utils import Bunch
import matplotlib.pyplot as plt
from scipy import signal
from datetime import datetime
import warnings

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

def plot_timeseries_diagnostics(masked_bold, sub_id, save_dir, tr=1.5):
    """Plot comprehensive diagnostics of BOLD timeseries.
    
    Args:
        masked_bold (np.ndarray): Shape (timepoints, voxels)
        sub_id (str): Subject ID for filename
        save_dir (str): Output directory
        tr (float): Repetition time in seconds
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'BOLD Diagnostics - Subject {sub_id}')
    
    # Power spectrum
    freqs, psd = signal.welch(masked_bold.mean(axis=1), fs=1/tr)
    axes[0,0].semilogy(freqs, psd)
    axes[0,0].set_xlabel('Frequency (Hz)')
    axes[0,0].set_ylabel('Power')
    axes[0,0].set_title('Power Spectrum')
    axes[0,0].grid(True)
    
    # Story-relevant frequency band power
    story_mask = (freqs >= 0.01) & (freqs <= 0.1)
    axes[0,0].fill_between(freqs[story_mask], psd[story_mask], alpha=0.3, color='r',
                          label='Story-relevant band (0.01-0.1 Hz)')
    axes[0,0].legend()
    
    # Mean timeseries
    axes[0,1].plot(np.arange(len(masked_bold))*tr, masked_bold.mean(axis=1))
    axes[0,1].set_xlabel('Time (s)')
    axes[0,1].set_ylabel('Mean BOLD')
    axes[0,1].set_title('Global Signal')
    axes[0,1].grid(True)
    
    # Variance across voxels
    axes[1,0].hist(np.std(masked_bold, axis=0), bins=50)
    axes[1,0].set_xlabel('Voxel Std Dev')
    axes[1,0].set_ylabel('Count')
    axes[1,0].set_title('Variance Distribution')
    
    # Temporal SNR
    tsnr = masked_bold.mean(axis=0) / masked_bold.std(axis=0)
    axes[1,1].hist(tsnr, bins=50)
    axes[1,1].set_xlabel('tSNR')
    axes[1,1].set_ylabel('Count')
    axes[1,1].set_title(f'Temporal SNR (median={np.median(tsnr):.1f})')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{sub_id}_bold_diagnostics.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'tsnr_median': np.median(tsnr),
        'tsnr_mean': np.mean(tsnr),
        'story_band_power_ratio': np.sum(psd[story_mask]) / np.sum(psd)
    }

def analyze_confounds(confounds_df):
    """Analyze confound regressors for quality assessment.
    
    Args:
        confounds_df (pd.DataFrame): Confounds from fMRIPrep
        
    Returns:
        dict: Motion and noise statistics
    """
    motion_stats = {}
    
    # Detailed motion analysis
    trans_cols = ['trans_x', 'trans_y', 'trans_z']
    rot_cols = ['rot_x', 'rot_y', 'rot_z']
    
    if all(col in confounds_df.columns for col in trans_cols + rot_cols):
        # Translation statistics (in mm)
        translations = confounds_df[trans_cols].values
        motion_stats.update({
            'mean_translation_x': float(confounds_df['trans_x'].abs().mean()),
            'mean_translation_y': float(confounds_df['trans_y'].abs().mean()),
            'mean_translation_z': float(confounds_df['trans_z'].abs().mean()),
            'max_translation': float(np.abs(translations).max()),
            'mean_translation': float(np.abs(translations).mean())
        })
        
        # Rotation statistics (in radians)
        rotations = confounds_df[rot_cols].values
        motion_stats.update({
            'mean_rotation_x': float(confounds_df['rot_x'].abs().mean()),
            'mean_rotation_y': float(confounds_df['rot_y'].abs().mean()),
            'mean_rotation_z': float(confounds_df['rot_z'].abs().mean()),
            'max_rotation': float(np.abs(rotations).max()),
            'mean_rotation': float(np.abs(rotations).mean())
        })
        
        # RMS of motion
        motion_rms = np.sqrt(np.mean(np.concatenate([translations, rotations], axis=1)**2, axis=1))
        motion_stats.update({
            'mean_motion_rms': float(motion_rms.mean()),
            'max_motion_rms': float(motion_rms.max())
        })
    
    # Framewise displacement
    if 'framewise_displacement' in confounds_df:
        fd = confounds_df['framewise_displacement'].fillna(0)
        motion_stats.update({
            'mean_fd': float(fd.mean()),
            'max_fd': float(fd.max()),
            'fd_over_0.5mm': int((fd > 0.5).sum()),
            'fd_over_0.2mm': int((fd > 0.2).sum()),
            'percent_fd_over_0.5mm': float((fd > 0.5).mean() * 100),
            'percent_fd_over_0.2mm': float((fd > 0.2).mean() * 100),
            'fd_percentile_95': float(np.percentile(fd, 95))
        })
    
    # Detailed aCompCor analysis
    acompcor_cols = [col for col in confounds_df.columns 
                     if col.startswith('a_comp_cor_')]
    if acompcor_cols:
        acompcor = confounds_df[acompcor_cols].values
        
        # Individual component variance
        var_explained = np.var(acompcor, axis=0)
        total_var = np.sum(var_explained)
        
        # Store first 5 components
        for i, var in enumerate(var_explained[:5]):
            motion_stats[f'acompcor_{i+1}_variance'] = float(var)
            motion_stats[f'acompcor_{i+1}_variance_percent'] = float((var / total_var) * 100)
        
        # Correlation between aCompCor and motion
        motion = confounds_df[trans_cols + rot_cols].values
        correlations = np.zeros((len(acompcor_cols[:5]), 6))
        for i, acomp in enumerate(acompcor_cols[:5]):
            for j, mcol in enumerate(trans_cols + rot_cols):
                correlations[i,j] = np.corrcoef(confounds_df[acomp], confounds_df[mcol])[0,1]
        
        motion_stats['max_acompcor_motion_correlation'] = float(np.abs(correlations).max())
        motion_stats['mean_acompcor_motion_correlation'] = float(np.abs(correlations).mean())
        
        # First few timepoints for inspection
        print("\naCompCor first 5 timepoints:")
        print(confounds_df[acompcor_cols[:5]].head())
    
    return motion_stats

def check_data_properties(masked_bold):
    """Verify data properties after preprocessing."""
    return {
        'n_timepoints': int(masked_bold.shape[0]),  # ensure int
        'n_voxels': int(masked_bold.shape[1]),      # ensure int
        'percent_nan': float(np.float64(np.isnan(masked_bold).mean() * 100)),  # convert to native float
        'percent_zero': float(np.float64((masked_bold == 0).mean() * 100)),    # convert to native float
        'volumes_with_high_nan': int(np.sum(np.isnan(masked_bold).mean(axis=1) > 0.01)),
        'volumes_with_high_zero': int(np.sum((masked_bold == 0).mean(axis=1) > 0.01))
    }

def load_task_confounds(sub_id, task_file):
    """Load and validate confound regressors from fMRIPrep.
    
    Args:
        sub_id (str): Subject ID
        task_file (str): Path to BOLD data
        
    Returns:
        tuple: (confounds DataFrame, QC metrics dict)
    """
    task = os.path.basename(task_file).split("_task-")[1].split("_")[0]
    confound_file = os.path.join(os.path.dirname(task_file), 
                                f"{sub_id}_task-{task}_desc-confounds_timeseries.tsv")
    
    if not os.path.exists(confound_file):
        raise FileNotFoundError(f"Confounds file not found: {confound_file}")
    
    # Load confounds
    confounds_df = pd.read_csv(confound_file, sep='\t')
    
    # Get confound QC metrics
    confound_qc = analyze_confounds(confounds_df)
    
    # Select minimal confound set for HMM
    basic_motion_names = ['trans_x', 'trans_y', 'trans_z', 
                         'rot_x', 'rot_y', 'rot_z']
    motion_cols = [col for col in confounds_df.columns 
                  if col in basic_motion_names]
    
    if len(motion_cols) != 6:
        raise ValueError(f"Expected 6 motion parameters, found {len(motion_cols)}")
    print(f"Selected motion parameters: {motion_cols}")
    
    # Get aCompCor components (first 5)
    acompcor_cols = [col for col in confounds_df.columns 
                     if col.startswith('a_comp_cor_')][:5]
    if len(acompcor_cols) < 5:
        warnings.warn(f"Found fewer than 5 aCompCor components: {len(acompcor_cols)}")
    
    # Get cosine regressors for high-pass filtering
    cosine_cols = [col for col in confounds_df.columns 
                   if 'cosine' in col]
    
    # Combine all confounds
    selected_cols = motion_cols + acompcor_cols + cosine_cols
    selected_confounds = confounds_df[selected_cols]
    
    # Handle missing values
    if selected_confounds.isnull().any().any():
        warnings.warn("Found NaN values in confounds. Will be replaced with 0.")
        selected_confounds = selected_confounds.fillna(0)
    
    return selected_confounds, confound_qc