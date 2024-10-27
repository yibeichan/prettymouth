import os
import glob
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.maskers import NiftiMasker
from dotenv import load_dotenv
from argparse import ArgumentParser

def load_task_confounds(sub_id, task_file):
    """Load and extract relevant confound regressors from fMRIPrep confounds file.
    
    Args:
        sub_id (str): Subject ID
        task_file (str): Path to the task file
        
    Returns:
        pd.DataFrame: DataFrame containing selected confound regressors
        
    Raises:
        FileNotFoundError: If confounds file cannot be found
        ValueError: If required confound columns are missing
    """
    task = os.path.basename(task_file).split("_task-")[1].split("_")[0]
    confound_file = os.path.join(os.path.dirname(task_file), 
                                f"{sub_id}_task-{task}_desc-confounds_timeseries.tsv")
    # Load confounds data
    confounds_df = pd.read_csv(confound_file, sep='\t')
    
    # Select relevant columns
    compcor_cols = confounds_df.filter(regex='a_comp_cor_').columns
    wm_csf_cols = confounds_df.filter(regex='csf_|white_matter_').columns
    motion_cols = confounds_df.filter(regex='motion_|trans_|rot_').columns
    cosine_cols = confounds_df.filter(regex='cosine').columns
    
    # Verify we found some columns
    if len(compcor_cols) + len(wm_csf_cols) + len(motion_cols) + len(cosine_cols) == 0:
        raise ValueError("No matching confound columns found in confounds file")
    
    # Return selected columns
    return confounds_df[list(compcor_cols) + list(wm_csf_cols) + list(motion_cols) + list(cosine_cols)]

def main(sub_id, task_file, save_dir, smoothing_fwhm=6):
    """
    This script processes the native-space BOLD data using an atlas-based binary mask.
    """
    confounds_df = load_task_confounds(sub_id, task_file)
    confounds = confounds_df.to_numpy()
    
    # Before the signal.clean() call
    if np.any(np.isnan(confounds)) or np.any(np.isinf(confounds)):
        print("Warning: Found NaN or inf values in confounds")
    confounds = np.nan_to_num(confounds)

    mask_file = task_file.replace("preproc_bold.nii.gz", "brain_mask.nii.gz")
    # Load BOLD data
    brain_img = nib.load(task_file)
    mask_img = nib.load(mask_file)

    masker = NiftiMasker(mask_img=mask_img,  
                         detrend=True, 
                         standardize="zscore_sample", 
                         smoothing_fwhm=smoothing_fwhm,
                         target_affine=brain_img.affine, 
                         target_shape=brain_img.shape[0:3], 
                         verbose=1)

    # Apply the masker (masking, denoising, and smoothing)
    masked_bold = masker.fit_transform(task_file, confounds=confounds)
    print(f"Masked bold shape: {masked_bold.shape}")

    img = masker.inverse_transform(masked_bold)

    # Save masked BOLD data
    # np.save(os.path.join(save_dir, f"{subject_id}_cleaned_smoothed_masked_bold.npy"), masked_bold)
    # print(f"Saved masked bold for {subject_id}")

    # Save masked and smoothed BOLD image as NIfTI file
    img.to_filename(os.path.join(save_dir, f"{sub_id}_cleaned_smoothed_masked_bold.nii.gz"))
    print(f"Saved masked bold img for {sub_id}")

if __name__ == "__main__":
    load_dotenv()
    parser = ArgumentParser(description="postproc")
    parser.add_argument("sub_id", help="subject id", type=str)
    args = parser.parse_args()
    sub_id = args.sub_id

    scratch_dir = os.getenv("SCRATCH_DIR")
    data_dir = os.path.join(scratch_dir, "data")
    parcellation_dir = os.path.join(data_dir, "combined_parcellations")
    output_dir = os.path.join(scratch_dir, "output")
    
    res = "2mm"
    save_dir = os.path.join(output_dir, f"postproc_{res}")
    os.makedirs(save_dir, exist_ok=True)

    fmriprep_data = "/orcd/scratch/bcs/001/yibei/prettymouth_babs/prettymouth_fmriprep/prettymouth_output/"
    if res == "native":
        # atlas_path = os.path.join(parcellation_dir, "combined_Schaefer2018_1000Parcels_Kong2022_17Networks_Tian_Subcortex_S4_3T_2009cAsym_native.nii.gz")
        task_file = os.path.join(fmriprep_data, f"{sub_id}_fmriprep-24-1-0/fmriprep/{sub_id}/func/{sub_id}_task-prettymouth_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")
    elif res == "2mm":
        # atlas_path = os.path.join(parcellation_dir, "combined_Schaefer2018_1000Parcels_Kong2022_17Networks_Tian_Subcortex_S4_3T_2mm.nii.gz")
        task_file = os.path.join(fmriprep_data, f"{sub_id}_fmriprep-24-1-0/fmriprep/{sub_id}/func/{sub_id}_task-prettymouth_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz")
    else:
        print(f"Resolution {res} not recognized.")
        exit(1)
    # chech whether the file exists
    if not os.path.exists(task_file):
        print(f"File {task_file} does not exist.")
        exit(1)
    main(sub_id, task_file, save_dir)