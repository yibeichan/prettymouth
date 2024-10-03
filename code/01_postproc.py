import os
import glob
import numpy as np
import nibabel as nib
from nilearn.interfaces.fmriprep import load_confounds
from nilearn.maskers import NiftiMasker
from dotenv import load_dotenv
from argparse import ArgumentParser
from memory_profiler import profile

# @profile
def main(bold_file, save_dir, smoothing_fwhm=6):
    """
    this script processes the native-space BOLD data
    """
    # get subject id
    subject_id = os.path.basename(bold_file).split("_task")[0]
    
    # Load confounds (assuming load_confounds_strategy is defined elsewhere)
    confounds, _ = load_confounds(bold_file, strategy = ("motion", "high_pass", "wm_csf", "compcor"))
    
    # Load BOLD data
    brain_data = nib.load(bold_file)
    
    # Initialize NiftiMasker with smoothing
    masker = NiftiMasker(mask_img=None, 
                         detrend=True, 
                         standardize="zscore_sample", 
                         smoothing_fwhm=smoothing_fwhm,
                         target_affine=brain_data.affine, 
                         target_shape=brain_data.shape[0:3], 
                         verbose=1)
    
    # Apply the masker (masking, denoising, and smoothing)
    masked_bold = masker.fit_transform(bold_file, confounds=confounds)
    print(f"Masked bold shape: {masked_bold.shape}")
    
    img = masker.inverse_transform(masked_bold)

    np.save(os.path.join(save_dir, f"{subject_id}_masked_bold.npy"), masked_bold)
    print(f"Saved masked bold for {subject_id}")
    
    # Save masked and smoothed BOLD image as NIfTI file
    img.to_filename(os.path.join(save_dir, f"{subject_id}_masked_bold.nii.gz"))
    print(f"Saved masked bold img for {subject_id}")

if __name__ == "__main__":
    load_dotenv()
    parser = ArgumentParser(description="postproc")
    parser.add_argument("sub_id", help="subject id", type=str)
    args = parser.parse_args()
    sub_id = args.sub_id

    scratch_dir = os.getenv("SCRATCH_DIR")
    nese_dir = os.getenv("NESE_DIR")
    data_dir = os.path.join(nese_dir, "data")
    parcellation_dir = os.path.join(data_dir, "combined_parcellations")
    output_dir = os.path.join(nese_dir, "output")
    
    res = "2mm"
    save_dir = os.path.join(output_dir, f"postproc_{res}")
    os.makedirs(save_dir, exist_ok=True)

    fmriprep_data = "/om2/scratch/tmp/yibei/prettymouth_babs/prettymouth_output/"
    if res == "native":
        task_file = os.path.join(fmriprep_data, f"{sub_id}_fmriprep-24-1-0/fmriprep/{sub_id}/func/{sub_id}_task-prettymouth_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")
    elif res == "2mm":
        task_file = os.path.join(fmriprep_data, f"{sub_id}_fmriprep-24-1-0/fmriprep/{sub_id}/func/{sub_id}_task-prettymouth_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz")
    else:
        print(f"Resolution {res} not recognized.")
        exit(1)
    # chech whether the file exists
    if not os.path.exists(task_file):
        print(f"File {task_file} does not exist.")
        exit(1)
    main(task_file, save_dir)