import os
import json
import numpy as np
import nibabel as nib
from nilearn.maskers import NiftiMasker
from dotenv import load_dotenv
from argparse import ArgumentParser
from datetime import datetime
from utils.preproc import (plot_timeseries_diagnostics,
                          check_data_properties, load_task_confounds)

def convert_to_native_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64, np.float16)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_native_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_native_types(item) for item in obj)
    return obj

def main(sub_id, task_file, save_dir, smoothing_fwhm=6):
    """Process BOLD data optimized for story listening HMM analysis.
    
    Key preprocessing choices:
    1. Smoothing: 6mm FWHM - balances noise reduction and network pattern preservation
    2. Detrending: Remove scanner drift
    3. Z-scoring: Normalize for HMM state detection
    4. Motion + aCompCor: Remove noise while preserving global story responses
    
    Args:
        sub_id (str): Subject ID
        task_file (str): Path to BOLD data
        save_dir (str): Output directory
        smoothing_fwhm (float): Smoothing kernel size in mm
    """
    print(f"\nProcessing subject: {sub_id}")
    start_time = datetime.now()
    
    # Load and check data
    print("Loading data...")
    brain_img = nib.load(task_file)
    mask_file = task_file.replace("preproc_bold.nii.gz", "brain_mask.nii.gz")
    mask_img = nib.load(mask_file)
    
    # Check image dimensions
    voxel_size = np.prod(brain_img.header.get_zooms()[:3])
    print(f"Original voxel volume: {voxel_size:.2f} mm³")
    print(f"Smoothing FWHM: {smoothing_fwhm} mm")
    
    # Load and check confounds
    confounds_df, confound_qc = load_task_confounds(sub_id, task_file)
    
    print("\nDetailed QC Metrics:")
    print("\n1. Motion Parameters:")
    for key, value in confound_qc.items():
        if isinstance(value, (int, float)):
            print(f"{key}: {value:.3f}")
    
    # Set up preprocessing
    masker = NiftiMasker(
        mask_img=mask_img,
        detrend=True,  # Remove linear trends
        standardize="zscore_sample",  # Normalize for HMM
        smoothing_fwhm=smoothing_fwhm,  # Balance noise reduction & spatial info
        t_r=1.5,  # From acquisition params
        target_affine=brain_img.affine,
        target_shape=brain_img.shape[0:3],
        verbose=1
    )
    
    # Apply preprocessing
    print("\nApplying preprocessing...")
    masked_bold = masker.fit_transform(task_file, confounds=confounds_df)
    print(f"Preprocessed data shape: {masked_bold.shape}")
    
    # Check data properties
    data_props = check_data_properties(masked_bold)
    
    # Generate QC plots and get metrics
    bold_qc = plot_timeseries_diagnostics(
        masked_bold,
        sub_id,
        save_dir,
        tr=1.5
    )
    
    # Save outputs
    output_img = masker.inverse_transform(masked_bold)
    output_path = os.path.join(save_dir, f"{sub_id}_preprocessed_bold.nii.gz")
    output_img.to_filename(output_path)
    print(f"\nSaved preprocessed data: {output_path}")
    
    # Save preprocessing report
    end_time = datetime.now()
    report = {
        'subject_id': sub_id,
        'processing_time': str(end_time - start_time),
        'timestamp': end_time.strftime('%Y-%m-%d %H:%M:%S'),
        'input': {
            'original_voxel_size': brain_img.header.get_zooms()[:3],
            'original_shape': brain_img.shape,
            'original_tr': 1.5
        },
        'preprocessing': {
            'smoothing_fwhm': smoothing_fwhm,
            'detrend': True,
            'standardization': 'zscore_sample',
            'confounds_used': list(confounds_df.columns)
        },
        'qc_metrics': {
            'motion': confound_qc,
            'bold': bold_qc,
            'data_properties': data_props
        }
    }
    
    # Before saving the report, ensure all nested structures are converted
    report = convert_to_native_types(report)
    
    # Double-check conversion of QC metrics specifically
    report['qc_metrics'] = convert_to_native_types(report['qc_metrics'])
    
    report_path = os.path.join(save_dir, f"{sub_id}_preprocessing_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Saved preprocessing report: {report_path}")

if __name__ == "__main__":
    load_dotenv()
    parser = ArgumentParser(description="fMRI preprocessing for story listening HMM")
    parser.add_argument("sub_id", help="subject id", type=str)
    parser.add_argument("--res", type=str, default="native",
                      help="Resolution of the atlas")
    parser.add_argument("--smoothing", type=float, default=6.0,
                      help="FWHM of smoothing kernel in mm")
    args = parser.parse_args()

    # Setup directories
    scratch_dir = os.getenv("SCRATCH_DIR")
    data_dir = os.path.join(scratch_dir, "data")
    output_dir = os.path.join(scratch_dir, "output_RR")
    save_dir = os.path.join(output_dir, f"01_postproc_{args.res}")
    os.makedirs(save_dir, exist_ok=True)

    # Construct file path
    fmriprep_data = "/orcd/scratch/bcs/001/yibei/prettymouth_babs/prettymouth_fmriprep/prettymouth_output/"
    task_file = os.path.join(
        fmriprep_data,
        f"{args.sub_id}_fmriprep-24-1-0/fmriprep/{args.sub_id}/func/"
        f"{args.sub_id}_task-prettymouth_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    )

    if not os.path.exists(task_file):
        raise FileNotFoundError(f"BOLD file not found: {task_file}")

    main(args.sub_id, task_file, save_dir, smoothing_fwhm=args.smoothing)