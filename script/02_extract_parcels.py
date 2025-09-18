import os
import nibabel as nib
import numpy as np
from dotenv import load_dotenv
from typing import List
import multiprocessing as mp
from functools import partial

def process_subject(sub_id: str, atlas_data: np.ndarray, atlas_affine: np.ndarray, data_dir: str, output_dir: str) -> None:
    """Process a single subject's data"""
    input_path = os.path.join(data_dir, f"{sub_id}_preprocessed_bold.nii.gz")
    # Create subject-specific output directory
    sub_output_dir = os.path.join(output_dir, sub_id)
    os.makedirs(sub_output_dir, exist_ok=True)
    
    try:
        # Load subject data
        img = nib.load(input_path)
        data = img.get_fdata()
        
        # Check spatial dimensions
        if data.shape[:3] != atlas_data.shape:
            raise ValueError(f"Shape mismatch: Atlas shape {atlas_data.shape}, Subject data shape {data.shape[:3]}")
        
        # Check spatial alignment (affine transformation matrices)
        if not np.allclose(img.affine, atlas_affine, atol=1e-3):
            error_message = (
                f"ERROR: Affine matrices don't match for {sub_id}\n"
                f"Atlas affine:\n{atlas_affine}\n"
                f"Subject affine:\n{img.affine}\n"
                "Data and atlas are not properly aligned! Processing halted."
            )
            raise ValueError(error_message)
        
        # Get unique parcel IDs (excluding 0 which is typically background)
        parcel_ids = np.unique(atlas_data)
        parcel_ids = parcel_ids[parcel_ids != 0]
        
        # Extract time series for each parcel
        for parcel_id in parcel_ids:
            parcel_mask = atlas_data == parcel_id
            parcel_data = data[parcel_mask, :]
            # parcel_timeseries = np.mean(parcel_data, axis=0)
            print(f"{sub_id} parcel_{int(parcel_id):04d}.shape", parcel_data.shape)
            
            output_path = os.path.join(sub_output_dir, f"parcel_{int(parcel_id):04d}_timeseries.npy")
            np.save(output_path, parcel_data)
        
        print(f"Successfully processed subject {sub_id}")
    except Exception as e:
        print(f"Error processing subject {sub_id}: {str(e)}")

def main(subject_ids: List[str], atlas_path: str, data_dir: str, output_dir: str) -> None:
    try:
        # Load atlas once
        atlas_img = nib.load(atlas_path)
        atlas_data = atlas_img.get_fdata()
        atlas_affine = atlas_img.affine
        print(f"Atlas data shape: {atlas_data.shape}")
        
        # Set up parallel processing
        num_cores = min(mp.cpu_count()-1, 8)  # More adaptive core usage
        print(f"Using {num_cores} cores for parallel processing")
        
        # Create partial function with fixed arguments
        process_subject_partial = partial(
            process_subject,
            atlas_data=atlas_data,
            atlas_affine=atlas_affine,  # Pass affine matrix
            data_dir=data_dir,
            output_dir=output_dir
        )
        
        # Process subjects in parallel
        with mp.Pool(num_cores) as pool:
            pool.map(process_subject_partial, subject_ids)
                
    except Exception as e:
        print(f"Error loading atlas: {str(e)}")
        return

if __name__ == "__main__":
    load_dotenv()
    scratch_dir = os.getenv("SCRATCH_DIR")
    affair_ids = os.getenv("AFFAIR_SUBJECTS").split(",")
    paranoia_ids = os.getenv("PARANOIA_SUBJECTS").split(",")
    parcellation_dir = os.path.join(scratch_dir, "data", "combined_parcellations")

    res = "native"
    atlas_name = f"combined_Schaefer2018_1000Parcels_Kong2022_17Networks_Tian_Subcortex_S4_3T_{res}.nii.gz"
    atlas_path = os.path.join(parcellation_dir, atlas_name)

    data_dir = os.path.join(scratch_dir, "output_RR", f"01_postproc_{res}")
    output_dir = os.path.join(scratch_dir, "output_RR", f"02_extract_parcels_{res}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Combine subject lists
    all_subjects = affair_ids + paranoia_ids
    
    main(all_subjects, atlas_path, data_dir, output_dir)