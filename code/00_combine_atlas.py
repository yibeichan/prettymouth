import os
import nibabel as nib
import numpy as np
from nilearn.image import resample_to_img
from pathlib import Path
from dotenv import load_dotenv

def validate_inputs(img1_path, img2_path, output_path, image_ref_path=None):
    """Validate input files exist and output directory is writable."""
    for path in [img1_path, img2_path] + ([image_ref_path] if image_ref_path else []):
        if not Path(path).exists():
            raise FileNotFoundError(f"Input file not found: {path}")
    
    output_dir = Path(output_path).parent
    if not output_dir.exists():
        raise NotADirectoryError(f"Output directory does not exist: {output_dir}")

def combine_atlases_nifti(img1_path, img2_path, output_path, resample=False, image_ref_path=None):
    """
    Combine subcortical and cortical atlases.
    
    Parameters
    ----------
    img1_path : str
        Path to subcortical atlas
    img2_path : str
        Path to cortical atlas (Schaefer 400/1000)
    output_path : str
        Path for saving combined atlas
    resample : bool
        Whether to resample to match reference image
    image_ref_path : str, optional
        Path to reference image for resampling
    """
    # Validate inputs
    validate_inputs(img1_path, img2_path, output_path, image_ref_path)
    
    # Load NIfTI images
    img1_nifti = nib.load(img1_path)  # subcortical
    img2_nifti = nib.load(img2_path)  # cortical

    # If resampling to native space
    if resample and image_ref_path:
        image_ref_nifti = nib.load(image_ref_path)
        print(f"Resampling atlases to match reference: {image_ref_nifti.shape}")
        
        # Resample both img1 and img2 to match the reference image
        img1_nifti = resample_to_img(
            img1_nifti, 
            image_ref_nifti, 
            interpolation='nearest',
            force_resample=True
        )
        img2_nifti = resample_to_img(
            img2_nifti, 
            image_ref_nifti, 
            interpolation='nearest',
            force_resample=True
        )
    else:
        # For 2mm case, resample subcortical to match cortical
        print(f"Resampling subcortical atlas to match cortical space: {img2_nifti.shape}")
        img1_nifti = resample_to_img(
            img1_nifti, 
            img2_nifti,  # Use cortical atlas as reference
            interpolation='nearest',
            force_resample=True
        )

    # Get the image data as NumPy arrays
    img1_data = np.round(img1_nifti.get_fdata()).astype(int)  # Round and convert to integer
    img2_data = np.round(img2_nifti.get_fdata()).astype(int)  # Round and convert to integer
    
    # Create a copy of img2_data to be the combined image
    combined_data = np.copy(img2_data)
    
    # Get the expected number of cortical parcels from the filename
    n_cortical = int(os.path.basename(img2_path).split('Parcels')[0].split('_')[-1])
    
    # Add subcortical parcels with new numbering
    subcortical_mask = img1_data > 0
    unique_subcortical = np.unique(img1_data[subcortical_mask])
    n_subcortical = len(unique_subcortical)
    
    # Map subcortical values to follow cortical values
    for i, old_val in enumerate(unique_subcortical, 1):
        new_val = n_cortical + i
        combined_data[img1_data == old_val] = new_val
    
    # Create a new NIfTI image using the combined data
    combined_nifti = nib.Nifti1Image(combined_data, img2_nifti.affine, img2_nifti.header)
    
    # Save the combined NIfTI image
    nib.save(combined_nifti, output_path)
    
    # Print detailed summary
    print("\nAtlas Combination Summary:")
    print(f"Cortical parcels (1-{n_cortical}): {len(np.unique(img2_data[img2_data > 0]))}")
    print(f"Subcortical parcels ({n_cortical+1}-{n_cortical+n_subcortical}): {n_subcortical}")
    print(f"Total parcels: {len(np.unique(combined_data[combined_data > 0]))}")
    
    return combined_nifti

if __name__ == "__main__":
    load_dotenv()
    
    # Get environment variables with error checking
    required_env_vars = ["BASE_DIR", "SCRATCH_DIR"]
    env_vars = {var: os.getenv(var) for var in required_env_vars}
    
    if None in env_vars.values():
        missing = [k for k, v in env_vars.items() if v is None]
        raise EnvironmentError(f"Missing required environment variables: {missing}")
    
    base_dir = env_vars["BASE_DIR"]
    scratch_dir = env_vars["SCRATCH_DIR"]
    
    data_dir = os.path.join(scratch_dir, "data", "combined_parcellations")
    subcortical_atlas_path = os.path.join(data_dir, "Tian_Subcortex_S4_3T_2009cAsym.nii.gz")
    
    # Resolution handling
    res = "native"  # or "native"
    need_resample = res == "native"
    
    for n_parcel in [400, 1000]:
        print(f"\nProcessing {n_parcel} parcels version...")
        cortical_atlas_path = os.path.join(
            data_dir, 
            f"Schaefer2018_{n_parcel}Parcels_Kong2022_17Networks_order_FSLMNI152_2mm.nii.gz"
        )
        output_path = os.path.join(
            data_dir, 
            f"combined_Schaefer2018_{n_parcel}Parcels_Kong2022_17Networks_Tian_Subcortex_S4_3T_{res}.nii.gz"
        )
        
        image_ref_path = None
        if need_resample:
            image_ref_path = "/orcd/scratch/bcs/001/yibei/prettymouth_babs/prettymouth_fmriprep/prettymouth_output/sub-023_fmriprep-24-1-0/fmriprep/sub-023/func/sub-023_task-prettymouth_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
        
        try:
            combined_image = combine_atlases_nifti(
                subcortical_atlas_path,
                cortical_atlas_path,
                output_path,
                resample=need_resample,
                image_ref_path=image_ref_path
            )
            print(f"Combined NIfTI image saved to: {output_path}")
        except Exception as e:
            print(f"Error processing {n_parcel} parcels: {e}")
            continue