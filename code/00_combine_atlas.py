import os
import nibabel as nib
import numpy as np

from dotenv import load_dotenv

##################
# this script combines two atlases into one (i.e. Tian_Subcortex_S4_3T.nii and Schaefer2018_400Parcels_Kong2022_17Networks_order_FSLMNI152_2mm.nii.gz)
# for the schaefer parcellation we have both 400 and 1000 parcel versions
##################

import nibabel as nib
import numpy as np
from nilearn.image import resample_to_img

def combine_atlases_nifti(img1_path, img2_path, output_path, resample=False, image_ref_path=None):
    # Load NIfTI images
    img1_nifti = nib.load(img1_path)
    img2_nifti = nib.load(img2_path)

    # If resampling is requested and a reference image is provided
    if resample and image_ref_path:
        image_ref_nifti = nib.load(image_ref_path)
        print(f"Resampling both atlases to match the reference image: {image_ref_nifti.shape}")
        
        # Resample both img1 and img2 to match the reference image
        img1_nifti = resample_to_img(img1_nifti, image_ref_nifti, interpolation='nearest')
        img2_nifti = resample_to_img(img2_nifti, image_ref_nifti, interpolation='nearest')
    
    # Get the image data as NumPy arrays
    img1_data = np.round(img1_nifti.get_fdata()).astype(int)  # Round and convert to integer
    img2_data = np.round(img2_nifti.get_fdata()).astype(int)  # Round and convert to integer
    
    # Create a copy of img2_data to be the combined image
    combined_data = np.copy(img2_data)
    
    # Find the maximum value in img2_data
    max_value_img2 = np.max(img2_data)
    
    # Add max_value_img2 to non-zero values in img1_data
    img1_non_zero = img1_data > 0
    combined_data[img1_non_zero] = img1_data[img1_non_zero] + max_value_img2
    
    # Create a new NIfTI image using the combined data and the affine of img2 (or reference image if resampled)
    combined_nifti = nib.Nifti1Image(combined_data, img2_nifti.affine, img2_nifti.header)
    
    # Save the combined NIfTI image
    nib.save(combined_nifti, output_path)
    
    return combined_nifti

if __name__ == "__main__":
    load_dotenv()
    
    base_dir = os.getenv("BASE_DIR")
    scratch_dir = os.getenv("SCRATCH_DIR")
    nese_dir = os.getenv("NESE_DIR")
    data_dir = os.path.join(nese_dir, "data", "combined_parcellations")
    # subcortical_atlas_path = os.path.join(data_dir, "Tian_Subcortex_S4_3T.nii")
    subcortical_atlas_path = os.path.join(data_dir, "Tian_Subcortex_S4_3T_2009cAsym.nii.gz")
    n_parcels = [400, 1000]
    for n_parcel in n_parcels:
        cortical_atlas_path = os.path.join(data_dir, f"Schaefer2018_{n_parcel}Parcels_Kong2022_17Networks_order_FSLMNI152_2mm.nii.gz")
        # output_path = os.path.join(data_dir, f"combined_Schaefer2018_{n_parcel}Parcels_Kong2022_17Networks_Tian_Subcortex_S4_3T_2mm.nii.gz")
        output_path = os.path.join(data_dir, f"combined_Schaefer2018_{n_parcel}Parcels_Kong2022_17Networks_Tian_Subcortex_S4_3T_2009cAsym_original.nii.gz")
        image_ref_path = "/om2/scratch/tmp/yibei/prettymouth_babs/prettymouth_output/sub-023_fmriprep-24-1-0/fmriprep/sub-023/func/sub-023_task-prettymouth_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
        # combined_image = combine_atlases_nifti(subcortical_atlas_path, cortical_atlas_path, output_path)
        combined_image = combine_atlases_nifti(subcortical_atlas_path, cortical_atlas_path, output_path, resample=True, image_ref_path=image_ref_path)
        print(f"Combined NIfTI image saved to: {output_path}")
