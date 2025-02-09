#!/usr/bin/env python3
"""
Atlas Masking Script for fMRI Data Processing
This script applies atlas-based masking to preprocessed fMRI data.
It handles multiple subjects and provides detailed logging and error handling.
"""

import os
import sys
import logging
from typing import List, Optional, Tuple
import nibabel as nib
import numpy as np
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('atlas_masking.log')
    ]
)
logger = logging.getLogger(__name__)

def load_and_validate_atlas(atlas_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and validate the atlas file.
    
    Args:
        atlas_path: Path to the atlas file
        
    Returns:
        Tuple of (mask_indices, atlas_shape)
        
    Raises:
        FileNotFoundError: If atlas file doesn't exist
        ValueError: If atlas is invalid
    """
    if not os.path.exists(atlas_path):
        raise FileNotFoundError(f"Atlas file not found: {atlas_path}")
        
    try:
        atlas_img = nib.load(atlas_path)
        atlas_data = atlas_img.get_fdata() > 0  # Convert to boolean mask
        mask_indices = atlas_data > 0
        n_voxels = np.sum(mask_indices)
        
        if n_voxels == 0:
            raise ValueError("Atlas mask is empty")
            
        logger.info(f"Atlas loaded successfully: shape={atlas_data.shape}, active_voxels={n_voxels}")
        return mask_indices, atlas_data.shape
        
    except Exception as e:
        logger.error(f"Error loading atlas: {str(e)}")
        raise

def process_subject(
    sub_id: str,
    mask_indices: np.ndarray,
    data_dir: str,
    output_dir: str,
    expected_shape: Optional[Tuple] = None
) -> bool:
    """
    Process a single subject's data with the atlas mask.
    
    Args:
        sub_id: Subject identifier
        mask_indices: Boolean mask from atlas
        data_dir: Directory containing input data
        output_dir: Directory for output data
        expected_shape: Expected shape of input data (optional)
        
    Returns:
        bool: True if processing successful, False otherwise
    """
    input_path = os.path.join(data_dir, f"{sub_id}_cleaned_smoothed_masked_bold.nii.gz")
    output_path = os.path.join(output_dir, f"{sub_id}_masked_data.npy")
    
    try:
        # Load and validate input data
        if not os.path.exists(input_path):
            logger.error(f"Input file not found for subject {sub_id}")
            return False
            
        data = nib.load(input_path).get_fdata()
        
        # Validate data shape
        if expected_shape and data.shape[:-1] != expected_shape:
            logger.error(f"Shape mismatch for {sub_id}: expected {expected_shape}, got {data.shape[:-1]}")
            return False
            
        # Apply mask and validate output
        masked_data = data[mask_indices, :]
        
        # Data quality checks
        if np.any(np.isnan(masked_data)):
            logger.warning(f"NaN values found in masked data for {sub_id}")
            
        if np.any(np.isinf(masked_data)):
            logger.warning(f"Infinite values found in masked data for {sub_id}")
            
        # Log data statistics
        logger.info(
            f"Subject {sub_id} - "
            f"Input shape: {data.shape} → "
            f"Masked shape: {masked_data.shape} | "
            f"Range: [{masked_data.min():.2f}, {masked_data.max():.2f}]"
        )
        
        # Save output
        np.save(output_path, masked_data)
        return True
        
    except Exception as e:
        logger.error(f"Error processing subject {sub_id}: {str(e)}")
        return False

def main(subject_ids: List[str], atlas_path: str, data_dir: str, output_dir: str) -> None:
    """
    Main function to process all subjects.
    
    Args:
        subject_ids: List of subject identifiers
        atlas_path: Path to atlas file
        data_dir: Directory containing input data
        output_dir: Directory for output data
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load and validate atlas
        mask_indices, atlas_shape = load_and_validate_atlas(atlas_path)
        
        # Process all subjects
        successful = 0
        failed = 0
        
        for sub_id in subject_ids:
            if process_subject(sub_id, mask_indices, data_dir, output_dir, atlas_shape):
                successful += 1
            else:
                failed += 1
                
        # Log summary
        logger.info(f"\nProcessing complete:")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Total: {len(subject_ids)}")
        
    except Exception as e:
        logger.error(f"Fatal error in main processing: {str(e)}")
        raise

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    scratch_dir = os.getenv("SCRATCH_DIR")
    if not scratch_dir:
        logger.error("SCRATCH_DIR environment variable not set")
        sys.exit(1)
        
    # Get subject IDs
    affair_ids = os.getenv("AFFAIR_SUBJECTS", "").split(",")
    paranoia_ids = os.getenv("PARANOIA_SUBJECTS", "").split(",")
    all_subjects = [s for s in (affair_ids + paranoia_ids) if s]  # Remove empty strings
    
    if not all_subjects:
        logger.error("No subject IDs found in environment variables")
        sys.exit(1)
    
    # Set up paths
    res = "native"  # or "native"
    parcellation_dir = os.path.join(scratch_dir, "data", "combined_parcellations")
    atlas_name = f"combined_Schaefer2018_1000Parcels_Kong2022_17Networks_Tian_Subcortex_S4_3T_{res}.nii.gz"
        
    atlas_path = os.path.join(parcellation_dir, atlas_name)
    data_dir = os.path.join(scratch_dir, "output", f"postproc_{res}")
    output_dir = os.path.join(scratch_dir, "output", f"atlas_masked_{res}")
    
    # Run main processing
    logger.info("Starting atlas masking pipeline")
    logger.info(f"Resolution: {res}")
    logger.info(f"Number of subjects: {len(all_subjects)}")
    
    main(all_subjects, atlas_path, data_dir, output_dir)