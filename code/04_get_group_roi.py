import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv
from argparse import ArgumentParser

@dataclass
class Config:
    """Configuration parameters for data processing."""
    N_CORTICAL_PARCELS: int = 1000  # Number of cortical parcels in Schaefer atlas
    FILE_ENCODING: str = 'utf-8'
    DELIMITER: str = '\t'

class DataProcessor:
    """Process neuroimaging data by averaging parcels within ROIs."""
    
    def __init__(self, atlas_file: str, output_dir: str):
        """
        Initialize the processor.
        
        Parameters:
        ----------
        atlas_file : str
            Path to the Schaefer atlas file
        output_dir : str
            Directory for output files
        """
        self.atlas_file = atlas_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cortex_df = self._load_atlas()
        
    def _load_atlas(self) -> pd.DataFrame:
        """Load and validate the atlas file."""
        try:
            df = pd.read_csv(
                self.atlas_file, 
                header=None, 
                sep=Config.DELIMITER,
                encoding=Config.FILE_ENCODING
            )
            if df.empty:
                raise ValueError("Atlas file is empty")
            return df
        except Exception as e:
            raise RuntimeError(f"Failed to load atlas file: {e}")

    @staticmethod
    def get_roi_name(full_name: str) -> str:
        """Extract ROI name from full parcel name."""
        return full_name.replace('17networks_', '').rsplit('_', 1)[0]

    def create_roi_mapping(self) -> Dict[str, List[int]]:
        """Create mapping from ROIs to parcel indices."""
        roi_dict = {}
        for _, row in self.cortex_df.iterrows():
            roi_name = self.get_roi_name(row[1])
            parcel_idx = row[0]
            roi_dict.setdefault(roi_name, []).append(parcel_idx)
        return roi_dict

    def average_by_roi(self, data: np.ndarray, roi_dict: Dict[str, List[int]]) -> np.ndarray:
        """
        Average data within each ROI.
        
        Parameters:
        ----------
        data : np.ndarray
            Input data of shape (n_subjects, n_voxels, n_timepoints)
        roi_dict : Dict[str, List[int]]
            Mapping of ROI names to parcel indices
            
        Returns:
        -------
        np.ndarray
            Averaged data of shape (n_subjects, n_rois, n_timepoints)
        """
        if data.size == 0:
            raise ValueError("Input data array is empty")
        if not roi_dict:
            raise ValueError("ROI dictionary is empty")
            
        n_subjects, n_voxels, n_timepoints = data.shape
        n_rois = len(roi_dict)
        
        # Validate indices
        all_indices = np.concatenate([np.array(indices) - 1 for indices in roi_dict.values()])
        if np.any(all_indices >= n_voxels) or np.any(all_indices < 0):
            raise ValueError("ROI dictionary contains invalid indices")
        
        averaged_data = np.zeros((n_subjects, n_rois, n_timepoints))
        
        for roi_idx, (_, indices) in enumerate(roi_dict.items()):
            indices = np.array(indices) - 1  # Convert to 0-based indexing
            averaged_data[:, roi_idx, :] = np.mean(data[:, indices, :], axis=1)
            
        return averaged_data

    def save_roi_mapping(self, roi_dict: Dict[str, List[int]], group: str) -> None:
        """Save ROI to parcel mapping."""
        roi_df = pd.DataFrame({
            'roi': list(roi_dict.keys()),
            'parcel_indices': list(roi_dict.values())
        })
        output_path = self.output_dir / f"{group}_roi_df.csv"
        roi_df.to_csv(output_path, index=False)

    def process_group_data(self, group: str) -> None:
        """
        Process data for a specific group.
        
        Parameters:
        ----------
        group : str
            Group identifier
        """
        # Create ROI mapping
        roi_dict = self.create_roi_mapping()
        self.save_roi_mapping(roi_dict, group)
        
        # Load and process data
        try:
            data_path = self.output_dir / f"{group}_group_data.npy"
            data = np.load(data_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find data file: {data_path}")
            
        # Split cortical and subcortical data
        cortical_data = data[:, :Config.N_CORTICAL_PARCELS, :]
        subcortical_data = data[:, Config.N_CORTICAL_PARCELS:, :]
        
        # Process cortical data
        averaged_data = self.average_by_roi(cortical_data, roi_dict)
        
        # Combine with subcortical data
        reduced_data = np.concatenate([averaged_data, subcortical_data], axis=1)
        
        # Save processed data
        output_path = self.output_dir / f"{group}_group_data_roi.npy"
        np.save(output_path, reduced_data)
        
        # Log processing summary
        print(f"Processing complete for group: {group}")
        print(f"Original shape: {data.shape}")
        print(f"Averaged shape: {averaged_data.shape}")
        print(f"Final shape: {reduced_data.shape}")

def parse_args():
    """Parse command line arguments."""
    parser = ArgumentParser(description="Process fMRI data by averaging within ROIs")
    parser.add_argument("--res", type=str, default="native",
                       help="Resolution for analysis (default: native)")
    parser.add_argument("--group", type=str, default="affair",
                       help="Group identifier (default: affair)")
    return parser.parse_args()

def main():
    """Main execution function."""
    load_dotenv()
    args = parse_args()
    
    # Set up paths
    scratch_dir = os.getenv("SCRATCH_DIR")
    if not scratch_dir:
        raise ValueError("SCRATCH_DIR environment variable not set")
        
    parcellation_dir = Path(scratch_dir) / "data" / "combined_parcellations"
    atlas_masked_dir = Path(scratch_dir) / "output" / f"atlas_masked_{args.res}"
    atlas_file = parcellation_dir / "Schaefer2018_1000Parcels_Kong2022_17Networks_order.txt"
    
    # Initialize processor and run
    processor = DataProcessor(str(atlas_file), str(atlas_masked_dir))
    processor.process_group_data(args.group)

if __name__ == "__main__":
    main()