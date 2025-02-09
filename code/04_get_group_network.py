import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
from dotenv import load_dotenv
from argparse import ArgumentParser

class NetworkProcessor:
    def __init__(self, atlas_masked_dir: str):
        self.atlas_masked_dir = Path(atlas_masked_dir)
        self.networks_dir = self.atlas_masked_dir / 'networks'
        self.networks_dir.mkdir(exist_ok=True)
        
    def organize_by_networks(self, cortex_df: pd.DataFrame) -> Dict[str, List[int]]:
        """
        Create dictionary of networks with their corresponding parcel indices,
        combining left and right hemisphere parcels for each network.
        
        Args:
            cortex_df: DataFrame containing cortical parcellation information
            
        Returns:
            Dictionary mapping network names to lists of parcel indices
        """
        network_dict: Dict[str, List[int]] = {}
        
        # Process cortical parcels
        for _, row in cortex_df.iterrows():
            parcel_name = row[1]
            parcel_idx = row[0]
            
            try:
                # Split the parcel name to extract network
                # Example: 17networks_LH_DefaultA_FPole_1
                # or: 17networks_RH_VisualC_ExStr_9
                parts = parcel_name.split('_')
                if len(parts) < 3:
                    raise ValueError(f"Invalid parcel name format: {parcel_name}")
                    
                # Get network name (e.g., DefaultA, VisualC)
                network_name = parts[2]
                
                # Add to network dictionary
                if network_name not in network_dict:
                    network_dict[network_name] = []
                network_dict[network_name].append(parcel_idx)
                
            except Exception as e:
                raise ValueError(f"Error processing parcel {parcel_idx}: {str(e)}")
        
        # Add subcortex as a separate network
        network_dict['Subcortex'] = list(range(1001, 1055))
        
        return network_dict
    
    def process_group(self, group: str, cortex_df: pd.DataFrame) -> None:
        """Process and save data for each network."""
        # Validate inputs
        if cortex_df.empty:
            raise ValueError("Empty cortex DataFrame provided")
            
        # Get network dictionary
        network_dict = self.organize_by_networks(cortex_df)
        
        # Save network mapping
        network_df = pd.DataFrame({
            'network': list(network_dict.keys()),
            'parcel_indices': list(network_dict.values()),
            'n_parcels': [len(indices) for indices in network_dict.values()]
        })
        network_df.to_csv(self.networks_dir / f"{group}_network_df.csv", index=False)
        
        # Load and validate data
        data_path = self.atlas_masked_dir / f"{group}_group_data.npy"
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
            
        # Use memory mapping for large files
        data = np.load(data_path, mmap_mode='r')
        expected_parcels = 1054  # 1000 cortical + 54 subcortical
        if data.shape[1] != expected_parcels:
            raise ValueError(f"Expected {expected_parcels} parcels, got {data.shape[1]}")
        
        # Process each network
        for network_name, indices in network_dict.items():
            # Convert to 0-based indexing
            indices = np.array(indices) - 1
            
            # Extract and save network data
            network_data = data[:, indices, :].copy()
            output_file = self.networks_dir / f"{group}_{network_name}_data.npy"
            np.save(output_file, network_data)
            print(f"Saved {network_name} data: shape={network_data.shape}")
            # This will now show the correct number of parcels per network
            # combining both hemispheres for each network

def main():
    args = parse_args()
    load_dotenv()
    
    scratch_dir = os.getenv("SCRATCH_DIR")
    if not scratch_dir:
        raise ValueError("SCRATCH_DIR environment variable not set")
    
    parcellation_dir = Path(scratch_dir) / "data" / "combined_parcellations"
    atlas_masked_dir = Path(scratch_dir) / "output" / f"atlas_masked_{args.res}"
    
    # Load cortex data
    cortex_file = parcellation_dir / "Schaefer2018_1000Parcels_Kong2022_17Networks_order.txt"
    cortex_df = pd.read_csv(cortex_file, header=None, sep="\t")
    
    # Process data
    processor = NetworkProcessor(atlas_masked_dir)
    processor.process_group(args.group, cortex_df)

def parse_args():
    parser = ArgumentParser(description="Process fMRI data by networks")
    parser.add_argument("--res", type=str, default="native",
                       help="Resolution for analysis")
    parser.add_argument("--group", type=str, default="affair",
                       help="Group identifier")
    return parser.parse_args()

if __name__ == "__main__":
    main()