import os
import glob
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from argparse import ArgumentParser

def load_group_data(subjects, data_dir):
    """Load and stack data for a group of subjects.
    
    Returns:
        np.ndarray: Stacked data with shape (n_subjects, n_files, n_timepoints)
        Each file's data is averaged across its first dimension (voxels/parcels)
    """
    all_subjects_data = []
    expected_shape = None
    
    for subject in subjects:
        try:
            # Find all .npy files in subject directory
            data_paths = glob.glob(str(data_dir / f"{subject}/" / "*.npy"))
            if not data_paths:
                raise FileNotFoundError(f"No .npy files found for subject {subject}")
            
            data_paths.sort()
            print(f"Found {len(data_paths)} files for subject {subject}")
            
            # Load all files for this subject
            subject_data = []
            for i, path in enumerate(data_paths):
                data = np.load(path)
                # Take mean across first dimension (voxels/parcels)
                data_mean = np.mean(data, axis=0)  # Shape: (n_timepoints,)
                if i < 3:  # Print first 3 files' shapes for debugging
                    print(f"{subject} - {Path(path).name}: original shape {data.shape}, after mean {data_mean.shape}")
                subject_data.append(data_mean)
            
            # Stack all files for this subject
            subject_array = np.stack(subject_data)  # Shape: (n_files, n_timepoints)
            
            # Check shape consistency across subjects
            if expected_shape is None:
                expected_shape = subject_array.shape
            elif subject_array.shape != expected_shape:
                print(f"Warning: Inconsistent shape for subject {subject}")
                print(f"Expected {expected_shape}, got {subject_array.shape}")
                continue
                
            all_subjects_data.append(subject_array)
                
        except Exception as e:
            print(f"Error loading data for subject {subject}: {e}")
            continue
    
    if not all_subjects_data:
        return None
        
    try:
        return np.stack(all_subjects_data)  # Shape: (n_subjects, n_files, n_timepoints)
    except ValueError as e:
        print("Failed to stack arrays. Shape summary:")
        for i, data in enumerate(all_subjects_data):
            print(f"Subject {subjects[i]}: shape {data.shape}")
        return None

if __name__ == "__main__":
    load_dotenv()
    parser = ArgumentParser(description="Combine masked data for affair and paranoia groups")
    parser.add_argument("--res", type=str, default="native")
    args = parser.parse_args()

    # Setup paths
    scratch_dir = os.getenv("SCRATCH_DIR")
    if not scratch_dir:
        raise ValueError("SCRATCH_DIR environment variable not set")
    
    data_dir = Path(scratch_dir) / "output" / f"atlas_masked_{args.res}"
    if not data_dir.exists():
        raise ValueError(f"Data directory does not exist: {data_dir}")

    # Get subject lists
    affair_subjects = os.getenv("AFFAIR_SUBJECTS", "").split(",")
    paranoia_subjects = os.getenv("PARANOIA_SUBJECTS", "").split(",")

    if not affair_subjects[0] or not paranoia_subjects[0]:
        raise ValueError("Subject lists are empty")

    # Load and combine data
    affair_data = load_group_data(affair_subjects, data_dir)
    paranoia_data = load_group_data(paranoia_subjects, data_dir)

    if affair_data is not None and paranoia_data is not None:
        print(f"Affair data shape: {affair_data.shape}")
        print(f"Paranoia data shape: {paranoia_data.shape}")

        np.save(data_dir / "affair_grouped_parcel_data.npy", affair_data)
        np.save(data_dir / "paranoia_grouped_parcel_data.npy", paranoia_data)
        print("Data successfully saved")
    else:
        print("Failed to load all data")