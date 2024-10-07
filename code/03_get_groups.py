import os
import numpy as np
import nibabel as nib
from dotenv import load_dotenv

def get_groups(subject_ids, data_dir):
    first_bold_file = os.path.join(data_dir, f"{subject_ids[0]}_cleaned_smoothed_masked_bold.npy")
    first_data = np.load(first_bold_file)
    
    n_subjects = len(subject_ids)
    group_data = np.zeros((n_subjects, *first_data.shape))  
    
    for i, subject_id in enumerate(subject_ids):
        print(f"Loading {subject_id}")
        bold_file = os.path.join(data_dir, f"{subject_id}_cleaned_smoothed_masked_bold.npy")
        group_data[i] = np.load(bold_file)
    
    return group_data

def main(affair_ids, paranoia_ids, data_dir, save_dir):
    affair_data = get_groups(affair_ids, data_dir)
    paranoia_data = get_groups(paranoia_ids, data_dir)
    
    np.save(os.path.join(save_dir, "affair_data_masked.npy"), affair_data)
    np.save(os.path.join(save_dir, "paranoia_data_masked.npy"), paranoia_data)

if __name__ == "__main__":
    load_dotenv()

    nese_dir = os.getenv("NESE_DIR")
    if nese_dir is None:
        raise EnvironmentError("NESE_DIR environment variable is not set.")

    data_dir = os.path.join(nese_dir, "output", "postproc_native")
    save_dir = os.path.join(nese_dir, "output", "group_data")
    os.makedirs(save_dir, exist_ok=True)

    affair_ids = ['sub-023', 'sub-032', 'sub-034', 'sub-038', 'sub-050', 'sub-083', 'sub-084', 'sub-085', 'sub-086', 'sub-087', 'sub-088', 'sub-089', 'sub-090', 'sub-091', 'sub-092', 'sub-093', 'sub-094', 'sub-095', 'sub-096', 'sub-097']
    paranoia_ids = ['sub-030', 'sub-052', 'sub-065', 'sub-066', 'sub-079', 'sub-081', 'sub-098', 'sub-099', 'sub-100', 'sub-101', 'sub-102', 'sub-103', 'sub-104', 'sub-105', 'sub-106', 'sub-107', 'sub-108', 'sub-109', 'sub-110', 'sub-111']

    main(affair_ids, paranoia_ids, data_dir, save_dir)