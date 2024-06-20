import os
import sys
from dotenv import load_dotenv
import gc
import numpy as np
from utils.coflt import get_eFC_chunked

def get_seg_eFC(boundaries, flattened_coflt, group, data_dir):
    delta = 3
    for i, r in enumerate(boundaries):
        if i == 0:
            start = 14 + delta
        else:
            start = r + delta
        if i != len(boundaries) - 1:
            end = boundaries[i+1] + delta
        else:
            end = 465 + delta
        seg_coflt = flattened_coflt[:, start:end]
        seg_eFC = get_eFC_chunked(seg_coflt)
        np.save(os.path.join(data_dir, f"{group}_eFC_event{i+1:02d}.npy"), seg_eFC)
        print(f"Saved {group}_eFC_event{i+1:02d}.npy")
        del seg_coflt, seg_eFC
        gc.collect()

def main(n_parcel, boundary_dir, eFC_output_dir):
    data_dir = os.path.join(eFC_output_dir, f"{n_parcel}parcel")
    
    boundary_file = os.path.join(boundary_dir, "prominent_peaks_010.txt")
    with open(boundary_file, 'r') as f:
        lines = f.readlines()
        boundaries = np.array([int(float(line.strip())) for line in lines], dtype=int)
    
    # groups = ["affair", "paranoia", "mix", "diff"]
    groups = ["affair", "paranoia"]
    for group in groups:
        flattened_data = np.load(os.path.join(data_dir, f"flattened_{group}_coflt.npy"), mmap_mode='r')
        get_seg_eFC(boundaries, flattened_data, group, data_dir)
        del flattened_data
        gc.collect()

if __name__ == "__main__":
    load_dotenv()
    scratch_dir = os.getenv("SCRATCH_DIR")
    boundary_dir = os.path.join(scratch_dir, "output", "behav_results")
    eFC_output_dir = os.path.join(scratch_dir, "output", "eFC")
    os.makedirs(eFC_output_dir, exist_ok=True)
    n_parcel = int(sys.argv[1])
    main(n_parcel, boundary_dir, eFC_output_dir)
