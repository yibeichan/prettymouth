import os
import sys
from dotenv import load_dotenv
import gc
import numpy as np
from utils.coflt import read_coflt_data, flatten_3D, get_eFC_chunked

def process_group(n_parcel, group_name, coft_output_dir, eFC_output_dir):
    # coflt_data = read_coflt_data(n_parcel, group_name, coft_output_dir)
    # # Average across subjects
    # mean_coflt = np.nanmean(coflt_data, axis=0)
    # del coflt_data
    # gc.collect()
    
    save_dir = os.path.join(eFC_output_dir, f"{n_parcel}parcel")
    os.makedirs(save_dir, exist_ok=True)

    # flattened_data = flatten_3D(mean_coflt)
    # np.save(os.path.join(save_dir, f"flattened_{group_name}_coflt.npy"), flattened_data)
    flattened_data = np.load(os.path.join(save_dir, f"flattened_{group_name}_coflt.npy"))
    delta = 3
    start = 14 + delta
    end = start + 451
    eFC_data = get_eFC_chunked(flattened_data[:, start:end])
    np.save(os.path.join(save_dir, f"{group_name}_eFC.npy"), eFC_data)
    del flattened_data, eFC_data
    gc.collect()

def process_computed_group(n_parcel, group_name, group_data, eFC_output_dir):
    save_dir = os.path.join(eFC_output_dir, f"{n_parcel}parcel")
    os.makedirs(save_dir, exist_ok=True)

    delta = 3
    start = 14 + delta
    end = start + 451
    # Compute and save eFC
    eFC_data = get_eFC_chunked(group_data[:, start:end])
    np.save(os.path.join(save_dir, f"{group_name}_eFC.npy"), eFC_data)
    
    del group_data, eFC_data
    gc.collect()

def main(n_parcel, coft_output_dir, eFC_output_dir):
    # Process individual groups
    process_group(n_parcel, "affair", coft_output_dir, eFC_output_dir)
    process_group(n_parcel, "paranoia", coft_output_dir, eFC_output_dir)
    
    # Compute the mix and diff groups
    flattened_affair_coflt = np.load(os.path.join(eFC_output_dir, f"{n_parcel}parcel", "flattened_affair_coflt.npy"), mmap_mode='r')
    flattened_paranoia_coflt = np.load(os.path.join(eFC_output_dir, f"{n_parcel}parcel", "flattened_paranoia_coflt.npy"), mmap_mode='r')
    
    flattened_mix_coflt = (flattened_affair_coflt + flattened_paranoia_coflt) / 2
    process_computed_group(n_parcel, "mix", flattened_mix_coflt, eFC_output_dir)
    
    flattened_diff_coflt = np.abs(flattened_affair_coflt - flattened_paranoia_coflt)
    process_computed_group(n_parcel, "diff", flattened_diff_coflt, eFC_output_dir)

if __name__ == "__main__":
    load_dotenv()
    scratch_dir = os.getenv("SCRATCH_DIR")
    coft_output_dir = os.path.join(scratch_dir, "output", "cofluctuation_LOO")
    eFC_output_dir = os.path.join(scratch_dir, "output", "eFC")
    n_parcel = int(sys.argv[1])
    main(n_parcel, coft_output_dir, eFC_output_dir)
