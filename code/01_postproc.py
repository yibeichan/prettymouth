import os
import glob
import nibabel as nib
import numpy as np
import scipy.io
from nilearn import signal
from nilearn.interfaces.fmriprep import load_confounds_strategy
import subprocess
import tempfile
from joblib import Parallel, delayed
from dotenv import load_dotenv
from argparse import ArgumentParser

def process_file(task_file, smoothing_mm, lh_surf, rh_surf, save_dir):
    try:
        confounds_df, _ = load_confounds_strategy(task_file, denoise_strategy="compcor")
        task_img = nib.load(task_file)
        clean_func_signal = signal.clean(task_img.dataobj[:], detrend=True, confounds=confounds_df, standardize='zscore_sample', t_r=1.5)
        print(f"Cleaned {task_file}")
        
        task_cln = nib.Cifti2Image(clean_func_signal, task_img.header)
        with tempfile.NamedTemporaryFile(suffix='.dtseries.nii', delete=False) as tmpfile:
            cleaned_file_path = tmpfile.name
        
        nib.save(task_cln, cleaned_file_path)
        smooth_output_file = os.path.join(save_dir, os.path.basename(task_file).replace('.dtseries.nii', '_cleaned_smoothed.dtseries.nii'))
        smoothing_command = f"wb_command -cifti-smoothing {cleaned_file_path} {smoothing_mm} {smoothing_mm} COLUMN {smooth_output_file} -left-surface {lh_surf} -right-surface {rh_surf}"
        
        subprocess.run(smoothing_command, shell=True, check=True)
        print(f"Smoothed {task_file}")
    finally:
        os.remove(cleaned_file_path)

def process_subject(subject_files, smoothing_mm, lh_surf, rh_surf, save_dir):
    if not subject_files:
        print("No subject files found.")
        return
    Parallel(n_jobs=-1)(
        delayed(process_file)(task_file, smoothing_mm, lh_surf, rh_surf, save_dir) for task_file in subject_files)

def reformat_parcellation(scratch_dir, n_parcel, yan_kong_path, medial_mask_path):
    # Load the parcellation data
    yan_kong17 = nib.load(yan_kong_path)
    # Extract ROIs of grayordinates in the cortex
    rois = yan_kong17.dataobj[0].astype(int)
    # Define a medial wall mask for fsLR
    mat = scipy.io.loadmat(medial_mask_path)
    mask = mat['medial_mask']
    rois = rois[np.where(mask != 0)[0]]

    # Extract parcel names and colors
    axis0 = yan_kong17.header.get_index_map(0)
    nmap = list(axis0.named_maps)[0]

    keys = [0]
    labels = ['']
    rgba = [(0.0, 0.0, 0.0, 0.0)]
    for i in range(1, n_parcel+1):
        roi = nmap.label_table[i]
        labels.append(roi.label[11:])
        rgba.append((roi.red, roi.green, roi.blue, roi.alpha))
        keys.append(i)

    # Define structures and their names for subcortical parts
    structures = {
        'accumbens_left_': slice(59412,59547),
        'accumbens_right_': slice(59547,59687),
        'amygdala_left_': slice(59687,60002),
        'amygdala_right_': slice(60002,60334),
        'brainStem_': slice(60334,63806),
        'caudate_left_': slice(63806,64534),
        'caudate_right_': slice(64534,65289),
        'cerebellum_left_': slice(65289,73998),
        'cerebellum_right_': slice(73998,83142),
        'diencephalon_left_': slice(83142,83848),
        'diencephalon_right_': slice(83848,84560),
        'hippocampus_left_': slice(84560,85324),
        'hippocampus_right_': slice(85324,86119),
        'pallidum_left_': slice(86119,86416),
        'pallidum_right_': slice(86416,86676),
        'putamen_left_': slice(86676,87736),
        'putamen_right_': slice(87736,88746),
        'thalamus_left_': slice(88746,90034),
        'thalamus_right_': slice(90034,None)
    }

    map_all = np.zeros(91282, dtype=int)
    map_all[:59412] = rois

    for i, (name, struct) in enumerate(structures.items()):
        map_all[struct] = int(n_parcel + 1) + i
        keys.append(int(n_parcel + 1) + i)
        labels.append(name)

    # Assign black color for subcortical parts
    rgba.extend([(0.0, 0.0, 0.0, 0.0)] * len(structures))

    # Save parcellation
    parcel_dir = os.path.join(scratch_dir, 'data', 'yan_parcellations')
    os.makedirs(parcel_dir, exist_ok=True)
    np.savez_compressed(os.path.join(parcel_dir, f'yan_kong17_{n_parcel}parcels.npz'), 
                        map_all=map_all, labels=labels, rgba=rgba, ids=keys)

if __name__ == "__main__":
    load_dotenv()
    
    base_dir = os.getenv("BASE_DIR")
    scratch_dir = os.getenv("SCRATCH_DIR")
    nese_dir = os.getenv("NESE_DIR")
    data_dir = os.path.join(nese_dir, 'data')
    fmriprep_dir = "/om2/scratch/Fri/gelbanna/gelbanna/prettymouth_project/merge_ds/fmriprep-23.1.4"
    
    if not base_dir or not scratch_dir:
        print("BASE_DIR or SCRATCH_DIR environment variables not set")
        exit(1)
    
    output_dir = os.path.join(scratch_dir, "output")
    save_dir = os.path.join(output_dir, "postproc")
    os.makedirs(save_dir, exist_ok=True)
    
    subject_files = glob.glob(os.path.join(fmriprep_dir, 'sub-*', 'func', '*prettymouth*.dtseries.nii'))
    # print(subject_files)
    # L-R surface templates
    lh_surf = os.path.join(data_dir, 'fsLR32k', 'tpl-fsLR_hemi-L_den-32k_sphere.surf.gii')
    rh_surf = os.path.join(data_dir, 'fsLR32k', 'tpl-fsLR_hemi-R_den-32k_sphere.surf.gii')
    
    smoothing_mm = 2
    process_subject(subject_files, smoothing_mm, lh_surf, rh_surf, save_dir)

    medial_mask_path = os.path.join(data_dir, 'fsLR32k','fsLR_32k_medial_mask.mat')
    # reformat all parcellations
    for n_parcel in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
        yan_kong_path = os.path.join(data_dir, 'parcellations', f'{n_parcel}Parcels_Kong2022_17Networks.dlabel.nii')
        reformat_parcellation(scratch_dir, n_parcel, yan_kong_path, medial_mask_path)