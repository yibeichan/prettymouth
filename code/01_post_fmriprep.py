import os
import numpy as np
import pandas as pd
import scipy.io
import nibabel as nib
from tqdm import tqdm
from nilearn import signal, image as nimg
from bids.layout import BIDSLayout
from nipype.interfaces.workbench.base import WBCommand
from dotenv import load_dotenv

def preprocess_subject(sub, layout, task, smoothing_mm, right_surface, left_surface, fmriprep_dir):
    sub_dir = os.path.join(fmriprep_dir, f'sub-{sub}')
    smoothed_dir = os.path.join(sub_dir, 'smoothed')
    cleaned_dir = os.path.join(sub_dir, 'cleaned')

    # Create directories for smoothed and cleaned data
    os.makedirs(smoothed_dir, exist_ok=True)
    os.makedirs(cleaned_dir, exist_ok=True)
    
    # Query cifti files and confound files for a subject and a run
    func_file = layout.get(subject=sub, task=task, datatype='func', extension='dtseries.nii', return_type='file')[0]
    confound_file = layout.get(subject=sub, task=task, datatype='func', desc='confounds', extension="tsv", return_type='file')[0]

    # Smoothing cifti files using connectome workbench
    smooth_output_file = os.path.join(smoothed_dir, os.path.basename(func_file))
    # wb_command = WBCommand(command='wb_command')
    # wb_command.inputs.args = f'-cifti-smoothing {func_file} {smoothing_mm} {smoothing_mm} COLUMN {smooth_output_file} -right-surface {right_surface} -left-surface {left_surface}'
    # wb_command.run()
    # Reading the confound variables
    confound_df = pd.read_csv(confound_file, delimiter='\t')
    cols = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z', 'a_comp_cor_00', 'a_comp_cor_01', 'a_comp_cor_02', 'a_comp_cor_03', 'a_comp_cor_04']
    cosine_cols = confound_df.filter(like='cosine').columns
    selected_confounds = cols + list(cosine_cols)
    confounds_matrix = confound_df[selected_confounds].fillna(0).values

    # Load smoothed func data and remove confounds
    smoothed_func_img = nimg.load_img(smooth_output_file)
    clean_func_signal = signal.clean(smoothed_func_img.get_fdata(), detrend=True, confounds=confounds_matrix, standardize='zscore_sample')
    func_cln = nib.Cifti2Image(clean_func_signal, smoothed_func_img.header)

    # Save cleaned data
    func_cln.to_filename(os.path.join(cleaned_dir, os.path.basename(func_file)))

def reformat_parcellation(fmriprep_dir, n_parcel, yan_kong_path, medial_mask_path):
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
    parcel_dir = os.path.join(fmriprep_dir, 'yan_parcellations')
    os.makedirs(parcel_dir, exist_ok=True)
    np.savez_compressed(os.path.join(parcel_dir, f'yan_kong17_{n_parcel}parcels.npz'), 
                        map_all=map_all, labels=labels, rgba=rgba, ids=keys)

def main():
    load_dotenv()
    scratch_dir = os.getenv("SCRATCH_DIR")
    data_dir = os.path.join(scratch_dir, 'data')
    fmriprep_dir = os.path.join(data_dir, 'fmriprep')
    parcellation_dir = os.path.join(data_dir, 'parcellations')
    # L-R surface templates
    left_surface = os.path.join(data_dir, 'fsLR32k', 'tpl-fsLR_hemi-L_den-32k_sphere.surf.gii')
    right_surface = os.path.join(data_dir, 'fsLR32k', 'tpl-fsLR_hemi-R_den-32k_sphere.surf.gii')

    layout = BIDSLayout(fmriprep_dir, validate=False, config=['bids', 'derivatives'])
    # Query list of subjects
    subjects = layout.get_subjects()
    task = 'prettymouth'
    smoothing_mm = 2

    for sub in tqdm(subjects, desc='Preprocessing data for subjects...'):
        preprocess_subject(sub, layout, task, smoothing_mm, right_surface, left_surface, fmriprep_dir)

    medial_mask_path = os.path.join(data_dir, 'fsLR32k','fsLR_32k_medial_mask.mat')
    # reformat all parcellations
    for n_parcel in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
        yan_kong_path = os.path.join(parcellation_dir, f'{n_parcel}Parcels_Kong2022_17Networks.dlabel.nii')
        reformat_parcellation(fmriprep_dir, n_parcel, yan_kong_path, medial_mask_path)

if __name__ == '__main__':
    main()