import warnings
warnings.filterwarnings('ignore')

import os
import logging
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import ast
from bisect import bisect_left
from collections import Counter
from scipy.signal import find_peaks

def setup_logging(base_dir, task, task_id):
    """
    Sets up logging for the given base directory, data input, and sub ID, and returns the log filename.
    
    Args:
        base_dir (str): The base directory for the log file.
        task (str): The task for the log file.
        task_id (str): The task ID for the log file.
    
    Returns:
        str: The generated log filename.
    """
    log_dir = os.path.join(base_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, f'{task}_{task_id}.log')
    logging.basicConfig(filename=log_filename, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    return log_filename

def seconds2TRs(resp_seconds, TR=1.5):
    logging.info("Converting seconds to TRs")
    resp_seconds = np.array(resp_seconds)
    resp_TRs = np.round(resp_seconds / TR).astype(int)
    return np.unique(resp_TRs)

def convert2binary(lst, size):
    logging.info("Converting list to binary array")
    binary_array = np.zeros(size, dtype=int)
    valid_indices = np.clip(lst, 0, size-1)
    binary_array[valid_indices] = 1
    return binary_array

def convert_to_TRs(df, TR, n_TR=465):
    logging.info("Converting response times to TRs")
    df['key_resp_2.rt'] = df['key_resp_2.rt'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df['key_resp_2.TR'] = df['key_resp_2.rt'].apply(lambda x: seconds2TRs(x, TR))
    df['binary_response'] = df['key_resp_2.TR'].apply(lambda x: convert2binary(x, n_TR))
    return df[['key_resp_2.TR', 'binary_response', 'participant']].copy()

def get_agreement(df, unit='TR'):
    logging.info("Calculating agreement")
    response_col = 'key_resp_2.TR'
    all_responses = np.concatenate(df[response_col].dropna().values)
    response_counts = Counter(all_responses)
    agreement_df = pd.DataFrame(list(response_counts.items()), columns=[unit, 'count']).sort_values(by=unit).reset_index(drop=True)
    agreement_df['agreement'] = agreement_df['count'] / df.shape[0]
    
    if unit == 'TR':
        agreement_df = agreement_df[(agreement_df[unit] >= 14) & (agreement_df[unit] < 466)]
    elif unit == 'second':
        agreement_df = agreement_df[(agreement_df[unit] >= 21) & (agreement_df[unit] < 699)]
    else:
        logging.error("Invalid unit provided")
        raise ValueError("Please enter a valid unit: TR or second")
    
    return agreement_df.reset_index(drop=True)

def filter_elements(A, list_B, delta=3):
    """
    Filters a given element A based on its proximity to elements in a sorted list B.

    This function determines whether a given key response timepoint (A) is close to 
    a boundary (e.g., beginning or end of a sentence, or a pause in the speech) 
    in a sorted list of boundary timepoints (list_B). If A is within a threshold 
    delta of a boundary, the closest boundary timepoint is returned; otherwise, None is returned.

    Args:
        A (float or int): The key response timepoint to be evaluated.
        list_B (list of float or int): A sorted list of boundary timepoints.
        delta (float or int, optional): The maximum allowable distance for A to be considered close to a boundary. Default is 3.

    Returns:
        float or None: The closest boundary timepoint if A is within delta of a boundary, otherwise None.
    
    Methodology:
        - Behavioral data (key response) can be delayed even if our brain realizes there is a boundary.
        - The function examines each key response timepoint (A) and checks whether it is close to a boundary
          in list_B within the given threshold delta.
        - If A is close to a boundary (within delta), the closest boundary timepoint is returned.
        - If no close boundary is found, the function returns None.
    
    Examples:
        >>> filter_elements(10, [7, 11, 15], delta=3)
        11
        >>> filter_elements(5, [7, 11, 15], delta=3)
        None
    """
    pos = bisect_left(list_B, A)
    if pos == 0 or pos == len(list_B):
        return None

    B = list_B[pos]
    B_prev = list_B[pos - 1]

    B_next = list_B[pos + 1] if pos + 1 < len(list_B) else float('inf')

    if A == B:
        return B
    elif B_prev <= A <= min(B_prev + delta, B):
        return B_prev
    elif B <= A <= min(B + delta, B_next):
        return B
    return None

def map_boundary2seg(seg_df, boundaries, filepath):
    result = []

    for i, r in enumerate(boundaries):
        if i == 0:  # First iteration
            seg_index = seg_df[seg_df['onset_TRs'] == r].index[-1]
            seg = seg_df.iloc[:seg_index]['Seg'].tolist()
        elif i == len(boundaries) - 1:  # Last iteration
            prev_r = boundaries[i-1]
            prev_seg_index = seg_df[seg_df['onset_TRs'] == prev_r].index[-1]
            seg = seg_df.iloc[prev_seg_index:]['Seg'].tolist()
        else:  # Subsequent iterations
            prev_r = boundaries[i-1]
            prev_seg_index = seg_df[seg_df['onset_TRs'] == prev_r].index[-1]
            seg_index = seg_df[seg_df['onset_TRs'] == r].index[-1]
            seg = seg_df.iloc[prev_seg_index:seg_index]['Seg'].tolist()

        result.append({'boundary': r, 'segments': seg[0]})

    pd.DataFrame(result).to_csv(filepath, index=False)

def map_boundary2word(word_df,boundaries, filepath):
    events = []
    for i, n in enumerate(boundaries):
        start = 14 if i == 0 else boundaries[i-1]
        end = word_df["TR"].max() if i == len(boundaries)-1 else n
        text = " ".join(word_df.loc[(word_df["TR"] >= start) & (word_df["TR"] <= end), "text"].values)
        events.append({"start_TR": start, "end_TR": end, "text": text})
    event_df = pd.DataFrame(events)
    event_df.to_csv(filepath, index=False)

def process_thresholds(threshold_list, agree_tr_df, output_prefix, word_df, seg_df, output_dir, filtered=False):
    peaks_info = []
    
    for th in threshold_list:
        prominent_peaks, _ = find_peaks(agree_tr_df['agreement'], prominence=th)
        logging.info(f"There are {len(prominent_peaks)} prominent (threshold = {th}) peaks in {output_prefix}")
        
        # Save the prominent peaks to a text file
        np.savetxt(os.path.join(output_dir, f"prominent_peaks_{output_prefix}_{int(th*100):03}.txt"), prominent_peaks, fmt='%d')
        
        # Create the filepath for the event dataframe CSV
        filepath = os.path.join(output_dir, f"event_df_prom_{output_prefix}_{int(th*100):03}.csv")
        
        # Determine boundaries and append the final TR value
        boundaries = agree_tr_df['TR'].iloc[prominent_peaks].tolist()
        boundaries.append(465.0)
        
        if not filtered:
            map_boundary2word(word_df, boundaries, filepath)
        else:
            map_boundary2seg(seg_df, boundaries, filepath)
        
        # Save threshold and number of peaks to the list
        peaks_info.append((th, len(prominent_peaks)))
    
    return peaks_info

def main(df1, df2a, df2b, word_df, seg_df, output_dir):
    logging.info("Starting main process")
    response_df1 = convert_to_TRs(df1, 1.5)
    response_df2a = convert_to_TRs(df2a, 1.5)
    response_df2b = convert_to_TRs(df2b, 1.5)

    response_df1.to_pickle(os.path.join(output_dir, "individual_response_eventseg.pkl"))
    response_df2a.to_pickle(os.path.join(output_dir, "individual_response_evidence_affair.pkl"))
    response_df2b.to_pickle(os.path.join(output_dir, "individual_response_evidence_paranoia.pkl"))

    agree_TR_df1 = get_agreement(response_df1)
    agree_TR_df2a = get_agreement(response_df2a)
    agree_TR_df2b = get_agreement(response_df2b)

    seg_df['onset_TRs'] = seg_df['onset_sec'].apply(lambda x: seconds2TRs(x)[0]) + 14 # annotated story starts at 0 not 14, so we need to add 14
    sentence_onset_TRs = np.unique(seg_df['onset_TRs']) 

    agree_TR_df1['filtered_TRs'] = agree_TR_df1['TR'].apply(lambda x: filter_elements(x, sentence_onset_TRs))

    # Filter out rows where 'filtered_TRs' is NaN before concatenation
    all_filtered_TRs = np.concatenate(
        [np.repeat(row['filtered_TRs'], int(row['count'])) 
        for _, row in agree_TR_df1.iterrows() 
        if not np.isnan(row['filtered_TRs'])]
    )

    filtered_response_counts = Counter(all_filtered_TRs)
    agree_TR_df1_filtered = pd.DataFrame(list(filtered_response_counts.items()), columns=['TR', 'count'])
    agree_TR_df1_filtered = agree_TR_df1_filtered.sort_values(by='TR').reset_index(drop=True)
    agree_TR_df1_filtered['agreement'] = agree_TR_df1_filtered['count'] / df1.shape[0]

    agree_TR_df1.to_pickle(os.path.join(output_dir, "agreement_eventseg.pkl"))
    agree_TR_df1_filtered.to_pickle(os.path.join(output_dir, "agreement_eventseg_filtered.pkl"))
    agree_TR_df2a.to_pickle(os.path.join(output_dir, "agreement_evidence_affair.pkl"))
    agree_TR_df2b.to_pickle(os.path.join(output_dir, "agreement_evidence_paranoia.pkl"))

    word_df["TR"] = word_df["onset"].apply(lambda x: np.round(x/1.5)).astype(int)

    th = np.around(agree_TR_df1['agreement'].quantile(0.85), decimals=2)
    th_filtered = np.around(agree_TR_df1_filtered['agreement'].quantile(0.85), decimals=2)

    th_lst = [round(th, 2) for th in np.arange(th-0.05, th+0.1, 0.01)]
    th_filtered_lst = [round(th_filtered, 2) for th_filtered in np.arange(th_filtered-0.05, th_filtered+0.1, 0.01)]


    peak_th = process_thresholds(th_lst, agree_TR_df1, "eventseg", word_df, seg_df, output_dir, filtered=False)
    peak_th_filtered = process_thresholds(th_filtered_lst, agree_TR_df1_filtered, "eventseg_filtered", word_df, seg_df, output_dir, filtered=True)

    # Convert the lists to NumPy arrays
    peak_th_array = np.array(peak_th, dtype=[('threshold', 'f4'), ('number_of_peaks', 'i4')])
    peak_th_filtered_array = np.array(peak_th_filtered, dtype=[('threshold', 'f4'), ('number_of_peaks', 'i4')])

    # Save the justification data as NumPy arrays
    np.save(os.path.join(output_dir, 'n_peak_prom_threshold.npy'), peak_th_array)
    np.save(os.path.join(output_dir, 'n_peak_prom_threshold_filtered.npy'), peak_th_filtered_array)

if __name__ == '__main__':
    logging.info("Loading environment variables")
    load_dotenv()
    base_dir = os.getenv("SCRATCH_DIR")
    
    if not base_dir:
        logging.error("BASE_DIR environment variable not set")
        raise EnvironmentError("BASE_DIR environment variable not set")

    data_dir = os.path.join(base_dir, "data")
    behav_dir = os.path.join(data_dir, "behav")
    stimuli_dir = os.path.join(data_dir, "stimuli")
    output_dir = os.path.join(base_dir, "output", "behav_results")
    os.makedirs(output_dir, exist_ok=True)

    stimuli_file1 = os.path.join(stimuli_dir, "word_by_onset.csv")
    word_df = pd.read_csv(stimuli_file1, encoding='latin1')
    stimuli_file2 = os.path.join(stimuli_dir, "segments_speaker.csv")
    seg_df = pd.read_csv(stimuli_file2, encoding='latin1')

    df_1 = pd.read_csv(os.path.join(behav_dir, "prettymouth1.csv"))
    df_2a = pd.read_csv(os.path.join(behav_dir, "prettymouth2a.csv"))
    df_2b = pd.read_csv(os.path.join(behav_dir, "prettymouth2b.csv"))
    df_1_postsurvey = pd.read_csv(os.path.join(behav_dir, "prettymouth1_postsurvey.csv")).iloc[2:]
    df_2_postsurvey = pd.read_csv(os.path.join(behav_dir, "prettymouth2_postsurvey.csv")).iloc[2:]

    df_1_response = df_1.dropna(subset=["key_resp_2.rt"])
    df_2a_response = df_2a.dropna(subset=["key_resp_2.rt"])
    df_2b_response = df_2b.dropna(subset=["key_resp_2.rt"])

    common_participants_1 = set(df_1_response["participant"]).intersection(df_1_postsurvey["participant"])
    common_participants_2a = set(df_2a_response["participant"]).intersection(df_2_postsurvey["participant"])
    common_participants_2b = set(df_2b_response["participant"]).intersection(df_2_postsurvey["participant"])

    df1 = df_1_response[df_1_response["participant"].isin(common_participants_1)]
    df2a = df_2a_response[df_2a_response["participant"].isin(common_participants_2a)]
    df2b = df_2b_response[df_2b_response["participant"].isin(common_participants_2b)]

    setup_logging(base_dir=base_dir, task="behav_data_proc", task_id="001")
    main(df1, df2a, df2b, word_df, seg_df, output_dir)