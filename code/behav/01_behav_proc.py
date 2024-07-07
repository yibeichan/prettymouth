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
    resp_TRs = np.unique(np.round(resp_seconds / TR).astype(int))
    return resp_TRs

def convert2binary(lst, size):
    logging.info("Converting list to binary array")
    binary_array = np.zeros(size, dtype=int)
    valid_indices = np.array(lst)
    valid_indices = valid_indices[valid_indices < size]
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
    agreement_df = pd.DataFrame(list(response_counts.items()), columns=[unit, 'count'])
    agreement_df = agreement_df.sort_values(by=unit).reset_index(drop=True)
    agreement_df['agreement'] = agreement_df['count'] / df.shape[0]
    
    if unit == 'TR':
        agreement_df = agreement_df[(agreement_df[unit] >= 14) & (agreement_df[unit] < 465)]
        # agreement_df[unit] = agreement_df[unit] - 14
    elif unit == 'second':
        agreement_df = agreement_df[(agreement_df[unit] >= 21) & (agreement_df[unit] < 697)]
        # agreement_df[unit] = agreement_df[unit] - 21
    else:
        logging.error("Invalid unit provided")
        raise ValueError("Please enter a valid unit: TR or second")
    
    agreement_df = agreement_df.reset_index(drop=True)
    return agreement_df

def get_tsdf(df, n_TR=465):
    logging.info("Generating time series DataFrame")
    unit = 'TR'
    ts_df = pd.DataFrame({unit: np.arange(n_TR)})
    
    ts_df = ts_df.merge(df[[unit, "agreement"]], on=unit, how='left').fillna(0)
    return ts_df

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

def create_event_df(prom, prominent_peaks, word_df, filepath):
    logging.info(f"Creating event DataFrame for prominence {prom}")
    events = []
    for i, n in enumerate(prominent_peaks):
        start = 14 if i == 0 else prominent_peaks[i-1]
        end = word_df["TR"].max() if i == len(prominent_peaks)-1 else n
        text = " ".join(word_df.loc[(word_df["TR"] >= start) & (word_df["TR"] <= end), "text"].values)
        events.append({"start_TR": start, "end_TR": end, "text": text})
    event_df = pd.DataFrame(events)
    event_df.to_csv(filepath, index=False)

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

    seg_df['onset_TRs'] = seg_df['onset_sec'].apply(lambda x: seconds2TRs(x)[0])
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

    agree_TS_df1 = get_tsdf(agree_TR_df1)
    agree_TS_df1_filtered = get_tsdf(agree_TR_df1_filtered)
    agree_TS_df2a = get_tsdf(agree_TR_df2a)
    agree_TS_df2b = get_tsdf(agree_TR_df2b)
    
    agree_TS_df1.to_pickle(os.path.join(output_dir, "agreement_eventseg_ts.pkl"))
    agree_TS_df1_filtered.to_pickle(os.path.join(output_dir, "agreement_eventseg_filtered_ts.pkl"))
    agree_TS_df2a.to_pickle(os.path.join(output_dir, "agreement_evidence_affair_ts.pkl"))
    agree_TS_df2b.to_pickle(os.path.join(output_dir, "agreement_evidence_paranoia_ts.pkl"))

    word_df["TR"] = word_df["onset"].apply(lambda x: np.round(x/1.5)).astype(int)

    for prom in [0.08, 0.09, 0.1, 0.11, 0.12]:
        prominent_peaks1, _ = find_peaks(agree_TS_df1['agreement'], prominence=prom, distance=5)
        logging.info(f"There are {len(prominent_peaks1)} prominent (agreement = {prom}) peaks in agreement")
        np.savetxt(os.path.join(output_dir, f"prominent_peaks_{int(prom*100):03}.txt"), prominent_peaks1, fmt='%d')
        filepath1 = os.path.join(output_dir, f"event_df_prom_{int(prom*100):03}.csv")
        create_event_df(prom, prominent_peaks1, word_df, filepath1)

        prominent_peaks2, _ = find_peaks(agree_TS_df1_filtered['agreement'], prominence=prom, distance=5)
        logging.info(f"There are {len(prominent_peaks2)} prominent (agreement = {prom}) peaks in agreement_filtered")
        np.savetxt(os.path.join(output_dir, f"prominent_peaks_filtered_{int(prom*100):03}.txt"), prominent_peaks2, fmt='%d')
        filepath2 = os.path.join(output_dir, f"event_df_prom_filtered_{int(prom*100):03}.csv")
        create_event_df(prom, prominent_peaks2, word_df, filepath2)

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