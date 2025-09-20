"""
08_behav_proc_bin_1s.py

Same logic as 08_behav_proc.py but using 1-second binning instead of TR (1.5s) binning.
This addresses reviewer concerns about temporal resolution.
"""

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
from kneed import KneeLocator
import matplotlib.pyplot as plt

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
    log_filename = os.path.join(log_dir, f'{task}_{task_id}_bin_1s.log')
    logging.basicConfig(filename=log_filename, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    return log_filename

def seconds2bins(resp_seconds, bin_size=1.0):
    """Convert seconds to 1-second bins."""
    logging.info("Converting seconds to 1s bins")
    resp_seconds = np.array(resp_seconds)
    resp_bins = np.round(resp_seconds / bin_size).astype(int)
    return np.unique(resp_bins)

def convert2binary(lst, size):
    logging.info("Converting list to binary array")
    binary_array = np.zeros(size, dtype=int)
    valid_indices = np.clip(lst, 0, size-1)
    binary_array[valid_indices] = 1
    return binary_array

def convert_to_bins(df, bin_size=1.0, n_bins=698):
    """Convert response times to 1-second bins."""
    logging.info("Converting response times to 1s bins")
    df['key_resp_2.rt'] = df['key_resp_2.rt'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df['key_resp_2.bin'] = df['key_resp_2.rt'].apply(lambda x: seconds2bins(x, bin_size))
    df['binary_response'] = df['key_resp_2.bin'].apply(lambda x: convert2binary(x, n_bins))
    return df[['key_resp_2.bin', 'binary_response', 'participant']].copy()

def get_agreement(df, unit='bin'):
    """Calculate agreement using 1-second bins."""
    logging.info("Calculating agreement")
    response_col = 'key_resp_2.bin'
    all_responses = np.concatenate(df[response_col].dropna().values)
    response_counts = Counter(all_responses)
    agreement_df = pd.DataFrame(list(response_counts.items()), columns=[unit, 'count']).sort_values(by=unit).reset_index(drop=True)
    agreement_df['agreement'] = agreement_df['count'] / df.shape[0]

    # Filter to story range (21s to 698s - matching 697.5 seconds rounded up)
    agreement_df = agreement_df[(agreement_df[unit] >= 21) & (agreement_df[unit] < 698)]

    return agreement_df.reset_index(drop=True)

def filter_elements(A, list_B, delta=3):
    """
    Filters a given element A based on its proximity to elements in a sorted list B.
    Delta is in seconds (3 seconds = 2 TRs).
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
            seg_index = seg_df[seg_df['onset_seconds'] < r].index[-1]
            seg = seg_df.iloc[:seg_index]['seg'].tolist()
        elif i == len(boundaries) - 1:  # Last iteration
            prev_r = boundaries[i-1]
            prev_seg_index = seg_df[seg_df['onset_seconds'] <= prev_r].index[-1]
            seg = seg_df.iloc[prev_seg_index:]['seg'].tolist()
        else:  # Subsequent iterations
            prev_r = boundaries[i-1]
            prev_seg_index = seg_df[seg_df['onset_seconds'] <= prev_r].index[-1]
            seg_index = seg_df[seg_df['onset_seconds'] < r].index[-1]
            seg = seg_df.iloc[prev_seg_index:seg_index]['seg'].tolist()

        result.append({'boundary': r, 'segments': " ".join(seg)})
    pd.DataFrame(result).to_csv(filepath, index=False)
    print(f"Segments saved to {filepath}")

def map_boundary2word(word_df, boundaries, filepath):
    events = []
    for i, n in enumerate(boundaries):
        start = 21 if i == 0 else boundaries[i-1]
        end = word_df["bin"].max() if i == len(boundaries)-1 else n
        text = " ".join(word_df.loc[(word_df["bin"] >= start) & (word_df["bin"] <= end), "text"].values)
        events.append({"start_bin": start, "end_bin": end, "text": text})
    pd.DataFrame(events).to_csv(filepath, index=False)
    print(f"Events saved to {filepath}")

def process_thresholds(threshold_list, agree_df, output_prefix, word_df, seg_df, output_dir, filtered=False):
    peaks_info = []

    for th in threshold_list:
        prominent_peaks, _ = find_peaks(agree_df['agreement'], prominence=th)
        print(f"There are {len(prominent_peaks)} prominent (threshold = {th}) peaks in {output_prefix}")

        formatted_th = f"{th:.2f}".replace('.', '')

        # Create the filepath for the event dataframe CSV
        filepath = os.path.join(output_dir, f"event_df_prom{output_prefix}_{formatted_th}.csv")
        print(f"setup event dataframe at {filepath}")

        # Determine boundaries and append the final bin value
        boundaries = agree_df['bin'].iloc[prominent_peaks].tolist()

        # Save the prominent peaks to a text file
        np.savetxt(os.path.join(output_dir, f"prominent_peaks{output_prefix}_{formatted_th}.txt"), boundaries, fmt='%d')

        boundaries.append(698)  # 698 seconds (465 TRs * 1.5 = 697.5, rounded up)

        if not filtered:
            map_boundary2word(word_df, boundaries, filepath)
        else:
            map_boundary2seg(seg_df, boundaries, filepath)

        # Save threshold and number of peaks to the list
        peaks_info.append((th, len(prominent_peaks)))

    return peaks_info

def plot_peaks_vs_thresholds(data, filepath, width=15, height=8, label_fontsize=15, tick_fontsize=13, dpi=100):
    """
    Plots the number of prominent peaks vs. agreement threshold and identifies the elbow point.
    Saves the plot as an SVG file with specified pixel dimensions.
    """

    # Extracting the columns
    thresholds = data['threshold']
    n_peaks = data['n_peaks']

    # Identify the elbow point using Kneedle algorithm
    kneedle = KneeLocator(thresholds, n_peaks, curve='convex', direction='decreasing', S=2.0)
    elbow_point = kneedle.elbow


    # Creating the line plot
    plt.figure(figsize=(width, height), dpi=dpi)
    plt.plot(thresholds, n_peaks, marker='o', linestyle='-', label='Number of Prominent Peaks')
    if elbow_point:
        plt.axvline(elbow_point, color='r', linestyle='--', label=f'Elbow Point: Prominence = {elbow_point:.2f}')
    plt.xlabel('Prominence', fontsize=label_fontsize)
    plt.ylabel('Number of Prominent Peaks', fontsize=label_fontsize)
    plt.xticks(np.arange(min(thresholds), max(thresholds) + 0.01, 0.01), fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.legend(fontsize=label_fontsize)
    plt.savefig(filepath, format='svg', dpi=dpi)
    plt.close()
    # Print the elbow point for reference
    logging.info(f'Elbow point: {elbow_point}')

def main(df1, df2a, df2b, word_df, seg_df, output_dir):
    logging.info("Starting main process with 1-second binning")

    # Convert to 1-second bins instead of TRs
    response_df1 = convert_to_bins(df1, bin_size=1.0)
    response_df2a = convert_to_bins(df2a, bin_size=1.0)
    response_df2b = convert_to_bins(df2b, bin_size=1.0)

    response_df1.to_pickle(os.path.join(output_dir, "individual_response_eventseg_1s.pkl"))
    response_df2a.to_pickle(os.path.join(output_dir, "individual_response_evidence_affair_1s.pkl"))
    response_df2b.to_pickle(os.path.join(output_dir, "individual_response_evidence_paranoia_1s.pkl"))

    agree_df1 = get_agreement(response_df1)
    agree_df2a = get_agreement(response_df2a)
    agree_df2b = get_agreement(response_df2b)

    # Convert segment onsets to seconds (already in seconds, just add offset)
    seg_df['onset_seconds'] = seg_df['onset_sec'] + 21  # Story starts at 21s (14 TRs * 1.5)
    sentence_onset_seconds = np.unique(seg_df['onset_seconds'])

    agree_df1['filtered_bins'] = agree_df1['bin'].apply(lambda x: filter_elements(x, sentence_onset_seconds))

    # Filter out rows where 'filtered_bins' is NaN before concatenation
    all_filtered_bins = np.concatenate(
        [np.repeat(row['filtered_bins'], int(row['count']))
        for _, row in agree_df1.iterrows()
        if not np.isnan(row['filtered_bins'])]
    )

    filtered_response_counts = Counter(all_filtered_bins)
    agree_df1_filtered = pd.DataFrame(list(filtered_response_counts.items()), columns=['bin', 'count'])
    agree_df1_filtered = agree_df1_filtered.sort_values(by='bin').reset_index(drop=True)
    agree_df1_filtered['agreement'] = agree_df1_filtered['count'] / df1.shape[0]

    agree_df1.to_pickle(os.path.join(output_dir, "agreement_eventseg_1s.pkl"))
    agree_df1_filtered.to_pickle(os.path.join(output_dir, "agreement_eventseg_filtered_1s.pkl"))
    agree_df2a.to_pickle(os.path.join(output_dir, "agreement_evidence_affair_1s.pkl"))
    agree_df2b.to_pickle(os.path.join(output_dir, "agreement_evidence_paranoia_1s.pkl"))

    # Convert word onsets to 1-second bins
    word_df["bin"] = word_df["onset"].apply(lambda x: np.round(x/1.0)).astype(int)

    th = np.around(agree_df1['agreement'].quantile(0.85), decimals=2)
    th2 = np.around(agree_df1['agreement'].quantile(0.95), decimals=2)
    th_filtered = np.around(agree_df1_filtered['agreement'].quantile(0.85), decimals=2)
    th_filtered2 = np.around(agree_df1_filtered['agreement'].quantile(0.95), decimals=2)

    th_lst = [round(t, 2) for t in np.arange(th-0.05, th2, 0.01)]
    th_filtered_lst = [round(t, 2) for t in np.arange(th_filtered-0.05, th_filtered2, 0.01)]

    peak_th = process_thresholds(th_lst, agree_df1, "", word_df, seg_df, output_dir, filtered=False)
    peak_th_filtered = process_thresholds(th_filtered_lst, agree_df1_filtered, "_filtered", word_df, seg_df, output_dir, filtered=True)

    # Convert the lists to NumPy arrays
    peak_th_array = np.array(peak_th, dtype=[('threshold', 'f4'), ('n_peaks', 'i4')])
    peak_th_filtered_array = np.array(peak_th_filtered, dtype=[('threshold', 'f4'), ('n_peaks', 'i4')])

    figurepath1 = os.path.join(output_dir, "elbow_prominent_peaks_eventseg_1s.svg")
    plot_peaks_vs_thresholds(peak_th_array, figurepath1)

    figurepath2 = os.path.join(output_dir, "elbow_prominent_peaks_eventseg_filtered_1s.svg")
    plot_peaks_vs_thresholds(peak_th_filtered_array, figurepath2)

    # Save the justification data as NumPy arrays
    np.save(os.path.join(output_dir, 'n_peak_prom_threshold_1s.npy'), peak_th_array)
    np.save(os.path.join(output_dir, 'n_peak_prom_threshold_filtered_1s.npy'), peak_th_filtered_array)

if __name__ == '__main__':
    logging.info("Loading environment variables")
    load_dotenv()
    base_dir = os.getenv("BASE_DIR")
    scratch_dir = os.getenv("SCRATCH_DIR")

    if not base_dir:
        logging.error("BASE_DIR environment variable not set")
        raise EnvironmentError("BASE_DIR environment variable not set")

    data_dir = os.path.join(scratch_dir, "data")
    behav_dir = os.path.join(data_dir, "behav")
    stimuli_dir = os.path.join(data_dir, "stimuli")
    output_dir = os.path.join(scratch_dir, "output_RR", "08_behav_results_bin_1s")
    os.makedirs(output_dir, exist_ok=True)

    log_filename = setup_logging(output_dir, 'behav_proc', 'bin_1s')

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

    log_filename = setup_logging(output_dir, 'behav_proc', 'bin_1s')
    main(df1, df2a, df2b, word_df, seg_df, output_dir)