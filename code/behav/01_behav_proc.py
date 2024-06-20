# ignore warnings 
import warnings
warnings.filterwarnings('ignore')
import os
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import ast
# import Counter
from collections import Counter
from scipy.signal import find_peaks

# resample seconds to TR
def seconds2TRs(resp_seconds):
    resp_seconds = np.array(resp_seconds)
    resp_TRs = np.ceil(resp_seconds / 1.5)
    resp_TRs = np.unique(resp_TRs)
    return resp_TRs

# get agreement 
def get_agreement(response_col, unit, df):
    all_responses = []
    for i, row in enumerate(df[response_col]):
        # Skip over None or NaN values
        if row is None or (isinstance(row, float) and np.isnan(row)):
            continue
        all_responses.extend(row)
    # Count the occurrences of each unique element
    response_counts = Counter(all_responses)
    # Create a new DataFrame
    agreement_df = pd.DataFrame(list(response_counts.items()), columns=[unit, 'count'])
    # drop nan based on TR
    agreement_df = agreement_df.dropna(subset=[unit])
    # Sort by TR for better readability
    agreement_df = agreement_df.sort_values(by=unit).reset_index(drop=True)
    agreement_df['agreement'] = agreement_df['count'] / df.shape[0]
    # remove the first 14 TRs/21s and the last 3 TRs
    if unit == 'TR':
        agreement_df = agreement_df[(agreement_df[unit] > 14)&(agreement_df[unit] < 467)]
    elif unit == 'second':
        agreement_df = agreement_df[(agreement_df[unit] > 21)&(agreement_df[unit] < 699)]
    else:
        print("Please enter a valid unit: TR or second")
    # reset index
    agreement_df = agreement_df.reset_index(drop=True)
    return agreement_df

# we need to convert countered responses to a time series dataframe
def get_tsdf(df, unit):
    df2 = pd.DataFrame(columns=[unit, "agreement"])
    min_val = 1
    max_val = df[unit].max()+1
    df2[unit] = np.arange(min_val, max_val, 1)
    for i, row in enumerate(df2[unit]):
        if row in df[unit].values:
            df2.at[i, "agreement"] = df[df[unit] == row]["agreement"].values[0]
        else:
            df2.at[i, "agreement"] = 0
    # convert to float
    df2["agreement"] = df2["agreement"].astype(float)
    return df2

def create_event_df(prom, prominent_peaks, story_df, output_dir):
    event_df = pd.DataFrame(columns=["start_TR", "end_TR", "text"])
    for i, n in enumerate(prominent_peaks):
        if i == 0:
            start = 14
        else:
            start = prominent_peaks[i-1]
        if i == len(prominent_peaks)-1:
            end = story_df["TR"].max()
        else:
            end = n
        text = story_df.loc[(story_df["TR"] >= start) & (story_df["TR"] <= end), "text"].values
        text = " ".join(text)
        event_df = event_df.append({"start_TR": start, "end_TR": end, "text": text}, ignore_index=True)
    event_df.to_csv(os.path.join(output_dir, f"event_df_{int(prom*100):03}.csv"), index=False)

def main():
    load_dotenv()
    base_dir = os.getenv("BASE_DIR")
    data_dir = os.path.join(base_dir, "data")
    behav_data_dir = os.path.join(data_dir, "behav")
    stimuli_dir = os.path.join(base_dir, "stimuli")
    output_dir = os.path.join(base_dir, "output", "behav_results")
    os.makedirs(output_dir, exist_ok=True)

    pavlovia_df = pd.read_excel(os.path.join(behav_data_dir, "pavlovia.xlsx"))
    postsurvey_df = pd.read_csv(os.path.join(behav_data_dir, "postsurvey.csv"))
    prolific_df = pd.read_csv(os.path.join(behav_data_dir, "prolific_ids.csv"))
    print(pavlovia_df.shape, postsurvey_df.shape, prolific_df.shape)

    pp_ids = prolific_df["Participant id"].values
    # select rows in pavlovia_df that have pp_ids in prolific_df
    pavlovia_df2 = pavlovia_df[pavlovia_df["participant"].isin(pp_ids)]
    # do the same for postsurvey_df
    postsurvey_df2 = postsurvey_df[postsurvey_df["participant"].isin(pp_ids)]
    postsurvey_df3 = postsurvey_df2[['ch1', 'ch2', 'ch3', 'recall', 'engagement', 'clarity', 'age', 'gender', 'participant']]
    print(pavlovia_df2.shape, postsurvey_df3.shape)

    # merge the two dataframes based on the participant column
    merged_df = pd.merge(pavlovia_df2, postsurvey_df3, on="participant")
    print(merged_df.shape)

    # Remove BOM
    merged_df['key_resp_2.rt'] = merged_df['key_resp_2.rt'].str.replace('\ufeff', '', regex=False)
    # Convert string representation of lists to actual lists
    merged_df['event_seg'] = merged_df['key_resp_2.rt'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else x)
    # Filter elements with exactly 3 digits after the decimal point using NumPy
    merged_df['event_seg'] = merged_df['event_seg'].apply(
        lambda x: np.around(x, decimals=3)
    )
    merged_df['event_seg_TRs'] = merged_df['event_seg'].apply(seconds2TRs)
    agree_TR_df = get_agreement('event_seg_TRs', 'TR', merged_df)
    agree_TR_df2 = get_tsdf(agree_TR_df, "TR")

    stimuli_file = os.path.join(stimuli_dir, "word_by_onset.csv")
    story_df = pd.read_csv(stimuli_file)
    story_df["TR"] = story_df["onset"].apply(lambda x: np.ceil(x/1.5)).astype(int)

    for prom in [0.08, 0.09, 0.1, 0.11, 0.12]:
        prominent_peaks, _ = find_peaks(agree_TR_df2['agreement'], prominence=prom, distance=5)
        print(f"There are {len(prominent_peaks)} prominent (agreement = {prom}) peaks in agreement")
        # save the peaks to a txt file
        np.savetxt(os.path.join(output_dir, f"prominent_peaks_{int(prom*100):03}.txt"), prominent_peaks, fmt='%d')
        # create event_df
        create_event_df(prom, prominent_peaks, story_df, output_dir)

if __name__ == '__main__':
    main()