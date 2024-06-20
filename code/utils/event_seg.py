import os
import numpy as np
import pandas as pd
import pickle

def process_event(n_event, ts_data, n_sub):
    from brainiak.eventseg.event import EventSegment
    from scipy.stats import zscore
    print('n_event: ', n_event)
    print('running train data')
    data_train = ts_data[:n_sub//2, :, :]
    if data_train.size > 0:
        data_train_mean = np.nanmean(data_train, axis=0).T
        data_train_mean_zs = zscore(data_train_mean, axis=0)
        # remove the whole row if there is nan in any column
        data_train_mean_zs = data_train_mean_zs[~np.isnan(data_train_mean_zs).any(axis=1), :]
        # check if how many rows are left, if less than 10, skip this n_event
        if data_train_mean_zs.shape[0] < 10:
            raise ValueError("Not enough data points for training.")
        else:         
            data_HMM = EventSegment(n_events=n_event, split_merge=True)
            data_HMM.fit(data_train_mean_zs)
    else:
        raise ValueError("Training data is empty.")

    print('running test data')
    data_test = ts_data[n_sub//2:, :, :]
    if data_test.size > 0:
        data_test_mean = np.nanmean(data_test, axis=0).T
        # fill nan with 0
        data_test_mean_zs = zscore(data_test_mean, axis=0)
        # remove the whole row if there is nan in any column
        data_test_mean_zs[~np.isnan(data_test_mean_zs).any(axis=1), :]
        # check how many rows are left, if less than 10, skip this n_event
        if data_test_mean_zs.shape[0] < 10:
            raise ValueError("Not enough data points for testing.")
        else:
            _, test_ll = data_HMM.find_events(data_test_mean_zs)
            print('test_ll: ', test_ll)
            # check whether there is nan in test_ll, if so, raise error
            if np.isnan(test_ll):
                raise ValueError("NaN value found in test_ll.")
    else:
        raise ValueError("Testing data is empty.")

    return test_ll

def get_HMM_test(ts_data, n_event_list):
    from multiprocessing import Pool
    n_sub = ts_data.shape[0]
    with Pool(2) as p:
        test_ll = p.starmap(process_event, [(n_event, ts_data, n_sub) for n_event in n_event_list])
    return np.array(test_ll)

def get_test_ll_df(files, roi_name):
    df = pd.DataFrame(columns=[roi_name])
    df["n_event"] = 0
    df = df.set_index("n_event")
    for file in files:
        with open(file, 'rb') as f:
            ll_interval, test_ll = pickle.load(f)
            for e, l in zip(ll_interval, test_ll):
                df.at[e, roi_name] = l
        # sort the index
        df = df.sort_index()
        # df = df.fillna(-100000000)
    return df

def get_ll_interval(df, roi_name):
    """
    get the interval of the max value in test_ll_df
    param: df: pd.DataFrame of test log likelihood
    """
    # Check if roi_name is a valid column in df
    if roi_name not in df.columns:
        raise ValueError(f"Column '{roi_name}' not found in DataFrame.")

    # Convert df to float and handle possible exceptions
    try:
        df = df.astype(float)
    except ValueError as e:
        raise ValueError("Could not convert DataFrame to float.") from e

    max_ll_dict = df.max(axis=0).to_dict()
    ll_interval_df = pd.DataFrame(df.idxmax(axis=0)).T

    col = roi_name
    max_ll = max_ll_dict[col]
    min_ll = max_ll - 0.1 * abs(max_ll)
    idice = df[(df[col] <= max_ll) & (df[col] >= min_ll)][col].index

    for i in idice:
        ll_interval_df = ll_interval_df.append({col: i}, ignore_index=True)

    ll_interval_df = ll_interval_df.reset_index(drop=True)

    for idx, row in ll_interval_df.iterrows():
        while True:
            value = row[col]
            # Ensure value is numeric
            if isinstance(value, str):
                try:
                    value = float(value)
                except ValueError:
                    raise ValueError(f"Cannot convert value '{value}' to float.")

            if value in df.index and not np.isnan(df.loc[value, col]):
                if value > 1:
                    ll_interval_df.at[idx, col] = value - 1
                else:
                    ll_interval_df.at[idx, col] = 0
            else:
                ll_interval_df.at[idx, col] = value
                break

    ll_interval = ll_interval_df[col].tolist()

    ll_interval = list(set([int(i) for i in ll_interval if not np.isnan(i) and i != 0]))

    ll_interval.sort(reverse=True)

    return ll_interval

def plot_test_ll(network_test_ll, n_event_lists, output_dir, movie_name, n_parcel, stage):
    """
    This function plots the test log-likelihood of each network for different number of events.
    1. plot each network and make into a gif
    2. plot all networks in one figure
    """
    import imageio
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style("whitegrid", {'axes.grid' : False})
    # plot each network and make into a gif
    images = []
    for i, ntw in zip(range(len(network_test_ll)),network_test_ll.keys()):
        # plot test_ll with different number of events
        plt.figure(figsize=(10, 5))
        sns.lineplot(x=n_event_lists[i], y=network_test_ll[ntw])
        plt.xlabel("Number of events")
        plt.ylabel("Log-likelihood")
        plt.title(f"Test log-likelihood of {ntw}")
        filename = os.path.join(output_dir, "HMM",  f"{movie_name}_test", f"{n_parcel}parcel_HMM_test_ll_{ntw}_p{stage:02d}.png")
        plt.savefig(filename, dpi=300)
        plt.close()
        images.append(imageio.imread(filename))
    imageio.mimsave(os.path.join(output_dir, "HMM",  f"{movie_name}_test", f"{n_parcel}parcel_HMM_test_ll_all_p{stage:02d}.gif"), images, duration=1)

    # plot network_test_ll with different number of events
    plt.figure(figsize=(40, 12))
    i = 0
    for k, v in network_test_ll.items():
        # if i is even, plot the line with different style
        if i % 2 == 0:
            # rondomly choose a color
            color = np.random.rand(3,)
            sns.lineplot(x=n_event_lists[i], y=v, label=k,  linestyle='--', color=color)
        else:
            # color = np.random.rand(3,)
            sns.lineplot(x=n_event_lists[i], y=v, label=k)
        i += 1
    plt.xlabel("Number of events")
    plt.ylabel("Log-likelihood")
    plt.title("Test log-likelihood of different networks")
    plt.legend(bbox_to_anchor=(1.005, 1), loc=2, borderaxespad=0.)
    plt.savefig(os.path.join(output_dir, "HMM",  f"{movie_name}_test", f"{n_parcel}parcel_HMM_test_ll_all_p{stage:02d}.png"), dpi=300)
    plt.close()

def plot_test_ll_df(test_ll_df, HMM_test_dir, movie_name, n_parcel, stage):
    import matplotlib.pyplot as plt
    import seaborn as sns
    new_test_ll_df = test_ll_df.copy(deep=True)
    # # remove columns vPFC
    # new_test_ll_df = new_test_ll_df.drop(columns=["vPFC"])
    # plot test_ll_df with n_event as the x-axis, each column is a line
    plt.figure(figsize=(40, 12))
    sns.lineplot(data=new_test_ll_df)
    plt.xlabel("Number of events")
    plt.ylabel("Log-likelihood")
    plt.title("Test log-likelihood of different networks")
    plt.legend(bbox_to_anchor=(1.005, 1), loc=2, borderaxespad=0.)
    # identify the highest point for each line
    for col in new_test_ll_df.columns:
        new_test_ll_df[col] = new_test_ll_df[col].astype(float)
        max_l = new_test_ll_df[col].max()
        max_e = new_test_ll_df[col].idxmax(axis=0)
        plt.scatter(max_e, max_l, color='red')
        plt.text(max_e, max_l, f"{col}: {max_e}, {max_l}", fontsize=12)
    plt.savefig(os.path.join(HMM_test_dir, f"{n_parcel}parcel_HMM_test_ll_df_p{stage:02d}.png"), dpi=300)
    plt.close()