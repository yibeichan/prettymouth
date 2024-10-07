# in this script, we will:
# 1. load each subject's data per network
# 2. concatenate the data across subjects and get the group data for each network
# 3. get HMM states for this one group data per network

import os
import glob
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.decomposition import PCA
from hmmlearn import hmm
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from argparse import ArgumentParser
import multiprocessing
from functools import partial
from memory_profiler import profile

@profile
def fit_model_with_multiple_inits(train_seq, test_seq, train_lengths, test_lengths, n_components, covariance_type, n_iter, random_state, n_inits=10):
    best_model = None
    best_score = float('-inf')
    
    for init in range(n_inits):
        model = hmm.GaussianHMM(
            n_components=n_components,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=random_state + init
        )
        
        try:
            model.fit(train_seq, lengths=train_lengths)
            score = model.score(test_seq, lengths=test_lengths)
            
            if score > best_score:
                best_score = score
                best_model = model
        
        except Exception as e:
            print(f"Error with initialization {init}: {e}")
            continue
    
    return best_model, best_score

@profile
def load_network_data(network):
    data_dir = "/nese/mit/group/sig/projects/yibei/prettymouth/output/network_data"
    files = glob.glob(os.path.join(data_dir, f"*{network}*.npy"))
    all_data = []
    for file in files:
        data = np.load(file)
        all_data.append(data)
    return np.array(all_data)

@profile
def process_network(network_name, data_pca, n_components, n_subj, n_TR, n_components_pca, n_params, covariance_type='full', random_state=42):
    print(f"\nProcessing network: {network_name}, n_components: {n_components}")
    
    # Prepare Leave-One-Out cross-validation
    loo = LeaveOneOut()
    
    results = {
        'n_components': n_components,
        'subject': [],
        'log_likelihood': [],
        'AIC': [],
        'BIC': []
    }
    
    for train_index, test_index in loo.split(range(n_subj)):
        train_data = data_pca[train_index]
        test_data = data_pca[test_index]
        
        # Reshape data for HMM
        train_seq = train_data.reshape(-1, n_components_pca)
        test_seq = test_data.reshape(-1, n_components_pca)
        train_lengths = [n_TR] * len(train_index)
        test_lengths = [n_TR]
        
        best_model, log_likelihood = fit_model_with_multiple_inits(
            train_seq, test_seq, train_lengths, test_lengths,
            n_components, covariance_type, n_iter=1000, random_state=random_state
        )
        
        if best_model is None:
            print(f"Failed to fit model with {n_components} components for subject {test_index[0]}")
            continue
        
        AIC = 2 * n_params - 2 * log_likelihood
        BIC = np.log(n_TR) * n_params - 2 * log_likelihood
        
        results['subject'].append(test_index[0])
        results['log_likelihood'].append(log_likelihood)
        results['AIC'].append(AIC)
        results['BIC'].append(BIC)
        
        print(f"Subject {test_index[0]}: Log Likelihood = {log_likelihood:.2f}, AIC = {AIC:.2f}, BIC = {BIC:.2f}")
    
    return results

@profile
def run_parallel_hmm(network_name, all_data, n_components_range, covariance_type='full', random_state=42):
    data = all_data[:, :, 17:468]  # Consider making these indices parameters
    n_subj, n_parcel, n_TR = data.shape
    
    # Apply PCA
    pca = PCA(n_components=0.95)  # Retain 95% of variance
    data_flat = data.reshape(n_subj * n_TR, n_parcel)
    data_pca = pca.fit_transform(data_flat)
    n_components_pca = data_pca.shape[1]
    print(f"Number of PCA components: {n_components_pca}")
    
    # Reshape data back to (n_subj, n_TR, n_components_pca)
    data_pca = data_pca.reshape(n_subj, n_TR, n_components_pca)
    
    # num_cores = multiprocessing.cpu_count()
    num_cores = 16
    pool = multiprocessing.Pool(processes=num_cores)
    
    results = []
    for n_components in n_components_range:
        n_params = (
            n_components * (n_components - 1) +
            n_components - 1 +
            n_components * n_components_pca +
            n_components * n_components_pca * (n_components_pca + 1) / 2
        )
        partial_process = partial(process_network, network_name, data_pca, n_components, n_subj, n_TR, n_components_pca, n_params, covariance_type=covariance_type, random_state=random_state)
        results.append(pool.apply_async(partial_process))
    
    pool.close()
    pool.join()
    
    return [r.get() for r in results]

@profile
def plot_results(network_name, results, save_dir):
    avg_metrics = {
        'n_components': sorted(set(results['n_components'])),
        'mean_log_likelihood': [],
        'mean_AIC': [],
        'mean_BIC': []
    }
    
    for n_components in avg_metrics['n_components']:
        indices = [i for i, x in enumerate(results['n_components']) if x == n_components]
        avg_metrics['mean_log_likelihood'].append(np.mean([results['log_likelihood'][i] for i in indices]))
        avg_metrics['mean_AIC'].append(np.mean([results['AIC'][i] for i in indices]))
        avg_metrics['mean_BIC'].append(np.mean([results['BIC'][i] for i in indices]))
    
    plt.figure(figsize=(10, 6))
    plt.plot(avg_metrics['n_components'], avg_metrics['mean_AIC'], label='AIC')
    plt.plot(avg_metrics['n_components'], avg_metrics['mean_BIC'], label='BIC')
    plt.xlabel('Number of Hidden States')
    plt.ylabel('Metric Value')
    plt.title(f'Model Selection Metrics for {network_name}')
    plt.legend()
    plt.savefig(os.path.join(save_dir, f"{network_name}_hmm_model_selection.png"))
    
    optimal_idx = np.argmin(avg_metrics['mean_BIC'])
    optimal_n_components = avg_metrics['n_components'][optimal_idx]
    print(f"Optimal number of hidden states for {network_name}: {optimal_n_components}")
    
    return optimal_n_components

if __name__ == "__main__":
    load_dotenv()
    parser = ArgumentParser()
    parser.add_argument("--network", type=str, required=True)
    args = parser.parse_args()
    network_name = args.network
    
    nese_dir = os.getenv("NESE_DIR")        
    if nese_dir is None:
        raise EnvironmentError("NESE_DIR environment variable is not set.")
    
    network_data_dir = os.path.join(nese_dir, "output", "network_data")
    save_dir = os.path.join(nese_dir, "output", "hmm_results", "level1", "all_subj")
    os.makedirs(save_dir, exist_ok=True)

    n_components_range = range(3, 26)
    all_data = load_network_data(network_name)
    results = run_parallel_hmm(network_name, all_data, n_components_range)
    
    # Combine results from all parallel processes
    combined_results = {
        'n_components': [],
        'subject': [],
        'log_likelihood': [],
        'AIC': [],
        'BIC': []
    }
    for r in results:
        for key in combined_results.keys():
            combined_results[key].extend(r[key] if key != 'n_components' else [r[key]] * len(r['subject']))
    
    optimal_n_components = plot_results(network_name, combined_results, save_dir)
    # save results and optimal_n_components
    np.save(os.path.join(save_dir, f"{network_name}_hmm_results.npy"), combined_results)
    np.save(os.path.join(save_dir, f"{network_name}_hmm_optimal_n_components.npy"), optimal_n_components)