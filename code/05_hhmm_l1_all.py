import numpy as np
from hmmlearn.vhmm import VariationalGaussianHMM
from sklearn.model_selection import KFold
import multiprocessing as mp
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Optional
import os
import json
from datetime import datetime
import logging
from time import time
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv
from argparse import ArgumentParser

@dataclass
class FoldData:
    """Data structure for cross-validation fold processing"""
    combined_data: np.ndarray
    n_states: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    n_init: int

def setup_logging(output_dir: str) -> None:
    """Configure logging with enhanced formatting"""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'hmm_analysis.log')),
            logging.StreamHandler()
        ]
    )

def load_and_preprocess_data(data_dir: str, network: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load and preprocess network data with vectorized operations"""
    group1_path = os.path.join(data_dir, f'affair_{network}_data.npy')
    group2_path = os.path.join(data_dir, f'paranoia_{network}_data.npy')
    
    if not (os.path.exists(group1_path) and os.path.exists(group2_path)):
        raise FileNotFoundError(f"Data files not found for network {network}")
    
    group1_data = np.load(group1_path)
    group2_data = np.load(group2_path)
    
    # Validate data
    for group_name, data in [("Group1", group1_data), ("Group2", group2_data)]:
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            raise ValueError(f"Invalid values found in {group_name} {network} data")
    
    # Vectorized standardization
    def standardize_group(data):
        orig_shape = data.shape
        data_2d = data.reshape(-1, orig_shape[-1])
        scaler = StandardScaler()
        data_2d = scaler.fit_transform(data_2d)
        return data_2d.reshape(orig_shape)
    
    group1_data = standardize_group(group1_data)
    group2_data = standardize_group(group2_data)
    
    logging.info(f"Loaded {network} data - Shape: G1 {group1_data.shape}, G2 {group2_data.shape}")
    return group1_data, group2_data

def setup_vhmm_priors(
    n_components: int, 
    n_features: int,
    tr: float = 1.5,
    expected_state_duration: float = 10.0
) -> Dict[str, np.ndarray]:
    """Setup informed priors for VariationalGaussianHMM based on actual parameters
    
    Args:
        n_components: Number of hidden states
        n_features: Number of observed features
        tr: Repetition time in seconds
        expected_state_duration: Expected duration of states in seconds
    """
    # Prior for initial state distribution
    startprob_prior = np.ones(n_components)
    
    # Prior for transition matrix - adjusted based on expected duration
    expected_transitions = expected_state_duration / (tr*3)
    transmat_prior = np.ones((n_components, n_components))
    np.fill_diagonal(transmat_prior, expected_transitions)
    
    # Means prior (location prior)
    means_prior = np.zeros((n_components, n_features))
    
    # Beta prior (precision scalar on mean prior)
    beta_prior = np.ones(n_components)
    
    # Degrees of freedom for the Wishart prior
    dof_prior = np.full(n_components, n_features + 2)
    
    # Scale matrix for the Wishart prior - Changed to 3D array
    scale_prior = np.array([np.eye(n_features) for _ in range(n_components)])
    
    return {
        'startprob_prior': startprob_prior,
        'transmat_prior': transmat_prior,
        'means_prior': means_prior,
        'beta_prior': beta_prior,
        'dof_prior': dof_prior,
        'scale_prior': scale_prior  # Now has shape (n_components, n_features, n_features)
    }

def calculate_model_scores(model: VariationalGaussianHMM, 
                         train_data: np.ndarray,
                         test_data: np.ndarray) -> Dict[str, float]:
    """Calculate multiple criteria for model selection"""
    # Primary criterion: Free Energy
    train_ll = model.score(train_data)
    test_ll = model.score(test_data)
    free_energy = -train_ll + model.monitor_.convergence_monitor.history[-1]
    
    # State uncertainty from test data
    posteriors = model.predict_proba(test_data)
    state_uncertainty = -np.mean(np.sum(posteriors * np.log(posteriors + 1e-10), axis=1))
    
    return {
        'free_energy': free_energy,
        'train_ll': train_ll,
        'test_ll': test_ll,
        'state_uncertainty': state_uncertainty
    }

def process_fold(fold_data: FoldData) -> Tuple[Dict[str, Any], Dict[str, int], Optional[VariationalGaussianHMM]]:
    n_features = fold_data.combined_data.shape[-1]
    
    train_data = fold_data.combined_data[fold_data.train_idx].reshape(-1, n_features)
    test_data = fold_data.combined_data[fold_data.test_idx].reshape(-1, n_features)
    
    best_score = {'free_energy': np.inf, 'train_ll': -np.inf, 'test_ll': -np.inf}
    best_model = None
    fold_stats = {'convergence_failures': 0, 'successful_fits': 0}
    
    priors = setup_vhmm_priors(fold_data.n_states, n_features)
    
    for init in range(fold_data.n_init):
        try:
            model = VariationalGaussianHMM(
                n_components=fold_data.n_states,
                covariance_type='full',
                n_iter=100,
                random_state=42 + init,
                implementation='log',
                **priors  # Now includes the correct prior parameters
            )
            
            model.fit(train_data)
            scores = calculate_model_scores(model, train_data, test_data)
            
            if scores['free_energy'] < best_score['free_energy']:
                best_score = scores
                best_model = model
            
            fold_stats['successful_fits'] += 1
            
        except Exception as e:
            fold_stats['convergence_failures'] += 1
            logging.warning(f"Convergence failure in fold (init {init}): {str(e)}")
            continue
    
    return best_score, fold_stats, best_model

def parallel_optimize_states(combined_data: np.ndarray, 
                           max_states: int = 10,
                           n_splits: int = 5,
                           n_init: int = 5) -> Tuple[int, Dict[str, Any]]:
    """Parallel optimization of HMM states using subject-level cross-validation"""
    n_subj = combined_data.shape[0]
    
    # Create shuffled subject indices
    subject_indices = np.arange(n_subj)
    np.random.shuffle(subject_indices)
    
    # Setup cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Generate fold data objects
    fold_data_list = [
        FoldData(
            combined_data=combined_data[subject_indices],
            n_states=n_states,
            train_idx=train_idx,
            test_idx=test_idx,
            n_init=n_init
        )
        for n_states in range(2, max_states + 1)
        for train_idx, test_idx in kf.split(range(n_subj))
    ]
    
    # Process folds in parallel with progress bar
    n_jobs = 16
    with mp.Pool(processes=n_jobs) as pool:
        results = list(tqdm(
            pool.imap(process_fold, fold_data_list),
            total=len(fold_data_list),
            desc="Processing state-fold combinations"
        ))
    
    # Organize results by number of states
    metrics = {}
    for i, (score, fold_stats, model) in enumerate(results):
        n_states = (i // n_splits) + 2
        if n_states not in metrics:
            metrics[n_states] = {
                'free_energy_scores': [],
                'train_ll_scores': [],
                'test_ll_scores': [],
                'state_uncertainties': [],
                'successful_fits': 0,
                'convergence_failures': 0
            }
        
        if score['train_ll'] > -np.inf:  # Valid result
            metrics[n_states]['free_energy_scores'].append(score['free_energy'])
            metrics[n_states]['train_ll_scores'].append(score['train_ll'])
            metrics[n_states]['test_ll_scores'].append(score['test_ll'])
            metrics[n_states]['state_uncertainties'].append(score['state_uncertainty'])
        
        metrics[n_states]['successful_fits'] += fold_stats['successful_fits']
        metrics[n_states]['convergence_failures'] += fold_stats['convergence_failures']
    
    # Calculate summary metrics for each number of states
    summary_metrics = {}
    for n_states, scores in metrics.items():
        if len(scores['free_energy_scores']) == n_splits:
            summary_metrics[n_states] = {
                'mean_free_energy': np.mean(scores['free_energy_scores']),
                'std_free_energy': np.std(scores['free_energy_scores']),
                'mean_train_ll': np.mean(scores['train_ll_scores']),
                'mean_test_ll': np.mean(scores['test_ll_scores']),
                'mean_state_uncertainty': np.mean(scores['state_uncertainties']),
                'successful_fits': scores['successful_fits'],
                'convergence_failures': scores['convergence_failures']
            }
    
    # Select optimal number of states based on free energy
    if summary_metrics:
        optimal_states = min(summary_metrics.keys(),
                           key=lambda k: summary_metrics[k]['mean_free_energy'])
    else:
        optimal_states = None
    
    return optimal_states, summary_metrics

def process_network(network_data: Tuple[str, str, int]) -> Tuple[str, Dict[str, Any]]:
    """Process a single network"""
    data_dir, network, max_states = network_data
    try:
        group1_data, group2_data = load_and_preprocess_data(data_dir, network)
        combined_data = np.concatenate([group1_data, group2_data], axis=0)
        
        optimal_states, metrics = parallel_optimize_states(
            combined_data,
            max_states=max_states,
            n_splits=5,
            n_init=5
        )
        
        if optimal_states is not None:
            # Fit final model with optimal states
            n_features = combined_data.shape[1]
            expected_state_duration = 10.0  # seconds
            tr = 1.5  # Adjust based on your acquisition parameters
            
            priors = setup_vhmm_priors(
                n_components=optimal_states, 
                n_features=n_features,
                tr=tr,
                expected_state_duration=expected_state_duration
            )
            
            final_model = VariationalGaussianHMM(
                n_components=optimal_states,
                covariance_type='full',
                n_iter=200,
                random_state=42,
                implementation='log',
                **priors
            ).fit(combined_data.reshape(-1, n_features))
            
            # Calculate final scores
            final_scores = calculate_model_scores(
                final_model, 
                combined_data.reshape(-1, n_features),
                combined_data.reshape(-1, n_features)  # Using all data for final evaluation
            )
            
            # Get posteriors and state sequences
            posteriors = {
                'means': final_model.means_posterior_,
                'covars': final_model.covars_,
                'transmat': final_model.transmat_posterior_,
                'startprob': final_model.startprob_posterior_
            }
            
            group1_states = [
                final_model.predict(subj_data.T) 
                for subj_data in group1_data
            ]
            group2_states = [
                final_model.predict(subj_data.T) 
                for subj_data in group2_data
            ]
            
            return network, {
                'optimal_states': optimal_states,
                'metrics': metrics,
                'final_scores': final_scores,
                'posteriors': posteriors,
                'group1_states': group1_states,
                'group2_states': group2_states
            }
        
        return network, {'error': 'No valid models found'}
        
    except Exception as e:
        logging.error(f"Error processing network {network}: {str(e)}", exc_info=True)
        return network, {'error': str(e)}

def main(data_dir: str, output_dir: str, network: str, max_states: int = 50) -> None:
    """Main function for processing a single network"""
    start_time = time()
    setup_logging(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Process single network
    result = process_network((data_dir, network, max_states))
    network, network_result = result
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = os.path.join(output_dir, f'vhmm_analysis_{network}_{timestamp}.json')
    
    serializable_results = {
        'global_stats': {
            'total_time': time() - start_time,
            'network_processed': 1 if 'error' not in network_result else 0,
            'network_failed': 1 if 'error' in network_result else 0
        },
        'network_results': {}
    }
    
    if 'error' not in network_result:
        serializable_results['network_results'][network] = {
            'optimal_states': network_result['optimal_states'],
            'metrics': {
                k: {mk: float(mv) for mk, mv in v.items()}
                for k, v in network_result['metrics'].items()
            },
            'final_scores': {k: float(v) for k, v in network_result['final_scores'].items()},
            'posteriors': {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in network_result['posteriors'].items()
            },
            'group1_states': [s.tolist() for s in network_result['group1_states']],
            'group2_states': [s.tolist() for s in network_result['group2_states']]
        }
    else:
        serializable_results['network_results'][network] = {'error': network_result['error']}
    
    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    logging.info(f"\nAnalysis completed in {serializable_results['global_stats']['total_time']:.2f} seconds")
    logging.info(f"Network processed: {network}")
    if 'error' in network_result:
        logging.error(f"Network failed with error: {network_result['error']}")

if __name__ == "__main__":
    load_dotenv()
    scratch_dir = os.getenv("SCRATCH_DIR")

    parser = ArgumentParser(description='HMM Analysis for neuroimaging data')
    parser.add_argument("--res", type=str, default="native", 
                       help="Resolution for analysis")
    parser.add_argument("--network", type=str, required=True,
                       help="Network name to process")
    args = parser.parse_args()

    data_dir = os.path.join(scratch_dir, "output", f"atlas_masked_{args.res}",
                                              "networks")
    output_dir = os.path.join(scratch_dir, "output", f"vhmm_l1_{args.res}")
    os.makedirs(output_dir, exist_ok=True)

    main(data_dir, output_dir, args.network)