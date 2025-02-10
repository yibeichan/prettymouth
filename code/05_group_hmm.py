import os
from dotenv import load_dotenv
import numpy as np
from argparse import ArgumentParser
import pickle
from hmmlearn import hmm
from scipy.stats import zscore, entropy
from sklearn.metrics import mutual_info_score
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import gc
from typing import List, Tuple, Dict
import time
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json

logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')

class GroupHMM:
    def __init__(self, 
                 n_components_range: Tuple[int, int], 
                 random_state: int, 
                 n_jobs: int, 
                 max_tries: int,
                 tr: float = 1.5):
        """
        Initialize the HMM analysis with configuration parameters.
        
        Args:
            n_components_range: Tuple of (min, max) number of states to test
            random_state: Random seed for reproducibility
            n_jobs: Number of parallel jobs
            max_tries: Maximum attempts for model fitting
            tr: Repetition time in seconds
        """
        if not isinstance(n_components_range, tuple) or len(n_components_range) != 2:
            raise ValueError("n_components_range must be a tuple of (min, max)")
        if n_components_range[0] >= n_components_range[1]:
            raise ValueError("n_components_range[0] must be less than n_components_range[1]")

        self.n_components_range = n_components_range
        self.random_state = random_state
        self.n_jobs = min(n_jobs, cpu_count())
        self.max_tries = max_tries
        self.tr = tr
        self.cv_results = {}
        self.timeout = 3600  # 1 hour default timeout
        self.checkpoint_dir = "checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Initialize storage for metrics
        self.state_durations = {}
        self.transition_stabilities = {}
        self.bic_scores = {}
        self.aic_scores = {}

    def initialize_informed_transition_matrix(self, n_states: int) -> np.ndarray:
        """Initialize transition matrix with state-specific durations"""
        transmat = np.zeros((n_states, n_states))
        
        for i in range(n_states):
            # Duration increases with state index
            avg_duration = (5 + i * 2) / self.tr  # 5s for first state, +2s for each subsequent
            stay_prob = np.exp(-1/avg_duration)
            transmat[i,i] = stay_prob
            # Distribute remaining probability
            transmat[i, [j for j in range(n_states) if j != i]] = \
                (1 - stay_prob) / (n_states - 1)
        
        return transmat

    def preprocess_data(self, group_data: np.ndarray) -> np.ndarray:
        """Z-score and reshape data for HMM"""
        processed_data = np.array([zscore(subject, axis=1, ddof=1) 
                                 for subject in group_data])
        return np.vstack([subj.T for subj in processed_data])

    def fit_single_model(self, data: np.ndarray, n_states: int) -> hmm.GaussianHMM:
        """Fit single HMM model with informed initialization"""
        transmat = self.initialize_informed_transition_matrix(n_states)
        
        model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type='diag',
            n_iter=500,
            random_state=self.random_state,
            params='stmc',
            init_params='',
            tol=1e-2
        )
        
        model.transmat_ = transmat
        model.fit(data)
        return model

    def calculate_information_criteria(self, 
                                    model: hmm.GaussianHMM, 
                                    data: np.ndarray) -> Tuple[float, float]:
        """Calculate AIC and BIC scores for the model"""
        n_samples = data.shape[0]
        n_features = data.shape[1]
        n_states = model.n_components
        
        n_parameters = (n_states * n_features +  # means
                       n_states * n_features +    # diagonal covariances
                       n_states * (n_states - 1) +  # transition matrix
                       (n_states - 1))             # initial probabilities
        
        log_likelihood = model.score(data)
        
        aic = -2 * log_likelihood + 2 * n_parameters
        bic = -2 * log_likelihood + n_parameters * np.log(n_samples)
        
        return aic, bic

    def _cv_worker(self, args) -> Tuple[hmm.GaussianHMM, float, float, int]:
        """Worker function for parallel cross-validation"""
        i, all_data, n_states = args
        try:
            # Create validation set (single subject)
            val_data = self.preprocess_data(all_data[i:i+1])
            
            # Create training set (all other subjects)
            train_data = self.preprocess_data(np.delete(all_data, i, axis=0))
            
            # Fit model
            model = self.fit_single_model(train_data, n_states)
            
            # Calculate information criteria using validation data
            aic, bic = self.calculate_information_criteria(model, val_data)
            val_score = model.score(val_data)
            
            return (model, val_score, aic, bic, i)
            
        except Exception as e:
            logging.warning(f"CV failed for subject {i}: {str(e)}")
            return None

    def calculate_transition_stability(self, 
                                    models: List[hmm.GaussianHMM], 
                                    data: np.ndarray) -> Tuple[float, float]:
        """Calculate stability of state transitions across cross-validation folds"""
        state_sequences = [model.predict(data) for model in models]
        n_models = len(models)
        mi_scores = []
        
        for i in range(n_models):
            for j in range(i + 1, n_models):
                mi = mutual_info_score(state_sequences[i], state_sequences[j])
                mi_scores.append(mi)
        
        return np.mean(mi_scores), np.std(mi_scores)

    def analyze_state_durations(self, 
                              model: hmm.GaussianHMM, 
                              data: np.ndarray, 
                              n_states: int) -> Dict:
        """Analyze the distribution of state durations"""
        state_sequence = model.predict(data)
        durations = {state: [] for state in range(n_states)}
        current_state = state_sequence[0]
        current_duration = 1
        
        for state in state_sequence[1:]:
            if state == current_state:
                current_duration += 1
            else:
                durations[current_state].append(current_duration * self.tr)
                current_state = state
                current_duration = 1
        
        durations[current_state].append(current_duration * self.tr)
        
        duration_stats = {}
        for state in range(n_states):
            if durations[state]:
                duration_stats[state] = {
                    'mean': np.mean(durations[state]),
                    'std': np.std(durations[state]),
                    'median': np.median(durations[state]),
                    'min': np.min(durations[state]),
                    'max': np.max(durations[state]),
                    'n_occurrences': len(durations[state]),
                    'total_time': sum(durations[state]),
                    'proportion': sum(durations[state]) / (len(state_sequence) * self.tr)
                }
            else:
                duration_stats[state] = {
                    'mean': 0, 'std': 0, 'median': 0,
                    'min': 0, 'max': 0, 'n_occurrences': 0,
                    'total_time': 0, 'proportion': 0
                }
        
        return {'raw_durations': durations, 'stats': duration_stats}

    def save_checkpoint(self, cv_results: Dict, current_n_states: int):
        """Save checkpoint during analysis"""
        checkpoint_file = os.path.join(self.checkpoint_dir, 
                                     f"checkpoint_{current_n_states}.pkl")
        with open(checkpoint_file, "wb") as f:
            pickle.dump(cv_results, f)

    def cross_validate_model(self, 
                           group1_data: np.ndarray, 
                           group2_data: np.ndarray, 
                           n_states: int) -> Dict:
        """Perform comprehensive cross-validation with multiple metrics"""
        all_data = np.concatenate([group1_data, group2_data], axis=0)
        n_subj = len(all_data)
        
        cv_args = [(i, all_data, n_states) for i in range(n_subj)]
        
        all_models = []
        cv_scores = []
        aic_scores = []
        bic_scores = []
        cv_indices = []
        
        with Pool(processes=self.n_jobs) as pool:
            results = list(tqdm(
                pool.imap(self._cv_worker, cv_args),
                total=len(cv_args),
                desc=f"CV for {n_states} states"
            ))
        
        valid_results = [r for r in results if r is not None]
        if not valid_results:
            raise ValueError("All cross-validation iterations failed")
        
        for model, val_score, aic, bic, idx in valid_results:
            all_models.append(model)
            cv_scores.append(val_score)
            aic_scores.append(aic)
            bic_scores.append(bic)
            cv_indices.append(idx)
        
        combined_data = self.preprocess_data(all_data)
        stability_mean, stability_std = self.calculate_transition_stability(
            all_models, combined_data)
        
        best_model_idx = np.argmin(bic_scores)
        duration_analysis = self.analyze_state_durations(
            all_models[best_model_idx], combined_data, n_states)
        
        return {
            'mean_score': np.mean(cv_scores),
            'std_score': np.std(cv_scores),
            'all_scores': cv_scores,
            'cv_indices': cv_indices,
            'n_states': n_states,
            'n_successful': len(valid_results),
            'n_total': len(cv_args),
            'information_criteria': {
                'aic': {
                    'mean': np.mean(aic_scores),
                    'std': np.std(aic_scores),
                    'min': np.min(aic_scores),
                    'max': np.max(aic_scores),
                    'best_model_idx': int(np.argmin(aic_scores))
                },
                'bic': {
                    'mean': np.mean(bic_scores),
                    'std': np.std(bic_scores),
                    'min': np.min(bic_scores),
                    'max': np.max(bic_scores),
                    'best_model_idx': int(best_model_idx)
                }
            },
            'stability': {
                'mean': stability_mean,
                'std': stability_std
            },
            'durations': duration_analysis
        }

    def find_optimal_states(self, 
                          group1_data: np.ndarray, 
                          group2_data: np.ndarray) -> Dict:
        """Find optimal number of states using comprehensive metrics"""
        cv_results = {}
        no_improvement_count = 0
        best_score = float('-inf')
        patience = 3
        
        for n_states in range(self.n_components_range[0], 
                            self.n_components_range[1] + 1):
            logging.info(f"Cross-validating model with {n_states} states")
            
            cv_result = self.cross_validate_model(
                group1_data, group2_data, n_states)
            
            cv_results[n_states] = cv_result
            
            self.plot_transition_matrix(
                self.initialize_informed_transition_matrix(n_states),
                n_states,
                f"initial_transition_matrix_{n_states}_states.png"
            )
            
            logging.info(f"CV Score for {n_states} states: "
                        f"{cv_result['mean_score']:.2f} ± "
                        f"{cv_result['std_score']:.2f}")
            logging.info(f"AIC: {cv_result['information_criteria']['aic']['mean']:.2f}")
            logging.info(f"BIC: {cv_result['information_criteria']['bic']['mean']:.2f}")
            logging.info(f"Stability: {cv_result['stability']['mean']:.3f}")
            
            if cv_result['mean_score'] > best_score:
                best_score = cv_result['mean_score']
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            self.save_checkpoint(cv_results, n_states)
            
            if no_improvement_count >= patience:
                logging.info(f"No improvement for {patience} iterations. Stopping.")
                break
        
        best_n_states = max(cv_results.items(), 
                          key=lambda x: x[1]['mean_score'])[0]
        
        return {
            'cv_results': cv_results,
            'best_n_states': best_n_states
        }

    def plot_transition_matrix(self, 
                             matrix: np.ndarray, 
                             n_states: int, 
                             filename: str):
        """Plot transition matrix heatmap"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix, annot=True, cmap='coolwarm', vmin=0, vmax=1)
        plt.title(f'Transition Matrix for {n_states} States')
        plt.xlabel('To State')
        plt.ylabel('From State')
        plt.savefig(filename)
        plt.close()

    def plot_duration_distributions(self, 
                                  duration_results: Dict, 
                                  n_states: int, 
                                  output_path: str):
        """Plot state duration distributions"""
        fig, axs = plt.subplots(n_states, 1, figsize=(12, 4*n_states))
        if n_states == 1:
            axs = [axs]
        
        for state in range(n_states):
            durations = duration_results['raw_durations'][state]
            stats = duration_results['stats'][state]
            
            if durations:
                sns.histplot(durations, ax=axs[state], bins=30)
                axs[state].axvline(stats['mean'], color='r', linestyle='--',
                                 label=f"Mean: {stats['mean']:.2f}s")
                axs[state].axvline(stats['median'], color='g', linestyle=':',
                                 label=f"Median: {stats['median']:.2f}s")
                axs[state].set_title(f"State {state} Durations\n"
                                   f"(n={stats['n_occurrences']}, "
                                   f"proportion={stats['proportion']:.3f})")
                axs[state].set_xlabel("Duration (seconds)")
                axs[state].set_ylabel("Count")
                axs[state].legend()
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def plot_metrics_comparison(self, results: Dict, output_path: str):
        """Plot comparison of different metrics"""
        n_states = sorted(results.keys())
        metrics = {
            'Cross-Validation Score': [results[n]['mean_score'] for n in n_states],
            'AIC': [results[n]['information_criteria']['aic']['mean'] for n in n_states],
            'BIC': [results[n]['information_criteria']['bic']['mean'] for n in n_states],
            'Stability': [results[n]['stability']['mean'] for n in n_states]
        }
        
        fig, axes = plt.subplots(4, 1, figsize=(12, 20))
        
        for ax, (metric_name, values) in zip(axes, metrics.items()):
            ax.plot(n_states, values, 'o-', linewidth=2)
            ax.set_xlabel('Number of States')
            ax.set_ylabel(metric_name)
            ax.grid(True)
            
            # Add trend line
            z = np.polyfit(n_states, values, 3)
            p = np.poly1d(z)
            ax.plot(n_states, p(n_states), '--', color='gray', alpha=0.5)
            
            # Add value labels
            for x, y in zip(n_states, values):
                ax.annotate(f'{y:.0f}', 
                          (x, y), 
                          textcoords="offset points", 
                          xytext=(0,10), 
                          ha='center')
        
        plt.suptitle('Model Selection Metrics Comparison', y=1.02, fontsize=14)
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()

    def plot_all_metrics(self, results: Dict, output_dir: str):
        """Generate all visualization plots"""
        plots_dir = os.path.join(output_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Plot metrics comparison
        self.plot_metrics_comparison(
            results['cv_results'],
            os.path.join(plots_dir, 'metrics_comparison.png')
        )
        
        # Plot duration distributions for best model
        best_n_states = results['best_n_states']
        best_results = results['cv_results'][best_n_states]
        self.plot_duration_distributions(
            best_results['durations'],
            best_n_states,
            os.path.join(plots_dir, f'duration_distributions_{best_n_states}_states.png')
        )
        
        # Plot transition matrices
        for n_states in results['cv_results'].keys():
            self.plot_transition_matrix(
                self.initialize_informed_transition_matrix(n_states),
                n_states,
                os.path.join(plots_dir, f'transition_matrix_{n_states}_states.png')
            )

def generate_detailed_report(results: Dict, output_dir: str) -> str:
    """Generate detailed analysis report with all metrics"""
    report_path = os.path.join(output_dir, 'detailed_analysis_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("="*50 + "\n")
        f.write("Detailed HMM Analysis Report\n")
        f.write("="*50 + "\n\n")
        
        f.write(f"Optimal number of states: {results['best_n_states']}\n\n")
        
        f.write("Model Selection Metrics Summary:\n")
        f.write("-"*30 + "\n")
        for n_states, result in results['cv_results'].items():
            f.write(f"\nResults for {n_states} states:\n")
            f.write(f"{'-'*20}\n")
            
            # Cross-validation score
            f.write(f"CV Score: {result['mean_score']:.2f} ± "
                   f"{result['std_score']:.2f}\n")
            
            # Information criteria
            f.write(f"AIC: {result['information_criteria']['aic']['mean']:.2f} ± "
                   f"{result['information_criteria']['aic']['std']:.2f}\n")
            f.write(f"BIC: {result['information_criteria']['bic']['mean']:.2f} ± "
                   f"{result['information_criteria']['bic']['std']:.2f}\n")
            
            # Stability
            f.write(f"Stability: {result['stability']['mean']:.3f} ± "
                   f"{result['stability']['std']:.3f}\n\n")
            
            # State durations
            f.write("State Duration Statistics:\n")
            for state, stats in result['durations']['stats'].items():
                f.write(f"State {state}:\n")
                f.write(f"  Mean duration: {stats['mean']:.2f}s\n")
                f.write(f"  Median duration: {stats['median']:.2f}s\n")
                f.write(f"  Occurrences: {stats['n_occurrences']}\n")
                f.write(f"  Total time: {stats['total_time']:.2f}s\n")
                f.write(f"  Proportion: {stats['proportion']:.3f}\n")
                f.write("\n")
    
    return report_path

def main(group1_data: np.ndarray, 
         group2_data: np.ndarray,
         group1_name: str,
         group2_name: str,
         output_dir: str,
         n_components_range: Tuple[int, int],
         random_state: int,
         n_jobs: int,
         max_tries: int,
         tr: float = 1.5) -> Dict:
    """
    Main execution function for HMM analysis
    """
    start_time = time.time()
    
    try:
        logging.info(f"Starting HMM analysis for {group1_name} and {group2_name}")
        logging.info(f"Group 1 shape: {group1_data.shape}")
        logging.info(f"Group 2 shape: {group2_data.shape}")
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize analyzer
        analyzer = GroupHMM(
            n_components_range=n_components_range,
            random_state=random_state,
            n_jobs=n_jobs,
            max_tries=max_tries,
            tr=tr
        )

        # Find optimal states
        results = analyzer.find_optimal_states(group1_data, group2_data)
        optimal_n_states = results['best_n_states']
        
        # Generate all plots
        analyzer.plot_all_metrics(results, output_dir)
        
        # Generate detailed report
        report_path = generate_detailed_report(results, output_dir)
        
        # Save full results
        results_file = os.path.join(output_dir, "full_hmm_results.pkl")
        with open(results_file, "wb") as f:
            pickle.dump(results, f)
        
        # Add execution metadata
        end_time = time.time()
        execution_time = end_time - start_time
        
        metadata = {
            'execution_time': execution_time,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'parameters': {
                'n_components_range': n_components_range,
                'random_state': random_state,
                'n_jobs': n_jobs,
                'max_tries': max_tries,
                'tr': tr
            },
            'data_info': {
                'group1_name': group1_name,
                'group2_name': group2_name,
                'group1_shape': group1_data.shape,
                'group2_shape': group2_data.shape
            }
        }
        
        # Save metadata
        metadata_file = os.path.join(output_dir, "analysis_metadata.json")
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=4)
        
        logging.info(f"Analysis completed in {execution_time:.2f} seconds")
        logging.info(f"Optimal number of states: {optimal_n_states}")
        logging.info(f"Results saved to {output_dir}")
        
        return results
        
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    # Set up logging
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('group_hmm_analysis.log')
    file_handler.setFormatter(log_formatter)
    logging.getLogger().addHandler(file_handler)
    
    # Parse arguments
    parser = ArgumentParser(description='HMM Analysis for neuroimaging data')
    parser.add_argument("--res", type=str, default="native", 
                       help="Resolution for analysis")
    parser.add_argument("--n_components_min", type=int, default=2, 
                       help="Minimum number of states")
    parser.add_argument("--n_components_max", type=int, default=20, 
                       help="Maximum number of states")
    parser.add_argument("--random_state", type=int, default=42, 
                       help="Random state for reproducibility")
    parser.add_argument("--n_jobs", type=int, default=20, 
                       help="Number of parallel jobs")
    parser.add_argument("--max_tries", type=int, default=5, 
                       help="Maximum attempts for model fitting")
    parser.add_argument("--tr", type=float, default=1.5,
                       help="TR in seconds")
    args = parser.parse_args()

    # Load environment variables
    load_dotenv()
    base_dir = os.getenv("BASE_DIR")
    scratch_dir = os.getenv("SCRATCH_DIR")

    # Prepare output directory
    output_dir = os.path.join(scratch_dir, "output", f"group_hmm_{args.res}")
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Load data
        group1_data = np.load(os.path.join(scratch_dir, "output", 
                                         f"atlas_masked_{args.res}", 
                                         "affair_group_data_roi.npy"))
        group2_data = np.load(os.path.join(scratch_dir, "output", 
                                         f"atlas_masked_{args.res}", 
                                         "paranoia_group_data_roi.npy"))
        
        # Run analysis
        results = main(
            group1_data=group1_data,
            group2_data=group2_data,
            group1_name="affair",
            group2_name="paranoia",
            output_dir=output_dir,
            n_components_range=(args.n_components_min, args.n_components_max),
            random_state=args.random_state,
            n_jobs=args.n_jobs,
            max_tries=args.max_tries,
            tr=args.tr
        )
        
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        raise