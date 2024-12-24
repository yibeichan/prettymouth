import os
from dotenv import load_dotenv
import numpy as np
from argparse import ArgumentParser
import pickle
from hmmlearn import hmm
from scipy.stats import zscore
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import gc
from typing import List, Tuple, Dict
import time
import logging
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')

class GroupHMMAnalysis:
    def __init__(self, n_components_range: Tuple[int, int], 
                 random_state: int, 
                 n_jobs: int, 
                 max_tries: int,
                 tr: float = 1.5):
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

    def cross_validate_model(self, 
                           group1_data: np.ndarray, 
                           group2_data: np.ndarray, 
                           n_states: int) -> Dict:
        """Perform balanced leave-one-out cross-validation"""
        n_subj1 = len(group1_data)
        n_subj2 = len(group2_data)
        cv_scores = []
        
        for i in tqdm(range(n_subj1), desc=f"CV for {n_states} states - Group 1"):
            for j in range(n_subj2):
                try:
                    # Create validation set
                    val_subj1 = group1_data[i:i+1]
                    val_subj2 = group2_data[j:j+1]
                    val_data = np.vstack([
                        self.preprocess_data(val_subj1),
                        self.preprocess_data(val_subj2)
                    ])
                    
                    # Create training set
                    train_subj1 = np.delete(group1_data, i, axis=0)
                    train_subj2 = np.delete(group2_data, j, axis=0)
                    train_data = np.vstack([
                        self.preprocess_data(train_subj1),
                        self.preprocess_data(train_subj2)
                    ])
                    
                    # Fit and evaluate model
                    model = self.fit_single_model(train_data, n_states)
                    val_score = model.score(val_data)
                    cv_scores.append(val_score)
                    
                except Exception as e:
                    logging.warning(f"CV failed for subjects {i},{j}: {str(e)}")
                    continue
        
        return {
            'mean_score': np.mean(cv_scores),
            'std_score': np.std(cv_scores),
            'all_scores': cv_scores,
            'n_states': n_states
        }

    def find_optimal_states(self, 
                          group1_data: np.ndarray, 
                          group2_data: np.ndarray) -> Dict:
        """Find optimal number of states using cross-validation"""
        cv_results = {}
        
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

def plot_cv_results(cv_results: Dict, output_path: str):
    """Plot cross-validation results"""
    n_states = sorted(cv_results.keys())
    mean_scores = [cv_results[n]['mean_score'] for n in n_states]
    std_scores = [cv_results[n]['std_score'] for n in n_states]
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(n_states, mean_scores, yerr=std_scores, 
                marker='o', linestyle='-')
    plt.xlabel('Number of States')
    plt.ylabel('Cross-Validation Score')
    plt.title('Cross-Validation Results for Different Numbers of States')
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

def main(group1_data: np.ndarray, 
         group2_data: np.ndarray,
         group1_name: str,
         group2_name: str,
         output_dir: str,
         n_components_range: Tuple[int, int],
         random_state: int,
         n_jobs: int,
         max_tries: int,
         tr: float = 1.5):
    """Main execution function"""
    
    start_time = time.time()
    
    try:
        logging.info(f"Starting HMM analysis for {group1_name} and {group2_name}")
        logging.info(f"Group 1 shape: {group1_data.shape}")
        logging.info(f"Group 2 shape: {group2_data.shape}")
        
        # Create output directories
        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Initialize analyzer
        analyzer = GroupHMMAnalysis(
            n_components_range=n_components_range,
            random_state=random_state,
            n_jobs=n_jobs,
            max_tries=max_tries,
            tr=tr
        )

        # Find optimal states using cross-validation
        cv_results = analyzer.find_optimal_states(group1_data, group2_data)
        optimal_n_states = cv_results['best_n_states']
        
        # Plot cross-validation results
        plot_cv_results(
            cv_results['cv_results'], 
            os.path.join(plots_dir, "cv_results.png")
        )
        
        # Prepare output
        output = {
            'cv_results': cv_results,
            'optimal_n_states': optimal_n_states,
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
        
        # Save results
        results_file = os.path.join(output_dir, "hmm_cv_results.pkl")
        with open(results_file, "wb") as f:
            pickle.dump(output, f)
        
        # Generate summary report
        end_time = time.time()
        total_time = end_time - start_time
        
        summary_file = os.path.join(output_dir, "analysis_summary.txt")
        with open(summary_file, "w") as f:
            f.write("="*50 + "\n")
            f.write("HMM Analysis Summary Report\n")
            f.write("="*50 + "\n\n")
            
            f.write(f"Groups: {group1_name} and {group2_name}\n")
            f.write(f"Optimal number of states: {optimal_n_states}\n\n")
            
            f.write("Cross-Validation Results:\n")
            for n_states, result in cv_results['cv_results'].items():
                f.write(f"States {n_states}: {result['mean_score']:.2f} ± "
                       f"{result['std_score']:.2f}\n")
            
            f.write(f"\nTotal processing time: {total_time:.2f} seconds\n")
        
        logging.info("Analysis completed successfully")
        logging.info(f"Optimal number of states: {optimal_n_states}")
        logging.info(f"Results saved to {output_dir}")
        
        return output
        
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    # Set up logging
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('hmm_analysis.log')
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

    # Load data
    try:
        # Load both groups
        group1_data = np.load(os.path.join(scratch_dir, "output", 
                                          f"atlas_masked_{args.res}", 
                                          "affair_group_data_roi.npy"))
        group2_data = np.load(os.path.join(scratch_dir, "output", 
                                          f"atlas_masked_{args.res}", 
                                          "paranoia_group_data_roi.npy"))
        
        # Run main analysis
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