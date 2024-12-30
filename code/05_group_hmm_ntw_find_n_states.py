import os
import numpy as np
from pathlib import Path
import logging
import pickle
from hmmlearn import hmm
from scipy.stats import zscore
from sklearn.metrics import mutual_info_score
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import traceback
from argparse import ArgumentParser

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GroupHMM:
    def __init__(self, n_components_range: Tuple[int, int], random_state: int, 
                 n_jobs: int, max_tries: int, tr: float = 1.5, output_dir: Path = None):
        self.n_components_range = n_components_range
        self.random_state = random_state
        self.n_jobs = min(n_jobs, cpu_count())
        self.max_tries = max_tries
        self.tr = tr
        self.output_dir = output_dir
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Initialize storage
        self.models = {}
        self.scores = {}
        self.information_criteria = {}
        
    def calculate_free_parameters(self, n_states: int, n_features: int) -> int:
        """Calculate number of free parameters in the model"""
        # Mean parameters for each state and feature
        mean_params = n_states * n_features
        
        # Diagonal covariance parameters
        cov_params = n_states * n_features
        
        # Transition matrix parameters (n_states x n_states - n_states)
        # Subtract n_states because each row must sum to 1
        trans_params = n_states * (n_states - 1)
        
        # Initial state probabilities (n_states - 1 because they sum to 1)
        init_params = n_states - 1
        
        return mean_params + cov_params + trans_params + init_params
        
    def calculate_information_criteria(self, model: hmm.GaussianHMM, 
                                    data: np.ndarray, n_states: int) -> Dict:
        """Calculate AIC and BIC using full model likelihood"""
        n_samples = data.shape[0]
        n_features = data.shape[1]
        n_params = self.calculate_free_parameters(n_states, n_features)
        
        # Calculate total log likelihood
        log_likelihood = model.score(data) * n_samples
        
        # Calculate information criteria
        aic = -2 * log_likelihood + 2 * n_params
        bic = -2 * log_likelihood + np.log(n_samples) * n_params
        
        return {
            'aic': aic,
            'bic': bic,
            'log_likelihood': log_likelihood,
            'n_params': n_params
        }

    def fit_single_model(self, data: np.ndarray, n_states: int) -> hmm.GaussianHMM:
        """Fit single HMM with informed initialization"""
        model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type='diag',
            n_iter=500,
            random_state=self.random_state,
            init_params='mc',  # Initialize means and covars from data
            params='stmc'
        )
        
        # Fit model
        for _ in range(self.max_tries):
            try:
                model.fit(data)
                return model
            except Exception as e:
                logging.warning(f"Fitting attempt failed: {str(e)}")
                continue
        
        raise ValueError(f"Failed to fit model after {self.max_tries} attempts")

    def evaluate_model(self, data: np.ndarray, n_states: int) -> Dict:
        """Evaluate model with cross-validation and information criteria"""
        # Preprocess full dataset
        processed_data = self.preprocess_data(data)
        
        # Fit model on full dataset
        model = self.fit_single_model(processed_data, n_states)
        
        # Calculate information criteria
        ic_metrics = self.calculate_information_criteria(model, processed_data, n_states)
        
        # Perform cross-validation
        cv_scores = self.cross_validate(data, n_states)
        
        return {
            'model': model,
            'ic_metrics': ic_metrics,
            'cv_scores': cv_scores
        }

    def preprocess_data(self, data: np.ndarray) -> np.ndarray:
        """Preprocess data for HMM analysis"""
        if data.ndim == 3:  # Multiple subjects
            return np.vstack([zscore(subj, axis=1, ddof=1).T for subj in data])
        return zscore(data, axis=1, ddof=1).T

    def cross_validate(self, data: np.ndarray, n_states: int) -> Dict:
        """Perform leave-one-out cross-validation"""
        n_subjects = len(data)
        scores = []
        
        for i in range(n_subjects):
            # Split data
            test_data = self.preprocess_data(data[i:i+1])
            train_data = self.preprocess_data(np.delete(data, i, axis=0))
            
            # Fit and evaluate
            try:
                model = self.fit_single_model(train_data, n_states)
                score = model.score(test_data)
                scores.append(score)
            except Exception as e:
                logging.warning(f"CV fold {i} failed: {str(e)}")
                continue
        
        return {
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'n_successful': len(scores)
        }

    def find_optimal_states(self, data: np.ndarray) -> Dict:
        """Find optimal number of states using multiple criteria"""
        results = {}
        best_bic = float('inf')
        best_n_states = None
        
        for n_states in range(self.n_components_range[0], 
                            self.n_components_range[1] + 1):
            try:
                # Evaluate model
                evaluation = self.evaluate_model(data, n_states)
                results[n_states] = evaluation
                
                # Track best BIC
                if evaluation['ic_metrics']['bic'] < best_bic:
                    best_bic = evaluation['ic_metrics']['bic']
                    best_n_states = n_states
                
                # Save checkpoint
                self.save_checkpoint(results, n_states)
                
                # Log progress
                logging.info(f"\nResults for {n_states} states:")
                logging.info(f"BIC: {evaluation['ic_metrics']['bic']:.2f}")
                logging.info(f"AIC: {evaluation['ic_metrics']['aic']:.2f}")
                logging.info(f"CV Score: {evaluation['cv_scores']['mean_score']:.2f} "
                           f"± {evaluation['cv_scores']['std_score']:.2f}")
                
            except Exception as e:
                logging.error(f"Error processing {n_states} states: {str(e)}")
                continue
        
        return {
            'results': results,
            'best_n_states': best_n_states
        }

    def save_checkpoint(self, results: Dict, current_n_states: int):
        """Save checkpoint during analysis"""
        checkpoint_file = self.checkpoint_dir / f"checkpoint_{current_n_states}.pkl"
        with open(checkpoint_file, "wb") as f:
            pickle.dump(results, f)

    def generate_report(self, results: Dict, output_dir: Path):
        """Generate comprehensive analysis report"""
        report_file = output_dir / 'analysis_report.txt'
        
        with open(report_file, 'w') as f:
            f.write("HMM Analysis Report\n")
            f.write("="*50 + "\n\n")
            
            f.write(f"Optimal number of states: {results['best_n_states']}\n\n")
            
            for n_states, result in results['results'].items():
                f.write(f"\nResults for {n_states} states:\n")
                f.write("-"*30 + "\n")
                f.write(f"BIC: {result['ic_metrics']['bic']:.2f}\n")
                f.write(f"AIC: {result['ic_metrics']['aic']:.2f}\n")
                f.write(f"Log Likelihood: {result['ic_metrics']['log_likelihood']:.2f}\n")
                f.write(f"Number of parameters: {result['ic_metrics']['n_params']}\n")
                f.write(f"CV Score: {result['cv_scores']['mean_score']:.2f} "
                       f"± {result['cv_scores']['std_score']:.2f}\n")

def load_and_preprocess_data(files: List[str], trim: bool = True) -> np.ndarray:
    """Load and preprocess data from multiple files"""
    data = []
    for file in sorted(files):
        raw_data = np.load(file)
        avg_data = np.mean(raw_data, axis=1)
        data.append(avg_data)
    data = np.stack(data, axis=1)
    
    if trim:
        data = data[:, :, 17:468]
    return data

def main():
    parser = ArgumentParser(description='HMM Analysis for neuroimaging data')
    parser.add_argument("--res", type=str, default="native")
    parser.add_argument("--n_components_min", type=int, default=2)
    parser.add_argument("--n_components_max", type=int, default=20)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--n_jobs", type=int, default=38)
    parser.add_argument("--max_tries", type=int, default=5)
    parser.add_argument("--tr", type=float, default=1.5)
    parser.add_argument("--trim", type=bool, default=True)
    args = parser.parse_args()

    load_dotenv()
    scratch_dir = Path(os.getenv("SCRATCH_DIR"))
    if args.trim:   
        output_dir = scratch_dir / "output" / f"group_hmm_ntw_{args.res}_trimmed"
    else:
        output_dir = scratch_dir / "output" / f"group_hmm_ntw_{args.res}"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Setup logging to file
        log_file = output_dir / 'analysis.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(file_handler)

        # Load data
        data_dir = scratch_dir / "output" / f"atlas_masked_{args.res}" / "networks"
        group1_files = sorted(list(data_dir.glob("affair_*.npy")))
        group2_files = sorted(list(data_dir.glob("paranoia_*.npy")))

        group1_data = load_and_preprocess_data(group1_files, args.trim)
        group2_data = load_and_preprocess_data(group2_files, args.trim)

        logging.info(f"Group 1 data shape: {group1_data.shape}")
        logging.info(f"Group 2 data shape: {group2_data.shape}")

        # Initialize and run analysis
        analyzer = GroupHMM(
            n_components_range=(args.n_components_min, args.n_components_max),
            random_state=args.random_state,
            n_jobs=args.n_jobs,
            max_tries=args.max_tries,
            tr=args.tr,
            output_dir=output_dir
        )

        all_data = np.concatenate([group1_data, group2_data], axis=0)

        results = analyzer.find_optimal_states(all_data)
        
        # Generate outputs
        analyzer.plot_metrics(results, output_dir)
        analyzer.generate_report(results, output_dir)
        
        # Save final results
        with open(output_dir / "final_results.pkl", "wb") as f:
            pickle.dump(results, f)
            
        logging.info(f"Analysis completed. Results saved to {output_dir}")
        logging.info(f"Optimal number of states: {results['best_n_states']}")

    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        logging.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()