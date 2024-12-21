import os
from dotenv import load_dotenv
import numpy as np
from argparse import ArgumentParser
import pickle
from hmmlearn import hmm
from scipy.stats import zscore
from tqdm import tqdm
from multiprocessing import Pool, shared_memory, cpu_count
import gc
from typing import List, Union, Tuple, Dict
import time
import psutil
import logging

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')

class GroupHMMAnalysis:
    def __init__(self, n_components_range: Tuple[int, int], 
                 random_state: int, 
                 n_jobs: int, 
                 max_tries: int,
                 chunk_size: int = None):
        if not isinstance(n_components_range, tuple) or len(n_components_range) != 2:
            raise ValueError("n_components_range must be a tuple of (min, max)")
        if n_components_range[0] >= n_components_range[1]:
            raise ValueError("n_components_range[0] must be less than n_components_range[1]")

        self.n_components_range = n_components_range
        self.random_state = random_state
        self.n_jobs = min(n_jobs, cpu_count())
        self.max_tries = max_tries
        self.chunk_size = chunk_size
        self.best_models = {}
        self.convergence_stats = {
            'total_attempts': 0,
            'convergence_failures': 0,
            'other_failures': 0
        }
        self.shared_data = None
        self.performance_stats = {
            'memory_usage': [],
            'processing_times': []
        }
        self._preprocessed_cache = {}

    def log_memory_usage(self):
        """Log current memory usage"""
        process = psutil.Process(os.getpid())
        mem_usage = process.memory_info().rss / 1024 / 1024  # Convert to MB
        self.performance_stats['memory_usage'].append(mem_usage)
        return mem_usage

    def optimize_chunk_size(self, n_total_tasks: int) -> int:
        """Determine optimal chunk size based on data and CPU count"""
        if self.chunk_size is not None:
            return self.chunk_size
            
        tasks_per_worker = max(1, n_total_tasks // (self.n_jobs * 4))
        return min(tasks_per_worker, 10)

    def preprocess_data(self, data_list: List[np.ndarray]) -> List[np.ndarray]:
        """Z-score each subject's data"""
        cache_key = hash(tuple(arr.tobytes() for arr in data_list))
        if cache_key in self._preprocessed_cache:
            return self._preprocessed_cache[cache_key]
        result = [zscore(subject, axis=1, ddof=1) for subject in data_list]
        self._preprocessed_cache[cache_key] = result
        return result

    def prepare_shared_data(self, data_list: List[np.ndarray]) -> List[Tuple]:
        """Convert data to shared memory format"""
        processed_data = self.preprocess_data(data_list)
        shared_data = []
        
        for subject_data in processed_data:
            shm = shared_memory.SharedMemory(create=True, size=subject_data.nbytes)
            shared_array = np.ndarray(subject_data.shape, 
                                    dtype=subject_data.dtype, 
                                    buffer=shm.buf)
            shared_array[:] = subject_data[:]
            shared_data.append((shm.name, subject_data.shape, subject_data.dtype))
        
        self.shared_data = shared_data
        return shared_data

    def cleanup_shared_memory(self):
        """Clean up shared memory"""
        if self.shared_data:
            for shm_name, _, _ in self.shared_data:
                try:
                    shm = shared_memory.SharedMemory(name=shm_name)
                    shm.close()
                    shm.unlink()
                except:
                    pass
            self.shared_data = None

    def get_data_from_shared(self, indices: List[int]) -> List[np.ndarray]:
        """Retrieve data from shared memory"""
        if not self.shared_data:
            raise ValueError("Shared data not initialized")
            
        result = []
        for idx in indices:
            shm_name, shape, dtype = self.shared_data[idx]
            shm = shared_memory.SharedMemory(name=shm_name)
            array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
            result.append(array.copy())
            shm.close()
        
        return result
    
    def evaluate_model(self, train_data: List[np.ndarray], 
                      test_data: List[np.ndarray], 
                      n_states: int) -> Dict:
        """Evaluate HMM model with enhanced error handling and performance tracking"""
        start_time = time.time()
        self.convergence_stats['total_attempts'] += 1
        
        for attempt in range(self.max_tries):
            try:
                # Track memory before model fitting
                initial_mem = self.log_memory_usage()
                
                # Initialize transition matrix
                transmat = np.eye(n_states) * 0.9
                off_diag = (1 - 0.9) / (n_states - 1)
                transmat[transmat == 0] = off_diag
                
                model = hmm.GaussianHMM(
                    n_components=n_states,
                    covariance_type='diag',
                    n_iter=500,
                    random_state=self.random_state + attempt,
                    params='stmc',
                    init_params='',
                    tol=1e-2
                )
                
                model.transmat_ = transmat
                
                # Prepare data for multiple sequence fitting
                train_lengths = [subject.shape[1] for subject in train_data]
                train_data_stacked = np.vstack([subj.T for subj in train_data])
                
                # Fit model
                model.fit(train_data_stacked, lengths=train_lengths)
                
                # Calculate log-likelihood
                train_ll = np.mean([model.score(subj.T) for subj in train_data])
                test_ll = np.mean([model.score(subj.T) for subj in test_data])
                
                # Calculate model parameters
                n_params = (n_states * (n_states - 1) +  # Transition parameters
                          n_states * train_data[0].shape[0] * 2)  # Emission parameters
                
                # Calculate information criteria
                total_test_timepoints = sum(subj.shape[1] for subj in test_data)
                aic = -2 * test_ll + 2 * n_params
                bic = -2 * test_ll + np.log(total_test_timepoints) * n_params
                
                # Track performance metrics
                end_time = time.time()
                final_mem = self.log_memory_usage()
                
                self.performance_stats['processing_times'].append(end_time - start_time)
                
                return {
                    'model': model,
                    'train_ll': train_ll,
                    'test_ll': test_ll,
                    'aic': aic,
                    'bic': bic,
                    'n_params': n_params,
                    'converged': True,
                    'n_attempts': attempt + 1,
                    'processing_time': end_time - start_time,
                    'memory_delta': final_mem - initial_mem
                }
                
            except (ValueError, np.linalg.LinAlgError) as e:
                if "converge" in str(e).lower():
                    self.convergence_stats['convergence_failures'] += 1
                    if attempt < self.max_tries - 1:
                        continue
                    else:
                        # Return last attempt results even if not converged
                        end_time = time.time()
                        return {
                            'model': model,
                            'train_ll': train_ll if 'train_ll' in locals() else None,
                            'test_ll': test_ll if 'test_ll' in locals() else None,
                            'aic': aic if 'aic' in locals() else None,
                            'bic': bic if 'bic' in locals() else None,
                            'n_params': n_params if 'n_params' in locals() else None,
                            'converged': False,
                            'n_attempts': attempt + 1,
                            'processing_time': end_time - start_time,
                            'error': str(e)
                        }
                else:
                    self.convergence_stats['other_failures'] += 1
                    if attempt == self.max_tries - 1:
                        logging.error(f"Failed to fit model with {n_states} states: {str(e)}")
                        raise
                    continue
            finally:
                # Force garbage collection after each attempt
                gc.collect()

    def process_chunk(self, chunk_params: List[Tuple]) -> List[Tuple]:
        """Process a chunk of parameters with memory management"""
        chunk_results = []
        for n_states, train_idx, test_idx in chunk_params:
            try:
                # Get data from shared memory
                train_data = self.get_data_from_shared(train_idx)
                test_data = self.get_data_from_shared(test_idx)
                
                # Evaluate model
                result = self.evaluate_model(train_data, test_data, n_states)
                chunk_results.append((n_states, result))
                
            except Exception as e:
                logging.error(f"Error processing chunk: {str(e)}")
                chunk_results.append((n_states, {'error': str(e)}))
            
            finally:
                # Clean up
                gc.collect()
                
        return chunk_results
    
    def find_optimal_states(self, data_list: List[np.ndarray]) -> Dict:
        """Optimized parallel implementation for finding optimal states"""
        n_subjects = len(data_list)
        results = {n_states: [] for n_states in range(self.n_components_range[0], 
                                                    self.n_components_range[1] + 1)}
        
        # Convert data to shared memory format
        try:
            logging.info("Preparing shared memory data...")
            self.prepare_shared_data(data_list)
            
            # Pre-compute CV splits
            cv_splits = [(list(set(range(n_subjects)) - {test_idx}), [test_idx]) 
                        for test_idx in range(n_subjects)]
            
            # Create parameter tuples
            param_tuples = []
            for n_states in results.keys():
                for train_idx, test_idx in cv_splits:
                    param_tuples.append((n_states, train_idx, test_idx))
            
            # Determine chunk size
            total_tasks = len(param_tuples)
            chunk_size = self.optimize_chunk_size(total_tasks)
            param_chunks = [param_tuples[i:i + chunk_size] 
                          for i in range(0, total_tasks, chunk_size)]
            
            logging.info(f"Starting parallel processing with {len(param_chunks)} chunks...")
            
            # Process chunks in parallel
            with Pool(processes=self.n_jobs) as pool:
                chunk_results = list(tqdm(
                    pool.imap(self.process_chunk, param_chunks),
                    total=len(param_chunks),
                    desc="Processing chunks"
                ))
            
            # Reorganize results
            for chunk in chunk_results:
                for n_states, result in chunk:
                    if 'error' not in result:
                        results[n_states].append(result)
            
        finally:
            # Clean up shared memory
            self.cleanup_shared_memory()
        
        return results

    def get_best_model(self, results: Dict, criterion='aic') -> Tuple:
        """Enhanced model selection with stability metrics"""
        best_score = float('inf')
        best_n_states = None
        
        scores_summary = {}
        convergence_summary = {}
        performance_summary = {}
        
        for n_states, metrics_list in results.items():
            if not metrics_list:
                continue
            
            # Filter converged models
            converged_metrics = [m for m in metrics_list if m.get('converged', False)]
            metrics_to_use = converged_metrics if converged_metrics else metrics_list
            
            # Calculate scores
            scores = [metrics[criterion] for metrics in metrics_to_use]
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            
            # Calculate performance metrics
            processing_times = [m.get('processing_time', 0) for m in metrics_to_use]
            memory_deltas = [m.get('memory_delta', 0) for m in metrics_to_use]
            
            scores_summary[n_states] = {
                'mean': mean_score,
                'std': std_score,
                'adjusted': mean_score + std_score,
                'n_converged': len(converged_metrics),
                'n_total': len(metrics_list)
            }
            
            convergence_summary[n_states] = {
                'converged_ratio': len(converged_metrics) / len(metrics_list),
                'avg_attempts': np.mean([m.get('n_attempts', self.max_tries) 
                                      for m in metrics_list])
            }
            
            performance_summary[n_states] = {
                'avg_processing_time': np.mean(processing_times),
                'avg_memory_delta': np.mean(memory_deltas)
            }
            
            # Update best model
            adjusted_score = mean_score + std_score
            if adjusted_score < best_score:
                best_score = adjusted_score
                best_n_states = n_states
        
        return best_n_states, scores_summary, convergence_summary, performance_summary

    def analyze_group(self, group_data: np.ndarray, group_name: str) -> Tuple:
        """Main analysis pipeline with enhanced monitoring"""
        start_time = time.time()
        
        try:
            logging.info(f"Starting analysis for group: {group_name}")
            results = self.find_optimal_states(group_data)
            
            best_n_states, scores_summary, convergence_summary, performance_summary = \
                self.get_best_model(results)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            self.best_models[group_name] = {
                'results': results,
                'best_n_states': best_n_states,
                'scores_summary': scores_summary,
                'convergence_summary': convergence_summary,
                'performance_summary': performance_summary,
                'total_processing_time': total_time,
                'convergence_stats': self.convergence_stats.copy()
            }
            
            return (results, best_n_states, scores_summary, 
                   convergence_summary, performance_summary)
            
        except Exception as e:
            logging.error(f"Error in group analysis: {str(e)}")
            raise

def main(group_data: np.ndarray, 
         group_name: str, 
         output_dir: str, 
         n_components_range: Tuple[int, int], 
         random_state: int, 
         n_jobs: int, 
         max_tries: int,
         chunk_size: int = None):
    """Main execution function with enhanced error handling and reporting"""
    
    start_time = time.time()
    
    try:
        logging.info(f"Starting HMM analysis for {group_name}")
        logging.info(f"Data shape: {group_data.shape}")
        
        # Initialize analyzer
        analyzer = GroupHMMAnalysis(
            n_components_range=n_components_range,
            random_state=random_state,
            n_jobs=n_jobs,
            max_tries=max_tries,
            chunk_size=chunk_size
        )

        # Run analysis
        results, best_n_states, scores_summary, convergence_summary, performance_summary = \
            analyzer.analyze_group(group_data, group_name)
        
        # Prepare comprehensive output
        output = {
            'results': results,
            'best_n_states': best_n_states,
            'scores_summary': scores_summary,
            'convergence_summary': convergence_summary,
            'performance_summary': performance_summary,
            'convergence_stats': analyzer.convergence_stats,
            'performance_stats': analyzer.performance_stats,
            'parameters': {
                'n_components_range': n_components_range,
                'random_state': random_state,
                'n_jobs': n_jobs,
                'max_tries': max_tries,
                'chunk_size': chunk_size
            }
        }
        
        # Save results
        output_file = os.path.join(output_dir, f"{group_name}_models.pkl")
        with open(output_file, "wb") as f:
            pickle.dump(output, f)
        
        # Generate summary report
        end_time = time.time()
        total_time = end_time - start_time
        
        print("\n" + "="*50)
        print("Analysis Summary Report")
        print("="*50)
        print(f"\nGroup: {group_name}")
        print(f"Best number of states: {best_n_states}")
        
        print("\nConvergence Statistics:")
        print(f"Total attempts: {analyzer.convergence_stats['total_attempts']}")
        print(f"Convergence failures: {analyzer.convergence_stats['convergence_failures']}")
        print(f"Other failures: {analyzer.convergence_stats['other_failures']}")
        
        print("\nPerformance Metrics:")
        print(f"Total processing time: {total_time:.2f} seconds")
        print(f"Average memory usage: {np.mean(analyzer.performance_stats['memory_usage']):.2f} MB")
        
        print("\nConvergence Summary by States:")
        for n_states in convergence_summary:
            print(f"\nStates {n_states}:")
            print(f"Convergence ratio: {convergence_summary[n_states]['converged_ratio']:.2f}")
            print(f"Average attempts: {convergence_summary[n_states]['avg_attempts']:.2f}")
            if n_states in performance_summary:
                print(f"Average processing time: {performance_summary[n_states]['avg_processing_time']:.2f} seconds")
        
        logging.info("Analysis completed successfully")
        return output
        
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    # Set up logging file
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('hmm_analysis.log')
    file_handler.setFormatter(log_formatter)
    logging.getLogger().addHandler(file_handler)
    
    # Parse arguments
    parser = ArgumentParser(description='HMM Analysis with optimized parallel processing')
    parser.add_argument("--res", type=str, default="native", 
                       help="Resolution for analysis")
    parser.add_argument("--group", type=str, default="affair", 
                       help="Group name for analysis")
    parser.add_argument("--n_components_min", type=int, default=2, 
                       help="Minimum number of states")
    parser.add_argument("--n_components_max", type=int, default=12, 
                       help="Maximum number of states")
    parser.add_argument("--random_state", type=int, default=42, 
                       help="Random state for reproducibility")
    parser.add_argument("--n_jobs", type=int, default=20, 
                       help="Number of parallel jobs")
    parser.add_argument("--max_tries", type=int, default=5, 
                       help="Maximum attempts for model fitting")
    parser.add_argument("--chunk_size", type=int, default=None, 
                       help="Size of chunks for parallel processing")
    args = parser.parse_args()

    # Load environment variables
    load_dotenv()
    base_dir = os.getenv("BASE_DIR")
    scratch_dir = os.getenv("SCRATCH_DIR")

    # Prepare output directory
    output_dir = os.path.join(scratch_dir, "output", f"group_hmm_{args.res}_{args.group}")
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    try:
        if args.group == "combined":
            group_data1 = np.load(os.path.join(scratch_dir, "output", 
                                              f"atlas_masked_{args.res}", 
                                              "affair_group_data_roi.npy"))
            group_data2 = np.load(os.path.join(scratch_dir, "output", 
                                              f"atlas_masked_{args.res}", 
                                              "paranoia_group_data_roi.npy"))
            group_data = np.concatenate([group_data1, group_data2], axis=0)
        else:
            group_data = np.load(os.path.join(scratch_dir, "output", 
                                             f"atlas_masked_{args.res}", 
                                             f"{args.group}_group_data.npy"))
        
        # Run main analysis
        main(
            group_data=group_data,
            group_name=args.group,
            output_dir=output_dir,
            n_components_range=(args.n_components_min, args.n_components_max),
            random_state=args.random_state,
            n_jobs=args.n_jobs,
            max_tries=args.max_tries,
            chunk_size=args.chunk_size
        )
        
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        raise