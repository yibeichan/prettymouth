import numpy as np
import matplotlib.pyplot as plt
import json
import glob
import os
import pandas as pd
import seaborn as sns
from pathlib import Path
from scipy.stats import entropy
from sklearn.metrics import silhouette_score
import pickle

def compare_hmm_models_comprehensive(base_dir, group_name):
    """
    Comprehensive comparison of HMM models with different numbers of states.
    
    Args:
        base_dir: Base directory containing all model folders
        group_name: Group identifier (e.g., 'combined', 'group1', 'group2')
    
    Returns:
        dict: Results of model comparison including recommended model
    """
    output_dir = os.path.join(base_dir, "05_model_comparison")
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all model directories
    model_dirs = sorted(glob.glob(f"{base_dir}/04_{group_name}_hmm_*states_ntw_native_trimmed"))
    if not model_dirs:
        raise ValueError(f"No model directories found for {group_name}")
    
    # Extract stats for all models
    models_data = []
    n_states_list = []
    
    for model_dir in model_dirs:
        # Extract n_states from directory name
        n_states = int(os.path.basename(model_dir).split('_')[3].replace('states', ''))
        n_states_list.append(n_states)
        
        stats_dir = os.path.join(model_dir, 'statistics')
        
        # Load CV results
        cv_path = os.path.join(stats_dir, f"{group_name}_cv_results.json")
        with open(cv_path, 'r') as f:
            cv_results = json.load(f)
        
        # Load summary stats
        summary_path = os.path.join(stats_dir, f"{group_name}_summary.json")
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        
        # Load metrics (contains additional information)
        metrics_path = os.path.join(stats_dir, f"{group_name}_metrics.pkl")
        with open(metrics_path, 'rb') as f:
            metrics = pickle.load(f)

        # Extract average duration
        avg_duration = np.nan
        
        # First, try to get durations from group_level metrics
        if 'state_metrics' in metrics and 'group_level' in metrics['state_metrics']:
            group_metrics = metrics['state_metrics']['group_level']
            
            # Check if durations are directly in group metrics
            if 'durations' in group_metrics:
                # Calculate average duration across all states
                durations = []
                for state, stats in group_metrics['durations'].items():
                    if 'mean' in stats:
                        durations.append(stats['mean'])
                
                if durations:
                    avg_duration = np.mean(durations)
            # If not found directly, check if there's average durations per state
            elif 'average_durations' in group_metrics:
                avg_duration = np.mean(list(group_metrics['average_durations'].values()))
        
        # If still not found, look at other potential locations
        if np.isnan(avg_duration):
            # Try direct structure
            if 'state_durations' in metrics:
                if isinstance(metrics['state_durations'], (list, np.ndarray)):
                    avg_duration = np.mean(metrics['state_durations'])
                elif isinstance(metrics['state_durations'], dict):
                    avg_duration = np.mean(list(metrics['state_durations'].values()))
        
        # Load state sequences for additional metrics
        seq_path = os.path.join(stats_dir, f"{group_name}_state_sequences.npy")
        state_sequences = np.load(seq_path)
        
        # Calculate additional metrics
        additional_metrics = calculate_additional_metrics(state_sequences, metrics)
        
        # Load model file to get information about multiple initializations
        model_path = os.path.join(model_dir, 'models', f"{group_name}_hmm_model.pkl")
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # If initialization scores available, calculate stability
        if 'initialization_scores' in model_data:
            scores = model_data['initialization_scores']
            # Calculate coefficient of variation of scores
            stability_metric = 1 - (np.std(scores) / np.mean(np.abs(scores)))
            additional_metrics['initialization_stability'] = stability_metric
        
        # As a fallback, calculate durations directly from state sequences if needed
        if np.isnan(avg_duration):
            # Calculate state durations from sequences
            durations_by_state = {}
            for subj in range(state_sequences.shape[0]):
                seq = state_sequences[subj, :]
                run_lengths = []
                run_states = []
                current_state = seq[0]
                current_run = 1
                
                for i in range(1, len(seq)):
                    if seq[i] == current_state:
                        current_run += 1
                    else:
                        run_lengths.append(current_run)
                        run_states.append(current_state)
                        current_state = seq[i]
                        current_run = 1
                
                # Don't forget the last run
                run_lengths.append(current_run)
                run_states.append(current_state)
                
                # Add to dictionary
                for state, duration in zip(run_states, run_lengths):
                    if state not in durations_by_state:
                        durations_by_state[state] = []
                    durations_by_state[state].append(duration)
            
            # Calculate average duration across all states
            all_durations = []
            for state, durations in durations_by_state.items():
                all_durations.extend(durations)
            
            if all_durations:
                avg_duration = np.mean(all_durations)
        
        # Compile model data
        model_data = {
            'n_states': n_states,
            'state_reliability': cv_results['state_reliability'],
            'mean_log_likelihood': cv_results['mean_log_likelihood'],
            'bic': summary['model_performance']['bic'],
            'avg_duration': avg_duration,
            'state_entropy': additional_metrics['state_entropy'],
            'transition_entropy': additional_metrics['transition_entropy'],
            'state_usage_balance': additional_metrics['state_usage_balance'],
            'temporal_consistency': additional_metrics['temporal_consistency'],
            'folder': model_dir
        }
        
        models_data.append(model_data)
    
    # Convert to DataFrame for easier analysis
    models_df = pd.DataFrame(models_data)
    models_df = models_df.sort_values('n_states')
    
    # Calculate basic rankings directly to ensure they exist
    models_df['reliability_rank'] = models_df['state_reliability'].rank(ascending=False)
    models_df['likelihood_rank'] = models_df['mean_log_likelihood'].rank(ascending=False)
    models_df['bic_rank'] = models_df['bic'].rank(ascending=True)
    models_df['basic_weighted_rank'] = (0.4 * models_df['reliability_rank'] + 
                                      0.3 * models_df['likelihood_rank'] + 
                                      0.3 * models_df['bic_rank'])
    
    # If the full rank_models function works, use its results
    try:
        rank_results = rank_models(models_df)
        # Add the full ranking results
        for col in rank_results.columns:
            models_df[col] = rank_results[col]
    except Exception as e:
        print(f"Warning: Error in rank_models: {e}")
        # We already have fallback rankings
        models_df['weighted_rank'] = models_df['basic_weighted_rank']
    
    # Ensure weighted_rank exists
    if 'weighted_rank' not in models_df.columns:
        models_df['weighted_rank'] = models_df['basic_weighted_rank']
    
    # Generate plots
    try:
        plot_comprehensive_comparison(models_df, group_name, output_dir)
    except Exception as e:
        print(f"Warning: Error in plotting: {e}")
    
    # Add new analyses from enhanced version
    try:
        sequence_metrics = analyze_state_sequences(model_dirs, group_name)
        stability_metrics = analyze_model_stability(model_dirs, group_name)
        cv_detail_metrics = analyze_cv_results(model_dirs, group_name)
        
        # Integrate new metrics with existing data
        for model_data in models_data:
            n_states = model_data['n_states']
            
            # Add sequence analysis metrics
            if n_states in sequence_metrics:
                for key, value in sequence_metrics[n_states].items():
                    model_data[key] = value
            
            # Add stability metrics
            if n_states in stability_metrics:
                for key, value in stability_metrics[n_states].items():
                    model_data[key] = value
            
            # Add detailed CV metrics
            if n_states in cv_detail_metrics:
                for key, value in cv_detail_metrics[n_states].items():
                    model_data[key] = value
        
        # Create a new DataFrame with extended metrics
        extended_df = pd.DataFrame(models_data)
        extended_df = extended_df.sort_values('n_states')
        
        # Add all the ranking columns back
        for col in models_df.columns:
            if col not in extended_df.columns and col.endswith('_rank'):
                extended_df[col] = models_df[col]
        
        # Make sure weighted_rank is present
        if 'weighted_rank' not in extended_df.columns:
            extended_df['weighted_rank'] = models_df['weighted_rank']
            
        # Update the main DataFrame
        models_df = extended_df
        
    except Exception as e:
        print(f"Warning: Error in additional metrics: {e}")
    
    # Create enhanced plots
    try:
        plot_enhanced_comparison(models_df, group_name, output_dir)
    except Exception as e:
        print(f"Warning: Error in enhanced plotting: {e}")
    
    # Add neuroscientific assessment
    try:
        models_df = neuroscientific_assessment(models_df)
    except Exception as e:
        print(f"Warning: Error in neuroscientific assessment: {e}")
        models_df['neuro_plausibility'] = 0.5  # Default value if assessment fails
    
    # Ensure neuro_plausibility exists
    if 'neuro_plausibility' not in models_df.columns:
        models_df['neuro_plausibility'] = 0.5  # Default value
    
    # Identify best model using weighted ranking with neuroscientific plausibility
    try:
        final_scores = models_df['weighted_rank'] * 0.7 + (1 - models_df['neuro_plausibility']) * 0.3
        models_df['final_score'] = final_scores
        best_model_idx = models_df['final_score'].idxmin()
    except Exception as e:
        print(f"Warning: Error in final scoring: {e}")
        # Fallback to simpler method
        if 'weighted_rank' in models_df.columns:
            best_model_idx = models_df['weighted_rank'].idxmin()
        else:
            best_model_idx = models_df['basic_weighted_rank'].idxmin()
    
    best_n_states = models_df.iloc[best_model_idx]['n_states']
    
    # Create a modified version of print_recommendation that handles missing columns
    def modified_print_recommendation(results_dict, df):
        print("\n" + "="*80)
        print(f"HMM MODEL COMPARISON SUMMARY")
        print("="*80)
        
        # Only print metrics if they exist
        for metric in ['optimal_by_reliability', 'optimal_by_likelihood', 'optimal_by_bic', 'optimal_by_weighted_rank']:
            if metric in results_dict:
                print(f"{metric}: {results_dict[metric]}")
        
        print("-"*80)
        print(f"RECOMMENDED MODEL: {results_dict['recommended_model']} states")
        
        if 'recommendation_rationale' in results_dict:
            print(f"Rationale: {results_dict['recommendation_rationale']}")
        
        print("="*80 + "\n")
    
    # Create a modified version of generate_recommendation_rationale
    def modified_recommendation_rationale(df, best_idx):
        best_model = df.iloc[best_idx]
        n_states = int(best_model['n_states'])
        
        rationale = [f"The {n_states}-state model is recommended based on overall performance."]
        
        # Only add specific metrics if they exist
        strengths = []
        if 'state_reliability' in df.columns:
            if df.loc[best_idx, 'state_reliability'] >= df['state_reliability'].median():
                strengths.append("good reliability")
                
        if 'bic' in df.columns:
            if df.loc[best_idx, 'bic'] <= df['bic'].median():
                strengths.append("favorable BIC")
        
        if strengths:
            rationale.append("Key strengths: " + ", ".join(strengths) + ".")
            
        return " ".join(rationale)
    
    # Save comparison results
    try:
        # Find best models for each metric (with error handling)
        reliability_best = int(models_df.loc[models_df['state_reliability'].idxmax()]['n_states']) if 'state_reliability' in models_df.columns else None
        likelihood_best = int(models_df.loc[models_df['mean_log_likelihood'].idxmax()]['n_states']) if 'mean_log_likelihood' in models_df.columns else None
        bic_best = int(models_df.loc[models_df['bic'].idxmin()]['n_states']) if 'bic' in models_df.columns else None
        
        results = {
            'models_data': models_df.to_dict('records'),
            'recommended_model': int(best_n_states)
        }
        
        # Only add metrics if they were successfully calculated
        if reliability_best is not None:
            results['optimal_by_reliability'] = reliability_best
        if likelihood_best is not None:
            results['optimal_by_likelihood'] = likelihood_best
        if bic_best is not None:
            results['optimal_by_bic'] = bic_best
        if 'weighted_rank' in models_df.columns:
            results['optimal_by_weighted_rank'] = int(best_n_states)
            
        # Generate rationale using the safer function
        results['recommendation_rationale'] = modified_recommendation_rationale(models_df, best_model_idx)
        
        # Save detailed comparison to file
        with open(os.path.join(output_dir, f"{group_name}_model_comparison.json"), 'w') as f:
            # Convert DataFrame to list for JSON serialization
            serializable_results = {k: (v if not isinstance(v, pd.DataFrame) else v.to_dict('records')) 
                                   for k, v in results.items()}
            json.dump(serializable_results, f, indent=2)
    except Exception as e:
        print(f"Warning: Error in saving results: {e}")
        # Minimal results if full results fail
        results = {
            'recommended_model': int(best_n_states),
            'error': str(e)
        }
    
    # Print recommendation using the safer function
    try:
        modified_print_recommendation(results, models_df)
    except Exception as e:
        print(f"Warning: Error in printing recommendation: {e}")
        print(f"RECOMMENDED MODEL: {best_n_states} states")
    
    return results

def print_recommendation(results):
    """
    Print recommendation summary to console.
    
    Args:
        results: Dictionary of comparison results
    """
    print("\n" + "="*80)
    print(f"HMM MODEL COMPARISON SUMMARY")
    print("="*80)
    
    # Only print metrics if they exist
    for metric in ['optimal_by_reliability', 'optimal_by_likelihood', 'optimal_by_bic', 'optimal_by_weighted_rank']:
        if metric in results:
            print(f"{metric}: {results[metric]}")
    
    print("-"*80)
    print(f"RECOMMENDED MODEL: {results['recommended_model']} states")
    
    if 'recommendation_rationale' in results:
        print(f"Rationale: {results['recommendation_rationale']}")
    
    print("="*80 + "\n")

def generate_recommendation_rationale(models_df, best_idx):
    """
    Generate a text explanation for model recommendation.
    
    Args:
        models_df: DataFrame with model metrics
        best_idx: Index of the best model
        
    Returns:
        str: Rationale for recommendation
    """
    try:
        best_model = models_df.iloc[best_idx]
        n_states = int(best_model['n_states'])
        
        # Find rankings for the best model (with error handling)
        rationale = [f"The {n_states}-state model is recommended based on weighted ranking across multiple metrics."]
        
        # Add specific strengths (only if columns exist)
        strengths = []
        for col_name, col_label in [
            ('reliability_rank', 'state pattern reliability'),
            ('bic_rank', 'optimal BIC'),
            ('state_usage_balance', 'balanced state usage'),
            ('temporal_consistency', 'temporal consistency')
        ]:
            try:
                if col_name in best_model and col_name in models_df.columns:
                    rank = best_model[col_name]
                    total = len(models_df)
                    if col_name.endswith('_rank') and rank <= 2:
                        strengths.append(f"{col_label} (ranked {int(rank)} of {total})")
                    elif not col_name.endswith('_rank') and best_model[col_name] >= models_df[col_name].median():
                        strengths.append(f"good {col_label}")
            except:
                continue
        
        if strengths:
            rationale.append("Key strengths: " + ", ".join(strengths) + ".")
            
        # Add neuroscientific context
        rationale.append(f"The {n_states}-state model provides a balance between model complexity and interpretability.")
        
        return " ".join(rationale)
    except Exception as e:
        return f"The {n_states}-state model is recommended based on overall performance."

def calculate_additional_metrics(state_sequences, metrics):
    """
    Calculate additional metrics for model evaluation.
    
    Args:
        state_sequences: State sequences from HMM model
        metrics: Dictionary of metrics with nested structure
    
    Returns:
        dict: Additional metrics
    """
    # State occupancy distribution
    state_counts = np.bincount(state_sequences.flatten())
    state_probs = state_counts / np.sum(state_counts)
    
    # State entropy (how evenly states are used)
    state_entropy = entropy(state_probs)
    
    # Transition matrix entropy (how predictable transitions are)
    transition_matrix = None
    
    # Try to extract transition matrix from the nested structure
    if 'state_metrics' in metrics and 'temporal' in metrics['state_metrics']:
        transition_matrix = metrics['state_metrics']['group_level']['transitions']
    
    transition_entropy = np.nan
    if transition_matrix is not None:
        # Calculate entropy of each row and average
        transition_entropy = np.mean([entropy(row) for row in transition_matrix])
    
    # State usage balance (coefficient of variation of state usage)
    state_usage_balance = 1 - (np.std(state_probs) / np.mean(state_probs) if np.mean(state_probs) > 0 else np.nan)
    
    # Temporal consistency (how consistent states are across subjects at same timepoints)
    # For each timepoint, calculate entropy of state distribution across subjects
    n_subjects, n_timepoints = state_sequences.shape
    timepoint_entropies = []
    
    for t in range(n_timepoints):
        state_dist = np.bincount(state_sequences[:, t], minlength=len(state_probs))
        state_dist = state_dist / n_subjects
        if np.sum(state_dist) > 0:
            timepoint_entropies.append(entropy(state_dist))
    
    temporal_consistency = 1 - (np.mean(timepoint_entropies) / np.log(len(state_probs)) 
                               if timepoint_entropies else np.nan)
    
    return {
        'state_entropy': state_entropy,
        'transition_entropy': transition_entropy,
        'state_usage_balance': state_usage_balance,
        'temporal_consistency': temporal_consistency
    }

def rank_models(models_df):
    """
    Rank models based on different metrics and create a weighted ranking.
    
    Args:
        models_df: DataFrame with model metrics
        
    Returns:
        DataFrame: Ranking results
    """
    # Create copy to avoid modifying original
    df = models_df.copy()
    
    # Initialize columns for ranks
    rank_columns = {}
    
    # Rank by each metric (lower rank is better)
    # For metrics where higher is better, multiply by -1 before ranking
    rank_columns['reliability_rank'] = df['state_reliability'].rank(ascending=False)
    rank_columns['likelihood_rank'] = df['mean_log_likelihood'].rank(ascending=False)
    rank_columns['bic_rank'] = df['bic'].rank(ascending=True)  # Lower BIC is better
    # No AIC rank
    rank_columns['entropy_rank'] = df['state_entropy'].rank(ascending=False)  # Higher entropy is better
    rank_columns['balance_rank'] = df['state_usage_balance'].rank(ascending=False)  # Higher balance is better
    rank_columns['consistency_rank'] = df['temporal_consistency'].rank(ascending=False)  # Higher consistency is better
    
    # Define weights for each metric - adjusted without AIC
    weights = {
        'reliability_rank': 0.30,  # Increased weight
        'likelihood_rank': 0.15,
        'bic_rank': 0.25,  
        'entropy_rank': 0.10,
        'balance_rank': 0.10,
        'consistency_rank': 0.10
    }
    
    # Calculate weighted rank
    weighted_rank = sum(rank_columns[col] * weights[col] for col in rank_columns)
    rank_columns['weighted_rank'] = weighted_rank
    
    return pd.DataFrame(rank_columns)

def plot_comprehensive_comparison(models_df, group_name, output_dir):
    """
    Create comprehensive plots for model comparison.
    
    Args:
        models_df: DataFrame with model metrics
        group_name: Group identifier
        output_dir: Directory to save outputs
    """
    n_states_list = models_df['n_states'].values
    
    # Create a more comprehensive visualization
    fig = plt.figure(figsize=(15, 18))
    gs = fig.add_gridspec(4, 2)
    
    # 1. Information criteria plot
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(n_states_list, models_df['bic'].values, 'o-', label='BIC')
    # No AIC plot
    ax1.set_xlabel('Number of States')
    ax1.set_ylabel('Information Criterion')
    ax1.set_title('Model Complexity Trade-off')
    ax1.legend()
    
    # 2. Cross-validation metrics
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(n_states_list, models_df['state_reliability'].values, 'o-', label='State Pattern Reliability')
    ax2.plot(n_states_list, models_df['mean_log_likelihood'].values / np.max(np.abs(models_df['mean_log_likelihood'].values)), 
             's-', label='Normalized Log-likelihood')
    ax2.set_xlabel('Number of States')
    ax2.set_ylabel('Cross-validation Metric')
    ax2.set_title('Cross-validation Performance')
    ax2.legend()
    
    # 3. State usage metrics
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(n_states_list, models_df['state_entropy'].values, 'o-', label='State Entropy')
    ax3.plot(n_states_list, models_df['state_usage_balance'].values, 's-', label='State Usage Balance')
    ax3.set_xlabel('Number of States')
    ax3.set_ylabel('Metric Value')
    ax3.set_title('State Usage Metrics')
    ax3.legend()
    
    # 4. Temporal metrics
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(n_states_list, models_df['temporal_consistency'].values, 'o-', label='Temporal Consistency')
    ax4.plot(n_states_list, models_df['transition_entropy'].values, 's-', label='Transition Entropy')
    ax4.set_xlabel('Number of States')
    ax4.set_ylabel('Metric Value')
    ax4.set_title('Temporal Dynamics Metrics')
    ax4.legend()
    
    # 5. Average state duration
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(n_states_list, models_df['avg_duration'].values, 'o-')
    ax5.set_xlabel('Number of States')
    ax5.set_ylabel('Average Duration (TRs)')
    ax5.set_title('Average State Duration')
    
    # 6. Rankings heatmap
    ax6 = fig.add_subplot(gs[2, 1])
    rank_cols = [col for col in models_df.columns if col.endswith('_rank')]
    rank_data = models_df[rank_cols].values
    sns.heatmap(rank_data, 
                annot=True, 
                cmap='viridis_r',
                xticklabels=[col.replace('_rank', '') for col in rank_cols],
                yticklabels=n_states_list,
                ax=ax6)
    ax6.set_title('Model Rankings by Metric (lower is better)')
    ax6.set_ylabel('Number of States')
    
    # 7. Weighted rank
    ax7 = fig.add_subplot(gs[3, :])
    bars = ax7.bar(n_states_list, models_df['weighted_rank'].values)
    best_idx = np.argmin(models_df['weighted_rank'].values)
    bars[best_idx].set_color('green')
    
    ax7.set_xlabel('Number of States')
    ax7.set_ylabel('Weighted Rank')
    ax7.set_title('Overall Model Ranking (lower is better)')
    
    # Add text showing the recommendation
    best_n_states = models_df.iloc[best_idx]['n_states']
    ax7.text(0.5, 0.9, f'Recommended model: {int(best_n_states)} states',
             horizontalalignment='center',
             transform=ax7.transAxes,
             bbox=dict(facecolor='white', alpha=0.8))
    
    # Finalize and save
    plt.tight_layout()
    fig.suptitle(f'Comprehensive HMM Model Comparison - {group_name}', fontsize=16, y=1.02)
    plt.savefig(os.path.join(output_dir, f"{group_name}_comprehensive_model_comparison.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create elbow plot for key metrics
    plt.figure(figsize=(10, 6))
    plt.plot(n_states_list, models_df['bic'].values, 'o-', label='BIC')
    plt.plot(n_states_list, models_df['state_reliability'].values * np.max(models_df['bic'].values) * 0.5, 's-', 
             label='Reliability (scaled)')
    
    # Find elbow points
    from kneed import KneeLocator
    if len(n_states_list) > 2:
        try:
            kneedle_bic = KneeLocator(n_states_list, models_df['bic'].values, 
                                      S=1.0, curve='convex', direction='decreasing')
            if kneedle_bic.knee:
                plt.axvline(x=kneedle_bic.knee, color='r', linestyle='--', label=f'BIC Elbow: {kneedle_bic.knee} states')
        except:
            pass  # Skip if knee detection fails
            
    plt.xlabel('Number of States')
    plt.ylabel('Metric Value')
    plt.title(f'Elbow Plot for Model Selection - {group_name}')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, f"{group_name}_model_elbow_plot.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()

def generate_recommendation_rationale(models_df, best_idx):
    """
    Generate a text explanation for model recommendation.
    
    Args:
        models_df: DataFrame with model metrics
        best_idx: Index of the best model
        
    Returns:
        str: Rationale for recommendation
    """
    best_model = models_df.iloc[best_idx]
    n_states = int(best_model['n_states'])
    
    # Find rankings for the best model
    reliability_rank = best_model['reliability_rank']
    bic_rank = best_model['bic_rank']
    balance_rank = best_model['balance_rank']
    consistency_rank = best_model['consistency_rank']
    
    # Count how many models we have
    total_models = len(models_df)
    
    rationale = [f"The {n_states}-state model is recommended based on weighted ranking across multiple metrics."]
    
    # Add specific strengths
    strengths = []
    if reliability_rank <= 2:
        strengths.append(f"high state pattern reliability (ranked {int(reliability_rank)} of {total_models})")
    if bic_rank <= 2:
        strengths.append(f"optimal BIC (ranked {int(bic_rank)} of {total_models})")
    if balance_rank <= 2:
        strengths.append(f"balanced state usage (ranked {int(balance_rank)} of {total_models})")
    if consistency_rank <= 2:
        strengths.append(f"temporal consistency (ranked {int(consistency_rank)} of {total_models})")
    
    if strengths:
        rationale.append("Key strengths: " + ", ".join(strengths) + ".")
    
    # Add comparison to more complex models
    more_complex = models_df[models_df['n_states'] > n_states]
    if not more_complex.empty:
        bic_improvement = (more_complex['bic'].min() - best_model['bic']) / best_model['bic'] * 100
        if bic_improvement < 5:
            rationale.append(f"More complex models offer marginal improvement ({bic_improvement:.1f}% BIC reduction at best).")
    
    # Add neuroscientific context
    rationale.append(f"The {n_states}-state model provides a balance between model complexity and neurobiological interpretability.")
    
    return " ".join(rationale)

def print_recommendation(results):
    """
    Print recommendation summary to console.
    
    Args:
        results: Dictionary of comparison results
    """
    print("\n" + "="*80)
    print(f"HMM MODEL COMPARISON SUMMARY")
    print("="*80)
    print(f"Optimal number of states by reliability: {results['optimal_by_reliability']}")
    print(f"Optimal number of states by log-likelihood: {results['optimal_by_likelihood']}")
    print(f"Optimal number of states by BIC: {results['optimal_by_bic']}")
    print(f"Optimal number of states by weighted rank: {results['optimal_by_weighted_rank']}")
    print("-"*80)
    print(f"RECOMMENDED MODEL: {results['recommended_model']} states")
    print(f"Rationale: {results['recommendation_rationale']}")
    print("="*80 + "\n")

def analyze_state_sequences(model_dirs, group_name):
    """Extract additional metrics from state sequences across models."""
    sequence_metrics = {}
    
    for model_dir in model_dirs:
        n_states = int(os.path.basename(model_dir).split('_')[3].replace('states', ''))
        seq_path = os.path.join(model_dir, 'statistics', f"{group_name}_state_sequences.npy")
        
        if os.path.exists(seq_path):
            sequences = np.load(seq_path)
            n_subjects, n_timepoints = sequences.shape
            
            # Calculate inter-subject synchrony
            subject_correlations = []
            for i in range(n_subjects):
                for j in range(i+1, n_subjects):
                    corr = np.corrcoef(sequences[i], sequences[j])[0,1]
                    subject_correlations.append(corr)
            
            # Calculate state transition rate
            transitions = 0
            for subj in range(n_subjects):
                transitions += np.sum(np.diff(sequences[subj]) != 0)
            transition_rate = transitions / (n_subjects * (n_timepoints-1))
            
            # Calculate mutual information across timepoints
            from sklearn.metrics import mutual_info_score
            time_mutual_info = []
            for t1 in range(0, n_timepoints, 10):  # Sample every 10 timepoints for efficiency
                for t2 in range(t1+10, n_timepoints, 10):
                    mi = mutual_info_score(sequences[:, t1], sequences[:, t2])
                    time_mutual_info.append(mi)
            
            sequence_metrics[n_states] = {
                'inter_subject_synchrony': np.mean(subject_correlations),
                'transition_rate': transition_rate,
                'temporal_mutual_info': np.mean(time_mutual_info) if time_mutual_info else 0
            }
    
    return sequence_metrics

def analyze_model_stability(model_dirs, group_name):
    """Analyze stability of models based on saved model files."""
    stability_metrics = {}
    
    for model_dir in model_dirs:
        n_states = int(os.path.basename(model_dir).split('_')[3].replace('states', ''))
        model_path = os.path.join(model_dir, 'models', f"{group_name}_hmm_model.pkl")
        
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                # Extract model object
                model = model_data.get('model', None)
                
                if model is not None:
                    # Calculate eigenvalues of transition matrix
                    eigenvals = np.linalg.eigvals(model.transmat_)
                    # Second largest eigenvalue indicates mixing rate
                    sorted_evals = np.sort(np.abs(eigenvals))
                    mixing_rate = 1 - sorted_evals[-2]
                    
                    # Calculate condition number of covariance matrices
                    cond_numbers = []
                    for i in range(n_states):
                        if hasattr(model, 'covars_'):
                            if model.covariance_type == 'full':
                                cov = model.covars_[i]
                                cond_num = np.linalg.cond(cov)
                                cond_numbers.append(cond_num)
                    
                    stability_metrics[n_states] = {
                        'mixing_rate': mixing_rate,
                        'mean_cov_condition_number': np.mean(cond_numbers) if cond_numbers else np.nan
                    }
            except Exception as e:
                print(f"Error analyzing model {n_states}: {str(e)}")
    
    return stability_metrics

def analyze_cv_results(model_dirs, group_name):
    """Extract detailed information from cross-validation results."""
    cv_metrics = {}
    
    for model_dir in model_dirs:
        n_states = int(os.path.basename(model_dir).split('_')[3].replace('states', ''))
        cv_path = os.path.join(model_dir, 'statistics', f"{group_name}_cv_results.json")
        
        if os.path.exists(cv_path):
            with open(cv_path, 'r') as f:
                cv_data = json.load(f)
            
            # Extract fold-specific metrics if available
            fold_likelihoods = cv_data.get('fold_likelihoods', [])
            fold_reliabilities = cv_data.get('fold_reliabilities', [])
            
            if fold_likelihoods and fold_reliabilities:
                cv_metrics[n_states] = {
                    'likelihood_std': np.std(fold_likelihoods),
                    'reliability_std': np.std(fold_reliabilities),
                    'min_reliability': np.min(fold_reliabilities),
                    'likelihood_range': np.max(fold_likelihoods) - np.min(fold_likelihoods)
                }
    
    return cv_metrics

def enhanced_rank_models(models_df):
    """
    Enhanced ranking function incorporating additional metrics.
    
    Args:
        models_df: DataFrame with model metrics
        
    Returns:
        DataFrame: Ranking results
    """
    # Create copy to avoid modifying original
    df = models_df.copy()
    
    # Initialize columns for ranks
    rank_columns = {}
    
    # Original rankings
    rank_columns['reliability_rank'] = df['state_reliability'].rank(ascending=False)
    rank_columns['likelihood_rank'] = df['mean_log_likelihood'].rank(ascending=False)
    rank_columns['bic_rank'] = df['bic'].rank(ascending=True)
    rank_columns['aic_rank'] = df['aic'].rank(ascending=True)
    
    # Add new metric rankings if available
    if 'inter_subject_synchrony' in df.columns:
        rank_columns['synchrony_rank'] = df['inter_subject_synchrony'].rank(ascending=False)
    
    if 'mixing_rate' in df.columns:
        rank_columns['mixing_rank'] = df['mixing_rate'].rank(ascending=False)
    
    if 'mean_cov_condition_number' in df.columns:
        # Lower condition number is better (more stable)
        rank_columns['stability_rank'] = df['mean_cov_condition_number'].rank(ascending=True)
    
    if 'likelihood_std' in df.columns:
        # Lower std is better (more consistent)
        rank_columns['consistency_rank'] = df['likelihood_std'].rank(ascending=True)
    
    # Define weights for each metric
    weights = {
        'reliability_rank': 0.20,
        'likelihood_rank': 0.15,
        'bic_rank': 0.15,
        'aic_rank': 0.10,
    }
    
    # Add weights for new metrics
    if 'synchrony_rank' in rank_columns:
        weights['synchrony_rank'] = 0.10
    if 'mixing_rank' in rank_columns:
        weights['mixing_rank'] = 0.10
    if 'stability_rank' in rank_columns:
        weights['stability_rank'] = 0.10
    if 'consistency_rank' in rank_columns:
        weights['consistency_rank'] = 0.10
    
    # Normalize weights to sum to 1
    weight_sum = sum(weights.values())
    weights = {k: v/weight_sum for k, v in weights.items()}
    
    # Calculate weighted rank
    weighted_rank = sum(rank_columns[col] * weights[col] for col in rank_columns)
    rank_columns['weighted_rank'] = weighted_rank
    
    # Add explainability score - penalize very high state counts
    state_penalty = np.log1p(df['n_states']) / np.log1p(np.max(df['n_states']))
    rank_columns['explainability_score'] = 1 - state_penalty
    
    # Calculate final score with explainability adjustment
    rank_columns['final_rank'] = weighted_rank * (1 + 0.2 * (1 - rank_columns['explainability_score']))
    
    return pd.DataFrame(rank_columns)

def plot_enhanced_comparison(models_df, group_name, output_dir):
    """
    Create enhanced visualizations for model comparison.
    
    Args:
        models_df: DataFrame with model metrics
        group_name: Group identifier
        output_dir: Directory to save outputs
    """
    # Create your original visualizations
    
    # Add a new figure for the extended metrics
    if any(col in models_df.columns for col in 
           ['inter_subject_synchrony', 'mixing_rate', 'mean_cov_condition_number']):
        
        plt.figure(figsize=(15, 10))
        
        # Plot inter-subject synchrony if available
        if 'inter_subject_synchrony' in models_df.columns:
            plt.subplot(2, 2, 1)
            plt.plot(models_df['n_states'], models_df['inter_subject_synchrony'], 'o-')
            plt.xlabel('Number of States')
            plt.ylabel('Inter-subject Synchrony')
            plt.title('Subject Synchronization by Model Complexity')
            
        # Plot mixing rate if available
        if 'mixing_rate' in models_df.columns:
            plt.subplot(2, 2, 2)
            plt.plot(models_df['n_states'], models_df['mixing_rate'], 'o-')
            plt.xlabel('Number of States')
            plt.ylabel('State Mixing Rate')
            plt.title('State Mixing by Model Complexity')
            
        # Plot stability metrics if available
        if 'mean_cov_condition_number' in models_df.columns:
            plt.subplot(2, 2, 3)
            plt.semilogy(models_df['n_states'], models_df['mean_cov_condition_number'], 'o-')
            plt.xlabel('Number of States')
            plt.ylabel('Covariance Condition Number (log scale)')
            plt.title('Numerical Stability by Model Complexity')
            
        # Plot consistency if available
        if 'likelihood_std' in models_df.columns:
            plt.subplot(2, 2, 4)
            plt.plot(models_df['n_states'], models_df['likelihood_std'], 'o-', 
                    label='Log-likelihood std')
            if 'reliability_std' in models_df.columns:
                plt.plot(models_df['n_states'], models_df['reliability_std'], 's-', 
                        label='Reliability std')
            plt.xlabel('Number of States')
            plt.ylabel('Standard Deviation')
            plt.title('Cross-validation Consistency')
            plt.legend()
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{group_name}_extended_metrics.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()

def neuroscientific_assessment(models_df):
    """
    Add a neuroscientific assessment of model plausibility.
    
    Args:
        models_df: DataFrame with model metrics
        
    Returns:
        DataFrame with added neuroscientific assessment
    """
    df = models_df.copy()
    
    # Calculate typical brain state durations (in seconds)
    if 'avg_duration' in df.columns:
        # Assuming TR value
        TR = 1.5  # seconds, adjust based on your acquisition
        df['avg_duration_seconds'] = df['avg_duration'] * TR
        
        # Neuroscientific plausibility scores
        # Higher score = more plausible
        
        # Duration plausibility (fMRI brain states typically last 5-30 seconds)
        duration_target = 15  # seconds
        df['duration_plausibility'] = 1 - np.abs(df['avg_duration_seconds'] - duration_target) / 30
        df['duration_plausibility'] = np.clip(df['duration_plausibility'], 0, 1)
        
        # State count plausibility (typical fMRI studies find 5-9 states)
        df['state_count_plausibility'] = 1 - np.abs(df['n_states'] - 7) / 10
        df['state_count_plausibility'] = np.clip(df['state_count_plausibility'], 0, 1)
        
        # Overall neuroscientific plausibility
        df['neuro_plausibility'] = 0.6 * df['duration_plausibility'] + 0.4 * df['state_count_plausibility']
    
    return df

# Usage example
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
        
        # Setup paths
    scratch_dir = os.getenv("SCRATCH_DIR")
    base_dir = os.path.join(scratch_dir, "output")
    # Compare multiple groups
    groups = ["affair", "paranoia", "combined"]
    group_results = {}
    
    for group in groups:
        print(f"\nAnalyzing {group}...")
        group_results[group] = compare_hmm_models_comprehensive(base_dir, group)
        
    # Compare optimal models between groups
    if len(groups) > 1:
        print("\nCross-group comparison:")
        for group in groups:
            print(f"{group}: Optimal states = {group_results[group]['recommended_model']}")