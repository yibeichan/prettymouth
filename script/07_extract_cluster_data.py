#!/usr/bin/env python
# extract_cluster_data.py

import os
import json
import numpy as np
import re
import argparse
import logging
from pathlib import Path
import pickle
from collections import defaultdict
from datetime import datetime


def setup_logging(output_dir):
    """Set up logging configuration."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'extract_cluster_data.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def extract_state_number(model_name):
    """Extract the number of states from a model name."""
    match = re.search(r'(\d+)states', model_name)
    if match:
        return int(match.group(1))
    return None


def load_cluster_info(threshold_dir):
    """
    Load the cluster to state mapping information for the given threshold.
    
    Args:
        threshold_dir: Directory with cluster information
        
    Returns:
        Dictionary with cluster mapping information
    """
    cluster_info_path = os.path.join(threshold_dir, 'cluster_info.json')
    
    if not os.path.exists(cluster_info_path):
        raise FileNotFoundError(f"Cluster info not found at {cluster_info_path}")
    
    with open(cluster_info_path, 'r') as f:
        return json.load(f)


def list_available_models(cluster_id, group, cluster_info):
    """
    List all available models for a specific cluster and group.
    
    Args:
        cluster_id: ID of the cluster to examine
        group: Group name to filter by
        cluster_info: Dictionary with cluster mapping information
        
    Returns:
        List of available model names
    """
    cluster_id_str = str(cluster_id)
    
    if cluster_id_str not in cluster_info:
        raise ValueError(f"Cluster {cluster_id} not found in cluster_info")
    
    cluster_data = cluster_info[cluster_id_str]
    
    # Filter members to only include the specified group
    available_models = [
        m['model'] for m in cluster_data['members'] 
        if m['group'] == group
    ]
    
    return sorted(set(available_models))  # Return unique sorted model names


def find_state_info_for_cluster(cluster_id, group, model_pattern, cluster_info, list_models=False):
    """
    Find the state information for a specific cluster, group, and model.
    
    Args:
        cluster_id: ID of the cluster to extract
        group: Group name ('affair', 'paranoia', 'combined', 'balanced')
        model_pattern: Pattern to match model name (e.g., '5states')
        cluster_info: Dictionary with cluster mapping information
        list_models: If True, just list available models and return None
        
    Returns:
        Dictionary with state information, or None if list_models is True
    """
    cluster_id_str = str(cluster_id)
    
    if cluster_id_str not in cluster_info:
        raise ValueError(f"Cluster {cluster_id} not found in cluster_info")
    
    cluster_data = cluster_info[cluster_id_str]
    
    # Extract the number of states from the model pattern
    n_states = extract_state_number(model_pattern)
    if not n_states:
        raise ValueError(f"Could not extract number of states from model pattern: {model_pattern}")
    
    # Format the model name to match the format in cluster_info
    model_name = f"{group} {n_states}states"
    
    # Filter members to only include the specified group
    group_members = [
        m for m in cluster_data['members'] 
        if m['group'] == group
    ]
    
    if not group_members:
        available_groups = list(set(m['group'] for m in cluster_data['members']))
        raise ValueError(f"No members found for cluster {cluster_id}, group {group}. Available groups: {available_groups}")
    
    if list_models:
        # Just return None since we'll print the list elsewhere
        return None
    
    # Filter to get matching model
    matching_members = [
        m for m in group_members 
        if m['model'] == model_name
    ]
    
    if not matching_members:
        available_models = list(set(m['model'] for m in group_members))
        raise ValueError(f"No matching members found for cluster {cluster_id}, group {group}, model {model_name}. Available models: {available_models}")
    
    # Since we're looking for a specific model, we should only have one match
    member = matching_members[0]
    
    return {
        'cluster_id': cluster_id,
        'group': group,
        'model': model_name,
        'model_pattern': model_pattern,
        'original_state_idx': member['original_state_idx'],
        'sorted_state_idx': member['sorted_state_idx'],
        'pattern_idx': member['pattern_idx'],
        'n_states': n_states
    }


def format_model_path(group, n_states):
    """
    Format the model path based on group and number of states.
    
    Args:
        group: Group name ('affair', 'paranoia', 'combined', 'balanced')
        n_states: Number of states in the model
        
    Returns:
        String with the model path format
    """
    return f"04_{group}_hmm_{n_states}states_ntw_native_trimmed"


def extract_state_sequence(state_info, output_dir):
    """
    Extract and save state sequences for a specific state index from a model.
    
    Args:
        state_info: Dictionary with state information
        output_dir: Directory to save output
        
    Returns:
        Binary state sequence array where 1 = target state, 0 = other states
    """
    # Construct the path to the state sequence file
    base_output_dir = os.getenv('SCRATCH_DIR')
    if not base_output_dir:
        base_output_dir = os.path.dirname(os.path.dirname(output_dir))  # Fallback if env var not set

    model_base_path = format_model_path(state_info['group'], state_info['n_states'])
    sequence_file = os.path.join(
        base_output_dir,
        'output_RR',
        model_base_path,
        'statistics',
        f"{state_info['group']}_state_sequences.npy"
    )
    
    # Check if the sequence file exists
    if not os.path.exists(sequence_file):
        raise FileNotFoundError(f"State sequence file not found at {sequence_file}")
    
    # Load state sequences
    try:
        sequences = np.load(sequence_file)
        logging.info(f"Loaded sequences with shape {sequences.shape} from {sequence_file}")
        
        # Get the original state index
        original_state_idx = state_info['original_state_idx']
        
        # Convert to int if possible
        try:
            if isinstance(original_state_idx, str):
                original_state_idx = int(original_state_idx)
        except ValueError:
            logging.warning(f"Could not convert state index to int: {original_state_idx}")
        
        # Create binary sequences where 1 = target state, 0 = other states
        binary_sequences = (sequences == original_state_idx).astype(int)
        
        # Save the binary sequences
        output_file = os.path.join(output_dir, f'cluster_{state_info["cluster_id"]}_sequences.npy')
        np.save(output_file, binary_sequences)
        logging.info(f"Saved binary sequences for cluster {state_info['cluster_id']} to {output_file}")
        
        # Calculate and save basic stats
        stats = {
            'cluster_id': state_info['cluster_id'],
            'group': state_info['group'],
            'model': state_info['model'],
            'original_state_idx': original_state_idx,
            'sorted_state_idx': state_info['sorted_state_idx'],
            'pattern_idx': state_info['pattern_idx'],
            'sequence_file': sequence_file,
            'sequence_shape': sequences.shape,
            'positive_samples': int(np.sum(binary_sequences)),
            'total_samples': binary_sequences.size,
            'occupancy': float(np.mean(binary_sequences))
        }
        
        with open(os.path.join(output_dir, f'cluster_{state_info["cluster_id"]}_stats.json'), 'w') as f:
            json.dump(stats, f, indent=2)
        
        return binary_sequences, stats
        
    except Exception as e:
        logging.error(f"Error processing sequence file {sequence_file}: {str(e)}")
        raise


def create_subject_timeseries(state_sequences, state_info, output_dir):
    """
    Create subject-level time series for the specified state to enable analysis.
    
    Args:
        state_sequences: Binary state sequences array
        state_info: Dictionary with state information
        output_dir: Directory to save output files
        
    Returns:
        Dictionary with subject-level time series data
    """
    # Create directory for subject timeseries
    timeseries_dir = os.path.join(output_dir, 'subject_timeseries')
    os.makedirs(timeseries_dir, exist_ok=True)
    
    # Get subject lists from environment variables
    affair_subjects = os.getenv('AFFAIR_SUBJECTS', '').split(",")
    paranoia_subjects = os.getenv('PARANOIA_SUBJECTS', '').split(",")
    
    if not affair_subjects or not paranoia_subjects:
        logging.warning("Subject lists not found in environment variables")
    
    # Create subject mapping
    subject_groups = {}
    for subj in affair_subjects:
        subject_groups[subj] = 'affair'
    for subj in paranoia_subjects:
        subject_groups[subj] = 'paranoia'
    
    # Determine which subjects to use based on group
    group = state_info['group']
    if group == 'affair':
        all_subjects = affair_subjects
    elif group == 'paranoia':
        all_subjects = paranoia_subjects
    else:  # combined or balanced
        all_subjects = affair_subjects + paranoia_subjects
    
    logging.info(f"Processing {len(all_subjects)} subjects from {group} group")
    
    # Check for mismatch in number of subjects
    if len(all_subjects) != state_sequences.shape[0]:
        logging.warning(f"Mismatch in subject count: {len(all_subjects)} subjects, but {state_sequences.shape[0]} sequences")
        # If there's a mismatch, ensure we don't exceed array bounds
        all_subjects = all_subjects[:state_sequences.shape[0]]
    
    # Save individual subject timeseries
    subject_data = {}
    cluster_id = state_info['cluster_id']
    original_state_idx = state_info['original_state_idx']
    
    for i, subject_id in enumerate(all_subjects):
        if i < state_sequences.shape[0]:
            # Create a dictionary with subject information and time series
            subject_data[subject_id] = {
                'cluster_id': cluster_id,
                'original_state_idx': original_state_idx,
                'sorted_state_idx': state_info['sorted_state_idx'],
                'timeseries': state_sequences[i].tolist(),
                'group': subject_groups.get(subject_id, 'unknown'),
                'mean_occupancy': float(np.mean(state_sequences[i])),
                'model': state_info['model']
            }
            
            # Format the subject ID correctly (ensure it has "sub-" prefix if needed)
            if not subject_id.startswith('sub-'):
                formatted_subject_id = f'sub-{subject_id}'
            else:
                formatted_subject_id = subject_id
            
            # Save individual subject file
            output_file = os.path.join(timeseries_dir, f'{formatted_subject_id}_cluster_{cluster_id}_timeseries.json')
            with open(output_file, 'w') as f:
                json.dump(subject_data[subject_id], f, indent=2)
            
            logging.info(f"Saved time series data for subject {subject_id} to {output_file}")
    
    # Also save as a combined pickle file for easier analysis
    combined_output = os.path.join(output_dir, f'all_subjects_cluster_{cluster_id}_timeseries.pkl')
    with open(combined_output, 'wb') as f:
        pickle.dump(subject_data, f)
    
    logging.info(f"Saved combined time series data to {combined_output}")
    
    return subject_data


def calculate_group_statistics(subject_data, state_info, output_dir):
    """
    Calculate and save group-level statistics.
    
    Args:
        subject_data: Dictionary with subject-level time series
        state_info: Dictionary with state information
        output_dir: Directory to save output files
        
    Returns:
        Dictionary with group-level statistics
    """
    # Initialize group stats
    group_stats = {
        'affair': {},
        'paranoia': {},
    }
    
    # Initialize stats collections for each group
    for group in group_stats:
        group_stats[group] = {
            'occupancy': [],
            'run_lengths': [],
            'interval_lengths': []
        }
    
    # Calculate statistics for each subject
    for subject_id, data in subject_data.items():
        group = data['group']
        
        # Skip unknown group
        if group == 'unknown':
            continue
        
        # Add occupancy
        group_stats[group]['occupancy'].append(data['mean_occupancy'])
        
        # Calculate run and interval lengths from time series
        timeseries = np.array(data['timeseries'])
        
        # Find runs (consecutive 1s)
        from itertools import groupby
        runs = [len(list(g)) for k, g in groupby(timeseries) if k == 1]
        if runs:
            group_stats[group]['run_lengths'].extend(runs)
        
        # Find intervals (consecutive 0s)
        intervals = [len(list(g)) for k, g in groupby(timeseries) if k == 0]
        if intervals:
            group_stats[group]['interval_lengths'].extend(intervals)
    
    # Compute summary statistics
    summary_stats = {}
    for group in group_stats:
        stats = group_stats[group]
        summary_stats[group] = {
            'mean_occupancy': np.mean(stats['occupancy']) if stats['occupancy'] else 0,
            'std_occupancy': np.std(stats['occupancy']) if stats['occupancy'] else 0,
            'mean_run_length': np.mean(stats['run_lengths']) if stats['run_lengths'] else 0,
            'std_run_length': np.std(stats['run_lengths']) if stats['run_lengths'] else 0,
            'mean_interval_length': np.mean(stats['interval_lengths']) if stats['interval_lengths'] else 0,
            'std_interval_length': np.std(stats['interval_lengths']) if stats['interval_lengths'] else 0,
            'n_subjects': len(stats['occupancy'])
        }
    
    # Add cluster and state information to the stats
    for group in summary_stats:
        summary_stats[group]['cluster_id'] = state_info['cluster_id']
        summary_stats[group]['original_state_idx'] = state_info['original_state_idx']
        summary_stats[group]['sorted_state_idx'] = state_info['sorted_state_idx']
        summary_stats[group]['model'] = state_info['model']
    
    # Save summary statistics
    cluster_id = state_info['cluster_id']
    with open(os.path.join(output_dir, f'cluster_{cluster_id}_group_statistics.json'), 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    logging.info(f"Saved group statistics to {os.path.join(output_dir, f'cluster_{cluster_id}_group_statistics.json')}")
    
    return summary_stats


def main():
    """Main function to extract state sequences for a specific cluster."""
    from dotenv import load_dotenv
    load_dotenv()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Extract state sequences for a specific cluster.')
    parser.add_argument('--cluster-id', type=int, required=True,
                        help='Cluster ID to extract')
    parser.add_argument('--group', type=str, required=True, 
                        choices=['affair', 'paranoia', 'combined', 'balanced'],
                        help='Group to process (affair, paranoia, combined, balanced)')
    parser.add_argument('--model-pattern', type=str,
                        help='Model pattern to match (e.g., "5states")')
    parser.add_argument('--threshold', type=float, required=True,
                        help='Threshold value (e.g., 0.75)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: SCRATCH_DIR/output_RR/07_extracted_cluster_data/TH_GROUP_MODEL_CLUSTER)')
    parser.add_argument('--list-models', action='store_true',
                        help='List available models for the specified cluster and group')
    
    args = parser.parse_args()
    
    # Get base directory from environment
    scratch_dir = os.getenv('SCRATCH_DIR')
    if not scratch_dir:
        raise ValueError("SCRATCH_DIR environment variable not set")
    
    # Set paths
    base_output_dir = os.path.join(scratch_dir, 'output_RR')
    
    # Format threshold string
    threshold_str = f"{args.threshold:.2f}".replace('.', '')
    
    # Determine the path to the cluster_info.json file
    threshold_dir = os.path.join(base_output_dir, '06_state_pattern_cluster', f'th_{threshold_str}')
    
    # Load cluster mapping information
    print(f"Loading cluster mapping for threshold {args.threshold} from {threshold_dir}")
    cluster_info = load_cluster_info(threshold_dir)
    
    # If list-models is specified, just print the available models and exit
    if args.list_models:
        models = list_available_models(args.cluster_id, args.group, cluster_info)
        print(f"\nAvailable models for cluster {args.cluster_id}, group {args.group}:")
        for model in models:
            print(f"  - {model}")
        return
    
    # Check if model-pattern is provided
    if not args.model_pattern:
        models = list_available_models(args.cluster_id, args.group, cluster_info)
        print(f"\nError: --model-pattern is required. Available models for cluster {args.cluster_id}, group {args.group}:")
        for model in models:
            print(f"  - {model}")
        print(f"\nPlease specify one of these models with --model-pattern. For example:")
        print(f"  --model-pattern {models[0].split()[1]}")
        return
    
    # Set output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(base_output_dir, '07_extracted_cluster_data', 
                                 f"th_{threshold_str}_{args.group}_{args.model_pattern}_cluster{args.cluster_id}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up logging
    logger = setup_logging(output_dir)
    logger.info(f"Starting extraction for cluster={args.cluster_id}, group={args.group}, "
                f"model_pattern={args.model_pattern}, threshold={args.threshold}")
    
    try:
        # Find state information for the specified cluster, group, and model
        logger.info(f"Finding state information for cluster {args.cluster_id}, group {args.group}, model_pattern {args.model_pattern}")
        state_info = find_state_info_for_cluster(args.cluster_id, args.group, args.model_pattern, cluster_info)
        
        # Log the state information
        logger.info(f"Found state information: {state_info}")
        
        # Extract state sequences
        logger.info(f"Extracting state sequences")
        state_sequences, stats = extract_state_sequence(state_info, output_dir)
        
        # Create subject-level time series
        logger.info("Creating subject-level time series")
        subject_data = create_subject_timeseries(
            state_sequences,
            state_info,
            output_dir
        )
        
        # Calculate group statistics
        logger.info("Calculating group statistics")
        group_stats = calculate_group_statistics(
            subject_data,
            state_info,
            output_dir
        )
        
        # Save configuration
        config = {
            'cluster_id': args.cluster_id,
            'group': args.group,
            'model_pattern': args.model_pattern,
            'threshold': args.threshold,
            'state_info': state_info,
            'extraction_time': datetime.now().isoformat()
        }
        
        with open(os.path.join(output_dir, 'extraction_config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Extraction complete. Results saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error in processing: {str(e)}", exc_info=True)
        
        # If we get a model-not-found error, try to list available models
        if "No matching members found" in str(e):
            models = list_available_models(args.cluster_id, args.group, cluster_info)
            print(f"\nAvailable models for cluster {args.cluster_id}, group {args.group}:")
            for model in models:
                print(f"  - {model}")
            print(f"\nPlease specify one of these models with --model-pattern. For example:")
            print(f"  --model-pattern {models[0].split()[1]}")
        
        raise


if __name__ == "__main__":
    main()