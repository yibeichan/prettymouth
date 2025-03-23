#!/usr/bin/env python
# 07_map_cluster2stateseq.py

import os
import json
import numpy as np
import re
import argparse
import logging
from pathlib import Path
import pickle
from collections import defaultdict

def setup_logging(output_dir):
    """Set up logging configuration."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, '07_map_cluster2stateseq.log')),
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
    """Load cluster information from the specified threshold directory."""
    cluster_info_path = os.path.join(threshold_dir, 'cluster_info.json')
    
    if not os.path.exists(cluster_info_path):
        raise FileNotFoundError(f"Cluster info not found at {cluster_info_path}")
    
    with open(cluster_info_path, 'r') as f:
        return json.load(f)

def find_first_model_for_clusters(cluster_info, output_dir, top_n_clusters=5, group_filter='combined'):
    """
    Find the first model where each of the top N clusters appears.
    
    Args:
        cluster_info: Dictionary with cluster information
        top_n_clusters: Number of top clusters to process
        group_filter: Only consider this group (default: 'combined')
        
    Returns:
        Dictionary mapping cluster IDs to their (model, state_idx) information
    """
    cluster_to_model_map = {}
    
    # Process only top N clusters by ID
    for cluster_id in range(1, top_n_clusters + 1):
        cluster_id_str = str(cluster_id)
        
        if cluster_id_str not in cluster_info:
            logging.warning(f"Cluster {cluster_id} not found in cluster_info")
            continue
            
        cluster_data = cluster_info[cluster_id_str]
        
        # Filter members to only include the specified group
        filtered_members = [m for m in cluster_data['members'] if m['group'] == group_filter]
        
        if not filtered_members:
            logging.warning(f"No '{group_filter}' group members found for cluster {cluster_id}")
            continue
            
        # Extract model information and sort by number of states
        model_info = []
        for member in filtered_members:
            model_name = member['model']
            n_states = extract_state_number(model_name)
            
            if n_states:
                model_info.append((n_states, member))
        
        # Sort by number of states
        model_info.sort(key=lambda x: x[0])
        
        if not model_info:
            logging.warning(f"No valid model information found for cluster {cluster_id}")
            continue
            
        # Get the earliest model (fewest states)
        min_states, first_member = model_info[0]
        
        cluster_to_model_map[cluster_id] = {
            'n_states': min_states,
            'model': first_member['model'],
            'original_state_idx': first_member['original_state_idx'],
            'pattern_idx': first_member['pattern_idx'],
            'sequence_file': f"{output_dir}/04_{group_filter}_hmm_{min_states}states_ntw_native_trimmed/statistics/{group_filter}_state_sequences.npy"
        }
    
    return cluster_to_model_map

def create_state_sequence_mapping(cluster_to_model_map, output_dir):
    """
    Create mapping from clusters to state sequences for further analysis.
    
    Args:
        cluster_to_model_map: Dictionary mapping clusters to model information
        base_dir: Base directory for data
        output_dir: Directory to save output files
        
    Returns:
        Dictionary with state sequence information for each cluster
    """
    state_sequences = {}
    mapping_info = {}
    
    for cluster_id, model_info in cluster_to_model_map.items():
        sequence_file = model_info['sequence_file']
        
        # Ensure sequence file exists
        if not os.path.exists(sequence_file):
            logging.error(f"Sequence file not found: {sequence_file}")
            continue
        
        # Load state sequences
        try:
            sequences = np.load(sequence_file)
            logging.info(f"Loaded sequences with shape {sequences.shape} from {sequence_file}")
            
            # Determine which state index corresponds to the cluster
            original_state_idx = model_info['original_state_idx']
            
            # Convert to int if possible
            try:
                if isinstance(original_state_idx, str):
                    original_state_idx = int(original_state_idx)
            except ValueError:
                logging.warning(f"Could not convert state index to int: {original_state_idx}")
            
            # Create binary sequences where 1 = target state, 0 = other states
            binary_sequences = (sequences == original_state_idx).astype(int)
            
            # Store the sequences and mapping information
            state_sequences[cluster_id] = binary_sequences
            
            mapping_info[cluster_id] = {
                'n_states': model_info['n_states'],
                'model': model_info['model'],
                'original_state_idx': original_state_idx,
                'pattern_idx': model_info['pattern_idx'],
                'sequence_file': model_info['sequence_file'],
                'sequence_shape': sequences.shape,
                'positive_samples': int(np.sum(binary_sequences)),
                'total_samples': binary_sequences.size,
                'occupancy': float(np.mean(binary_sequences))
            }
            
        except Exception as e:
            logging.error(f"Error processing sequence file {sequence_file}: {str(e)}")
    
    # Save mapping information
    with open(os.path.join(output_dir, 'cluster_state_mapping.json'), 'w') as f:
        json.dump(mapping_info, f, indent=2)
    
    # Save binary sequences for each cluster
    for cluster_id, sequences in state_sequences.items():
        output_file = os.path.join(output_dir, f'cluster_{cluster_id}_sequences.npy')
        np.save(output_file, sequences)
        logging.info(f"Saved binary sequences for cluster {cluster_id} to {output_file}")
    
    return state_sequences, mapping_info

def create_subject_timeseries(state_sequences, group, mapping_info, output_dir):
    """
    Create subject-level time series for each cluster to enable GLMM analysis.
    
    Args:
        state_sequences: Dictionary of state sequences by cluster
        mapping_info: Dictionary with mapping information
        output_dir: Directory to save output files
    """
    # Create directories for each analysis type
    timeseries_dir = os.path.join(output_dir, 'subject_timeseries')
    os.makedirs(timeseries_dir, exist_ok=True)
    
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
    
    # Save state time series for each subject and cluster
    subject_data = defaultdict(dict)
    
    for cluster_id, sequences in state_sequences.items():
        # Ensure all_subjects is a list of unique subject IDs
        if group == 'affair':
            all_subjects = affair_subjects
        elif group == 'paranoia':
            all_subjects = paranoia_subjects
        else:
            all_subjects = affair_subjects + paranoia_subjects
        print(group, all_subjects)
        
        if len(all_subjects) != sequences.shape[0]:
            logging.warning(f"Mismatch in subject count: {len(all_subjects)} subjects, but {sequences.shape[0]} sequences")
            # If there's a mismatch, ensure we don't exceed array bounds
            all_subjects = all_subjects[:sequences.shape[0]]
        
        # Save individual subject timeseries
        for i, subject_id in enumerate(all_subjects):
            if i < sequences.shape[0]:
                # Create a dictionary with subject information and time series
                subject_data[subject_id][f'cluster_{cluster_id}'] = {
                    'timeseries': sequences[i].tolist(),
                    'group': subject_groups.get(subject_id, 'unknown'),
                    'mean_occupancy': float(np.mean(sequences[i])),
                    'cluster_id': int(cluster_id),
                    'original_state_idx': mapping_info[cluster_id]['original_state_idx'],
                    'model': mapping_info[cluster_id]['model']
                }
    
    # Save each subject's data as an individual file with correct filename format
    for subject_id, data in subject_data.items():
        print(subject_id)
        # Format the subject ID correctly (ensure it has "sub-" prefix if needed)
        if not subject_id.startswith('sub-'):
            formatted_subject_id = f'sub-{subject_id}'
        else:
            formatted_subject_id = subject_id
        
        # Create a proper filename with the subject ID first
        output_file = os.path.join(timeseries_dir, f'{formatted_subject_id}_cluster_timeseries.json')
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        logging.info(f"Saved time series data for subject {subject_id} to {output_file}")
    
    # Also save as a combined pickle file for easier analysis
    with open(os.path.join(output_dir, 'all_subject_timeseries.pkl'), 'wb') as f:
        pickle.dump(subject_data, f)
    
    logging.info(f"Saved combined time series data to {os.path.join(output_dir, 'all_subject_timeseries.pkl')}")
    
    return subject_data

def calculate_group_statistics(subject_data, group, output_dir):
    """
    Calculate and save group-level statistics for each cluster.
    
    Args:
        subject_data: Dictionary with subject-level time series
        output_dir: Directory to save output files
    """
    # Initialize group stats
    group_stats = {
        'affair': defaultdict(dict),
        'paranoia': defaultdict(dict),
    }
    
    # Calculate statistics for each group and cluster
    for subject_id, clusters in subject_data.items():
        for cluster_key, data in clusters.items():
            group = data['group']
            print(f"{group} {subject_id} in {cluster_key}")
            cluster_id = data['cluster_id']
            
            # Skip unknown group
            if group == 'unknown':
                continue
                
            # Initialize lists if needed
            if 'occupancy' not in group_stats[group][cluster_id]:
                group_stats[group][cluster_id] = {
                    'occupancy': [],
                    'run_lengths': [],
                    'interval_lengths': []
                }
            
            # Add occupancy
            group_stats[group][cluster_id]['occupancy'].append(data['mean_occupancy'])
            
            # Calculate run and interval lengths from time series
            timeseries = np.array(data['timeseries'])
            
            # Find runs (consecutive 1s)
            from itertools import groupby
            runs = [len(list(g)) for k, g in groupby(timeseries) if k == 1]
            if runs:
                group_stats[group][cluster_id]['run_lengths'].extend(runs)
            
            # Find intervals (consecutive 0s)
            intervals = [len(list(g)) for k, g in groupby(timeseries) if k == 0]
            if intervals:
                group_stats[group][cluster_id]['interval_lengths'].extend(intervals)
    
    # Compute summary statistics
    summary_stats = {}
    for group in group_stats:
        summary_stats[group] = {}
        
        for cluster_id, stats in group_stats[group].items():
            summary_stats[group][cluster_id] = {
                'mean_occupancy': np.mean(stats['occupancy']),
                'std_occupancy': np.std(stats['occupancy']),
                'mean_run_length': np.mean(stats['run_lengths']) if stats['run_lengths'] else 0,
                'std_run_length': np.std(stats['run_lengths']) if stats['run_lengths'] else 0,
                'mean_interval_length': np.mean(stats['interval_lengths']) if stats['interval_lengths'] else 0,
                'std_interval_length': np.std(stats['interval_lengths']) if stats['interval_lengths'] else 0,
                'n_subjects': len(stats['occupancy'])
            }
    
    # Save summary statistics
    with open(os.path.join(output_dir, 'group_statistics.json'), 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    logging.info(f"Saved group statistics to {os.path.join(output_dir, 'group_statistics.json')}")
    
    return summary_stats

def main():
    """Main function to map clusters to state sequences."""
    from dotenv import load_dotenv
    load_dotenv()
    # Parse command line argume
    
    # Create paths
    scratch_dir = os.getenv('SCRATCH_DIR')
    output_dir = os.path.join(scratch_dir, 'output')
    thresholds = [0.60, 0.65, 0.70, 0.75, 0.8, 0.85, 0.9]
    groups = ['affair', 'paranoia', 'combined']
    for threshold in thresholds:
        if threshold > 0.8:
            top_clusters = 3
        else:
            top_clusters = 5
        for group in groups:
            print("processing", group, "with threshold", threshold)
            threshold_str = f"{threshold:.2f}".replace('.', '')
            threshold_dir = os.path.join(output_dir, '06_state_pattern_cluster', f'th_{threshold_str}')
            save_dir = os.path.join(output_dir, '07_cluster_state_mapping', f'th_{threshold_str}', f'{group}')
            
            # Set up logging
            logger = setup_logging(save_dir)
            logger.info(f"Starting cluster to state sequence mapping for threshold {threshold}")
            
            try:
                # Load cluster information
                logger.info(f"Loading cluster info from {threshold_dir}")
                cluster_info = load_cluster_info(threshold_dir)
                
                # Find first model for each cluster
                logger.info(f"Finding first models for top {top_clusters} clusters in group '{group}'")
                cluster_to_model_map = find_first_model_for_clusters(
                    cluster_info, 
                    output_dir,
                    top_n_clusters=top_clusters,
                    group_filter=group
                )
                
                # Save the mapping information
                with open(os.path.join(save_dir, 'cluster_model_mapping.json'), 'w') as f:
                    json.dump(cluster_to_model_map, f, indent=2)
                
                # Create state sequence mapping
                logger.info("Creating state sequence mapping")
                state_sequences, mapping_info = create_state_sequence_mapping(
                    cluster_to_model_map, 
                    save_dir
                )
                
                # Create subject-level time series
                logger.info("Creating subject-level time series")
                subject_data = create_subject_timeseries(
                    state_sequences,
                    group,
                    mapping_info,
                    save_dir
                )
                
                # Calculate group statistics
                logger.info("Calculating group statistics")
                group_stats = calculate_group_statistics(
                    subject_data,
                    group,
                    save_dir
                )
                
                logger.info(f"Complete. Results saved to {save_dir}")
                
            except Exception as e:
                logger.error(f"Error in processing: {str(e)}", exc_info=True)
                raise

if __name__ == "__main__":
    main()
