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
        
        # Extract the actual group from the model name
        actual_group = first_member['group']

        cluster_to_model_map[cluster_id] = {
            'n_states': min_states,
            'model': first_member['model'],
            'original_state_idx': first_member['original_state_idx'],
            'pattern_idx': first_member['pattern_idx'],
            'model_group': actual_group,
            'sequence_file': f"{output_dir}/04_{actual_group}_hmm_{min_states}states_ntw_native_trimmed/statistics/{actual_group}_state_sequences.npy"
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

            # Validate sequence dimensions
            if sequences.ndim != 2:
                raise ValueError(f"Expected 2D array, got {sequences.ndim}D array")
            if sequences.shape[1] == 0:
                raise ValueError(f"Sequence has no time points")
            
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
            if binary_sequences.size > 0:
                state_sequences[cluster_id] = binary_sequences
            else:
                logging.warning(f"Empty binary sequences for cluster {cluster_id}, skipping")
                continue
            
            mapping_info[cluster_id] = {
                'n_states': model_info['n_states'],
                'model': model_info['model'],
                'original_state_idx': original_state_idx,
                'pattern_idx': model_info['pattern_idx'],
                'model_group': model_info.get('model_group', 'combined'),
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

    # Remove empty strings from lists
    affair_subjects = [s for s in affair_subjects if s]
    paranoia_subjects = [s for s in paranoia_subjects if s]

    if not affair_subjects or not paranoia_subjects:
        logging.warning("Subject lists not found in environment variables, attempting to load from files")
        # Try loading from files as fallback
        try:
            base_dir = os.path.dirname(os.path.dirname(output_dir))
            with open(os.path.join(base_dir, 'data', 'affair_ids.txt'), 'r') as f:
                affair_subjects = [line.strip() for line in f if line.strip()]
            with open(os.path.join(base_dir, 'data', 'paranoia_ids.txt'), 'r') as f:
                paranoia_subjects = [line.strip() for line in f if line.strip()]
            logging.info(f"Loaded {len(affair_subjects)} affair subjects and {len(paranoia_subjects)} paranoia subjects from files")
        except Exception as e:
            logging.error(f"Failed to load subject lists: {e}")

    # Create subject mapping
    subject_groups = {}
    for subj in affair_subjects:
        subject_groups[subj] = 'affair'
    for subj in paranoia_subjects:
        subject_groups[subj] = 'paranoia'

    # Save state time series for each subject and cluster
    subject_data = defaultdict(dict)

    for cluster_id, sequences in state_sequences.items():
        # Get the model info to understand which group's sequences we're dealing with
        model_info = mapping_info.get(cluster_id, {})

        # Use the model_group field which indicates the actual group of the model
        model_group = model_info.get('model_group', 'combined')

        # Determine expected subjects based on the actual model group
        if model_group == 'affair':
            expected_subjects = affair_subjects
        elif model_group == 'paranoia':
            expected_subjects = paranoia_subjects
        elif model_group == 'balanced':
            # Balanced group uses a subset of 19 subjects (9 affair + 10 paranoia)
            # Load the actual subjects used in balanced models from saved indices
            try:
                # Find any balanced model directory to get the selected indices
                balanced_model_path = model_info.get('sequence_file', '').rsplit('/statistics/', 1)[0]
                indices_file = os.path.join(balanced_model_path, 'statistics', 'balanced_selected_indices.json')

                if os.path.exists(indices_file):
                    with open(indices_file, 'r') as f:
                        selected_data = json.load(f)
                        balanced_affair = selected_data['selected_subjects']['affair']
                        balanced_paranoia = selected_data['selected_subjects']['paranoia']
                        expected_subjects = balanced_affair + balanced_paranoia
                        logging.debug(f"Loaded balanced subjects from {indices_file}")
                else:
                    # Fallback: use indices if file not found
                    logging.warning(f"Balanced indices file not found at {indices_file}, using first 9 affair and 10 paranoia")
                    expected_subjects = affair_subjects[:9] + paranoia_subjects[:10]
            except Exception as e:
                logging.warning(f"Error loading balanced subjects: {e}, using fallback")
                expected_subjects = affair_subjects[:9] + paranoia_subjects[:10]
        else:  # combined
            expected_subjects = affair_subjects + paranoia_subjects

        logging.info(f"Processing cluster {cluster_id} from {model_group} model with {len(expected_subjects)} subjects, sequences shape: {sequences.shape}")

        # Validate sequence dimensions match expected subjects
        if len(expected_subjects) != sequences.shape[0]:
            logging.error(f"Mismatch for cluster {cluster_id}: expected {len(expected_subjects)} {model_group} subjects, got {sequences.shape[0]} sequences")
            continue

        # Save individual subject timeseries
        for i, subject_id in enumerate(expected_subjects):
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
        logging.debug(f"Processing subject: {subject_id}")
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
            logging.debug(f"Processing {group} {subject_id} in {cluster_key}")
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

def create_extracted_cluster_format(subject_data, cluster_id, mapping_info, output_dir, group, model_pattern):
    """
    Create data in the format expected by script 09 (07_extracted_cluster_data structure).
    This creates a separate directory for each cluster with individual subject data.

    Args:
        subject_data: Dictionary with subject-level time series (from create_subject_timeseries)
        cluster_id: The cluster ID to extract
        mapping_info: Dictionary with mapping information
        output_dir: Base output directory
        group: Group name (affair, paranoia, combined, balanced)
        model_pattern: Model pattern string (e.g., "2states", "3states")
    """
    import logging
    logger = logging.getLogger(__name__)

    # Create output directory for this specific cluster
    cluster_dir = output_dir / '07_map_cluster2stateseq' / f'th_{mapping_info[cluster_id].get("threshold", "060")}_{group}_{model_pattern}_cluster{cluster_id}'
    os.makedirs(cluster_dir, exist_ok=True)

    logger.info(f"Creating extracted cluster format for cluster {cluster_id} in {cluster_dir}")

    # Create simplified data structure for this specific cluster
    cluster_specific_data = {}

    for subject_id, subject_clusters in subject_data.items():
        # Check if this subject has data for the requested cluster
        cluster_key = f'cluster_{cluster_id}'
        if cluster_key in subject_clusters:
            cluster_data = subject_clusters[cluster_key]

            # Create entry in the format expected by script 09
            cluster_specific_data[subject_id] = {
                'cluster_id': cluster_id,
                'original_state_idx': cluster_data.get('original_state_idx', mapping_info[cluster_id]['original_state_idx']),
                'sorted_state_idx': 0,  # This is always 0 for binary sequences
                'timeseries': cluster_data['timeseries'],
                'group': cluster_data['group'],
                'mean_occupancy': cluster_data['mean_occupancy'],
                'model': cluster_data.get('model', mapping_info[cluster_id]['model'])
            }

    # Save the cluster-specific data
    if cluster_specific_data:
        pickle_path = cluster_dir / f'all_subjects_cluster_{cluster_id}_timeseries.pkl'
        with open(pickle_path, 'wb') as f:
            pickle.dump(cluster_specific_data, f)
        logger.info(f"Saved extracted cluster data for {len(cluster_specific_data)} subjects to {pickle_path}")

        # Also save as individual JSON files for inspection
        json_dir = cluster_dir / 'individual_subjects'
        os.makedirs(json_dir, exist_ok=True)

        for subject_id, data in cluster_specific_data.items():
            json_path = json_dir / f'{subject_id}_cluster_{cluster_id}.json'
            # Convert for JSON serialization
            json_data = data.copy()
            json_data['timeseries'] = data['timeseries']  # Already a list
            with open(json_path, 'w') as f:
                json.dump(json_data, f, indent=2)

        logger.info(f"Saved individual JSON files to {json_dir}")
    else:
        logger.warning(f"No data found for cluster {cluster_id}")

    return cluster_specific_data

def main():
    """Main function to map clusters to state sequences."""
    from dotenv import load_dotenv
    load_dotenv()

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Map clusters to state sequences')
    parser.add_argument('--thresholds', nargs='+', type=float,
                        default=[0.60, 0.65, 0.70, 0.75, 0.8, 0.85, 0.9],
                        help='Thresholds to process (default: 0.60-0.90)')
    parser.add_argument('--groups', nargs='+',
                        default=['affair', 'paranoia', 'combined', 'balanced'],
                        help='Groups to process (default: all)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: SCRATCH_DIR/output_RR)')
    parser.add_argument('--top-clusters', type=int, default=None,
                        help='Number of top clusters to process (default: 5 for th<=0.8, 3 for th>0.8)')
    args = parser.parse_args()

    # Create paths
    scratch_dir = os.getenv('SCRATCH_DIR')
    output_dir = args.output_dir or os.path.join(scratch_dir, 'output_RR')
    thresholds = args.thresholds
    groups = args.groups
    for threshold in thresholds:
        # Use user-specified top_clusters if provided, otherwise default based on threshold
        if args.top_clusters is not None:
            top_clusters = args.top_clusters
        elif threshold > 0.8:
            top_clusters = 3
        else:
            top_clusters = 5
        for group in groups:
            logging.info(f"Processing {group} group with threshold {threshold}")
            threshold_str = f"{threshold:.2f}".replace('.', '')
            threshold_dir = os.path.join(output_dir, '06_state_pattern_cluster', f'th_{threshold_str}')
            save_dir = os.path.join(output_dir, '07_map_cluster2stateseq', f'th_{threshold_str}', f'{group}')
            
            # Set up logging
            logger = setup_logging(save_dir)
            logger.info(f"Starting cluster to state sequence mapping for {group} group with threshold {threshold}")
            
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

                # Create extracted cluster format for each cluster (for script 09 compatibility)
                logger.info("Creating extracted cluster format for script 09 compatibility")
                for cluster_id in state_sequences.keys():
                    # Determine model pattern from the mapping info
                    n_states = mapping_info[cluster_id]['n_states']
                    model_pattern = f"{n_states}states"

                    # Add threshold to mapping_info if not present
                    mapping_info[cluster_id]['threshold'] = threshold_str

                    # Create the extracted format
                    create_extracted_cluster_format(
                        subject_data=subject_data,
                        cluster_id=cluster_id,
                        mapping_info=mapping_info,
                        output_dir=Path(output_dir),
                        group=group,
                        model_pattern=model_pattern
                    )

                logger.info(f"Complete. Results saved to {save_dir}")
                
            except Exception as e:
                logger.error(f"Error in processing: {str(e)}", exc_info=True)
                raise

if __name__ == "__main__":
    main()
