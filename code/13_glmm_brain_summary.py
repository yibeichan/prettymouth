import numpy as np
import pandas as pd
import re
import json
import matplotlib.pyplot as plt
import seaborn as sns

def get_brain_content_data(state_affair, state_paranoia):
    # Add debugging
    print(f"Getting brain content data for state_affair={state_affair}, state_paranoia={state_paranoia}")
    
    # Use Path.glob() method instead of glob module
    pattern = f'hierarchical_analysis_results_*.npz'
    search_path = Path(OUTPUT_DIR) / '11_brain_content_analysis' / f'state_affair_{state_affair}_state_paranoia_{state_paranoia}'
    
    print(f"Searching in: {search_path}")
    glob_results = list(search_path.glob(pattern))
    print(f"Found {len(glob_results)} matching files")
    
    if not glob_results:
        raise FileNotFoundError(f"No files matching {pattern} found in {search_path}")
    
    brain_content_path = glob_results[0]
    print(f"Loading data from: {brain_content_path}")
    
    brain_content_data = np.load(brain_content_path, allow_pickle=True)
    print(f"Loaded data with keys: {list(brain_content_data.keys())}")
    
    meta_data = brain_content_data['metadata']
    main_analysis = brain_content_data['main_analysis']
    cross_validation = brain_content_data['cross_validation']
    
    # Check what's in main_analysis
    if isinstance(main_analysis, dict):
        print(f"Main analysis keys: {list(main_analysis.keys())}")
        if 'feature_data' in main_analysis:
            print(f"Feature data keys: {list(main_analysis['feature_data'].keys())}")
    elif isinstance(main_analysis, np.ndarray) and main_analysis.dtype == np.dtype('O'):
        print("Main analysis is an object array, attempting to convert to dict")
        main_analysis = main_analysis.item() if main_analysis.size == 1 else dict(main_analysis)
    else:
        print(f"Main analysis type: {type(main_analysis)}")
    
    return meta_data, main_analysis, cross_validation

def parse_glmm_results(content):
    """
    Parse the GLMM results from the text file.
    """
    # If it's already a dictionary, use it directly
    if isinstance(content, dict):
        print("Content is already a dictionary")
        # Check if the dictionary has the expected structure
        if 'feature_data' not in content:
            print(f"Dictionary keys: {list(content.keys())}")
            
            # Create a properly structured dictionary
            results = {
                'feature_data': {},
                'credible_features': [],  # Changed from 'significant_features' to 'credible_features'
                'group_stats': content.get('group_stats', {})
            }
            
            # The key issue: we need to extract feature data from 'feature_results'
            if 'feature_results' in content:
                print("Found 'feature_results' key, extracting feature data")
                for feature, data in content['feature_results'].items():
                    results['feature_data'][feature] = {}
                    
                    # Extract coefficients
                    if 'coefficients' in data:
                        if 'group' in data['coefficients']:
                            results['feature_data'][feature]['group_coef'] = data['coefficients']['group']
                            
                            # Calculate odds ratio from coefficient (exp(coef))
                            if 'group' in data['coefficients']:
                                odds_ratio = np.exp(data['coefficients']['group'])
                                results['feature_data'][feature]['odds_ratio'] = {
                                    'value': odds_ratio
                                }
                                
                        if 'interaction' in data['coefficients']:
                            results['feature_data'][feature]['interaction_coef'] = data['coefficients']['interaction']
                    
                    # Extract posterior probabilities
                    if 'posterior_prob' in data:
                        if 'group' in data['posterior_prob']:
                            results['feature_data'][feature]['group_prob'] = data['posterior_prob']['group']
                            # Determine if feature has credible effect (posterior probability > threshold)
                            if data['posterior_prob']['group'] > 0.95 or data['posterior_prob']['group'] < 0.05:
                                results['credible_features'].append(feature)
                        if 'interaction' in data['posterior_prob']:
                            results['feature_data'][feature]['interaction_prob'] = data['posterior_prob']['interaction']
                    
                    # Extract HDI
                    if 'hdi' in data and 'group' in data['hdi']:
                        results['feature_data'][feature]['group_hdi'] = data['hdi']['group']
                        
                        # Calculate odds ratio HDI from coefficient HDI
                        if 'group' in data['hdi']:
                            lower_or = np.exp(data['hdi']['group']['lower'])
                            upper_or = np.exp(data['hdi']['group']['upper'])
                            
                            if 'odds_ratio' not in results['feature_data'][feature]:
                                results['feature_data'][feature]['odds_ratio'] = {}
                                
                            results['feature_data'][feature]['odds_ratio']['lower'] = lower_or
                            results['feature_data'][feature]['odds_ratio']['upper'] = upper_or
                    
                    # Extract effect sizes
                    if 'effect_sizes' in content:
                        print("Found 'effect_sizes' key, extracting standardized effects")
                        for feature, effect_data in content['effect_sizes'].items():
                            if feature not in results['feature_data']:
                                results['feature_data'][feature] = {}
                            
                            if 'standardized_effect' in effect_data:
                                results['feature_data'][feature]['std_effect'] = effect_data['standardized_effect']
            
            # Extract practical significance
            if 'practical_significance' in content:
                print("Found 'practical_significance' key")
                for feature, sig_data in content['practical_significance'].items():
                    if feature not in results['feature_data']:
                        results['feature_data'][feature] = {}
                    
                    if 'group' in sig_data and 'interpretation' in sig_data['group']:
                        results['feature_data'][feature]['practical_significance'] = sig_data['group']['interpretation']
            
            # Extract credible features from bayesian_multiple_comparison if not already found
            if not results['credible_features'] and 'bayesian_multiple_comparison' in content:
                if 'credible_features' in content['bayesian_multiple_comparison']:
                    results['credible_features'] = content['bayesian_multiple_comparison']['credible_features']
                elif 'significant_features' in content['bayesian_multiple_comparison']:
                    # For backward compatibility
                    results['credible_features'] = content['bayesian_multiple_comparison']['significant_features']
            
            print(f"Extracted data for {len(results['feature_data'])} features")
            print(f"Credible features: {results['credible_features']}")
            
            return results
        return content
    
    # Read file content if a string (assumed to be a file path)
    if isinstance(content, str) and Path(content).exists():
        with open(content, 'rb') as f:
            content_bytes = f.read()
    else:
        # Assume it's already the content (string or bytes)
        content_bytes = content if isinstance(content, bytes) else str(content).encode('utf-8')
    
    # Try to decode the content with different encodings
    encodings = ['utf-8', 'latin-1', 'cp1252']
    content = None
    
    for encoding in encodings:
        try:
            content = content_bytes.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    
    if content is None:
        raise ValueError("Could not decode the file content with any of the attempted encodings")
    
    # Define the features we're interested in
    features = [
        'lee_girl_together', 'has_verb', 'lee_speaking', 'girl_speaking', 
        'arthur_speaking', 'has_adj', 'has_adv', 'has_noun', 
        'lee_girl_verb', 'arthur_adj'
    ]
    
    # Initialize results dictionary with updated terminology
    results = {
        'feature_data': {},
        'credible_features': [],  # Changed from 'significant_features'
        'group_stats': {}
    }
    
    # Extract credible features
    sig_features_match = re.search(r"'(significant|credible)_features': \[([^\]]+)\]", content)
    if sig_features_match:
        sig_features_text = sig_features_match.group(2)
        sig_features = re.findall(r"'([^']+)'", sig_features_text)
        results['credible_features'] = sig_features
    
    # Extract group stats
    # Occupancy
    occupancy_match = re.search(r"'occupancy': {'affair': ([0-9.]+), 'paranoia': ([0-9.]+)}", content)
    if occupancy_match:
        results['group_stats']['occupancy'] = {
            'affair': float(occupancy_match.group(1)),
            'paranoia': float(occupancy_match.group(2))
        }
    
    # Transitions
    transitions_match = re.search(r"'transitions': {'affair': {'entry_rate': ([0-9.]+), 'exit_rate': ([0-9.]+)}, 'paranoia': {'entry_rate': ([0-9.]+), 'exit_rate': ([0-9.]+)}}", content)
    if transitions_match:
        results['group_stats']['transitions'] = {
            'affair': {
                'entry_rate': float(transitions_match.group(1)),
                'exit_rate': float(transitions_match.group(2))
            },
            'paranoia': {
                'entry_rate': float(transitions_match.group(3)),
                'exit_rate': float(transitions_match.group(4))
            }
        }
    
    # Extract data for each feature
    for feature in features:
        results['feature_data'][feature] = {}
        
        # Extract coefficients
        coef_pattern = rf"'{feature}': \{{[^}}]*'coefficients': \{{([^}}]+)\}}"
        coef_match = re.search(coef_pattern, content, re.DOTALL)
        if coef_match:
            coef_text = coef_match.group(1)
            # Extract group coefficient
            group_coef_match = re.search(r"'group': ([0-9.-]+)", coef_text)
            if group_coef_match:
                results['feature_data'][feature]['group_coef'] = float(group_coef_match.group(1))
            # Extract interaction coefficient
            interaction_coef_match = re.search(r"'interaction': ([0-9.-]+)", coef_text)
            if interaction_coef_match:
                results['feature_data'][feature]['interaction_coef'] = float(interaction_coef_match.group(1))
        
        # Extract posterior probabilities
        prob_pattern = rf"'{feature}': \{{[^}}]*'posterior_prob': \{{([^}}]+)\}}"
        prob_match = re.search(prob_pattern, content, re.DOTALL)
        if prob_match:
            prob_text = prob_match.group(1)
            # Extract group probability
            group_prob_match = re.search(r"'group': ([0-9.]+)", prob_text)
            if group_prob_match:
                results['feature_data'][feature]['group_prob'] = float(group_prob_match.group(1))
            # Extract interaction probability
            interaction_prob_match = re.search(r"'interaction': ([0-9.]+)", prob_text)
            if interaction_prob_match:
                results['feature_data'][feature]['interaction_prob'] = float(interaction_prob_match.group(1))
        
        # Extract HDI (highest density interval)
        hdi_pattern = rf"'{feature}': \{{[^}}]*'hdi': \{{[^}}]*'group': \{{'lower': ([0-9.-]+), 'upper': ([0-9.-]+)\}}"
        hdi_match = re.search(hdi_pattern, content, re.DOTALL)
        if hdi_match:
            results['feature_data'][feature]['group_hdi'] = {
                'lower': float(hdi_match.group(1)),
                'upper': float(hdi_match.group(2))
            }
        
        # Extract odds ratio
        or_pattern = rf"'{feature}': \{{[^}}]*'odds_ratio': \{{'odds_ratio': ([0-9.]+), 'lower': ([0-9.]+), 'upper': ([0-9.]+)\}}"
        or_match = re.search(or_pattern, content, re.DOTALL)
        if or_match:
            results['feature_data'][feature]['odds_ratio'] = {
                'value': float(or_match.group(1)),
                'lower': float(or_match.group(2)),
                'upper': float(or_match.group(3))
            }
        
        # Extract standardized effect from effect_sizes
        effect_pattern = rf"'effect_sizes': \{{[^}}]*'{feature}': \{{'standardized_effect': ([0-9.]+)"
        effect_match = re.search(effect_pattern, content, re.DOTALL)
        if effect_match:
            results['feature_data'][feature]['std_effect'] = float(effect_match.group(1))
        
        # Extract practical significance
        practical_pattern = rf"'practical_significance': \{{[^}}]*'{feature}': \{{'group': \{{[^}}]*'interpretation': '([^']+)'"
        practical_match = re.search(practical_pattern, content, re.DOTALL)
        if practical_match:
            results['feature_data'][feature]['practical_significance'] = practical_match.group(1)
        
        # If odds_ratio is missing, try to extract it with a more flexible pattern
        if 'odds_ratio' not in results['feature_data'][feature]:
            print(f"Trying alternative extraction for odds_ratio in {feature}")
            alt_or_pattern = rf"{feature}.*?odds_ratio.*?([0-9.]+).*?lower.*?([0-9.]+).*?upper.*?([0-9.]+)"
            alt_or_match = re.search(alt_or_pattern, content, re.DOTALL | re.IGNORECASE)
            if alt_or_match:
                results['feature_data'][feature]['odds_ratio'] = {
                    'value': float(alt_or_match.group(1)),
                    'lower': float(alt_or_match.group(2)),
                    'upper': float(alt_or_match.group(3))
                }
                print(f"  Found odds_ratio: {results['feature_data'][feature]['odds_ratio']}")
        
        # If std_effect is missing, try to extract it with a more flexible pattern
        if 'std_effect' not in results['feature_data'][feature]:
            print(f"Trying alternative extraction for std_effect in {feature}")
            alt_effect_pattern = rf"{feature}.*?standardized_effect.*?([0-9.]+)"
            alt_effect_match = re.search(alt_effect_pattern, content, re.DOTALL | re.IGNORECASE)
            if alt_effect_match:
                results['feature_data'][feature]['std_effect'] = float(alt_effect_match.group(1))
                print(f"  Found std_effect: {results['feature_data'][feature]['std_effect']}")
    
    # For each feature, calculate odds ratio from coefficient if not already present
    for feature in features:
        if 'group_coef' in results['feature_data'][feature] and 'odds_ratio' not in results['feature_data'][feature]:
            coef = results['feature_data'][feature]['group_coef']
            odds_ratio = np.exp(coef)
            results['feature_data'][feature]['odds_ratio'] = {
                'value': odds_ratio
            }
            
            # If HDI is available, calculate odds ratio HDI
            if 'group_hdi' in results['feature_data'][feature]:
                hdi = results['feature_data'][feature]['group_hdi']
                results['feature_data'][feature]['odds_ratio']['lower'] = np.exp(hdi['lower'])
                results['feature_data'][feature]['odds_ratio']['upper'] = np.exp(hdi['upper'])
    
    # Add debugging at the end to check what was extracted
    print(f"Extracted {len(results['credible_features'])} credible features")
    print(f"Features with odds ratio data: {[f for f in results['feature_data'] if 'odds_ratio' in results['feature_data'][f]]}")
    print(f"Features with std_effect data: {[f for f in results['feature_data'] if 'std_effect' in results['feature_data'][f]]}")
    
    return results

def summarize_for_paper(results):
    """
    Generate summary statistics and formatted text for paper.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing parsed GLMM results
    
    Returns:
    --------
    dict
        Dictionary containing summary statistics and formatted text
    """
    # Add debugging at the beginning
    print("Starting summarize_for_paper")
    print(f"Number of features: {len(results['feature_data'])}")
    print(f"Credible features: {results['credible_features']}")
    
    # Check what data is available for each feature
    for feature, data in results['feature_data'].items():
        print(f"\nFeature: {feature}")
        print(f"  Has odds_ratio: {'odds_ratio' in data}")
        if 'odds_ratio' in data:
            print(f"  Odds ratio value: {data['odds_ratio'].get('value', 'missing')}")
        print(f"  Has std_effect: {'std_effect' in data}")
        if 'std_effect' in data:
            print(f"  Std effect value: {data['std_effect']}")
    
    summary = {
        'main_text': '',
        'supplement': '',
        'tables': {},
        'figures': {}
    }
    
    # Extract data for calculation
    feature_data = results['feature_data']
    credible_features = results['credible_features']
    non_credible_features = [f for f in feature_data.keys() if f not in credible_features]
    
    # Calculate average odds ratio and effect sizes
    credible_ors = [feature_data[f]['odds_ratio']['value'] for f in credible_features if 'odds_ratio' in feature_data[f]]
    non_credible_ors = [feature_data[f]['odds_ratio']['value'] for f in non_credible_features if 'odds_ratio' in feature_data[f]]
    
    # Print what we found
    print(f"Credible features with odds ratios: {len(credible_ors)}/{len(credible_features)}")
    print(f"Non-credible features with odds ratios: {len(non_credible_ors)}/{len(non_credible_features)}")
    
    avg_credible_odd_ratio = np.mean(credible_ors) if credible_ors else np.nan
    avg_non_credible_odd_ratio = np.mean(non_credible_ors) if non_credible_ors else np.nan
    
    credible_effects = [feature_data[f]['std_effect'] for f in credible_features if 'std_effect' in feature_data[f]]
    non_credible_effects = [feature_data[f]['std_effect'] for f in non_credible_features if 'std_effect' in feature_data[f]]
    
    print(f"Credible features with std effects: {len(credible_effects)}/{len(credible_features)}")
    print(f"Non-credible features with std effects: {len(non_credible_effects)}/{len(non_credible_features)}")
    
    avg_credible_effect = np.mean(credible_effects) if credible_effects else np.nan
    avg_non_credible_effect = np.mean(non_credible_effects) if non_credible_effects else np.nan
    
    # Calculate range of odds ratios and effect sizes
    all_ors = credible_ors + non_credible_ors
    odd_ratio_range = (min(all_ors), max(all_ors)) if all_ors else (np.nan, np.nan)
    
    all_effects = credible_effects + non_credible_effects
    effect_range = (min(all_effects), max(all_effects)) if all_effects else (np.nan, np.nan)
    
    # Create a dataframe for all features
    feature_rows = []
    for feature in feature_data:
        data = feature_data[feature]
        row = {
            'Feature': feature,
            'Group Coefficient': data.get('group_coef', np.nan),
            'Posterior Probability': data.get('group_prob', np.nan),
            'HDI Lower': data.get('group_hdi', {}).get('lower', np.nan),
            'HDI Upper': data.get('group_hdi', {}).get('upper', np.nan),
            'Odds Ratio': data.get('odds_ratio', {}).get('value', np.nan),
            'OR Lower': data.get('odds_ratio', {}).get('lower', np.nan),
            'OR Upper': data.get('odds_ratio', {}).get('upper', np.nan),
            'Standardized Effect': data.get('std_effect', np.nan),
            'Practical Significance': data.get('practical_significance', ''),
            'Credible Effect': feature in credible_features,  # Changed from 'Significant'
            'Interaction Coefficient': data.get('interaction_coef', np.nan),
            'Interaction Probability': data.get('interaction_prob', np.nan)
        }
        feature_rows.append(row)
    
    df = pd.DataFrame(feature_rows)
    
    # Sort by effect size
    df_sorted = df.sort_values('Standardized Effect', ascending=False)
    
    # Create summary tables
    summary['tables']['all_features'] = df
    summary['tables']['sorted_features'] = df_sorted
    
    # Create tables for credible and non-credible features
    summary['tables']['credible'] = df[df['Credible Effect']]
    summary['tables']['non_credible'] = df[~df['Credible Effect']]
    
    # Generate main text summary
    main_text = []
    main_text.append(f"### Bayesian GLMM Results Summary")
    main_text.append("\nBayesian GLMM analysis revealed credible group effects in {n_credible} out of {n_total} neural state features.".format(
        n_credible=len(credible_features),
        n_total=len(feature_data)
    ))
    
    # Add information about credible features
    if credible_features:
        main_text.append("The features with credible effects were: {features}.".format(
            features=", ".join([f"'{f}'" for f in credible_features])
        ))
        main_text.append("For these features, the mean standardized effect size was {effect:.2f} (odds ratio: {odd_ratio:.2f}).".format(
            effect=avg_credible_effect,
            odd_ratio=avg_credible_odd_ratio
        ))
    
    # Add information about top features
    top_features = df_sorted.head(3)
    main_text.append("\nThe three features with the largest effect sizes were:")
    for _, row in top_features.iterrows():
        credible_status = "credible" if row['Credible Effect'] else "non-credible"
        feature_name = row['Feature']
        
        # Get interaction data if available
        interaction_coef = feature_data[feature_name].get('interaction_coef', np.nan)
        interaction_prob = feature_data[feature_name].get('interaction_prob', np.nan)
        
        # Determine effect direction description
        group_coef = row['Group Coefficient']
        effect_direction = "higher in paranoia condition" if group_coef > 0 else "higher in affair condition"
        
        # Determine if there's a significant interaction that modifies the interpretation
        interaction_significant = (interaction_prob > 0.95 or interaction_prob < 0.05) if not np.isnan(interaction_prob) else False
        
        if interaction_significant:
            # If group is positive and interaction is negative, the effect is stronger in affair
            # If group is positive and interaction is positive, the effect is even stronger in paranoia
            if group_coef > 0 and interaction_coef < 0:
                interaction_note = f", though the effect is reduced in paranoia compared to affair (interaction={interaction_coef:.3f})"
            elif group_coef > 0 and interaction_coef > 0:
                interaction_note = f", and the effect is enhanced in paranoia compared to affair (interaction={interaction_coef:.3f})"
            elif group_coef < 0 and interaction_coef < 0:
                interaction_note = f", and the effect is enhanced in affair compared to paranoia (interaction={interaction_coef:.3f})"
            elif group_coef < 0 and interaction_coef > 0:
                interaction_note = f", though the effect is reduced in affair compared to paranoia (interaction={interaction_coef:.3f})"
            else:
                interaction_note = ""
        else:
            interaction_note = ""
        
        main_text.append("- '{feature}': coefficient = {coef:.3f} [{lower:.3f}, {upper:.3f}], OR = {odd_ratio:.2f}, standardized effect = {effect:.2f}, {status} effect. Neural states for this feature were {direction}{interaction}.".format(
            feature=feature_name,
            coef=row['Group Coefficient'],
            lower=row['HDI Lower'],
            upper=row['HDI Upper'],
            odd_ratio=row['Odds Ratio'],
            effect=row['Standardized Effect'],
            status=credible_status,
            direction=effect_direction,
            interaction=interaction_note
        ))
    
    # Add information about group statistics
    if 'group_stats' in results and 'occupancy' in results['group_stats']:
        occupancy = results['group_stats']['occupancy']
        main_text.append("\nNeural state occupancy rates were {affair:.1f}% for the affair condition and {paranoia:.1f}% for the paranoia condition.".format(
            affair=occupancy['affair'] * 100,
            paranoia=occupancy['paranoia'] * 100
        ))
    
    if 'group_stats' in results and 'transitions' in results['group_stats']:
        transitions = results['group_stats']['transitions']
        main_text.append("State transition rates were {affair:.2f}% for the affair condition and {paranoia:.2f}% for the paranoia condition.".format(
            affair=transitions['affair']['entry_rate'] * 100,
            paranoia=transitions['paranoia']['entry_rate'] * 100
        ))
    
    summary['main_text'] = "\n".join(main_text)
    
    # Generate supplementary material
    supplement = []
    supplement.append("## Detailed Bayesian GLMM Results")
    supplement.append("\nThe Bayesian generalized linear mixed model (GLMM) was used to analyze the relationship between experimental conditions and neural state features.")
    supplement.append("\nWe report posterior probabilities and 95% highest density intervals (HDI) rather than p-values, as is appropriate for Bayesian analysis. Features with posterior probabilities > 0.95 or < 0.05 are considered to have credible effects.")
    
    # Add detailed results for all features
    supplement.append("\n### Results for All Features")
    supplement.append("Table S1 presents the detailed results for all features included in the analysis, sorted by standardized effect size.")
    
    # Add table with all features
    feature_table = df_sorted.copy()
    feature_table['Credible Effect'] = feature_table['Credible Effect'].map({True: 'Yes', False: 'No'})
    feature_table = feature_table[['Feature', 'Group Coefficient', 'HDI Lower', 'HDI Upper', 'Posterior Probability', 
                                   'Odds Ratio', 'Standardized Effect', 'Practical Significance', 'Credible Effect']]
    supplement.append("\nTable S1: GLMM Results for All Features")
    supplement.append(feature_table.to_string(index=False, float_format='%.3f'))
    
    # Add detailed results for credible features
    if credible_features:
        main_text.append("The features with credible effects were: {features}.".format(
            features=", ".join([f"'{f}'" for f in credible_features])
        ))
        main_text.append("For these features, the mean standardized effect size was {effect:.2f} (odds ratio: {odd_ratio:.2f}).".format(
            effect=avg_credible_effect,
            odd_ratio=avg_credible_odd_ratio
        ))
        
        # Add explanation about coefficient direction
        positive_coef_features = [f for f in credible_features if feature_data[f].get('group_coef', 0) > 0]
        negative_coef_features = [f for f in credible_features if feature_data[f].get('group_coef', 0) < 0]
        
        if positive_coef_features:
            pos_features_str = ", ".join([f"'{f}'" for f in positive_coef_features])
            main_text.append(f"The following features showed higher activation in the paranoia condition: {pos_features_str}.")
        
        if negative_coef_features:
            neg_features_str = ", ".join([f"'{f}'" for f in negative_coef_features])
            main_text.append(f"The following features showed higher activation in the affair condition: {neg_features_str}.")
            
        supplement.append("\n\n### Features with Credible Effects")
        supplement.append("The following features showed credible group effects (posterior probability > 0.95 or < 0.05):")
        for feature in credible_features:
            data = feature_data[feature]
            supplement.append("\n#### {feature}".format(feature=feature))
            supplement.append("- Group coefficient: {coef:.3f} [{lower:.3f}, {upper:.3f}]".format(
                coef=data.get('group_coef', np.nan),
                lower=data.get('group_hdi', {}).get('lower', np.nan),
                upper=data.get('group_hdi', {}).get('upper', np.nan)
            ))
            supplement.append("- Posterior probability: {prob:.4f}".format(
                prob=data.get('group_prob', np.nan)
            ))
            supplement.append("- Odds ratio: {odd_ratio:.3f} [{lower:.3f}, {upper:.3f}]".format(
                odd_ratio=data.get('odds_ratio', {}).get('value', np.nan),
                lower=data.get('odds_ratio', {}).get('lower', np.nan),
                upper=data.get('odds_ratio', {}).get('upper', np.nan)
            ))
            supplement.append("- Standardized effect: {effect:.3f}".format(
                effect=data.get('std_effect', np.nan)
            ))
            supplement.append("- Practical significance: {sig}".format(
                sig=data.get('practical_significance', '')
            ))
    
    # Add information about interaction effects
    supplement.append("\n\n### Interaction Effects")
    supplement.append("Interaction effects (feature × group) were examined for all features.")
    
    significant_interactions = []
    for feature, data in feature_data.items():
        if data.get('interaction_prob', 0) > 0.95 or data.get('interaction_prob', 0) < 0.05:
            significant_interactions.append((feature, data))
    
    if significant_interactions:
        supplement.append("Significant interaction effects were found for the following features:")
        for feature, data in significant_interactions:
            supplement.append("\n- {feature}: coefficient = {coef:.3f}, posterior probability = {prob:.4f}".format(
                feature=feature,
                coef=data.get('interaction_coef', np.nan),
                prob=data.get('interaction_prob', np.nan)
            ))
    else:
        supplement.append("No significant interaction effects were found.")
    
    summary['supplement'] = "\n".join(supplement)
    
    return summary

def plot_results(results, output_dir='.'):
    """
    Generate plots for the GLMM results.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing parsed GLMM results
    output_dir : str
        Directory to save the plots
    
    Returns:
    --------
    dict
        Dictionary of generated figures
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    figures = {}
    
    # Create dataframe from results
    feature_data = results['feature_data']
    credible_features = results['credible_features']  # Changed from sig_features
    
    feature_rows = []
    for feature in feature_data:
        data = feature_data[feature]
        row = {
            'Feature': feature,
            'Group Coefficient': data.get('group_coef', np.nan),
            'Posterior Probability': data.get('group_prob', np.nan),
            'HDI Lower': data.get('group_hdi', {}).get('lower', np.nan),
            'HDI Upper': data.get('group_hdi', {}).get('upper', np.nan),
            'Odds Ratio': data.get('odds_ratio', {}).get('value', np.nan),
            'Standardized Effect': data.get('std_effect', np.nan),
            'Credible Effect': feature in credible_features  # Changed from 'Significant'
        }
        feature_rows.append(row)
    
    df = pd.DataFrame(feature_rows)
    
    # Sort by standardized effect
    df_sorted = df.sort_values('Standardized Effect', ascending=False)
    
    # 1. Forest plot of group coefficients with HDIs
    plt.figure(figsize=(10, 6))
    
    # Plot HDI ranges
    for i, (_, row) in enumerate(df_sorted.iterrows()):
        color = 'darkblue' if row['Credible Effect'] else 'lightblue'  # Changed from 'Significant'
        plt.plot([row['HDI Lower'], row['HDI Upper']], [i, i], color=color, linewidth=2)
        plt.plot(row['Group Coefficient'], i, 'o', color=color, markersize=8)
    
    # Add feature names
    plt.yticks(range(len(df_sorted)), df_sorted['Feature'])
    
    # Add vertical line at zero
    plt.axvline(x=0, color='gray', linestyle='--')
    
    plt.xlabel('Group Coefficient (with 95% HDI)')
    plt.title('Bayesian GLMM Group Coefficients by Feature')
    plt.tight_layout()
    
    plt.savefig(f'{output_dir}/group_coefficients_forest_plot.png', dpi=300, bbox_inches='tight')
    figures['forest_plot'] = plt
    
    # 2. Bar plot of odds ratios
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Odds Ratio', y='Feature', data=df_sorted, 
                hue='Credible Effect', palette=['lightblue', 'darkblue'])  # Changed from 'Significant'
    
    plt.axvline(x=1, color='gray', linestyle='--')
    plt.title('Odds Ratios by Feature')
    plt.tight_layout()
    
    plt.savefig(f'{output_dir}/odds_ratios_bar_plot.png', dpi=300, bbox_inches='tight')
    figures['odds_ratio_plot'] = plt
    
    # 3. Bar plot of standardized effects
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Standardized Effect', y='Feature', data=df_sorted, 
                hue='Credible Effect', palette=['lightblue', 'darkblue'])  # Changed from 'Significant'
    
    plt.title('Standardized Effects by Feature')
    plt.tight_layout()
    
    plt.savefig(f'{output_dir}/standardized_effects_bar_plot.png', dpi=300, bbox_inches='tight')
    figures['std_effect_plot'] = plt
    
    # 4. Posterior probability plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Posterior Probability', y='Feature', data=df_sorted, 
                hue='Credible Effect', palette=['lightblue', 'darkblue'])
    
    # Add reference lines at 0.05 and 0.95
    plt.axvline(x=0.05, color='red', linestyle='--', alpha=0.5)
    plt.axvline(x=0.95, color='red', linestyle='--', alpha=0.5)
    
    plt.title('Posterior Probabilities by Feature')
    plt.tight_layout()
    
    plt.savefig(f'{output_dir}/posterior_probabilities_bar_plot.png', dpi=300, bbox_inches='tight')
    figures['posterior_prob_plot'] = plt
    
    # 5. Group stats visualization
    if 'group_stats' in results and 'occupancy' in results['group_stats']:
        plt.figure(figsize=(8, 6))
        occupancy = results['group_stats']['occupancy']
        
        labels = ['Affair', 'Paranoia']
        values = [occupancy['affair'] * 100, occupancy['paranoia'] * 100]
        
        plt.bar(labels, values, color=['darkblue', 'lightblue'])
        plt.title('Neural State Occupancy by Condition')
        plt.ylabel('Occupancy (%)')
        plt.ylim(0, 100)
        
        plt.savefig(f'{output_dir}/occupancy_bar_plot.png', dpi=300, bbox_inches='tight')
        figures['occupancy_plot'] = plt
    
    return figures

def main(state_affair, save_dir):
    """
    Main function to parse, summarize, and visualize GLMM results.
    
    Parameters:
    -----------
    file_path : str
        Path to the text file containing GLMM results
    output_dir : str
        Directory to save outputs
    """
    os.makedirs(save_dir, exist_ok=True)
    # Parse results
    state_map = {"affair_to_paranoia": {0:1, 1:2, 2:0}, "paranoia_to_affair": {1:0, 2:1, 0:2}}
    state_paranoia = state_map["affair_to_paranoia"][state_affair]
    
    print(f"Getting data for state_affair={state_affair}, state_paranoia={state_paranoia}")
    meta_data_a0p1, main_analysis_a0p1, cross_validation_a0p1 = get_brain_content_data(state_affair, state_paranoia)
    
    # Check what's in main_analysis before parsing
    print("Type of main_analysis:", type(main_analysis_a0p1))
    if isinstance(main_analysis_a0p1, dict):
        print("Keys in main_analysis:", list(main_analysis_a0p1.keys()))
        
        # Check if feature_results exists and print a sample
        if 'feature_results' in main_analysis_a0p1:
            features = list(main_analysis_a0p1['feature_results'].keys())
            if features:
                sample_feature = features[0]
                print(f"Sample feature: {sample_feature}")
                print(f"Data for {sample_feature}:", main_analysis_a0p1['feature_results'][sample_feature])
    
    results = parse_glmm_results(main_analysis_a0p1)
    
    # Check what's in results after parsing
    print("Keys in parsed results:", list(results.keys()))
    if 'feature_data' in results:
        print("Features in results:", list(results['feature_data'].keys()))
        for feature in list(results['feature_data'].keys())[:3]:  # First 3 features
            print(f"\nData for {feature}:")
            for key, value in results['feature_data'][feature].items():
                print(f"  {key}: {value}")
    
    # Generate summary
    summary = summarize_for_paper(results)
    
    # Save main text and supplement
    with open(f'{save_dir}/main_text_summary.md', 'w') as f:
        f.write(summary['main_text'])
    
    with open(f'{save_dir}/supplementary_material.md', 'w') as f:
        f.write(summary['supplement'])
    
    # Save tables as CSV
    for name, df in summary['tables'].items():
        df.to_csv(f'{save_dir}/{name}_table.csv', index=False)
    
    # Generate plots
    figures = plot_results(results, save_dir)
    
    print(f"Analysis complete. Results saved to {save_dir}")
    print("\nMain Text Summary Preview:")
    print("==========================")
    print(summary['main_text'])
    
    return results, summary, figures

if __name__ == "__main__":
    import os
    from pathlib import Path
    from dotenv import load_dotenv

    
    load_dotenv()
    OUTPUT_DIR = Path(os.getenv("SCRATCH_DIR")) / "output"
    
    main(0, OUTPUT_DIR / "13_glmm_brain_summary")