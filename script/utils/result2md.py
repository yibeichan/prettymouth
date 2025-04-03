import os
from pathlib import Path
import json
import pandas as pd
import numpy as np

# ************************************************************************
# Brain State Functions (for .npz files)
# ************************************************************************

def brain_create_main_effects_table(data):
    """Creates main effects table for brain state data"""
    # Initialize lists to store results
    features = []
    coefficients = []
    posterior_sds = []
    prob_positive = []
    bayes_factors = []
    evidence_categories = []
    odds_ratios = []
    odds_lower = []
    odds_upper = []
    
    # Extract feature results from main analysis
    feature_results = data['main_analysis']['feature_results']
    
    # Iterate through each feature
    for feature_name, results in feature_results.items():
        # Add feature main effect
        features.append(f"{feature_name}")
        coefficients.append(results['coefficients'][feature_name])
        posterior_sds.append(results['posterior_sds'][feature_name])
        prob_positive.append(results['prob_positive'][feature_name])
        bayes_factors.append(results['bayes_factors'][feature_name])
        evidence_categories.append(results['evidence_categories'][feature_name])
        odds_ratios.append(results['odds_ratios'][feature_name]['odds_ratio'])
        odds_lower.append(results['odds_ratios'][feature_name]['lower'])
        odds_upper.append(results['odds_ratios'][feature_name]['upper'])
        
        # Add group interaction effect
        interaction_key = f"group_{feature_name}_interaction"
        features.append(f"{feature_name}:Group Interaction")
        coefficients.append(results['coefficients'][interaction_key])
        posterior_sds.append(results['posterior_sds'][interaction_key])
        prob_positive.append(results['prob_positive'][interaction_key])
        bayes_factors.append(results['bayes_factors'][interaction_key])
        evidence_categories.append(results['evidence_categories'][interaction_key])
        odds_ratios.append(results['odds_ratios'][interaction_key]['odds_ratio'])
        odds_lower.append(results['odds_ratios'][interaction_key]['lower'])
        odds_upper.append(results['odds_ratios'][interaction_key]['upper'])
    
    # Create DataFrame
    df = pd.DataFrame({
        'Feature': features,
        'Coefficient': coefficients,
        'Std. Error': posterior_sds,
        'P(Effect > 0)': prob_positive,
        'Bayes Factor': bayes_factors,
        'Evidence Category': evidence_categories,
        'Odds Ratio': odds_ratios,
        'OR Lower': odds_lower,
        'OR Upper': odds_upper
    })
    
    # Sort by Bayes Factor (descending)
    df = df.sort_values(by='Bayes Factor', ascending=False)
    
    # Format numeric columns
    df['Coefficient'] = df['Coefficient'].round(3)
    df['Std. Error'] = df['Std. Error'].round(3)
    df['P(Effect > 0)'] = df['P(Effect > 0)'].round(3)
    df['Bayes Factor'] = df['Bayes Factor'].round(2)
    df['Odds Ratio'] = df['Odds Ratio'].round(3)
    df['OR Lower'] = df['OR Lower'].round(3)
    df['OR Upper'] = df['OR Upper'].round(3)
    
    return df

def brain_create_group_effects_table(data):
    """Creates group effects table for brain state data"""
    # Initialize lists to store results
    features = []
    affair_coefs = []
    affair_ors = []
    affair_prob_pos = []
    paranoia_coefs = []
    paranoia_ors = []
    paranoia_prob_pos = []
    diff_coefs = []
    prob_stronger_affair = []
    
    # Extract feature results from main analysis
    feature_results = data['main_analysis']['feature_results']
    
    # Iterate through each feature
    for feature_name, results in feature_results.items():
        # Check if group_specific_effects exists for this feature
        if 'group_specific_effects' in results and feature_name in results['group_specific_effects']:
            gse = results['group_specific_effects'][feature_name]
            
            features.append(feature_name)
            
            # Affair group
            affair_coefs.append(gse['affair_group']['coefficient'])
            affair_ors.append(gse['affair_group']['odds_ratio'])
            affair_prob_pos.append(gse['affair_group']['prob_positive'])
            
            # Paranoia group
            paranoia_coefs.append(gse['paranoia_group']['coefficient'])
            paranoia_ors.append(gse['paranoia_group']['odds_ratio'])
            paranoia_prob_pos.append(gse['paranoia_group']['prob_positive'])
            
            # Difference between groups
            diff_coefs.append(gse['diff_between_groups']['coefficient'])
            prob_stronger_affair.append(gse['diff_between_groups']['prob_stronger_in_affair'])
    
    # Create DataFrame
    df = pd.DataFrame({
        'Feature': features,
        'Affair Coef': affair_coefs,
        'Affair OR': affair_ors,
        'Affair P(>0)': affair_prob_pos,
        'Paranoia Coef': paranoia_coefs,
        'Paranoia OR': paranoia_ors,
        'Paranoia P(>0)': paranoia_prob_pos,
        'Diff (A-P)': diff_coefs,
        'P(Stronger in Affair)': prob_stronger_affair
    })
    
    # Format numeric columns
    df['Affair Coef'] = df['Affair Coef'].round(3)
    df['Affair OR'] = df['Affair OR'].round(3)
    df['Affair P(>0)'] = df['Affair P(>0)'].round(3)
    df['Paranoia Coef'] = df['Paranoia Coef'].round(3)
    df['Paranoia OR'] = df['Paranoia OR'].round(3)
    df['Paranoia P(>0)'] = df['Paranoia P(>0)'].round(3)
    df['Diff (A-P)'] = df['Diff (A-P)'].round(3)
    df['P(Stronger in Affair)'] = df['P(Stronger in Affair)'].round(3)
    
    # Sort by absolute difference magnitude
    df = df.sort_values(by='Diff (A-P)', key=abs, ascending=False)
    
    return df

def brain_create_multiple_comparison_table(data):
    """Creates multiple comparison table for brain state data"""
    # Extract multiple comparison results
    mc_results = data['main_analysis']['multiple_comparison']
    fdr_threshold = mc_results['fdr_threshold']
    credible_effects = mc_results['credible_effects']
    all_effects = mc_results['all_effects']
    
    # Create DataFrame
    effects = []
    posterior_probs = []
    fdrs = []
    is_credible = []
    
    for effect, values in all_effects.items():
        effects.append(effect)
        posterior_probs.append(values['posterior_prob'])
        fdrs.append(values['fdr'])
        is_credible.append(effect in credible_effects)
    
    df = pd.DataFrame({
        'Effect': effects,
        'Posterior Probability': posterior_probs,
        'FDR': fdrs,
        'Credible (FDR < 0.05)': is_credible
    })
    
    # Sort by FDR (ascending)
    df = df.sort_values(by='FDR')
    
    # Format numeric columns
    df['Posterior Probability'] = df['Posterior Probability'].round(3)
    df['FDR'] = df['FDR'].round(3)
    
    return df

def brain_create_cv_summary(data):
    """Creates cross-validation summary table for brain state data"""
    cv_results = data['cross_validation']['subject_cv']
    
    # Group by group
    affair_results = [r for r in cv_results if r['group'] == 'affair']
    paranoia_results = [r for r in cv_results if r['group'] == 'paranoia']
    
    # Extract coefficients for each interaction term
    def extract_interaction_coeffs(results, interaction_name):
        return [r['coefficients'].get(f'group_{interaction_name}_interaction', np.nan) for r in results]
    
    # Get all interaction names (excluding 'const' and 'group_coded')
    sample_coeffs = cv_results[0]['coefficients']
    interaction_names = [k.replace('group_', '').replace('_interaction', '') for k in sample_coeffs.keys() 
                         if k not in ['const', 'group_coded'] and '_interaction' in k]
    
    # Create DataFrames for each group
    df_affair = pd.DataFrame({
        'Subject': [r['subject'] for r in affair_results],
        **{name: extract_interaction_coeffs(affair_results, name) for name in interaction_names}
    })
    
    df_paranoia = pd.DataFrame({
        'Subject': [r['subject'] for r in paranoia_results],
        **{name: extract_interaction_coeffs(paranoia_results, name) for name in interaction_names}
    })
    
    # Calculate summary statistics
    summary_stats = []
    for name in interaction_names:
        affair_mean = df_affair[name].mean()
        affair_std = df_affair[name].std()
        paranoia_mean = df_paranoia[name].mean()
        paranoia_std = df_paranoia[name].std()
        
        summary_stats.append({
            'Interaction': name,
            'Affair Mean': affair_mean,
            'Affair Std': affair_std,
            'Paranoia Mean': paranoia_mean,
            'Paranoia Std': paranoia_std,
            'Stability Ratio': min(affair_std, paranoia_std) / max(affair_std, paranoia_std)
        })
    
    df_summary = pd.DataFrame(summary_stats)
    
    # Format numeric columns
    for col in df_summary.columns:
        if col != 'Interaction':
            df_summary[col] = df_summary[col].round(4)
    
    # Sort by stability ratio (descending)
    df_summary = df_summary.sort_values(by='Stability Ratio', ascending=False)
    
    return df_summary

def brain_create_all_tables(data):
    """Creates all tables for brain state data"""
    # Create tables
    main_effects_table = brain_create_main_effects_table(data)
    group_effects_table = brain_create_group_effects_table(data)
    multiple_comparison_table = brain_create_multiple_comparison_table(data)
    cv_summary_table = brain_create_cv_summary(data)
    
    # Group occupancy and transition statistics
    group_stats = data['main_analysis']['group_stats']
    
    occupancy_df = pd.DataFrame({
        'Group': ['Affair', 'Paranoia'],
        'Occupancy Rate': [group_stats['occupancy']['affair'], group_stats['occupancy']['paranoia']]
    })
    
    transition_df = pd.DataFrame({
        'Group': ['Affair', 'Paranoia'],
        'Entry Rate': [group_stats['transitions']['affair']['entry_rate'], 
                      group_stats['transitions']['paranoia']['entry_rate']],
        'Exit Rate': [group_stats['transitions']['affair']['exit_rate'], 
                     group_stats['transitions']['paranoia']['exit_rate']]
    })
    
    return {
        'main_effects': main_effects_table,
        'group_effects': group_effects_table,
        'multiple_comparison': multiple_comparison_table,
        'cv_summary': cv_summary_table,
        'occupancy': occupancy_df,
        'transitions': transition_df
    }

def brain_create_paper_tables(data):
    """Generate publication-ready tables for a brain state dynamics paper"""
    tables = brain_create_all_tables(data)
    
    # Table 1: Main Effects (Publication Format)
    pub_main_effects = tables['main_effects'][['Feature', 'Coefficient', 'Std. Error', 
                                              'P(Effect > 0)', 'Bayes Factor', 'Odds Ratio']].copy()
    
    # Add significance markers based on Bayes Factor
    def get_significance(bf):
        if bf >= 100:
            return "***"
        elif bf >= 10:
            return "**"
        elif bf >= 3:
            return "*"
        else:
            return ""
    
    pub_main_effects['Significance'] = pub_main_effects['Bayes Factor'].apply(get_significance)
    pub_main_effects['Coef ± SE'] = pub_main_effects.apply(
        lambda row: f"{row['Coefficient']} ± {row['Std. Error']} {row['Significance']}", axis=1)
    
    pub_main_effects = pub_main_effects[['Feature', 'Coef ± SE', 'P(Effect > 0)', 'Bayes Factor', 'Odds Ratio']]
    
    # Table 2: Group Comparison (Publication Format)
    pub_group_effects = tables['group_effects'][['Feature', 'Affair Coef', 'Paranoia Coef', 
                                                'Diff (A-P)', 'P(Stronger in Affair)']].copy()
    
    # Mark credible differences
    credible_effects = tables['multiple_comparison'][tables['multiple_comparison']['Credible (FDR < 0.05)'] == True]['Effect']
    pub_group_effects['Credible Diff'] = pub_group_effects['Feature'].apply(
        lambda x: any(x in effect for effect in credible_effects))
    
    # Format group differences
    pub_group_effects['Group Difference'] = pub_group_effects.apply(
        lambda row: f"{row['Diff (A-P)']}{' †' if row['Credible Diff'] else ''}", axis=1)
    
    pub_group_effects = pub_group_effects[['Feature', 'Affair Coef', 'Paranoia Coef', 
                                          'Group Difference', 'P(Stronger in Affair)']]
    
    # Create the dictionary of publication tables
    pub_tables = {
        'main_effects': pub_main_effects,
        'group_effects': pub_group_effects,
        'occupancy': tables['occupancy'],
        'transitions': tables['transitions'],
    }
    
    return pub_tables

def brain_generate_markdown(data_path, output_path="brain_state_analysis_results.md"):
    """Generate a single markdown file with all brain state tables"""
    # Load data from .npz file
    loaded_data = np.load(data_path, allow_pickle=True)
    data = loaded_data["results"].item()
    
    # Get all tables
    tables = brain_create_all_tables(data)
    paper_tables = brain_create_paper_tables(data)
    
    # Create markdown string with all tables
    markdown_content = """# Brain State Dynamics Analysis Supplementary Material

This document contains all statistical tables from the brain state dynamics analysis.

## Table of Contents

1. [Main Effects and Interactions](#1-main-effects-and-interactions)
2. [Group-Specific Effects](#2-group-specific-effects)
3. [Multiple Comparisons Analysis](#3-multiple-comparisons-analysis)
4. [Cross-Validation Summary](#4-cross-validation-summary)
5. [State Occupancy Rates](#5-state-occupancy-rates)
6. [State Transition Rates](#6-state-transition-rates)
7. [Publication-Ready Main Effects](#7-publication-ready-main-effects)
8. [Publication-Ready Group Effects](#8-publication-ready-group-effects)

---

"""
    
    # 1. Main Effects Table
    markdown_content += """## 1. Main Effects and Interactions

This table presents the main effects and group interactions for all features analyzed in the brain state dynamics study. Features are sorted by Bayes Factor (strongest evidence first).

"""
    markdown_content += tables['main_effects'].to_markdown(index=False)
    markdown_content += "\n\n---\n\n"
    
    # 2. Group Effects Table
    markdown_content += """## 2. Group-Specific Effects

This table presents the feature effects separately for each group (Affair and Paranoia) as well as the difference between groups. Features are sorted by the absolute magnitude of the difference between groups.

"""
    markdown_content += tables['group_effects'].to_markdown(index=False)
    markdown_content += "\n\n---\n\n"
    
    # 3. Multiple Comparison Table
    markdown_content += """## 3. Multiple Comparisons Analysis

This table presents the results of multiple comparisons correction using False Discovery Rate (FDR). Effects with FDR < 0.05 are considered credible.

"""
    markdown_content += tables['multiple_comparison'].to_markdown(index=False)
    markdown_content += "\n\n---\n\n"
    
    # 4. Cross-Validation Summary
    markdown_content += """## 4. Cross-Validation Summary

This table presents the summary of cross-validation results, showing the stability of feature effects across subjects within each group.

"""
    markdown_content += tables['cv_summary'].to_markdown(index=False)
    markdown_content += "\n\n---\n\n"
    
    # 5. State Occupancy
    markdown_content += """## 5. State Occupancy Rates

This table presents the state occupancy rates for each experimental group.

"""
    markdown_content += tables['occupancy'].to_markdown(index=False)
    markdown_content += "\n\n---\n\n"
    
    # 6. State Transitions
    markdown_content += """## 6. State Transition Rates

This table presents the state entry and exit rates for each experimental group.

"""
    markdown_content += tables['transitions'].to_markdown(index=False)
    markdown_content += "\n\n---\n\n"
    
    # 7. Publication-Ready Main Effects
    markdown_content += """## 7. Publication-Ready Main Effects

Publication-ready table of main effects and interactions. Significance: * BF > 3, ** BF > 10, *** BF > 100.

"""
    markdown_content += paper_tables['main_effects'].to_markdown(index=False)
    markdown_content += "\n\n---\n\n"
    
    # 8. Publication-Ready Group Effects
    markdown_content += """## 8. Publication-Ready Group Effects

Publication-ready table of group-specific effects. † indicates credible differences (FDR < 0.05).

"""
    markdown_content += paper_tables['group_effects'].to_markdown(index=False)
    
    # Write markdown content to file
    with open(output_path, 'w') as f:
        f.write(markdown_content)
    
    print(f"Markdown file generated: {output_path}")
    return output_path

def process_brain_state_data(data_path, output_path=None):
    """Process brain state data and generate a markdown file"""
    if output_path is None:
        # Generate output path from input path and include cluster/state information
        dir_name = os.path.dirname(data_path)
        base_dir = os.path.basename(dir_name)
        output_path = f"{base_dir}_results.md"
    
    return brain_generate_markdown(data_path, output_path)


# ************************************************************************
# Behavioral GLMM Functions (for JSON files)
# ************************************************************************

def load_behavioral_data(json_file_path):
    """Load behavioral GLMM results data from a JSON file"""
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        return data
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error loading JSON file: {e}")
        return None

def behavioral_create_main_effects_table(data):
    """Create a DataFrame with main effects from behavioral GLMM results"""
    # Extract the combined results section
    results = data["content_relationships"]["combined_result"]
    
    # Create list to store main effects
    main_effects = []
    
    # Get all features that have coefficients
    all_features = [k for k in results["coefficients"].keys() 
                   if not k.startswith("group_") and 
                      not k.startswith("target_lag") and 
                      k != "const"]
    
    for feature in all_features:
        # Get coefficient
        coefficient = results["coefficients"][feature]
        
        # Get standard error from posterior_sds
        std_error = results["posterior_sds"].get(feature, np.nan)
        
        # Get p(effect > 0)
        prob_positive = results["prob_positive"].get(feature, np.nan)
        
        # Get Bayes factor
        bayes_factor = results["bayes_factors"].get(feature, np.nan)
        
        # Get odds ratio and CI
        if feature in results["odds_ratios"]:
            odds_ratio = results["odds_ratios"][feature]["odds_ratio"]
            lower_ci = results["odds_ratios"][feature]["lower"]
            upper_ci = results["odds_ratios"][feature]["upper"]
        else:
            odds_ratio = np.nan
            lower_ci = np.nan
            upper_ci = np.nan
        
        # Get evidence category
        evidence_category = results["evidence_categories"].get(feature, "N/A")
        
        # Add to main effects list
        main_effects.append({
            "Feature": feature,
            "Coefficient": coefficient,
            "Std.Error": std_error,
            "P(Effect>0)": prob_positive,
            "Bayes Factor": bayes_factor,
            "Odds Ratio": odds_ratio,
            "Lower CI": lower_ci,
            "Upper CI": upper_ci,
            "Evidence": evidence_category
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(main_effects)
    
    # Sort by Bayes Factor (descending)
    if "Bayes Factor" in df.columns and not df["Bayes Factor"].isna().all():
        df = df.sort_values("Bayes Factor", ascending=False)
    
    # Format numeric columns
    for col in ["Coefficient", "Std.Error", "P(Effect>0)", "Lower CI", "Upper CI"]:
        if col in df.columns:
            df[col] = df[col].map(lambda x: f"{x:.3f}" if not pd.isna(x) else "N/A")
    
    # Format Bayes Factor (scientific notation for large values)
    if "Bayes Factor" in df.columns:
        df["Bayes Factor"] = df["Bayes Factor"].map(
            lambda x: f"{x:.2e}" if not pd.isna(x) and x > 10000 else
                     (f"{x:.2f}" if not pd.isna(x) else "N/A")
        )
    
    # Format Odds Ratio
    if "Odds Ratio" in df.columns:
        df["Odds Ratio"] = df["Odds Ratio"].map(lambda x: f"{x:.3f}" if not pd.isna(x) else "N/A")
    
    return df

def behavioral_create_interaction_effects_table(data):
    """Create a DataFrame with interaction effects from behavioral GLMM results"""
    # Extract the combined results section
    results = data["content_relationships"]["combined_result"]
    
    # Create list to store interaction effects
    interaction_effects = []
    
    # Get all interaction features
    interaction_features = [k for k in results["coefficients"].keys() 
                           if k.startswith("group_") and k.endswith("_interaction")]
    
    for feature in interaction_features:
        # Extract base feature name
        base_feature = feature.replace("group_", "").replace("_interaction", "")
        
        # Get coefficient
        coefficient = results["coefficients"][feature]
        
        # Get standard error from posterior_sds
        std_error = results["posterior_sds"].get(feature, np.nan)
        
        # Get p(effect > 0)
        prob_positive = results["prob_positive"].get(feature, np.nan)
        
        # Get Bayes factor
        bayes_factor = results["bayes_factors"].get(feature, np.nan)
        
        # Get odds ratio and CI
        if feature in results["odds_ratios"]:
            odds_ratio = results["odds_ratios"][feature]["odds_ratio"]
            lower_ci = results["odds_ratios"][feature]["lower"]
            upper_ci = results["odds_ratios"][feature]["upper"]
        else:
            odds_ratio = np.nan
            lower_ci = np.nan
            upper_ci = np.nan
        
        # Get evidence category
        evidence_category = results["evidence_categories"].get(feature, "N/A")
        
        # Add to interaction effects list
        interaction_effects.append({
            "Feature": base_feature,
            "Coefficient": coefficient,
            "Std.Error": std_error,
            "P(Effect>0)": prob_positive,
            "Bayes Factor": bayes_factor,
            "Odds Ratio": odds_ratio,
            "Lower CI": lower_ci,
            "Upper CI": upper_ci,
            "Evidence": evidence_category
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(interaction_effects)
    
    # Sort by Bayes Factor (descending)
    if "Bayes Factor" in df.columns and not df["Bayes Factor"].isna().all():
        df = df.sort_values("Bayes Factor", ascending=False)
    
    # Format numeric columns
    for col in ["Coefficient", "Std.Error", "P(Effect>0)", "Lower CI", "Upper CI"]:
        if col in df.columns:
            df[col] = df[col].map(lambda x: f"{x:.3f}" if not pd.isna(x) else "N/A")
    
    # Format Bayes Factor (scientific notation for large values)
    if "Bayes Factor" in df.columns:
        df["Bayes Factor"] = df["Bayes Factor"].map(
            lambda x: f"{x:.2e}" if not pd.isna(x) and x > 10000 else
                     (f"{x:.2f}" if not pd.isna(x) else "N/A")
        )
    
    # Format Odds Ratio
    if "Odds Ratio" in df.columns:
        df["Odds Ratio"] = df["Odds Ratio"].map(lambda x: f"{x:.3f}" if not pd.isna(x) else "N/A")
    
    return df

def behavioral_create_group_specific_effects_table(data):
    """Create a DataFrame with group-specific effects from behavioral GLMM results"""
    # Extract the combined results section
    results = data["content_relationships"]["combined_result"]
    
    # Create list to store group-specific effects
    group_effects = []
    
    # Get all features that have group-specific effects
    gs_features = results.get("group_specific_effects", {}).keys()
    
    for feature in gs_features:
        # Get group-specific data
        gs_data = results["group_specific_effects"][feature]
        
        # Extract affair group data
        affair_coef = gs_data["affair_group"]["coefficient"]
        affair_or = gs_data["affair_group"]["odds_ratio"]
        affair_prob = gs_data["affair_group"]["prob_positive"]
        
        # Extract paranoia group data
        paranoia_coef = gs_data["paranoia_group"]["coefficient"]
        paranoia_or = gs_data["paranoia_group"]["odds_ratio"]
        paranoia_prob = gs_data["paranoia_group"]["prob_positive"]
        
        # Extract difference data
        diff_coef = gs_data["diff_between_groups"]["coefficient"]
        prob_stronger_affair = gs_data["diff_between_groups"]["prob_stronger_in_affair"]
        
        # Format prob values
        affair_prob_str = f"{affair_prob:.3f}"
        paranoia_prob_str = f"{paranoia_prob:.3f}"
        prob_stronger_str = f"{prob_stronger_affair:.3f}"
        
        # Handle extreme probabilities
        if affair_prob > 0.999:
            affair_prob_str = ">0.999"
        elif affair_prob < 0.001:
            affair_prob_str = "<0.001"
            
        if paranoia_prob > 0.999:
            paranoia_prob_str = ">0.999"
        elif paranoia_prob < 0.001:
            paranoia_prob_str = "<0.001"
            
        if prob_stronger_affair > 0.999:
            prob_stronger_str = ">0.999"
        elif prob_stronger_affair < 0.001:
            prob_stronger_str = "<0.001"
        
        # Add to group effects list
        group_effects.append({
            "Feature": feature,
            "Affair Coef": f"{affair_coef:.3f}",
            "Affair OR": f"{affair_or:.3f}",
            "Affair P(>0)": affair_prob_str,
            "Paranoia Coef": f"{paranoia_coef:.3f}",
            "Paranoia OR": f"{paranoia_or:.3f}",
            "Paranoia P(>0)": paranoia_prob_str,
            "Diff (A-P)": f"{diff_coef:.3f}",
            "P(Stronger in Affair)": prob_stronger_str
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(group_effects)
    
    # Sort by absolute difference magnitude
    if "Diff (A-P)" in df.columns and not df.empty:
        df["abs_diff"] = df["Diff (A-P)"].apply(lambda x: abs(float(x)))
        df = df.sort_values("abs_diff", ascending=False)
        df = df.drop("abs_diff", axis=1)
    
    return df

def behavioral_create_multiple_comparison_table(data):
    """Create a DataFrame with multiple comparison results from behavioral GLMM results"""
    # Extract the combined results section
    results = data["content_relationships"]["combined_result"]
    
    # Check if multiple comparison data exists
    if "multiple_comparison" not in results:
        return pd.DataFrame()
    
    # Extract multiple comparison data
    mc_data = results["multiple_comparison"]
    
    # Create list to store multiple comparison results
    mc_results = []
    
    # Get all effects
    all_effects = mc_data.get("all_effects", {})
    
    # If all_effects is a dictionary with nested values
    if isinstance(all_effects, dict):
        for effect, effect_data in all_effects.items():
            posterior_prob = effect_data.get("posterior_prob", np.nan)
            fdr = effect_data.get("fdr", np.nan)
            is_credible = effect in mc_data.get("credible_effects", [])
            
            mc_results.append({
                "Effect": effect,
                "Posterior Probability": posterior_prob,
                "FDR": fdr,
                "Credible (FDR < 0.05)": is_credible
            })
    # If all_effects is a list
    elif isinstance(all_effects, list):
        posterior_probs = mc_data.get("posterior_probs", {})
        
        # Check if cumulative_fdr exists
        if "cumulative_fdr" in mc_data:
            for effect in all_effects:
                posterior_prob = posterior_probs.get(effect, np.nan)
                fdr = mc_data["cumulative_fdr"].get(effect, np.nan)
                is_credible = effect in mc_data.get("credible_effects", [])
                
                mc_results.append({
                    "Effect": effect,
                    "Posterior Probability": posterior_prob,
                    "FDR": fdr,
                    "Credible (FDR < 0.05)": is_credible
                })
    
    # Convert to DataFrame
    df = pd.DataFrame(mc_results)
    
    # Sort by FDR (ascending)
    if "FDR" in df.columns and not df.empty:
        df = df.sort_values("FDR")
    
    # Format numeric columns
    if "Posterior Probability" in df.columns:
        df["Posterior Probability"] = df["Posterior Probability"].map(
            lambda x: f"{x:.3f}" if not pd.isna(x) else "N/A"
        )
    
    if "FDR" in df.columns:
        df["FDR"] = df["FDR"].map(
            lambda x: f"{x:.3f}" if not pd.isna(x) else "N/A"
        )
    
    return df

def behavioral_generate_markdown(data, output_path=None):
    """Generate a markdown file with behavioral GLMM results"""
    if output_path is None:
        output_path = "behavioral_glmm_results.md"
    
    # Create tables
    main_effects_df = behavioral_create_main_effects_table(data)
    interaction_effects_df = behavioral_create_interaction_effects_table(data)
    group_effects_df = behavioral_create_group_specific_effects_table(data)
    mc_df = behavioral_create_multiple_comparison_table(data)
    
    # Get metadata
    metadata = data.get("metadata", {})
    coding_type = metadata.get("coding_type", "N/A")
    reference_group = metadata.get("reference_group", "N/A")
    n_subjects = metadata.get("n_subjects", {})
    n_timepoints = metadata.get("n_timepoints", "N/A")
    timestamp = metadata.get("timestamp", "N/A")
    
    # Create markdown content
    markdown_content = "# Behavioral GLMM Results for Content Analysis\n\n"
    
    # Add metadata section
    # markdown_content += "## Analysis Information\n\n"
    # markdown_content += f"- **Coding Type**: {coding_type}\n"
    # markdown_content += f"- **Reference Group**: {reference_group}\n"
    
    if n_subjects:
        markdown_content += f"- **Subjects (Affair)**: {n_subjects.get('affair', 'N/A')}\n"
        markdown_content += f"- **Subjects (Paranoia)**: {n_subjects.get('paranoia', 'N/A')}\n"
    
    # markdown_content += f"- **Timepoints**: {n_timepoints}\n"
    # markdown_content += f"- **Analysis Date**: {timestamp}\n\n"
    
    # Add main effects table
    markdown_content += "## Main Effects\n\n"
    if not main_effects_df.empty:
        markdown_content += "This table presents the main effects for all features in the analysis.\n\n"
        markdown_content += main_effects_df.to_markdown(index=False)
    else:
        markdown_content += "No main effects data available.\n"
    markdown_content += "\n\n"
    
    # Add interaction effects table
    markdown_content += "## Group Interaction Effects\n\n"
    if not interaction_effects_df.empty:
        markdown_content += "This table presents the interaction effects between group and each feature.\n\n"
        markdown_content += interaction_effects_df.to_markdown(index=False)
    else:
        markdown_content += "No interaction effects data available.\n"
    markdown_content += "\n\n"
    
    # Add group-specific effects table
    markdown_content += "## Group-Specific Effects\n\n"
    if not group_effects_df.empty:
        markdown_content += "This table presents effects separately for each context group and their differences.\n\n"
        markdown_content += group_effects_df.to_markdown(index=False)
    else:
        markdown_content += "No group-specific effects data available.\n"
    markdown_content += "\n\n"
    
    # Add multiple comparison table
    markdown_content += "## Multiple Comparisons Analysis\n\n"
    if not mc_df.empty:
        markdown_content += "This table presents the results of multiple comparisons correction using False Discovery Rate (FDR).\n\n"
        markdown_content += mc_df.to_markdown(index=False)
    else:
        markdown_content += "No multiple comparison data available.\n"
    markdown_content += "\n\n"
    
    # Add statistical notation
    markdown_content += "## Statistical Notation\n\n"
    markdown_content += "- **Coefficient**: The log odds effect size from the GLMM model\n"
    markdown_content += "- **Std.Error**: Posterior standard deviation of the coefficient\n"
    markdown_content += "- **P(Effect>0)**: Posterior probability that the effect is positive\n"
    markdown_content += "- **Bayes Factor**: Relative evidence in favor of the effect existing vs. not existing\n"
    markdown_content += "- **Odds Ratio**: Exponentiated coefficient, representing the multiplicative effect on odds\n"
    markdown_content += "- **Lower/Upper CI**: Lower and upper bounds of the 95% highest density interval for the odds ratio\n"
    markdown_content += "- **Evidence**: Categorical interpretation of the Bayes factor strength\n"
    markdown_content += "- **FDR**: False Discovery Rate, corrected for multiple comparisons\n"
    
    # Write to file
    with open(output_path, 'w') as f:
        f.write(markdown_content)
    
    print(f"Markdown file generated: {output_path}")
    return output_path

def process_behavioral_data(json_file_path, output_path=None):
    """Process behavioral GLMM data and generate a markdown file"""
    # Load the data
    data = load_behavioral_data(json_file_path)
    if data is None:
        return None
    
    # If output_path is not specified, create one based on input path
    if output_path is None:
        dir_name = os.path.dirname(json_file_path)
        base_name = os.path.splitext(os.path.basename(json_file_path))[0]
        output_path = os.path.join(dir_name, f"{base_name}_glmm_results.md")
    
    # Generate markdown
    return behavioral_generate_markdown(data, output_path)

# ************************************************************************
# Main function to process any type of data
# ************************************************************************

def process_data_to_markdown(file_path, output_path=None, data_type=None):
    """
    Process any type of data (brain state or behavioral) to markdown
    
    Parameters:
    -----------
    file_path : str
        Path to the data file (NPZ for brain state, JSON for behavioral)
    output_path : str, optional
        Path to save the output markdown file
    data_type : str, optional
        Type of data to process: 'brain' or 'behavioral' (if not specified, will be auto-detected)
        
    Returns:
    --------
    str
        Path to the generated markdown file
    """
    # Auto-detect data type if not specified
    if data_type is None:
        if file_path.endswith('.npz'):
            data_type = 'brain'
        elif file_path.endswith('.json'):
            data_type = 'behavioral'
        else:
            print(f"Unable to determine data type from file extension: {file_path}")
            return None
    
    # Process based on data type
    if data_type.lower() == 'brain':
        return process_brain_state_data(file_path, output_path)
    elif data_type.lower() == 'behavioral':
        return process_behavioral_data(file_path, output_path)
    else:
        print(f"Unknown data type: {data_type}")
        return None