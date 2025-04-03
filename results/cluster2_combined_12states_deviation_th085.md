# Brain State GLMM Analysis Results

This document contains all statistical tables from the brain state GLMM analysis.

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

## 1. Main Effects and Interactions

This table presents the main effects and group interactions for all features analyzed in the brain state dynamics study. Features are sorted by Bayes Factor (strongest evidence first).

| Feature                             |   Coefficient |   Std. Error |   P(Effect > 0) |   Bayes Factor | Evidence Category    |   Odds Ratio |   OR Lower |   OR Upper |
|:------------------------------------|--------------:|-------------:|----------------:|---------------:|:---------------------|-------------:|-----------:|-----------:|
| arthur_speaking                     |         0.5   |        0.082 |           1     |    1.32804e+08 | Extreme evidence     |        1.648 |      1.404 |      1.934 |
| girl_speaking                       |        -0.946 |        0.315 |           0.001 |   89.69        | Very strong evidence |        0.388 |      0.209 |      0.721 |
| has_adj                             |        -0.253 |        0.093 |           0.003 |   39.64        | Very strong evidence |        0.776 |      0.647 |      0.932 |
| lee_speaking                        |        -0.211 |        0.09  |           0.01  |   15.21        | Strong evidence      |        0.81  |      0.678 |      0.967 |
| arthur_adj:Group Interaction        |        -0.269 |        0.151 |           0.037 |    4.94        | Moderate evidence    |        0.764 |      0.568 |      1.026 |
| has_noun                            |         0.097 |        0.081 |           0.886 |    2.07        | Anecdotal evidence   |        1.102 |      0.941 |      1.291 |
| lee_girl_verb                       |         0.211 |        0.214 |           0.838 |    1.63        | Anecdotal evidence   |        1.235 |      0.812 |      1.881 |
| has_adv:Group Interaction           |         0.084 |        0.092 |           0.821 |    1.53        | Anecdotal evidence   |        1.088 |      0.909 |      1.303 |
| has_verb                            |        -0.061 |        0.081 |           0.224 |    1.33        | Anecdotal evidence   |        0.941 |      0.803 |      1.102 |
| lee_girl_together                   |         0.14  |        0.196 |           0.762 |    1.29        | Anecdotal evidence   |        1.15  |      0.784 |      1.687 |
| has_adj:Group Interaction           |        -0.066 |        0.093 |           0.24  |    1.28        | Anecdotal evidence   |        0.937 |      0.781 |      1.124 |
| arthur_adj                          |        -0.102 |        0.151 |           0.249 |    1.26        | Anecdotal evidence   |        0.903 |      0.672 |      1.214 |
| has_noun:Group Interaction          |         0.044 |        0.08  |           0.709 |    1.16        | Anecdotal evidence   |        1.045 |      0.893 |      1.224 |
| lee_girl_together:Group Interaction |         0.101 |        0.195 |           0.697 |    1.14        | Anecdotal evidence   |        1.106 |      0.754 |      1.622 |
| girl_speaking:Group Interaction     |         0.156 |        0.315 |           0.689 |    1.13        | Anecdotal evidence   |        1.168 |      0.63  |      2.167 |
| lee_speaking:Group Interaction      |         0.033 |        0.09  |           0.64  |    1.07        | Anecdotal evidence   |        1.033 |      0.865 |      1.233 |
| has_verb:Group Interaction          |        -0.025 |        0.081 |           0.379 |    1.05        | Anecdotal evidence   |        0.975 |      0.833 |      1.143 |
| has_adv                             |         0.017 |        0.092 |           0.573 |    1.02        | Anecdotal evidence   |        1.017 |      0.85  |      1.218 |
| arthur_speaking:Group Interaction   |         0.014 |        0.082 |           0.569 |    1.02        | Anecdotal evidence   |        1.014 |      0.864 |      1.19  |
| lee_girl_verb:Group Interaction     |         0.013 |        0.214 |           0.523 |    1           | Anecdotal evidence   |        1.013 |      0.665 |      1.541 |

---

## 2. Group-Specific Effects

This table presents the feature effects separately for each group (Affair and Paranoia) as well as the difference between groups. Features are sorted by the absolute magnitude of the difference between groups.

| Feature           |   Affair Coef |   Affair OR |   Affair P(>0) |   Paranoia Coef |   Paranoia OR |   Paranoia P(>0) |   Diff (A-P) |   P(Stronger in Affair) |
|:------------------|--------------:|------------:|---------------:|----------------:|--------------:|-----------------:|-------------:|------------------------:|
| arthur_adj        |        -0.371 |       0.69  |          0.041 |           0.167 |         1.182 |            0.784 |       -0.269 |                   0.037 |
| girl_speaking     |        -0.79  |       0.454 |          0.038 |          -1.101 |         0.333 |            0.007 |        0.156 |                   0.689 |
| lee_girl_together |         0.24  |       1.271 |          0.807 |           0.039 |         1.04  |            0.556 |        0.101 |                   0.697 |
| has_adv           |         0.101 |       1.107 |          0.783 |          -0.067 |         0.935 |            0.302 |        0.084 |                   0.821 |
| has_adj           |        -0.319 |       0.727 |          0.008 |          -0.187 |         0.829 |            0.077 |       -0.066 |                   0.24  |
| has_noun          |         0.141 |       1.152 |          0.893 |           0.053 |         1.054 |            0.679 |        0.044 |                   0.709 |
| lee_speaking      |        -0.178 |       0.837 |          0.082 |          -0.243 |         0.784 |            0.028 |        0.033 |                   0.64  |
| has_verb          |        -0.086 |       0.917 |          0.225 |          -0.036 |         0.964 |            0.375 |       -0.025 |                   0.379 |
| arthur_speaking   |         0.514 |       1.672 |          1     |           0.485 |         1.625 |            1     |        0.014 |                   0.569 |
| lee_girl_verb     |         0.224 |       1.251 |          0.77  |           0.199 |         1.22  |            0.744 |        0.013 |                   0.523 |

---

## 3. Multiple Comparisons Analysis

This table presents the results of multiple comparisons correction using False Discovery Rate (FDR). Effects with FDR < 0.05 are considered credible.

| Effect                                                |   Posterior Probability |   FDR | Credible (FDR < 0.05)   |
|:------------------------------------------------------|------------------------:|------:|:------------------------|
| arthur_adj:group_arthur_adj_interaction               |                   0.963 | 0.037 | True                    |
| has_adv:group_has_adv_interaction                     |                   0.821 | 0.108 | False                   |
| has_adj:group_has_adj_interaction                     |                   0.76  | 0.152 | False                   |
| has_noun:group_has_noun_interaction                   |                   0.709 | 0.187 | False                   |
| lee_girl_together:group_lee_girl_together_interaction |                   0.697 | 0.21  | False                   |
| girl_speaking:group_girl_speaking_interaction         |                   0.689 | 0.227 | False                   |
| lee_speaking:group_lee_speaking_interaction           |                   0.64  | 0.246 | False                   |
| has_verb:group_has_verb_interaction                   |                   0.621 | 0.262 | False                   |
| arthur_speaking:group_arthur_speaking_interaction     |                   0.569 | 0.281 | False                   |
| lee_girl_verb:group_lee_girl_verb_interaction         |                   0.523 | 0.301 | False                   |

---

## 4. Cross-Validation Summary

This table presents the summary of cross-validation results, showing the stability of feature effects across subjects within each group.

| Interaction       |   Affair Mean |   Affair Std |   Paranoia Mean |   Paranoia Std |   Stability Ratio |
|:------------------|--------------:|-------------:|----------------:|---------------:|------------------:|
| arthur_speaking   |        0.0242 |       0.0135 |          0.0243 |         0.0135 |            0.9988 |
| lee_girl_together |        0.1054 |       0.0342 |          0.1071 |         0.035  |            0.9774 |
| has_verb          |       -0.0266 |       0.0092 |         -0.0265 |         0.016  |            0.5793 |

---

## 5. State Occupancy Rates

This table presents the state occupancy rates for each experimental group.

| Group    |   Occupancy Rate |
|:---------|-----------------:|
| Affair   |        0.134905  |
| Paranoia |        0.0679192 |

---

## 6. State Transition Rates

This table presents the state entry and exit rates for each experimental group.

| Group    |   Entry Rate |   Exit Rate |
|:---------|-------------:|------------:|
| Affair   |    0.0261988 |   0.0261988 |
| Paranoia |    0.0173099 |   0.0173099 |

---

## 7. Publication-Ready Main Effects

Publication-ready table of main effects and interactions. Significance: * BF > 3, ** BF > 10, *** BF > 100.

| Feature                             | Coef ± SE         |   P(Effect > 0) |   Bayes Factor |   Odds Ratio |
|:------------------------------------|:------------------|----------------:|---------------:|-------------:|
| arthur_speaking                     | 0.5 ± 0.082 ***   |           1     |    1.32804e+08 |        1.648 |
| girl_speaking                       | -0.946 ± 0.315 ** |           0.001 |   89.69        |        0.388 |
| has_adj                             | -0.253 ± 0.093 ** |           0.003 |   39.64        |        0.776 |
| lee_speaking                        | -0.211 ± 0.09 **  |           0.01  |   15.21        |        0.81  |
| arthur_adj:Group Interaction        | -0.269 ± 0.151 *  |           0.037 |    4.94        |        0.764 |
| has_noun                            | 0.097 ± 0.081     |           0.886 |    2.07        |        1.102 |
| lee_girl_verb                       | 0.211 ± 0.214     |           0.838 |    1.63        |        1.235 |
| has_adv:Group Interaction           | 0.084 ± 0.092     |           0.821 |    1.53        |        1.088 |
| has_verb                            | -0.061 ± 0.081    |           0.224 |    1.33        |        0.941 |
| lee_girl_together                   | 0.14 ± 0.196      |           0.762 |    1.29        |        1.15  |
| has_adj:Group Interaction           | -0.066 ± 0.093    |           0.24  |    1.28        |        0.937 |
| arthur_adj                          | -0.102 ± 0.151    |           0.249 |    1.26        |        0.903 |
| has_noun:Group Interaction          | 0.044 ± 0.08      |           0.709 |    1.16        |        1.045 |
| lee_girl_together:Group Interaction | 0.101 ± 0.195     |           0.697 |    1.14        |        1.106 |
| girl_speaking:Group Interaction     | 0.156 ± 0.315     |           0.689 |    1.13        |        1.168 |
| lee_speaking:Group Interaction      | 0.033 ± 0.09      |           0.64  |    1.07        |        1.033 |
| has_verb:Group Interaction          | -0.025 ± 0.081    |           0.379 |    1.05        |        0.975 |
| has_adv                             | 0.017 ± 0.092     |           0.573 |    1.02        |        1.017 |
| arthur_speaking:Group Interaction   | 0.014 ± 0.082     |           0.569 |    1.02        |        1.014 |
| lee_girl_verb:Group Interaction     | 0.013 ± 0.214     |           0.523 |    1           |        1.013 |

---

## 8. Publication-Ready Group Effects

Publication-ready table of group-specific effects. † indicates credible differences (FDR < 0.05).

| Feature           |   Affair Coef |   Paranoia Coef | Group Difference   |   P(Stronger in Affair) |
|:------------------|--------------:|----------------:|:-------------------|------------------------:|
| arthur_adj        |        -0.371 |           0.167 | -0.269 †           |                   0.037 |
| girl_speaking     |        -0.79  |          -1.101 | 0.156              |                   0.689 |
| lee_girl_together |         0.24  |           0.039 | 0.101              |                   0.697 |
| has_adv           |         0.101 |          -0.067 | 0.084              |                   0.821 |
| has_adj           |        -0.319 |          -0.187 | -0.066             |                   0.24  |
| has_noun          |         0.141 |           0.053 | 0.044              |                   0.709 |
| lee_speaking      |        -0.178 |          -0.243 | 0.033              |                   0.64  |
| has_verb          |        -0.086 |          -0.036 | -0.025             |                   0.379 |
| arthur_speaking   |         0.514 |           0.485 | 0.014              |                   0.569 |
| lee_girl_verb     |         0.224 |           0.199 | 0.013              |                   0.523 |