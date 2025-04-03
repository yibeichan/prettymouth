# Brain State Dynamics Analysis Supplementary Material

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

## 1. Main Effects and Interactions

This table presents the main effects and group interactions for all features analyzed in the brain state dynamics study. Features are sorted by Bayes Factor (strongest evidence first).

| Feature                             |   Coefficient |   Std. Error |   P(Effect > 0) |   Bayes Factor | Evidence Category    |   Odds Ratio |   OR Lower |   OR Upper |
|:------------------------------------|--------------:|-------------:|----------------:|---------------:|:---------------------|-------------:|-----------:|-----------:|
| arthur_speaking                     |         0.214 |        0.059 |           1     |         729.83 | Extreme evidence     |        1.239 |      1.104 |      1.391 |
| has_verb                            |         0.15  |        0.057 |           0.996 |          33.65 | Very strong evidence |        1.162 |      1.04  |      1.298 |
| has_adj:Group Interaction           |        -0.142 |        0.063 |           0.012 |          12.6  | Strong evidence      |        0.867 |      0.766 |      0.982 |
| arthur_adj:Group Interaction        |        -0.166 |        0.108 |           0.063 |           3.24 | Moderate evidence    |        0.847 |      0.685 |      1.047 |
| lee_girl_verb                       |         0.21  |        0.149 |           0.921 |           2.71 | Anecdotal evidence   |        1.234 |      0.922 |      1.652 |
| lee_girl_together:Group Interaction |         0.156 |        0.131 |           0.883 |           2.03 | Anecdotal evidence   |        1.168 |      0.904 |      1.509 |
| lee_girl_together                   |         0.151 |        0.131 |           0.876 |           1.94 | Anecdotal evidence   |        1.163 |      0.9   |      1.503 |
| arthur_adj                          |         0.104 |        0.108 |           0.832 |           1.59 | Anecdotal evidence   |        1.11  |      0.898 |      1.372 |
| lee_girl_verb:Group Interaction     |         0.136 |        0.149 |           0.82  |           1.52 | Anecdotal evidence   |        1.146 |      0.856 |      1.533 |
| girl_speaking:Group Interaction     |        -0.107 |        0.151 |           0.239 |           1.29 | Anecdotal evidence   |        0.898 |      0.669 |      1.207 |
| has_adj                             |        -0.04  |        0.063 |           0.261 |           1.23 | Anecdotal evidence   |        0.96  |      0.849 |      1.087 |
| has_noun                            |        -0.033 |        0.055 |           0.275 |           1.2  | Anecdotal evidence   |        0.967 |      0.868 |      1.078 |
| arthur_speaking:Group Interaction   |         0.033 |        0.059 |           0.714 |           1.17 | Anecdotal evidence   |        1.034 |      0.921 |      1.161 |
| has_verb:Group Interaction          |         0.031 |        0.056 |           0.71  |           1.17 | Anecdotal evidence   |        1.032 |      0.924 |      1.152 |
| lee_speaking                        |        -0.029 |        0.06  |           0.314 |           1.12 | Anecdotal evidence   |        0.971 |      0.864 |      1.092 |
| has_adv                             |        -0.027 |        0.064 |           0.334 |           1.1  | Anecdotal evidence   |        0.973 |      0.859 |      1.102 |
| lee_speaking:Group Interaction      |        -0.018 |        0.06  |           0.382 |           1.05 | Anecdotal evidence   |        0.982 |      0.873 |      1.104 |
| has_adv:Group Interaction           |        -0.014 |        0.064 |           0.415 |           1.02 | Anecdotal evidence   |        0.987 |      0.871 |      1.117 |
| girl_speaking                       |        -0.012 |        0.151 |           0.468 |           1    | Anecdotal evidence   |        0.988 |      0.735 |      1.328 |
| has_noun:Group Interaction          |         0.003 |        0.055 |           0.524 |           1    | Anecdotal evidence   |        1.003 |      0.9   |      1.119 |

---

## 2. Group-Specific Effects

This table presents the feature effects separately for each group (Affair and Paranoia) as well as the difference between groups. Features are sorted by the absolute magnitude of the difference between groups.

| Feature           |   Affair Coef |   Affair OR |   Affair P(>0) |   Paranoia Coef |   Paranoia OR |   Paranoia P(>0) |   Diff (A-P) |   P(Stronger in Affair) |
|:------------------|--------------:|------------:|---------------:|----------------:|--------------:|-----------------:|-------------:|------------------------:|
| arthur_adj        |        -0.062 |       0.94  |          0.343 |           0.27  |         1.31  |            0.961 |       -0.166 |                   0.063 |
| lee_girl_together |         0.306 |       1.358 |          0.951 |          -0.005 |         0.995 |            0.49  |        0.156 |                   0.883 |
| has_adj           |        -0.183 |       0.833 |          0.02  |           0.102 |         1.107 |            0.873 |       -0.142 |                   0.012 |
| lee_girl_verb     |         0.346 |       1.414 |          0.95  |           0.074 |         1.077 |            0.638 |        0.136 |                   0.82  |
| girl_speaking     |        -0.119 |       0.888 |          0.288 |           0.095 |         1.1   |            0.672 |       -0.107 |                   0.239 |
| arthur_speaking   |         0.248 |       1.281 |          0.998 |           0.181 |         1.198 |            0.985 |        0.033 |                   0.714 |
| has_verb          |         0.181 |       1.198 |          0.988 |           0.119 |         1.126 |            0.932 |        0.031 |                   0.71  |
| lee_speaking      |        -0.047 |       0.954 |          0.289 |          -0.011 |         0.989 |            0.448 |       -0.018 |                   0.382 |
| has_adv           |        -0.041 |       0.96  |          0.325 |          -0.014 |         0.986 |            0.44  |       -0.014 |                   0.415 |
| has_noun          |        -0.03  |       0.971 |          0.352 |          -0.037 |         0.964 |            0.321 |        0.003 |                   0.524 |

---

## 3. Multiple Comparisons Analysis

This table presents the results of multiple comparisons correction using False Discovery Rate (FDR). Effects with FDR < 0.05 are considered credible.

| Effect                                                |   Posterior Probability |   FDR | Credible (FDR < 0.05)   |
|:------------------------------------------------------|------------------------:|------:|:------------------------|
| has_adj:group_has_adj_interaction                     |                   0.988 | 0.012 | True                    |
| arthur_adj:group_arthur_adj_interaction               |                   0.937 | 0.037 | True                    |
| lee_girl_together:group_lee_girl_together_interaction |                   0.883 | 0.064 | False                   |
| lee_girl_verb:group_lee_girl_verb_interaction         |                   0.82  | 0.093 | False                   |
| girl_speaking:group_girl_speaking_interaction         |                   0.761 | 0.122 | False                   |
| arthur_speaking:group_arthur_speaking_interaction     |                   0.714 | 0.149 | False                   |
| has_verb:group_has_verb_interaction                   |                   0.71  | 0.17  | False                   |
| lee_speaking:group_lee_speaking_interaction           |                   0.618 | 0.196 | False                   |
| has_adv:group_has_adv_interaction                     |                   0.585 | 0.22  | False                   |
| has_noun:group_has_noun_interaction                   |                   0.524 | 0.246 | False                   |

---

## 4. Cross-Validation Summary

This table presents the summary of cross-validation results, showing the stability of feature effects across subjects within each group.

| Interaction       |   Affair Mean |   Affair Std |   Paranoia Mean |   Paranoia Std |   Stability Ratio |
|:------------------|--------------:|-------------:|----------------:|---------------:|------------------:|
| lee_girl_together |        0.1646 |       0.0225 |          0.1646 |         0.0206 |            0.9151 |
| has_verb          |        0.0233 |       0.0086 |          0.0233 |         0.0072 |            0.8293 |
| arthur_speaking   |        0.0434 |       0.0074 |          0.0434 |         0.006  |            0.8072 |

---

## 5. State Occupancy Rates

This table presents the state occupancy rates for each experimental group.

| Group    |   Occupancy Rate |
|:---------|-----------------:|
| Affair   |         0.618158 |
| Paranoia |         0.536469 |

---

## 6. State Transition Rates

This table presents the state entry and exit rates for each experimental group.

| Group    |   Entry Rate |   Exit Rate |
|:---------|-------------:|------------:|
| Affair   |    0.0410526 |   0.040117  |
| Paranoia |    0.0425731 |   0.0421053 |

---

## 7. Publication-Ready Main Effects

Publication-ready table of main effects and interactions. Significance: * BF > 3, ** BF > 10, *** BF > 100.

| Feature                             | Coef ± SE         |   P(Effect > 0) |   Bayes Factor |   Odds Ratio |
|:------------------------------------|:------------------|----------------:|---------------:|-------------:|
| arthur_speaking                     | 0.214 ± 0.059 *** |           1     |         729.83 |        1.239 |
| has_verb                            | 0.15 ± 0.057 **   |           0.996 |          33.65 |        1.162 |
| has_adj:Group Interaction           | -0.142 ± 0.063 ** |           0.012 |          12.6  |        0.867 |
| arthur_adj:Group Interaction        | -0.166 ± 0.108 *  |           0.063 |           3.24 |        0.847 |
| lee_girl_verb                       | 0.21 ± 0.149      |           0.921 |           2.71 |        1.234 |
| lee_girl_together:Group Interaction | 0.156 ± 0.131     |           0.883 |           2.03 |        1.168 |
| lee_girl_together                   | 0.151 ± 0.131     |           0.876 |           1.94 |        1.163 |
| arthur_adj                          | 0.104 ± 0.108     |           0.832 |           1.59 |        1.11  |
| lee_girl_verb:Group Interaction     | 0.136 ± 0.149     |           0.82  |           1.52 |        1.146 |
| girl_speaking:Group Interaction     | -0.107 ± 0.151    |           0.239 |           1.29 |        0.898 |
| has_adj                             | -0.04 ± 0.063     |           0.261 |           1.23 |        0.96  |
| has_noun                            | -0.033 ± 0.055    |           0.275 |           1.2  |        0.967 |
| arthur_speaking:Group Interaction   | 0.033 ± 0.059     |           0.714 |           1.17 |        1.034 |
| has_verb:Group Interaction          | 0.031 ± 0.056     |           0.71  |           1.17 |        1.032 |
| lee_speaking                        | -0.029 ± 0.06     |           0.314 |           1.12 |        0.971 |
| has_adv                             | -0.027 ± 0.064    |           0.334 |           1.1  |        0.973 |
| lee_speaking:Group Interaction      | -0.018 ± 0.06     |           0.382 |           1.05 |        0.982 |
| has_adv:Group Interaction           | -0.014 ± 0.064    |           0.415 |           1.02 |        0.987 |
| girl_speaking                       | -0.012 ± 0.151    |           0.468 |           1    |        0.988 |
| has_noun:Group Interaction          | 0.003 ± 0.055     |           0.524 |           1    |        1.003 |

---

## 8. Publication-Ready Group Effects

Publication-ready table of group-specific effects. † indicates credible differences (FDR < 0.05).

| Feature           |   Affair Coef |   Paranoia Coef | Group Difference   |   P(Stronger in Affair) |
|:------------------|--------------:|----------------:|:-------------------|------------------------:|
| arthur_adj        |        -0.062 |           0.27  | -0.166 †           |                   0.063 |
| lee_girl_together |         0.306 |          -0.005 | 0.156              |                   0.883 |
| has_adj           |        -0.183 |           0.102 | -0.142 †           |                   0.012 |
| lee_girl_verb     |         0.346 |           0.074 | 0.136              |                   0.82  |
| girl_speaking     |        -0.119 |           0.095 | -0.107             |                   0.239 |
| arthur_speaking   |         0.248 |           0.181 | 0.033              |                   0.714 |
| has_verb          |         0.181 |           0.119 | 0.031              |                   0.71  |
| lee_speaking      |        -0.047 |          -0.011 | -0.018             |                   0.382 |
| has_adv           |        -0.041 |          -0.014 | -0.014             |                   0.415 |
| has_noun          |        -0.03  |          -0.037 | 0.003              |                   0.524 |