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

| Feature                             |   Coefficient |   Std. Error |   P(Effect > 0) |   Bayes Factor | Evidence Category   |   Odds Ratio |   OR Lower |   OR Upper |
|:------------------------------------|--------------:|-------------:|----------------:|---------------:|:--------------------|-------------:|-----------:|-----------:|
| arthur_speaking                     |        -0.371 |        0.084 |           0     |       17990.2  | Extreme evidence    |        0.69  |      0.586 |      0.813 |
| has_noun:Group Interaction          |         0.132 |        0.075 |           0.96  |           4.67 | Moderate evidence   |        1.142 |      0.985 |      1.324 |
| arthur_speaking:Group Interaction   |        -0.142 |        0.084 |           0.044 |           4.25 | Moderate evidence   |        0.867 |      0.736 |      1.022 |
| has_noun                            |        -0.094 |        0.075 |           0.107 |           2.17 | Anecdotal evidence  |        0.91  |      0.785 |      1.055 |
| has_adv:Group Interaction           |        -0.104 |        0.086 |           0.113 |           2.08 | Anecdotal evidence  |        0.901 |      0.761 |      1.067 |
| has_adj:Group Interaction           |         0.103 |        0.087 |           0.881 |           2.01 | Anecdotal evidence  |        1.108 |      0.935 |      1.314 |
| has_verb                            |        -0.088 |        0.076 |           0.123 |           1.95 | Anecdotal evidence  |        0.915 |      0.788 |      1.063 |
| lee_girl_together                   |        -0.211 |        0.188 |           0.131 |           1.88 | Anecdotal evidence  |        0.809 |      0.559 |      1.171 |
| has_adj                             |         0.086 |        0.087 |           0.839 |           1.63 | Anecdotal evidence  |        1.09  |      0.919 |      1.293 |
| girl_speaking                       |        -0.202 |        0.205 |           0.162 |           1.63 | Anecdotal evidence  |        0.817 |      0.547 |      1.221 |
| arthur_adj:Group Interaction        |        -0.138 |        0.155 |           0.187 |           1.48 | Anecdotal evidence  |        0.871 |      0.643 |      1.181 |
| lee_speaking:Group Interaction      |         0.058 |        0.08  |           0.767 |           1.3  | Anecdotal evidence  |        1.06  |      0.906 |      1.24  |
| girl_speaking:Group Interaction     |        -0.14  |        0.205 |           0.248 |           1.26 | Anecdotal evidence  |        0.87  |      0.582 |      1.299 |
| lee_speaking                        |         0.034 |        0.08  |           0.666 |           1.1  | Anecdotal evidence  |        1.035 |      0.885 |      1.211 |
| lee_girl_together:Group Interaction |         0.08  |        0.188 |           0.663 |           1.09 | Anecdotal evidence  |        1.083 |      0.748 |      1.567 |
| lee_girl_verb:Group Interaction     |         0.063 |        0.202 |           0.622 |           1.05 | Anecdotal evidence  |        1.065 |      0.717 |      1.583 |
| has_adv                             |         0.026 |        0.086 |           0.619 |           1.05 | Anecdotal evidence  |        1.026 |      0.867 |      1.215 |
| arthur_adj                          |         0.039 |        0.155 |           0.599 |           1.03 | Anecdotal evidence  |        1.04  |      0.767 |      1.41  |
| lee_girl_verb                       |        -0.031 |        0.202 |           0.439 |           1.01 | Anecdotal evidence  |        0.969 |      0.652 |      1.44  |
| has_verb:Group Interaction          |        -0.01  |        0.076 |           0.446 |           1.01 | Anecdotal evidence  |        0.99  |      0.852 |      1.149 |

---

## 2. Group-Specific Effects

This table presents the feature effects separately for each group (Affair and Paranoia) as well as the difference between groups. Features are sorted by the absolute magnitude of the difference between groups.

| Feature           |   Affair Coef |   Affair OR |   Affair P(>0) |   Paranoia Coef |   Paranoia OR |   Paranoia P(>0) |   Diff (A-P) |   P(Stronger in Affair) |
|:------------------|--------------:|------------:|---------------:|----------------:|--------------:|-----------------:|-------------:|------------------------:|
| arthur_speaking   |        -0.513 |       0.599 |          0     |          -0.228 |         0.796 |            0.027 |       -0.142 |                   0.044 |
| girl_speaking     |        -0.342 |       0.71  |          0.119 |          -0.063 |         0.939 |            0.415 |       -0.14  |                   0.248 |
| arthur_adj        |        -0.099 |       0.906 |          0.327 |           0.177 |         1.193 |            0.79  |       -0.138 |                   0.187 |
| has_noun          |         0.039 |       1.039 |          0.641 |          -0.226 |         0.797 |            0.017 |        0.132 |                   0.96  |
| has_adv           |        -0.078 |       0.925 |          0.261 |           0.13  |         1.139 |            0.858 |       -0.104 |                   0.113 |
| has_adj           |         0.189 |       1.208 |          0.937 |          -0.016 |         0.984 |            0.447 |        0.103 |                   0.881 |
| lee_girl_together |        -0.132 |       0.876 |          0.31  |          -0.291 |         0.747 |            0.137 |        0.08  |                   0.663 |
| lee_girl_verb     |         0.032 |       1.033 |          0.545 |          -0.094 |         0.91  |            0.371 |        0.063 |                   0.622 |
| lee_speaking      |         0.093 |       1.097 |          0.794 |          -0.024 |         0.976 |            0.416 |        0.058 |                   0.767 |
| has_verb          |        -0.099 |       0.906 |          0.18  |          -0.078 |         0.925 |            0.234 |       -0.01  |                   0.446 |

---

## 3. Multiple Comparisons Analysis

This table presents the results of multiple comparisons correction using False Discovery Rate (FDR). Effects with FDR < 0.05 are considered credible.

| Effect                                                |   Posterior Probability |   FDR | Credible (FDR < 0.05)   |
|:------------------------------------------------------|------------------------:|------:|:------------------------|
| has_noun:group_has_noun_interaction                   |                   0.96  | 0.04  | True                    |
| arthur_speaking:group_arthur_speaking_interaction     |                   0.956 | 0.042 | True                    |
| has_adv:group_has_adv_interaction                     |                   0.887 | 0.066 | False                   |
| has_adj:group_has_adj_interaction                     |                   0.881 | 0.079 | False                   |
| arthur_adj:group_arthur_adj_interaction               |                   0.813 | 0.101 | False                   |
| lee_speaking:group_lee_speaking_interaction           |                   0.767 | 0.123 | False                   |
| girl_speaking:group_girl_speaking_interaction         |                   0.752 | 0.141 | False                   |
| lee_girl_together:group_lee_girl_together_interaction |                   0.663 | 0.165 | False                   |
| lee_girl_verb:group_lee_girl_verb_interaction         |                   0.622 | 0.189 | False                   |
| has_verb:group_has_verb_interaction                   |                   0.554 | 0.214 | False                   |

---

## 4. Cross-Validation Summary

This table presents the summary of cross-validation results, showing the stability of feature effects across subjects within each group.

| Interaction       |   Affair Mean |   Affair Std |   Paranoia Mean |   Paranoia Std |   Stability Ratio |
|:------------------|--------------:|-------------:|----------------:|---------------:|------------------:|
| arthur_speaking   |       -0.1394 |       0.0135 |         -0.1393 |         0.0129 |            0.9529 |
| has_verb          |       -0.0039 |       0.0144 |         -0.0038 |         0.0122 |            0.8481 |
| lee_girl_together |        0.0422 |       0.0268 |          0.0428 |         0.021  |            0.7832 |

---

## 5. State Occupancy Rates

This table presents the state occupancy rates for each experimental group.

| Group    |   Occupancy Rate |
|:---------|-----------------:|
| Affair   |        0.0956938 |
| Paranoia |        0.122768  |

---

## 6. State Transition Rates

This table presents the state entry and exit rates for each experimental group.

| Group    |   Entry Rate |   Exit Rate |
|:---------|-------------:|------------:|
| Affair   |    0.0225731 |   0.0214035 |
| Paranoia |    0.0266667 |   0.025614  |

---

## 7. Publication-Ready Main Effects

Publication-ready table of main effects and interactions. Significance: * BF > 3, ** BF > 10, *** BF > 100.

| Feature                             | Coef ± SE          |   P(Effect > 0) |   Bayes Factor |   Odds Ratio |
|:------------------------------------|:-------------------|----------------:|---------------:|-------------:|
| arthur_speaking                     | -0.371 ± 0.084 *** |           0     |       17990.2  |        0.69  |
| has_noun:Group Interaction          | 0.132 ± 0.075 *    |           0.96  |           4.67 |        1.142 |
| arthur_speaking:Group Interaction   | -0.142 ± 0.084 *   |           0.044 |           4.25 |        0.867 |
| has_noun                            | -0.094 ± 0.075     |           0.107 |           2.17 |        0.91  |
| has_adv:Group Interaction           | -0.104 ± 0.086     |           0.113 |           2.08 |        0.901 |
| has_adj:Group Interaction           | 0.103 ± 0.087      |           0.881 |           2.01 |        1.108 |
| has_verb                            | -0.088 ± 0.076     |           0.123 |           1.95 |        0.915 |
| lee_girl_together                   | -0.211 ± 0.188     |           0.131 |           1.88 |        0.809 |
| has_adj                             | 0.086 ± 0.087      |           0.839 |           1.63 |        1.09  |
| girl_speaking                       | -0.202 ± 0.205     |           0.162 |           1.63 |        0.817 |
| arthur_adj:Group Interaction        | -0.138 ± 0.155     |           0.187 |           1.48 |        0.871 |
| lee_speaking:Group Interaction      | 0.058 ± 0.08       |           0.767 |           1.3  |        1.06  |
| girl_speaking:Group Interaction     | -0.14 ± 0.205      |           0.248 |           1.26 |        0.87  |
| lee_speaking                        | 0.034 ± 0.08       |           0.666 |           1.1  |        1.035 |
| lee_girl_together:Group Interaction | 0.08 ± 0.188       |           0.663 |           1.09 |        1.083 |
| lee_girl_verb:Group Interaction     | 0.063 ± 0.202      |           0.622 |           1.05 |        1.065 |
| has_adv                             | 0.026 ± 0.086      |           0.619 |           1.05 |        1.026 |
| arthur_adj                          | 0.039 ± 0.155      |           0.599 |           1.03 |        1.04  |
| lee_girl_verb                       | -0.031 ± 0.202     |           0.439 |           1.01 |        0.969 |
| has_verb:Group Interaction          | -0.01 ± 0.076      |           0.446 |           1.01 |        0.99  |

---

## 8. Publication-Ready Group Effects

Publication-ready table of group-specific effects. † indicates credible differences (FDR < 0.05).

| Feature           |   Affair Coef |   Paranoia Coef | Group Difference   |   P(Stronger in Affair) |
|:------------------|--------------:|----------------:|:-------------------|------------------------:|
| arthur_speaking   |        -0.513 |          -0.228 | -0.142 †           |                   0.044 |
| girl_speaking     |        -0.342 |          -0.063 | -0.14              |                   0.248 |
| arthur_adj        |        -0.099 |           0.177 | -0.138             |                   0.187 |
| has_noun          |         0.039 |          -0.226 | 0.132 †            |                   0.96  |
| has_adv           |        -0.078 |           0.13  | -0.104             |                   0.113 |
| has_adj           |         0.189 |          -0.016 | 0.103              |                   0.881 |
| lee_girl_together |        -0.132 |          -0.291 | 0.08               |                   0.663 |
| lee_girl_verb     |         0.032 |          -0.094 | 0.063              |                   0.622 |
| lee_speaking      |         0.093 |          -0.024 | 0.058              |                   0.767 |
| has_verb          |        -0.099 |          -0.078 | -0.01              |                   0.446 |