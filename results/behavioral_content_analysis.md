# Behavioral GLMM Results for Content Analysis

## Main Effects

This table presents the main effects for all features in the analysis.

| Feature           |   Coefficient |   Std.Error |   P(Effect>0) |   Bayes Factor |   Odds Ratio |   Lower CI |   Upper CI | Evidence           |
|:------------------|--------------:|------------:|--------------:|---------------:|-------------:|-----------:|-----------:|:-------------------|
| lee_speaking      |         0.588 |       0.09  |         1     |       1.49e+09 |        1.801 |      1.508 |      2.15  | Extreme evidence   |
| arthur_adj        |        -0.888 |       0.166 |         0     |       1.63e+06 |        0.411 |      0.297 |      0.57  | Extreme evidence   |
| has_noun          |         0.301 |       0.058 |         1     |  618000        |        1.351 |      1.205 |      1.515 | Extreme evidence   |
| arthur_speaking   |         1.009 |       0.093 |         1     |    1000        |        2.743 |      2.286 |      3.291 | Extreme evidence   |
| lee_girl_together |        -0.81  |       0.496 |         0.051 |       3.8      |        0.445 |      0.168 |      1.175 | Moderate evidence  |
| girl_speaking     |        -0.608 |       0.448 |         0.088 |       2.51     |        0.545 |      0.226 |      1.311 | Anecdotal evidence |
| has_verb          |         0.045 |       0.058 |         0.78  |       1.35     |        1.046 |      0.933 |      1.172 | Anecdotal evidence |
| has_adv           |         0.037 |       0.065 |         0.715 |       1.17     |        1.037 |      0.914 |      1.178 | Anecdotal evidence |
| has_adj           |         0.043 |       0.09  |         0.682 |       1.12     |        1.044 |      0.874 |      1.246 | Anecdotal evidence |
| lee_girl_verb     |        -0.059 |       0.565 |         0.458 |       1.01     |        0.943 |      0.311 |      2.855 | Anecdotal evidence |

## Group Interaction Effects

This table presents the interaction effects between group and each feature.

| Feature           |   Coefficient |   Std.Error |   P(Effect>0) |   Bayes Factor |   Odds Ratio |   Lower CI |   Upper CI | Evidence           |
|:------------------|--------------:|------------:|--------------:|---------------:|-------------:|-----------:|-----------:|:-------------------|
| arthur_speaking   |        -1.096 |       0.093 |         0     |        1000    |        0.334 |      0.279 |      0.401 | Extreme evidence   |
| has_noun          |        -0.21  |       0.058 |         0     |         657.01 |        0.811 |      0.723 |      0.909 | Extreme evidence   |
| has_adv           |        -0.206 |       0.065 |         0.001 |         157.93 |        0.814 |      0.717 |      0.924 | Extreme evidence   |
| girl_speaking     |         1.052 |       0.448 |         0.991 |          15.71 |        2.864 |      1.189 |      6.894 | Strong evidence    |
| lee_speaking      |        -0.207 |       0.09  |         0.011 |          13.85 |        0.813 |      0.681 |      0.97  | Strong evidence    |
| has_verb          |        -0.087 |       0.058 |         0.067 |           3.08 |        0.917 |      0.818 |      1.027 | Moderate evidence  |
| arthur_adj        |         0.144 |       0.166 |         0.807 |           1.46 |        1.155 |      0.834 |      1.599 | Anecdotal evidence |
| lee_girl_verb     |        -0.259 |       0.565 |         0.323 |           1.11 |        0.772 |      0.255 |      2.337 | Anecdotal evidence |
| lee_girl_together |         0.132 |       0.496 |         0.605 |           1.04 |        1.142 |      0.432 |      3.016 | Anecdotal evidence |
| has_adj           |        -0.007 |       0.09  |         0.471 |           1    |        0.993 |      0.832 |      1.186 | Anecdotal evidence |

## Group-Specific Effects

This table presents effects separately for each context group and their differences.

| Feature           |   Affair Coef |   Affair OR | Affair P(>0)   |   Paranoia Coef |   Paranoia OR | Paranoia P(>0)   |   Diff (A-P) | P(Stronger in Affair)   |
|:------------------|--------------:|------------:|:---------------|----------------:|--------------:|:-----------------|-------------:|:------------------------|
| arthur_speaking   |        -0.087 |       0.917 | 0.254          |           2.105 |         8.205 | >0.999           |       -1.096 | <0.001                  |
| girl_speaking     |         0.444 |       1.559 | 0.758          |          -1.66  |         0.19  | 0.004            |        1.052 | 0.991                   |
| lee_girl_verb     |        -0.318 |       0.727 | 0.345          |           0.2   |         1.222 | 0.599            |       -0.259 | 0.323                   |
| has_noun          |         0.091 |       1.095 | 0.865          |           0.511 |         1.667 | >0.999           |       -0.21  | <0.001                  |
| lee_speaking      |         0.381 |       1.463 | 0.999          |           0.796 |         2.216 | >0.999           |       -0.207 | 0.011                   |
| has_adv           |        -0.169 |       0.844 | 0.032          |           0.243 |         1.275 | 0.996            |       -0.206 | <0.001                  |
| arthur_adj        |        -0.744 |       0.475 | <0.001         |          -1.032 |         0.356 | <0.001           |        0.144 | 0.807                   |
| lee_girl_together |        -0.677 |       0.508 | 0.167          |          -0.942 |         0.39  | 0.089            |        0.132 | 0.605                   |
| has_verb          |        -0.042 |       0.959 | 0.303          |           0.132 |         1.141 | 0.946            |       -0.087 | 0.067                   |
| has_adj           |         0.036 |       1.037 | 0.611          |           0.049 |         1.05  | 0.650            |       -0.007 | 0.471                   |

## Multiple Comparisons Analysis

This table presents the results of multiple comparisons correction using False Discovery Rate (FDR).

| Effect                              |   Posterior Probability |   FDR | Credible (FDR < 0.05)   |
|:------------------------------------|------------------------:|------:|:------------------------|
| arthur_speaking                     |                   1     | 0     | True                    |
| group_arthur_speaking_interaction   |                   1     | 0     | True                    |
| lee_speaking                        |                   1     | 0     | True                    |
| arthur_adj                          |                   1     | 0     | True                    |
| has_noun                            |                   1     | 0     | True                    |
| group_has_noun_interaction          |                   1     | 0     | True                    |
| group_has_adv_interaction           |                   0.999 | 0     | True                    |
| group_girl_speaking_interaction     |                   0.991 | 0.001 | True                    |
| group_lee_speaking_interaction      |                   0.989 | 0.002 | True                    |
| lee_girl_together                   |                   0.949 | 0.007 | True                    |
| group_has_verb_interaction          |                   0.933 | 0.013 | True                    |
| girl_speaking                       |                   0.912 | 0.019 | True                    |
| group_arthur_adj_interaction        |                   0.807 | 0.032 | True                    |
| has_verb                            |                   0.78  | 0.046 | True                    |
| has_adv                             |                   0.715 | 0.062 | False                   |
| has_adj                             |                   0.682 | 0.078 | False                   |
| group_lee_girl_verb_interaction     |                   0.677 | 0.092 | False                   |
| group_lee_girl_together_interaction |                   0.605 | 0.109 | False                   |
| lee_girl_verb                       |                   0.542 | 0.127 | False                   |
| group_has_adj_interaction           |                   0.529 | 0.145 | False                   |

## Statistical Notation

- **Coefficient**: The log odds effect size from the GLMM model
- **Std.Error**: Posterior standard deviation of the coefficient
- **P(Effect>0)**: Posterior probability that the effect is positive
- **Bayes Factor**: Relative evidence in favor of the effect existing vs. not existing
- **Odds Ratio**: Exponentiated coefficient, representing the multiplicative effect on odds
- **Lower/Upper CI**: Lower and upper bounds of the 95% highest density interval for the odds ratio
- **Evidence**: Categorical interpretation of the Bayes factor strength
- **FDR**: False Discovery Rate, corrected for multiple comparisons
