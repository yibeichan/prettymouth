# Behavioral GLMM Results for Content Analysis

## Main Effects

This table presents the main effects for all features in the analysis.

| Feature           |   Coefficient |   Std.Error |   P(Effect>0) |   Bayes Factor |   Odds Ratio |   Lower CI |   Upper CI | Evidence           |
|:------------------|--------------:|------------:|--------------:|---------------:|-------------:|-----------:|-----------:|:-------------------|
| lee_speaking      |         0.453 |       0.087 |         1     |      659000    |        1.572 |      1.325 |      1.866 | Extreme evidence   |
| arthur_adj        |        -0.912 |       0.201 |         0     |       28700    |        0.402 |      0.271 |      0.596 | Extreme evidence   |
| arthur_speaking   |         0.882 |       0.085 |         1     |        1000    |        2.416 |      2.047 |      2.851 | Extreme evidence   |
| has_noun          |         0.199 |       0.058 |         1     |         340.52 |        1.22  |      1.088 |      1.367 | Extreme evidence   |
| lee_girl_together |        -0.982 |       0.443 |         0.013 |          11.71 |        0.375 |      0.157 |      0.892 | Strong evidence    |
| has_verb          |        -0.084 |       0.057 |         0.07  |           2.96 |        0.92  |      0.823 |      1.028 | Anecdotal evidence |
| girl_speaking     |        -0.614 |       0.452 |         0.087 |           2.51 |        0.541 |      0.223 |      1.313 | Anecdotal evidence |
| has_adv           |         0.075 |       0.069 |         0.861 |           1.8  |        1.078 |      0.941 |      1.234 | Anecdotal evidence |
| lee_girl_verb     |         0.167 |       0.547 |         0.62  |           1.05 |        1.182 |      0.405 |      3.454 | Anecdotal evidence |
| has_adj           |         0     |       0.104 |         0.502 |           1    |        1     |      0.816 |      1.227 | Anecdotal evidence |

## Group Interaction Effects

This table presents the interaction effects between group and each feature.

| Feature           |   Coefficient |   Std.Error |   P(Effect>0) |   Bayes Factor |   Odds Ratio |   Lower CI |   Upper CI | Evidence           |
|:------------------|--------------:|------------:|--------------:|---------------:|-------------:|-----------:|-----------:|:-------------------|
| arthur_speaking   |        -1.134 |       0.085 |         0     |        1000    |        0.322 |      0.273 |      0.38  | Extreme evidence   |
| has_adv           |        -0.169 |       0.069 |         0.007 |          19.8  |        0.845 |      0.737 |      0.967 | Strong evidence    |
| has_noun          |        -0.131 |       0.058 |         0.012 |          12.55 |        0.877 |      0.783 |      0.983 | Strong evidence    |
| girl_speaking     |         0.952 |       0.452 |         0.982 |           9.2  |        2.591 |      1.069 |      6.283 | Moderate evidence  |
| lee_girl_verb     |        -0.615 |       0.547 |         0.13  |           1.88 |        0.541 |      0.185 |      1.58  | Anecdotal evidence |
| lee_girl_together |         0.495 |       0.443 |         0.868 |           1.87 |        1.641 |      0.689 |      3.908 | Anecdotal evidence |
| has_adj           |        -0.103 |       0.104 |         0.161 |           1.63 |        0.902 |      0.736 |      1.106 | Anecdotal evidence |
| has_verb          |        -0.043 |       0.057 |         0.225 |           1.33 |        0.958 |      0.857 |      1.071 | Anecdotal evidence |
| lee_speaking      |        -0.059 |       0.087 |         0.248 |           1.26 |        0.942 |      0.794 |      1.118 | Anecdotal evidence |
| arthur_adj        |         0.085 |       0.201 |         0.664 |           1.09 |        1.089 |      0.734 |      1.615 | Anecdotal evidence |

## Group-Specific Effects

This table presents effects separately for each context group and their differences.

| Feature           |   Affair Coef |   Affair OR | Affair P(>0)   |   Paranoia Coef |   Paranoia OR | Paranoia P(>0)   |   Diff (A-P) | P(Stronger in Affair)   |
|:------------------|--------------:|------------:|:---------------|----------------:|--------------:|:-----------------|-------------:|:------------------------|
| arthur_speaking   |        -0.252 |       0.777 | 0.018          |           2.016 |         7.511 | >0.999           |       -1.134 | <0.001                  |
| girl_speaking     |         0.338 |       1.403 | 0.702          |          -1.566 |         0.209 | 0.007            |        0.952 | 0.982                   |
| lee_girl_verb     |        -0.448 |       0.639 | 0.281          |           0.783 |         2.187 | 0.844            |       -0.615 | 0.130                   |
| lee_girl_together |        -0.487 |       0.615 | 0.219          |          -1.477 |         0.228 | 0.009            |        0.495 | 0.868                   |
| has_adv           |        -0.094 |       0.91  | 0.168          |           0.244 |         1.276 | 0.994            |       -0.169 | 0.007                   |
| has_noun          |         0.068 |       1.07  | 0.795          |           0.33  |         1.39  | >0.999           |       -0.131 | 0.012                   |
| has_adj           |        -0.103 |       0.903 | 0.243          |           0.103 |         1.109 | 0.759            |       -0.103 | 0.161                   |
| arthur_adj        |        -0.827 |       0.438 | 0.002          |          -0.997 |         0.369 | <0.001           |        0.085 | 0.664                   |
| lee_speaking      |         0.393 |       1.482 | >0.999         |           0.512 |         1.669 | >0.999           |       -0.059 | 0.248                   |
| has_verb          |        -0.126 |       0.881 | 0.058          |          -0.041 |         0.96  | 0.305            |       -0.043 | 0.225                   |

## Multiple Comparisons Analysis

This table presents the results of multiple comparisons correction using False Discovery Rate (FDR).

| Effect                              |   Posterior Probability |   FDR | Credible (FDR < 0.05)   |
|:------------------------------------|------------------------:|------:|:------------------------|
| arthur_speaking                     |                   1     | 0     | True                    |
| group_arthur_speaking_interaction   |                   1     | 0     | True                    |
| lee_speaking                        |                   1     | 0     | True                    |
| arthur_adj                          |                   1     | 0     | True                    |
| has_noun                            |                   1     | 0     | True                    |
| group_has_adv_interaction           |                   0.993 | 0.001 | True                    |
| group_has_noun_interaction          |                   0.988 | 0.003 | True                    |
| lee_girl_together                   |                   0.987 | 0.004 | True                    |
| group_girl_speaking_interaction     |                   0.982 | 0.006 | True                    |
| has_verb                            |                   0.93  | 0.012 | True                    |
| girl_speaking                       |                   0.913 | 0.019 | True                    |
| group_lee_girl_verb_interaction     |                   0.87  | 0.028 | True                    |
| group_lee_girl_together_interaction |                   0.868 | 0.036 | True                    |
| has_adv                             |                   0.861 | 0.044 | True                    |
| group_has_adj_interaction           |                   0.839 | 0.051 | False                   |
| group_has_verb_interaction          |                   0.775 | 0.062 | False                   |
| group_lee_speaking_interaction      |                   0.752 | 0.073 | False                   |
| group_arthur_adj_interaction        |                   0.664 | 0.088 | False                   |
| lee_girl_verb                       |                   0.62  | 0.103 | False                   |
| has_adj                             |                   0.502 | 0.123 | False                   |

## Statistical Notation

- **Coefficient**: The log odds effect size from the GLMM model
- **Std.Error**: Posterior standard deviation of the coefficient
- **P(Effect>0)**: Posterior probability that the effect is positive
- **Bayes Factor**: Relative evidence in favor of the effect existing vs. not existing
- **Odds Ratio**: Exponentiated coefficient, representing the multiplicative effect on odds
- **Lower/Upper CI**: Lower and upper bounds of the 95% highest density interval for the odds ratio
- **Evidence**: Categorical interpretation of the Bayes factor strength
- **FDR**: False Discovery Rate, corrected for multiple comparisons
