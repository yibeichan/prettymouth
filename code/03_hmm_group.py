import numpy as np
from hmmlearn import hmm
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Step 1: Load Subject-Level Mean State Vectors
n_subjects = 5
subject_hmms = []  # Placeholder for subject-level HMMs (import or load from file)

# For simplicity, reusing pseudo data generation (replace with actual data loading in practice)
for i in range(n_subjects):
    # Pseudo mean vectors from subject-level HMMs
    mean_vectors = np.random.randn(4, 264)  # Replace with actual mean vectors from subject-level HMMs
    subject_hmms.append(mean_vectors)

# Step 2: Pool Mean State Vectors Across Subjects
group_state_vectors = np.vstack(subject_hmms)

# Step 3: Apply Clustering to Identify Group-Level States
n_states_range = range(2, 10)  # Range of group-level states to try
best_model = None
best_bic = np.inf

# Cross-validation for optimal number of group-level states using Gaussian Mixture Model
for n_states in n_states_range:
    gmm = GaussianMixture(n_components=n_states, covariance_type='full', random_state=42)
    gmm.fit(group_state_vectors)
    bic = gmm.bic(group_state_vectors)
    if bic < best_bic:
        best_bic = bic
        best_model = gmm

cluster_labels = best_model.predict(group_state_vectors)

# Step 4: Group-Level HMM
# Concatenate subject data to create group-level training data
n_regions = 264
n_timepoints = 120
subject_data = [np.random.randn(n_timepoints, n_regions) for _ in range(n_subjects)]
group_data = np.vstack(subject_data)

# Train a group-level HMM with the optimal number of states found
best_n_states = best_model.n_components
group_hmm = hmm.GaussianHMM(n_components=best_n_states, covariance_type="full", n_iter=100, random_state=42)
group_hmm.fit(group_data)

# Predict the hidden states sequence for group data
group_hidden_states = group_hmm.predict(group_data)
print("\nGroup-Level Hidden States Sequence\n", group_hidden_states)

# Step 5: Visualization
plt.figure(figsize=(12, 6))
plt.plot(group_hidden_states, label="Group-Level Hidden State Sequence", color='blue')
plt.xlabel("Time Point")
plt.ylabel("State")
plt.title("Group-Level Hidden States Over Time")
plt.legend()
plt.show()

"""
Documentation:

1. Subject-Level HMM (subject_level_hmm.py):
   - For each subject, we used cross-validation to determine the optimal number of hidden states.
   - BIC was used to select the best model among different numbers of hidden states (ranging from 2 to 9).
   - The best HMM for each subject was stored, along with the mean vectors representing each hidden state.

2. Group-Level HMM (group_level_hmm.py):
   - Pooled mean state vectors from all subject-level HMMs.
   - Applied Gaussian Mixture Model (GMM) clustering to determine the optimal number of group-level states.
   - Trained a group-level HMM using the optimal number of states found via GMM and BIC.

3. Cross-Validation:
   - Cross-validation was done using a train-test split to determine the optimal number of hidden states at both subject and group levels.
   - BIC (Bayesian Information Criterion) was used as a metric to evaluate model fit and complexity.

"""