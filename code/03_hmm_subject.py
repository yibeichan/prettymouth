# subject_level_hmm.py
import numpy as np
from hmmlearn import hmm
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Step 1: Generate Pseudo Data (fMRI time series for multiple subjects)
n_subjects = 5  # Number of subjects
n_regions = 264  # Number of brain regions
n_timepoints = 120  # Number of time points (TRs)

# Generate pseudo fMRI data for each subject (normally distributed)
np.random.seed(42)
subject_data = [np.random.randn(n_timepoints, n_regions) for _ in range(n_subjects)]

# Step 2: Subject-Level HMM with Model Selection
subject_hmms = []  # List to store trained HMMs for each subject

for i, data in enumerate(subject_data):
    # Split data into training and validation sets for cross-validation
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
    
    best_model = None
    best_bic = np.inf
    n_states_range = range(2, 10)  # Range of hidden states to try
    
    # Train HMMs with different numbers of hidden states and select the best model using BIC
    for n_states in n_states_range:
        model = hmm.GaussianHMM(n_components=n_states, covariance_type="full", n_iter=100, random_state=42)
        model.fit(train_data)
        bic = model.score(val_data)  # BIC as a measure of model quality
        if bic < best_bic:
            best_bic = bic
            best_model = model
    
    subject_hmms.append(best_model)
    
    # Predict the most likely sequence of hidden states
    hidden_states = best_model.predict(data)
    print(f"Subject {i + 1}: Hidden States Sequence\n", hidden_states)
    
    # Calculate and store mean vectors for each state
    mean_vectors = []
    for state in range(best_model.n_components):
        state_data = data[hidden_states == state]
        if len(state_data) > 0:
            mean_vectors.append(np.mean(state_data, axis=0))
        else:
            mean_vectors.append(np.zeros(n_regions))  # Handle empty states if necessary
    
    subject_hmms[i].mean_vectors = np.array(mean_vectors)

# End of subject_level_hmm.py