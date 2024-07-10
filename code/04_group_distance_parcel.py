import os
import sys
import glob
import logging
import numpy as np
from utils import setup_logging

from scipy.spatial.distance import euclidean, cityblock, correlation
from scipy.stats import gaussian_kde, entropy
from sklearn.metrics import mutual_info_score
from dtaidistance import dtw

from dotenv import load_dotenv

def get_average(npy_files):
    """Calculate the average of numpy arrays loaded from the given files."""
    sum_array = None
    num_files = len(npy_files)
    
    if num_files == 0:
        logging.warning("No files were processed. Returning None.")
        return None
    
    for file in npy_files:
        try:
            data = np.load(file)
        except Exception as e:
            logging.error(f"Error loading {file}: {e}")
            continue

        if sum_array is None:
            sum_array = np.zeros_like(data)

        sum_array += data

    return sum_array / num_files

def calculate_and_save_average(npy_files, output_file):
    """Calculate and save the average array if it doesn't exist, or load it from a file."""
    if os.path.exists(output_file):
        logging.info(f"Loading existing average from {output_file}...")
        return np.load(output_file)
    else:
        logging.info(f"Calculating average for {output_file}...")
        avg_array = get_average(npy_files)
        if avg_array is not None:
            np.save(output_file, avg_array)
        return avg_array

def get_kl(a, b, num_points=1000, smoothing=1e-10):
    """Calculate the KL divergence between two time series."""
    kde_a = gaussian_kde(a)
    kde_b = gaussian_kde(b)

    min_val = min(min(a), min(b))
    max_val = max(max(a), max(b))
    x_eval = np.linspace(min_val, max_val, num_points)

    pdf_a = kde_a(x_eval) + smoothing
    pdf_b = kde_b(x_eval) + smoothing

    pdf_a /= np.sum(pdf_a)
    pdf_b /= np.sum(pdf_b)

    return entropy(pdf_a, pdf_b)

def get_mutinfo(a, b):
    a_discretized = np.digitize(a, bins=np.histogram_bin_edges(a, bins='auto'))
    b_discretized = np.digitize(b, bins=np.histogram_bin_edges(b, bins='auto'))

    mutual_info = mutual_info_score(a_discretized, b_discretized)
    return 1 - mutual_info

def calculate_distances(row_A, row_B):
    """Calculate various distances between two rows."""
    euclidean_distance = euclidean(row_A, row_B)
    dtw_distance = dtw.distance_fast(row_A, row_B)
    pearson_distance = 1 - correlation(row_A, row_B)
    manhattan_distance = cityblock(row_A, row_B)
    kl_distance = get_kl(row_A, row_B)
    mutual_info_distance = get_mutinfo(row_A, row_B)

    return euclidean_distance, dtw_distance, pearson_distance, manhattan_distance, kl_distance, mutual_info_distance

def normalize(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

def main(n_parcel, affair_coflt_files, paranoia_coflt_files, dtw_output_dir):
    affair_avg_file = os.path.join(dtw_output_dir, "affair_coflt_avg.npy")
    paranoia_avg_file = os.path.join(dtw_output_dir, "paranoia_coflt_avg.npy")

    affair_coflt_avg = calculate_and_save_average(affair_coflt_files, affair_avg_file)
    paranoia_coflt_avg = calculate_and_save_average(paranoia_coflt_files, paranoia_avg_file)

    if affair_coflt_avg is None or paranoia_coflt_avg is None:
        logging.error("One or both of the average arrays could not be calculated. Exiting.")
        return

    start, end = 14, 465
    A = affair_coflt_avg[:, :, start:end]
    B = paranoia_coflt_avg[:, :, start:end]

    # Initialize matrices to store distances
    distance_measures = ['euclidean', 'dtw', 'pearson', 'manhattan', 'kl', 'mutual_info']
    distance_matrices = {measure: np.zeros((n_parcel, n_parcel)) for measure in distance_measures}

    for i in range(n_parcel):
        for j in range(i, n_parcel):
            row_A = A[i, j, :]
            row_B = B[i, j, :]

            distances = calculate_distances(row_A, row_B)

            for measure, dist in zip(distance_measures, distances):
                distance_matrices[measure][i, j] = dist
                if i != j:
                    distance_matrices[measure][j, i] = dist

    np.savez_compressed(os.path.join(dtw_output_dir, "parcel_distances"), **distance_matrices)

    # Normalize and save distances
    normalized_distance_matrices = {measure: normalize(matrix) for measure, matrix in distance_matrices.items()}
    np.savez_compressed(os.path.join(dtw_output_dir, "parcel_distances_normalized"), **normalized_distance_matrices)

if __name__ == '__main__':
    load_dotenv()
    scratch_dir = os.getenv("SCRATCH_DIR")
    base_dir = os.getenv("BASE_DIR")
    output_dir = os.path.join(scratch_dir, "output")
    coflt_LOO = os.path.join(output_dir, "cofluctuation_LOO")
    
    try:
        n_parcel = int(sys.argv[1])
    except (IndexError, ValueError):
        logging.error("Please provide the number of parcels as a command line argument.")
        sys.exit(1)

    affair_coflt_LOO = os.path.join(coflt_LOO, f"{n_parcel}parcel", "affair")
    paranoia_coflt_LOO = os.path.join(coflt_LOO, f"{n_parcel}parcel", "paranoia")

    affair_coflt_files = sorted(glob.glob(os.path.join(affair_coflt_LOO, "*.npy")))
    paranoia_coflt_files = sorted(glob.glob(os.path.join(paranoia_coflt_LOO, "*.npy")))

    dtw_output_dir = os.path.join(output_dir, "group_distance", f"{n_parcel}parcel")
    os.makedirs(dtw_output_dir, exist_ok=True)

    setup_logging(base_dir=base_dir, task="group_distance", task_id=f"{n_parcel}parcel")

    main(n_parcel, affair_coflt_files, paranoia_coflt_files, dtw_output_dir)