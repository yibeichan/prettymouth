import os
import sys
import glob
import logging
import numpy as np
from scipy.spatial.distance import euclidean, correlation, cityblock
from dtaidistance import dtw
from utils import setup_logging
from dotenv import load_dotenv

def get_average(npy_files):
    """Calculate the average of numpy arrays loaded from the given files."""
    sum_array = None
    num_files = len(npy_files)
    
    for file in npy_files:
        try:
            data = np.load(file)
        except Exception as e:
            logging.error(f"Error loading {file}: {e}")
            continue

        if sum_array is None:
            sum_array = np.zeros_like(data)

        sum_array += data

    if num_files == 0:
        logging.warning("No files were processed. Returning None.")
        return None

    return sum_array / num_files

def calculate_and_save_average(npy_files, output_file):
    """Calculate and save the average array if it doesn't exist, or load it from a file."""
    if os.path.exists(output_file):
        logging.info(f"Loading existing average from {output_file}...")
        return np.load(output_file)
    else:
        logging.info(f"Calculating average for {output_file}...")
        avg_array = get_average(npy_files)
        np.save(output_file, avg_array)
        return avg_array

def get_diagonal(array_3d):
    """Extract the diagonal elements from the 3D array along the third dimension."""
    _, _, n_TR = array_3d.shape
    return np.array([np.diagonal(array_3d[:, :, i]) for i in range(n_TR)]).T

def calculate_distances(row_A, row_B):
    """Calculate various distances between two rows."""
    # Euclidean Distance using scipy
    euclidean_distance = euclidean(row_A, row_B)

    # DTW Distance using dtw library
    dtw_distance = dtw.distance_fast(row_A, row_B)
    
    # Pearson Correlation Distance using scipy
    pearson_distance = 1-correlation(row_A, row_B)

    # Manhattan Distance using scipy
    manhattan_distance = cityblock(row_A, row_B)

    return euclidean_distance, dtw_distance, pearson_distance, manhattan_distance

# Function to normalize an array using Min-Max scaling
def normalize(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

def main(n_parcel, affair_coflt_files, paranoia_coflt_files, dtw_output_dir):
    affair_avg_file = os.path.join(dtw_output_dir, "affair_coflt_avg.npy")
    paranoia_avg_file = os.path.join(dtw_output_dir, "paranoia_coflt_avg.npy")

    affair_coflt_avg = calculate_and_save_average(affair_coflt_files, affair_avg_file)
    paranoia_coflt_avg = calculate_and_save_average(paranoia_coflt_files, paranoia_avg_file)

    logging.info("Extracting diagonals...")
    affair_coflt_avg_diag = get_diagonal(affair_coflt_avg)
    paranoia_coflt_avg_diag = get_diagonal(paranoia_coflt_avg)

    start, end = 14, 465
    A = affair_coflt_avg_diag[:, start:end]
    B = paranoia_coflt_avg_diag[:, start:end]

    # Initialize arrays to store distances
    euclidean_distances = np.zeros(n_parcel)
    dtw_distances = np.zeros(n_parcel)
    pearson_distances = np.zeros(n_parcel)
    manhattan_distances = np.zeros(n_parcel)

    for i in range(n_parcel):
        row_A = A[i, :]
        row_B = B[i, :]

        euclidean_distances[i], dtw_distances[i], pearson_distances[i], manhattan_distances[i] = calculate_distances(row_A, row_B)

    np.savez_compressed(os.path.join(dtw_output_dir, "parcel_distances"), 
                        euclidean=euclidean_distances, dtw=dtw_distances, 
                        pearson=pearson_distances, manhattan=manhattan_distances)

    # Normalize the distances
    euclidean_distances_normalized = normalize(euclidean_distances)
    dtw_distances_normalized = normalize(dtw_distances)
    pearson_distances_normalized = normalize(pearson_distances)
    manhattan_distances_normalized = normalize(manhattan_distances)

    # save normalized distances
    np.savez_compressed(os.path.join(dtw_output_dir, "parcel_distances_normalized"), 
                        euclidean=euclidean_distances_normalized, dtw=dtw_distances_normalized, 
                        pearson=pearson_distances_normalized, manhattan=manhattan_distances_normalized)

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