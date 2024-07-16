import os
import logging

def setup_logging(base_dir, task, task_id):
    """
    Sets up logging for the given base directory, data input, and sub ID, and returns the log filename.
    
    Args:
        base_dir (str): The base directory for the log file.
        data_input (str): The data input for the log file.
        sub_id (str): The sub ID for the log file.
    
    Returns:
        str: The generated log filename.
    """
    log_filename = os.path.join(base_dir, 'logs', f'{task}_{task_id}.log')
    logging.basicConfig(filename=log_filename, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    return log_filename

def read_output_path(file_path):
    with open(file_path, 'r') as file:
        return file.readline().strip()