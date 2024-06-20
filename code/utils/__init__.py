def read_output_path(file_path):
    with open(file_path, 'r') as file:
        return file.readline().strip()