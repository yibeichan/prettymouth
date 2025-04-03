from utils.result2md import process_data_to_markdown
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

base_dir = os.getenv("BASE_DIR")
scratch_dir = os.getenv("SCRATCH_DIR")
base_dir = os.getenv("BASE_DIR")
scratch_dir = os.getenv("SCRATCH_DIR")

brain_glmm_dir = os.path.join(scratch_dir, "output", "09_brain_content_glmm")
behav_glmm_dir = os.path.join(scratch_dir, "output", "09_behavior_content_glmm")
result_dir = Path(base_dir) / "results"
result_dir.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    cluster_id = 1
    model_id = 2
    file = os.path.join(brain_glmm_dir, f"cluster{cluster_id}_combined_{model_id}states_deviation_th085", "complete_results.npz")
    output_file = os.path.join(result_dir, f"cluster{cluster_id}_combined_{model_id}states_deviation_th085.md")
    process_data_to_markdown(file, output_file)

    cluster_id = 2
    model_id = 12
    file = os.path.join(brain_glmm_dir, f"cluster{cluster_id}_combined_{model_id}states_deviation_th085", "complete_results.npz")
    output_file = os.path.join(result_dir, f"cluster{cluster_id}_combined_{model_id}states_deviation_th085.md")
    process_data_to_markdown(file, output_file)

    cluster_id = 3
    model_id = 2
    file = os.path.join(brain_glmm_dir, f"cluster{cluster_id}_combined_{model_id}states_deviation_th085", "complete_results.npz")
    output_file = os.path.join(result_dir, f"cluster{cluster_id}_combined_{model_id}states_deviation_th085.md")
    process_data_to_markdown(file, output_file)

    cluster_id = 4
    model_id = 8
    file = os.path.join(brain_glmm_dir, f"cluster{cluster_id}_combined_{model_id}states_deviation_th085", "complete_results.npz")
    output_file = os.path.join(result_dir, f"cluster{cluster_id}_combined_{model_id}states_deviation_th085.md")
    process_data_to_markdown(file, output_file)

    json_file_path = os.path.join(behav_glmm_dir, "deviation", "behavioral_content_analysis.json")
    output_file = os.path.join(result_dir, "behavioral_content_analysis.md")
    process_data_to_markdown(json_file_path, output_file, "behavioral")
