import os
import json
import uuid
import argparse

# Make sure image_generator is found in the same directory
from .image_generator import generate_circuit_image

# --- Configuration ---
DATASET_BASE_DIR = "datasets"
NUM_CASES_PER_N = 10 # Number of test cases to generate for each N value
N_VALUES_TO_GENERATE = list(range(2, 7)) # Generate for N=2, 3, 4, 5, 6
DEFAULT_TASK_TYPE = "identify_all" # Could add more types later
DEFAULT_EVAL_METHOD = "detailed_comparison"

def create_base_test_case(n_value: int, case_num: int, task_type: str, instruction: str, image_path: str, expected_outcome: dict, evaluation_method: str) -> dict:
    """Creates the basic structure for a test case JSON."""
    test_case_id = f"n{n_value}_{case_num:03d}_{task_type}"
    return {
        "test_case_id": test_case_id,
        "n_value": n_value,
        "task_type": task_type,
        "instruction": instruction,
        "image_path": image_path.replace("\\", "/"), # Ensure forward slashes for consistency
        "expected_outcome": expected_outcome,
        "evaluation_method": evaluation_method
    }

def write_test_case(test_case: dict, output_dir: str):
    """Writes a test case dictionary to its JSON file in the correct subdir."""
    file_path = os.path.join(output_dir, f"{test_case['test_case_id']}.json")
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(test_case, f, indent=4)
        # print(f"Successfully wrote {file_path}")
    except Exception as e:
        print(f"Error writing {file_path}: {e}")

def generate_dataset(n_values: list[int], num_per_n: int):
    """Generates the full dataset for the specified N values."""
    print(f"Starting dataset generation for N values: {n_values}")
    total_generated = 0
    total_failed = 0

    # Define instructions based on task type (expandable)
    instructions = {
        "identify_all": "Analyze the provided image. For each numbered element, identify the shape it points to, its color, and its type. Return the result as a structured JSON object mapping each number (as a string key) to an object containing 'shape_type' and 'color'.",
        # Add other task types here if needed
    }

    for n in n_values:
        print(f"--- Generating for N={n} ---")
        n_dir = os.path.join(DATASET_BASE_DIR, f"n{n}")
        images_dir = os.path.join(n_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        
        generated_for_n = 0
        failed_for_n = 0

        for i in range(num_per_n):
            case_num = i + 1
            task_type = DEFAULT_TASK_TYPE # For now, only one type
            instruction = instructions[task_type]
            
            test_case_base_id = f"n{n}_{case_num:03d}_{task_type}"
            image_filename = f"{test_case_base_id}.png"
            image_save_path = os.path.join(images_dir, image_filename)
            relative_image_path = os.path.join(f"n{n}", "images", image_filename)

            # Generate a deterministic seed based on N and case number
            # Using a simple combination, can be made more complex if needed
            current_seed = (n * 1000) + case_num 

            # Generate the image and get ground truth, passing the seed
            ground_truth = generate_circuit_image(N=n, image_save_path=image_save_path, seed=current_seed)

            if ground_truth:
                # Create the test case JSON
                test_case_json = create_base_test_case(
                    n_value=n,
                    case_num=case_num,
                    task_type=task_type,
                    instruction=instruction,
                    image_path=relative_image_path,
                    expected_outcome=ground_truth,
                    evaluation_method=DEFAULT_EVAL_METHOD
                )
                # Write the JSON file to the n{N} directory
                write_test_case(test_case_json, n_dir)
                generated_for_n += 1
            else:
                print(f"Failed to generate image for case {case_num} with N={n}")
                failed_for_n += 1
        
        print(f"Completed N={n}: {generated_for_n} generated, {failed_for_n} failed.")
        total_generated += generated_for_n
        total_failed += failed_for_n

    print(f"\nDataset generation complete.")
    print(f"Total Generated: {total_generated}")
    print(f"Total Failed:    {total_failed}")

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate the CeledonBenchCircuits dataset.")
    parser.add_argument("-n", "--n_values", type=int, nargs='+', default=N_VALUES_TO_GENERATE,
                        help=f"List of N values (number of shapes) to generate datasets for (default: {N_VALUES_TO_GENERATE})")
    parser.add_argument("--num_per_n", type=int, default=NUM_CASES_PER_N,
                        help=f"Number of test cases to generate for each N value (default: {NUM_CASES_PER_N})")
    
    args = parser.parse_args()
    
    generate_dataset(n_values=sorted(list(set(args.n_values))), num_per_n=args.num_per_n) 