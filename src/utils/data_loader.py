import os
import json
import glob

DATASETS_DIR = "datasets"

def load_single_test_case(test_case_id: str) -> dict | None:
    """Loads a single test case JSON file by its ID, searching all dataset subdirs."""
    # Extract N from the test_case_id (e.g., "n2_001_...")
    try:
        n_str = test_case_id.split('_')[0] # Assumes format like "nX_..."
        dataset_subdir = os.path.join(DATASETS_DIR, n_str)
        file_path = os.path.join(dataset_subdir, f"{test_case_id}.json")
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except (IndexError, FileNotFoundError):
        # Fallback: search all N subdirectories if direct path fails or ID format is unexpected
        print(f"Warning: Could not directly find {test_case_id} in expected path. Searching all dataset dirs.")
        for n_dir in os.listdir(DATASETS_DIR):
            potential_path = os.path.join(DATASETS_DIR, n_dir, f"{test_case_id}.json")
            if os.path.isfile(potential_path):
                try:
                    with open(potential_path, 'r', encoding='utf-8') as f:
                        return json.load(f)
                except Exception as e:
                    print(f"Error loading {potential_path}: {e}")

    print(f"Test case {test_case_id} not found.")
    return None


def load_test_cases_for_n(n_value: int) -> list[dict]:
    """Loads all test case JSON files for a specific N value."""
    dataset_subdir = os.path.join(DATASETS_DIR, f"n{n_value}")
    test_cases = []
    if not os.path.isdir(dataset_subdir):
        print(f"Dataset directory not found for N={n_value}: {dataset_subdir}")
        return test_cases

    json_files = glob.glob(os.path.join(dataset_subdir, "*.json"))
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                test_case = json.load(f)
                test_cases.append(test_case)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {file_path}: {e}")
        except Exception as e:
            print(f"Error loading test case from {file_path}: {e}")

    print(f"Loaded {len(test_cases)} test cases for N={n_value}")
    return test_cases

def get_all_available_n_values() -> list[int]:
    """Scans the datasets directory and returns a sorted list of found N values."""
    n_values = []
    if not os.path.isdir(DATASETS_DIR):
        return []
        
    for item in os.listdir(DATASETS_DIR):
        if os.path.isdir(os.path.join(DATASETS_DIR, item)) and item.startswith('n'):
            try:
                n_val = int(item[1:])
                n_values.append(n_val)
            except ValueError:
                print(f"Warning: Skipping directory with non-integer N value: {item}")
                
    return sorted(n_values) 