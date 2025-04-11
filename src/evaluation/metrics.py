import os
import json
import glob
import argparse
import traceback # Added for detailed error reporting
from typing import Any, Dict, Tuple, List

# Project-specific imports
from src.utils.data_loader import load_single_test_case

# --- Evaluation Functions ---

def parse_llm_output(llm_output_str: str | None) -> Tuple[Dict[str, Any] | None, str | None]:
    """Parses the LLM output string, expecting JSON. Returns (parsed_dict, error_message)."""
    if not llm_output_str:
        return None, "LLM output string is missing or empty."
    
    try:
        # The string should already be cleaned and verified JSON by the wrapper's _parse_response
        parsed_output = json.loads(llm_output_str)
        if not isinstance(parsed_output, dict):
             # If the top level isn't a dict, it doesn't match the expected format
             return None, f"LLM output is not a JSON object (dictionary). Found type: {type(parsed_output)}"
        return parsed_output, None
    except json.JSONDecodeError as e:
        return None, f"Failed to decode LLM output JSON: {e}. Output: {llm_output_str[:200]}..."
    except Exception as e:
        return None, f"Unexpected error parsing LLM output: {type(e).__name__}: {e}"

def compare_identify_all_task(llm_result: Dict[str, Any], ground_truth: Dict[str, Any]) -> Tuple[float, Dict[str, str]]:
    """Compares LLM output against ground truth for the 'identify_all' task.

    Args:
        llm_result: The parsed JSON dictionary from the LLM.
        ground_truth: The 'expected_outcome' dictionary from the test case JSON.

    Returns:
        A tuple containing:
            - pass_rate (float): 0.0 (fail) or 1.0 (pass) based on exact match.
            - details (dict): Breakdown of comparison results (e.g., correct_items, incorrect_items, missing_items).
    """
    gt_shapes = {s['id']: s for s in ground_truth.get('shapes', [])}
    gt_connections = {str(k): v for k, v in ground_truth.get('connections', {}).items()} # Ensure keys are strings
    n_value = ground_truth.get("n_value", 0)

    correct_count = 0
    incorrect_shapes_count = 0 # New counter
    incorrect_colors_count = 0 # New counter
    items_with_errors = set()  # Track items (numbers) with any error
    incorrect_details_str = {} # Keep detailed string messages per number
    missing_numbers = set(gt_connections.keys())
    extra_numbers = set()

    if not isinstance(llm_result, dict):
        # Handle cases where parsing yielded something unexpected (though parse_llm_output should catch JSON errors)
        return 0.0, {"error": "LLM output was not a valid dictionary.", "raw_output_preview": str(llm_result)[:200]}

    for num_str, llm_answer in llm_result.items():
        item_had_error = False # Flag for this specific number
        if not num_str.isdigit():
             incorrect_details_str[f"invalid_key_{num_str}"] = f"Key '{num_str}' is not a valid number string."
             extra_numbers.add(num_str) # Treat as extra if key invalid
             items_with_errors.add(num_str) # Count extra keys as items with errors
             continue

        if num_str in missing_numbers:
            missing_numbers.remove(num_str)
        else:
            extra_numbers.add(num_str)
            incorrect_details_str[f"extra_key_{num_str}"] = f"Number '{num_str}' was not expected."
            items_with_errors.add(num_str) # Count extra keys as items with errors
            continue # Don't process extra keys further

        if not isinstance(llm_answer, dict):
            incorrect_details_str[num_str] = f"Value for number '{num_str}' is not a dictionary."
            items_with_errors.add(num_str)
            item_had_error = True
            continue

        # Get ground truth shape details for this number
        gt_shape_id = gt_connections.get(num_str)
        if not gt_shape_id or gt_shape_id not in gt_shapes:
             incorrect_details_str[num_str] = f"Internal Error: Ground truth mapping missing for number '{num_str}'."
             items_with_errors.add(num_str) # Count internal error as item with error
             item_had_error = True
             continue 
        gt_shape_details = gt_shapes[gt_shape_id]

        # Check required fields in LLM answer
        llm_shape_type = llm_answer.get('shape_type')
        llm_color = llm_answer.get('color')
        # llm_shape_id is no longer requested or checked
        # llm_shape_id = llm_answer.get('shape_id')

        errors_for_item = [] # Errors specific to this number (for string message)
        
        # Check Shape Type
        if llm_shape_type is None:
            errors_for_item.append("Missing 'shape_type' field.")
            incorrect_shapes_count += 1 # Count missing as incorrect shape
            item_had_error = True
        elif llm_shape_type != gt_shape_details.get('type'):
            errors_for_item.append(f"Incorrect shape_type: Expected '{gt_shape_details.get('type')}', Got '{llm_shape_type}'.")
            incorrect_shapes_count += 1
            item_had_error = True
            
        # Check Color
        if llm_color is None:
            errors_for_item.append("Missing 'color' field.")
            incorrect_colors_count += 1 # Count missing as incorrect color
            item_had_error = True
        elif llm_color != gt_shape_details.get('color'):
            errors_for_item.append(f"Incorrect color: Expected '{gt_shape_details.get('color')}', Got '{llm_color}'.")
            incorrect_colors_count += 1
            item_had_error = True

        # Removed optional check for shape ID
        # if llm_shape_id is None:
        #     errors.append("Missing 'shape_id' field.") 
        # elif llm_shape_id != gt_shape_id:
        #     errors.append(f"Incorrect shape_id: Expected '{gt_shape_id}', Got '{llm_shape_id}'.")

        if not item_had_error:
            correct_count += 1
        else:
            items_with_errors.add(num_str)
            incorrect_details_str[num_str] = " | ".join(errors_for_item)

    # Add details about missing numbers
    if missing_numbers:
        incorrect_details_str["missing_numbers"] = f"LLM output did not include expected numbers: {sorted(list(missing_numbers))}"
        # Add missing numbers to the set of items with errors
        for missing_num in missing_numbers:
            items_with_errors.add(missing_num)
            
    # Determine pass/fail: For exact match, all N items must be correct, no missing, no extras.
    is_pass = (correct_count == n_value) and not missing_numbers and not extra_numbers
    pass_rate = 1.0 if is_pass else 0.0

    details = {
        "total_expected": n_value,
        "correct_items": correct_count, # Items where BOTH shape and color are correct
        "incorrect_items": len(items_with_errors - extra_numbers), # Count of expected items with >= 1 error (shape or color or missing)
        "incorrect_shapes_count": incorrect_shapes_count, # Total count of shape mismatches across all items
        "incorrect_colors_count": incorrect_colors_count, # Total count of color mismatches across all items
        "missing_item_keys": sorted(list(missing_numbers)),
        "extra_item_keys": sorted(list(extra_numbers)),
        "error_details": incorrect_details_str, # Keep detailed string messages
        "pass_status": "PASS" if is_pass else "FAIL"
    }
    
    return pass_rate, details

# --- Function for Pair Correctness (Updated Logic) --- 
def compare_pair_correctness(llm_result: Dict[str, Any], ground_truth: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """Compares the SET of (shape_type, color) pairs, ignoring number assignment.
    Returns the PROPORTION (0.0 to 1.0) of ground truth pairs found in the LLM output.
    """
    details = {"proportion_correct": 0.0} # Start with 0
    try:
        # Extract ground truth pairs
        gt_shapes_list = ground_truth.get("shapes", [])
        if not gt_shapes_list:
            details["error"] = "Ground truth 'shapes' data missing or empty."
            return 0.0, details
        gt_pairs = frozenset((s.get("type"), s.get("color")) for s in gt_shapes_list if s.get("type") and s.get("color"))
        
        # Extract LLM output pairs
        if not isinstance(llm_result, dict):
            details["error"] = "LLM output was not a dictionary."
            return 0.0, details
        llm_pairs = frozenset((item.get("shape_type"), item.get("color")) for item in llm_result.values() if isinstance(item, dict) and item.get("shape_type") and item.get("color"))

        # Calculate proportion of correct pairs
        num_gt_pairs = len(gt_pairs)
        if num_gt_pairs == 0:
            details["error"] = "No valid pairs found in ground truth."
            return 0.0, details # Or 1.0 if empty set match is desired?
            
        intersection = gt_pairs.intersection(llm_pairs)
        num_correct_pairs = len(intersection)
        proportion_correct = num_correct_pairs / num_gt_pairs
        
        details["proportion_correct"] = proportion_correct
        details["num_gt_pairs"] = num_gt_pairs
        details["num_llm_pairs"] = len(llm_pairs)
        details["num_correct_pairs"] = num_correct_pairs
        # Add sets for debugging if needed
        # details["expected_pairs_set"] = sorted(list(gt_pairs))
        # details["llm_pairs_set"] = sorted(list(llm_pairs))
        # details["intersection"] = sorted(list(intersection))
            
        return proportion_correct, details
            
    except Exception as e:
        details["error"] = f"Exception during pair comparison: {e}"
        traceback.print_exc()
        return 0.0, details

def evaluate_test_case(raw_result_entry: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluates a single raw result entry."""
    test_case_id = raw_result_entry.get("test_case_id", "UNKNOWN")
    eval_details = {
        "test_case_id": test_case_id,
        "n_value": raw_result_entry.get("n_value", "unknown"),
        "llm_provider": raw_result_entry.get("llm_provider", "unknown"),
        "llm_model": raw_result_entry.get("llm_model", "unknown"),
        "task_type": raw_result_entry.get("task_type", "unknown"),
        "evaluation_method": "unknown", # Will be updated
        "pass_rate": 0.0,
        "details": {},
        "evaluation_error": None,
        "raw_llm_error": raw_result_entry.get("error_message") # Carry over LLM error
    }

    # 1. Check for raw LLM errors first
    if eval_details["raw_llm_error"]:
        eval_details["evaluation_error"] = "LLM failed to produce output."
        eval_details["details"] = {"error": "LLM execution failed.", "llm_error_message": eval_details["raw_llm_error"]}
        return eval_details # Cannot evaluate further

    # 2. Load Ground Truth
    test_case_data = load_single_test_case(test_case_id)
    if not test_case_data:
        eval_details["evaluation_error"] = f"Failed to load ground truth test case data for {test_case_id}."
        eval_details["details"] = {"error": "Ground truth unavailable."}
        return eval_details

    ground_truth = test_case_data.get("expected_outcome")
    evaluation_method = test_case_data.get("evaluation_method")
    eval_details["evaluation_method"] = evaluation_method

    if not ground_truth or not evaluation_method:
        eval_details["evaluation_error"] = f"Ground truth ('expected_outcome') or 'evaluation_method' missing in test case {test_case_id}."
        eval_details["details"] = {"error": "Ground truth incomplete."}
        return eval_details

    # 3. Parse LLM Output
    llm_output_str = raw_result_entry.get("llm_output")
    parsed_llm_output, parse_error = parse_llm_output(llm_output_str)

    if parse_error:
        eval_details["evaluation_error"] = f"Failed to parse LLM output: {parse_error}"
        eval_details["details"] = {"error": "LLM output parsing failed.", "parsing_error_message": parse_error}
        # Pass rate remains 0.0
        return eval_details

    # 4. Perform Comparison based on Evaluation Method
    try:
        if evaluation_method == "detailed_comparison":
            # Detailed Comparison (Pass/Fail based on exact match)
            pass_rate, details = compare_identify_all_task(parsed_llm_output, ground_truth)
            eval_details["pass_rate"] = pass_rate
            eval_details["details"] = details
            
            # Also calculate and store pair correctness PROPORTION
            pair_proportion, pair_details = compare_pair_correctness(parsed_llm_output, ground_truth)
            eval_details["pair_correctness_proportion"] = pair_proportion # Store proportion 0.0 to 1.0
            # Add details if needed for debugging pair issues
            # eval_details["pair_correctness_details"] = pair_details 
            
        # Add elif for other primary evaluation methods here
        # elif evaluation_method == "some_other_method":
        #     pass_rate, details = compare_other_task(...)
        #     eval_details["pass_rate"] = pass_rate
        #     eval_details["details"] = details
        #     # Need to decide if pair correctness applies to other methods
        else:
            eval_details["evaluation_error"] = f"Unsupported evaluation method: {evaluation_method}"
            eval_details["details"] = {"error": "Unknown evaluation method specified."}
    except Exception as compare_e:
        eval_details["evaluation_error"] = f"Error during comparison: {type(compare_e).__name__}: {compare_e}"
        eval_details["details"] = {"error": "Comparison logic failed.", "exception": str(compare_e)}
        traceback.print_exc() # Print traceback for debugging comparison errors

    return eval_details


def calculate_summary_metrics(evaluation_details_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculates summary statistics from a list of evaluation details."""
    total_cases = len(evaluation_details_list)
    if total_cases == 0:
        return {"error": "No evaluation details provided.", "total_cases": 0}

    passed_count = 0
    failed_llm_execution_count = 0
    failed_parsing_count = 0
    failed_comparison_count = 0
    evaluation_logic_errors = 0 # Ground truth loading, unknown method etc.
    
    pass_rates_by_n = {}
    counts_by_n = {}

    for entry in evaluation_details_list:
        n_val = entry.get("n_value", "unknown")
        if n_val not in pass_rates_by_n:
            pass_rates_by_n[n_val] = 0
            counts_by_n[n_val] = 0
        counts_by_n[n_val] += 1

        if entry.get("pass_rate", 0.0) == 1.0:
            passed_count += 1
            pass_rates_by_n[n_val] += 1
        elif entry.get("raw_llm_error"):
            failed_llm_execution_count += 1
        elif entry.get("evaluation_error") and "parse LLM output" in entry["evaluation_error"]:
            failed_parsing_count += 1
        elif entry.get("evaluation_error") and "during comparison" in entry["evaluation_error"]:
            failed_comparison_count += 1
        elif entry.get("evaluation_error"):
            evaluation_logic_errors += 1
        # else: # Pass rate is 0 but no specific error category hit - counts as comparison fail
            # failed_comparison_count += 1
            
    # Calculate overall pass rate
    overall_pass_rate = (passed_count / total_cases) * 100 if total_cases > 0 else 0

    # Calculate pass rates per N
    summary_pass_rates_by_n = {}
    for n_val, n_count in counts_by_n.items():
        n_passed = pass_rates_by_n.get(n_val, 0)
        summary_pass_rates_by_n[f"n_{n_val}"] = (n_passed / n_count) * 100 if n_count > 0 else 0
        
    summary = {
        "total_cases": total_cases,
        "overall_pass_rate_percent": round(overall_pass_rate, 2),
        "passed_count": passed_count,
        "failed_count": total_cases - passed_count,
        "failure_breakdown": {
            "llm_execution_error": failed_llm_execution_count,
            "output_parsing_error": failed_parsing_count,
            "comparison_error": failed_comparison_count + (total_cases - passed_count - failed_llm_execution_count - failed_parsing_count - evaluation_logic_errors), # Include implicit comparison fails
            "evaluation_setup_error": evaluation_logic_errors
        },
        "pass_rate_percent_by_n": summary_pass_rates_by_n
    }
    return summary

def main():
    parser = argparse.ArgumentParser(description="Evaluate CeledonBenchCircuits results for a specific run.")
    parser.add_argument("--run_dir", required=True, help="Path to the specific run directory containing raw results.")
    
    args = parser.parse_args()

    run_dir = args.run_dir
    if not os.path.isdir(run_dir):
        print(f"Error: Run directory not found: {run_dir}")
        return

    # Find the raw results file (assuming one per run directory)
    raw_results_files = glob.glob(os.path.join(run_dir, "*_raw_results.jsonl"))
    if not raw_results_files:
        print(f"Error: No '*_raw_results.jsonl' file found in {run_dir}")
        return
    if len(raw_results_files) > 1:
        print(f"Warning: Multiple raw results files found in {run_dir}. Using the first one: {os.path.basename(raw_results_files[0])}")
    
    raw_results_path = raw_results_files[0]
    evaluation_details_path = os.path.join(run_dir, "evaluation_details.jsonl")
    summary_metrics_path = os.path.join(run_dir, "summary_metrics.json")

    print(f"Evaluating results from: {os.path.basename(raw_results_path)}")

    all_evaluation_details = []
    processed_count = 0
    error_count = 0

    try:
        # Correctly handle multiple file openings
        with open(raw_results_path, 'r', encoding='utf-8') as infile, open(evaluation_details_path, 'w', encoding='utf-8') as outfile:
            
            for line in infile:
                processed_count += 1
                try:
                    raw_result = json.loads(line.strip())
                    eval_result = evaluate_test_case(raw_result)
                    all_evaluation_details.append(eval_result)
                    outfile.write(json.dumps(eval_result) + '\n')
                    if eval_result.get("evaluation_error") or eval_result.get("raw_llm_error"):
                         # Print brief error during processing
                         error_type = "LLM Error" if eval_result.get("raw_llm_error") else "Eval Error"
                         error_msg = eval_result.get("raw_llm_error") or eval_result.get("evaluation_error")
                         print(f"  > Error ({error_type}) - Test Case {eval_result.get('test_case_id')}: {str(error_msg)[:100]}...")
                         error_count += 1

                except json.JSONDecodeError as json_e:
                    print(f"Error decoding raw result line {processed_count}: {json_e} - Line: {line.strip()[:100]}...")
                    error_count += 1
                except Exception as e:
                    print(f"Unexpected error processing line {processed_count}: {e}")
                    error_count += 1
                    traceback.print_exc()
            
    except Exception as file_e:
        print(f"Fatal error reading/writing files: {file_e}")
        return

    print(f"\nProcessed {processed_count} raw results.")
    print(f"Wrote evaluation details to: {os.path.basename(evaluation_details_path)}")
    if error_count > 0:
         print(f"Encountered {error_count} errors during evaluation processing.")

    # Calculate and save summary metrics
    if all_evaluation_details:
        print("Calculating summary metrics...")
        summary_metrics = calculate_summary_metrics(all_evaluation_details)
        try:
            with open(summary_metrics_path, 'w', encoding='utf-8') as f:
                json.dump(summary_metrics, f, indent=4)
            print(f"Wrote summary metrics to: {os.path.basename(summary_metrics_path)}")
            # Print summary to console
            print("\n--- Summary Metrics ---")
            print(json.dumps(summary_metrics, indent=2))
            print("---------------------")
        except Exception as summary_e:
            print(f"Error writing summary metrics file: {summary_e}")
    else:
        print("No evaluation details generated, skipping summary metrics calculation.")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main() 