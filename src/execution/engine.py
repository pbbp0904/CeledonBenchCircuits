import os
import json
import argparse
import time
import datetime
import uuid
import traceback
import asyncio
import operator # For sorting
import glob # For finding directories

# Project-specific imports
from src.utils.data_loader import load_test_cases_for_n, load_single_test_case
from src.execution.prompt_utils import format_prompt
from src.execution.llm_wrappers import get_llm_wrapper, LLMWrapper

# Define the base directory for datasets relative to the project root
DATASET_BASE_DIR = "datasets"

# --- Concurrency Settings ---
# Adjust based on API limits and system resources
MAX_CONCURRENT_LLM_CALLS = 5 

async def process_test_case(
    test_case: dict, 
    llm: LLMWrapper, 
    run_id: str, 
    provider: str, 
    model: str, 
    semaphore: asyncio.Semaphore, 
    output_lock: asyncio.Lock, 
    temp_results_file_path: str 
):
    """Processes a single test case asynchronously and writes result to a temp file."""
    test_case_id = test_case.get('test_case_id', 'UNKNOWN')
    n_value = test_case.get('n_value', 'unknown')
    task_name = f"task_{test_case_id}" 
    asyncio.current_task().set_name(task_name)

    result_entry = {
        "run_id": run_id,
        "test_case_id": test_case_id,
        "n_value": n_value,
        "task_type": test_case.get("task_type", "unknown"),
        "instruction": test_case.get("instruction", "unknown"),
        "image_path": test_case.get("image_path", "unknown"),
        "llm_provider": provider,
        "llm_model": model,
        "llm_output": None, # Store the parsed output (string) or raw text
        "error_message": None,
        "execution_time_sec": 0,
        "timestamp": None 
    }

    try:
        prompt = format_prompt(test_case)
        relative_image_path = test_case.get("image_path")
        
        if not relative_image_path:
            raise FileNotFoundError("Image path missing in test case JSON.")
            
        # Construct the full path relative to the project root
        full_image_path = os.path.join(DATASET_BASE_DIR, relative_image_path)
        
        # Check existence using the full path
        if not os.path.exists(full_image_path):
             raise FileNotFoundError(f"Image file not found at expected location: {full_image_path}")

        # print(f"Task {task_name}: Waiting for semaphore...") # Verbose
        async with semaphore:
            # print(f"Task {task_name}: Semaphore acquired, calling LLM...") # Verbose
            start_time = time.time()
            # Call the async generate method with the FULL image path
            llm_output_str, error_msg = await llm.generate(prompt, image_path=full_image_path)
            end_time = time.time()
        # print(f"Task {task_name}: LLM call finished.") # Verbose
        
        result_entry["llm_output"] = llm_output_str
        result_entry["error_message"] = error_msg
        result_entry["execution_time_sec"] = end_time - start_time
        result_entry["timestamp"] = datetime.datetime.now().isoformat()

        if error_msg:
            print(f"Task {task_name}: LLM call failed (recorded). Error: {error_msg[:100]}...")
        # else:
            # print(f"Task {task_name}: LLM call successful. Time: {result_entry['execution_time_sec']:.2f}s") # Verbose
            
    except FileNotFoundError as fnf_err:
        print(f"Task {task_name}: ERROR - {fnf_err}")
        result_entry["error_message"] = str(fnf_err)
    except Exception as e:
        print(f"Task {task_name}: FATAL ERROR during processing: {type(e).__name__}: {e}")
        tb_str = traceback.format_exc()
        result_entry["error_message"] = f"FATAL ENGINE ERROR: {type(e).__name__}: {e}\nTRACEBACK:\n{tb_str}"
        result_entry["execution_time_sec"] = 0 # Reset time on fatal error
        result_entry["timestamp"] = datetime.datetime.now().isoformat()

    # Safely write the result incrementally to the TEMPORARY file
    try:
        async with output_lock:
            with open(temp_results_file_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result_entry) + '\n')
    except Exception as log_e:
         print(f"Task {task_name}: ERROR writing result to temp log file {temp_results_file_path}: {log_e}")

    return result_entry 

# --- Helper function to find latest run directory for a specific N --- 
def find_latest_run_dir(base_results_dir: str, provider: str, model: str, n_value: int) -> str | None:
    """Finds the latest timestamped run directory for a given N config."""
    # Ensure model name is filesystem-safe
    safe_model_name = model.replace('/', '-').replace(':', '_')
    pattern = os.path.join(base_results_dir, f"*_{provider}_{safe_model_name}_n{n_value}")
    run_dirs = glob.glob(pattern)
    if not run_dirs:
        return None
    run_dirs.sort(reverse=True) # Sort by timestamp descending
    return run_dirs[0]

async def run_benchmark_async(n_value: int, provider: str, model: str, results_dir: str, test_case_id_to_run: str | None = None):
    """Runs the benchmark asynchronously for a given N, saves sorted results."""
    if test_case_id_to_run:
        print(f"*** TARGETING SINGLE TEST CASE: {test_case_id_to_run} for N={n_value} / {provider} / {model} ***")
    else:
        print(f"Starting ASYNC benchmark run: N={n_value}, Provider={provider}, Model={model}")

    # --- Setup --- #
    try:
        all_test_cases = load_test_cases_for_n(n_value)
        if not all_test_cases:
            print(f"No test cases found for N={n_value}. Exiting.")
            return
        
        # Filter test cases if a specific ID is provided
        if test_case_id_to_run:
            test_cases = [tc for tc in all_test_cases if tc.get('test_case_id') == test_case_id_to_run]
            if not test_cases:
                 print(f"ERROR: Specified test case ID '{test_case_id_to_run}' not found for N={n_value}. Exiting.")
                 return
        else:
            test_cases = all_test_cases
            
        llm = get_llm_wrapper(provider, model)
    except (FileNotFoundError, ValueError, ImportError) as e:
        print(f"Error during setup: {e}")
        print("Ensure API keys are set in .env and required packages are installed.")
        # print({k: v for k, v in os.environ.items() if 'API_KEY' in k}) # Debug: Print keys
        return
    except Exception as setup_e:
        print(f"Unexpected error during setup: {setup_e}")
        traceback.print_exc()
        return

    # --- Determine Run Directory and File Paths --- 
    target_run_dir = None
    is_rerun = False
    safe_model_name = model.replace('/', '-').replace(':', '_') # Sanitize model name
    
    if test_case_id_to_run:
        # For reruns, find the *latest* existing directory for this config
        target_run_dir = find_latest_run_dir(results_dir, provider, model, n_value)
        if target_run_dir:
            print(f"Found latest existing run directory to update: {os.path.basename(target_run_dir)}")
            is_rerun = True
        else:
            print(f"No existing run directory found for N={n_value}/{provider}/{model}. Creating a new one.")
            
    # Create a new run directory if not doing a targeted re-run or if no existing dir was found
    if not target_run_dir:
        run_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        run_id = f"{run_timestamp}_{provider}_{safe_model_name}_n{n_value}"
        target_run_dir = os.path.join(results_dir, run_id)
        try:
            os.makedirs(target_run_dir, exist_ok=True)
        except OSError as e:
            print(f"FATAL: Could not create results directory {target_run_dir}: {e}")
            return
    else:
        # Extract run_id from the existing directory name if needed
        run_id = os.path.basename(target_run_dir)
        
    # Define FINAL and TEMP file paths
    final_results_file_path = os.path.join(target_run_dir, f"n{n_value}_raw_results.jsonl")
    # Use a unique temp file name to avoid conflicts during crashes/reruns
    temp_file_uuid = uuid.uuid4()
    temp_results_file_path = os.path.join(target_run_dir, f"n{n_value}_temp_results_{temp_file_uuid}.jsonl")
    
    print(f"Run Directory: {target_run_dir}")
    print(f"Results file: {os.path.basename(final_results_file_path)}")
    # --- End Directory/Path Logic --- 

    # --- Async Execution --- #
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_LLM_CALLS)
    output_lock = asyncio.Lock() 
    tasks = []

    print(f"Creating {len(test_cases)} task(s) for N={n_value}...") 
    for test_case in test_cases:
        tasks.append(process_test_case(
            test_case, llm, run_id, provider, model, 
            semaphore, output_lock, temp_results_file_path
        ))

    start_run_time = time.time()
    print(f"Dispatching {len(tasks)} task(s) with concurrency limit {MAX_CONCURRENT_LLM_CALLS}...")
    # Use return_exceptions=True to prevent one task failure from stopping others
    results = await asyncio.gather(*tasks, return_exceptions=True)
    end_run_time = time.time()
    print(f"\nAll {len(tasks)} task(s) completed or failed for N={n_value}.")

    # Check for exceptions returned by gather
    task_exceptions = [res for res in results if isinstance(res, Exception)]
    if task_exceptions:
        print(f"WARNING: {len(task_exceptions)} task(s) raised exceptions during execution.")
        # Optionally print details of the first few exceptions for debugging
        for i, exc in enumerate(task_exceptions[:3]):
             print(f"  Exception {i+1}: {type(exc).__name__}: {exc}")

    # --- Post-processing: Sort and write final results --- 
    new_results_data = []
    try:
        if not os.path.exists(temp_results_file_path):
             print(f"Warning: Temporary results file not found: {os.path.basename(temp_results_file_path)}. No new results to process.")
             return # Exit post-processing
             
        with open(temp_results_file_path, 'r', encoding='utf-8') as temp_f:
            for line in temp_f:
                try:
                    new_results_data.append(json.loads(line.strip()))
                except json.JSONDecodeError as json_e:
                    print(f"Warning: Skipping invalid JSON line in temp file: {json_e} - Line: {line.strip()[:100]}...")
        
        if not new_results_data:
            print("Warning: No valid results found in temporary file.")
        else:
            # --- Merge logic for reruns --- 
            final_data_to_write = []
            if is_rerun and test_case_id_to_run:
                print(f"Merging new result for {test_case_id_to_run} into existing results: {os.path.basename(final_results_file_path)}")
                existing_results = []
                if os.path.exists(final_results_file_path):
                    try:
                        with open(final_results_file_path, 'r', encoding='utf-8') as existing_f:
                            for line in existing_f:
                                try:
                                    entry = json.loads(line.strip())
                                    # Only keep results that are NOT the one being rerun
                                    if entry.get('test_case_id') != test_case_id_to_run:
                                        existing_results.append(entry)
                                except json.JSONDecodeError:
                                    print(f"Warning: Skipping invalid JSON line in EXISTING results file.")
                    except Exception as read_err:
                        print(f"Error reading existing results file {final_results_file_path}: {read_err}. Proceeding with only new results.")
                else:
                     print(f"Existing results file not found. Saving new result directly.")
                     
                final_data_to_write = existing_results
                final_data_to_write.extend(new_results_data)
            else:
                 # Not a targeted rerun, or initial run: use only new results
                 final_data_to_write = new_results_data
            # --- End Merge Logic ---
                
            if not final_data_to_write:
                 print("Warning: No results to write to the final file.")
            else:
                # Sort final list by test_case_id
                final_data_to_write.sort(key=operator.itemgetter('test_case_id'))
                
                with open(final_results_file_path, 'w', encoding='utf-8') as final_f:
                    for entry in final_data_to_write:
                        final_f.write(json.dumps(entry) + '\n')
                print(f"Final results saved/updated ({len(final_data_to_write)} entries) in {os.path.basename(final_results_file_path)}")
                
                # --- Invalidate summary if this was a targeted rerun --- 
                if is_rerun and test_case_id_to_run:
                    summary_file_path = os.path.join(target_run_dir, "summary_metrics.json")
                    if os.path.exists(summary_file_path):
                        try:
                            os.remove(summary_file_path)
                            print(f"Deleted existing summary file to trigger re-evaluation: {os.path.basename(summary_file_path)}")
                        except OSError as del_err:
                            print(f"Warning: Could not delete existing summary file {summary_file_path}: {del_err}")

    except Exception as post_e:
        print(f"Error during post-processing (reading temp/writing final results): {post_e}")
        traceback.print_exc()
    finally:
        # Clean up temporary file only if it exists
        if os.path.exists(temp_results_file_path):
            try:
                os.remove(temp_results_file_path)
            except OSError as del_e:
                 print(f"Error deleting temporary file {temp_results_file_path}: {del_e}")

    # --- Final Summary --- 
    run_duration = end_run_time - start_run_time
    print(f"\nBenchmark task for N={n_value} duration: {run_duration:.2f}s")
    print(f"Results directory: {target_run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the CeledonBenchCircuits benchmark (Async Engine)." )
    parser.add_argument("--n_value", type=int, required=True, help="The N value (number of shapes) dataset to run.")
    parser.add_argument("--provider", required=True, help="LLM provider (e.g., OpenAI, Anthropic, Google, Groq)." )
    parser.add_argument("--model", required=True, help="Specific LLM model name (ensure it's vision-capable for this benchmark)." )
    parser.add_argument("--results_dir", default="results", help="Base directory to save run results." )
    parser.add_argument("--concurrency", type=int, default=MAX_CONCURRENT_LLM_CALLS, help="Max concurrent LLM API calls.")
    parser.add_argument(
        "--test_case_id", 
        type=str, 
        default=None,
        help="Optional: Run only a specific test case ID, overwriting its previous result in the latest run directory for the given N/Provider/Model."
    )
    
    args = parser.parse_args()
    
    # Update global concurrency limit if provided via args
    MAX_CONCURRENT_LLM_CALLS = args.concurrency
    
    # Ensure results directory exists
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Run the async benchmark function
    asyncio.run(run_benchmark_async(
        n_value=args.n_value, 
        provider=args.provider, 
        model=args.model, 
        results_dir=args.results_dir, 
        test_case_id_to_run=args.test_case_id
    )) 