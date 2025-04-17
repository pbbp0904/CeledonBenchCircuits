import subprocess
import sys
import os
import datetime
import concurrent.futures
import argparse

# --- Benchmark Configurations --- #
# Define the models, providers, and N values to test.
CONFIGURATIONS = []
PROVIDERS_MODELS = {
    "OpenAI": ["gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "o4-mini"],
    "Anthropic": ["claude-3-5-haiku-20241022", "claude-3-5-sonnet-20240620", "claude-3-7-sonnet-20250219"],
    "Google": ["gemini-2.0-flash-lite", "gemini-2.0-flash", "gemini-2.5-pro-preview-03-25"]
}
N_VALUES = [2, 3, 4, 5, 6]

for provider, models in PROVIDERS_MODELS.items():
    for model in models:
        for n in N_VALUES:
            CONFIGURATIONS.append({"provider": provider, "model": model, "n_value": n})

print(f"Generated {len(CONFIGURATIONS)} configurations.")

# --- Script Settings --- #
RESULTS_BASE_DIR = "results"
PYTHON_EXECUTABLE = sys.executable
ENGINE_MODULE_PATH = "src.execution.engine" # Make sure this is the correct module path
# Adjust max workers based on your system and API rate limits
# Running too many vision models concurrently can be resource-intensive
MAX_WORKERS = min(os.cpu_count(), 5)

# --- Function to Run Single Configuration --- 
def run_single_configuration(config, index, total, test_case_id=None):
    """Runs a single benchmark configuration using subprocess."""
    provider = config["provider"]
    model = config["model"]
    n_value = config["n_value"]
    safe_model_name = model.replace('/', '-').replace(':', '_') # Sanitize model name for logs/paths
    
    # Combine print statements
    print(f"\n[{index+1}/{total}] Starting Config: N={n_value} / {provider} / {safe_model_name}\n" + "-" * 30)
    
    command = [
        PYTHON_EXECUTABLE,
        "-m", ENGINE_MODULE_PATH,
        "--provider", provider,
        "--model", model,
        "--n_value", str(n_value),
        "--results_dir", RESULTS_BASE_DIR,
        # Add concurrency limit for the engine itself if needed (e.g., "--concurrency", "5")
    ]
    
    # Conditionally add test_case_id
    if test_case_id:
        command.extend(["--test_case_id", test_case_id])
        print(f"  Targeting specific test case: {test_case_id}")

    success = False
    stdout_output = ""
    stderr_output = ""
    start_time = datetime.datetime.now()
    
    try:
        process = subprocess.run(
            command,
            check=False, # Don't raise exception on non-zero exit code immediately
            capture_output=True,
            text=True, 
            encoding='utf-8',
            errors='replace' # Handle potential encoding errors
        )
        
        stdout_output = process.stdout
        stderr_output = process.stderr
        
        if process.returncode == 0:
            print(f"Config [{index+1}/{total}] (N={n_value}/{provider}/{safe_model_name}) completed successfully.")
            success = True
        else:
            print(f"ERROR: Config [{index+1}/{total}] (N={n_value}/{provider}/{safe_model_name}) failed! (Return Code: {process.returncode})" )
            
    except Exception as e:
        print(f"ERROR: Subprocess execution failed for config [{index+1}/{total}]: {e}")
        stderr_output += f"\nSubprocess Execution Exception: {e}" 
        
    end_time = datetime.datetime.now()
    duration = end_time - start_time
    print(f"--- Config [{index+1}/{total}] Duration: {duration} ---")
        
    # Optionally log stdout/stderr to files for easier debugging
    log_dir = os.path.join(RESULTS_BASE_DIR, "_run_logs")
    os.makedirs(log_dir, exist_ok=True)
    log_filename_base = f"{start_time.strftime('%Y%m%d_%H%M%S')}_N{n_value}_{provider}_{safe_model_name}"
    if test_case_id:
        log_filename_base += f"_{test_case_id}"
        
    try:
        with open(os.path.join(log_dir, f"{log_filename_base}.stdout.log"), 'w', encoding='utf-8') as f_out:
            f_out.write(stdout_output)
        if stderr_output:
            with open(os.path.join(log_dir, f"{log_filename_base}.stderr.log"), 'w', encoding='utf-8') as f_err:
                f_err.write(stderr_output)
    except Exception as log_e:
        print(f"Warning: Failed to write log files for config [{index+1}/{total}]: {log_e}")

    # Print summary of captured output (might be long)
    # print(f"--- Stdout for Config [{index+1}/{total}] --- ")
    # print(stdout_output.strip()[:500] + ("..." if len(stdout_output.strip()) > 500 else "")) # Print preview
    # print("--- End Stdout ---")
    if stderr_output:
        print(f"--- Stderr for Config [{index+1}/{total}] --- ")
        print(stderr_output.strip()[:1000] + ("..." if len(stderr_output.strip()) > 1000 else "")) # Print preview
        print("--- End Stderr ---")
    print("-" * 60)
        
    return success

# --- Main Parallel Function --- 
def main():
    parser = argparse.ArgumentParser(description="Run multiple CeledonBenchCircuits benchmarks in parallel.")
    parser.add_argument(
        "--test_case_id", 
        type=str, 
        default=None,
        help="Optional: Run only a specific test case ID across all matching N-value configurations."
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=MAX_WORKERS,
        help=f"Maximum number of parallel benchmark processes (default: {MAX_WORKERS}). Adjust based on resources/API limits."
    )
    args = parser.parse_args()
    
    start_time = datetime.datetime.now()
    print(f"Starting PARALLEL multi-benchmark run at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Found {len(CONFIGURATIONS)} configurations to run.")
    print(f"Using up to {args.max_workers} parallel workers.")
    if args.test_case_id:
        print(f"*** Running ONLY test case: {args.test_case_id} (will run on all configs where N matches) ***")
    
    successful_runs = 0
    failed_runs = 0
    futures = []
    
    # Filter configurations if test_case_id is provided
    configs_to_run = CONFIGURATIONS
    if args.test_case_id:
        try:
            target_n = int(args.test_case_id.split('_')[0][1:]) # Extract N from test case id
            configs_to_run = [cfg for cfg in CONFIGURATIONS if cfg["n_value"] == target_n]
            if not configs_to_run:
                 print(f"Warning: No configurations found matching N={target_n} for test case {args.test_case_id}. Exiting.")
                 return
            print(f"Filtered to {len(configs_to_run)} configurations matching N={target_n} for test case {args.test_case_id}")
        except (IndexError, ValueError):
             print(f"Error: Could not parse N value from test_case_id '{args.test_case_id}'. Running all configurations.")
             # Proceed with all configs if parsing fails
             configs_to_run = CONFIGURATIONS
             args.test_case_id = None # Clear invalid test_case_id 

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        for i, config in enumerate(configs_to_run):
            # Pass the potentially filtered test_case_id
            futures.append(executor.submit(run_single_configuration, config, i, len(configs_to_run), args.test_case_id))
        
        for future in concurrent.futures.as_completed(futures):
            try:
                if future.result():
                    successful_runs += 1
                else:
                    failed_runs += 1
            except Exception as exc:
                print(f'ERROR: A configuration run generated an exception in the runner: {exc}')
                failed_runs += 1
    
    end_time = datetime.datetime.now()
    duration = end_time - start_time
    
    print("\n" + "=" * 40)
    print(f"PARALLEL Multi-benchmark run finished at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total Duration: {duration}")
    print(f"Configurations Attempted: {successful_runs + failed_runs}")
    print(f"  Successful: {successful_runs}")
    print(f"  Failed:     {failed_runs}")
    print("Note: Check '_run_logs' directory for detailed stdout/stderr.")
    print("=" * 40)

if __name__ == "__main__":
    main() 