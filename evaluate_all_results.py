import os
import subprocess
import sys
import datetime
import argparse
import glob

# --- Settings ---
RESULTS_BASE_DIR = "results"
PYTHON_EXECUTABLE = sys.executable
METRICS_MODULE_PATH = "src.evaluation.metrics"

def evaluate_single_run(run_dir_path: str):
    """Constructs and runs the evaluation command for a single run directory."""
    run_name = os.path.basename(run_dir_path)
    # Don't print evaluating run here, main loop does it.
    # print(f"\n--- Evaluating Run: {run_name} ---") 

    if not os.path.isdir(run_dir_path):
        print(f"Skipping (not a directory): {run_dir_path}")
        return False

    # REMOVED Check if summary already exists 
    # summary_file = os.path.join(run_dir_path, "summary_metrics.json")
    # if os.path.exists(summary_file):
    #     print(f"Summary file already exists: {os.path.basename(summary_file)}. Re-evaluating.")
    #     # return True # Don't return early, proceed to re-evaluate

    # Check if raw results file exists before attempting evaluation
    raw_results_files = glob.glob(os.path.join(run_dir_path, "*_raw_results.jsonl"))
    if not raw_results_files:
        print(f"Skipping (No raw results file found): {run_name}")
        return False # Cannot evaluate if raw results don't exist

    command = [
        PYTHON_EXECUTABLE,
        "-m", METRICS_MODULE_PATH,
        "--run_dir", run_dir_path
    ]

    print(f"Running evaluation for {run_name}...")
    # print(f"Command: {' '.join(command)}") # Less verbose 

    try:
        process = subprocess.run(
            command,
            check=True, # Raise exception on non-zero exit code
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        # Print only if there's significant output, keep it concise
        stdout_strip = process.stdout.strip()
        stderr_strip = process.stderr.strip()
        if len(stdout_strip) > 10 or stderr_strip:
            print("--- Evaluation Output Start ---")
            if stdout_strip:
                 print(stdout_strip)
            if stderr_strip:
                 print("--- Stderr ---")
                 print(stderr_strip)
            print("--- Evaluation Output End ---")
            
        print(f"Evaluation successful for {run_name}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"ERROR: Evaluation failed for {run_name}! (Return Code: {e.returncode})")
        # Print captured output on error for debugging
        print("--- Evaluation Stdout Start ---")
        print(e.stdout.strip())
        print("--- Evaluation Stdout End ---")
        print("--- Evaluation Stderr Start ---")
        print(e.stderr.strip())
        print("--- Evaluation Stderr End ---")
        return False
    except Exception as e:
        print(f"UNEXPECTED ERROR during evaluation of {run_name}: {e}")
        return False

def main(results_dir: str):
    start_time = datetime.datetime.now()
    print(f"Starting evaluation of all runs in '{results_dir}' at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    if not os.path.isdir(results_dir):
        print(f"Error: Results base directory '{results_dir}' not found.")
        return

    # Find potential run directories (any subdirectory in results/)
    try:
        run_dirs = [
            os.path.join(results_dir, d)
            for d in os.listdir(results_dir)
            if os.path.isdir(os.path.join(results_dir, d)) and not d.startswith('_') # Exclude helper dirs like _run_logs
        ]
    except OSError as e:
        print(f"Error listing directories in {results_dir}: {e}")
        return

    if not run_dirs:
        print(f"No run directories found in '{results_dir}'.")
        return

    print(f"Found {len(run_dirs)} potential run directories to evaluate.")

    success_count = 0
    fail_count = 0
    evaluated_count = 0 # Count runs where evaluation was actually performed

    # Sort directories for consistent processing order
    run_dirs.sort()

    for run_dir in run_dirs:
        run_name = os.path.basename(run_dir)
        raw_results_files = glob.glob(os.path.join(run_dir, "*_raw_results.jsonl"))

        if not raw_results_files:
            print(f"Skipping (No raw results): {run_name}")
            # Don't count as failure if raw results never existed
            continue 
            
        # If raw results exist, attempt evaluation regardless of existing summary
        print(f"Processing run: {run_name}")
        evaluated_count += 1
        if evaluate_single_run(run_dir):
            success_count += 1
        else:
            fail_count += 1

    end_time = datetime.datetime.now()
    duration = end_time - start_time

    print("\n" + "=" * 40)
    print(f"Evaluation completed at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total Duration: {duration}")
    print(f"Runs Processed (Attempted Eval): {evaluated_count}")
    print(f"  Successful Evals: {success_count}")
    print(f"  Failed Evals:     {fail_count}")
    print("=" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate all CeledonBenchCircuits benchmark runs.")
    parser.add_argument("--results_dir", default=RESULTS_BASE_DIR, 
                        help=f"Base directory containing the benchmark run result folders (default: {RESULTS_BASE_DIR})")
    args = parser.parse_args()
    main(args.results_dir) 