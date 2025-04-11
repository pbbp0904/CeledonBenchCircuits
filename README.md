# CeledonBenchCircuits

A benchmark suite for evaluating the visual understanding capabilities of large multimodal models (LMMs) on circuit-like diagrams.

This benchmark generates images containing randomly placed colored shapes and numbers, connected by curvy arrows. Models are tasked with analyzing these images and identifying the connections and properties of the elements based on the provided instructions.

## Example Image

![Example circuit diagram from the benchmark](datasets/n6/images/n6_015_identify_all.png)

## Project Structure

```
CeledonBenchCircuits/
├── datasets/             # Generated test case data
│   ├── n2/               # Test cases with N=2 shapes
│   │   ├── images/       # PNG images for N=2
│   │   └── n2_001_identify_all.json
│   │   └── ...
│   ├── n3/
│   └── ...               # Subdirectories for N=2 through N=6 (default)
├── results/              # Raw outputs and evaluation results from benchmark runs
│   ├── 20250410_190017_Google_gemini-2.0-flash-lite_n2/ # Example run directory
│   │   ├── n2_raw_results.jsonl        # Raw LLM outputs
│   │   ├── evaluation_details.jsonl  # Per-test-case evaluation results
│   │   └── summary_metrics.json      # Aggregated pass/fail metrics for the run
│   └── ...
│   └── _run_logs/        # Detailed stdout/stderr logs from run_multiple_benchmarks
├── analysis_plots/       # Output plots and tables from analysis
│   ├── pass_rate_by_n_value.png
│   ├── pass_rate_by_model.png
│   ├── summary_pass_rate_table.png
│   ├── correctness_rate_heatmap_by_model.png
│   ├── subitem_correctness_rate_table.png
│   ├── pair_correctness_heatmap_by_model.png
│   └── pair_correctness_rate_table.png
├── src/
│   ├── generation/       # Scripts for generating datasets (image_generator, dataset_generator)
│   ├── execution/        # Scripts for running benchmarks (llm_wrappers, prompt_utils, engine)
│   ├── evaluation/       # Scripts for evaluating results (metrics)
│   ├── analysis/         # Scripts for analyzing and plotting results (plotter, analyze_results)
│   └── utils/            # Utility scripts (data_loader)
├── .env.example        # Example environment file for API keys
├── requirements.txt    # Python dependencies
├── run_multiple_benchmarks.py # Script to run benchmarks for multiple configurations
├── evaluate_all_results.py   # Script to evaluate all completed benchmark runs
└── README.md           # This file
```

## Setup

1.  **Clone the Repository:**
    ```bash
    # (If applicable)
    # git clone <repository_url>
    # cd CeledonBenchCircuits
    ```

2.  **Create and Activate Virtual Environment (Recommended):**
    ```bash
    # Create the environment
    python -m venv venv
    
    # Activate the environment
    # Windows (PowerShell):
    .\venv\Scripts\Activate.ps1
    # macOS/Linux:
    # source venv/bin/activate
    ```
    *(If activation fails on PowerShell due to execution policy, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` and try again.)*

3.  **Install Dependencies:**
    ```bash
    # Make sure your virtual environment is active first!
    pip install -r requirements.txt
    # Optional, for natural sorting of test case IDs in heatmaps:
    # pip install natsort 
    ```

4.  **Configure API Keys:**
    *   Copy `.env.example` to a new file named `.env`.
    *   Open `.env` and replace the placeholder values with your actual API keys for the LLM providers you intend to use (OpenAI, Anthropic, Google).
    ```env
    # .env
    OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    ANTHROPIC_API_KEY="sk-ant-api03-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx-xxxxxxxxxx"
    GOOGLE_API_KEY="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    ```

## Usage

The benchmark workflow involves generating the dataset, running models against it, evaluating the raw results, and analyzing the aggregated metrics.

### Step 1: Generate Dataset

*   Run the dataset generator script. This creates the JSON metadata and PNG images in the `datasets/` directory.
*   By default, it generates 20 test cases for N values 2 through 6.
    ```bash
    python -m src.generation.dataset_generator
    ```
*   You can customize the N values and the number of cases per N:
    ```bash
    # Generate only for N=2 and N=3, with 10 cases each
    python -m src.generation.dataset_generator -n 2 3 --num_per_n 10
    ```
*   The script generates images with specific constraints (e.g., using the first N shapes/colors, deterministic seeding) and creates JSON files containing the ground truth, instructions, and image paths.
*   The current default task (`identify_all`) asks the LLM to output a JSON mapping each number to the shape type and color it points to. The prompt includes hints about valid shape/color names and a fallback message if the image isn't detected.

### Step 2: Run Benchmarks

*   Edit the `PROVIDERS_MODELS` dictionary and `N_VALUES` list near the top of `run_multiple_benchmarks.py` to define which models and N values you want to test. Ensure the models listed are appropriate for vision tasks.
*   Run the script:
    ```bash
    python run_multiple_benchmarks.py
    ```
*   This script executes the benchmark engine (`src.execution.engine`) for each configuration (Model x N Value) defined in the script, potentially in parallel (up to 4 workers by default, adjustable via `--max_workers`).
*   Raw results (`*_raw_results.jsonl`) are saved in timestamped directories within `results/`. Detailed stdout/stderr logs for each run are saved in `results/_run_logs/`.
*   To rerun only a specific test case (e.g., for debugging or after fixing an issue), use the `--test_case_id` flag. This will run the specified test case ID across all configurations matching its N value.
    ```bash
    # Example: Rerun only n3_005 for all configured models that run on N=3
    python run_multiple_benchmarks.py --test_case_id n3_005_identify_all
    ```

### Step 3: Evaluate Results

*   After benchmark runs are complete (or partially complete), evaluate the generated raw results:
    ```bash
    python evaluate_all_results.py
    ```
*   This script scans the `results/` directory for run folders containing raw results (`*_raw_results.jsonl`).
*   For each run found, it calls the evaluation logic (`src.evaluation.metrics`).
*   The evaluation compares the LLM's output (parsed from the raw JSONL file) against the ground truth from the corresponding dataset JSON file.
*   It generates two files within each run directory:
    *   `evaluation_details.jsonl`: Contains detailed results for each test case, including:
        *   `pass_rate`: 1.0 if the output JSON exactly matched the expected structure and values, 0.0 otherwise.
        *   `details`: Granular error breakdown (e.g., incorrect shape/color counts).
        *   `pair_correctness_proportion`: The proportion (0.0-1.0) of ground truth (shape, color) pairs found in the LLM's output set, ignoring number assignment.
    *   `summary_metrics.json`: Aggregated metrics for the entire run (e.g., overall pass rate, failure breakdown).
*   **Important:** By default, this script forces re-evaluation of all runs where raw results exist, overwriting previous evaluation files. This is useful if the evaluation logic has been updated.

### Step 4: Analyze Results

*   Generate aggregated plots and tables from all evaluated runs:
    ```bash
    python -m src.analysis.analyze_results
    ```
*   This script reads the `summary_metrics.json` and `evaluation_details.jsonl` files from all evaluated run directories in `results/`.
*   It generates the following files in the `analysis_plots/` directory:
    *   `pass_rate_by_n_value.png`: Bar plot showing Overall Pass Rate (%) vs. N, grouped by Model.
    *   `pass_rate_by_model.png`: Bar plot showing Overall Pass Rate (%) vs. Model, grouped by N.
    *   `summary_pass_rate_table.png`: Formatted table image showing Overall Pass Rate (%) vs. Model and N, including baseline chance rows.
    *   `correctness_rate_heatmap_by_model.png`: Heatmap showing Sub-Item Correctness Rate (%) vs. Test Case ID and Model.
    *   `subitem_correctness_rate_table.png`: Formatted table image showing average Sub-Item Correctness Rate (%) vs. Model and N.
    *   `pair_correctness_heatmap_by_model.png`: Heatmap showing Pair Correctness Rate (%) vs. Test Case ID and Model.
    *   `pair_correctness_rate_table.png`: Formatted table image showing average Pair Correctness Rate (%) vs. Model and N.
*   Models in plots and tables are ordered according to `MODEL_PLOT_ORDER` defined in `src/analysis/plotter.py`.
*   Colors are assigned based on `MODEL_PALETTE` in `src/analysis/plotter.py`.

## Extending the Benchmark

*   **Adding N Values:** Modify `N_VALUES_TO_GENERATE` in `src/generation/dataset_generator.py` or pass values via the command line. Add corresponding N values to the `N_VALUES` list in `run_multiple_benchmarks.py`.
*   **Adding Task Types:** Define new task types, corresponding instructions in `src/generation/dataset_generator.py`, implement new comparison logic in `src/evaluation/metrics.py`, potentially add new evaluation methods, and update `src/execution/prompt_utils.py` if needed.
*   **Changing Generation Parameters:** Modify constants in `src/generation/image_generator.py` (e.g., `IMG_SIZE`, `MIN_DISTANCE`, `SHAPE_TYPES`, `COLORS`, arrow parameters).
*   **Adding Models:** Add model details to `PROVIDERS_MODELS` in `run_multiple_benchmarks.py` and assign colors in `MODEL_PALETTE` in `src/analysis/plotter.py`.
