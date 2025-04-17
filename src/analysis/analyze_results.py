import os
import glob
import argparse

# Import necessary functions from the dedicated plotter script
from src.analysis.plotter import load_and_prepare_data, create_n_value_grouped_plot, create_model_grouped_plot, load_detailed_evaluation_data, create_pass_fail_heatmap, create_summary_pass_rate_table, create_pair_correctness_table, create_pair_correctness_heatmap, create_subitem_correctness_table, MODEL_PLOT_ORDER

# --- Settings ---
RESULTS_BASE_DIR = "results"
OUTPUT_DIR = "analysis_plots"

def find_run_dirs(results_dir: str) -> list[str]:
    """Finds all immediate subdirectories within results_dir containing a summary_metrics.json file."""
    if not os.path.isdir(results_dir):
        print(f"Error: Results directory not found: {results_dir}")
        return []
        
    # Updated glob pattern: Look for the summary file directly in subdirectories
    summary_files = glob.glob(os.path.join(results_dir, "*", "summary_metrics.json")) 
    
    if not summary_files:
        # Also check for directories starting with underscore (like _run_logs) and exclude them
        potential_dirs = [d for d in glob.glob(os.path.join(results_dir, "*")) if os.path.isdir(d) and not os.path.basename(d).startswith('_')]
        found_summaries = False
        summary_files = []
        for d in potential_dirs:
            if os.path.exists(os.path.join(d, "summary_metrics.json")):
                 summary_files.append(os.path.join(d, "summary_metrics.json"))
                 found_summaries = True
        
        if not found_summaries:
            print(f"No summary_metrics.json files found directly within immediate subdirectories of {results_dir}")
            return []

    # Extract the directory path from the summary file path and sort
    run_dirs = sorted([os.path.dirname(f) for f in summary_files])
    print(f"Found {len(run_dirs)} run directories with summary files to analyze.")
    return run_dirs

def main(results_dir: str, output_dir: str):
    """Main function to find results and trigger plotting using src.analysis.plotter."""
    print(f"Starting analysis of results in: {results_dir}")
    run_directories = find_run_dirs(results_dir)

    if run_directories:
        # Ensure output directory exists before calling plotter
        os.makedirs(output_dir, exist_ok=True)
        print(f"Analysis plots will be saved to: {output_dir}")
        
        # Load and prepare data using the plotter's function
        df_summary, models_list, n_values_list = load_and_prepare_data(run_directories)

        if df_summary is not None and not df_summary.empty:
            # Call the plotting functions from the plotter module
            print(f"Summary data prepared. Generating bar plots for {len(models_list)} models and N values: {n_values_list}...")
            create_n_value_grouped_plot(df_summary, models_list, n_values_list, output_dir)
            create_model_grouped_plot(df_summary, models_list, n_values_list, output_dir)
            
            # --- Generate Summary Table Image ---
            # Pass the same summary dataframe and model order
            print("\nGenerating summary pass rate table...")
            create_summary_pass_rate_table(df_summary, models_list, n_values_list, output_dir)
        else:
            print("Failed to load or prepare summary data. Skipping bar plots and summary table.")
            
        # Load detailed evaluation data and create heatmap
        print("\nLoading detailed evaluation data...")
        df_detailed = load_detailed_evaluation_data(run_directories)
        
        if df_detailed is not None and not df_detailed.empty:
            # Pass detailed data to sub-item correctness heatmap
            # Note: Assuming create_pass_fail_heatmap was correctly updated to show correctness_rate
            print("\nGenerating sub-item correctness heatmap...")
            create_pass_fail_heatmap(df_detailed, MODEL_PLOT_ORDER, output_dir)
            
            # Pass detailed data to sub-item correctness table
            print("\nGenerating sub-item correctness table...")
            create_subitem_correctness_table(df_detailed, models_list, n_values_list, output_dir)
            
            # Pass detailed data to pair correctness table
            print("\nGenerating pair correctness table...")
            create_pair_correctness_table(df_detailed, models_list, n_values_list, output_dir)
            
            # Pass detailed data to pair correctness heatmap
            print("\nGenerating pair correctness heatmap...")
            create_pair_correctness_heatmap(df_detailed, MODEL_PLOT_ORDER, output_dir)
        else:
            print("Failed to load detailed evaluation data. Skipping heatmap and pair correctness plots.")
            
    else:
        print("Analysis finished: No results data found to plot.")

    print("\nAnalysis complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze CeledonBenchCircuits results and generate plots.")
    parser.add_argument("--results_dir", default=RESULTS_BASE_DIR,
                        help=f"Base directory containing benchmark run result folders (default: {RESULTS_BASE_DIR})")
    parser.add_argument("--output_dir", default=OUTPUT_DIR, 
                        help=f"Directory to save analysis plots (default: {OUTPUT_DIR})")
    
    args = parser.parse_args()
    
    main(args.results_dir, args.output_dir) 