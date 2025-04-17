import os
import json
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from typing import List, Tuple, Dict
import traceback
import math # Add math import for factorial

# Style settings
sns.set_theme(style="whitegrid")

# Define a color palette for consistency
MODEL_PALETTE = {
    # Google Models (Greens)
    'gemini-2.0-flash-lite': '#98df8a',        # Light Green
    'gemini-2.0-flash': '#2ca02c',             # Dark Green
    'gemini-2.5-pro-preview-03-25': '#8fbc8f', # Dark Sea Green

    # OpenAI Models (Blues)
    'gpt-4o-mini-2024-07-18': '#eeeee8',          # Light Blue
    'gpt-4o': '#cccce4',                          # Blue
    'o1': '#bbbbdd',                              # Very Light Blue
    'gpt-4.1-nano': '#aec7e8',                    # Light Blue
    'gpt-4.1-mini': '#1f77b4',                    # Blue
    'gpt-4.1': '#72bcd4',                         # Medium Blue/Cyan
    'o4-mini': '#222280',                         # Darker Blue
    
    # Anthropic Models (Oranges/Reds)
    'claude-3-5-haiku-20241022': '#ffbb78',       # Light Orange
    'claude-3-5-sonnet-20240620': '#ff7f0e',      # Orange
    'claude-3-7-sonnet-20250219': '#d62728',      # Red 
}
DEFAULT_COLOR = '#7f7f7f' # Grey for unknown models

# Define the desired order for models in plots
MODEL_PLOT_ORDER = [
    # Google
    'gemini-2.0-flash-lite',
    'gemini-2.0-flash',
    'gemini-2.5-pro-preview-03-25',
    # OpenAI
    'gpt-4o-mini-2024-07-18',
    'gpt-4o',
    'o1',
    'gpt-4.1-nano',
    'gpt-4.1-mini',
    'gpt-4.1',
    'o4-mini',
    # Anthropic
    'claude-3-5-haiku-20241022',
    'claude-3-5-sonnet-20240620',
    'claude-3-7-sonnet-20250219',
]

# --- Add new function to load detailed data ---
def load_detailed_evaluation_data(run_dirs: List[str]) -> pd.DataFrame | None:
    """Loads detailed evaluation results from multiple run directories.
       Calculates Correctness_Rate: % of correctly identified sub-items (shape/color).
    """
    all_details = []

    for run_dir in run_dirs:
        detail_path = os.path.join(run_dir, "evaluation_details.jsonl")
        if not os.path.exists(detail_path):
            print(f"Warning: evaluation_details.jsonl not found in {run_dir}, skipping detailed analysis for this run.")
            continue

        try:
            with open(detail_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        detail_data = json.loads(line.strip())
                        
                        # Extract necessary fields for correctness rate calculation
                        details_dict = detail_data.get("details", {})
                        total_expected = details_dict.get("total_expected", 0)
                        incorrect_shapes = details_dict.get("incorrect_shapes_count", total_expected) # Assume all incorrect if missing
                        incorrect_colors = details_dict.get("incorrect_colors_count", total_expected) # Assume all incorrect if missing
                        pass_status = details_dict.get("pass_status", "FAIL")

                        correctness_rate = 0.0
                        if total_expected > 0 and pass_status != "ERROR": # Avoid division by zero and skip errored cases
                             correct_shapes = total_expected - incorrect_shapes
                             correct_colors = total_expected - incorrect_colors
                             total_sub_items = 2 * total_expected
                             correctness_rate = (correct_shapes + correct_colors) / total_sub_items

                        all_details.append({
                            "test_case_id": detail_data.get("test_case_id"),
                            "Model": detail_data.get("llm_model"),
                            "N_Value": detail_data.get("n_value"),
                            "Pass_Rate_Binary": 1 if detail_data.get("pass_rate", 0.0) == 1.0 else 0, # Keep original pass/fail if needed
                            "Correctness_Rate": correctness_rate * 100, # Store as percentage
                            "Pair_Correctness_Proportion": detail_data.get("pair_correctness_proportion", 0.0) # Load the new proportion field (0.0 to 1.0)
                        })
                    except json.JSONDecodeError as json_e:
                        print(f"Error decoding JSON line in {detail_path}: {json_e}")
        except Exception as e:
            print(f"Error reading detailed evaluation file {detail_path}: {e}")
            
    if not all_details:
        print("No valid detailed evaluation data loaded.")
        return None
        
    df = pd.DataFrame(all_details)
    # Ensure N_Value is numeric for potential sorting
    df['N_Value'] = pd.to_numeric(df['N_Value'], errors='coerce')
    df.dropna(subset=['N_Value', 'Model', 'test_case_id'], inplace=True) # Drop rows with missing crucial info
    df['N_Value'] = df['N_Value'].astype(int)
    
    print(f"Loaded {len(df)} detailed evaluation entries.")
    return df

# --- Helper to extract base model name ---
def get_base_model_name(model_name: str) -> str:
    """Extracts a base model name for grouping (heuristic)."""
    name = model_name.lower()
    # Don't strip -lite globally, handle specific cases if needed
    # name = name.replace("-lite", "") 
    parts = name.split('-')
             
    # Specific known model cleanups (add more if needed)
    if name.startswith("claude-3-5-") or name.startswith("claude-3-7-"):
        pass # Keep version number for Claude 3.x
    if name == "gemini-2.5-pro-preview-03-25": # Fix potential over-strip
         name = "gemini-2.5-pro"

    # Special case for simple names like 'o1' or 'gpt-4o'
    if name == 'o1' or name == 'gpt-4o':
        return name
        
    # General structure cleanup (might need refinement)
    name = name.replace("-2024-07-18","") # Remove specific known dates
    name = name.replace("-20241022","")
    name = name.replace("-20240620","")
    name = name.replace("-20250219","")
    name = name.replace("-03-25","") # From gemini preview
    
    return name

# --- Updated heatmap plotting function ---
def create_pass_fail_heatmap(df_detailed: pd.DataFrame, models_order: List[str], output_dir: str):
    """Creates a heatmap showing CORRECTNESS RATE (%) for each test case vs. BASE model."""
    if df_detailed is None or df_detailed.empty:
        print("Cannot create heatmap: No detailed evaluation data.")
        return

    try:
        # Add Base_Model column
        df_detailed['Base_Model'] = df_detailed['Model'].apply(get_base_model_name)
        
        # Aggregate: Calculate MEAN CORRECTNESS RATE per test case per base model
        # Group by test case and base model, calculate the mean of 'Correctness_Rate'
        agg_data = df_detailed.groupby(['test_case_id', 'Base_Model'])['Correctness_Rate'].mean().reset_index()
        # The rate is already a percentage (0-100) from the loading function
        
        # Pivot to create the matrix: test_case_id vs Base_Model
        heatmap_data = agg_data.pivot_table(index='test_case_id', columns='Base_Model', values='Correctness_Rate')
        
        # Define the desired base model column order
        base_models_in_order = []
        seen_base_models = set()
        for full_model in models_order: # Use the full model order list passed in
             base = get_base_model_name(full_model)
             if base in heatmap_data.columns and base not in seen_base_models:
                 base_models_in_order.append(base)
                 seen_base_models.add(base)
                 
        other_base_models = sorted([c for c in heatmap_data.columns if c not in seen_base_models])
        final_column_order = base_models_in_order + other_base_models
        
        # Sort rows (test_case_id) naturally
        try:
             import natsort
             sorted_index = natsort.natsorted(heatmap_data.index)
        except ImportError:
             print("Warning: natsort package not found. Falling back to simple sort for test case IDs.")
             sorted_index = sorted(heatmap_data.index)
        
        # Reindex both rows and columns, fill missing with 0 (0% correctness)
        heatmap_data = heatmap_data.reindex(index=sorted_index, columns=final_column_order).fillna(0)

        # Clean up test case IDs for display
        heatmap_data.index = heatmap_data.index.str.replace("_identify_all", "", regex=False)

        # --- Plotting --- 
        num_tests = len(heatmap_data.index)
        num_configs = len(heatmap_data.columns)
        fig_width = max(10, num_configs * 0.8)
        fig_height = max(8, num_tests * 0.18)
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        # Use a colorblind-friendly sequential colormap (e.g., viridis)
        cmap = 'viridis' 

        sns.heatmap(heatmap_data, 
                    ax=ax, 
                    cmap=cmap, 
                    linewidths=0.5, 
                    linecolor='lightgrey',
                    annot=True, # Show percentages
                    fmt=".0f", # Format as integer percentage
                    cbar=True, 
                    cbar_kws={'label': 'Correctness Rate (%)'},
                    vmin=0, vmax=100)

        ax.set_title('Sub-Item Correctness Rate (%) by Test Case and Model', fontsize=16, pad=20)
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Test Case ID', fontsize=12)
        # Move Y-axis ticks and labels to the left
        ax.yaxis.tick_left()
        ax.yaxis.set_label_position("left")
        
        # Explicitly set ticks and labels to ensure all are shown
        ax.set_yticks([i + 0.5 for i in range(len(heatmap_data.index))])
        # Increase font size slightly
        ylabel_fontsize = 10 if num_tests <= 50 else 8
        ax.set_yticklabels(heatmap_data.index, fontsize=ylabel_fontsize)
        
        # Add thicker lines between N value transitions
        last_n = None
        for i, test_id in enumerate(heatmap_data.index):
            try:
                current_n = int(test_id.split('_')[0][1:]) # Extract N from 'nX_...' 
                if last_n is not None and current_n != last_n:
                    ax.axhline(i, color='black', linewidth=1.5)
                last_n = current_n
            except (ValueError, IndexError):
                # Handle cases where test_id format might be unexpected
                continue 
                
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        plt.tight_layout()
        
        plot_filename = os.path.join(output_dir, "correctness_rate_heatmap_by_model.png") # Updated filename
        try:
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"Saved Correctness Rate heatmap to: {plot_filename}")
        except Exception as e:
            print(f"Error saving heatmap plot {plot_filename}: {e}")
        plt.close(fig)
        
    except Exception as plot_e:
        print(f"Error creating heatmap: {plot_e}")
        traceback.print_exc()

# --- New Pair Correctness Heatmap function ---
def create_pair_correctness_heatmap(df_detailed: pd.DataFrame, models_order: List[str], output_dir: str):
    """Creates a heatmap showing PAIR CORRECTNESS RATE (%) for each test case vs. BASE model."""
    # Check for the proportion column
    if df_detailed is None or df_detailed.empty or 'Pair_Correctness_Proportion' not in df_detailed.columns:
        print("Cannot create pair correctness heatmap: Missing data or 'Pair_Correctness_Proportion' column.")
        return

    try:
        # Add Base_Model column
        df_detailed['Base_Model'] = df_detailed['Model'].apply(get_base_model_name)
        
        # Aggregate: Calculate MEAN PAIR CORRECTNESS PROPORTION per test case per base model
        # NOTE: No need to average here, the proportion is already per test case
        # We just need to pivot the existing proportion data
        agg_data = df_detailed # Use the detailed data directly
        agg_data['Pair_Correctness_Rate'] = agg_data['Pair_Correctness_Proportion'] * 100 # Ensure it's percentage
        
        # Pivot to create the matrix: test_case_id vs Base_Model
        heatmap_data = agg_data.pivot_table(index='test_case_id', columns='Base_Model', values='Pair_Correctness_Rate')
        
        # --- Reorder rows and columns (remains the same) --- 
        base_models_in_order = []
        seen_base_models = set()
        for full_model in models_order:
            base = get_base_model_name(full_model)
            if base in heatmap_data.columns and base not in seen_base_models:
                base_models_in_order.append(base)
                seen_base_models.add(base)
        other_base_models = sorted([c for c in heatmap_data.columns if c not in seen_base_models])
        final_column_order = base_models_in_order + other_base_models
        try:
             import natsort
             sorted_index = natsort.natsorted(heatmap_data.index)
        except ImportError:
             print("Warning: natsort package not found. Falling back to simple sort for test case IDs.")
             sorted_index = sorted(heatmap_data.index)
        heatmap_data = heatmap_data.reindex(index=sorted_index, columns=final_column_order).fillna(0)

        # Clean up test case IDs for display
        heatmap_data.index = heatmap_data.index.str.replace("_identify_all", "", regex=False)

        # --- Plotting (Remains the same, already uses correct rate) --- 
        num_tests = len(heatmap_data.index)
        num_configs = len(heatmap_data.columns)
        fig_width = max(10, num_configs * 0.8)
        fig_height = max(8, num_tests * 0.18)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        cmap = 'viridis' 
        sns.heatmap(heatmap_data, 
                    ax=ax, cmap=cmap, linewidths=0.5, linecolor='lightgrey',
                    annot=True, fmt=".0f", cbar=True, 
                    cbar_kws={'label': 'Pair Correctness Rate (%)'},
                    vmin=0, vmax=100)
        ax.set_title('Pair Correctness Rate (%) by Test Case and Model', fontsize=16, pad=20)
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Test Case ID', fontsize=12)
        # Move Y-axis ticks and labels to the left
        ax.yaxis.tick_left()
        ax.yaxis.set_label_position("left")

        # Explicitly set ticks and labels to ensure all are shown
        ax.set_yticks([i + 0.5 for i in range(len(heatmap_data.index))])
        # Increase font size slightly
        ylabel_fontsize = 10 if num_tests <= 50 else 8
        ax.set_yticklabels(heatmap_data.index, fontsize=ylabel_fontsize)

        # Add N-transition lines
        last_n = None
        for i, test_id in enumerate(heatmap_data.index):
            try:
                current_n = int(test_id.split('_')[0][1:])
                if last_n is not None and current_n != last_n:
                    ax.axhline(i, color='black', linewidth=1.5)
                last_n = current_n
            except (ValueError, IndexError): continue 
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        plt.tight_layout()
        plot_filename = os.path.join(output_dir, "pair_correctness_heatmap_by_model.png")
        try:
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"Saved Pair Correctness heatmap to: {plot_filename}")
        except Exception as e:
            print(f"Error saving heatmap plot {plot_filename}: {e}")
        plt.close(fig)
        
    except Exception as plot_e:
        print(f"Error creating pair correctness heatmap: {plot_e}")
        traceback.print_exc()

def load_and_prepare_data(run_dirs: List[str]) -> Tuple[pd.DataFrame | None, List[str], List[int]]:
    """Loads summary metrics from multiple run directories and prepares a DataFrame.
    
    Returns:
        - DataFrame with columns ['Model', 'Provider', 'N_Value', 'Pass_Rate']
        - List of unique models found
        - List of unique N values found (sorted)
    """
    all_data = []
    models_found = set()
    n_values_found = set()

    for run_dir in run_dirs:
        summary_path = os.path.join(run_dir, "summary_metrics.json")
        if not os.path.exists(summary_path):
            print(f"Warning: Summary file not found in {run_dir}, skipping.")
            continue

        try:
            with open(summary_path, 'r', encoding='utf-8') as f:
                summary_data = json.load(f)
            
            # Extract info from directory name (assuming format *_{provider}_{model}_n{N})
            dir_name = os.path.basename(run_dir)
            parts = dir_name.split('_')
            if len(parts) < 4 or not parts[-1].startswith('n'):
                print(f"Warning: Could not parse provider/model/N from dir name: {dir_name}, skipping.")
                continue
                
            provider = parts[-3]
            model_name = parts[-2]
            n_value_str = parts[-1][1:] # Remove 'n' prefix
            
            try:
                n_value = int(n_value_str)
            except ValueError:
                 print(f"Warning: Could not parse N value from {parts[-1]} in dir: {dir_name}, skipping.")
                 continue

            pass_rate = summary_data.get("overall_pass_rate_percent", 0.0)
            
            all_data.append({
                "Model": model_name,
                "Provider": provider,
                "N_Value": n_value,
                "Pass_Rate": pass_rate
            })
            models_found.add(model_name)
            n_values_found.add(n_value)
            
        except json.JSONDecodeError as json_e:
            print(f"Error reading JSON from {summary_path}: {json_e}")
        except Exception as e:
            print(f"Error processing summary file {summary_path}: {e}")

    if not all_data:
        print("No valid summary data loaded.")
        return None, [], []

    df = pd.DataFrame(all_data)
    
    # Sort models based on MODEL_PLOT_ORDER
    models_in_data = list(models_found)
    ordered_models = [m for m in MODEL_PLOT_ORDER if m in models_in_data]
    # Add any models found in data but not in the predefined order list, sorted alphabetically
    other_models = sorted([m for m in models_in_data if m not in MODEL_PLOT_ORDER])
    final_model_order = ordered_models + other_models
    
    sorted_n_values = sorted(list(n_values_found))
    
    print(f"Loaded data for {len(df)} runs. Models (Plot Order): {final_model_order}, N Values: {sorted_n_values}")
    return df, final_model_order, sorted_n_values

def create_n_value_grouped_plot(df: pd.DataFrame, models: List[str], n_values: List[int], output_dir: str):
    """Creates a bar plot with N Value on X-axis, grouped by Model."""
    if df is None or df.empty:
        print("Cannot create N-Value grouped plot: No data.")
        return
        
    num_n_values = len(n_values)
    num_models = len(models)
    if num_n_values == 0 or num_models == 0:
        print("Cannot create N-Value grouped plot: Missing N values or models.")
        return

    # Pivot the data
    pivot_df = df.pivot(index='N_Value', columns='Model', values='Pass_Rate').fillna(0)
    # Ensure columns are in sorted order
    pivot_df = pivot_df[models]
    # Ensure index (N_Value) is sorted
    pivot_df = pivot_df.reindex(sorted(pivot_df.index))
    
    # --- Plotting --- 
    fig, ax = plt.subplots(figsize=(max(8, num_n_values * num_models * 0.5), 6))

    # Get colors for models
    colors = [MODEL_PALETTE.get(model, DEFAULT_COLOR) for model in models]

    pivot_df.plot(kind='bar', ax=ax, color=colors, width=0.8)

    ax.set_title('LLM Pass Rate by Number of Shapes (N)', fontsize=16, pad=20)
    ax.set_xlabel('Number of Shapes (N)', fontsize=12)
    ax.set_ylabel('Pass Rate (%)', fontsize=12)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100.0))
    ax.set_ylim(0, 105) # Extend y-axis slightly
    ax.tick_params(axis='x', rotation=0)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend(title='Model', bbox_to_anchor=(1.02, 1), loc='upper left')
    
    # Add value labels on bars (optional, can get crowded)
    # for container in ax.containers:
    #     ax.bar_label(container, fmt='%.1f%%', label_type='edge', padding=3, fontsize=8)

    plt.tight_layout(rect=[0, 0, 0.88, 1]) # Adjust layout to make space for legend
    
    plot_filename = os.path.join(output_dir, "pass_rate_by_n_value.png")
    try:
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Saved N-Value grouped plot to: {plot_filename}")
    except Exception as e:
        print(f"Error saving plot {plot_filename}: {e}")
    plt.close(fig)

def create_model_grouped_plot(df: pd.DataFrame, models: List[str], n_values: List[int], output_dir: str):
    """Creates a bar plot with Model on X-axis, grouped by N Value."""
    if df is None or df.empty:
        print("Cannot create Model grouped plot: No data.")
        return
    
    num_n_values = len(n_values)
    num_models = len(models)
    if num_n_values == 0 or num_models == 0:
        print("Cannot create Model grouped plot: Missing N values or models.")
        return
        
    # Pivot the data
    pivot_df = df.pivot(index='Model', columns='N_Value', values='Pass_Rate').fillna(0)
    # Ensure columns (N_Value) are sorted
    pivot_df = pivot_df[sorted(pivot_df.columns)]
    # Ensure index (Model) is sorted
    pivot_df = pivot_df.reindex(models)
    
    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(max(8, num_models * num_n_values * 0.4), 6))

    # Use a suitable colormap for N values (e.g., 'viridis', 'plasma', 'Blues')
    cmap = plt.get_cmap('viridis')
    colors = [cmap(i / num_n_values) for i in range(num_n_values)]

    pivot_df.plot(kind='bar', ax=ax, color=colors, width=0.8)

    ax.set_title('LLM Pass Rate by Model (Grouped by N)', fontsize=16, pad=20)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Pass Rate (%)', fontsize=12)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100.0))
    ax.set_ylim(0, 105)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend(title='N Value', bbox_to_anchor=(1.02, 1), loc='upper left')

    # Rotate X axis labels using plt.setp for better compatibility
    if len(models) > 4: 
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust layout
    
    plot_filename = os.path.join(output_dir, "pass_rate_by_model.png")
    try:
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Saved Model grouped plot to: {plot_filename}")
    except Exception as e:
        print(f"Error saving plot {plot_filename}: {e}")
    plt.close(fig)

# --- New summary table plotting function ---
def create_summary_pass_rate_table(df_summary: pd.DataFrame, models_order: List[str], n_values: List[int], output_dir: str):
    """Creates a formatted table image summarizing pass rates by Base Model and N Value."""
    if df_summary is None or df_summary.empty:
        print("Cannot create summary table: No data.")
        return

    try:
        # Add Base_Model column
        df_summary['Base_Model'] = df_summary['Model'].apply(get_base_model_name)
        
        # Ensure N_Value is integer for column headers
        df_summary['N_Value'] = df_summary['N_Value'].astype(int)
        
        # Aggregate: Get the pass rate for each Base_Model / N_Value combo
        # (Data might already be unique per Model/N, but mean() handles potential duplicates)
        agg_data = df_summary.groupby(['Base_Model', 'N_Value'])['Pass_Rate'].mean().reset_index()
        
        # Pivot the table
        table_data = agg_data.pivot(index='Base_Model', columns='N_Value', values='Pass_Rate')
        table_data.index.name = 'Model' # Rename index for display
        
        # Define the desired base model row order
        base_models_in_order = []
        seen_base_models = set()
        for full_model in models_order:
            base = get_base_model_name(full_model)
            if base in table_data.index and base not in seen_base_models:
                base_models_in_order.append(base)
                seen_base_models.add(base)
        other_base_models = sorted([idx for idx in table_data.index if idx not in seen_base_models])
        final_row_order = base_models_in_order + other_base_models
        
        # Ensure columns (N Values) are sorted numerically
        final_col_order = sorted(table_data.columns)
        
        # Reindex and fill missing values with 0
        table_data = table_data.reindex(index=final_row_order, columns=final_col_order).fillna(0)

        # --- Calculate and Add Baseline Rows --- 
        random_chance_rates = {}
        unique_assign_chance_rates = {}
        correct_assign_chance_rates = {}
        
        for n_val in final_col_order:
            prob_random = 0.0
            prob_unique_assign = 0.0
            prob_correct_assign = 0.0
            if n_val > 0:
                # Random Chance: 1 / N^(2N)
                try:
                    prob_random = 1 / (n_val**(2*n_val))
                except OverflowError: pass # Stays 0.0
                
                # Factorial calculation for other two
                fact_n = 0
                try:
                    fact_n = math.factorial(n_val)
                except OverflowError: pass # Stays 0

                if fact_n > 0:
                     # Unique Assignment Chance: 1 / (N!)^2
                    try:
                        prob_unique_assign = 1 / (fact_n * fact_n)
                    except OverflowError: pass # Stays 0.0
                    
                    # Correct Assignment Chance: 1 / N!
                    prob_correct_assign = 1 / fact_n
            
            random_chance_rates[n_val] = prob_random * 100
            unique_assign_chance_rates[n_val] = prob_unique_assign * 100
            correct_assign_chance_rates[n_val] = prob_correct_assign * 100
                
        random_chance_series = pd.Series(random_chance_rates, name="Random Chance")
        unique_assign_series = pd.Series(unique_assign_chance_rates, name="Unique Assign Chance") # New Name
        correct_assign_series = pd.Series(correct_assign_chance_rates, name="Correct Assign Chance") # New Name
    
        # Use pd.concat to add all new rows
        table_data = pd.concat([
            table_data, 
            pd.DataFrame(random_chance_series).T, 
            pd.DataFrame(unique_assign_series).T,
            pd.DataFrame(correct_assign_series).T
        ])
    
        # Define baseline rows (used for styling checks)
        baseline_rows = ["Random Chance", "Unique Assign Chance", "Correct Assign Chance"]
        # Define the desired order for appending to the index
        reversed_baseline_order = baseline_rows[::-1] # Reverse the list
        final_row_order.extend(reversed_baseline_order)
        # Reindex again to ensure correct order including baselines
        table_data = table_data.reindex(index=final_row_order, columns=final_col_order)

        # --- Create Table Image --- 
        fig, ax = plt.subplots(figsize=(max(6, len(final_col_order) * 1.2), 
                                        max(4, len(table_data.index) * 0.5)))
        ax.axis('off') 
        ax.set_title('Overall Pass Rate (%) by Model and N Value (with Baselines)', 
                     fontweight="bold", pad=20, fontsize=14)
        
        # Format data using fixed-point, adjusting precision based on value
        def format_cell(x):
            if pd.isna(x): return "N/A"
            if x == 100: return "100%" # Handle 100% explicitly
            if x == 0: return "0.0%"
            if x >= 0.1: # Use 1 decimal place for most common rates
                return f"{x:.1f}%" 
            if x > 0.0001: # Use more precision for small rates
                return f"{x:.4f}%" 
            return "<0.0001%" # For extremely small values
            
        table_display_data = table_data.map(format_cell)
        table_display_data.columns = [f"N={col}" for col in table_display_data.columns]
        table_obj = ax.table(cellText=table_display_data.values,
                             rowLabels=table_display_data.index,
                             colLabels=table_display_data.columns,
                             cellLoc='center', rowLoc='left', loc='center',
                             colWidths=[0.15]*len(table_display_data.columns))
        table_obj.set_fontsize(10)
        table_obj.scale(1.2, 1.2)
        cmap = plt.get_cmap('viridis') 
        norm = plt.Normalize(vmin=0, vmax=100)
        baseline_row_color = '#e8e8e8'
        for i in range(table_data.shape[0]):
            is_baseline = table_data.index[i] in baseline_rows # Check against updated list
            for j in range(table_data.shape[1]):
                cell = table_obj.get_celld()[(i + 1, j)]
                try:
                    numeric_value = table_data.iloc[i, j]
                except IndexError: continue
                cell_color = baseline_row_color if is_baseline else cmap(norm(numeric_value))
                cell.set_facecolor(cell_color)
                bg_val = numeric_value if not is_baseline else 30
                text_color = 'white' if norm(bg_val) > 0.5 else 'black' 
                cell.get_text().set_color(text_color)
        for k, cell in table_obj.get_celld().items():
            cell.set_edgecolor('lightgrey')
            cell.set_linewidth(0.5)
            is_baseline_idx = table_data.index[k[0]-1] in baseline_rows if k[0]>0 and k[1]==-1 else False
            if k[0] == 0: # Header row
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#f0f0f0')
                cell.get_text().set_color('black')
            elif k[1] == -1: # Index column
                cell.set_text_props(weight='bold', ha='left')
                cell.set_facecolor(baseline_row_color if is_baseline_idx else '#f0f0f0')
                cell.get_text().set_color('black')
                cell.set_width(0.3)
        plt.tight_layout()
        plot_filename = os.path.join(output_dir, "summary_pass_rate_table.png")
        try:
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"Saved Summary Pass Rate table to: {plot_filename}")
        except Exception as e:
            print(f"Error saving table plot {plot_filename}: {e}")
        plt.close(fig)
        
    except Exception as table_e:
        print(f"Error creating summary table: {table_e}")
        traceback.print_exc()

# --- New Pair Correctness table plotting function ---
def create_pair_correctness_table(df_detailed: pd.DataFrame, models_order: List[str], n_values: List[int], output_dir: str):
    """Creates a formatted table image summarizing pair correctness rates by Base Model and N Value."""
    # Check for the new proportion column
    if df_detailed is None or df_detailed.empty or 'Pair_Correctness_Proportion' not in df_detailed.columns:
        print("Cannot create pair correctness table: Missing data or 'Pair_Correctness_Proportion' column.")
        return

    try:
        # Add Base_Model column
        df_detailed['Base_Model'] = df_detailed['Model'].apply(get_base_model_name)
        
        # Ensure N_Value is integer for column headers
        df_detailed['N_Value'] = df_detailed['N_Value'].astype(int)
        
        # Aggregate: Calculate the MEAN Pair Correctness PROPORTION
        agg_data = df_detailed.groupby(['Base_Model', 'N_Value'])['Pair_Correctness_Proportion'].mean().reset_index()
        agg_data.rename(columns={'Pair_Correctness_Proportion': 'Pair_Correctness_Rate'}, inplace=True)
        agg_data['Pair_Correctness_Rate'] *= 100 # Convert mean proportion (0-1) to percentage
        
        # Pivot the table
        table_data = agg_data.pivot(index='Base_Model', columns='N_Value', values='Pair_Correctness_Rate')
        table_data.index.name = 'Model' # Rename index for display
        
        # --- Reorder rows and columns (similar to other table) --- 
        base_models_in_order = []
        seen_base_models = set()
        for full_model in models_order:
            base = get_base_model_name(full_model)
            if base in table_data.index and base not in seen_base_models:
                base_models_in_order.append(base)
                seen_base_models.add(base)
        other_base_models = sorted([idx for idx in table_data.index if idx not in seen_base_models])
        final_row_order = base_models_in_order + other_base_models
        final_col_order = sorted(table_data.columns)
        table_data = table_data.reindex(index=final_row_order, columns=final_col_order).fillna(0)

        # --- Create Table Image (Similar styling to summary table) --- 
        fig, ax = plt.subplots(figsize=(max(6, len(final_col_order) * 1.2), 
                                        max(4, len(final_row_order) * 0.5)))
        ax.axis('off')
        ax.set_title('Pair Correctness Rate (%) by Model and N Value', 
                     fontweight="bold", pad=20, fontsize=14)

        # Format data using fixed-point, adjusting precision based on value
        def format_cell(x):
            if pd.isna(x): return "N/A"
            if x == 100: return "100%"
            if x == 0: return "0.0%"
            if x >= 0.1: return f"{x:.1f}%"
            if x > 0.0001: return f"{x:.4f}%"
            return "<0.0001%"
        table_display_data = table_data.map(format_cell)
        table_display_data.columns = [f"N={col}" for col in table_display_data.columns]

        table_obj = ax.table(cellText=table_display_data.values,
                             rowLabels=table_display_data.index,
                             colLabels=table_display_data.columns,
                             cellLoc='center', rowLoc='left', loc='center',
                             colWidths=[0.15]*len(table_display_data.columns))
                             
        table_obj.set_fontsize(10)
        table_obj.scale(1.2, 1.2)

        cmap = plt.get_cmap('viridis')
        norm = plt.Normalize(vmin=0, vmax=100)
        
        for i in range(table_data.shape[0]):
            for j in range(table_data.shape[1]):
                cell = table_obj.get_celld()[(i + 1, j)]
                try:
                    numeric_value = table_data.iloc[i, j]
                except IndexError: continue
                cell.set_facecolor(cmap(norm(numeric_value)))
                text_color = 'white' if norm(numeric_value) > 0.5 else 'black' 
                cell.get_text().set_color(text_color)
                
        # Style headers and index column
        for k, cell in table_obj.get_celld().items():
            cell.set_edgecolor('lightgrey')
            cell.set_linewidth(0.5)
            if k[0] == 0: # Header row
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#f0f0f0')
                cell.get_text().set_color('black')
            elif k[1] == -1: # Index column
                cell.set_text_props(weight='bold', ha='left')
                cell.set_facecolor('#f0f0f0')
                cell.get_text().set_color('black')
                cell.set_width(0.3)

        plt.tight_layout()

        plot_filename = os.path.join(output_dir, "pair_correctness_rate_table.png")
        try:
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"Saved Pair Correctness Rate table to: {plot_filename}")
        except Exception as e:
            print(f"Error saving table plot {plot_filename}: {e}")
        plt.close(fig)
        
    except Exception as table_e:
        print(f"Error creating pair correctness table: {table_e}")
        traceback.print_exc()

# --- New Sub-Item Correctness table plotting function ---
def create_subitem_correctness_table(df_detailed: pd.DataFrame, models_order: List[str], n_values: List[int], output_dir: str):
    """Creates a formatted table image summarizing sub-item correctness rates by Base Model and N Value."""
    # Check for the required column
    if df_detailed is None or df_detailed.empty or 'Correctness_Rate' not in df_detailed.columns:
        print("Cannot create sub-item correctness table: Missing data or 'Correctness_Rate' column.")
        return

    try:
        # Add Base_Model column
        df_detailed['Base_Model'] = df_detailed['Model'].apply(get_base_model_name)
        
        # Ensure N_Value is integer for column headers
        df_detailed['N_Value'] = df_detailed['N_Value'].astype(int)
        
        # Aggregate: Calculate the MEAN Sub-Item Correctness Rate
        agg_data = df_detailed.groupby(['Base_Model', 'N_Value'])['Correctness_Rate'].mean().reset_index()
        # Rate is already percentage (0-100)
        
        # Pivot the table
        table_data = agg_data.pivot(index='Base_Model', columns='N_Value', values='Correctness_Rate')
        table_data.index.name = 'Model' # Rename index for display
        
        # --- Reorder rows and columns (similar to other tables) --- 
        base_models_in_order = []
        seen_base_models = set()
        for full_model in models_order:
            base = get_base_model_name(full_model)
            if base in table_data.index and base not in seen_base_models:
                base_models_in_order.append(base)
                seen_base_models.add(base)
        other_base_models = sorted([idx for idx in table_data.index if idx not in seen_base_models])
        final_row_order = base_models_in_order + other_base_models
        final_col_order = sorted(table_data.columns)
        table_data = table_data.reindex(index=final_row_order, columns=final_col_order).fillna(0)

        # --- Create Table Image (Similar styling to other tables) --- 
        fig, ax = plt.subplots(figsize=(max(6, len(final_col_order) * 1.2), 
                                        max(4, len(final_row_order) * 0.5)))
        ax.axis('off')
        ax.set_title('Sub-Item Correctness Rate (%) by Model and N Value', 
                     fontweight="bold", pad=20, fontsize=14)

        # Format data using fixed-point, adjusting precision based on value
        def format_cell(x):
            if pd.isna(x): return "N/A"
            if x == 100: return "100%"
            if x == 0: return "0.0%"
            if x >= 0.1: return f"{x:.1f}%"
            if x > 0.0001: return f"{x:.4f}%"
            return "<0.0001%"
            
        table_display_data = table_data.map(format_cell)
        table_display_data.columns = [f"N={col}" for col in table_display_data.columns]

        table_obj = ax.table(cellText=table_display_data.values,
                             rowLabels=table_display_data.index,
                             colLabels=table_display_data.columns,
                             cellLoc='center', rowLoc='left', loc='center',
                             colWidths=[0.15]*len(table_display_data.columns))
                             
        table_obj.set_fontsize(10)
        table_obj.scale(1.2, 1.2)

        cmap = plt.get_cmap('viridis')
        norm = plt.Normalize(vmin=0, vmax=100)
        
        for i in range(table_data.shape[0]):
            for j in range(table_data.shape[1]):
                cell = table_obj.get_celld()[(i + 1, j)]
                try:
                    numeric_value = table_data.iloc[i, j]
                except IndexError: continue
                cell.set_facecolor(cmap(norm(numeric_value)))
                text_color = 'white' if norm(numeric_value) > 0.5 else 'black' 
                cell.get_text().set_color(text_color)
                
        # Style headers and index column
        for k, cell in table_obj.get_celld().items():
            cell.set_edgecolor('lightgrey')
            cell.set_linewidth(0.5)
            if k[0] == 0: # Header row
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#f0f0f0')
                cell.get_text().set_color('black')
            elif k[1] == -1: # Index column
                cell.set_text_props(weight='bold', ha='left')
                cell.set_facecolor('#f0f0f0')
                cell.get_text().set_color('black')
                cell.set_width(0.3)

        plt.tight_layout()

        plot_filename = os.path.join(output_dir, "subitem_correctness_rate_table.png") # New filename
        try:
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"Saved Sub-Item Correctness Rate table to: {plot_filename}")
        except Exception as e:
            print(f"Error saving table plot {plot_filename}: {e}")
        plt.close(fig)
        
    except Exception as table_e:
        print(f"Error creating sub-item correctness table: {table_e}")
        traceback.print_exc()

# --- Main (for testing plotter directly) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate plots from CeledonBenchCircuits summary metrics.")
    parser.add_argument("--results_dir", default="../../results", help="Base directory containing run results.")
    parser.add_argument("--output_dir", default="../../analysis_plots", help="Directory to save plots.")
    args = parser.parse_args()

    # Find all run directories with summary files
    summary_files = glob.glob(os.path.join(args.results_dir, "*", "summary_metrics.json"))
    run_dirs = sorted([os.path.dirname(f) for f in summary_files])

    if not run_dirs:
        print(f"No run directories with summary files found in {args.results_dir}")
    else:
        print(f"Found {len(run_dirs)} run directories for analysis.")
        os.makedirs(args.output_dir, exist_ok=True)
        
        df, models, n_values = load_and_prepare_data(run_dirs)
        
        if df is not None and not df.empty:
            create_n_value_grouped_plot(df, models, n_values, args.output_dir)
            create_model_grouped_plot(df, models, n_values, args.output_dir)
            create_summary_pass_rate_table(df, models, n_values, args.output_dir)
            create_pair_correctness_table(df, models, n_values, args.output_dir)
            create_subitem_correctness_table(df, models, n_values, args.output_dir)
            create_pass_fail_heatmap(df, models, args.output_dir)
            create_pair_correctness_heatmap(df, models, args.output_dir)
        else:
            print("No data loaded, cannot generate plots.")

    print("Plotter script finished.") 