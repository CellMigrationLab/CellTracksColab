
import ipywidgets as widgets
from ipywidgets import Layout, VBox, Button, Accordion, SelectMultiple, IntText
import scipy.stats as stats
import numpy as np
from multiprocessing import Pool
import pandas as pd
import os
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FixedLocator
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from matplotlib.gridspec import GridSpec
import requests


def get_selectable_columns(df):
    exclude_cols = ['Condition', 'experiment_nb', 'File_name', 'Repeat', 'Unique_ID', 'LABEL', 'TRACK_INDEX', 'TRACK_ID', 'TRACK_X_LOCATION', 'TRACK_Y_LOCATION', 'TRACK_Z_LOCATION', 'Exemplar', 'TRACK_STOP', 'TRACK_START', 'Cluster_UMAP', 'Cluster_tsne']
    return [col for col in df.columns if (df[col].dtype.kind in 'biufc') and (col not in exclude_cols)]

def display_variable_checkboxes(selectable_columns):
    variable_checkboxes = [widgets.Checkbox(value=False, description=col) for col in selectable_columns]
    checkboxes_widget = widgets.VBox([
        widgets.Label('Variables to Plot:'),
        widgets.GridBox(variable_checkboxes, layout=widgets.Layout(grid_template_columns="repeat(3, 300px)"))
    ])
    return variable_checkboxes, checkboxes_widget

def create_condition_selector(df, column_name):
    conditions = df[column_name].unique()
    return SelectMultiple(
        options=conditions,
        description='Conditions:',
        disabled=False,
        layout=Layout(width='100%')
    )

def display_condition_selection(df, column_name):
    condition_selector = create_condition_selector(df, column_name)
    condition_accordion = Accordion(children=[VBox([condition_selector])])
    condition_accordion.set_title(0, 'Select Conditions')
    return condition_selector, condition_accordion

def safe_log10_p_values(matrix):
    """Apply a safe logarithmic transformation to p-values, handling p=1 specifically."""
    # Replace non-positive values with a very small number just greater than 0
    small_value = np.nextafter(0, 1)
    adjusted_matrix = np.where(matrix > 0, matrix, small_value)

    logged_matrix = -np.log10(adjusted_matrix)
    logged_matrix[matrix == 1] = -np.log10(0.999)
    return logged_matrix

def plot_heatmap(ax, matrix, title, cmap='viridis'):
    """Plot a heatmap with logarithmic scaling of p-values and real p-values as annotations."""
    log_matrix = safe_log10_p_values(matrix.fillna(1))  # Handle NaN values by assuming non-significant (p=1)

    # Use LogNorm for the color scale, ensuring no invalid vmin or vmax
    vmin = np.min(log_matrix[np.isfinite(log_matrix)])
    vmax = np.max(log_matrix[np.isfinite(log_matrix)])
    norm = LogNorm(vmin=max(vmin, 1e-10), vmax=vmax)  # Ensure vmin is positive and small

    # Annotations as real p-values, formatted in scientific notation, special handling for p=1 to show as "1.00"
    formatted_annotations = matrix.applymap(lambda x: f"{x:.2e}" if pd.notna(x) and x != 1 else "1.00")

    # Plot the heatmap with string annotations for clarity
    sns.heatmap(log_matrix, ax=ax, cmap=cmap, norm=norm, annot=formatted_annotations,
                fmt="", xticklabels=matrix.columns, yticklabels=matrix.index)
    ax.set_title(title)

def cohen_d(group1, group2):
    """Calculate Cohen's d for measuring effect size between two groups."""
    diff = group1.mean() - group2.mean()
    n1, n2 = len(group1), len(group2)
    var1 = group1.var(ddof=1)  # ddof=1 for sample variance
    var2 = group2.var(ddof=1)
    pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
    d = diff / np.sqrt(pooled_var)
    return d

def perform_randomization_test(df, cond1, cond2, var, n_iterations=1000):
    """Perform a randomization test using Cohen's d as the effect size metric."""
    group1 = df[df['Condition'] == cond1][var]
    group2 = df[df['Condition'] == cond2][var]
    observed_effect_size = cohen_d(group1, group2)
    combined = np.concatenate([group1, group2])
    count_extreme = 0
    # Perform the randomization test
    for _ in range(n_iterations):
        np.random.shuffle(combined)
        new_group1 = combined[:len(group1)]
        new_group2 = combined[len(group1):]
        new_effect_size = cohen_d(new_group1, new_group2)
        if abs(new_effect_size) >= abs(observed_effect_size):
            count_extreme += 1

    p_value = (count_extreme + 1) / (n_iterations +1)

    return p_value

def run_batch(params):
    """ Function to run a batch of randomization tests. """
    group1, group2, combined, observed_effect_size, n_iter = params
    count_extreme = 0
    for _ in range(n_iter):
        np.random.shuffle(combined)
        new_group1 = combined[:len(group1)]
        new_group2 = combined[len(group1):]
        new_effect_size = cohen_d(new_group1, new_group2)
        if abs(new_effect_size) >= abs(observed_effect_size):
            count_extreme += 1
    return count_extreme

def perform_randomization_test_parallel(df, cond1, cond2, var, n_iterations=1000, n_cores=4):
    group1 = df[df['Condition'] == cond1][var].to_numpy()
    group2 = df[df['Condition'] == cond2][var].to_numpy()
    observed_effect_size = cohen_d(group1, group2)
    combined = np.concatenate([group1, group2])
    
    # Split iterations across multiple cores
    iter_per_core = [(group1, group2, combined.copy(), observed_effect_size, n_iterations // n_cores) for _ in range(n_cores)]
    for i in range(n_iterations % n_cores):
        iter_per_core[i] = (group1, group2, combined.copy(), observed_effect_size, iter_per_core[i][-1] + 1)

    # Create a multiprocessing pool and map the execution to the pool
    with Pool(n_cores) as pool:
        results = pool.map(run_batch, iter_per_core)

    total_extreme = sum(results)
    p_value = (total_extreme + 1) / (n_iterations + 1)
    return p_value

def perform_t_test(df, cond1, cond2, var):
    """Perform a t-test using the average of each repeat for the given conditions and calculate Cohen's d."""
    group1 = df[df['Condition'] == cond1].groupby('Repeat')[var].mean()
    group2 = df[df['Condition'] == cond2].groupby('Repeat')[var].mean()

    # Perform the t-test on these group means
    t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)  # Use Welch's t-test for unequal variances

    return p_value

def plot_selected_vars(button, variable_checkboxes, df, Conditions, Results_Folder, condition_selector, stat_method_selector):
    plt.clf()  # Clear the current figure before creating a new plot
    print("Plotting in progress...")

  # Get selected variables
    variables_to_plot = [box.description for box in variable_checkboxes if box.value]
    n_plots = len(variables_to_plot)
    method = stat_method_selector.value

    if n_plots == 0:
        print("No variables selected for plotting")
        return

  # Get selected conditions
    selected_conditions = condition_selector.value
    n_selected_conditions = len(selected_conditions)

    if n_selected_conditions == 0:
        print("No conditions selected for plotting")
        return

    effect_size_matrices = {}
    p_value_matrices = {}
    bonferroni_matrices = {}

# Use only selected and ordered conditions
    filtered_df = df[df[Conditions].isin(selected_conditions)].copy()

    unique_conditions = filtered_df[Conditions].unique().tolist()
    num_comparisons = len(unique_conditions) * (len(unique_conditions) - 1) // 2
    n_iterations = 10000

    for var in variables_to_plot:
        pdf_pages = PdfPages(f"{Results_Folder}/pdf/{var}_Boxplots_and_Statistics.pdf")
        effect_size_matrices[var] = pd.DataFrame(0, index=unique_conditions, columns=unique_conditions)
        p_value_matrices[var] = pd.DataFrame(1, index=unique_conditions, columns=unique_conditions)
        bonferroni_matrices[var] = pd.DataFrame(1, index=unique_conditions, columns=unique_conditions)

        for cond1, cond2 in itertools.combinations(unique_conditions, 2):
            group1 = filtered_df[filtered_df[Conditions] == cond1][var]
            group2 = filtered_df[filtered_df[Conditions] == cond2][var]

            effect_size = cohen_d(group1, group2)

            if method == 't-test':
                p_value = perform_t_test(filtered_df, cond1, cond2, var)
            if method == 'randomization test':
                #p_value = perform_randomization_test(filtered_df, cond1, cond2, var, n_iterations=n_iterations)
                p_value = perform_randomization_test_parallel(filtered_df, cond1, cond2, var, n_iterations=n_iterations)


            # Set and mirror effect sizes and p-values
            effect_size_matrices[var].loc[cond1, cond2] = effect_size_matrices[var].loc[cond2, cond1] = effect_size
            p_value_matrices[var].loc[cond1, cond2] = p_value_matrices[var].loc[cond2, cond1] = p_value
            bonferroni_corrected_p_value = min(p_value * num_comparisons, 1.0)
            bonferroni_matrices[var].loc[cond1, cond2] = bonferroni_matrices[var].loc[cond2, cond1] = bonferroni_corrected_p_value

        # Save to CSV
        combined_df = pd.concat([
            effect_size_matrices[var].rename(columns=lambda x: f"{x} (Effect Size)"),
            p_value_matrices[var].rename(columns=lambda x: f"{x} ({method} P-Value)"),
            bonferroni_matrices[var].rename(columns=lambda x: f"{x} ({method} Bonferroni-corrected P-Value)")
        ], axis=1)

        combined_df.to_csv(f"{Results_Folder}/csv/{var}_statistics_combined.csv")

    # Create a new figure
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 3, height_ratios=[1.5, 1])
        ax_box = fig.add_subplot(gs[0, :])
    # Extract the data for this variable
        data_for_var = df[[Conditions, var, 'Repeat', 'File_name' ]]
    # Save the data_for_var to a CSV for replotting
        data_for_var.to_csv(f"{Results_Folder}/csv/{var}_boxplot_data.csv", index=False)

    # Calculate the Interquartile Range (IQR) using the 25th and 75th percentiles
        Q1 = df[var].quantile(0.2)
        Q3 = df[var].quantile(0.8)
        IQR = Q3 - Q1

    # Define bounds for the outliers
        multiplier = 10
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR

    # Plotting
        sns.boxplot(x=Conditions, y=var, data=filtered_df, ax=ax_box, color='lightgray')  # Boxplot
        sns.stripplot(x=Conditions, y=var, data=filtered_df, ax=ax_box, hue='Repeat', dodge=True, jitter=True, alpha=0.2)  # Individual data points
        ax_box.set_ylim([max(min(filtered_df[var]), lower_bound), min(max(filtered_df[var]), upper_bound)])
        ax_box.set_title(f"{var}")
        ax_box.set_xlabel('Condition')
        ax_box.set_ylabel(var)
        tick_labels = ax_box.get_xticklabels()
        tick_locations = ax_box.get_xticks()
        ax_box.xaxis.set_major_locator(FixedLocator(tick_locations))
        ax_box.set_xticklabels(tick_labels, rotation=90)
        ax_box.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Repeat')

    # Statistical Analyses and Heatmaps

      # Effect Size heatmap
        ax_d = fig.add_subplot(gs[1, 0])
        sns.heatmap(effect_size_matrices[var].fillna(0), annot=True, cmap="viridis", cbar=True, square=True, ax=ax_d, vmax=1)
        ax_d.set_title(f"Effect Size (Cohen's d)")

      # p-value heatmap using the new function
        ax_p = fig.add_subplot(gs[1, 1])
        plot_heatmap(ax_p, p_value_matrices[var], f"{method} p-value")

      # Bonferroni corrected p-value heatmap using the new function
        ax_bonf = fig.add_subplot(gs[1, 2])
        plot_heatmap(ax_bonf, bonferroni_matrices[var], "Bonferroni-corrected p-value")

        plt.tight_layout()
        pdf_pages.savefig(fig)