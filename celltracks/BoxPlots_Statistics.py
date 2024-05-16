
import ipywidgets as widgets
from ipywidgets import Layout, VBox, Button, Accordion, SelectMultiple, IntText
import scipy.stats as stats
import numpy as np
from multiprocessing import Pool, get_context
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
from scipy.stats import zscore
from scipy.stats import ks_2samp
from sklearn.preprocessing import MinMaxScaler


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

def format_scientific_for_ticks(x):
    """Format p-values for ticks: use scientific notation for values below 0.001, otherwise use standard notation."""
    if x < 0.001:
        return f"{x:.1e}"
    else:
        return f"{x:.4f}"

def format_p_value(x):
    """Format p-values to four significant digits."""
    if x < 0.001:
        return "< 0.001"
    else:
        return f"{x:.4g}"  # .4g ensures four significant digits


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
    log_matrix = safe_log10_p_values(matrix.fillna(1))

    # Define the normalization range
    vmin = -np.log10(0.1)  # Set vmin to the log-transformed value of 0.1
    vmax = np.max(log_matrix[np.isfinite(log_matrix)])

    if vmin > vmax:
      vmin = vmax        

    # Format annotations
    formatted_annotations = matrix.applymap(lambda x: format_p_value(x) if pd.notna(x) else "NaN")

    # Plot the heatmap without the color bar
    heatmap = sns.heatmap(log_matrix, ax=ax, cmap=cmap, annot=formatted_annotations,
                          fmt="", xticklabels=matrix.columns, yticklabels=matrix.index, cbar=False, vmin=vmin, vmax=vmax)
    ax.set_title(title)

    # Create a color bar with conditional formatting for ticks
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = ax.figure.colorbar(sm, ax=ax)

    # Set custom ticks and labels for the color bar
    num_ticks = 5
    tick_locs = np.linspace(vmin, vmax, num_ticks)
    tick_labels = [format_scientific_for_ticks(10**-tick) for tick in tick_locs]
    cbar.set_ticks(tick_locs)
    cbar.set_ticklabels(tick_labels)

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

    p_value = (count_extreme + 1) / (n_iterations + 1)
    return p_value

def run_batch(params):
    """Function to run a batch of randomization tests."""
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

    # Create a multiprocessing pool with the 'spawn' start method to avoid threading issues
    with get_context("spawn").Pool(n_cores) as pool:
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


def calculate_ks_p_value(df1, df2, column):
    """
    Calculate the KS p-value for a given column between two dataframes.

    Parameters:
    df1 (pandas.DataFrame): Original DataFrame.
    df2 (pandas.DataFrame): DataFrame after downsampling.
    column (str): Column name to compare.

    Returns:
    float: KS p-value.
    """
    return ks_2samp(df1[column].dropna(), df2[column].dropna())[1]

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
        print("No conditions selected for plotting, therefore all available conditions are selected by default")        
        selected_conditions = df[Conditions].unique().tolist()
    
    n_selected_conditions = len(selected_conditions)

    effect_size_matrices = {}
    p_value_matrices = {}
    bonferroni_matrices = {}

    # Use only selected and ordered conditions
    filtered_df = df[df[Conditions].isin(selected_conditions)].copy()

    unique_conditions = filtered_df[Conditions].unique().tolist()
    num_comparisons = len(unique_conditions) * (len(unique_conditions) - 1) // 2
    n_iterations = 1000

    for var in variables_to_plot:
        effect_size_matrices[var] = pd.DataFrame(0, index=unique_conditions, columns=unique_conditions)
        p_value_matrices[var] = pd.DataFrame(1, index=unique_conditions, columns=unique_conditions)
        bonferroni_matrices[var] = pd.DataFrame(1, index=unique_conditions, columns=unique_conditions)

        for cond1, cond2 in itertools.combinations(unique_conditions, 2):
            group1 = filtered_df[filtered_df[Conditions] == cond1][var]
            group2 = filtered_df[filtered_df[Conditions] == cond2][var]

            effect_size = abs(cohen_d(group1, group2))

            if method == 't-test':
                p_value = perform_t_test(filtered_df, cond1, cond2, var)
            if method == 'randomization test':
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
        pdf_pages = PdfPages(f"{Results_Folder}/pdf/{var}_Boxplots_and_Statistics.pdf")
        pdf_pages.savefig(fig)  
        pdf_pages.close()
        plt.show()   
        
def count_tracks_by_condition_and_repeat(df, Results_Folder, condition_col='Condition', repeat_col='Repeat', track_id_col='Unique_ID'):
    """
    Counts the number of unique tracks for each combination of condition and repeat in the given DataFrame and
    saves a stacked histogram plot as a PDF in the QC folder with annotations for each stack.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data.
    Results_Folder (str): The base folder where the results will be saved.
    condition_col (str): The name of the column representing the condition. Default is 'Condition'.
    repeat_col (str): The name of the column representing the repeat. Default is 'Repeat'.
    track_id_col (str): The name of the column representing the track ID. Default is 'Unique_ID'.
    """
    track_counts = df.groupby([condition_col, repeat_col])[track_id_col].nunique()
    track_counts_df = track_counts.reset_index()
    track_counts_df.rename(columns={track_id_col: 'Number_of_Tracks'}, inplace=True)

    # Pivot the data for plotting
    pivot_df = track_counts_df.pivot(index=condition_col, columns=repeat_col, values='Number_of_Tracks').fillna(0)

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = pivot_df.plot(kind='bar', stacked=True, ax=ax)
    ax.set_xlabel('Condition')
    ax.set_ylabel('Number of Tracks')
    ax.set_title('Stacked Histogram of Track Counts per Condition and Repeat')
    ax.legend(title=repeat_col)
    ax.grid(axis='y', linestyle='--')

    # Hide horizontal grid lines
    ax.yaxis.grid(False)

    # Add number annotations on each stack
    for bar in bars.patches:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_y() + bar.get_height() / 2,
                int(bar.get_height()),
                ha='center', va='center', color='black', fontweight='bold', fontsize=8)

    # Save the plot as a PDF
    pdf_file = os.path.join(Results_Folder, 'Track_Counts_Histogram.pdf')
    plt.savefig(pdf_file, bbox_inches='tight')
    print(f"Saved histogram to {pdf_file}")

    plt.show()

    return track_counts_df        


def heatmap_comparison(df, Results_Folder, Conditions, normalization='minmax', variables_per_page=40):
    # Get all the selectable columns
    variables_to_plot = get_selectable_columns(df)

    # Work on a copy of the DataFrame to avoid SettingWithCopyWarning
    df_mod = df.copy()
    
    # Drop rows where all elements are NaNs in the variables_to_plot columns
    df_mod = df_mod.dropna(subset=variables_to_plot)

    # Normalize the entire dataset for each variable
    if normalization == 'zscore':
        df_mod.loc[:, variables_to_plot] = df_mod[variables_to_plot].apply(zscore)
    elif normalization == 'minmax':
        scaler = MinMaxScaler(feature_range=(-1, 1))
        df_mod.loc[:, variables_to_plot] = df_mod[variables_to_plot].apply(lambda x: scaler.fit_transform(x.values.reshape(-1, 1)).flatten())
    else:
        raise ValueError("Unsupported normalization type. Use 'zscore' or 'minmax'.")

    # Compute median for each variable across Conditions
    median_values = df_mod.groupby(Conditions)[variables_to_plot].median().transpose()

    # Number of pages
    total_variables = len(variables_to_plot)
    num_pages = int(np.ceil(total_variables / variables_per_page))

    # Initialize an empty DataFrame to store all pages' data
    all_pages_data = pd.DataFrame()

    # Create a PDF file to save the heatmaps
    with PdfPages(f"{Results_Folder}/Heatmaps_Normalized_Median_Values_by_Condition.pdf") as pdf:
        for page in range(num_pages):
            start = page * variables_per_page
            end = min(start + variables_per_page, total_variables)
            page_data = median_values.iloc[start:end]

            # Append this page's data to the all_pages_data DataFrame
            all_pages_data = pd.concat([all_pages_data, page_data])

            plt.figure(figsize=(16, 10))
            sns.heatmap(page_data, cmap='coolwarm', annot=True, linewidths=.1)
            plt.title(f"{normalization.capitalize()} Normalized Median Values of Variables by Condition (Page {page + 1})")
            plt.tight_layout()

            pdf.savefig()  # saves the current figure into a pdf page
            plt.show()
            plt.close()

    # Save all pages data to a single CSV file
    all_pages_data.to_csv(f"{Results_Folder}/Normalized_Median_Values_by_Condition.csv")

    print(f"Heatmaps saved to {Results_Folder}/Heatmaps_Normalized_Median_Values_by_Condition.pdf")
    print(f"All data saved to {Results_Folder}/Normalized_Median_Values_by_Condition.csv")



def balance_dataset(df, condition_col='Condition', repeat_col='Repeat', track_id_col='Unique_ID', random_seed=None):
    """
    Balances the dataset by downsampling tracks for each condition and repeat combination.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data.
    condition_col (str): The name of the column representing the condition.
    repeat_col (str): The name of the column representing the repeat.
    track_id_col (str): The name of the column representing the track ID.
    random_seed (int, optional): The seed for the random number generator. Default is None.

    Returns:
    pandas.DataFrame: A new DataFrame with balanced track counts.
    """
    # Group by condition and repeat, and find the minimum track count
    min_track_count = df.groupby([condition_col, repeat_col])[track_id_col].nunique().min()

    # Function to sample min_track_count tracks from each group
    def sample_tracks(group):
        return group.sample(n=min_track_count, random_state=random_seed)

    # Apply sampling to each group and concatenate the results
    balanced_merged_tracks_df = df.groupby([condition_col, repeat_col]).apply(sample_tracks).reset_index(drop=True)

    return balanced_merged_tracks_df

