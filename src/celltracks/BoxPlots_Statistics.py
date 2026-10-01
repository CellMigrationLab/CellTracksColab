
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
import warnings
from matplotlib.gridspec import GridSpec
import requests
from scipy.stats import ks_2samp


computed_metrics = {
    'Track Metrics': [
        'Track Duration', 'Mean Speed', 'Median Speed', 'Max Speed', 'Min Speed',
        'Speed Standard Deviation', 'Directionality', 'Total Distance Traveled', 'Spatial Coverage', 'Tortuosity', 'Total Turning Angle', 'FMI_x_plus','FMI_x_minus', 'FMI_y_plus','FMI_y_minus'],
    'Rolling Track Metrics': [
        'Mean Speed Rolling', 'Median Speed Rolling', 'Max Speed Rolling',
        'Min Speed Rolling', 'Min Speed Rolling Rolling',
        'Speed Standard Deviation Rolling',
        'Total Distance Traveled Rolling', 'Directionality Rolling', 'Tortuosity Rolling', 'Total Turning Angle Rolling', 'Spatial Coverage Rolling'
    ],
    'Morphological Metrics': [
        'MEAN_', 'MEDIAN_', 'STD_', 'MIN_', 'MAX_'
    ],
    'Distance to ROI Metrics': [
        'MaxDistance_', 'MinDistance_', 'StartDistance_', 'EndDistance_',
        'MedianDistance_', 'StdDevDistance_', 'DirectionMovement_', 
        'AvgRateChange_', 'PercentageChange_', 'TrendSlope_'
    ]
}


def categorize_columns(df):
    exclude_cols = ['Condition', 'experiment_nb', 'File_name', 'Repeat', 'Unique_ID', 'LABEL', 'TRACK_INDEX', 'TRACK_ID', 'TRACK_X_LOCATION', 'TRACK_Y_LOCATION', 'TRACK_Z_LOCATION', 'Exemplar', 'TRACK_STOP', 'TRACK_START', 'Cluster_UMAP', 'Cluster_tsne']
    all_columns = [col for col in df.columns if col not in exclude_cols]
    
    tracking_software_metrics = [
        'MAX_DISTANCE_TRAVELED',
        'MEAN_STRAIGHT_LINE_SPEED',
        'MEAN_DIRECTIONAL_CHANGE'
    ]
    
    categorized_columns = {
        'Metrics Computed in CellTracksColab': {
            'Track Metrics': [],
            'Rolling Track Metrics': [],
            'Morphological Metrics': [],
            'Distance to ROI Metrics': []
        },
        'Metrics Imported from your Tracking Software': []
    }
    
    for col in all_columns:
        if col in tracking_software_metrics:
            categorized_columns['Metrics Imported from your Tracking Software'].append(col)
        else:
            added = False
            for category, metrics in computed_metrics.items():
                if category in categorized_columns['Metrics Computed in CellTracksColab']:
                    if col in metrics:
                        categorized_columns['Metrics Computed in CellTracksColab'][category].append(col)
                        added = True
                        break
                    # Handle prefix matching for Morphological Metrics and Distance to ROI Metrics
                    if (category in ['Morphological Metrics', 'Distance to ROI Metrics'] 
                            and any(col.startswith(prefix) for prefix in metrics)):
                        categorized_columns['Metrics Computed in CellTracksColab'][category].append(col)
                        added = True
                        break
            if not added:
                categorized_columns['Metrics Imported from your Tracking Software'].append(col)
    
    return categorized_columns

def get_selectable_columns(df):
    exclude_cols = ['Condition', 'experiment_nb', 'File_name', 'Repeat', 'Unique_ID', 'LABEL', 'TRACK_INDEX', 'TRACK_ID', 'TRACK_X_LOCATION', 'TRACK_Y_LOCATION', 'TRACK_Z_LOCATION', 'Exemplar', 'TRACK_STOP', 'TRACK_START', 'Cluster_UMAP', 'Cluster_tsne']
    return [col for col in df.columns if (df[col].dtype.kind in 'biufc') and (col not in exclude_cols)]

def get_selectable_columns_plots(df):
    computed_patterns = ['directionality', 'tortuosity', 'area', 'perimeter', 'basic_metric', 'advanced_metric']
    exclude_cols = ['Condition', 'experiment_nb', 'File_name', 'Repeat', 'Unique_ID', 'LABEL', 'TRACK_INDEX', 'TRACK_ID', 'TRACK_X_LOCATION', 'TRACK_Y_LOCATION', 'TRACK_Z_LOCATION', 'Exemplar', 'TRACK_STOP', 'TRACK_START', 'Cluster_UMAP', 'Cluster_tsne']
    columns = [col for col in df.columns if (df[col].dtype.kind in 'biufc') and (col not in exclude_cols)]
    
    computed_columns = [col for col in columns if any(pattern in col for pattern in computed_patterns)]
    imported_columns = [col for col in columns if col not in computed_columns]
    
    return {'Computed in CellTracksColab': computed_columns, 'Imported from Tracking Software': imported_columns}

def display_variable_checkboxes(categorized_columns):
    def create_select_all_checkbox(category, checkboxes):
        def toggle_all(change):
            for checkbox in checkboxes:
                checkbox.value = change.new
        select_all_checkbox = widgets.Checkbox(value=False, description=f'Select All {category}')
        select_all_checkbox.observe(toggle_all, names='value')
        return select_all_checkbox

    accordion_items = []
    checkboxes_dict = {}
    for main_category, sub_categories in categorized_columns.items():
        if isinstance(sub_categories, dict):
            sub_accordion_items = []
            checkboxes_dict[main_category] = {}
            for sub_category, columns in sub_categories.items():
                checkboxes = [widgets.Checkbox(value=False, description=col) for col in columns]
                checkboxes_grid = widgets.GridBox(checkboxes, layout=widgets.Layout(grid_template_columns="repeat(3, 300px)"))
                sub_accordion_items.append(VBox([create_select_all_checkbox(sub_category, checkboxes), checkboxes_grid]))
                checkboxes_dict[main_category][sub_category] = checkboxes
            sub_accordion = Accordion(children=sub_accordion_items)
            for i, sub_category in enumerate(sub_categories.keys()):
                sub_accordion.set_title(i, sub_category)
            sub_accordion.selected_index = None  # Close all accordion sections by default
            accordion_items.append(sub_accordion)
        else:
            columns = sub_categories
            checkboxes = [widgets.Checkbox(value=False, description=col) for col in columns]
            checkboxes_grid = widgets.GridBox(checkboxes, layout=widgets.Layout(grid_template_columns="repeat(3, 300px)"))
            accordion_items.append(VBox([create_select_all_checkbox(main_category, checkboxes), checkboxes_grid]))
            checkboxes_dict[main_category] = checkboxes
    
    accordion = Accordion(children=accordion_items)
    for i, main_category in enumerate(categorized_columns.keys()):
        accordion.set_title(i, main_category)
    accordion.selected_index = None  # Close all accordion sections by default
    
    return checkboxes_dict, accordion


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
    """Format finite p-values for color-bar ticks."""
    if not np.isfinite(x):
        return "N/A"
    if x < 0.001:
        return f"{x:.1e}"
    return f"{x:.4f}"


def format_p_value(x):
    """Format a finite p-value for a heatmap annotation."""
    if not np.isfinite(x):
        return "N/A"
    if x < 0.001:
        return "< 0.001"
    return f"{x:.4g}"


def safe_log10_p_values(matrix):
    """Return -log10(p), retaining a finite plotting range for p-values."""
    values = np.asarray(matrix, dtype=float)
    smallest_positive = np.nextafter(0.0, 1.0)
    values = np.clip(values, smallest_positive, 1.0)
    return -np.log10(values)


def plot_heatmap(ax, matrix, title, cmap='viridis'):
    """Plot p-values as -log10(p), with the original p-values annotated."""
    numeric_matrix = matrix.astype(float)
    missing_mask = ~np.isfinite(numeric_matrix.to_numpy())
    plot_matrix = numeric_matrix.fillna(1.0)
    log_matrix = safe_log10_p_values(plot_matrix)

    finite_values = log_matrix[np.isfinite(log_matrix)]
    vmax = max(1.0, float(finite_values.max())) if finite_values.size else 1.0
    vmin = 0.0

    formatted_annotations = numeric_matrix.map(
        lambda value: format_p_value(value) if np.isfinite(value) else ""
    )

    sns.heatmap(
        log_matrix,
        ax=ax,
        cmap=cmap,
        annot=formatted_annotations,
        fmt="",
        xticklabels=numeric_matrix.columns,
        yticklabels=numeric_matrix.index,
        cbar=False,
        vmin=vmin,
        vmax=vmax,
        square=True,
    )
    ax.set_title(title)
    ax.tick_params(axis='x', labelrotation=90)
    ax.tick_params(axis='y', labelrotation=0)

    # Mark undefined comparisons explicitly instead of coloring them as p=1.
    for row, col in zip(*np.where(missing_mask)):
        ax.add_patch(
            plt.Rectangle(
                (col, row), 1, 1,
                facecolor='lightgrey',
                edgecolor='white',
                linewidth=0.5,
                zorder=2,
            )
        )
        ax.text(col + 0.5, row + 0.5, "N/A", ha='center', va='center', zorder=3)

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = ax.figure.colorbar(sm, ax=ax)

    tick_locs = np.linspace(vmin, vmax, 5)
    tick_labels = [format_scientific_for_ticks(10 ** -tick) for tick in tick_locs]
    cbar.set_ticks(tick_locs)
    cbar.set_ticklabels(tick_labels)


def _finite_numeric_array(values):
    """Convert array-like values to a one-dimensional finite float array."""
    numeric = pd.to_numeric(pd.Series(values), errors='coerce').to_numpy(dtype=float)
    return numeric[np.isfinite(numeric)]


def _cohen_d_from_finite_arrays(group1, group2):
    """Calculate Cohen's d from arrays that have already been cleaned."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return np.nan

    diff = group1.mean() - group2.mean()
    var1 = group1.var(ddof=1)
    var2 = group2.var(ddof=1)
    pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)

    if not np.isfinite(pooled_var) or pooled_var < 0:
        return np.nan
    if np.isclose(pooled_var, 0.0):
        return 0.0 if np.isclose(diff, 0.0) else np.nan

    return diff / np.sqrt(pooled_var)


def cohen_d(group1, group2):
    """Calculate Cohen's d after removing NaN and infinite observations."""
    return _cohen_d_from_finite_arrays(
        _finite_numeric_array(group1),
        _finite_numeric_array(group2),
    )


def perform_randomization_test(df, cond1, cond2, var, n_iterations=1000):
    """Perform a randomization test using Cohen's d as the effect size metric."""
    group1 = _finite_numeric_array(df.loc[df['Condition'] == cond1, var])
    group2 = _finite_numeric_array(df.loc[df['Condition'] == cond2, var])
    observed_effect_size = _cohen_d_from_finite_arrays(group1, group2)

    if not np.isfinite(observed_effect_size):
        return np.nan

    combined = np.concatenate([group1, group2])
    rng = np.random.default_rng()
    count_extreme = 0

    for _ in range(n_iterations):
        permuted = rng.permutation(combined)
        new_group1 = permuted[:len(group1)]
        new_group2 = permuted[len(group1):]
        new_effect_size = _cohen_d_from_finite_arrays(new_group1, new_group2)
        if np.isfinite(new_effect_size) and abs(new_effect_size) >= abs(observed_effect_size):
            count_extreme += 1

    return (count_extreme + 1) / (n_iterations + 1)


def run_batch(params):
    """Run one batch of randomization-test permutations."""
    group1_size, combined, observed_effect_size, n_iter = params
    rng = np.random.default_rng()
    count_extreme = 0

    for _ in range(n_iter):
        permuted = rng.permutation(combined)
        new_group1 = permuted[:group1_size]
        new_group2 = permuted[group1_size:]
        new_effect_size = _cohen_d_from_finite_arrays(new_group1, new_group2)
        if np.isfinite(new_effect_size) and abs(new_effect_size) >= abs(observed_effect_size):
            count_extreme += 1

    return count_extreme


def perform_randomization_test_parallel(df, cond1, cond2, var, n_iterations=1000, n_cores=4):
    """Perform a NaN-safe randomization test, optionally across worker processes."""
    group1 = _finite_numeric_array(df.loc[df['Condition'] == cond1, var])
    group2 = _finite_numeric_array(df.loc[df['Condition'] == cond2, var])
    observed_effect_size = _cohen_d_from_finite_arrays(group1, group2)

    if not np.isfinite(observed_effect_size):
        return np.nan

    n_iterations = int(n_iterations)
    if n_iterations < 1:
        raise ValueError("n_iterations must be at least 1")

    combined = np.concatenate([group1, group2])
    n_cores = max(1, min(int(n_cores), n_iterations))

    if n_cores == 1:
        return perform_randomization_test(df, cond1, cond2, var, n_iterations=n_iterations)

    iterations = [n_iterations // n_cores] * n_cores
    for i in range(n_iterations % n_cores):
        iterations[i] += 1

    batches = [
        (len(group1), combined.copy(), observed_effect_size, batch_iterations)
        for batch_iterations in iterations
        if batch_iterations > 0
    ]

    with get_context("spawn").Pool(len(batches)) as pool:
        results = pool.map(run_batch, batches)

    total_extreme = sum(results)
    return (total_extreme + 1) / (n_iterations + 1)


def perform_t_test(df, cond1, cond2, var):
    """Perform Welch's t-test on repeat-level means after removing non-finite values."""
    group1 = (
        df.loc[df['Condition'] == cond1]
        .groupby('Repeat')[var]
        .mean()
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .to_numpy(dtype=float)
    )
    group2 = (
        df.loc[df['Condition'] == cond2]
        .groupby('Repeat')[var]
        .mean()
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .to_numpy(dtype=float)
    )

    if len(group1) < 2 or len(group2) < 2:
        return np.nan

    _, p_value = stats.ttest_ind(group1, group2, equal_var=False, nan_policy='omit')
    return float(p_value) if np.isfinite(p_value) else np.nan

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


def plot_selected_vars(button, checkboxes_dict, df, Conditions, Results_Folder, condition_selector, stat_method_selector):
    plt.close('all')
    print("Plotting in progress...")

    variables_to_plot = []
    for category, checkboxes in checkboxes_dict.items():
        if isinstance(checkboxes, dict):
            for subcheckboxes in checkboxes.values():
                variables_to_plot.extend(
                    checkbox.description for checkbox in subcheckboxes if checkbox.value
                )
        else:
            variables_to_plot.extend(
                checkbox.description for checkbox in checkboxes if checkbox.value
            )

    method = stat_method_selector.value
    if not variables_to_plot:
        print("No variables selected for plotting")
        return

    selected_conditions = list(condition_selector.value)
    if not selected_conditions:
        print("No conditions selected; using all available conditions.")
        selected_conditions = df[Conditions].dropna().unique().tolist()

    filtered_df = df[df[Conditions].isin(selected_conditions)].copy()
    unique_conditions = [
        condition for condition in selected_conditions
        if condition in set(filtered_df[Conditions].dropna())
    ]

    if not unique_conditions:
        print("No rows are available for the selected conditions.")
        return

    num_comparisons = len(unique_conditions) * (len(unique_conditions) - 1) // 2
    n_iterations = 1000

    for var in variables_to_plot:
        if var not in filtered_df.columns:
            print(f"Skipping '{var}': column not found.")
            continue

        # Float matrices accept computed effect sizes and p-values without dtype warnings.
        effect_size_matrix = pd.DataFrame(
            0.0, index=unique_conditions, columns=unique_conditions, dtype=float
        )
        p_value_matrix = pd.DataFrame(
            1.0, index=unique_conditions, columns=unique_conditions, dtype=float
        )
        bonferroni_matrix = pd.DataFrame(
            1.0, index=unique_conditions, columns=unique_conditions, dtype=float
        )

        valid_counts = {
            condition: len(_finite_numeric_array(filtered_df.loc[filtered_df[Conditions] == condition, var]))
            for condition in unique_conditions
        }
        missing_conditions = [condition for condition, count in valid_counts.items() if count < 2]
        if missing_conditions:
            print(
                f"{var}: fewer than two valid observations for "
                + ", ".join(map(str, missing_conditions))
                + "; affected comparisons are marked N/A."
            )

        for cond1, cond2 in itertools.combinations(unique_conditions, 2):
            group1 = filtered_df.loc[filtered_df[Conditions] == cond1, var]
            group2 = filtered_df.loc[filtered_df[Conditions] == cond2, var]
            effect_size = abs(cohen_d(group1, group2))

            if method == 't-test':
                p_value = perform_t_test(filtered_df, cond1, cond2, var)
            elif method == 'randomization test':
                p_value = perform_randomization_test_parallel(
                    filtered_df, cond1, cond2, var, n_iterations=n_iterations
                )
            else:
                raise ValueError(f"Unsupported statistical method: {method}")

            bonferroni_p = (
                min(p_value * num_comparisons, 1.0)
                if np.isfinite(p_value) and num_comparisons > 0
                else np.nan
            )

            effect_size_matrix.loc[cond1, cond2] = effect_size
            effect_size_matrix.loc[cond2, cond1] = effect_size
            p_value_matrix.loc[cond1, cond2] = p_value
            p_value_matrix.loc[cond2, cond1] = p_value
            bonferroni_matrix.loc[cond1, cond2] = bonferroni_p
            bonferroni_matrix.loc[cond2, cond1] = bonferroni_p

        combined_df = pd.concat([
            effect_size_matrix.rename(columns=lambda x: f"{x} (Effect Size)"),
            p_value_matrix.rename(columns=lambda x: f"{x} ({method} P-Value)"),
            bonferroni_matrix.rename(
                columns=lambda x: f"{x} ({method} Bonferroni-corrected P-Value)"
            ),
        ], axis=1)
        combined_df.to_csv(f"{Results_Folder}/csv/{var}_statistics_combined.csv")

        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 3, height_ratios=[1.5, 1])
        ax_box = fig.add_subplot(gs[0, :])

        data_for_var = filtered_df[[Conditions, var, 'Repeat', 'File_name']].copy()
        data_for_var.to_csv(
            f"{Results_Folder}/csv/{var}_boxplot_data.csv", index=False
        )

        plot_df = filtered_df[[Conditions, var, 'Repeat']].copy()
        plot_df[var] = pd.to_numeric(plot_df[var], errors='coerce')
        plot_df[var] = plot_df[var].replace([np.inf, -np.inf], np.nan)
        plot_df = plot_df.dropna(subset=[var])

        if plot_df.empty:
            plt.close(fig)
            print(f"Skipping '{var}': no finite values are available.")
            continue

        # Seaborn 0.13.2 still passes Matplotlib's deprecated ``vert``
        # argument internally. Suppress only that dependency-level warning.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore',
                message=r'vert: bool will be deprecated.*',
                category=PendingDeprecationWarning,
            )
            sns.boxplot(
                x=Conditions, y=var, data=plot_df, order=unique_conditions,
                ax=ax_box, color='lightgray'
            )
        sns.stripplot(
            x=Conditions,
            y=var,
            data=plot_df,
            order=unique_conditions,
            ax=ax_box,
            hue='Repeat',
            dodge=True,
            jitter=True,
            alpha=0.2,
            palette='tab10',
        )

        finite_values = plot_df[var]
        q1 = finite_values.quantile(0.2)
        q3 = finite_values.quantile(0.8)
        iqr = q3 - q1
        lower_bound = max(finite_values.min(), q1 - 10 * iqr)
        upper_bound = min(finite_values.max(), q3 + 10 * iqr)
        if np.isfinite(lower_bound) and np.isfinite(upper_bound):
            if np.isclose(lower_bound, upper_bound):
                padding = max(abs(lower_bound) * 0.05, 0.5)
                lower_bound -= padding
                upper_bound += padding
            ax_box.set_ylim(lower_bound, upper_bound)

        ax_box.set_title(var)
        ax_box.set_xlabel('Condition')
        ax_box.set_ylabel(var)
        ax_box.tick_params(axis='x', labelrotation=90)
        handles, labels = ax_box.get_legend_handles_labels()
        if handles:
            ax_box.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Repeat')

        ax_d = fig.add_subplot(gs[1, 0])
        effect_annotations = effect_size_matrix.map(
            lambda value: f"{value:.3g}" if np.isfinite(value) else ""
        )
        effect_missing = ~np.isfinite(effect_size_matrix.to_numpy())
        sns.heatmap(
            effect_size_matrix.fillna(0.0),
            annot=effect_annotations,
            fmt="",
            cmap="viridis",
            cbar=True,
            square=True,
            ax=ax_d,
            vmin=0,
            vmax=max(1.0, float(np.nanmax(effect_size_matrix.to_numpy()))),
        )
        for row, col in zip(*np.where(effect_missing)):
            ax_d.add_patch(
                plt.Rectangle(
                    (col, row), 1, 1,
                    facecolor='lightgrey', edgecolor='white', linewidth=0.5, zorder=2
                )
            )
            ax_d.text(col + 0.5, row + 0.5, "N/A", ha='center', va='center', zorder=3)
        ax_d.set_title("Effect Size (Cohen's d)")
        ax_d.tick_params(axis='x', labelrotation=90)
        ax_d.tick_params(axis='y', labelrotation=0)

        ax_p = fig.add_subplot(gs[1, 1])
        plot_heatmap(ax_p, p_value_matrix, f"{method} p-value")

        ax_bonf = fig.add_subplot(gs[1, 2])
        plot_heatmap(ax_bonf, bonferroni_matrix, "Bonferroni-corrected p-value")

        fig.tight_layout()
        output_path = f"{Results_Folder}/pdf/{var}_Boxplots_and_Statistics.pdf"
        fig.savefig(output_path, format='pdf', bbox_inches='tight')
        plt.show()
        plt.close(fig)

    print("Plotting completed.")

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



def handle_nans_in_selected_columns(selected_df, selected_columns, df_name='DataFrame', nan_threshold=30):
    """Drop high-missingness columns, then complete-case rows, and return the result."""
    selected_df = selected_df.copy()
    selected_columns = [column for column in selected_columns if column in selected_df.columns]

    nan_percentages = selected_df[selected_columns].isna().mean().mul(100)
    columns_with_nans = nan_percentages[nan_percentages > 0]

    if not columns_with_nans.empty:
        print(f"Missing values found in {df_name}:")
        for column, percentage in columns_with_nans.items():
            print(f"  {column}: {percentage:.2f}% NaN")

    columns_to_drop = nan_percentages[nan_percentages > nan_threshold].index.tolist()
    if columns_to_drop:
        print(
            f"Dropping columns with more than {nan_threshold}% NaN: "
            + ", ".join(columns_to_drop)
        )
        selected_df = selected_df.drop(columns=columns_to_drop)

    remaining_columns = [
        column for column in selected_columns if column in selected_df.columns
    ]
    rows_before = len(selected_df)
    selected_df = selected_df.dropna(subset=remaining_columns)
    rows_after = len(selected_df)

    print(f"Rows before NaN filtering: {rows_before}")
    print(f"Rows after NaN filtering: {rows_after}")
    return selected_df


def _normalize_series_preserving_missing(series, normalization):
    """Normalize finite values while retaining missing observations as NaN."""
    numeric = pd.to_numeric(series, errors='coerce').astype(float)
    numeric = numeric.replace([np.inf, -np.inf], np.nan)

    if normalization == 'zscore':
        mean = numeric.mean(skipna=True)
        std = numeric.std(skipna=True, ddof=0)
        if not np.isfinite(mean):
            return pd.Series(np.nan, index=numeric.index, dtype=float)
        if not np.isfinite(std) or np.isclose(std, 0.0):
            return pd.Series(
                np.where(numeric.notna(), 0.0, np.nan),
                index=numeric.index,
                dtype=float,
            )
        return (numeric - mean) / std

    if normalization == 'minmax':
        minimum = numeric.min(skipna=True)
        maximum = numeric.max(skipna=True)
        value_range = maximum - minimum
        if not np.isfinite(minimum) or not np.isfinite(maximum):
            return pd.Series(np.nan, index=numeric.index, dtype=float)
        if np.isclose(value_range, 0.0):
            return pd.Series(
                np.where(numeric.notna(), 0.0, np.nan),
                index=numeric.index,
                dtype=float,
            )
        return 2.0 * (numeric - minimum) / value_range - 1.0

    raise ValueError("Unsupported normalization type. Use 'zscore' or 'minmax'.")


def heatmap_comparison(df, Results_Folder, Conditions, normalization='zscore', variables_per_page=40):
    """Plot condition medians after NaN-safe per-variable normalization."""
    variables_to_plot = get_selectable_columns(df)
    if not variables_to_plot:
        print("No numeric variables are available for the heatmap.")
        return pd.DataFrame()

    if Conditions not in df.columns:
        raise KeyError(f"Condition column '{Conditions}' was not found in the DataFrame.")

    working = pd.DataFrame(index=df.index)
    working[Conditions] = df[Conditions]

    numeric_values = pd.DataFrame(index=df.index)
    for variable in variables_to_plot:
        numeric_values[variable] = pd.to_numeric(df[variable], errors='coerce').replace(
            [np.inf, -np.inf], np.nan
        )
        working[variable] = _normalize_series_preserving_missing(
            numeric_values[variable], normalization
        )

    missing_percentages = numeric_values.isna().mean().mul(100)
    missing_percentages = missing_percentages[missing_percentages > 0]
    if not missing_percentages.empty:
        print(
            "Missing values are retained per metric; "
            "available values are still used for each condition median."
        )
        for variable, percentage in missing_percentages.items():
            print(f"  {variable}: {percentage:.2f}% missing")

    working = working.dropna(subset=[Conditions])
    median_values = (
        working.groupby(Conditions, sort=False)[variables_to_plot]
        .median()
        .transpose()
    )
    valid_counts = (
        numeric_values.assign(**{Conditions: df[Conditions]})
        .dropna(subset=[Conditions])
        .groupby(Conditions, sort=False)[variables_to_plot]
        .count()
        .transpose()
    )

    unavailable_variables = median_values.index[median_values.isna().all(axis=1)].tolist()
    if unavailable_variables:
        print(
            "No finite values were available for these variables, so they were omitted: "
            + ", ".join(unavailable_variables)
        )
        median_values = median_values.drop(index=unavailable_variables)
        valid_counts = valid_counts.drop(index=unavailable_variables, errors='ignore')

    if median_values.empty:
        print("No finite values are available to plot after normalization.")
        return median_values

    os.makedirs(Results_Folder, exist_ok=True)
    csv_path = f"{Results_Folder}/Normalized_Median_Values_by_Condition.csv"
    counts_path = f"{Results_Folder}/Normalized_Median_Valid_Counts_by_Condition.csv"
    pdf_path = f"{Results_Folder}/Heatmaps_Normalized_Median_Values_by_Condition.pdf"

    median_values.to_csv(csv_path)
    valid_counts.to_csv(counts_path)

    total_variables = len(median_values)
    num_pages = int(np.ceil(total_variables / variables_per_page))
    finite_medians = median_values.to_numpy(dtype=float)
    finite_medians = finite_medians[np.isfinite(finite_medians)]
    max_abs = float(np.max(np.abs(finite_medians))) if finite_medians.size else 1.0
    if np.isclose(max_abs, 0.0):
        max_abs = 1.0

    with PdfPages(pdf_path) as pdf:
        for page in range(num_pages):
            start = page * variables_per_page
            end = min(start + variables_per_page, total_variables)
            page_data = median_values.iloc[start:end]
            missing_mask = page_data.isna().to_numpy()
            annotations = page_data.map(
                lambda value: f"{value:.2f}" if np.isfinite(value) else ""
            )

            fig_width = max(12, 1.4 * len(page_data.columns) + 5)
            fig_height = max(8, 0.38 * len(page_data.index) + 3)
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            sns.heatmap(
                page_data.fillna(0.0),
                cmap='coolwarm',
                annot=annotations,
                fmt="",
                linewidths=0.1,
                center=0,
                vmin=-max_abs,
                vmax=max_abs,
                ax=ax,
                cbar_kws={'label': f'{normalization} normalized median'},
            )

            for row, col in zip(*np.where(missing_mask)):
                ax.add_patch(
                    plt.Rectangle(
                        (col, row), 1, 1,
                        facecolor='lightgrey',
                        edgecolor='white',
                        linewidth=0.5,
                        zorder=2,
                    )
                )
                ax.text(col + 0.5, row + 0.5, "N/A", ha='center', va='center', zorder=3)

            ax.set_title(
                f"{normalization.capitalize()} Normalized Median Values of Variables "
                f"by Condition (Page {page + 1})"
            )
            ax.tick_params(axis='x', labelrotation=90)
            ax.tick_params(axis='y', labelrotation=0)
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.show()
            plt.close(fig)

    print(f"Heatmaps saved to {pdf_path}")
    print(f"Normalized medians saved to {csv_path}")
    print(f"Valid observation counts saved to {counts_path}")
    return median_values

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

