import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import os
import glob

TYPES = ["RGB", "RGBT", "RGBSOE"]

def load_all_csv_files_for_type(type_name, base_dir="."):
    """Find and load all CSV files matching Eval_{TYPE}_*.csv pattern"""
    pattern = os.path.join(base_dir, f"Eval_{type_name}_*.csv")
    csv_files = glob.glob(pattern)
    
    if not csv_files:
        print(f"Warning: No CSV files found for type {type_name} with pattern {pattern}")
        return None
    
    print(f"Found {len(csv_files)} CSV files for type {type_name}")
    
    # Read and combine all CSV files
    all_dataframes = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            all_dataframes.append(df)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
    
    if not all_dataframes:
        print(f"Warning: No valid data loaded for type {type_name}")
        return None
    
    # Combine all dataframes
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    print(f"Combined {len(combined_df)} rows for type {type_name}")
    
    return combined_df

def create_heatmap_plot_for_type(type_name, df):
    # Parameters
    total_width = 430
    total_height = 839
    horizontal_margin = total_width * 0.1
    vertical_margin = total_height * 0.1
    horizontal_interval = total_width * 0.8 / 4.0
    vertical_interval = total_height * 0.8 / 8.0

    border_thickness = 1
    
    # Color settings for heatmap
    color_low = (0.0, 0.0, 1.0)  # Blue for low loss
    color_high = (1.0, 0.0, 0.0)  # Red for high loss
    
    # Calculate number of areas
    divide_x_count = 1
    divide_y_count = 1
    
    print(f"horizontal_interval: {horizontal_interval}, vertical_interval: {vertical_interval}")
    x_ranges = []
    start = horizontal_margin - horizontal_interval / (2 + divide_x_count - 1)
    interval = horizontal_interval / divide_x_count
    while start + interval <= total_width:
        x_ranges.append([start, start + interval])
        start += interval

    y_ranges = []
    start = vertical_margin - vertical_interval / (2 + divide_y_count - 1)
    interval = vertical_interval / divide_y_count
    while start + interval <= total_height:
        y_ranges.append([start, start + interval])
        start += interval


    def create_heatmap_for_data(data_df, title, filename):
        # Initialize heatmap matrix: rows = y_ranges (vertical), cols = x_ranges (horizontal)
        # We want 9 vertical steps (rows) and 5 horizontal steps (cols)
        heatmap_matrix = np.full((len(y_ranges), len(x_ranges)), np.nan)
        
        # Process each area
        for i in range(len(x_ranges)):
            for j in range(len(y_ranges)):
                # Calculate area boundaries
                x_start = x_ranges[i][0]
                x_end = x_ranges[i][1]
                y_start = y_ranges[j][0]
                y_end = y_ranges[j][1]
                
                # Find data points within this area
                mask = (
                    (data_df['converted_gt_x'] >= x_start) & 
                    (data_df['converted_gt_x'] < x_end) &
                    (data_df['converted_gt_y'] >= y_start) & 
                    (data_df['converted_gt_y'] < y_end)
                )
                
                area_data = data_df[mask]
                
                if len(area_data) > 0:
                    # Calculate mean loss for this area
                    # Store as [j, i] because j is row (y) and i is col (x)
                    mean_loss = area_data['loss'].mean()
                    heatmap_matrix[j, i] = mean_loss
        
        # Calculate aspect ratio for square blocks
        # For square cells: (height per row) = (width per col)
        # So: (total_height / num_rows) = (total_width / num_cols)
        # Therefore: aspect = (num_rows / num_cols) to make cells square
        if len(x_ranges) > 0 and len(y_ranges) > 0:
            aspect_ratio = len(y_ranges) / len(x_ranges)  # 9/5 for square cells
        else:
            aspect_ratio = 'auto'
        
        # Create the plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 20))
        
        # Create custom colormap
        colors = [color_low, color_high]
        n_bins = 100
        cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
        
        # Plot heatmap (vertically flipped)
        # aspect='equal' or calculated aspect to make blocks square
        print(heatmap_matrix)
        im = ax.imshow(heatmap_matrix, cmap=cmap, aspect=aspect_ratio, origin='upper', vmin=0.96, vmax=3.31)
        
        # Add text labels for mean loss values in each area
        # for i in range(num_areas_y):
        #     for j in range(num_areas_x):
        #         if not np.isnan(heatmap_matrix[i, j]):
        #             # Center position for text
        #             x_center = j
        #             y_center = i
        #             # Format the loss value to 2 decimal places
        #             loss_text = f"{heatmap_matrix[i, j]:.2f}"
        #             ax.text(x_center, y_center, loss_text, 
        #                    ha='center', va='center', 
        #                    fontsize=30, fontweight='bold', 
        #                    color='white', fontfamily='Arial')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.2, pad=0.1, aspect=5)
        cbar.ax.tick_params(labelsize=36)
        for label in cbar.ax.get_yticklabels():
            label.set_fontweight('bold')
        
        # Remove axes, labels, and grid
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved {filename}")
    
    # Create overall heatmap
    create_heatmap_for_data(
        df, 
        f"{type_name} Error Heatmap - All Participants", 
        f"{type_name}_error_heatmap_all.png"
    )
    
    print(f"Total areas: {len(x_ranges)} x {len(y_ranges)} = {len(x_ranges) * len(y_ranges)}")

def create_heatmap_plot():
    """Main function to process all types"""
    for type_name in TYPES:
        print(f"\n{'='*60}")
        print(f"Processing type: {type_name}")
        print(f"{'='*60}")
        
        # Load all CSV files for this type
        df = load_all_csv_files_for_type(type_name, base_dir="tmp")
        
        if df is None or len(df) == 0:
            print(f"Skipping type {type_name} - no data available")
            continue
        
        # Transform coordinates
        df['converted_gt_x'] = df['gt_x'] * (430.0 / 7.1)
        df['converted_gt_y'] = df['gt_y'] * (871.6 / 14.3915)
        
        # Create heatmaps for this type
        create_heatmap_plot_for_type(type_name, df)

if __name__ == "__main__":
    create_heatmap_plot()