import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import os

# Define types and cameras
TYPES = ["RGB", "RGBSOE", "RGBT"]
CAMS = ["top_camera", "bottom_camera"]

def create_heatmap_plot(TYPE, CAM):
    # Parameters
    total_width = 430
    total_height = 872
    area_width = 86
    area_height = 83.9
    border_thickness = 1
    
    # Color settings for heatmap
    color_low = (0.0, 0.0, 1.0)  # Blue for low loss
    color_high = (1.0, 0.0, 0.0)  # Red for high loss
    
    # Read CSV data
    csv_path = f"{CAM}/{TYPE}_eval_pall.csv"
    df = pd.read_csv(csv_path)
    
    # Transform coordinates
    df['converted_gt_x'] = df['gt_x'] * (430.0 / 7.1)
    df['converted_gt_y'] = df['gt_y'] * (871.6 / 14.3915)
    
    # Calculate number of areas
    num_areas_x = int(np.ceil(total_width / area_width))
    num_areas_y = int(np.ceil(total_height / area_height))
    
    def create_heatmap_for_data(data_df, title, filename):
        # Initialize heatmap matrix
        heatmap_matrix = np.full((num_areas_y, num_areas_x), np.nan)
        
        # Process each area
        for i in range(num_areas_y):
            for j in range(num_areas_x):
                # Calculate area boundaries
                x_start = j * area_width
                x_end = min((j + 1) * area_width, total_width)
                y_start = i * area_height
                y_end = min((i + 1) * area_height, total_height)
                
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
                    mean_loss = area_data['loss'].mean()
                    heatmap_matrix[i, j] = mean_loss
        
        # Create the plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 20))
        
        # Create custom colormap
        colors = [color_low, color_high]
        n_bins = 100
        cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
        
        # Plot heatmap (vertically flipped)
        print(heatmap_matrix)
        im = ax.imshow(heatmap_matrix, cmap=cmap, aspect='auto', origin='upper', vmin=0, vmax=4)
        
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
            label.set_fontfamily('Arial')
        
        # Remove axes, labels, and grid
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved {filename}")
        
        # Return heatmap values for global statistics
        return heatmap_matrix[~np.isnan(heatmap_matrix)].flatten()
    
    # Create overall heatmap and collect values
    # heatmap_values = create_heatmap_for_data(
    #     df, 
    #     f"{CAM}_{TYPE} Error Heatmap - All Participants", 
    #     f"{CAM}_{TYPE}_error_heatmap_all.png"
    # )
    
    # Create individual participant heatmaps
    unique_subjects = sorted(df['subject'].unique())
    for subject in unique_subjects:
        subject_data = df[df['subject'] == subject]
        create_heatmap_for_data(
            subject_data,
            f"{CAM}_{TYPE} Error Heatmap - Participant {subject}",
            f"{CAM}_{TYPE}_error_heatmap_p{subject}.png"
        )
    
    print(f"Total areas: {num_areas_x} x {num_areas_y} = {num_areas_x * num_areas_y}")
    
    # Return heatmap values for global statistics
    return heatmap_values

if __name__ == "__main__":
    # Store all heatmap values to find global min/max
    all_heatmap_values = []
    
    # Create heatmaps for all combinations
    for TYPE in TYPES:
        for CAM in CAMS:
            try:
                print(f"\n=== Creating heatmap for {TYPE} - {CAM} ===")
                heatmap_values = create_heatmap_plot(TYPE, CAM)
                if heatmap_values is not None:
                    all_heatmap_values.extend(heatmap_values)
            except Exception as e:
                print(f"Error creating heatmap for {TYPE} - {CAM}: {e}")
                continue
    
    # Print global min/max across all heatmaps
    if all_heatmap_values:
        print(f"\n=== Global Heatmap Statistics ===")
        print(f"Global minimum: {min(all_heatmap_values):.4f}")
        print(f"Global maximum: {max(all_heatmap_values):.4f}")
        print(f"Total data points: {len(all_heatmap_values)}")
    
    print(f"\nCompleted all heatmap processing!")