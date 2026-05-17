import pandas as pd
import numpy as np
import os

TYPE = "RGBSOE"

class OneEuroFilter:
    """
    One Euro Filter implementation for smoothing noisy signals
    """
    def __init__(self, freq=30, mincutoff=1.0, beta=0.007, dcutoff=1.0):
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        
        # Initialize filters
        self.x = LowPassFilter(self.alpha(self.mincutoff))
        self.dx = LowPassFilter(self.alpha(self.dcutoff))
        self.lasttime = None
    
    def alpha(self, cutoff):
        te = 1.0 / self.freq
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / te)
    
    def __call__(self, x, timestamp=None):
        if timestamp is None:
            timestamp = len(self.x.y) if hasattr(self.x, 'y') else 0
        
        if self.lasttime is None:
            self.lasttime = timestamp
        
        # Calculate time delta
        dt = timestamp - self.lasttime if timestamp != self.lasttime else 1.0 / self.freq
        self.lasttime = timestamp
        
        # Calculate derivative
        if hasattr(self.x, 'y') and len(self.x.y) > 0:
            dx = (x - self.x.y[-1]) / dt
        else:
            dx = 0.0
        
        # Filter derivative
        edx = self.dx(dx, self.alpha(self.dcutoff))
        
        # Calculate cutoff frequency
        cutoff = self.mincutoff + self.beta * abs(edx)
        
        # Filter signal
        return self.x(x, self.alpha(cutoff))

class LowPassFilter:
    """
    Simple low-pass filter
    """
    def __init__(self, alpha):
        self.alpha = alpha
        self.y = []
    
    def __call__(self, x, alpha=None):
        if alpha is None:
            alpha = self.alpha
        
        if len(self.y) == 0:
            self.y.append(x)
        else:
            self.y.append(alpha * x + (1.0 - alpha) * self.y[-1])
        
        return self.y[-1]

def apply_one_euro_filter(data, freq=30, mincutoff=1.0, beta=0.007, dcutoff=1.0):
    """
    Apply one-euro filter to smooth the data
    
    Args:
        data: 1D array of values to filter
        freq: Sampling frequency in Hz
        mincutoff: Minimum cutoff frequency
        beta: Speed coefficient
        dcutoff: Derivative cutoff frequency
    
    Returns:
        Filtered data array
    """
    filter_instance = OneEuroFilter(freq=freq, mincutoff=mincutoff, beta=beta, dcutoff=dcutoff)
    filtered_data = []
    
    for i, value in enumerate(data):
        filtered_value = filter_instance(value, timestamp=i)
        filtered_data.append(filtered_value)
    
    return np.array(filtered_data)

def main():
    # Read the CSV file
    csv_path = f"data/{TYPE}_eval_screen2.csv"
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV file {csv_path} not found.")
        return
    
    print(f"Reading data from {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"Loaded {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
        
    # Apply one-euro filter to x and y coordinates
    print("Applying one-euro filter...")
    
    # Filter x coordinates
    filtered_x = apply_one_euro_filter(df['pred_px_norm_x'].values, freq=30, mincutoff=5.0, beta=1.5, dcutoff=10.0)
    df['filtered_pred_px_norm_x'] = filtered_x
    
    # Filter y coordinates  
    filtered_y = apply_one_euro_filter(df['pred_px_norm_y'].values, freq=30, mincutoff=5.0, beta=1.5, dcutoff=10.0)
    df['filtered_pred_px_norm_y'] = filtered_y
    
    # Save the filtered results
    output_path = f"data/{TYPE}_eval_screen2_filtered.csv"
    df.to_csv(output_path, index=False)
    
    print(f"Filtered results saved to {output_path}")
    
    # Print some statistics
    print(f"\nOriginal vs Filtered Statistics:")
    print(f"X coordinates:")
    print(f"  Original range: [{df['pred_px_norm_x'].min():.4f}, {df['pred_px_norm_x'].max():.4f}]")
    print(f"  Filtered range: [{df['filtered_pred_px_norm_x'].min():.4f}, {df['filtered_pred_px_norm_x'].max():.4f}]")
    print(f"  Original std: {df['pred_px_norm_x'].std():.4f}")
    print(f"  Filtered std: {df['filtered_pred_px_norm_x'].std():.4f}")
    
    print(f"\nY coordinates:")
    print(f"  Original range: [{df['pred_px_norm_y'].min():.4f}, {df['pred_px_norm_y'].max():.4f}]")
    print(f"  Filtered range: [{df['filtered_pred_px_norm_y'].min():.4f}, {df['filtered_pred_px_norm_y'].max():.4f}]")
    print(f"  Original std: {df['pred_px_norm_y'].std():.4f}")
    print(f"  Filtered std: {df['filtered_pred_px_norm_y'].std():.4f}")

if __name__ == "__main__":
    main()
