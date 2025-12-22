import re
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Remove SimHei font config to fix "findfont" error
# No Chinese font required since all text is English

def parse_latency_data(file_path):
    """Parse latency data from TXT file"""
    # Data structure: data[input_len][batch][output_len] = latency
    data = defaultdict(lambda: defaultdict(dict))
    # Regex pattern to match key information
    pattern = r"Batch-(\d+) Input len-(\d+) Output len-(\d+)\nAvg latency: ([\d\.]+) seconds"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        # Match all data entries
        matches = re.findall(pattern, content)
        for batch, input_len, output_len, latency in matches:
            # Convert data types
            batch = int(batch)
            input_len = int(input_len)
            output_len = int(output_len)
            latency = float(latency)
            # Store in data structure
            data[input_len][batch][output_len] = latency
    return data

def plot_latency(data):
    """Plot latency comparison line chart"""
    # Define colors and line styles (for Batch 1-8)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
    
    # Create subplots for each Input length
    input_lens = sorted(data.keys())
    n_cols = 2  # Number of subplot columns
    n_rows = (len(input_lens) + 1) // 2  # Number of subplot rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten()  # Flatten axes array for easy iteration
    
    for idx, input_len in enumerate(input_lens):
        ax = axes[idx]
        batch_data = data[input_len]
        batches = sorted(batch_data.keys())
        
        # Plot line for each Batch
        for batch_idx, batch in enumerate(batches):
            output_lens = sorted(batch_data[batch].keys())
            latencies = [batch_data[batch][ol] for ol in output_lens]
            
            ax.plot(
                output_lens, latencies,
                label=f'Batch-{batch}',
                color=colors[batch_idx % len(colors)],
                linestyle=linestyles[batch_idx % len(linestyles)],
                marker='o', markersize=4, linewidth=1.5
            )
        
        # Set subplot properties
        ax.set_title(f'Input Length = {input_len}')
        ax.set_xlabel('Output Length')
        ax.set_ylabel('Average Latency (seconds)')
        ax.set_xscale('log', base=2)  # Log scale for 2^n output lengths
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # Hide unused subplots (if any)
    for idx in range(len(input_lens), len(axes)):
        axes[idx].set_visible(False)
    
    # Global plot settings
    fig.suptitle('Average Latency Comparison by Batch/Input/Output Length', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('latency_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

# Main execution
if __name__ == '__main__':
    # Replace with your TXT file path
    file_path = 'latency_data.txt'
    try:
        latency_data = parse_latency_data(file_path)
        if not latency_data:
            print("Error: No data parsed! Please check file format.")
        else:
            plot_latency(latency_data)
            print("Chart generated successfully!")
    except FileNotFoundError:
        print(f"Error: File {file_path} not found! Please check the path.")
    except Exception as e:
        print(f"Error: {e}")