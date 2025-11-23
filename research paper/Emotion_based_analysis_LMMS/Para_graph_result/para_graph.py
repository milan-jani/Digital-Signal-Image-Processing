import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set IEEE style parameters for matplotlib
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8
})

def load_data(file_path):
    """Load the Excel file and return the dataframe"""
    try:
        df = pd.read_excel(file_path)
        print(f"Data loaded successfully! Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print("\nFirst few rows:")
        print(df.head())
        return df
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

def create_ieee_plot(x_data, y_data, title, xlabel, ylabel, save_name, plot_type='bar'):
    """Create IEEE-style professional plots"""
    fig, ax = plt.subplots(figsize=(6, 4))
    
    if plot_type == 'bar':
        bars = ax.bar(x_data, y_data, color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    elif plot_type == 'line':
        ax.plot(x_data, y_data, marker='o', linewidth=2, markersize=6, 
                color='steelblue', markerfacecolor='white', markeredgecolor='steelblue')
        # Add value labels on points
        for i, (x, y) in enumerate(zip(x_data, y_data)):
            ax.annotate(f'{y:.2f}', (x, y), textcoords="offset points", 
                       xytext=(0,10), ha='center', fontsize=8)
    
    elif plot_type == 'scatter':
        ax.scatter(x_data, y_data, s=100, alpha=0.7, color='steelblue', 
                  edgecolors='black', linewidth=0.5)
    
    # Formatting
    ax.set_title(title, fontweight='bold', pad=20)
    ax.set_xlabel(xlabel, fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    
    # Rotate x-axis labels if they're text and long
    if isinstance(x_data[0], str) and max(len(str(x)) for x in x_data) > 8:
        plt.xticks(rotation=45, ha='right')
    
    # Add subtle background
    ax.set_facecolor('#f8f9fa')
    
    # Tight layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'{save_name}.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.savefig(f'{save_name}.pdf', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()
    
    print(f"Graph saved as {save_name}.png and {save_name}.pdf")

def create_correlation_heatmap(df, save_name):
    """Create a correlation heatmap for numerical columns"""
    # Select only numerical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) > 1:
        fig, ax = plt.subplots(figsize=(7, 5.5))
        
        # Calculate correlation matrix
        corr_matrix = df[numeric_cols].corr()
        
        # Create heatmap
        sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0,
                   square=True, fmt='.2f', cbar_kws={'shrink': 0.8},
                   ax=ax, linewidths=0.5)
        
        ax.set_title('Correlation Matrix of Numerical Variables', 
                    fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(f'{save_name}_correlation.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{save_name}_correlation.pdf', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Correlation heatmap saved as {save_name}_correlation.png and .pdf")

def main():
    # File path
    file_path = r"D:\CANVAS\SEM 5\DSIP\research paper\Emotion_based_analysis_LMMS\Parameter_dsip_result.xlsx"
    
    # Set output directory to same location as script
    output_dir = Path(__file__).parent
    
    # Load data
    df = load_data(file_path)
    
    if df is None:
        return
    
    print("\n" + "="*50)
    print("CREATING IEEE-FORMAT GRAPHS")
    print("="*50)
    
    # Get column names
    columns = df.columns.tolist()
    
    # Assume first column is emotions or categories
    emotion_col = columns[0]
    emotion_values = df[emotion_col].tolist()
    
    # Create individual graphs for each parameter vs emotions
    for i, col in enumerate(columns[1:], 1):
        values = df[col].tolist()
        
        # Determine appropriate plot type based on data
        if df[col].dtype in ['int64', 'float64']:
            plot_type = 'bar'
        else:
            plot_type = 'bar'
        
        title = f'{col} by {emotion_col}'
        xlabel = emotion_col
        ylabel = col
        # Create clean filename and save in script directory
        clean_col_name = col.replace(" ", "_").replace("/", "_").replace("\\", "_").replace(":", "").replace("?", "")
        save_name = output_dir / f'emotion_{emotion_col.replace(" ", "_")}_{clean_col_name}'
        
        print(f"\nCreating graph {i}: {title}")
        create_ieee_plot(emotion_values, values, title, xlabel, ylabel, str(save_name), plot_type)
    
    # Create correlation heatmap if there are multiple numerical columns
    create_correlation_heatmap(df, str(output_dir / 'emotion_analysis'))
    
    print("\n" + "="*50)
    print("ALL GRAPHS CREATED SUCCESSFULLY!")
    print("Files saved in PNG and PDF formats for IEEE publication quality")
    print("="*50)

if __name__ == "__main__":
    main()