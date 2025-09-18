#!/usr/bin/env python3
"""
Create Pie Chart Visualizations for Failure Mode Analysis
TASK 1: Individual Model Failure Distribution Plot (6 pie charts in 2x3 grid)
TASK 2: Overall Research Priority Pie Chart (single large pie chart)
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set up plotting style
plt.style.use('seaborn-v0_8')
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

def load_failure_data():
    """Load failure analysis data from JSON files"""
    analysis_file = Path("../analysis_output/detailed_failure_analysis.json")
    summary_file = Path("../analysis_output/summary_statistics.json")
    
    with open(analysis_file, 'r') as f:
        failure_data = json.load(f)
    
    with open(summary_file, 'r') as f:
        summary_data = json.load(f)
    
    return failure_data, summary_data

def create_failure_mode_mapping():
    """Create mapping for failure mode names and colors"""
    failure_modes = {
        'cross_image_attention_failure': {
            'display_name': 'Cross-Image Attention Failure',
            'color': '#FF6B6B',
            'priority': 1
        },
        'evidence_aggregation_failure': {
            'display_name': 'Evidence Aggregation Failure', 
            'color': '#4ECDC4',
            'priority': 2
        },
        'temporal_reasoning_failure': {
            'display_name': 'Temporal Reasoning Failure',
            'color': '#45B7D1',
            'priority': 3
        },
        'spatial_relationship_failure': {
            'display_name': 'Spatial Relationship Failure',
            'color': '#96CEB4',
            'priority': 4
        },
        'error_propagation': {
            'display_name': 'Error Propagation',
            'color': '#FFEAA7',
            'priority': 5
        },
        'others': {
            'display_name': 'Others',
            'color': '#DDA0DD',
            'priority': 6
        }
    }
    return failure_modes

def create_individual_model_pie_charts(failure_data, failure_modes):
    """TASK 1: Create 6 individual pie charts in 2x3 grid layout"""
    print("Creating Individual Model Failure Distribution Plot...")
    
    # Model order for display
    model_order = ['LLaVA-Med', 'BiomedCLIP', 'MedGemma', 'Biomedical-LLaMA', 'Qwen2.5-VL', 'PMC-VQA']
    
    # Create figure with 2x3 subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Individual Model Failure Mode Distribution', fontsize=16, fontweight='bold', y=0.95)
    
    # Flatten axes for easier indexing
    axes_flat = axes.flatten()
    
    for i, model_name in enumerate(model_order):
        if model_name in failure_data['model_analysis']:
            model_data = failure_data['model_analysis'][model_name]
            failure_breakdown = model_data['failure_breakdown']
            total_failures = model_data['total_failures']
            
            # Prepare data for pie chart
            labels = []
            sizes = []
            colors = []
            counts = []
            
            # Add the 5 main failure modes
            for mode_key, mode_info in failure_modes.items():
                if mode_key != 'others' and mode_key in failure_breakdown:
                    labels.append(mode_info['display_name'])
                    sizes.append(failure_breakdown[mode_key]['percentage'])
                    colors.append(mode_info['color'])
                    counts.append(failure_breakdown[mode_key]['count'])
            
            # Add "Others" category (0% for this data structure)
            labels.append('Others')
            sizes.append(0.0)  # No "others" category in current data
            colors.append(failure_modes['others']['color'])
            counts.append(0)
            
            # Create pie chart
            wedges, texts, autotexts = axes_flat[i].pie(
                sizes, 
                labels=labels,
                colors=colors,
                autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*total_failures)})' if pct > 0 else '',
                startangle=90,
                textprops={'fontsize': 8}
            )
            
            # Set title for each subplot
            axes_flat[i].set_title(f'{model_name}\nTotal Failures: {total_failures:,}', 
                                 fontsize=11, fontweight='bold', pad=20)
            
            # Make text more readable
            for autotext in autotexts:
                autotext.set_fontsize(8)
                autotext.set_fontweight('bold')
    
    # Add overall legend
    legend_elements = []
    for mode_key, mode_info in failure_modes.items():
        if mode_key != 'others':
            legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=mode_info['color'], 
                                               label=mode_info['display_name']))
    
    fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, 0.02), 
              ncol=3, fontsize=10)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.15)
    
    # Save the plot
    plots_dir = Path("../plots")
    plots_dir.mkdir(exist_ok=True)
    plt.savefig(plots_dir / 'individual_model_failure_piecharts.png', 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Saved: individual_model_failure_piecharts.png")

def create_overall_priority_pie_chart(summary_data, failure_modes):
    """TASK 2: Create overall research priority pie chart"""
    print("Creating Overall Research Priority Pie Chart...")
    
    # Get overall failure mode percentages
    failure_percentages = summary_data['failure_mode_percentages']
    total_failures = summary_data['total_failures']
    
    # Prepare data for pie chart
    labels = []
    sizes = []
    colors = []
    priorities = []
    
    # Sort by priority (most common first)
    sorted_modes = sorted(failure_percentages.items(), 
                         key=lambda x: failure_modes[x[0]]['priority'])
    
    for mode_key, percentage in sorted_modes:
        if mode_key in failure_modes:
            labels.append(failure_modes[mode_key]['display_name'])
            sizes.append(percentage)
            colors.append(failure_modes[mode_key]['color'])
            priorities.append(failure_modes[mode_key]['priority'])
    
    # Create the pie chart
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create pie chart with custom formatting
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*total_failures):,})',
        startangle=90,
        textprops={'fontsize': 11, 'fontweight': 'bold'}
    )
    
    # Add priority rankings
    for i, (wedge, priority) in enumerate(zip(wedges, priorities)):
        angle = (wedge.theta2 + wedge.theta1) / 2
        x = 1.3 * np.cos(np.radians(angle))
        y = 1.3 * np.sin(np.radians(angle))
        ax.annotate(f'#{priority}', xy=(x, y), xytext=(x*1.1, y*1.1),
                   fontsize=12, fontweight='bold', ha='center', va='center',
                   bbox=dict(boxstyle='circle,pad=0.3', facecolor='white', 
                            edgecolor='black', alpha=0.8))
    
    # Set title and subtitle
    ax.set_title('Overall Research Priority: Failure Mode Distribution\n' + 
                f'Total Failures Analyzed: {total_failures:,} across 6 models', 
                fontsize=16, fontweight='bold', pad=30)
    
    # Add priority explanation
    priority_text = "Priority Rankings:\n#1 = Most Critical, #5 = Least Critical"
    ax.text(0.02, 0.02, priority_text, transform=ax.transAxes, 
           fontsize=10, bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    # Make text more readable
    for autotext in autotexts:
        autotext.set_fontsize(11)
        autotext.set_fontweight('bold')
    
    # Ensure pie chart is circular
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    # Save the plot
    plots_dir = Path("../plots")
    plots_dir.mkdir(exist_ok=True)
    plt.savefig(plots_dir / 'research_priority_failure_distribution.png', 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Saved: research_priority_failure_distribution.png")

def create_summary_statistics(failure_data, summary_data):
    """Create summary statistics for the pie charts"""
    print("\n" + "="*60)
    print("PIE CHART SUMMARY STATISTICS")
    print("="*60)
    
    total_failures = summary_data['total_failures']
    failure_percentages = summary_data['failure_mode_percentages']
    
    print(f"Total Failures Analyzed: {total_failures:,}")
    print(f"Models Analyzed: 6")
    print(f"Dataset: MedFrameQA")
    print("\nOverall Failure Mode Distribution:")
    
    failure_modes = create_failure_mode_mapping()
    for mode_key, percentage in sorted(failure_percentages.items(), 
                                     key=lambda x: failure_modes[x[0]]['priority']):
        count = int(percentage / 100 * total_failures)
        mode_name = failure_modes[mode_key]['display_name']
        print(f"  {mode_name}: {percentage:.1f}% ({count:,} failures)")
    
    print("\nModel-Specific Total Failures:")
    for model_name, model_data in failure_data['model_analysis'].items():
        print(f"  {model_name}: {model_data['total_failures']:,} failures")

def main():
    """Main execution function"""
    print("="*80)
    print("CREATING PIE CHART VISUALIZATIONS")
    print("="*80)
    
    # Load data
    failure_data, summary_data = load_failure_data()
    failure_modes = create_failure_mode_mapping()
    
    # Create individual model pie charts
    create_individual_model_pie_charts(failure_data, failure_modes)
    
    # Create overall priority pie chart
    create_overall_priority_pie_chart(summary_data, failure_modes)
    
    # Create summary statistics
    create_summary_statistics(failure_data, summary_data)
    
    print("\n" + "="*80)
    print("PIE CHART GENERATION COMPLETE!")
    print("="*80)
    print("Generated files:")
    print("📊 plots/individual_model_failure_piecharts.png - 6 pie charts in 2x3 grid")
    print("📊 plots/research_priority_failure_distribution.png - Overall priority chart")
    print("="*80)

if __name__ == "__main__":
    main()
