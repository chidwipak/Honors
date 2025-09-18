#!/usr/bin/env python3
"""
Create CORRECTED Pie Chart Visualizations for Failure Mode Analysis
Fixed version that properly calculates percentages based on actual failure counts
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

def calculate_correct_percentages(failure_breakdown, total_failures):
    """Calculate correct percentages based on actual failure counts"""
    corrected_breakdown = {}
    
    for mode_key, mode_data in failure_breakdown.items():
        count = mode_data['count']
        percentage = (count / total_failures) * 100 if total_failures > 0 else 0
        
        corrected_breakdown[mode_key] = {
            'count': count,
            'percentage': round(percentage, 1),
            'severity': mode_data['severity'],
            'description': mode_data['description']
        }
    
    return corrected_breakdown

def create_individual_model_pie_charts(failure_data, failure_modes):
    """TASK 1: Create 6 individual pie charts in 2x3 grid layout with CORRECT percentages"""
    print("Creating CORRECTED Individual Model Failure Distribution Plot...")
    
    # Model order for display
    model_order = ['LLaVA-Med', 'BiomedCLIP', 'MedGemma', 'Biomedical-LLaMA', 'Qwen2.5-VL', 'PMC-VQA']
    
    # Create figure with 2x3 subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Individual Model Failure Mode Distribution (Corrected)', fontsize=16, fontweight='bold', y=0.95)
    
    # Flatten axes for easier indexing
    axes_flat = axes.flatten()
    
    for i, model_name in enumerate(model_order):
        if model_name in failure_data['model_analysis']:
            model_data = failure_data['model_analysis'][model_name]
            total_failures = model_data['total_failures']
            
            # Calculate CORRECT percentages
            corrected_breakdown = calculate_correct_percentages(model_data['failure_breakdown'], total_failures)
            
            # Prepare data for pie chart
            labels = []
            sizes = []
            colors = []
            counts = []
            
            # Add the 5 main failure modes
            for mode_key, mode_info in failure_modes.items():
                if mode_key != 'others' and mode_key in corrected_breakdown:
                    labels.append(mode_info['display_name'])
                    sizes.append(corrected_breakdown[mode_key]['percentage'])
                    colors.append(mode_info['color'])
                    counts.append(corrected_breakdown[mode_key]['count'])
            
            # Add "Others" category (calculate remaining failures)
            other_count = total_failures - sum(counts)
            other_percentage = (other_count / total_failures) * 100 if total_failures > 0 else 0
            
            labels.append('Others')
            sizes.append(round(other_percentage, 1))
            colors.append(failure_modes['others']['color'])
            counts.append(other_count)
            
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
            
            # Print corrected percentages for verification
            print(f"\n{model_name} - Corrected Percentages:")
            for mode_key, mode_data in corrected_breakdown.items():
                print(f"  {mode_key}: {mode_data['percentage']:.1f}% ({mode_data['count']} failures)")
            print(f"  Others: {other_percentage:.1f}% ({other_count} failures)")
    
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
    plt.savefig(plots_dir / 'individual_model_failure_piecharts_corrected.png', 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Saved: individual_model_failure_piecharts_corrected.png")

def create_overall_priority_pie_chart(failure_data, failure_modes):
    """TASK 2: Create overall research priority pie chart with CORRECT percentages"""
    print("Creating CORRECTED Overall Research Priority Pie Chart...")
    
    # Calculate overall failure counts across all models
    total_failures = 0
    overall_failure_counts = {}
    
    for model_name, model_data in failure_data['model_analysis'].items():
        total_failures += model_data['total_failures']
        for mode_key, mode_data in model_data['failure_breakdown'].items():
            if mode_key not in overall_failure_counts:
                overall_failure_counts[mode_key] = 0
            overall_failure_counts[mode_key] += mode_data['count']
    
    # Calculate overall percentages
    overall_percentages = {}
    for mode_key, count in overall_failure_counts.items():
        percentage = (count / total_failures) * 100 if total_failures > 0 else 0
        overall_percentages[mode_key] = round(percentage, 1)
    
    # Add "Others" category
    other_count = total_failures - sum(overall_failure_counts.values())
    other_percentage = (other_count / total_failures) * 100 if total_failures > 0 else 0
    overall_failure_counts['others'] = other_count
    overall_percentages['others'] = round(other_percentage, 1)
    
    # Prepare data for pie chart
    labels = []
    sizes = []
    colors = []
    priorities = []
    counts = []
    
    # Sort by priority (most common first)
    sorted_modes = sorted(overall_percentages.items(), 
                         key=lambda x: failure_modes[x[0]]['priority'] if x[0] in failure_modes else 6)
    
    for mode_key, percentage in sorted_modes:
        if mode_key in failure_modes:
            labels.append(failure_modes[mode_key]['display_name'])
            sizes.append(percentage)
            colors.append(failure_modes[mode_key]['color'])
            priorities.append(failure_modes[mode_key]['priority'])
            counts.append(overall_failure_counts[mode_key])
        else:  # Others
            labels.append('Others')
            sizes.append(percentage)
            colors.append(failure_modes['others']['color'])
            priorities.append(6)
            counts.append(overall_failure_counts[mode_key])
    
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
    ax.set_title('Overall Research Priority: Failure Mode Distribution (Corrected)\n' + 
                f'Total Failures Analyzed: {total_failures:,} across 6 models', 
                fontsize=16, fontweight='bold', pad=30)
    
    # Add priority explanation
    priority_text = "Priority Rankings:\n#1 = Most Critical, #6 = Least Critical"
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
    plt.savefig(plots_dir / 'research_priority_failure_distribution_corrected.png', 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Saved: research_priority_failure_distribution_corrected.png")
    
    # Print corrected overall percentages
    print(f"\nOverall Corrected Percentages (Total Failures: {total_failures:,}):")
    for mode_key, percentage in sorted_modes:
        count = overall_failure_counts[mode_key]
        print(f"  {mode_key}: {percentage:.1f}% ({count:,} failures)")

def main():
    """Main execution function"""
    print("="*80)
    print("CREATING CORRECTED PIE CHART VISUALIZATIONS")
    print("="*80)
    
    # Load data
    failure_data, summary_data = load_failure_data()
    failure_modes = create_failure_mode_mapping()
    
    # Create individual model pie charts with corrected percentages
    create_individual_model_pie_charts(failure_data, failure_modes)
    
    # Create overall priority pie chart with corrected percentages
    create_overall_priority_pie_chart(failure_data, failure_modes)
    
    print("\n" + "="*80)
    print("CORRECTED PIE CHART GENERATION COMPLETE!")
    print("="*80)
    print("Generated files:")
    print("📊 plots/individual_model_failure_piecharts_corrected.png - 6 pie charts with CORRECT percentages")
    print("📊 plots/research_priority_failure_distribution_corrected.png - Overall chart with CORRECT percentages")
    print("="*80)

if __name__ == "__main__":
    main()
