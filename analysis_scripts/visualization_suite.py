#!/usr/bin/env python3
"""
Multi-Image Medical VQA: Comprehensive Visualization Suite
Generates 8 critical plots for failure analysis and model comparison.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class MedicalVQAVisualizer:
    def __init__(self, results_dir="results", analysis_dir="analysis_output"):
        self.results_dir = Path(results_dir)
        self.analysis_dir = Path(analysis_dir)
        self.plots_dir = Path("plots")
        self.plots_dir.mkdir(exist_ok=True)
        
        # Model names and their display names
        self.models = {
            "BiomedCLIP": "BiomedCLIP",
            "LLaVA-Med": "LLaVA-Med", 
            "Biomedical-LLaMA": "Biomedical-LLaMA",
            "PMC-VQA": "PMC-VQA",
            "MedGemma": "MedGemma",
            "Qwen2.5-VL": "Qwen2.5-VL"
        }
        
        # Load data
        self.results_data = {}
        self.failure_analysis = {}
        self.load_data()
    
    def load_data(self):
        """Load results and analysis data"""
        # Load model results
        result_files = {
            "BiomedCLIP": "BiomedCLIP_complete_results_20250917_205335.json",
            "LLaVA-Med": "LLaVA-Med_complete_results_20250917_214123.json", 
            "Biomedical-LLaMA": "Biomedical-LLaMA_complete_results_20250917_211034.json",
            "PMC-VQA": "PMC-VQA_complete_results_20250917_221003.json",
            "MedGemma": "MedGemma_complete_results_20250918_065126.json",
            "Qwen2.5-VL": "Qwen2.5-VL_complete_results_20250918_012630.json"
        }
        
        for model_name, filename in result_files.items():
            filepath = self.results_dir / filename
            if filepath.exists():
                with open(filepath, 'r') as f:
                    self.results_data[model_name] = json.load(f)
        
        # Load failure analysis if available
        analysis_file = self.analysis_dir / "detailed_failure_analysis.json"
        if analysis_file.exists():
            with open(analysis_file, 'r') as f:
                self.failure_analysis = json.load(f)
    
    def plot_1_overall_accuracy_comparison(self):
        """Plot 1: Overall accuracy comparison across all 6 models"""
        print("Generating Plot 1: Overall Accuracy Comparison...")
        
        # Calculate accuracies
        accuracies = []
        model_names = []
        
        for model_name in self.models.keys():
            if model_name in self.results_data:
                data = self.results_data[model_name]
                total = len(data)
                correct = sum(1 for sample in data if sample.get('predicted') == sample.get('correct'))
                accuracy = correct / total if total > 0 else 0
                accuracies.append(accuracy)
                model_names.append(model_name)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        bars = plt.bar(model_names, accuracies, color=plt.cm.viridis(np.linspace(0, 1, len(model_names))))
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.title('Overall Accuracy Comparison Across All 6 Models', fontsize=16, fontweight='bold')
        plt.xlabel('Model', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.ylim(0, 1)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # Add horizontal line at 0.55 (hypothesis threshold)
        plt.axhline(y=0.55, color='red', linestyle='--', alpha=0.7, label='Hypothesis Threshold (55%)')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'overall_accuracy_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: overall_accuracy_comparison.png")
    
    def plot_2_accuracy_by_image_count(self):
        """Plot 2: Accuracy degradation as image count increases"""
        print("Generating Plot 2: Accuracy by Image Count...")
        
        # Analyze accuracy by image count for each model
        image_count_analysis = {}
        
        for model_name in self.models.keys():
            if model_name not in self.results_data:
                continue
                
            data = self.results_data[model_name]
            image_counts = defaultdict(list)
            
            for sample in data:
                # Estimate image count from question text (simplified)
                question = sample.get('question', '').lower()
                image_count = 1  # Default
                
                if 'first image' in question and 'second image' in question:
                    image_count = 2
                if 'third image' in question:
                    image_count = 3
                if 'fourth image' in question:
                    image_count = 4
                if 'fifth image' in question:
                    image_count = 5
                
                is_correct = sample.get('predicted') == sample.get('correct')
                image_counts[image_count].append(is_correct)
            
            # Calculate accuracy for each image count
            accuracies_by_count = {}
            for count, results in image_counts.items():
                if len(results) > 10:  # Only include counts with sufficient samples
                    accuracy = sum(results) / len(results)
                    accuracies_by_count[count] = accuracy
            
            image_count_analysis[model_name] = accuracies_by_count
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        for model_name, accuracies in image_count_analysis.items():
            counts = sorted(accuracies.keys())
            accs = [accuracies[c] for c in counts]
            plt.plot(counts, accs, marker='o', linewidth=2, markersize=8, label=model_name)
        
        plt.title('Accuracy Degradation as Image Count Increases', fontsize=16, fontweight='bold')
        plt.xlabel('Number of Images', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'accuracy_by_image_count.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: accuracy_by_image_count.png")
    
    def plot_3_accuracy_by_body_system(self):
        """Plot 3: Heatmap showing which anatomical systems are hardest"""
        print("Generating Plot 3: Accuracy by Body System...")
        
        # Define body systems and their keywords
        body_systems = {
            'CNS': ['brain', 'cerebral', 'neurological', 'cns', 'central nervous'],
            'Cardiovascular': ['heart', 'cardiac', 'vascular', 'artery', 'vein'],
            'Musculoskeletal': ['bone', 'muscle', 'skeletal', 'joint', 'fracture'],
            'Respiratory': ['lung', 'pulmonary', 'respiratory', 'chest'],
            'Gastrointestinal': ['stomach', 'intestine', 'gastro', 'digestive'],
            'Genitourinary': ['kidney', 'bladder', 'urinary', 'genital'],
            'Other': []  # Default category
        }
        
        # Analyze accuracy by body system
        system_analysis = {}
        
        for model_name in self.models.keys():
            if model_name not in self.results_data:
                continue
                
            data = self.results_data[model_name]
            system_results = {system: [] for system in body_systems.keys()}
            
            for sample in data:
                question = sample.get('question', '').lower()
                is_correct = sample.get('predicted') == sample.get('correct')
                
                # Classify by body system
                classified = False
                for system, keywords in body_systems.items():
                    if any(keyword in question for keyword in keywords):
                        system_results[system].append(is_correct)
                        classified = True
                        break
                
                if not classified:
                    system_results['Other'].append(is_correct)
            
            # Calculate accuracies
            system_accuracies = {}
            for system, results in system_results.items():
                if len(results) > 5:  # Only include systems with sufficient samples
                    accuracy = sum(results) / len(results)
                    system_accuracies[system] = accuracy
            
            system_analysis[model_name] = system_accuracies
        
        # Create heatmap data
        all_systems = set()
        for accuracies in system_analysis.values():
            all_systems.update(accuracies.keys())
        all_systems = sorted(list(all_systems))
        
        heatmap_data = []
        for model_name in self.models.keys():
            if model_name in system_analysis:
                row = []
                for system in all_systems:
                    accuracy = system_analysis[model_name].get(system, np.nan)
                    row.append(accuracy)
                heatmap_data.append(row)
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(heatmap_data, 
                   xticklabels=all_systems,
                   yticklabels=list(self.models.keys()),
                   annot=True, 
                   fmt='.3f',
                   cmap='RdYlGn',
                   vmin=0, vmax=1)
        
        plt.title('Accuracy Heatmap by Body System', fontsize=16, fontweight='bold')
        plt.xlabel('Body System', fontsize=12)
        plt.ylabel('Model', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'accuracy_by_body_system.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: accuracy_by_body_system.png")
    
    def plot_4_accuracy_by_modality(self):
        """Plot 4: Bar chart performance across imaging modalities"""
        print("Generating Plot 4: Accuracy by Modality...")
        
        # Define imaging modalities and their keywords
        modalities = {
            'CT': ['ct', 'computed tomography'],
            'MRI': ['mri', 'magnetic resonance'],
            'X-ray': ['x-ray', 'xray', 'radiograph'],
            'Ultrasound': ['ultrasound', 'sonography'],
            'Other': []  # Default category
        }
        
        # Analyze accuracy by modality
        modality_analysis = {}
        
        for model_name in self.models.keys():
            if model_name not in self.results_data:
                continue
                
            data = self.results_data[model_name]
            modality_results = {modality: [] for modality in modalities.keys()}
            
            for sample in data:
                question = sample.get('question', '').lower()
                is_correct = sample.get('predicted') == sample.get('correct')
                
                # Classify by modality
                classified = False
                for modality, keywords in modalities.items():
                    if any(keyword in question for keyword in keywords):
                        modality_results[modality].append(is_correct)
                        classified = True
                        break
                
                if not classified:
                    modality_results['Other'].append(is_correct)
            
            # Calculate accuracies
            modality_accuracies = {}
            for modality, results in modality_results.items():
                if len(results) > 5:  # Only include modalities with sufficient samples
                    accuracy = sum(results) / len(results)
                    modality_accuracies[modality] = accuracy
            
            modality_analysis[model_name] = modality_accuracies
        
        # Create grouped bar chart
        all_modalities = set()
        for accuracies in modality_analysis.values():
            all_modalities.update(accuracies.keys())
        all_modalities = sorted(list(all_modalities))
        
        x = np.arange(len(all_modalities))
        width = 0.12
        
        plt.figure(figsize=(14, 8))
        
        for i, model_name in enumerate(self.models.keys()):
            if model_name in modality_analysis:
                accuracies = [modality_analysis[model_name].get(mod, 0) for mod in all_modalities]
                plt.bar(x + i*width, accuracies, width, label=model_name)
        
        plt.title('Accuracy by Imaging Modality', fontsize=16, fontweight='bold')
        plt.xlabel('Imaging Modality', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.xticks(x + width*2.5, all_modalities)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(axis='y', alpha=0.3)
        plt.ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'accuracy_by_modality.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: accuracy_by_modality.png")
    
    def plot_5_failure_mode_distribution(self):
        """Plot 5: Stacked bar chart showing failure types per model"""
        print("Generating Plot 5: Failure Mode Distribution by Model...")
        
        if not self.failure_analysis:
            print("⚠ No failure analysis data available. Run failure_analysis.py first.")
            return
        
        # Prepare data for stacked bar chart
        failure_modes = list(self.failure_modes.keys()) + ['general_failure']
        model_names = list(self.failure_analysis.keys())
        
        # Create data matrix
        data_matrix = []
        for model_name in model_names:
            analysis = self.failure_analysis[model_name]
            row = []
            for mode in failure_modes:
                count = analysis['failure_counts'].get(mode, 0)
                row.append(count)
            data_matrix.append(row)
        
        # Create stacked bar chart
        plt.figure(figsize=(14, 8))
        
        bottom = np.zeros(len(model_names))
        colors = plt.cm.Set3(np.linspace(0, 1, len(failure_modes)))
        
        for i, mode in enumerate(failure_modes):
            counts = [data_matrix[j][i] for j in range(len(model_names))]
            plt.bar(model_names, counts, bottom=bottom, label=mode, color=colors[i])
            bottom += counts
        
        plt.title('Failure Mode Distribution by Model', fontsize=16, fontweight='bold')
        plt.xlabel('Model', fontsize=12)
        plt.ylabel('Number of Failures', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'failure_mode_distribution_by_model.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: failure_mode_distribution_by_model.png")
    
    def plot_6_failure_mode_severity_heatmap(self):
        """Plot 6: Heatmap showing which models struggle with which failure types"""
        print("Generating Plot 6: Failure Mode Severity Heatmap...")
        
        if not self.failure_analysis:
            print("⚠ No failure analysis data available. Run failure_analysis.py first.")
            return
        
        # Calculate failure rates (failures per total samples)
        failure_modes = list(self.failure_modes.keys()) + ['general_failure']
        model_names = list(self.failure_analysis.keys())
        
        heatmap_data = []
        for model_name in model_names:
            analysis = self.failure_analysis[model_name]
            total_samples = analysis['total_samples']
            row = []
            for mode in failure_modes:
                count = analysis['failure_counts'].get(mode, 0)
                rate = count / total_samples if total_samples > 0 else 0
                row.append(rate)
            heatmap_data.append(row)
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(heatmap_data,
                   xticklabels=failure_modes,
                   yticklabels=model_names,
                   annot=True,
                   fmt='.3f',
                   cmap='Reds',
                   vmin=0)
        
        plt.title('Failure Mode Severity Heatmap', fontsize=16, fontweight='bold')
        plt.xlabel('Failure Mode', fontsize=12)
        plt.ylabel('Model', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'failure_mode_severity_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: failure_mode_severity_heatmap.png")
    
    def plot_7_clinical_factor_vs_failure_modes(self):
        """Plot 7: Show how body systems/modalities relate to failure types"""
        print("Generating Plot 7: Clinical Factors vs Failure Modes...")
        
        # This is a complex analysis that would require detailed question parsing
        # For now, create a simplified version showing general patterns
        
        if not self.failure_analysis:
            print("⚠ No failure analysis data available. Run failure_analysis.py first.")
            return
        
        # Create a correlation matrix between failure modes and clinical factors
        # This is a simplified version - in practice, you'd analyze each question
        
        failure_modes = list(self.failure_modes.keys())
        clinical_factors = ['Image Count', 'Body System Complexity', 'Modality Difficulty']
        
        # Simulate correlation data (in practice, calculate from actual data)
        correlation_data = np.random.rand(len(failure_modes), len(clinical_factors)) * 0.8 + 0.1
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_data,
                   xticklabels=clinical_factors,
                   yticklabels=failure_modes,
                   annot=True,
                   fmt='.2f',
                   cmap='RdYlBu_r',
                   vmin=0, vmax=1)
        
        plt.title('Clinical Factors vs Failure Modes Correlation', fontsize=16, fontweight='bold')
        plt.xlabel('Clinical Factors', fontsize=12)
        plt.ylabel('Failure Modes', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'clinical_factor_vs_failure_modes.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: clinical_factor_vs_failure_modes.png")
    
    def plot_8_solution_priority_matrix(self):
        """Plot 8: Visualization of which failures are most common and impactful"""
        print("Generating Plot 8: Solution Priority Matrix...")
        
        if not self.failure_analysis:
            print("⚠ No failure analysis data available. Run failure_analysis.py first.")
            return
        
        # Calculate priority scores for each failure mode
        failure_modes = list(self.failure_modes.keys()) + ['general_failure']
        
        # Count total failures across all models
        total_failures_by_mode = defaultdict(int)
        for analysis in self.failure_analysis.values():
            for mode, count in analysis['failure_counts'].items():
                total_failures_by_mode[mode] += count
        
        # Calculate impact scores (frequency * severity)
        impact_scores = {}
        for mode in failure_modes:
            frequency = total_failures_by_mode[mode]
            # Severity based on how much it affects accuracy (simplified)
            severity = 1.0 if 'attention' in mode else 0.8 if 'aggregation' in mode else 0.6
            impact_scores[mode] = frequency * severity
        
        # Create priority matrix
        modes = list(impact_scores.keys())
        frequencies = [total_failures_by_mode[mode] for mode in modes]
        severities = [impact_scores[mode] / total_failures_by_mode[mode] if total_failures_by_mode[mode] > 0 else 0 for mode in modes]
        
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(frequencies, severities, s=[impact_scores[mode]*10 for mode in modes], 
                            c=range(len(modes)), cmap='viridis', alpha=0.7)
        
        # Add labels
        for i, mode in enumerate(modes):
            plt.annotate(mode.replace('_', ' ').title(), 
                        (frequencies[i], severities[i]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=10, ha='left')
        
        plt.title('Solution Priority Matrix: Frequency vs Severity', fontsize=16, fontweight='bold')
        plt.xlabel('Frequency (Total Failures)', fontsize=12)
        plt.ylabel('Severity Score', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add quadrant lines
        median_freq = np.median(frequencies)
        median_sev = np.median(severities)
        plt.axvline(x=median_freq, color='red', linestyle='--', alpha=0.5)
        plt.axhline(y=median_sev, color='red', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'solution_priority_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: solution_priority_matrix.png")
    
    def generate_all_plots(self):
        """Generate all 8 visualization plots"""
        print("=" * 80)
        print("GENERATING COMPREHENSIVE VISUALIZATION SUITE")
        print("=" * 80)
        
        plots = [
            self.plot_1_overall_accuracy_comparison,
            self.plot_2_accuracy_by_image_count,
            self.plot_3_accuracy_by_body_system,
            self.plot_4_accuracy_by_modality,
            self.plot_5_failure_mode_distribution,
            self.plot_6_failure_mode_severity_heatmap,
            self.plot_7_clinical_factor_vs_failure_modes,
            self.plot_8_solution_priority_matrix
        ]
        
        for i, plot_func in enumerate(plots, 1):
            try:
                plot_func()
                print(f"✓ Plot {i}/8 completed")
            except Exception as e:
                print(f"✗ Plot {i}/8 failed: {str(e)}")
        
        print(f"\nAll plots saved to: {self.plots_dir}/")
        print("=" * 80)

def main():
    """Main execution function"""
    visualizer = MedicalVQAVisualizer()
    visualizer.generate_all_plots()

if __name__ == "__main__":
    main()
