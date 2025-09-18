#!/usr/bin/env python3
"""
Multi-Image Medical VQA: Critical Failure Mode Analysis
Analyzes all 6 models to identify specific failure patterns and prioritize solutions.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class MedicalVQAFailureAnalyzer:
    def __init__(self, results_dir="results"):
        self.results_dir = Path(results_dir)
        self.models = {
            "BiomedCLIP": "BiomedCLIP_complete_results_20250917_205335.json",
            "LLaVA-Med": "LLaVA-Med_complete_results_20250917_214123.json", 
            "Biomedical-LLaMA": "Biomedical-LLaMA_complete_results_20250917_211034.json",
            "PMC-VQA": "PMC-VQA_complete_results_20250917_221003.json",
            "MedGemma": "MedGemma_complete_results_20250918_065126.json",
            "Qwen2.5-VL": "Qwen2.5-VL_complete_results_20250918_012630.json"
        }
        
        # Define the 5 critical failure modes
        self.failure_modes = {
            "cross_image_attention_failure": {
                "description": "Model ignores some images, focuses only on 1-2 images",
                "keywords": ["first image", "second image", "third image", "multiple images", "across images", "both images", "all images"],
                "detection_patterns": [
                    r"based on (?:the )?(?:first|second|third|multiple|all|both) images?",
                    r"in (?:the )?(?:first|second|third|multiple|all|both) images?",
                    r"from (?:the )?(?:first|second|third|multiple|all|both) images?",
                    r"compare.*images?",
                    r"across.*images?"
                ]
            },
            "temporal_reasoning_failure": {
                "description": "Cannot understand progression/sequence across images",
                "keywords": ["progression", "sequence", "before", "after", "temporal", "time", "chronological", "follow-up", "evolution"],
                "detection_patterns": [
                    r"progression",
                    r"sequence",
                    r"before.*after",
                    r"temporal",
                    r"follow.?up",
                    r"evolution",
                    r"chronological",
                    r"time.*sequence"
                ]
            },
            "spatial_relationship_failure": {
                "description": "Misses anatomical connections across different views/orientations",
                "keywords": ["anatomical", "spatial", "view", "orientation", "position", "location", "relationship", "connection"],
                "detection_patterns": [
                    r"anatomical",
                    r"spatial",
                    r"view",
                    r"orientation", 
                    r"position",
                    r"relationship",
                    r"connection"
                ]
            },
            "evidence_aggregation_failure": {
                "description": "Cannot combine findings from multiple images into single diagnosis",
                "keywords": ["diagnosis", "findings", "combine", "aggregate", "synthesize", "conclusion", "assessment"],
                "detection_patterns": [
                    r"diagnosis",
                    r"findings",
                    r"combine",
                    r"aggregate",
                    r"synthesize",
                    r"conclusion",
                    r"assessment",
                    r"based on.*findings"
                ]
            },
            "error_propagation": {
                "description": "Early mistake in one image cascades to wrong final conclusion",
                "keywords": ["mistake", "error", "cascade", "propagation", "misinterpretation"],
                "detection_patterns": [
                    r"mistake",
                    r"error",
                    r"misinterpret",
                    r"incorrect.*interpretation"
                ]
            }
        }
        
        self.results_data = {}
        self.failure_analysis = {}
        
    def load_all_results(self):
        """Load results from all 6 models"""
        print("Loading results from all 6 models...")
        
        for model_name, filename in self.models.items():
            filepath = self.results_dir / filename
            if filepath.exists():
                with open(filepath, 'r') as f:
                    self.results_data[model_name] = json.load(f)
                print(f"✓ Loaded {model_name}: {len(self.results_data[model_name])} samples")
            else:
                print(f"✗ Missing results for {model_name}")
                
        print(f"Total models loaded: {len(self.results_data)}")
        return self.results_data
    
    def categorize_failures(self, model_name, sample):
        """Categorize a single failure into specific failure modes"""
        question = sample.get('question', '').lower()
        predicted = sample.get('predicted', '')
        correct = sample.get('correct', '')
        
        if predicted == correct:
            return []  # Not a failure
            
        detected_failures = []
        
        for failure_mode, config in self.failure_modes.items():
            # Check keywords
            keyword_matches = sum(1 for keyword in config['keywords'] if keyword in question)
            
            # Check regex patterns
            pattern_matches = 0
            for pattern in config['detection_patterns']:
                if re.search(pattern, question, re.IGNORECASE):
                    pattern_matches += 1
            
            # If we found matches, this might be this failure mode
            if keyword_matches > 0 or pattern_matches > 0:
                confidence = (keyword_matches + pattern_matches) / (len(config['keywords']) + len(config['detection_patterns']))
                detected_failures.append({
                    'failure_mode': failure_mode,
                    'confidence': confidence,
                    'keyword_matches': keyword_matches,
                    'pattern_matches': pattern_matches
                })
        
        # If no specific failure mode detected, classify as general failure
        if not detected_failures:
            detected_failures.append({
                'failure_mode': 'general_failure',
                'confidence': 0.1,
                'keyword_matches': 0,
                'pattern_matches': 0
            })
            
        return detected_failures
    
    def analyze_model_failures(self, model_name):
        """Analyze failures for a specific model"""
        if model_name not in self.results_data:
            return None
            
        data = self.results_data[model_name]
        total_samples = len(data)
        failures = []
        
        print(f"Analyzing failures for {model_name}...")
        
        for i, sample in enumerate(data):
            if i % 500 == 0:
                print(f"  Processed {i}/{total_samples} samples...")
                
            predicted = sample.get('predicted', '')
            correct = sample.get('correct', '')
            
            if predicted != correct:
                failure_categories = self.categorize_failures(model_name, sample)
                failures.append({
                    'sample_id': i,
                    'question': sample.get('question', ''),
                    'predicted': predicted,
                    'correct': correct,
                    'failure_categories': failure_categories
                })
        
        # Count failure modes
        failure_counts = defaultdict(int)
        for failure in failures:
            for category in failure['failure_categories']:
                failure_counts[category['failure_mode']] += 1
        
        total_failures = len(failures)
        failure_percentages = {mode: (count/total_failures)*100 for mode, count in failure_counts.items()}
        
        return {
            'model_name': model_name,
            'total_samples': total_samples,
            'total_failures': total_failures,
            'accuracy': (total_samples - total_failures) / total_samples,
            'failure_counts': dict(failure_counts),
            'failure_percentages': failure_percentages,
            'detailed_failures': failures[:100]  # Keep first 100 for detailed analysis
        }
    
    def analyze_all_models(self):
        """Analyze failures across all models"""
        print("Starting comprehensive failure analysis across all 6 models...")
        
        for model_name in self.models.keys():
            if model_name in self.results_data:
                self.failure_analysis[model_name] = self.analyze_model_failures(model_name)
                print(f"✓ Completed analysis for {model_name}")
            else:
                print(f"✗ Skipping {model_name} - no data available")
        
        return self.failure_analysis
    
    def generate_summary_statistics(self):
        """Generate overall summary statistics"""
        if not self.failure_analysis:
            return None
            
        # Overall accuracy across all models
        accuracies = [analysis['accuracy'] for analysis in self.failure_analysis.values()]
        overall_accuracy = np.mean(accuracies)
        
        # Most common failure modes across all models
        all_failure_counts = defaultdict(int)
        for analysis in self.failure_analysis.values():
            for mode, count in analysis['failure_counts'].items():
                all_failure_counts[mode] += count
        
        total_failures = sum(all_failure_counts.values())
        failure_rankings = sorted(all_failure_counts.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'overall_accuracy': overall_accuracy,
            'model_accuracies': {name: analysis['accuracy'] for name, analysis in self.failure_analysis.items()},
            'total_failures_across_models': total_failures,
            'failure_mode_rankings': failure_rankings,
            'failure_mode_percentages': {mode: (count/total_failures)*100 for mode, count in all_failure_counts.items()}
        }
    
    def save_analysis_results(self):
        """Save detailed analysis results"""
        output_dir = Path("analysis_output")
        output_dir.mkdir(exist_ok=True)
        
        # Save detailed failure analysis
        with open(output_dir / "detailed_failure_analysis.json", 'w') as f:
            json.dump(self.failure_analysis, f, indent=2)
        
        # Save summary statistics
        summary = self.generate_summary_statistics()
        with open(output_dir / "summary_statistics.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Analysis results saved to {output_dir}/")
        return output_dir

def main():
    """Main execution function"""
    print("=" * 80)
    print("MULTI-IMAGE MEDICAL VQA: CRITICAL FAILURE MODE ANALYSIS")
    print("=" * 80)
    
    # Initialize analyzer
    analyzer = MedicalVQAFailureAnalyzer()
    
    # Load all results
    analyzer.load_all_results()
    
    # Analyze failures across all models
    analyzer.analyze_all_models()
    
    # Generate and save results
    output_dir = analyzer.save_analysis_results()
    
    # Print summary
    summary = analyzer.generate_summary_statistics()
    if summary:
        print("\n" + "=" * 50)
        print("ANALYSIS SUMMARY")
        print("=" * 50)
        print(f"Overall Accuracy: {summary['overall_accuracy']:.3f}")
        print(f"Total Failures: {summary['total_failures_across_models']}")
        print("\nFailure Mode Rankings:")
        for i, (mode, count) in enumerate(summary['failure_mode_rankings'][:5], 1):
            percentage = summary['failure_mode_percentages'][mode]
            print(f"  {i}. {mode}: {count} failures ({percentage:.1f}%)")
    
    print(f"\nDetailed analysis saved to: {output_dir}")
    return analyzer

if __name__ == "__main__":
    analyzer = main()
