import os
import json
import logging
import time
import torch
import numpy as np
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict, Counter
import pandas as pd

from real_data_loader import load_real_medframeqa_dataset
from real_model_evaluators_updated import get_real_model_evaluator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealMedFrameQAEvaluator:
    """Comprehensive real evaluation pipeline for MedFrameQA dataset"""
    
    def __init__(self, data_dir, models_dir, results_dir, device=None):
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.results_dir = results_dir
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
        
        # Model configurations with correct paths
        self.model_configs = {
            'BiomedCLIP': {
                'path': os.path.join(models_dir, 'biomedclip'),
                'batch_size': 8
            },
            'Biomedical-LLaMA': {
                'path': os.path.join(models_dir, 'biomedical_llama'),
                'batch_size': 4
            },
            'LLaVA-Med': {
                'path': os.path.join(models_dir, 'llava_med'),
                'batch_size': 4
            },
            'MedGemma': {
                'path': os.path.join(models_dir, 'medgemma'),
                'batch_size': 4
            },
            'PMC-VQA': {
                'path': os.path.join(models_dir, 'pmc_vqa'),
                'batch_size': 4
            },
            'Qwen2.5-VL': {
                'path': os.path.join(models_dir, 'qwen25_vl'),
                'batch_size': 4
            }
        }
        
        # Results storage
        self.results = {}
        self.detailed_results = {}
        
    def load_dataset(self, max_samples=None):
        """Load the MedFrameQA dataset"""
        logger.info(f"Loading MedFrameQA dataset from {self.data_dir}")
        self.dataloader, self.dataset = load_real_medframeqa_dataset(
            self.data_dir, 
            max_samples=max_samples,
            batch_size=1,  # Process one at a time for now
            shuffle=False
        )
        logger.info(f"Dataset loaded: {len(self.dataset)} samples")
        return self.dataset
    
    def evaluate_model(self, model_name, max_samples=None):
        """Evaluate a single model on the dataset"""
        logger.info(f"Starting evaluation of {model_name}")
        
        if model_name not in self.model_configs:
            raise ValueError(f"Unknown model: {model_name}")
        
        config = self.model_configs[model_name]
        
        # Check if model directory exists
        if not os.path.exists(config['path']):
            logger.error(f"Model directory not found: {config['path']}")
            self.results[model_name] = {
                'accuracy': 0.0,
                'correct_predictions': 0,
                'total_samples': 0,
                'evaluation_time': 0,
                'samples_per_second': 0,
                'error': f"Model directory not found: {config['path']}"
            }
            return self.results[model_name]
        
        # Load the model
        try:
            evaluator = get_real_model_evaluator(model_name, config['path'], self.device)
            evaluator.load_model()
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            self.results[model_name] = {
                'accuracy': 0.0,
                'correct_predictions': 0,
                'total_samples': 0,
                'evaluation_time': 0,
                'samples_per_second': 0,
                'error': str(e)
            }
            return self.results[model_name]
        
        # Load dataset
        if not hasattr(self, 'dataset'):
            self.load_dataset(max_samples)
        
        # Evaluation results
        predictions = []
        ground_truths = []
        sample_metadata = []
        correct_predictions = 0
        total_samples = 0
        
        # Process each sample
        logger.info(f"Processing {len(self.dataset)} samples...")
        start_time = time.time()
        
        for i, sample in enumerate(tqdm(self.dataset, desc=f"Evaluating {model_name}")):
            try:
                # Get prediction
                prediction = evaluator.predict(
                    sample['images'],
                    sample['question'],
                    sample['options']
                )
                
                # Store results
                predictions.append(prediction)
                ground_truths.append(sample['answer'])
                sample_metadata.append(sample['metadata'])
                
                # Check if correct
                if prediction == sample['answer']:
                    correct_predictions += 1
                total_samples += 1
                
                # Log progress every 100 samples
                if (i + 1) % 100 == 0:
                    current_accuracy = correct_predictions / total_samples
                    logger.info(f"Processed {i+1}/{len(self.dataset)} samples, "
                              f"Current accuracy: {current_accuracy:.4f}")
                
            except Exception as e:
                logger.error(f"Error processing sample {i}: {e}")
                # Add fallback prediction
                predictions.append("A")
                ground_truths.append(sample['answer'])
                sample_metadata.append(sample['metadata'])
                total_samples += 1
        
        # Calculate final metrics
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0
        evaluation_time = time.time() - start_time
        
        # Store results
        self.results[model_name] = {
            'accuracy': accuracy,
            'correct_predictions': correct_predictions,
            'total_samples': total_samples,
            'evaluation_time': evaluation_time,
            'samples_per_second': total_samples / evaluation_time if evaluation_time > 0 else 0
        }
        
        self.detailed_results[model_name] = {
            'predictions': predictions,
            'ground_truths': ground_truths,
            'sample_metadata': sample_metadata,
            'accuracy': accuracy
        }
        
        logger.info(f"Evaluation completed for {model_name}: "
                   f"Accuracy = {accuracy:.4f}, "
                   f"Time = {evaluation_time:.2f}s")
        
        return self.results[model_name]
    
    def evaluate_all_models(self, max_samples=None):
        """Evaluate all models on the dataset"""
        logger.info("Starting evaluation of all models")
        
        # Load dataset once
        self.load_dataset(max_samples)
        
        # Evaluate each model
        for model_name in self.model_configs.keys():
            try:
                self.evaluate_model(model_name, max_samples)
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {e}")
                # Store error result
                self.results[model_name] = {
                    'accuracy': 0.0,
                    'correct_predictions': 0,
                    'total_samples': 0,
                    'evaluation_time': 0,
                    'samples_per_second': 0,
                    'error': str(e)
                }
        
        return self.results
    
    def calculate_clinical_breakdown(self, model_name):
        """Calculate clinical performance breakdown for a model"""
        if model_name not in self.detailed_results:
            logger.error(f"No results found for {model_name}")
            return {}
        
        results = self.detailed_results[model_name]
        predictions = results['predictions']
        ground_truths = results['ground_truths']
        metadata = results['sample_metadata']
        
        # Group by different clinical categories
        breakdown = {
            'body_system': defaultdict(lambda: {'correct': 0, 'total': 0}),
            'modality': defaultdict(lambda: {'correct': 0, 'total': 0}),
            'image_count': defaultdict(lambda: {'correct': 0, 'total': 0}),
            'organ': defaultdict(lambda: {'correct': 0, 'total': 0})
        }
        
        for pred, gt, meta in zip(predictions, ground_truths, metadata):
            is_correct = pred == gt
            
            # Body system breakdown
            body_system = meta.get('body_system', 'unknown')
            breakdown['body_system'][body_system]['total'] += 1
            if is_correct:
                breakdown['body_system'][body_system]['correct'] += 1
            
            # Modality breakdown
            modality = meta.get('modality', 'unknown')
            breakdown['modality'][modality]['total'] += 1
            if is_correct:
                breakdown['modality'][modality]['correct'] += 1
            
            # Image count breakdown
            image_count = meta.get('image_count', 0)
            breakdown['image_count'][image_count]['total'] += 1
            if is_correct:
                breakdown['image_count'][image_count]['correct'] += 1
            
            # Organ breakdown
            organ = meta.get('organ', 'unknown')
            breakdown['organ'][organ]['total'] += 1
            if is_correct:
                breakdown['organ'][organ]['correct'] += 1
        
        # Calculate accuracies
        for category in breakdown:
            for key, stats in breakdown[category].items():
                if stats['total'] > 0:
                    stats['accuracy'] = stats['correct'] / stats['total']
                else:
                    stats['accuracy'] = 0.0
        
        return breakdown
    
    def analyze_failure_modes(self, model_name):
        """Analyze failure modes for a model"""
        if model_name not in self.detailed_results:
            logger.error(f"No results found for {model_name}")
            return {}
        
        results = self.detailed_results[model_name]
        predictions = results['predictions']
        ground_truths = results['ground_truths']
        metadata = results['sample_metadata']
        
        # Find incorrect predictions
        failures = []
        for i, (pred, gt, meta) in enumerate(zip(predictions, ground_truths, metadata)):
            if pred != gt:
                failures.append({
                    'sample_id': i,
                    'prediction': pred,
                    'ground_truth': gt,
                    'metadata': meta
                })
        
        # Analyze failure patterns
        failure_analysis = {
            'total_failures': len(failures),
            'failure_rate': len(failures) / len(predictions) if predictions else 0,
            'failure_by_body_system': Counter(),
            'failure_by_modality': Counter(),
            'failure_by_image_count': Counter(),
            'common_wrong_answers': Counter(),
            'sample_failures': failures[:50]  # First 50 failures for inspection
        }
        
        for failure in failures:
            meta = failure['metadata']
            failure_analysis['failure_by_body_system'][meta.get('body_system', 'unknown')] += 1
            failure_analysis['failure_by_modality'][meta.get('modality', 'unknown')] += 1
            failure_analysis['failure_by_image_count'][meta.get('image_count', 0)] += 1
            failure_analysis['common_wrong_answers'][failure['prediction']] += 1
        
        return failure_analysis
    
    def save_results(self, filename_prefix="real_evaluation"):
        """Save all results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save summary results
        summary_file = os.path.join(self.results_dir, f"{filename_prefix}_summary_{timestamp}.json")
        with open(summary_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save detailed results for each model
        for model_name, results in self.detailed_results.items():
            detailed_file = os.path.join(self.results_dir, f"{filename_prefix}_{model_name}_{timestamp}.json")
            with open(detailed_file, 'w') as f:
                json.dump(results, f, indent=2)
        
        # Save clinical breakdowns
        clinical_breakdowns = {}
        for model_name in self.detailed_results.keys():
            clinical_breakdowns[model_name] = self.calculate_clinical_breakdown(model_name)
        
        clinical_file = os.path.join(self.results_dir, f"{filename_prefix}_clinical_breakdown_{timestamp}.json")
        with open(clinical_file, 'w') as f:
            json.dump(clinical_breakdowns, f, indent=2)
        
        # Save failure analysis
        failure_analyses = {}
        for model_name in self.detailed_results.keys():
            failure_analyses[model_name] = self.analyze_failure_modes(model_name)
        
        failure_file = os.path.join(self.results_dir, f"{filename_prefix}_failure_analysis_{timestamp}.json")
        with open(failure_file, 'w') as f:
            json.dump(failure_analyses, f, indent=2)
        
        # Create summary report
        self.create_summary_report(timestamp)
        
        logger.info(f"Results saved with timestamp: {timestamp}")
        return timestamp
    
    def create_summary_report(self, timestamp):
        """Create a human-readable summary report"""
        report_file = os.path.join(self.results_dir, f"real_evaluation_report_{timestamp}.md")
        
        with open(report_file, 'w') as f:
            f.write("# Real MedFrameQA Evaluation Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Overall results table
            f.write("## Overall Results\n\n")
            f.write("| Model | Accuracy | Correct | Total | Time (s) | Samples/s |\n")
            f.write("|-------|----------|---------|-------|----------|----------|\n")
            
            for model_name, results in self.results.items():
                if 'error' in results:
                    f.write(f"| {model_name} | ERROR | - | - | - | - |\n")
                else:
                    f.write(f"| {model_name} | {results['accuracy']:.4f} | "
                           f"{results['correct_predictions']} | {results['total_samples']} | "
                           f"{results['evaluation_time']:.2f} | {results['samples_per_second']:.2f} |\n")
            
            # Clinical breakdown
            f.write("\n## Clinical Performance Breakdown\n\n")
            for model_name in self.detailed_results.keys():
                f.write(f"### {model_name}\n\n")
                breakdown = self.calculate_clinical_breakdown(model_name)
                
                # Body system breakdown
                f.write("#### By Body System\n")
                f.write("| System | Accuracy | Correct | Total |\n")
                f.write("|--------|----------|---------|-------|\n")
                for system, stats in breakdown['body_system'].items():
                    f.write(f"| {system} | {stats['accuracy']:.4f} | "
                           f"{stats['correct']} | {stats['total']} |\n")
                
                # Modality breakdown
                f.write("\n#### By Modality\n")
                f.write("| Modality | Accuracy | Correct | Total |\n")
                f.write("|----------|----------|---------|-------|\n")
                for modality, stats in breakdown['modality'].items():
                    f.write(f"| {modality} | {stats['accuracy']:.4f} | "
                           f"{stats['correct']} | {stats['total']} |\n")
                
                # Image count breakdown
                f.write("\n#### By Image Count\n")
                f.write("| Images | Accuracy | Correct | Total |\n")
                f.write("|--------|----------|---------|-------|\n")
                for count, stats in breakdown['image_count'].items():
                    f.write(f"| {count} | {stats['accuracy']:.4f} | "
                           f"{stats['correct']} | {stats['total']} |\n")
                
                f.write("\n")
        
        logger.info(f"Summary report saved: {report_file}")

def main():
    """Main function to run the real evaluation"""
    # Configuration
    data_dir = "/home/mohanganesh/VQAhonors/data/MedFrameQA/data"
    models_dir = "/home/mohanganesh/VQAhonors/models"
    results_dir = "/home/mohanganesh/VQAhonors/results"
    
    # Create evaluator
    evaluator = RealMedFrameQAEvaluator(data_dir, models_dir, results_dir)
    
    # Run evaluation (start with small sample for testing)
    logger.info("Starting real evaluation pipeline")
    evaluator.evaluate_all_models(max_samples=10)  # Start with 10 samples for testing
    
    # Save results
    timestamp = evaluator.save_results()
    logger.info(f"Evaluation completed. Results saved with timestamp: {timestamp}")

if __name__ == "__main__":
    main()
