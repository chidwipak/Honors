#!/usr/bin/env python3
"""
Full MedFrameQA evaluation with checkpointing and error handling
Tests all 2,851 questions on each model with 100-question checkpoints
"""

import os
import sys
import json
import logging
import time
from datetime import datetime
import traceback

# Add src to path
sys.path.append('src')

from real_evaluation_pipeline_updated import RealMedFrameQAEvaluator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('full_evaluation_with_checkpoints.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CheckpointedEvaluator:
    """Evaluator with checkpointing and error handling"""
    
    def __init__(self, data_dir, models_dir, results_dir):
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.results_dir = results_dir
        self.checkpoint_interval = 100  # Check every 100 questions
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
        
        # Model configurations
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
        
    def load_dataset(self):
        """Load the complete MedFrameQA dataset"""
        logger.info("Loading complete MedFrameQA dataset...")
        evaluator = RealMedFrameQAEvaluator(self.data_dir, self.models_dir, self.results_dir)
        self.dataset = evaluator.load_dataset()  # Load all samples
        logger.info(f"Dataset loaded: {len(self.dataset)} samples")
        return self.dataset
    
    def evaluate_model_with_checkpoints(self, model_name):
        """Evaluate a single model with checkpointing"""
        logger.info(f"Starting evaluation of {model_name} on all {len(self.dataset)} samples")
        
        if model_name not in self.model_configs:
            raise ValueError(f"Unknown model: {model_name}")
        
        config = self.model_configs[model_name]
        
        # Check if model directory exists
        if not os.path.exists(config['path']):
            logger.error(f"Model directory not found: {config['path']}")
            return {
                'accuracy': 0.0,
                'correct_predictions': 0,
                'total_samples': 0,
                'evaluation_time': 0,
                'samples_per_second': 0,
                'error': f"Model directory not found: {config['path']}"
            }
        
        # Load the model
        try:
            from real_model_evaluators_updated import get_real_model_evaluator
            evaluator = get_real_model_evaluator(model_name, config['path'])
            evaluator.load_model()
            logger.info(f"Model {model_name} loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return {
                'accuracy': 0.0,
                'correct_predictions': 0,
                'total_samples': 0,
                'evaluation_time': 0,
                'samples_per_second': 0,
                'error': str(e)
            }
        
        # Evaluation results
        predictions = []
        ground_truths = []
        sample_metadata = []
        correct_predictions = 0
        total_samples = 0
        errors = []
        
        start_time = time.time()
        
        # Process samples with checkpointing
        for i, sample in enumerate(self.dataset):
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
                
                # Checkpoint every 100 samples
                if (i + 1) % self.checkpoint_interval == 0:
                    current_accuracy = correct_predictions / total_samples
                    elapsed_time = time.time() - start_time
                    samples_per_second = total_samples / elapsed_time if elapsed_time > 0 else 0
                    
                    logger.info(f"CHECKPOINT {i+1}/{len(self.dataset)}: "
                              f"Accuracy = {current_accuracy:.4f}, "
                              f"Correct = {correct_predictions}/{total_samples}, "
                              f"Time = {elapsed_time:.2f}s, "
                              f"Speed = {samples_per_second:.2f} samples/s")
                    
                    # Save checkpoint
                    checkpoint_data = {
                        'model_name': model_name,
                        'checkpoint': i + 1,
                        'total_samples': len(self.dataset),
                        'processed_samples': total_samples,
                        'correct_predictions': correct_predictions,
                        'accuracy': current_accuracy,
                        'elapsed_time': elapsed_time,
                        'samples_per_second': samples_per_second,
                        'errors': len(errors)
                    }
                    
                    checkpoint_file = os.path.join(
                        self.results_dir, 
                        f"{model_name}_checkpoint_{i+1}.json"
                    )
                    with open(checkpoint_file, 'w') as f:
                        json.dump(checkpoint_data, f, indent=2)
                    
                    # Check for too many errors
                    if len(errors) > 10:  # More than 10 errors in 100 samples
                        logger.error(f"Too many errors detected: {len(errors)} errors in {total_samples} samples")
                        logger.error("Stopping evaluation to analyze issues")
                        return {
                            'accuracy': current_accuracy,
                            'correct_predictions': correct_predictions,
                            'total_samples': total_samples,
                            'evaluation_time': elapsed_time,
                            'samples_per_second': samples_per_second,
                            'error': f"Too many errors: {len(errors)} errors in {total_samples} samples",
                            'errors': errors
                        }
                
            except Exception as e:
                error_msg = f"Error processing sample {i}: {str(e)}"
                logger.error(error_msg)
                errors.append({
                    'sample_id': i,
                    'error': str(e),
                    'traceback': traceback.format_exc()
                })
                
                # Add fallback prediction
                predictions.append("A")
                ground_truths.append(sample['answer'])
                sample_metadata.append(sample['metadata'])
                total_samples += 1
        
        # Calculate final metrics
        final_accuracy = correct_predictions / total_samples if total_samples > 0 else 0
        evaluation_time = time.time() - start_time
        
        # Store results
        result = {
            'accuracy': final_accuracy,
            'correct_predictions': correct_predictions,
            'total_samples': total_samples,
            'evaluation_time': evaluation_time,
            'samples_per_second': total_samples / evaluation_time if evaluation_time > 0 else 0,
            'errors': len(errors),
            'error_details': errors[:10]  # First 10 errors for analysis
        }
        
        self.results[model_name] = result
        self.detailed_results[model_name] = {
            'predictions': predictions,
            'ground_truths': ground_truths,
            'sample_metadata': sample_metadata,
            'accuracy': final_accuracy
        }
        
        logger.info(f"Evaluation completed for {model_name}: "
                   f"Accuracy = {final_accuracy:.4f}, "
                   f"Time = {evaluation_time:.2f}s, "
                   f"Errors = {len(errors)}")
        
        return result
    
    def save_model_results(self, model_name):
        """Save results for a specific model"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        if model_name in self.detailed_results:
            detailed_file = os.path.join(
                self.results_dir, 
                f"{model_name}_complete_results_{timestamp}.json"
            )
            with open(detailed_file, 'w') as f:
                json.dump(self.detailed_results[model_name], f, indent=2)
            logger.info(f"Detailed results saved: {detailed_file}")
        
        # Save summary
        if model_name in self.results:
            summary_file = os.path.join(
                self.results_dir, 
                f"{model_name}_summary_{timestamp}.json"
            )
            with open(summary_file, 'w') as f:
                json.dump(self.results[model_name], f, indent=2)
            logger.info(f"Summary saved: {summary_file}")

def main():
    """Main function to run full evaluation with checkpoints"""
    logger.info("="*80)
    logger.info("FULL MEDFRAMEQA EVALUATION WITH CHECKPOINTS")
    logger.info("="*80)
    logger.info("Testing all 2,851 questions on each model with 100-question checkpoints")
    logger.info("="*80)
    
    # Configuration
    data_dir = "/home/mohanganesh/VQAhonors/data/MedFrameQA/data"
    models_dir = "/home/mohanganesh/VQAhonors/models"
    results_dir = "/home/mohanganesh/VQAhonors/results"
    
    # Create evaluator
    evaluator = CheckpointedEvaluator(data_dir, models_dir, results_dir)
    
    # Load dataset
    dataset = evaluator.load_dataset()
    
    # Models to evaluate
    models_to_evaluate = [
        'BiomedCLIP',
        'Biomedical-LLaMA', 
        'LLaVA-Med',
        'MedGemma',
        'PMC-VQA',
        'Qwen2.5-VL'
    ]
    
    # Evaluate each model
    for model_name in models_to_evaluate:
        logger.info(f"\n{'='*60}")
        logger.info(f"EVALUATING MODEL: {model_name}")
        logger.info(f"{'='*60}")
        
        try:
            result = evaluator.evaluate_model_with_checkpoints(model_name)
            
            if 'error' in result and 'Too many errors' in result['error']:
                logger.error(f"Evaluation stopped for {model_name} due to too many errors")
                logger.error("Please analyze the errors and fix issues before continuing")
                break
            else:
                logger.info(f"✅ {model_name} evaluation completed successfully")
                evaluator.save_model_results(model_name)
                
        except Exception as e:
            logger.error(f"❌ Critical error evaluating {model_name}: {e}")
            logger.error("Stopping evaluation to analyze issues")
            break
    
    # Final summary
    logger.info(f"\n{'='*60}")
    logger.info("FINAL EVALUATION SUMMARY")
    logger.info(f"{'='*60}")
    for model_name, result in evaluator.results.items():
        if 'error' in result:
            logger.info(f"{model_name}: ERROR - {result['error']}")
        else:
            logger.info(f"{model_name}: {result['accuracy']:.4f} accuracy "
                      f"({result['correct_predictions']}/{result['total_samples']}) "
                      f"in {result['evaluation_time']:.2f}s")
    logger.info(f"{'='*60}")

if __name__ == "__main__":
    main()
