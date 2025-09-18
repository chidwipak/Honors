#!/usr/bin/env python3
"""
Complete real evaluation script for MedFrameQA dataset
This script replaces ALL mock data generation with actual model inference
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime

# Add src to path
sys.path.append('src')

from real_evaluation_pipeline_updated import RealMedFrameQAEvaluator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('complete_real_evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Main function to run complete real MedFrameQA evaluation"""
    parser = argparse.ArgumentParser(description='Complete Real MedFrameQA Evaluation')
    parser.add_argument('--data_dir', type=str, 
                       default='/home/mohanganesh/VQAhonors/data/MedFrameQA/data',
                       help='Path to MedFrameQA data directory')
    parser.add_argument('--models_dir', type=str,
                       default='/home/mohanganesh/VQAhonors/models',
                       help='Path to models directory')
    parser.add_argument('--results_dir', type=str,
                       default='/home/mohanganesh/VQAhonors/results',
                       help='Path to results directory')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to evaluate (for testing)')
    parser.add_argument('--model', type=str, default=None,
                       help='Specific model to evaluate (if not provided, evaluates all)')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for evaluation')
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("COMPLETE REAL MEDFRAMEQA EVALUATION PIPELINE")
    logger.info("="*80)
    logger.info("This script replaces ALL mock data generation with real model inference")
    logger.info("="*80)
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Models directory: {args.models_dir}")
    logger.info(f"Results directory: {args.results_dir}")
    logger.info(f"Max samples: {args.max_samples or 'All 2,851 samples'}")
    logger.info(f"Specific model: {args.model or 'All 6 models'}")
    logger.info("="*80)
    
    # Create evaluator
    evaluator = RealMedFrameQAEvaluator(
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        results_dir=args.results_dir
    )
    
    # Load dataset
    logger.info("Loading MedFrameQA dataset...")
    dataset = evaluator.load_dataset(max_samples=args.max_samples)
    logger.info(f"Dataset loaded: {len(dataset)} samples")
    
    # Run evaluation
    if args.model:
        # Evaluate specific model
        logger.info(f"Evaluating model: {args.model}")
        try:
            result = evaluator.evaluate_model(args.model, args.max_samples)
            logger.info(f"Evaluation completed for {args.model}: {result}")
        except Exception as e:
            logger.error(f"Error evaluating {args.model}: {e}")
    else:
        # Evaluate all models
        logger.info("Evaluating all 6 models...")
        try:
            results = evaluator.evaluate_all_models(args.max_samples)
            logger.info("All model evaluations completed")
            
            # Print summary
            logger.info("\n" + "="*60)
            logger.info("REAL EVALUATION SUMMARY")
            logger.info("="*60)
            for model_name, result in results.items():
                if 'error' in result:
                    logger.info(f"{model_name}: ERROR - {result['error']}")
                else:
                    logger.info(f"{model_name}: {result['accuracy']:.4f} accuracy "
                              f"({result['correct_predictions']}/{result['total_samples']}) "
                              f"in {result['evaluation_time']:.2f}s")
            logger.info("="*60)
            
        except Exception as e:
            logger.error(f"Error in evaluation: {e}")
    
    # Save results
    logger.info("Saving comprehensive results...")
    try:
        timestamp = evaluator.save_results("complete_real_evaluation")
        logger.info(f"Results saved with timestamp: {timestamp}")
        
        # Print file locations
        logger.info("\n" + "="*60)
        logger.info("RESULT FILES SAVED")
        logger.info("="*60)
        logger.info(f"- Summary: {args.results_dir}/complete_real_evaluation_summary_{timestamp}.json")
        logger.info(f"- Clinical breakdown: {args.results_dir}/complete_real_evaluation_clinical_breakdown_{timestamp}.json")
        logger.info(f"- Failure analysis: {args.results_dir}/complete_real_evaluation_failure_analysis_{timestamp}.json")
        logger.info(f"- Report: {args.results_dir}/real_evaluation_report_{timestamp}.md")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Error saving results: {e}")
    
    logger.info("Complete real evaluation pipeline finished!")
    logger.info("All mock data has been replaced with genuine model inference results.")

if __name__ == "__main__":
    main()
