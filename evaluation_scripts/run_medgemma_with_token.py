#!/usr/bin/env python3
"""
Run evaluation for MedGemma with HuggingFace token
"""

import os
import sys
import json
import logging
import time
from datetime import datetime

# Add src to path
sys.path.append('src')

from real_evaluation_pipeline_updated import RealMedFrameQAEvaluator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('medgemma_with_token_evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Main function to run MedGemma evaluation with token"""
    logger.info("Starting MedGemma evaluation with HuggingFace token...")
    
    # Set HuggingFace token
    os.environ['HF_TOKEN'] = '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'
    os.environ['HUGGINGFACE_HUB_TOKEN'] = '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'
    
    # Configuration
    data_dir = "/home/mohanganesh/VQAhonors/data/MedFrameQA/data"
    models_dir = "/home/mohanganesh/VQAhonors/models"
    results_dir = "/home/mohanganesh/VQAhonors/results"
    
    try:
        # Create evaluator
        evaluator = RealMedFrameQAEvaluator(
            data_dir=data_dir,
            models_dir=models_dir,
            results_dir=results_dir
        )
        
        # Load dataset
        logger.info("Loading dataset...")
        dataset = evaluator.load_dataset()
        logger.info(f"Loaded {len(dataset)} samples")
        
        # Run evaluation
        logger.info("Starting MedGemma evaluation...")
        result = evaluator.evaluate_model('MedGemma', max_samples=None)  # Run all samples
        
        logger.info(f"Evaluation completed: {result}")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = os.path.join(results_dir, f"MedGemma_complete_results_{timestamp}.json")
        
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"Results saved to: {result_file}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
