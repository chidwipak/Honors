import os
import torch
from torchvision import transforms
from data_loader import load_medframeqa_dataset
from models.model_evaluators import BiomedCLIPEvaluator
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("test_evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_transform():
    """Get image transformation for the dataset"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def test_biomedclip():
    """Test BiomedCLIP model evaluation"""
    logger.info("Testing BiomedCLIP evaluation")
    
    # Load a small sample of the dataset
    data_dir = "/home/mohanganesh/VQAhonors/data/MedFrameQA/data"
    transform = get_transform()
    dataloader = load_medframeqa_dataset(data_dir, transform, batch_size=2, max_samples=5)
    
    # Initialize evaluator
    evaluator = BiomedCLIPEvaluator(save_dir="test_results")
    
    # Run evaluation
    results = evaluator.evaluate(dataloader)
    
    logger.info(f"Test evaluation completed for BiomedCLIP")
    logger.info(f"Metrics: {results['metrics']}")
    
    return results

if __name__ == "__main__":
    # Create test results directory
    os.makedirs("test_results", exist_ok=True)
    
    # Test BiomedCLIP
    test_biomedclip()