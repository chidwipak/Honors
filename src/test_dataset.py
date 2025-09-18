import os
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from data_loader import load_medframeqa_dataset, visualize_sample
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dataset_test.log"),
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

def test_dataset_loading():
    """Test loading the MedFrameQA dataset"""
    logger.info("Testing MedFrameQA dataset loading")
    
    data_dir = "/home/mohanganesh/VQAhonors/data/MedFrameQA/data"
    transform = get_transform()
    
    # Load a small sample of the dataset
    dataloader = load_medframeqa_dataset(data_dir, transform, batch_size=2, max_samples=10)
    
    # Check if dataloader is working
    for batch in dataloader:
        logger.info(f"Batch size: {len(batch['question'])}")
        logger.info(f"Number of images in first sample: {len(batch['images'][0])}")
        logger.info(f"Question: {batch['question'][0]}")
        logger.info(f"Answer: {batch['answer'][0]}")
        
        # Save a visualization of the first sample
        sample = {
            'images': batch['images'][0],
            'question': batch['question'][0],
            'answer': batch['answer'][0]
        }
        
        # Create results directory if it doesn't exist
        os.makedirs("dataset_test", exist_ok=True)
        
        # Visualize and save the sample
        n_images = len(sample['images'])
        fig, axes = plt.subplots(1, n_images, figsize=(5*n_images, 5))
        
        if n_images == 1:
            axes = [axes]
        
        for i, img in enumerate(sample['images']):
            if isinstance(img, torch.Tensor):
                img = img.permute(1, 2, 0).numpy()
                img = (img - img.min()) / (img.max() - img.min())
            axes[i].imshow(img)
            axes[i].set_title(f"Image {i+1}")
            axes[i].axis('off')
        
        plt.suptitle(f"Q: {sample['question']}\nA: {sample['answer']}")
        plt.tight_layout()
        plt.savefig("dataset_test/sample_visualization.png")
        plt.close()
        
        logger.info("Sample visualization saved to dataset_test/sample_visualization.png")
        break
    
    logger.info("Dataset loading test completed successfully")

if __name__ == "__main__":
    test_dataset_loading()