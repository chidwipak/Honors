import os
import torch
from PIL import Image
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from ..evaluation import ModelEvaluator
import logging

logger = logging.getLogger(__name__)

class BiomedCLIPEvaluator(ModelEvaluator):
    """Evaluator for BiomedCLIP model"""
    
    def __init__(self, model_path="/home/mohanganesh/VQAhonors/models/biomedclip", save_dir="results"):
        """
        Initialize the BiomedCLIP evaluator
        
        Args:
            model_path (str): Path to the model
            save_dir (str): Directory to save results
        """
        super().__init__("BiomedCLIP", save_dir)
        
        logger.info(f"Loading BiomedCLIP model from {model_path}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            self.model = CLIPModel.from_pretrained(model_path)
            self.processor = CLIPProcessor.from_pretrained(model_path)
            self.model.to(self.device)
            logger.info("BiomedCLIP model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading BiomedCLIP model: {e}")
            raise
        
        # Define a set of possible answers for medical VQA
        self.possible_answers = [
            "yes", "no", "normal", "abnormal", "present", "absent",
            "pneumonia", "fracture", "tumor", "infection", "inflammation",
            "cardiomegaly", "effusion", "nodule", "mass", "opacity",
            "lesion", "calcification", "atelectasis", "consolidation",
            "edema", "emphysema", "fibrosis", "pneumothorax", "hernia"
        ]
        
    def preprocess_batch(self, batch):
        """
        Preprocess a batch of data for BiomedCLIP
        
        Args:
            batch (dict): Batch of data from dataloader
            
        Returns:
            dict: Preprocessed batch
        """
        processed_batch = {
            "questions": [],
            "images": [],
            "answers": batch["answer"]
        }
        
        # Process each sample in the batch
        for i in range(len(batch["question"])):
            question = batch["question"][i]
            processed_batch["questions"].append(question)
            
            # Process images for this sample
            sample_images = batch["images"][i]
            if not isinstance(sample_images, list):
                sample_images = [sample_images]
                
            # Convert tensor images to PIL images if needed
            pil_images = []
            for img in sample_images:
                if isinstance(img, torch.Tensor):
                    # Convert tensor to PIL image
                    img = img.permute(1, 2, 0).cpu().numpy()
                    img = (img * 255).astype(np.uint8)
                    pil_img = Image.fromarray(img)
                    pil_images.append(pil_img)
                else:
                    pil_images.append(img)
            
            processed_batch["images"].append(pil_images)
            
        return processed_batch
    
    def predict_batch(self, batch):
        """
        Generate predictions for a batch of data using BiomedCLIP
        
        Args:
            batch (dict): Preprocessed batch of data
            
        Returns:
            list: List of predictions
        """
        predictions = []
        
        for i in range(len(batch["questions"])):
            question = batch["questions"][i]
            images = batch["images"][i]
            
            # For each image, compute similarity with question + possible answers
            best_scores = []
            
            for img in images:
                answer_scores = {}
                
                for answer in self.possible_answers:
                    text = f"{question} {answer}"
                    
                    # Process inputs
                    inputs = self.processor(
                        text=text,
                        images=img,
                        return_tensors="pt",
                        padding=True
                    ).to(self.device)
                    
                    # Get similarity score
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        logits_per_image = outputs.logits_per_image
                        score = logits_per_image.item()
                    
                    answer_scores[answer] = score
                
                # Get best answer for this image
                best_answer = max(answer_scores.items(), key=lambda x: x[1])[0]
                best_scores.append((best_answer, answer_scores[best_answer]))
            
            # Select the answer with the highest score across all images
            best_prediction = max(best_scores, key=lambda x: x[1])[0]
            predictions.append(best_prediction)
        
        return predictions