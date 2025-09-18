import os
import sys
import json
import logging
import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime
from transformers import CLIPProcessor, CLIPModel
from torchvision import transforms
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge

# Add src directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_medframeqa_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("biomedclip_evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BiomedCLIPEvaluator:
    """Evaluator for BiomedCLIP model"""
    
    def __init__(self, model_path="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"):
        self.model_name = "BiomedCLIP"
        
        logger.info(f"Loading {self.model_name} model from {model_path}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            self.model = CLIPModel.from_pretrained(model_path)
            self.processor = CLIPProcessor.from_pretrained(model_path)
            self.model.to(self.device)
            logger.info(f"{self.model_name} model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading {self.model_name} model: {e}")
            # Fallback to CLIP model if BiomedCLIP fails
            logger.info("Falling back to standard CLIP model")
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.model.to(self.device)
        
        # Define a set of possible answers for medical VQA
        self.possible_answers = [
            "yes", "no", "normal", "abnormal", "present", "absent",
            "pneumonia", "fracture", "tumor", "infection", "inflammation",
            "cardiomegaly", "effusion", "nodule", "mass", "opacity",
            "lesion", "calcification", "atelectasis", "consolidation",
            "edema", "emphysema", "fibrosis", "pneumothorax", "hernia"
        ]
    
    def preprocess_batch(self, batch):
        processed_batch = {
            "questions": [],
            "images": [],
            "answers": batch["answer"]
        }
        
        for i in range(len(batch["question"])):
            question = batch["question"][i]
            processed_batch["questions"].append(question)
            
            sample_images = batch["images"][i]
            if not isinstance(sample_images, list):
                sample_images = [sample_images]
                
            pil_images = []
            for img in sample_images:
                if isinstance(img, torch.Tensor):
                    img = img.permute(1, 2, 0).cpu().numpy()
                    img = (img * 255).astype(np.uint8)
                    pil_img = Image.fromarray(img)
                    pil_images.append(pil_img)
                else:
                    pil_images.append(img)
            
            processed_batch["images"].append(pil_images)
            
        return processed_batch
    
    def predict_batch(self, batch):
        predictions = []
        
        for i in range(len(batch["questions"])):
            question = batch["questions"][i]
            images = batch["images"][i]
            
            best_answers = []
            
            for img in images:
                answer_scores = {}
                
                for answer in self.possible_answers:
                    text = f"{question} {answer}"
                    
                    try:
                        inputs = self.processor(
                            text=text,
                            images=img,
                            return_tensors="pt",
                            padding=True
                        ).to(self.device)
                        
                        with torch.no_grad():
                            outputs = self.model(**inputs)
                            logits_per_image = outputs.logits_per_image
                            score = logits_per_image.item()
                        
                        answer_scores[answer] = score
                    except Exception as e:
                        logger.warning(f"Error processing answer '{answer}': {e}")
                        answer_scores[answer] = -float('inf')
                
                if answer_scores:
                    best_answer = max(answer_scores.items(), key=lambda x: x[1])[0]
                    best_answers.append(best_answer)
                else:
                    best_answers.append("unknown")
            
            # Combine answers from multiple images
            if best_answers:
                # Use the most frequent answer
                from collections import Counter
                prediction = Counter(best_answers).most_common(1)[0][0]
            else:
                prediction = "unknown"
            
            predictions.append(prediction)
        
        return predictions

def calculate_metrics(predictions, ground_truth):
    """Calculate evaluation metrics"""
    metrics = {
        'bleu': [],
        'meteor': [],
        'rouge_1': [],
        'rouge_2': [],
        'rouge_l': [],
        'exact_match': []
    }
    
    rouge = Rouge()
    
    for pred, gt in zip(predictions, ground_truth):
        # BLEU score
        try:
            bleu = sentence_bleu([gt.split()], pred.split())
            metrics['bleu'].append(bleu)
        except Exception as e:
            logger.warning(f"Error calculating BLEU: {e}")
            metrics['bleu'].append(0)
        
        # METEOR score
        try:
            meteor = meteor_score([gt.split()], pred.split())
            metrics['meteor'].append(meteor)
        except Exception as e:
            logger.warning(f"Error calculating METEOR: {e}")
            metrics['meteor'].append(0)
        
        # ROUGE scores
        try:
            rouge_scores = rouge.get_scores(pred, gt)[0]
            metrics['rouge_1'].append(rouge_scores['rouge-1']['f'])
            metrics['rouge_2'].append(rouge_scores['rouge-2']['f'])
            metrics['rouge_l'].append(rouge_scores['rouge-l']['f'])
        except Exception as e:
            logger.warning(f"Error calculating ROUGE: {e}")
            metrics['rouge_1'].append(0)
            metrics['rouge_2'].append(0)
            metrics['rouge_l'].append(0)
        
        # Exact match
        metrics['exact_match'].append(1 if pred.lower() == gt.lower() else 0)
    
    # Calculate averages
    results = {}
    for metric, values in metrics.items():
        results[metric] = np.mean(values) if values else 0
    
    return results

def main():
    from PIL import Image
    
    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load the dataset
    data_dir = "/home/mohanganesh/VQAhonors/data/MedFrameQA/data"
    batch_size = 8
    
    logger.info(f"Loading MedFrameQA dataset from {data_dir}")
    dataloader, dataset = load_medframeqa_dataset(
        data_dir=data_dir,
        transform=transform,
        batch_size=batch_size,
        max_samples=None,  # Use the full dataset
        shuffle=False,     # No need to shuffle for evaluation
        num_workers=4
    )
    logger.info(f"Loaded {len(dataset)} samples")
    
    # Initialize model evaluator
    evaluator = BiomedCLIPEvaluator()
    
    # Evaluate model
    logger.info(f"Evaluating {evaluator.model_name} model")
    
    all_predictions = []
    all_ground_truth = []
    all_questions = []
    all_sample_ids = []
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Evaluating {evaluator.model_name}")):
        try:
            # Preprocess batch
            processed_batch = evaluator.preprocess_batch(batch)
            
            # Get predictions
            predictions = evaluator.predict_batch(processed_batch)
            
            # Store results
            all_predictions.extend(predictions)
            all_ground_truth.extend(batch["answer"])
            all_questions.extend(batch["question"])
            all_sample_ids.extend([str(id) for id in batch["id"]])
            
            # Log progress
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Processed {(batch_idx + 1) * batch_size} samples")
                
        except Exception as e:
            logger.error(f"Error processing batch {batch_idx}: {e}")
    
    # Calculate metrics
    metrics = calculate_metrics(all_predictions, all_ground_truth)
    
    # Save results
    results = {
        "model_name": evaluator.model_name,
        "metrics": metrics,
        "samples": [
            {
                "id": id,
                "question": q,
                "ground_truth": gt,
                "prediction": pred
            }
            for id, q, gt, pred in zip(all_sample_ids, all_questions, all_ground_truth, all_predictions)
        ]
    }
    
    # Save to file
    os.makedirs("results", exist_ok=True)
    results_file = os.path.join("results", f"{evaluator.model_name.lower()}_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {results_file}")
    logger.info(f"Metrics for {evaluator.model_name}: {metrics}")

if __name__ == "__main__":
    main()