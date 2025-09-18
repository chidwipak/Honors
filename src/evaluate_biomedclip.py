import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
import logging
from tqdm import tqdm
import json
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download NLTK data
nltk.download('wordnet')
nltk.download('punkt')

class BiomedCLIPEvaluator:
    """Evaluator for BiomedCLIP model"""
    
    def __init__(self, model_path="openai/clip-vit-base-patch32", save_dir="results"):
        self.model_name = "BiomedCLIP"
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        logger.info(f"Loading CLIP model from {model_path}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            self.model = CLIPModel.from_pretrained(model_path)
            self.processor = CLIPProcessor.from_pretrained(model_path)
            self.model.to(self.device)
            logger.info("BiomedCLIP model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading BiomedCLIP model: {e}")
            raise
        
        # Set of possible medical answers for zero-shot classification
        self.possible_answers = [
            "yes", "no", "normal", "abnormal", "pneumonia", "fracture", "tumor",
            "cardiomegaly", "edema", "atelectasis", "pneumothorax", "consolidation",
            "effusion", "infiltration", "mass", "nodule", "fibrosis", "emphysema",
            "pleural thickening", "hernia", "calcification", "opacity", "lesion",
            "fluid", "hemorrhage", "inflammation", "infection", "cancer", "cyst",
            "stone", "gallstone", "kidney stone", "appendicitis", "arthritis",
            "osteoporosis", "degenerative", "stenosis", "sclerosis", "scoliosis"
        ]
    
    def preprocess_batch(self, batch):
        """Preprocess a batch of data for the model"""
        images = []
        questions = []
        
        for item in batch:
            image = item["image"]
            question = item["question"]
            
            images.append(image)
            questions.append(question)
            
        # Process images and text
        inputs = self.processor(
            text=questions,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        return inputs, questions
    
    def predict_batch(self, inputs, questions):
        """Make predictions for a batch of data"""
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Get image and text features
        image_features = outputs.image_embeds
        text_features = outputs.text_embeds
        
        predictions = []
        
        # For each question, find the most similar answer
        for i, question in enumerate(questions):
            # Get the image embedding for this question
            img_emb = image_features[i].unsqueeze(0)
            
            # Process all possible answers with the question
            answer_inputs = self.processor(
                text=[f"{question} {answer}" for answer in self.possible_answers],
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)
            
            with torch.no_grad():
                answer_outputs = self.model.get_text_features(**answer_inputs)
            
            # Normalize embeddings
            img_emb = img_emb / img_emb.norm(dim=1, keepdim=True)
            answer_outputs = answer_outputs / answer_outputs.norm(dim=1, keepdim=True)
            
            # Calculate similarity scores
            similarity = torch.matmul(img_emb, answer_outputs.T).squeeze(0)
            
            # Get the most similar answer
            best_idx = similarity.argmax().item()
            prediction = self.possible_answers[best_idx]
            
            predictions.append(prediction)
        
        return predictions
    
    def evaluate(self, dataset, batch_size=8):
        """Evaluate the model on the dataset"""
        logger.info(f"Evaluating {self.model_name} on {len(dataset)} samples")
        
        all_results = []
        
        # Process in batches
        for i in tqdm(range(0, len(dataset), batch_size)):
            batch = dataset[i:i+batch_size]
            
            # Preprocess batch
            inputs, questions = self.preprocess_batch(batch)
            
            # Make predictions
            predictions = self.predict_batch(inputs, questions)
            
            # Store results
            for j, item in enumerate(batch):
                result = {
                    "id": item["id"],
                    "question": item["question"],
                    "reference": item["answer"],
                    "prediction": predictions[j]
                }
                all_results.append(result)
        
        # Calculate metrics
        metrics = self.calculate_metrics(all_results)
        
        # Save results
        self.save_results(all_results, metrics)
        
        return all_results, metrics
    
    def calculate_metrics(self, results):
        """Calculate evaluation metrics"""
        references = [r["reference"] for r in results]
        predictions = [r["prediction"] for r in results]
        
        # Calculate BLEU score
        bleu_scores = []
        for ref, pred in zip(references, predictions):
            ref_tokens = nltk.word_tokenize(ref.lower())
            pred_tokens = nltk.word_tokenize(pred.lower())
            bleu_scores.append(sentence_bleu([ref_tokens], pred_tokens))
        
        bleu = np.mean(bleu_scores)
        
        # Calculate METEOR score
        meteor_scores = []
        for ref, pred in zip(references, predictions):
            meteor_scores.append(meteor_score([ref], pred))
        
        meteor = np.mean(meteor_scores)
        
        # Calculate ROUGE scores
        rouge = Rouge()
        rouge_scores = rouge.get_scores(predictions, references, avg=True)
        
        # Calculate exact match
        exact_matches = [1 if r == p else 0 for r, p in zip(references, predictions)]
        exact_match = np.mean(exact_matches)
        
        metrics = {
            "bleu": bleu,
            "meteor": meteor,
            "rouge-1": rouge_scores["rouge-1"]["f"],
            "rouge-2": rouge_scores["rouge-2"]["f"],
            "rouge-l": rouge_scores["rouge-l"]["f"],
            "exact_match": exact_match
        }
        
        return metrics
    
    def save_results(self, results, metrics):
        """Save evaluation results to file"""
        output = {
            "model_name": self.model_name,
            "predictions": results,
            "metrics": metrics
        }
        
        output_path = os.path.join(self.save_dir, f"{self.model_name}_results.json")
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
        logger.info(f"Metrics: {metrics}")

def load_sample_dataset(size=10):
    """Load a sample dataset for testing"""
    # Create a mock dataset
    dataset = []
    
    for i in range(size):
        # Create a blank image for testing
        image = Image.new('RGB', (224, 224), color='white')
        
        # Create a sample item
        item = {
            "id": i,
            "image": image,
            "question": f"Is this image normal? {i}",
            "answer": "yes" if i % 2 == 0 else "no"
        }
        
        dataset.append(item)
    
    return dataset

if __name__ == "__main__":
    # Load sample dataset
    dataset = load_sample_dataset(size=10)
    
    # Initialize evaluator
    evaluator = BiomedCLIPEvaluator(save_dir="results")
    
    # Evaluate model
    results, metrics = evaluator.evaluate(dataset, batch_size=2)
    
    print(f"Evaluation completed with metrics: {metrics}")