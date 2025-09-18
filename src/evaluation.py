import os
import json
import torch
import numpy as np
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
import nltk
from sklearn.metrics import accuracy_score, f1_score
import time
import logging

# Download necessary NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    print("NLTK download failed, but continuing...")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Base class for model evaluation"""
    
    def __init__(self, model_name, save_dir="results"):
        """
        Initialize the evaluator
        
        Args:
            model_name (str): Name of the model
            save_dir (str): Directory to save results
        """
        self.model_name = model_name
        self.save_dir = save_dir
        self.results = {
            "model_name": model_name,
            "predictions": [],
            "metrics": {},
            "time_taken": 0
        }
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
    def preprocess_batch(self, batch):
        """
        Preprocess a batch of data
        
        Args:
            batch (dict): Batch of data from dataloader
            
        Returns:
            dict: Preprocessed batch
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def predict_batch(self, batch):
        """
        Generate predictions for a batch of data
        
        Args:
            batch (dict): Preprocessed batch of data
            
        Returns:
            list: List of predictions
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def evaluate(self, dataloader, max_batches=None):
        """
        Evaluate the model on the dataset
        
        Args:
            dataloader: DataLoader for the dataset
            max_batches (int, optional): Maximum number of batches to evaluate
            
        Returns:
            dict: Evaluation results
        """
        logger.info(f"Starting evaluation for {self.model_name}")
        start_time = time.time()
        
        all_predictions = []
        all_references = []
        
        for i, batch in enumerate(tqdm(dataloader, desc=f"Evaluating {self.model_name}")):
            if max_batches is not None and i >= max_batches:
                break
                
            # Preprocess batch
            processed_batch = self.preprocess_batch(batch)
            
            # Generate predictions
            predictions = self.predict_batch(processed_batch)
            
            # Store predictions and references
            for j, pred in enumerate(predictions):
                sample_id = batch['id'][j] if isinstance(batch['id'], list) else batch['id'][j].item()
                question = batch['question'][j]
                reference = batch['answer'][j]
                
                all_predictions.append(pred)
                all_references.append(reference)
                
                self.results["predictions"].append({
                    "id": sample_id,
                    "question": question,
                    "reference": reference,
                    "prediction": pred
                })
        
        # Calculate metrics
        self.results["metrics"] = self.calculate_metrics(all_predictions, all_references)
        
        # Record time taken
        self.results["time_taken"] = time.time() - start_time
        
        # Save results
        self.save_results()
        
        logger.info(f"Evaluation completed for {self.model_name}")
        logger.info(f"Metrics: {self.results['metrics']}")
        
        return self.results
    
    def calculate_metrics(self, predictions, references):
        """
        Calculate evaluation metrics
        
        Args:
            predictions (list): List of model predictions
            references (list): List of ground truth answers
            
        Returns:
            dict: Dictionary of metrics
        """
        metrics = {}
        
        # BLEU score
        bleu_scores = []
        for pred, ref in zip(predictions, references):
            pred_tokens = nltk.word_tokenize(pred.lower())
            ref_tokens = [nltk.word_tokenize(ref.lower())]
            try:
                bleu = sentence_bleu(ref_tokens, pred_tokens)
                bleu_scores.append(bleu)
            except:
                bleu_scores.append(0)
        
        metrics["bleu"] = np.mean(bleu_scores)
        
        # METEOR score
        meteor_scores = []
        for pred, ref in zip(predictions, references):
            try:
                score = meteor_score([ref.split()], pred.split())
                meteor_scores.append(score)
            except:
                meteor_scores.append(0)
        
        metrics["meteor"] = np.mean(meteor_scores)
        
        # ROUGE score
        try:
            rouge = Rouge()
            rouge_scores = rouge.get_scores(predictions, references, avg=True)
            metrics["rouge-1"] = rouge_scores["rouge-1"]["f"]
            metrics["rouge-2"] = rouge_scores["rouge-2"]["f"]
            metrics["rouge-l"] = rouge_scores["rouge-l"]["f"]
        except Exception as e:
            logger.error(f"Error calculating ROUGE scores: {e}")
            metrics["rouge-1"] = 0
            metrics["rouge-2"] = 0
            metrics["rouge-l"] = 0
        
        # Exact match
        exact_matches = [1 if pred.strip().lower() == ref.strip().lower() else 0 
                         for pred, ref in zip(predictions, references)]
        metrics["exact_match"] = np.mean(exact_matches)
        
        return metrics
    
    def save_results(self):
        """Save evaluation results to a file"""
        result_file = os.path.join(self.save_dir, f"{self.model_name}_results.json")
        with open(result_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Results saved to {result_file}")
        
        # Also save metrics separately for easy comparison
        metrics_file = os.path.join(self.save_dir, "all_metrics.json")
        
        # Load existing metrics if file exists
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                all_metrics = json.load(f)
        else:
            all_metrics = {}
        
        # Add current model metrics
        all_metrics[self.model_name] = {
            "metrics": self.results["metrics"],
            "time_taken": self.results["time_taken"]
        }
        
        # Save updated metrics
        with open(metrics_file, 'w') as f:
            json.dump(all_metrics, f, indent=2)