import os
import argparse
import torch
import logging
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import json

# Import our modules
from data_loader import load_medframeqa_dataset
from models.model_evaluators import (
    BiomedCLIPEvaluator,
    BiomedicalLLaMAEvaluator,
    LLaVAMedEvaluator,
    MedGemmaEvaluator,
    PMCVQAEvaluator,
    Qwen25VLEvaluator
)

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

def get_transform():
    """Get image transformation for the dataset"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def evaluate_model(model_name, data_dir, results_dir, max_samples=None, batch_size=8):
    """
    Evaluate a specific model on the dataset
    
    Args:
        model_name (str): Name of the model to evaluate
        data_dir (str): Directory containing the dataset
        results_dir (str): Directory to save results
        max_samples (int, optional): Maximum number of samples to evaluate
        batch_size (int): Batch size for evaluation
    """
    logger.info(f"Starting evaluation for {model_name}")
    
    # Load dataset
    transform = get_transform()
    dataloader = load_medframeqa_dataset(data_dir, transform, batch_size, max_samples)
    
    # Initialize evaluator based on model name
    if model_name == "biomedclip":
        evaluator = BiomedCLIPEvaluator(save_dir=results_dir)
    elif model_name == "biomedllama":
        evaluator = BiomedicalLLaMAEvaluator(save_dir=results_dir)
    elif model_name == "llavamed":
        evaluator = LLaVAMedEvaluator(save_dir=results_dir)
    elif model_name == "medgemma":
        evaluator = MedGemmaEvaluator(save_dir=results_dir)
    elif model_name == "pmcvqa":
        evaluator = PMCVQAEvaluator(save_dir=results_dir)
    elif model_name == "qwen25vl":
        evaluator = Qwen25VLEvaluator(save_dir=results_dir)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Run evaluation
    results = evaluator.evaluate(dataloader)
    
    logger.info(f"Evaluation completed for {model_name}")
    logger.info(f"Metrics: {results['metrics']}")
    
    return results

def evaluate_all_models(data_dir, results_dir, max_samples=None, batch_size=8):
    """
    Evaluate all models on the dataset
    
    Args:
        data_dir (str): Directory containing the dataset
        results_dir (str): Directory to save results
        max_samples (int, optional): Maximum number of samples to evaluate
        batch_size (int): Batch size for evaluation
    """
    models = [
        "biomedclip",
        "biomedllama",
        "llavamed",
        "medgemma",
        "pmcvqa",
        "qwen25vl"
    ]
    
    all_results = {}
    
    for model_name in models:
        try:
            results = evaluate_model(model_name, data_dir, results_dir, max_samples, batch_size)
            all_results[model_name] = results["metrics"]
        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {e}")
    
    # Save combined results
    with open(os.path.join(results_dir, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Create comparison plots
    create_comparison_plots(all_results, results_dir)
    
    return all_results

def create_comparison_plots(all_results, results_dir):
    """
    Create comparison plots for all models
    
    Args:
        all_results (dict): Dictionary of results for all models
        results_dir (str): Directory to save plots
    """
    # Create a DataFrame for easier plotting
    metrics = ["bleu", "meteor", "rouge-1", "rouge-2", "rouge-l", "exact_match"]
    data = []
    
    for model_name, metrics_dict in all_results.items():
        row = {"model": model_name}
        for metric in metrics:
            if metric in metrics_dict:
                row[metric] = metrics_dict[metric]
            else:
                row[metric] = 0
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Create bar plots for each metric
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        plt.bar(df["model"], df[metric])
        plt.title(f"{metric.upper()} Score Comparison")
        plt.xlabel("Model")
        plt.ylabel(f"{metric.upper()} Score")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"{metric}_comparison.png"))
        plt.close()
    
    # Create a combined plot
    plt.figure(figsize=(12, 8))
    
    x = range(len(df["model"]))
    width = 0.1
    
    for i, metric in enumerate(metrics):
        plt.bar([p + width*i for p in x], df[metric], width=width, label=metric.upper())
    
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.title("All Metrics Comparison")
    plt.xticks([p + width*2 for p in x], df["model"], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "all_metrics_comparison.png"))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Medical VQA Evaluation")
    parser.add_argument("--model", type=str, default="all", 
                        help="Model to evaluate (biomedclip, biomedllama, llavamed, medgemma, pmcvqa, qwen25vl, or all)")
    parser.add_argument("--data_dir", type=str, default="/home/mohanganesh/VQAhonors/data/MedFrameQA/data",
                        help="Directory containing the dataset")
    parser.add_argument("--results_dir", type=str, default="/home/mohanganesh/VQAhonors/results",
                        help="Directory to save results")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to evaluate")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for evaluation")
    
    args = parser.parse_args()
    
    # Create results directory if it doesn't exist
    os.makedirs(args.results_dir, exist_ok=True)
    
    if args.model == "all":
        evaluate_all_models(args.data_dir, args.results_dir, args.max_samples, args.batch_size)
    else:
        evaluate_model(args.model, args.data_dir, args.results_dir, args.max_samples, args.batch_size)

if __name__ == "__main__":
    main()