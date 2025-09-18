import os
import sys
import json
import logging
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from transformers import set_seed
from torchvision import transforms
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge

# Add src directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_medframeqa_dataset
from src.models.model_evaluators import (
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

# Set random seed for reproducibility
set_seed(42)

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

def evaluate_model(model_evaluator, dataloader, batch_size=8):
    """Evaluate a model on the dataset"""
    logger.info(f"Evaluating {model_evaluator.model_name} model")
    
    all_predictions = []
    all_ground_truth = []
    all_questions = []
    all_sample_ids = []
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Evaluating {model_evaluator.model_name}")):
        try:
            # Preprocess batch
            processed_batch = model_evaluator.preprocess_batch(batch)
            
            # Get predictions
            predictions = model_evaluator.predict_batch(processed_batch)
            
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
        "model_name": model_evaluator.model_name,
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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("results", timestamp)
    os.makedirs(results_dir, exist_ok=True)
    
    results_file = os.path.join(results_dir, f"{model_evaluator.model_name.lower()}_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {results_file}")
    logger.info(f"Metrics for {model_evaluator.model_name}: {metrics}")
    
    return results

def compare_models(results_list):
    """Compare results from different models"""
    comparison = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "models": {}
    }
    
    for result in results_list:
        model_name = result["model_name"]
        comparison["models"][model_name] = result["metrics"]
    
    # Determine best model for each metric
    best_models = {}
    for metric in ["bleu", "meteor", "rouge_1", "rouge_2", "rouge_l", "exact_match"]:
        best_score = -1
        best_model = None
        
        for model_name, metrics in comparison["models"].items():
            score = metrics.get(metric, 0)
            if score > best_score:
                best_score = score
                best_model = model_name
        
        best_models[metric] = {
            "model": best_model,
            "score": best_score
        }
    
    comparison["best_models"] = best_models
    
    # Determine overall best model
    model_scores = {}
    for model_name in comparison["models"].keys():
        model_scores[model_name] = sum(
            comparison["models"][model_name].get(metric, 0)
            for metric in ["bleu", "meteor", "rouge_l", "exact_match"]
        )
    
    best_model = max(model_scores.items(), key=lambda x: x[1])
    comparison["overall_best_model"] = {
        "model": best_model[0],
        "score": best_model[1]
    }
    
    # Save comparison to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("results", timestamp)
    os.makedirs(results_dir, exist_ok=True)
    
    comparison_file = os.path.join(results_dir, "model_comparison.json")
    with open(comparison_file, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    logger.info(f"Model comparison saved to {comparison_file}")
    
    return comparison

def visualize_results(comparison):
    """Visualize model comparison results"""
    models = list(comparison["models"].keys())
    metrics = ["bleu", "meteor", "rouge_1", "rouge_2", "rouge_l", "exact_match"]
    
    # Create a DataFrame for easier plotting
    data = []
    for model in models:
        model_metrics = comparison["models"][model]
        row = [model]
        for metric in metrics:
            row.append(model_metrics.get(metric, 0))
        data.append(row)
    
    df = pd.DataFrame(data, columns=["Model"] + metrics)
    df.set_index("Model", inplace=True)
    
    # Plot
    plt.figure(figsize=(12, 8))
    df.plot(kind="bar", figsize=(12, 8))
    plt.title("Model Performance Comparison")
    plt.ylabel("Score")
    plt.xlabel("Model")
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("results", timestamp)
    os.makedirs(results_dir, exist_ok=True)
    
    plot_file = os.path.join(results_dir, "model_comparison.png")
    plt.savefig(plot_file)
    
    logger.info(f"Visualization saved to {plot_file}")

def generate_report(comparison):
    """Generate a summary report"""
    report = []
    
    report.append("# Medical VQA Model Evaluation Report")
    report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    report.append("## Overall Best Model")
    best_model = comparison["overall_best_model"]["model"]
    best_score = comparison["overall_best_model"]["score"]
    report.append(f"The best performing model overall is **{best_model}** with a combined score of {best_score:.4f}.\n")
    
    report.append("## Best Models by Metric")
    for metric, info in comparison["best_models"].items():
        report.append(f"- **{metric}**: {info['model']} ({info['score']:.4f})")
    report.append("")
    
    report.append("## All Model Metrics")
    report.append("| Model | BLEU | METEOR | ROUGE-1 | ROUGE-2 | ROUGE-L | Exact Match |")
    report.append("| ----- | ---- | ------ | ------- | ------- | ------- | ----------- |")
    
    for model, metrics in comparison["models"].items():
        row = [
            model,
            f"{metrics.get('bleu', 0):.4f}",
            f"{metrics.get('meteor', 0):.4f}",
            f"{metrics.get('rouge_1', 0):.4f}",
            f"{metrics.get('rouge_2', 0):.4f}",
            f"{metrics.get('rouge_l', 0):.4f}",
            f"{metrics.get('exact_match', 0):.4f}"
        ]
        report.append("| " + " | ".join(row) + " |")
    
    report.append("\n## Conclusion")
    report.append(f"Based on the evaluation metrics, the {best_model} model performs best overall for medical visual question answering on the MedFrameQA dataset.")
    
    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("results", timestamp)
    os.makedirs(results_dir, exist_ok=True)
    
    report_file = os.path.join(results_dir, "evaluation_report.md")
    with open(report_file, 'w') as f:
        f.write("\n".join(report))
    
    logger.info(f"Report saved to {report_file}")

def main():
    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load the complete dataset
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
    
    # Initialize model evaluators
    model_evaluators = [
        BiomedCLIPEvaluator(model_path="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"),
        BiomedicalLLaMAEvaluator(model_path="ContactDoctor/Bio-Medical-Llama-3-8B"),
        LLaVAMedEvaluator(model_path="microsoft/llava-med-v1.5-mistral-7b"),
        MedGemmaEvaluator(model_path="google/medgemma-4b-it"),
        PMCVQAEvaluator(model_path="microsoft/pmc-vqa"),
        Qwen25VLEvaluator(model_path="Qwen/Qwen2.5-VL-7B-Instruct")
    ]
    
    # Evaluate each model
    results_list = []
    for evaluator in model_evaluators:
        try:
            results = evaluate_model(evaluator, dataloader, batch_size)
            results_list.append(results)
        except Exception as e:
            logger.error(f"Error evaluating {evaluator.model_name}: {e}")
    
    # Compare models
    comparison = compare_models(results_list)
    
    # Visualize results
    visualize_results(comparison)
    
    # Generate report
    generate_report(comparison)
    
    logger.info("Evaluation complete!")

if __name__ == "__main__":
    main()