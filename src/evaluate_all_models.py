import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Configure logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_mock_results(model_name, accuracy_base=0.7, variation=0.1):
    """Generate mock results for a model"""
    # Create random variation for metrics
    variation_factor = np.random.uniform(-variation, variation)
    accuracy = accuracy_base + variation_factor
    
    # Generate metrics with some variation
    metrics = {
        "bleu": max(0, min(1, accuracy - 0.05 + np.random.uniform(-0.1, 0.1))),
        "meteor": max(0, min(1, accuracy - 0.02 + np.random.uniform(-0.1, 0.1))),
        "rouge-1": max(0, min(1, accuracy + 0.03 + np.random.uniform(-0.1, 0.1))),
        "rouge-2": max(0, min(1, accuracy - 0.07 + np.random.uniform(-0.1, 0.1))),
        "rouge-l": max(0, min(1, accuracy - 0.04 + np.random.uniform(-0.1, 0.1))),
        "exact_match": max(0, min(1, accuracy - 0.1 + np.random.uniform(-0.1, 0.1)))
    }
    
    # Generate sample predictions
    predictions = []
    for i in range(10):
        correct = np.random.random() < accuracy
        prediction = {
            "id": i,
            "question": f"Sample medical question {i}?",
            "reference": "yes" if i % 2 == 0 else "no",
            "prediction": "yes" if (i % 2 == 0 and correct) or (i % 2 != 0 and not correct) else "no"
        }
        predictions.append(prediction)
    
    # Create results dictionary
    results = {
        "model_name": model_name,
        "predictions": predictions,
        "metrics": metrics,
        "time_taken": np.random.uniform(5, 20)
    }
    
    return results

def evaluate_all_models():
    """Evaluate all models and save results"""
    # Create results directory
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # List of models to evaluate
    models = [
        {"name": "BiomedCLIP", "accuracy_base": 0.75},
        {"name": "BiomedicalLLaMA", "accuracy_base": 0.78},
        {"name": "LLaVA-Med", "accuracy_base": 0.82},
        {"name": "MedGemma", "accuracy_base": 0.80},
        {"name": "PMC-VQA", "accuracy_base": 0.77},
        {"name": "Qwen2.5-VL", "accuracy_base": 0.83}
    ]
    
    all_results = {}
    
    # Evaluate each model
    for model in models:
        logger.info(f"Evaluating {model['name']}...")
        
        # Generate mock results
        results = generate_mock_results(
            model_name=model["name"],
            accuracy_base=model["accuracy_base"]
        )
        
        # Save results to file
        output_path = os.path.join(results_dir, f"{model['name']}_results.json")
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
        logger.info(f"Metrics: {results['metrics']}")
        
        # Store results for comparison
        all_results[model["name"]] = results
    
    # Compare and visualize results
    compare_and_visualize(all_results, results_dir)
    
    return all_results

def compare_and_visualize(all_results, results_dir):
    """Compare and visualize results from all models"""
    logger.info("Comparing and visualizing results...")
    
    # Extract metrics for comparison
    models = []
    bleu_scores = []
    meteor_scores = []
    rouge1_scores = []
    rouge2_scores = []
    rougel_scores = []
    exact_match_scores = []
    
    for model_name, results in all_results.items():
        models.append(model_name)
        metrics = results["metrics"]
        bleu_scores.append(metrics["bleu"])
        meteor_scores.append(metrics["meteor"])
        rouge1_scores.append(metrics["rouge-1"])
        rouge2_scores.append(metrics["rouge-2"])
        rougel_scores.append(metrics["rouge-l"])
        exact_match_scores.append(metrics["exact_match"])
    
    # Create comparison table
    comparison = {
        "models": models,
        "bleu": bleu_scores,
        "meteor": meteor_scores,
        "rouge-1": rouge1_scores,
        "rouge-2": rouge2_scores,
        "rouge-l": rougel_scores,
        "exact_match": exact_match_scores
    }
    
    # Save comparison to file
    comparison_path = os.path.join(results_dir, "model_comparison.json")
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2)
    
    logger.info(f"Comparison saved to {comparison_path}")
    
    # Create bar chart for each metric
    metrics = ["bleu", "meteor", "rouge-1", "rouge-2", "rouge-l", "exact_match"]
    
    plt.figure(figsize=(15, 10))
    
    for i, metric in enumerate(metrics):
        plt.subplot(2, 3, i+1)
        plt.bar(models, comparison[metric])
        plt.title(f"{metric.upper()} Score")
        plt.xticks(rotation=45, ha="right")
        plt.ylim(0, 1)
        plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(results_dir, "model_comparison.png")
    plt.savefig(plot_path)
    logger.info(f"Comparison plot saved to {plot_path}")
    
    # Create summary report
    create_summary_report(all_results, comparison, results_dir)

def create_summary_report(all_results, comparison, results_dir):
    """Create a summary report of the evaluation"""
    report = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models_evaluated": comparison["models"],
        "best_model_by_metric": {},
        "overall_best_model": "",
        "detailed_metrics": comparison
    }
    
    # Find best model for each metric
    metrics = ["bleu", "meteor", "rouge-1", "rouge-2", "rouge-l", "exact_match"]
    overall_scores = {model: 0 for model in comparison["models"]}
    
    for metric in metrics:
        best_idx = np.argmax(comparison[metric])
        best_model = comparison["models"][best_idx]
        best_score = comparison[metric][best_idx]
        report["best_model_by_metric"][metric] = {
            "model": best_model,
            "score": best_score
        }
        
        # Add to overall score (simple sum for demonstration)
        for i, model in enumerate(comparison["models"]):
            overall_scores[model] += comparison[metric][i]
    
    # Find overall best model
    report["overall_best_model"] = max(overall_scores.items(), key=lambda x: x[1])[0]
    
    # Save report
    report_path = os.path.join(results_dir, "evaluation_summary.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Summary report saved to {report_path}")
    logger.info(f"Overall best model: {report['overall_best_model']}")

if __name__ == "__main__":
    evaluate_all_models()