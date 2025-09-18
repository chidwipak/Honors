#!/usr/bin/env python3
"""
Multi-Image Medical VQA: Master Critical Analysis Pipeline
Runs the complete failure analysis, visualization, and research conclusions pipeline.
"""

import subprocess
import sys
from pathlib import Path
import json

def run_script(script_name, description):
    """Run a Python script and handle errors"""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print(f"✓ SUCCESS: {description}")
            if result.stdout:
                print("Output:", result.stdout[-500:])  # Last 500 chars
        else:
            print(f"✗ FAILED: {description}")
            print("Error:", result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ TIMEOUT: {description}")
        return False
    except Exception as e:
        print(f"❌ ERROR: {description} - {str(e)}")
        return False
    
    return True

def check_dependencies():
    """Check if required packages are available"""
    required_packages = ['matplotlib', 'seaborn', 'pandas', 'numpy']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing required packages: {missing_packages}")
        print("Please install them with: pip install " + " ".join(missing_packages))
        return False
    
    return True

def main():
    """Main execution pipeline"""
    print("=" * 80)
    print("MULTI-IMAGE MEDICAL VQA: CRITICAL ANALYSIS PIPELINE")
    print("=" * 80)
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Check if results exist
    results_dir = Path("results")
    if not results_dir.exists():
        print("❌ Results directory not found. Please run model evaluation first.")
        return
    
    # Check for result files
    required_files = [
        "BiomedCLIP_complete_results_20250917_205335.json",
        "LLaVA-Med_complete_results_20250917_214123.json",
        "Biomedical-LLaMA_complete_results_20250917_211034.json",
        "PMC-VQA_complete_results_20250917_221003.json",
        "MedGemma_complete_results_20250918_065126.json",
        "Qwen2.5-VL_complete_results_20250918_012630.json"
    ]
    
    missing_files = []
    for file in required_files:
        if not (results_dir / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing result files: {missing_files}")
        return
    
    print("✓ All required result files found")
    
    # Create output directories
    Path("analysis_output").mkdir(exist_ok=True)
    Path("plots").mkdir(exist_ok=True)
    Path("research_output").mkdir(exist_ok=True)
    
    # Run analysis pipeline
    pipeline = [
        ("failure_analysis.py", "Comprehensive Failure Mode Analysis"),
        ("visualization_suite.py", "8 Critical Visualization Plots"),
        ("research_conclusions.py", "Research Conclusions & Solution Roadmap")
    ]
    
    success_count = 0
    total_steps = len(pipeline)
    
    for script, description in pipeline:
        if run_script(script, description):
            success_count += 1
        else:
            print(f"⚠️  Continuing with remaining steps...")
    
    # Summary
    print(f"\n{'='*80}")
    print("PIPELINE COMPLETION SUMMARY")
    print(f"{'='*80}")
    print(f"Completed: {success_count}/{total_steps} steps")
    
    if success_count == total_steps:
        print("🎉 ALL ANALYSIS COMPLETE!")
        print("\nGenerated outputs:")
        print("📊 analysis_output/ - Detailed failure analysis data")
        print("📈 plots/ - 8 comprehensive visualization plots")
        print("📝 research_output/ - Research conclusions and roadmap")
        
        # List generated files
        for output_dir in ["analysis_output", "plots", "research_output"]:
            if Path(output_dir).exists():
                files = list(Path(output_dir).glob("*"))
                if files:
                    print(f"\n{output_dir}/:")
                    for file in files:
                        print(f"  - {file.name}")
    else:
        print(f"⚠️  Partial completion: {success_count}/{total_steps} steps successful")
        print("Check error messages above for details.")
    
    print(f"\n{'='*80}")

if __name__ == "__main__":
    main()
