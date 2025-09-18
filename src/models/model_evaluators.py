import os
import torch
import logging
from PIL import Image
import numpy as np
from transformers import (
    CLIPProcessor, CLIPModel,
    AutoModelForCausalLM, AutoProcessor, AutoTokenizer,
    AutoModelForVision2Seq
)
from ..evaluation import ModelEvaluator

logger = logging.getLogger(__name__)

class BiomedCLIPEvaluator(ModelEvaluator):
    """Evaluator for BiomedCLIP model"""
    
    def __init__(self, model_path="/home/mohanganesh/VQAhonors/models/biomedclip", save_dir="results"):
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
            
            best_scores = []
            
            for img in images:
                answer_scores = {}
                
                for answer in self.possible_answers:
                    text = f"{question} {answer}"
                    
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
                
                best_answer = max(answer_scores.items(), key=lambda x: x[1])[0]
                best_scores.append((best_answer, answer_scores[best_answer]))
            
            best_prediction = max(best_scores, key=lambda x: x[1])[0]
            predictions.append(best_prediction)
        
        return predictions


class BiomedicalLLaMAEvaluator(ModelEvaluator):
    """Evaluator for Biomedical LLaMA model"""
    
    def __init__(self, model_path="/home/mohanganesh/VQAhonors/models/biomedical_llama", save_dir="results"):
        super().__init__("BiomedicalLLaMA", save_dir)
        
        logger.info(f"Loading Biomedical LLaMA model from {model_path}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model.to(self.device)
            logger.info("Biomedical LLaMA model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading Biomedical LLaMA model: {e}")
            raise
    
    def preprocess_batch(self, batch):
        processed_batch = {
            "questions": batch["question"],
            "images": batch["images"],
            "answers": batch["answer"]
        }
        return processed_batch
    
    def predict_batch(self, batch):
        predictions = []
        
        for i in range(len(batch["questions"])):
            question = batch["questions"][i]
            
            # Format prompt for medical question
            prompt = f"You are a medical expert. Answer the following question: {question}\nAnswer:"
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    num_beams=3,
                    early_stopping=True
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract answer from response (remove the prompt)
            answer = response.split("Answer:")[1].strip() if "Answer:" in response else response.strip()
            predictions.append(answer)
        
        return predictions


class LLaVAMedEvaluator(ModelEvaluator):
    """Evaluator for LLaVA-Med model"""
    
    def __init__(self, model_path="/home/mohanganesh/VQAhonors/models/llava_med", save_dir="results"):
        super().__init__("LLaVA-Med", save_dir)
        
        logger.info(f"Loading LLaVA-Med model from {model_path}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            self.model = AutoModelForVision2Seq.from_pretrained(model_path)
            self.processor = AutoProcessor.from_pretrained(model_path)
            self.model.to(self.device)
            logger.info("LLaVA-Med model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading LLaVA-Med model: {e}")
            raise
    
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
            
            # For multi-image input, we'll process each image and combine results
            all_responses = []
            
            for img in images:
                prompt = f"<image>\n{question}"
                
                inputs = self.processor(
                    text=prompt,
                    images=img,
                    return_tensors="pt"
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=100,
                        do_sample=False
                    )
                
                response = self.processor.decode(outputs[0], skip_special_tokens=True)
                all_responses.append(response)
            
            # Combine responses from multiple images
            if len(all_responses) > 1:
                # Simple approach: take the most common answer
                from collections import Counter
                prediction = Counter(all_responses).most_common(1)[0][0]
            else:
                prediction = all_responses[0]
            
            predictions.append(prediction)
        
        return predictions


class MedGemmaEvaluator(ModelEvaluator):
    """Evaluator for MedGemma model"""
    
    def __init__(self, model_path="/home/mohanganesh/VQAhonors/models/medgemma", save_dir="results"):
        super().__init__("MedGemma", save_dir)
        
        logger.info(f"Loading MedGemma model from {model_path}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model.to(self.device)
            logger.info("MedGemma model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading MedGemma model: {e}")
            raise
    
    def preprocess_batch(self, batch):
        processed_batch = {
            "questions": batch["question"],
            "answers": batch["answer"]
        }
        return processed_batch
    
    def predict_batch(self, batch):
        predictions = []
        
        for i in range(len(batch["questions"])):
            question = batch["questions"][i]
            
            # Format prompt for medical question
            prompt = f"<|im_start|>user\nYou are a medical expert. Answer the following question: {question}<|im_end|>\n<|im_start|>assistant\n"
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    num_beams=3,
                    early_stopping=True
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract answer from response
            answer = response.split("<|im_start|>assistant\n")[-1].strip()
            predictions.append(answer)
        
        return predictions


class PMCVQAEvaluator(ModelEvaluator):
    """Evaluator for PMC-VQA model"""
    
    def __init__(self, model_path="/home/mohanganesh/VQAhonors/models/pmc_vqa", save_dir="results"):
        super().__init__("PMC-VQA", save_dir)
        
        logger.info(f"Loading PMC-VQA model from {model_path}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            # PMC-VQA is an adapter model, so we need to load the base model first
            # For this example, we'll assume it's based on LLaVA
            self.model = LlavaForConditionalGeneration.from_pretrained(
                "llava-hf/llava-1.5-7b-hf",
                device_map="auto"
            )
            # Load the adapter weights
            self.model.load_adapter(model_path)
            self.processor = LlavaProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
            logger.info("PMC-VQA model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading PMC-VQA model: {e}")
            raise
    
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
            
            # For multi-image input, we'll process each image and combine results
            all_responses = []
            
            for img in images:
                prompt = f"<image>\nUser: {question}\nAssistant:"
                
                inputs = self.processor(
                    text=prompt,
                    images=img,
                    return_tensors="pt"
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=100,
                        do_sample=False
                    )
                
                response = self.processor.decode(outputs[0], skip_special_tokens=True)
                # Extract answer from response
                answer = response.split("Assistant:")[-1].strip()
                all_responses.append(answer)
            
            # Combine responses from multiple images
            if len(all_responses) > 1:
                # Simple approach: take the most common answer
                from collections import Counter
                prediction = Counter(all_responses).most_common(1)[0][0]
            else:
                prediction = all_responses[0]
            
            predictions.append(prediction)
        
        return predictions


class Qwen25VLEvaluator(ModelEvaluator):
    """Evaluator for Qwen2.5-VL model"""
    
    def __init__(self, model_path="/home/mohanganesh/VQAhonors/models/qwen25_vl", save_dir="results"):
        super().__init__("Qwen2.5-VL", save_dir)
        
        logger.info(f"Loading Qwen2.5-VL model from {model_path}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            self.model = AutoModelForVision2Seq.from_pretrained(model_path)
            self.processor = AutoProcessor.from_pretrained(model_path)
            self.model.to(self.device)
            logger.info("Qwen2.5-VL model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading Qwen2.5-VL model: {e}")
            raise
    
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
            
            # For multi-image input, we'll process each image and combine results
            all_responses = []
            
            for img in images:
                prompt = f"<image>\n{question}"
                
                inputs = self.processor(
                    text=prompt,
                    images=img,
                    return_tensors="pt"
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=100,
                        do_sample=False
                    )
                
                response = self.processor.decode(outputs[0], skip_special_tokens=True)
                all_responses.append(response)
            
            # Combine responses from multiple images
            if len(all_responses) > 1:
                # Simple approach: take the most common answer
                from collections import Counter
                prediction = Counter(all_responses).most_common(1)[0][0]
            else:
                prediction = all_responses[0]
            
            predictions.append(prediction)
        
        return predictions