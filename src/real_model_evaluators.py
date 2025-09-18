import os
import torch
import logging
import numpy as np
from PIL import Image
from transformers import (
    CLIPProcessor, CLIPModel,
    AutoModelForCausalLM, AutoProcessor, AutoTokenizer,
    AutoModelForVision2Seq, LlavaForConditionalGeneration, LlavaProcessor
)
from tqdm import tqdm
import time

logger = logging.getLogger(__name__)

class RealModelEvaluator:
    """Base class for real model evaluators"""
    
    def __init__(self, model_name, model_path, device=None):
        self.model_name = model_name
        self.model_path = model_path
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.processor = None
        self.tokenizer = None
        
    def load_model(self):
        """Load the model and processor"""
        raise NotImplementedError("Subclasses must implement this method")
    
    def predict(self, images, question, options):
        """Generate prediction for a single sample"""
        raise NotImplementedError("Subclasses must implement this method")
    
    def predict_batch(self, batch):
        """Generate predictions for a batch of samples"""
        predictions = []
        
        # Handle different batch structures
        if isinstance(batch['images'], list):
            # List of images (from custom collate)
            for i in range(len(batch['images'])):
                images = batch['images'][i]
                question = batch['question'][i]
                options = batch['options'][i]
                
                pred = self.predict(images, question, options)
                predictions.append(pred)
        else:
            # Tensor batch
            batch_size = batch['images'].shape[0]
            for i in range(batch_size):
                images = batch['images'][i]
                question = batch['question'][i]
                options = batch['options'][i]
                
                pred = self.predict(images, question, options)
                predictions.append(pred)
        
        return predictions

class RealBiomedCLIPEvaluator(RealModelEvaluator):
    """Real BiomedCLIP evaluator"""
    
    def load_model(self):
        logger.info(f"Loading BiomedCLIP model from {self.model_path}")
        try:
            self.model = CLIPModel.from_pretrained(self.model_path)
            self.processor = CLIPProcessor.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            logger.info("BiomedCLIP model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading BiomedCLIP model: {e}")
            # Fallback to base CLIP model
            logger.info("Falling back to base CLIP model")
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.model.to(self.device)
            self.model.eval()
    
    def predict(self, images, question, options):
        """Generate prediction using BiomedCLIP"""
        try:
            # Handle multiple images - use first image for CLIP models
            if isinstance(images, torch.Tensor) and images.dim() == 4:
                # Multiple images, use the first one
                image = images[0]
            else:
                image = images
            
            # Convert tensor to PIL if needed
            if isinstance(image, torch.Tensor):
                image = self._tensor_to_pil(image)
            
            # Score each option
            option_scores = {}
            for option_key, option_text in options.items():
                text = f"{question} {option_text}"
                
                inputs = self.processor(
                    text=text,
                    images=image,
                    return_tensors="pt",
                    padding=True
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits_per_image = outputs.logits_per_image
                    score = logits_per_image.item()
                
                option_scores[option_key] = score
            
            # Return the option with highest score
            best_option = max(option_scores.items(), key=lambda x: x[1])[0]
            return best_option
            
        except Exception as e:
            logger.error(f"Error in BiomedCLIP prediction: {e}")
            # Return first option as fallback
            return list(options.keys())[0] if options else "A"
    
    def _tensor_to_pil(self, tensor):
        """Convert tensor to PIL Image"""
        # Denormalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = tensor * std + mean
        tensor = torch.clamp(tensor, 0, 1)
        
        # Convert to PIL
        tensor = tensor.permute(1, 2, 0).cpu()
        array = (tensor.numpy() * 255).astype(np.uint8)
        return Image.fromarray(array)

class RealBiomedicalLLaMAEvaluator(RealModelEvaluator):
    """Real Biomedical-LLaMA evaluator"""
    
    def load_model(self):
        logger.info(f"Loading Biomedical-LLaMA model from {self.model_path}")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            logger.info("Biomedical-LLaMA model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading Biomedical-LLaMA model: {e}")
            # Fallback to a smaller model
            logger.info("Falling back to smaller model")
            self.model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
            self.model.to(self.device)
    
    def predict(self, images, question, options):
        """Generate prediction using Biomedical-LLaMA"""
        try:
            # Format the question with options
            prompt = f"Question: {question}\n\nOptions:\n"
            for option_key, option_text in options.items():
                prompt += f"{option_key}. {option_text}\n"
            prompt += "\nAnswer with the letter of the correct option:"
            
            # Tokenize and generate
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=10,
                    num_return_sequences=1,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode the output
            output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract the answer option
            for option_key in options.keys():
                if option_key in output_text[-20:]:  # Check last 20 chars
                    return option_key
            
            # If no option found, return first option
            return list(options.keys())[0] if options else "A"
            
        except Exception as e:
            logger.error(f"Error in Biomedical-LLaMA prediction: {e}")
            return list(options.keys())[0] if options else "A"

class RealLLaVAMedEvaluator(RealModelEvaluator):
    """Real LLaVA-Med evaluator"""
    
    def load_model(self):
        logger.info(f"Loading LLaVA-Med model from {self.model_path}")
        try:
            self.model = LlavaForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self.processor = LlavaProcessor.from_pretrained(self.model_path)
            logger.info("LLaVA-Med model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading LLaVA-Med model: {e}")
            # Fallback to base LLaVA
            logger.info("Falling back to base LLaVA model")
            self.model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
            self.processor = LlavaProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
            self.model.to(self.device)
    
    def predict(self, images, question, options):
        """Generate prediction using LLaVA-Med"""
        try:
            # Handle multiple images - use first image
            if isinstance(images, torch.Tensor) and images.dim() == 4:
                image = images[0]
            else:
                image = images
            
            # Convert tensor to PIL if needed
            if isinstance(image, torch.Tensor):
                image = self._tensor_to_pil(image)
            
            # Format the question with options
            prompt = f"Question: {question}\n\nOptions:\n"
            for option_key, option_text in options.items():
                prompt += f"{option_key}. {option_text}\n"
            prompt += "\nAnswer with the letter of the correct option:"
            
            # Process inputs
            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate answer
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=20,
                    temperature=0.1,
                    do_sample=False
                )
            
            # Decode the output
            output_text = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            # Extract the answer option
            for option_key in options.keys():
                if option_key in output_text[-30:]:  # Check last 30 chars
                    return option_key
            
            # If no option found, return first option
            return list(options.keys())[0] if options else "A"
            
        except Exception as e:
            logger.error(f"Error in LLaVA-Med prediction: {e}")
            return list(options.keys())[0] if options else "A"
    
    def _tensor_to_pil(self, tensor):
        """Convert tensor to PIL Image"""
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = tensor * std + mean
        tensor = torch.clamp(tensor, 0, 1)
        
        tensor = tensor.permute(1, 2, 0).cpu()
        array = (tensor.numpy() * 255).astype(np.uint8)
        return Image.fromarray(array)

class RealMedGemmaEvaluator(RealModelEvaluator):
    """Real MedGemma evaluator"""
    
    def load_model(self):
        logger.info(f"Loading MedGemma model from {self.model_path}")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            logger.info("MedGemma model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading MedGemma model: {e}")
            # Fallback to base Gemma
            logger.info("Falling back to base Gemma model")
            self.model = AutoModelForCausalLM.from_pretrained("google/gemma-2b")
            self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
            self.model.to(self.device)
    
    def predict(self, images, question, options):
        """Generate prediction using MedGemma"""
        try:
            # Format the question with options
            prompt = f"Question: {question}\n\nOptions:\n"
            for option_key, option_text in options.items():
                prompt += f"{option_key}. {option_text}\n"
            prompt += "\nAnswer with the letter of the correct option:"
            
            # Tokenize and generate
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=10,
                    num_return_sequences=1,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode the output
            output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract the answer option
            for option_key in options.keys():
                if option_key in output_text[-20:]:
                    return option_key
            
            return list(options.keys())[0] if options else "A"
            
        except Exception as e:
            logger.error(f"Error in MedGemma prediction: {e}")
            return list(options.keys())[0] if options else "A"

class RealPMCVQAEvaluator(RealModelEvaluator):
    """Real PMC-VQA evaluator"""
    
    def load_model(self):
        logger.info(f"Loading PMC-VQA model from {self.model_path}")
        try:
            # PMC-VQA is typically based on LLaVA
            self.model = LlavaForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self.processor = LlavaProcessor.from_pretrained(self.model_path)
            logger.info("PMC-VQA model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading PMC-VQA model: {e}")
            # Fallback to base LLaVA
            logger.info("Falling back to base LLaVA model")
            self.model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
            self.processor = LlavaProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
            self.model.to(self.device)
    
    def predict(self, images, question, options):
        """Generate prediction using PMC-VQA"""
        try:
            # Handle multiple images - use first image
            if isinstance(images, torch.Tensor) and images.dim() == 4:
                image = images[0]
            else:
                image = images
            
            # Convert tensor to PIL if needed
            if isinstance(image, torch.Tensor):
                image = self._tensor_to_pil(image)
            
            # Format the question with options
            prompt = f"Question: {question}\n\nOptions:\n"
            for option_key, option_text in options.items():
                prompt += f"{option_key}. {option_text}\n"
            prompt += "\nAnswer with the letter of the correct option:"
            
            # Process inputs
            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate answer
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=20,
                    temperature=0.1,
                    do_sample=False
                )
            
            # Decode the output
            output_text = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            # Extract the answer option
            for option_key in options.keys():
                if option_key in output_text[-30:]:
                    return option_key
            
            return list(options.keys())[0] if options else "A"
            
        except Exception as e:
            logger.error(f"Error in PMC-VQA prediction: {e}")
            return list(options.keys())[0] if options else "A"
    
    def _tensor_to_pil(self, tensor):
        """Convert tensor to PIL Image"""
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = tensor * std + mean
        tensor = torch.clamp(tensor, 0, 1)
        
        tensor = tensor.permute(1, 2, 0).cpu()
        array = (tensor.numpy() * 255).astype(np.uint8)
        return Image.fromarray(array)

class RealQwen25VLEvaluator(RealModelEvaluator):
    """Real Qwen2.5-VL evaluator"""
    
    def load_model(self):
        logger.info(f"Loading Qwen2.5-VL model from {self.model_path}")
        try:
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self.processor = AutoProcessor.from_pretrained(self.model_path)
            logger.info("Qwen2.5-VL model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading Qwen2.5-VL model: {e}")
            # Fallback to base Qwen-VL
            logger.info("Falling back to base Qwen-VL model")
            self.model = AutoModelForVision2Seq.from_pretrained("Qwen/Qwen-VL")
            self.processor = AutoProcessor.from_pretrained("Qwen/Qwen-VL")
            self.model.to(self.device)
    
    def predict(self, images, question, options):
        """Generate prediction using Qwen2.5-VL"""
        try:
            # Handle multiple images - use first image
            if isinstance(images, torch.Tensor) and images.dim() == 4:
                image = images[0]
            else:
                image = images
            
            # Convert tensor to PIL if needed
            if isinstance(image, torch.Tensor):
                image = self._tensor_to_pil(image)
            
            # Format the question with options
            prompt = f"Question: {question}\n\nOptions:\n"
            for option_key, option_text in options.items():
                prompt += f"{option_key}. {option_text}\n"
            prompt += "\nAnswer with the letter of the correct option:"
            
            # Process inputs
            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate answer
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=20,
                    temperature=0.1,
                    do_sample=False
                )
            
            # Decode the output
            output_text = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            # Extract the answer option
            for option_key in options.keys():
                if option_key in output_text[-30:]:
                    return option_key
            
            return list(options.keys())[0] if options else "A"
            
        except Exception as e:
            logger.error(f"Error in Qwen2.5-VL prediction: {e}")
            return list(options.keys())[0] if options else "A"
    
    def _tensor_to_pil(self, tensor):
        """Convert tensor to PIL Image"""
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = tensor * std + mean
        tensor = torch.clamp(tensor, 0, 1)
        
        tensor = tensor.permute(1, 2, 0).cpu()
        array = (tensor.numpy() * 255).astype(np.uint8)
        return Image.fromarray(array)

# Model factory function
def get_real_model_evaluator(model_name, model_path, device=None):
    """Factory function to get the appropriate model evaluator"""
    evaluators = {
        'BiomedCLIP': RealBiomedCLIPEvaluator,
        'Biomedical-LLaMA': RealBiomedicalLLaMAEvaluator,
        'LLaVA-Med': RealLLaVAMedEvaluator,
        'MedGemma': RealMedGemmaEvaluator,
        'PMC-VQA': RealPMCVQAEvaluator,
        'Qwen2.5-VL': RealQwen25VLEvaluator
    }
    
    if model_name not in evaluators:
        raise ValueError(f"Unknown model: {model_name}")
    
    return evaluators[model_name](model_name, model_path, device)
