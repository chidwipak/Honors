import os
import pandas as pd
import numpy as np
from PIL import Image
import io
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms

class RealMedFrameQADataset(Dataset):
    """
    Real MedFrameQA dataset loader that properly handles the actual data structure
    """
    def __init__(self, data_dir, transform=None, max_samples=None):
        """
        Initialize the dataset
        
        Args:
            data_dir (str): Directory containing the dataset files
            transform (callable, optional): Optional transform to be applied on images
            max_samples (int, optional): Maximum number of samples to load (for testing)
        """
        self.data_dir = data_dir
        self.transform = transform
        
        # Load all parquet files and concatenate them
        parquet_files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]
        parquet_files.sort()  # Ensure consistent ordering
        
        dfs = []
        print(f"Loading {len(parquet_files)} parquet files from {data_dir}")
        for file in tqdm(parquet_files):
            file_path = os.path.join(data_dir, file)
            df = pd.read_parquet(file_path)
            dfs.append(df)
        
        self.data = pd.concat(dfs, ignore_index=True)
        
        # Limit samples if specified
        if max_samples is not None and max_samples < len(self.data):
            self.data = self.data.sample(max_samples, random_state=42)
            
        print(f"Loaded {len(self.data)} samples")
        
        # Set up default transform if none provided
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            dict: A dictionary containing the sample data
        """
        item = self.data.iloc[idx]
        
        # Extract images from image_1 to image_5 columns
        images = []
        for i in range(1, 6):  # image_1 to image_5
            img_col = f'image_{i}'
            if img_col in item and pd.notna(item[img_col]) and item[img_col] is not None:
                try:
                    # Image data is stored as dict with 'bytes' key
                    img_dict = item[img_col]
                    if isinstance(img_dict, dict) and 'bytes' in img_dict:
                        img_bytes = img_dict['bytes']
                        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                        
                        # Apply transform
                        if self.transform:
                            img = self.transform(img)
                        images.append(img)
                except Exception as e:
                    print(f"Error loading image from {img_col}: {e}")
                    continue
        
        # If no images were loaded, create a placeholder
        if not images:
            placeholder = Image.new('RGB', (224, 224), color='white')
            if self.transform:
                placeholder = self.transform(placeholder)
            images.append(placeholder)
        
        # Extract question and correct answer
        question = item['question'] if 'question' in item else ""
        correct_answer_letter = item['correct_answer'] if 'correct_answer' in item else ""
        
        # Extract options and convert to dictionary
        options = {}
        if 'options' in item and item['options'] is not None:
            try:
                options_list = item['options'].tolist() if isinstance(item['options'], np.ndarray) else item['options']
                for i, option_text in enumerate(options_list):
                    options[chr(65 + i)] = option_text  # A, B, C, D, E, F
            except Exception as e:
                print(f"Error processing options: {e}")
                options = {}
        
        # Get the correct answer text
        correct_answer_text = options.get(correct_answer_letter, "")
        
        # Extract metadata for research analysis
        metadata = {
            'question_id': item.get('question_id', f'q{idx}'),
            'body_system': item.get('system', 'unknown'),
            'modality': item.get('modality', 'unknown'),
            'organ': item.get('organ', 'unknown'),
            'reasoning_type': item.get('reasoning_chain', 'unknown'),
            'image_count': len(images),
            'video_id': item.get('video_id', ''),
            'keyword': item.get('keyword', '')
        }
        
        # Ensure all images are the same size before stacking
        if len(images) > 1:
            # All images should be the same size due to transform, but let's be safe
            try:
                images = torch.stack(images)
            except RuntimeError as e:
                print(f"Error stacking images: {e}")
                # If stacking fails, just take the first image
                images = images[0]
        else:
            images = images[0]
            
        return {
            'images': images,
            'question': question,
            'answer': correct_answer_letter,  # The letter (A, B, C, D, E, F)
            'answer_text': correct_answer_text,  # The full text of the correct answer
            'options': options,
            'metadata': metadata
        }

def custom_collate_fn(batch):
    """
    Custom collate function to handle different data types in the batch
    """
    # Separate different types of data
    images = []
    questions = []
    answers = []
    answer_texts = []
    options = []
    metadata = []
    
    for item in batch:
        images.append(item['images'])
        questions.append(item['question'])
        answers.append(item['answer'])
        answer_texts.append(item['answer_text'])
        options.append(item['options'])
        metadata.append(item['metadata'])
    
    # Stack images if they're tensors
    if len(images) > 0 and isinstance(images[0], torch.Tensor):
        try:
            images = torch.stack(images)
        except RuntimeError:
            # If stacking fails, create a list instead
            pass
    
    return {
        'images': images,
        'question': questions,
        'answer': answers,
        'answer_text': answer_texts,
        'options': options,
        'metadata': metadata
    }

def load_real_medframeqa_dataset(data_dir, transform=None, batch_size=8, max_samples=None, shuffle=True, num_workers=0):
    """
    Load the real MedFrameQA dataset
    
    Args:
        data_dir (str): Directory containing the dataset files
        transform (callable, optional): Optional transform to be applied on images
        batch_size (int): Batch size for the dataloader
        max_samples (int, optional): Maximum number of samples to load (for testing)
        shuffle (bool): Whether to shuffle the dataset
        num_workers (int): Number of workers for data loading (set to 0 to avoid multiprocessing issues)
        
    Returns:
        DataLoader: DataLoader for the dataset
    """
    dataset = RealMedFrameQADataset(data_dir, transform, max_samples)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=custom_collate_fn
    )
    return dataloader, dataset

def visualize_sample(sample):
    """
    Visualize a sample from the dataset
    
    Args:
        sample (dict): A sample from the dataset
    """
    images = sample['images']
    question = sample['question']
    answer = sample['answer']
    answer_text = sample['answer_text']
    options = sample['options']
    
    # Handle single image case
    if images.dim() == 3:  # Single image
        images = [images]
    
    n_images = len(images)
    fig, axes = plt.subplots(1, n_images, figsize=(5*n_images, 5))
    
    if n_images == 1:
        axes = [axes]
    
    for i, img in enumerate(images):
        if isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0).numpy()
            img = (img - img.min()) / (img.max() - img.min())
        axes[i].imshow(img)
        axes[i].set_title(f"Image {i+1}")
        axes[i].axis('off')
    
    plt.suptitle(f"Q: {question}\nA: {answer} - {answer_text}")
    plt.tight_layout()
    plt.show()
    
    print(f"Options: {options}")

if __name__ == "__main__":
    # Test the dataset loader
    data_dir = "/home/mohanganesh/VQAhonors/data/MedFrameQA/data"
    dataloader, dataset = load_real_medframeqa_dataset(data_dir, max_samples=5)
    
    # Test a sample
    for batch in dataloader:
        sample = {
            'images': batch['images'][0],
            'question': batch['question'][0],
            'answer': batch['answer'][0],
            'answer_text': batch['answer_text'][0],
            'options': batch['options'][0],
            'metadata': batch['metadata'][0]
        }
        print("Sample loaded successfully:")
        print(f"Question: {sample['question']}")
        print(f"Answer: {sample['answer']} - {sample['answer_text']}")
        print(f"Options: {sample['options']}")
        print(f"Metadata: {sample['metadata']}")
        print(f"Images shape: {sample['images'].shape}")
        break
