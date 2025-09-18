import os
import pandas as pd
import numpy as np
from PIL import Image
import io
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

class MedFrameQADataset(Dataset):
    """
    Dataset class for MedFrameQA dataset
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
        
        # Extract images
        images = []
        img_cols = [col for col in item.index if col.startswith('image_')]
        
        for img_col in img_cols:
            # Check if the column exists and has valid image data
            try:
                # Handle different types of image data storage in pandas
                if img_col in item:
                    img_data = item[img_col]
                    
                    # Handle different image data formats
                    if isinstance(img_data, np.ndarray):
                        img_bytes = img_data.tobytes()
                        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                    elif isinstance(img_data, bytes):
                        img_bytes = img_data
                        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                    elif isinstance(img_data, dict):
                        # Handle dictionary format with various possible keys
                        if 'bytes' in img_data and isinstance(img_data['bytes'], bytes):
                            img_bytes = img_data['bytes']
                            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                        elif 'data' in img_data and isinstance(img_data['data'], bytes):
                            img_bytes = img_data['data']
                            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                        elif 'image' in img_data and isinstance(img_data['image'], bytes):
                            img_bytes = img_data['image']
                            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                        else:
                            # Try to find any bytes-like key in the dictionary
                            bytes_found = False
                            for key, value in img_data.items():
                                if isinstance(value, bytes):
                                    try:
                                        img = Image.open(io.BytesIO(value)).convert('RGB')
                                        bytes_found = True
                                        break
                                    except:
                                        continue
                            
                            if not bytes_found:
                                # Use placeholder for invalid bytes
                                img = Image.new('RGB', (224, 224), color='white')
                    elif pd.notna(img_data):  # Handle other non-null types
                        # Use placeholder for unsupported types
                        img = Image.new('RGB', (224, 224), color='white')
                    else:
                        continue  # Skip null/NaN values
                    if self.transform:
                        img = self.transform(img)
                    else:
                        # If no transform is provided, create a basic tensor transform
                        from torchvision import transforms
                        transform = transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ])
                        img = transform(img)
                    images.append(img)
            except Exception as e:
                print(f"Error loading image from {img_col}: {e}")
                # Add a placeholder tensor
                from torchvision import transforms
                placeholder = Image.new('RGB', (224, 224), color='white')
                if self.transform:
                    placeholder_tensor = self.transform(placeholder)
                else:
                    transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                    placeholder_tensor = transform(placeholder)
                images.append(placeholder_tensor)
        
        # If no images were loaded, use a blank image tensor
        if not images:
            from torchvision import transforms
            placeholder = Image.new('RGB', (224, 224), color='white')
            if self.transform:
                placeholder_tensor = self.transform(placeholder)
            else:
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                placeholder_tensor = transform(placeholder)
            images.append(placeholder_tensor)
        
        # Extract question, options, and answer
        question = item['question'] if 'question' in item else ""
        answer = item['answer'] if 'answer' in item else ""
        
        # Extract options (A, B, C, D, E, F)
        options = {}
        for opt in ['A', 'B', 'C', 'D', 'E', 'F']:
            option_key = f'option_{opt}'
            if option_key in item:
                options[opt] = item[option_key]
        
        # Extract metadata for research analysis
        metadata = {}
        if 'body_system' in item:
            metadata['body_system'] = item['body_system']
        if 'modality' in item:
            metadata['modality'] = item['modality']
        if 'reasoning_type' in item:
            metadata['reasoning_type'] = item['reasoning_type']
        
        # Count number of images
        metadata['image_count'] = len(images)
        
        # Add question_id if available
        if 'question_id' in item:
            metadata['question_id'] = item['question_id']
        
        # Stack images if there are multiple
        if len(images) > 1:
            images = torch.stack(images)
        else:
            images = images[0]
            
        # Return all data needed for research evaluation
        return {
            'images': images,
            'question': question,
            'answer': answer,
            'options': options,
            'metadata': metadata
        }

def load_medframeqa_dataset(data_dir, transform=None, batch_size=8, max_samples=None, shuffle=True, num_workers=4):
    """
    Load the MedFrameQA dataset
    
    Args:
        data_dir (str): Directory containing the dataset files
        transform (callable, optional): Optional transform to be applied on images
        batch_size (int): Batch size for the dataloader
        max_samples (int, optional): Maximum number of samples to load (for testing)
        shuffle (bool): Whether to shuffle the dataset
        num_workers (int): Number of workers for data loading
        
    Returns:
        DataLoader: DataLoader for the dataset
    """
    dataset = MedFrameQADataset(data_dir, transform, max_samples)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
        pin_memory=True
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
    
    plt.suptitle(f"Q: {question}\nA: {answer}")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Test the dataset loader
    data_dir = "/home/mohanganesh/VQAhonors/data/MedFrameQA/data"
    dataloader = load_medframeqa_dataset(data_dir, max_samples=10)
    
    # Visualize a sample
    for batch in dataloader:
        sample = {
            'images': batch['images'][0],
            'question': batch['question'][0],
            'answer': batch['answer'][0]
        }
        visualize_sample(sample)
        break