import os
from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image
import glob
import torchvision.transforms as transforms

class MOBIUSDataset(Dataset):
    """Dataset class for MOBIUS data with numbered folders."""
    
    def __init__(self, data_dir, training=True, transform=None):
        """
        Args:
            data_dir (str): Root directory of MOBIUS dataset
            training (bool): Whether this is training data
            transform (callable, optional): Transform to be applied to images
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.training = training
        
        # Get all numbered folders
        self.folder_numbers = []
        for i in range(1, 36):  # 1 to 35
            if (self.data_dir / "Images" / str(i)).exists():
                self.folder_numbers.append(str(i))
        
        # Collect all image paths and corresponding mask paths
        self.img_paths = []
        self.mask_paths = []
        
        for folder_num in self.folder_numbers:
            img_folder = self.data_dir / "Images" / folder_num
            mask_folder = self.data_dir / "Masks" / folder_num
            
            # Get all image files in this folder
            img_files = sorted(glob.glob(str(img_folder / "*")))
            mask_files = sorted(glob.glob(str(mask_folder / "*")))
            
            # Verify matching pairs
            for img_path, mask_path in zip(img_files, mask_files):
                img_name = Path(img_path).stem
                mask_name = Path(mask_path).stem
                if img_name == mask_name:  # Only add if names match
                    self.img_paths.append(img_path)
                    self.mask_paths.append(mask_path)
        
        if len(self.img_paths) == 0:
            raise RuntimeError(f"No images found in {data_dir}")
            
        print(f"Found {len(self.img_paths)} images in {len(self.folder_numbers)} folders")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        
        # Convert to tensor first
        to_tensor = transforms.ToTensor()
        image = to_tensor(image)
        mask = torch.from_numpy(np.array(mask))
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
            # Make sure mask is also resized to match image size if resize is part of transforms
            if any(isinstance(t, transforms.Resize) for t in self.transform.transforms):
                resize = next(t for t in self.transform.transforms if isinstance(t, transforms.Resize))
                mask = transforms.Resize(resize.size, interpolation=transforms.InterpolationMode.NEAREST)(mask.unsqueeze(0)).squeeze(0)
        
        return image, mask