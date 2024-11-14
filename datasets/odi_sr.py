import os

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

from datasets import register

IMAGE_EXTS = ('.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG', '.webp')

@register('image_folder')
class ImageFolder(Dataset):

    def __init__(self, root_path):
        files = sorted(os.listdir(root_path))
        files = [os.path.join(root_path, _) for _ in files if _.endswith(IMAGE_EXTS)]
        self.files = files
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')
        return img