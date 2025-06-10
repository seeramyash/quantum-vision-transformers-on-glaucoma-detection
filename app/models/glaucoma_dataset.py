import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class GlaucomaDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['ImagePath']
        label = self.data.iloc[idx]['Label']
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return image, label