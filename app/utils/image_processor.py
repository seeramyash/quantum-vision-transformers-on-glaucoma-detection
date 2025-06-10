import os
from PIL import Image
from torchvision import transforms
from backend.app.config import RESIZE_DIR

def load_image_from_row(row, use_path=True):
    if use_path:
        image_path = row['ImagePath'].strip()
        if os.path.isabs(image_path):
            final_path = image_path
        else:
            final_path = os.path.join(RESIZE_DIR, image_path)
    else:
        image_filename = row['ImageFilename'].strip()
        final_path = os.path.join(RESIZE_DIR, image_filename)
    if not os.path.exists(final_path):
        raise FileNotFoundError(f"Image not found: {final_path}")
    return Image.open(final_path)

def preprocess_image(img):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform(img).unsqueeze(0)  # Add batch dimension