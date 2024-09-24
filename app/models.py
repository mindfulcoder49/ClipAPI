from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import io

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

async def get_image_embedding(file):
    image = Image.open(io.BytesIO(await file.read()))
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_embedding = model.get_image_features(**inputs)
    return image_embedding.tolist()

def get_text_embedding(text):
    inputs = processor(text=[text], return_tensors="pt")
    with torch.no_grad():
        text_embedding = model.get_text_features(**inputs)
    return text_embedding.tolist()
