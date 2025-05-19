import torch
from PIL import Image
from transformers import AutoModelForImageClassification, AutoImageProcessor
from scipy.ndimage import gaussian_filter
import numpy as np

# Load NSFW Image Detection Model
nsfw_model = AutoModelForImageClassification.from_pretrained("AimanZaharin/ViT-NSFW-Image-Detection")
processor = AutoImageProcessor.from_pretrained("AimanZaharin/ViT-NSFW-Image-Detection")

def add_blur(image, sigma=60.0):
    image = np.array(image)
    blurred_image = gaussian_filter(image, sigma=sigma)
    return Image.fromarray(np.uint8(blurred_image))

def check_nsfw(image, model, processor):
    with torch.no_grad():
        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_label = logits.argmax(-1).item()
        label = model.config.id2label[predicted_label]
        score = torch.softmax(logits, dim=-1)[0, predicted_label].item()
        return label, score

def generate_image_vit(pipe, prompt, params):
    images = pipe(prompt, **params).images

    for img in images:
        label, _ = check_nsfw(img, nsfw_model, processor)

        if label == "nsfw":
            caution_img = Image.open("assets\ImageBlocked.png")
            blurred_img = add_blur(img, sigma=60.0)

            return caution_img, blurred_img, f"Image classified as UNSAFE"

        else:
             return img, None, f"Image classified as SAFE"