import gradio as gr
from NSFWImageDetection import generate_image_vit
from diffusers import StableDiffusionPipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from PIL import Image

# Load the Stable Diffusion Model
model_id = "dreamlike-art/dreamlike-diffusion-1.0"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.to('cuda')

# Load NSFW Text Detection Model
tokenizer = AutoTokenizer.from_pretrained("AimanZaharin/RoBERTa-based-NSFW-Text-Detection")
text_model = AutoModelForSequenceClassification.from_pretrained("AimanZaharin/RoBERTa-based-NSFW-Text-Detection")

params = {
    'num_inference_steps': 250, 
    'num_images_per_prompt': 1, 
    'negative_prompt': 'worst quality, normal quality, low quality, low res, blurry, distortion, text, watermark, logo, banner, extra digits, cropped, jpeg artifacts, signature, username, error, sketch, duplicate, ugly, monochrome, horror, geometry, mutation, disgusting, bad anatomy, bad proportions, bad quality, deformed, disconnected limbs, out of frame, out of focus, dehydrated, disfigured, extra arms, extra limbs, extra hands, fused fingers, gross proportions, long neck, jpeg, malformed limbs, mutated, mutated hands, mutated limbs, missing arms, missing fingers, picture frame, poorly drawn hands, poorly drawn face, collage, pixel, pixelated, grainy, color aberration, amputee, autograph, bad illustration, beyond the borders, blank background, body out of frame, boring background, branding, cut off, dismembered, disproportioned, distorted, draft, duplicated features, extra fingers, extra legs, fault, flaw, grains, hazy, identifying mark, improper scale, incorrect physiology, incorrect ratio, indistinct, kitsch, low resolution, macabre, malformed, mark, misshapen, missing hands, missing legs, mistake, morbid, mutilated, off-screen, outside the picture, poorly drawn feet, printed words, render, repellent, replicate, reproduce, revolting dimensions, script, shortened, sign, split image, squint, storyboard, tiling, trimmed, unfocused, unattractive, unnatural pose, unreal engine, unsightly, written language'
    }

def process_prompt_and_generate(prompt):
    inputs = tokenizer(prompt, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = text_model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

    labels = ["SAFE", "QUESTIONABLE", "UNSAFE"]
    prediction_index = torch.argmax(probabilities, dim=1).item()
    prediction = labels[prediction_index]
    confidence = probabilities[0, prediction_index].item()

    if prediction == "SAFE":
        image, blurred_img, status = generate_image_vit(pipe, prompt, params)
        return image, blurred_img, status

    else:
        blocked_image = Image.open("assets\PromptBlocked.png")
        return blocked_image, None, f"Prompt classified as {prediction}"

with gr.Blocks(css="""
    body {
        font-family: 'Arial', sans-serif;
        background-color: #121212; 
        color: #ffffff;
    }
    .gr-button {
        border-radius: 8px;
        background-color: #0078ff;
        color: #fff;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .gr-button:hover {
        background-color: #005bb5;
        transform: scale(1.05);
    }
    .gr-image {
        border-radius: 8px;
    }
    h1, p {
        text-align: center;
    }
""") as demo:
    gr.Markdown("<h1>✨ AI Image Generator with NSFW Filtering ✨</h1>")
    gr.Markdown("<p>Enter a prompt below. NSFW content will be blocked and blurred.</p>")

    with gr.Row():
        with gr.Column(scale=1, elem_id="center_column"):
            prompt_input = gr.Textbox(label="Enter your prompt", placeholder="Type your creative prompt here...")
            generate_button = gr.Button("🚀 Generate Image")

    with gr.Row():
        output_image_safe = gr.Image(label="Generated Image", interactive=False, visible=True, height=256)
        output_image_blurred = gr.Image(label="Blurred Image (NSFW)", interactive=False, visible=True, height=256)
        output_text = gr.Textbox(label="Status", interactive=False)

    generate_button.click(
        fn=process_prompt_and_generate,
        inputs=prompt_input,
        outputs=[output_image_safe, output_image_blurred, output_text]
    )

demo.launch(share = True)
