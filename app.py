import torch
import gradio as gr
import numpy as np
import clip
from torchvision.datasets import CIFAR100
import os
from config import IMAGE_FEATURES_PATH, TEXT_FEATURES_PATH

# Load precomputed features
image_features = torch.load(IMAGE_FEATURES_PATH)
text_features = torch.load(TEXT_FEATURES_PATH)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Load CIFAR-100 dataset
cifar100 = CIFAR100(root="./data", train=False, download=False, transform=preprocess)

def find_relevant_images(prompt):
    # Tokenize the prompt
    text_inputs = clip.tokenize([prompt]).to(device)
    
    # Compute the features of the text prompt
    with torch.no_grad():
        prompt_features = model.encode_text(text_inputs).float()
        prompt_features /= prompt_features.norm(dim=-1, keepdim=True)
    
    # Compute the similarity between the prompt and the text features
    text_similarities = torch.nn.functional.cosine_similarity(prompt_features, text_features, dim=-1)
    
    # Find the most similar class indices
    top_text_indices = text_similarities.topk(20, largest=True).indices
    
    # Compute the similarity between image features and the most similar text features
    image_similarities = text_features[top_text_indices] @ image_features.T
    
    # Get the indices of the most similar images
    top_image_indices = image_similarities.topk(5, largest=True).indices
    
    top_image_indices = top_image_indices[0]
    # Retrieve the images based on the indices
    images = [cifar100[int(idx)][0].permute(1, 2, 0).numpy() for idx in top_image_indices]
    
    images_normalized = [((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8) for image in images]
    
    return images_normalized

# Create Gradio interface
iface = gr.Interface(
    fn=find_relevant_images,
    inputs=gr.Textbox(lines=1, label="Enter your prompt"),
    outputs=[gr.Image(type="numpy", label=f"Image {i+1}") for i in range(5)],
    examples=[
        ["This is a photo of a dog"],
        ["This is a photo of a flower"],
        ["This is a photo of an car"],
        ["This is a photo of a truck"]
    ]
)

# Launch the interface
iface.launch(share = True)
