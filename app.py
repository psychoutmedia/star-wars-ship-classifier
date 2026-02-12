import os
import torch
import gradio as gr
from fastai.vision.all import *

# Keep CPU Basic stable (prevents thread thrash / weird hangs)
torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

MODEL_PATH = "starwars_model.pkl"

# Load the exported fastai learner on CPU
learn = load_learner(MODEL_PATH, cpu=True)
categories = learn.dls.vocab

# If vocab is nested (sometimes happens), flatten it
if isinstance(categories, (list, tuple)) and len(categories) == 1 and isinstance(categories[0], (list, tuple)):
    categories = categories[0]

def classify_image(image_path: str):
    # Use filepath input for maximum stability on Spaces
    img = PILImage.create(image_path)

    # KEY: build a test dataloader with num_workers=0 explicitly
    dl = learn.dls.test_dl([img], num_workers=0)

    preds, _ = learn.get_preds(dl=dl)
    probs = preds[0]

    return {categories[i]: float(probs[i]) for i in range(len(categories))}

demo = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="filepath", label="Upload an image"),
    outputs=gr.Label(num_top_classes=3),
    title="Star Wars Ship Classifier",
)

# Queue prevents request handling edge-cases, and single concurrency suits CPU Basic
demo.queue(concurrency_count=1, max_size=20)

demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.getenv("PORT", "7860")),
)
