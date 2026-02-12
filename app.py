import os
import torch
import gradio as gr
from fastai.vision.all import *

# CPU Basic stability
torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

learn = load_learner("starwars_model.pkl", cpu=True)
learn.dls.num_workers = 0
categories = learn.dls.vocab

def classify_image(image_path: str):
    img = PILImage.create(image_path)

    # Avoid predict() hanging: make a test dataloader with num_workers=0 explicitly
    dl = learn.dls.test_dl([img], num_workers=0)
    preds, _ = learn.get_preds(dl=dl)
    probs = preds[0]

    return {categories[i]: float(probs[i]) for i in range(len(categories))}

demo = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="filepath"),
    outputs=gr.Label(num_top_classes=3),
    title="Star Wars Ship Classifier",
    concurrency_limit=1,  # valid in Gradio 5 Interface
)

demo.queue(max_size=20, default_concurrency_limit=1)

demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.getenv("PORT", "7860")),
)
