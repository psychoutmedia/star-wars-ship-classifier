import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
torch.set_num_threads(1)

import gradio as gr
from fastai.vision.all import *
from fastai.callback.progress import ProgressCallback

# 1. Load the model
learn = load_learner('starwars_model.pkl')
learn.remove_cb(ProgressCallback)
categories = learn.dls.vocab

def classify_image(img):
    img = PILImage.create(img)
    pred, idx, probs = learn.predict(img)
    return dict(zip(categories, map(float, probs)))

# 2. Build the interface
demo = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label()
)

# 3. Launch
demo.launch(ssr_mode=False)
