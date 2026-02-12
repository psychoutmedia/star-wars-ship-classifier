import gradio as gr
from fastai.vision.all import *

# 1. Load the model
learn = load_learner('starwars_model.pkl')
learn.dls.num_workers = 0
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
