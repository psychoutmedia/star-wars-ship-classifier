import gradio as gr
from fastai.vision.all import *

# Load the brain 
learn = load_learner('starwars_model.pkl')

categories = ('tie fighter star wars', 'x-wing starfighter')

def classify_image(img):
    # handles .webpformat fix
    img = PILImage.create(img).convert("RGB")
    pred, idx, probs = learn.predict(img)
    return dict(zip(categories, map(float, probs)))

image = gr.Image(type="pil")
label = gr.Label()


intf = gr.Interface(fn=classify_image, inputs=image, outputs=label)
intf.launch()