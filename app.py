import gradio as gr
from fastai.vision.all import *

# Load the brain 
learn = load_learner('starwars_model.pkl')

categories = ('tie fighter star wars', 'x-wing starfighter')

def classify_image(img):
    # Ensure it's a PIL Image and in RGB mode
    img = PILImage.create(img).convert("RGB")
    
    # Using 'with learn.no_bar()' can sometimes help with 'dict' errors 
    # as it prevents fastai from trying to 'add' progress bar dictionaries
    with learn.no_bar():
        pred, idx, probs = learn.predict(img)
        
    return dict(zip(categories, map(float, probs)))

image = gr.Image(type="pil")
label = gr.Label()


intf = gr.Interface(fn=classify_image, inputs=image, outputs=label)
# Enable the queue and simplify the launch
intf.launch()