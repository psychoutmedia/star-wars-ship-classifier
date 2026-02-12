import gradio as gr
from fastai.vision.all import *

# Load the model
learn = load_learner('starwars_model.pkl')

categories = ('tie fighter star wars', 'x-wing starfighter')

def classify_image(img):
    # Ensure image is in RGB mode (Fastai expects 3 channels)
    img = PILImage.create(img).convert("RGB")
    
    # Run prediction - we use float() to ensure it's a simple number
    pred, idx, probs = learn.predict(img)
    return dict(zip(categories, map(float, probs)))

# Use the 'image' and 'label' shortcuts for maximum stability
image_input = gr.Image()
label_output = gr.Label()

intf = gr.Interface(fn=classify_image, inputs=image_input, outputs=label_output)

# CRITICAL: .queue() prevents the "Infinite Spin" by handling requests one-by-one
intf.queue().launch()