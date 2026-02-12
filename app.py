import gradio as gr
from fastai.vision.all import *

# 1. Load the model
learn = load_learner('starwars_model.pkl')

categories = ('tie fighter star wars', 'x-wing starfighter')

def classify_image(img):
    import traceback
    try:
        print(f"Called with type: {type(img)}", flush=True)
        img = PILImage.create(img)
        print("PILImage created", flush=True)
        pred, idx, probs = learn.predict(img)
        print(f"Prediction: {pred}", flush=True)
        return dict(zip(categories, map(float, probs)))
    except Exception as e:
        traceback.print_exc()
        raise

# 2. Build the interface
demo = gr.Interface(
    fn=classify_image, 
    inputs=gr.Image(type="pil"),
    outputs=gr.Label()
)

# 3. Launch at module level for HF Spaces (Gradio 5 SSR imports the module)
demo.launch(ssr_mode=False)