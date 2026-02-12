import os
import torch
import gradio as gr
from fastai.vision.all import *

# Force single-threaded CPU behavior (helps avoid deadlocks)
torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
# Extra safety (some torch backends like it)
torch.backends.cudnn.benchmark = False

# Load model once at startup
learn = load_learner("starwars_model.pkl", cpu=True)
learn.dls.num_workers = 0   # redundant but harmless
categories = learn.dls.vocab   # usually list of str

def classify_image(image_path: str):
    print(f"Starting classification: {image_path}")
    try:
        # learn.predict returns: (predicted class, probs tensor, class names)
        pred_class, probs, classes = learn.predict(PILImage.create(image_path))
        print("Prediction done")
        
        # Return dict for gr.Label (keys = class names, values = probs)
        return {str(c): float(p) for c, p in zip(classes, probs)}
    
    except Exception as e:
        print(f"Error in classification: {e}")
        raise  # let Gradio show the error

# Interface without concurrency/queue extras
demo = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="filepath"),
    outputs=gr.Label(num_top_classes=3),
    title="Star Wars Ship Classifier",
    # Optional: add examples if you have sample images uploaded
    # examples=["tie-fighter.jpg", "x-wing.jpg"],
)

# NO .queue() call — this is key on HF CPU
demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.getenv("PORT", "7860")),
    debug=True,          # more verbose logs in HF
    show_error=True,     # helps show exceptions in UI
)