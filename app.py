import os
import time
import torch
import gradio as gr
from fastai.vision.all import *

# Force single-threaded / stable CPU behavior
torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.backends.cudnn.benchmark = False  # avoid any unexpected behavior

print("Loading model...")
start_load = time.time()
learn = load_learner("starwars_model.pkl", cpu=True)
load_time = time.time() - start_load
print(f"Model loaded in {load_time:.2f} seconds")

learn.dls.num_workers = 0
categories = learn.dls.vocab
print(f"Categories: {categories}")

def classify_image(image_path: str):
    print(f"Starting classification: {image_path}")
    overall_start = time.time()
    
    try:
        print("Loading image...")
        img_start = time.time()
        img = PILImage.create(image_path)
        img_load_time = time.time() - img_start
        print(f"Image loaded in {img_load_time:.2f}s")
        
        print("Running prediction...")
        pred_start = time.time()
        with torch.no_grad():
            pred_class, probs, classes = learn.predict(img)
        pred_time = time.time() - pred_start
        print(f"Prediction completed in {pred_time:.2f}s")
        print(f"Predicted class: {pred_class}")
        print(f"Probabilities: {probs.tolist()}")
        
        overall_time = time.time() - overall_start
        print(f"Total classification time: {overall_time:.2f}s")
        
        # Return dictionary for gr.Label
        return {str(c): float(p) for c, p in zip(classes, probs)}
    
    except Exception as e:
        error_time = time.time() - overall_start
        print(f"ERROR during classification (after {error_time:.2f}s): {str(e)}")
        import traceback
        traceback.print_exc()
        raise  # Let Gradio show the error in UI if possible

# Create interface
demo = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="filepath"),
    outputs=gr.Label(num_top_classes=3),
    title="Star Wars Ship Classifier",
    description="Upload an image of a Star Wars ship to classify it.",
    # You can add examples=["example1.jpg", "example2.jpg"] if you upload sample images
)

# Launch with debug + show errors
demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.getenv("PORT", "7860")),
    debug=True,          # More verbose logs — very helpful on HF
    show_error=True,     # Shows exceptions in the UI when possible
)