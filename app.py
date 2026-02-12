import os
import time
import torch
import gradio as gr
from fastai.vision.all import *
import fastai  # for torch_core.defaults

# Force single-threaded / stable CPU behavior
torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.backends.cudnn.benchmark = False

print("Loading model...")
start_load = time.time()
learn = load_learner("starwars_model.pkl", cpu=True)

# === THE SURGICAL CPU FIX ===
fastai.torch_core.defaults.device = 'cpu'  # global tensor default
learn.model.cpu()                           # model explicitly on CPU
learn.dls.cpu()                             # dataloaders explicitly on CPU
# (num_workers=0 already below, but reinforced)

print("Model configured for stable CPU inference")

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
        
        print("Resizing image to 224x224...")
        resize_start = time.time()
        img = Resize(224)(img)  # adjust to your model's training size if not 224
        resize_time = time.time() - resize_start
        print(f"Image resized in {resize_time:.2f}s")
        
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
        
        return {str(c): float(p) for c, p in zip(classes, probs)}
    
    except Exception as e:
        error_time = time.time() - overall_start
        print(f"ERROR during classification (after {error_time:.2f}s): {str(e)}")
        import traceback
        traceback.print_exc()
        raise

# Interface
demo = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="filepath"),
    outputs=gr.Label(num_top_classes=3),
    title="Star Wars Ship Classifier",
    description="Upload a Star Wars ship image (resized to 224×224 for faster CPU inference).",
)

demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.getenv("PORT", "7860")),
    debug=True,
    show_error=True,
)