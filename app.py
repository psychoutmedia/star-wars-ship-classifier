import os
import time
import torch
import gradio as gr
from fastai.vision.all import *

# Force single-threaded / stable CPU behavior
torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.backends.cudnn.benchmark = False

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
        
        print("Resizing image to 224x224...")
        resize_start = time.time()
        img = Resize(224)(img)  # Adjust to 299/384/etc. if your model was trained at different size
        resize_time = time.time() - resize_start
        print(f"Image resized in {resize_time:.2f}s")
        
        print("Applying transforms manually...")
        trans_start = time.time()
        # Apply the same item transforms as would happen in test_dl
        x = learn.dls.after_item(img)
        # Add batch dimension and apply batch transforms (normalization etc.)
        x = x.unsqueeze(0)  # shape: [1, C, H, W]
        x = learn.dls.after_batch(x)
        trans_time = time.time() - trans_start
        print(f"Transforms applied in {trans_time:.2f}s")
        
        print("Running manual forward pass...")
        inf_start = time.time()
        with torch.no_grad():
            # Direct model prediction
            preds = learn.model(x)
            # Get probabilities (assuming standard classification head)
            probs = learn.dls.loss_func.module.activation(preds).squeeze(0)
            pred_idx = probs.argmax().item()
            pred_class = categories[pred_idx]
        inf_time = time.time() - inf_start
        print(f"Manual inference completed in {inf_time:.2f}s")
        print(f"Predicted class: {pred_class}")
        print(f"Probabilities: {probs.tolist()}")
        
        overall_time = time.time() - overall_start
        print(f"Total classification time: {overall_time:.2f}s")
        
        # Return dict for Gradio Label
        return {str(c): float(p) for c, p in zip(categories, probs)}
    
    except Exception as e:
        error_time = time.time() - overall_start
        print(f"ERROR during classification (after {error_time:.2f}s): {str(e)}")
        import traceback
        traceback.print_exc()
        raise  # Let Gradio display the error if possible

# Interface setup
demo = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="filepath"),
    outputs=gr.Label(num_top_classes=3),
    title="Star Wars Ship Classifier",
    description="Classifies Star Wars ships (using manual inference to avoid fastai predict hangs).",
)

demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.getenv("PORT", "7860")),
    debug=True,
    show_error=True,
)