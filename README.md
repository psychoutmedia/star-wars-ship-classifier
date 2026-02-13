---
title: Star Wars Ship Classifier
emoji: 🚀
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 5.23.1
python_version: 3.11
app_file: app.py
pinned: false
---


# 🚀 Star Wars Ship Classifier (Fastai + Gradio)
### An AI that distinguishes between X-Wings, Y-Wings and TIE Fighters with 96% accuracy.

[![Live Demo](https://img.shields.io/badge/Demo-Hugging%20Face-ffcc00?style=for-the-badge&logo=huggingface)](https://huggingface.co/spaces/Psychoutmedia/star_wars_ship_classification)

## 📖 Overview
This project is a deep learning image classifier built using the **fastai** library. It was developed as part of my journey through the Fast.ai "Practical Deep Learning for Coders" course. The model can identify three iconic Star Wars starfighters, even when presented with "Stealth" variants or different artistic styles.

🛠️ The "Hanging-Inference" Fix
Standard Fastai deployments often suffer from an "infinite spinning" bug on Hugging Face due to Dataloader deadlocks in restricted container environments.

This project implements a custom "Manual Inference" pipeline to solve this:

Bypasses learn.predict(): Directly calls learn.model(x) to avoid silent multiprocessing deadlocks.

CPU Hardening: Explicitly locks PyTorch to a single thread (torch.set_num_threads(1)) to prevent CPU contention on shared hardware.

Manual Transforms: Hand-codes the after_item and after_batch pipeline to ensure 100% reliability without background worker overhead.

🧪 Technical Stack
Framework: Fastai 2.7.x (PyTorch backend)

Interface: Gradio 4.44.0 (Pinning for SSR stability)

Optimization: Manual Logit processing & Softmax calculation

Deployment: Hugging Face Spaces (CPU Basic)

📈 Performance
Inference Speed: ~0.4s per image (Post-model load)

Memory Footprint: < 500MB RAM

Input: Automatically resizes any image to 224x224 for optimal CPU processing.

## 🛠️ The Tech Stack
* **Language:** Python
* **Framework:** [fastai](https://docs.fast.ai/) / PyTorch
* **Model:** ResNet34 (Transfer Learning)
* **Deployment:** [Gradio](https://gradio.app/) hosted on Hugging Face Spaces
* **Data Source:** DuckDuckGo Image Search

## 🧪 The Process
1.  **Data Collection:** Scraped ~200 images of X-Wings and TIE Fighters.
2.  **Data Cleaning:** Used the `ImageClassifierCleaner` to manually prune "garbage" data (toasters, logos, and incorrect ship types) from the training set.
3.  **Training:** Fine-tuned a pre-trained **ResNet34** model for 3 epochs.
4.  **Optimization:** Utilized the 1cycle policy and standard image augmentations (Resize/Crop) to ensure the model generalizes well to new, unseen images.

## 📈 Results
The model achieved a **96% accuracy** rate on the validation set. 
* **Successful Edge Case:** Correctly identified a "Black Stealth X-Wing" with **99.9% confidence**, proving it learned shape features rather than just color bias.

## 🚀 How to Run Locally
1. Clone the repo:
   ```bash
   git clone git@github.com:YOUR_USERNAME/star-wars-classifier.git
   ```
2. Install dependencies:
   ```bash   
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash   
   python app.py
   ```
Developed by Mark Stephenson for PsychoutMedia Feb 2026
