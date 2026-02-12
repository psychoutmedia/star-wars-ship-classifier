# 🚀 Star Wars Ship Classifier
### An AI that distinguishes between X-Wings and TIE Fighters with 96% accuracy.

[![Live Demo](https://img.shields.io/badge/Demo-Hugging%20Face-ffcc00?style=for-the-badge&logo=huggingface)](YOUR_HUGGING_FACE_SPACE_URL_HERE)

## 📖 Overview
This project is a deep learning image classifier built using the **fastai** library. It was developed as part of my journey through the Fast.ai "Practical Deep Learning for Coders" course. The model can identify two iconic Star Wars starfighters, even when presented with "Stealth" variants or different artistic styles.

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
