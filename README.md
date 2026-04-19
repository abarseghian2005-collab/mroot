# Mroot

This project is a plant disease classification system. The goal is to predict the disease from an image, ignoring the plant type.

For example, both “tomato late blight” and “potato late blight” should be predicted as “late_blight”.

---

## Model

- Backbone: EfficientNet-B0  
- Classes: 39 (disease-only)  
- Input size: 288x288  

---

## Results

- ~0.88 mAP on our validation set  
- ~0.86 mAP on test set  

Note: higher (~0.89) was observed earlier, but that was before fixing the validation split.

---

## Files

- clean39_effb0_50ep_best.pth — trained model  
- class_mapping.json — class index mapping  
- clean39_effb0_50ep_results.json — evaluation results  
- main.py — API for inference  

---

experiments tracking:
https://wandb.ai/abarseghian2005-yerevan-state-university-ysu/plant-disease-classification
## How to run

Install dependencies:

pip install fastapi uvicorn torch torchvision timm pillow

Run the API:

uvicorn main:app --reload

Open in browser:

http://127.0.0.1:8000/docs

---

## API

POST /predict

Upload an image and it returns:

{
  "prediction": "...",
  "confidence": 0.91
}