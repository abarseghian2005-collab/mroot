# Mroot

This project is a plant disease classification system. The goal is to predict the disease from an image, ignoring the plant type.

For example, both “tomato late blight” and “potato late blight” should be predicted as “late_blight”.

---

## Model

Backbone: EfficientNet-B0  
Classes: 39 (disease-only)  
Input size: 288x288  

---

## Results

~0.88 mAP on validation set  
~0.86 mAP on test set  

Note: slightly higher (~0.89) was observed earlier before fixing the validation split.

---

## Files

clean39_effb0_50ep_best.pth — trained model  
class_mapping.json — class index mapping  
clean39_effb0_50ep_results.json — evaluation results  
main.py — API for inference  

---

## Experiments

During development, multiple experiments were conducted to explore different setups, including:

- Different dataset splits (original vs corrected)
- Regularization techniques (dropout, mixup)
- Class merging strategies
- Backbone comparison (EfficientNet-B0 vs B1)

Only the most relevant runs are included below to keep things clear.

Baseline (corrected split):
https://api.wandb.ai/links/abarseghian2005-yerevan-state-university-ysu/7iyri28i

Merged classes experiment:
https://api.wandb.ai/links/abarseghian2005-yerevan-state-university-ysu/i5v3eqzb

Final best model (50 epochs):
https://wandb.ai/abarseghian2005-yerevan-state-university-ysu/plant-disease-classification/reports/Final-Best-Model--VmlldzoxNjU4OTQ3NQ?accessToken=v77dev8fndl1zocph733m3g6qqo6q7cj3vv3314z1fub1rv4h1z7pff8ichaio5m

Backbone comparison (EfficientNet-B1):
https://wandb.ai/abarseghian2005-yerevan-state-university-ysu/plant-disease-classification/reports/Backbone-Comparison-EfficentNet-B1---VmlldzoxNjU4OTQ5MQ?accessToken=yju3huojr9tvibt37i08u5bcu82cw26a2a577bthpgn571wd1q77zfo70y179nte

Note: WandB workspace is private, so direct workspace links may not be accessible.  
That’s why key experiments are shared via report links above.

---

## Useful Links

Experiments tracking (workspace):
https://wandb.ai/abarseghian2005-yerevan-state-university-ysu/plant-disease-classification

Model weights:
https://huggingface.co/Armennnn/mroot-plant-disease-classification

Live API:
https://mroot.onrender.com/docs

---

## How to run locally

Install dependencies:

pip install fastapi uvicorn torch torchvision timm pillow python-multipart

Run the API:

uvicorn Mroot.main:app --reload

Open in browser:

http://127.0.0.1:8000/docs

---

## Deployed API

Swagger / OpenAPI docs:
https://mroot.onrender.com/docs

Root endpoint:
https://mroot.onrender.com/

Note: the deployed API uses Render free tier, so the first request may take some time if the service is waking up.

---

## API

POST /predict

Upload an image and it returns:

{
  "prediction": "mosaic_virus",
  "confidence": 0.93
}
