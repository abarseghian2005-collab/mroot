Mroot

This project is a plant disease classification system. The goal is to predict the disease from an image, ignoring the plant type.

For example, both “tomato late blight” and “potato late blight” should be predicted as “late_blight”.

Model
Backbone: EfficientNet-B0
Classes: 39 (disease-only)
Input size: 288x288
Results
~0.88 mAP on our validation set
~0.86 mAP on test set

Note: higher (~0.89) was observed earlier, but that was before fixing the validation split.

Files
clean39_effb0_50ep_best.pth — trained model
class_mapping.json — class index mapping
clean39_effb0_50ep_results.json — evaluation results
main.py — API for inference
Useful links

Experiments tracking:
https://wandb.ai/abarseghian2005-yerevan-state-university-ysu/plant-disease-classification

Model weights:
https://huggingface.co/Armennnn/mroot-plant-disease-classification

Live API:
https://mroot.onrender.com/docs

How to run locally

Install dependencies:

pip install fastapi uvicorn torch torchvision timm pillow python-multipart

Run the API locally:

uvicorn Mroot.main:app --reload

Open locally in browser:

http://127.0.0.1:8000/docs

Deployed API

Swagger / OpenAPI docs:
https://mroot.onrender.com/docs

Root endpoint:
https://mroot.onrender.com/

Note: the deployed API uses Render free tier, so the first request may take some time if the service is waking up.

API

POST /predict

Upload an image and it returns JSON like this:

{
"prediction": "mosaic_virus",
"confidence": 0.93
}
