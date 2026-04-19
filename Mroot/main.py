
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import io
import json
import torch
import timm
import torchvision.transforms as transforms

MODEL_PATH = "/kaggle/input/models/armenbars/clean39-effb0-final-completee/pytorch/default/1/clean39_effb0_50ep_best.pth"
MAPPING_PATH = "/kaggle/input/models/armenbars/clean39-effb0-final-completee/pytorch/default/1/class_mapping (1).json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(MAPPING_PATH, "r") as f:
    class_to_idx = json.load(f)

idx_to_class = {v: k for k, v in class_to_idx.items()}

transform = transforms.Compose([
    transforms.Resize((288, 288)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

model = timm.create_model(
    "efficientnet_b0",
    pretrained=False,
    num_classes=len(class_to_idx)
).to(device)

state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)
model.eval()

def predict_pil_image(image: Image.Image):
    image = image.convert("RGB")

    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs1 = model(x)

        flipped_img = image.transpose(Image.FLIP_LEFT_RIGHT)
        x_flip = transform(flipped_img).unsqueeze(0).to(device)
        outputs2 = model(x_flip)

        probs1 = torch.softmax(outputs1, dim=1)
        probs2 = torch.softmax(outputs2, dim=1)

        probs = (probs1 + probs2) / 2.0
        conf, pred_idx = torch.max(probs, dim=1)

    return {
        "prediction": idx_to_class[pred_idx.item()],
        "confidence": float(conf.item())
    }

app = FastAPI(title="Mroot Plant Disease API")

@app.get("/")
def root():
    return {"message": "Mroot API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Only JPG and PNG images are allowed.")

    image_bytes = await file.read()

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    result = predict_pil_image(image)
    return result
