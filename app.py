import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

st.set_page_config(page_title="Chest X-Ray Pneumonia Screening", layout="centered")

CLASSES = ["NORMAL", "PNEUMONIA"]

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


@st.cache_resource
def load_model():
    # Rebuild the exact architecture we trained, then load the saved weights
    model = models.efficientnet_b3(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(model.classifier[1].in_features, 2),
    )
    model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
    model.eval()
    return model


model = load_model()

st.title("Chest X-Ray Pneumonia Screening")
st.caption("AI decision-support prototype - Michael Kosarachi, Dept. of Biomedical Engineering, UNN")

st.warning(
    "This tool is a research prototype, not a medical device. Its output is not a diagnosis "
    "and must be reviewed by a qualified healthcare professional. It was trained on paediatric "
    "chest X-rays (ages 1-5) from one hospital and has not been validated on adults or on "
    "Nigerian hospital images."
)

uploaded = st.file_uploader("Upload a chest X-ray (JPEG or PNG)", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    pred = int(probs.argmax())

    # Grad-CAM heatmap on the class the model chose
    with GradCAM(model=model, target_layers=[model.features[-1]]) as cam:
        heat = cam(input_tensor=tensor, targets=[ClassifierOutputTarget(pred)])[0]
    rgb = np.array(image.resize((224, 224))).astype(np.float32) / 255.0
    overlay = show_cam_on_image(rgb, heat, use_rgb=True)

    st.subheader(f"Prediction: {CLASSES[pred]}")
    st.write(f"Model confidence: **{probs[pred].item():.1%}**")
    st.progress(float(probs[1]), text=f"Pneumonia probability: {probs[1].item():.1%}")

    col1, col2 = st.columns(2)
    col1.image(image, caption="Uploaded X-ray", use_container_width=True)
    col2.image(overlay, caption="Grad-CAM heatmap", use_container_width=True)
    st.caption("Red/yellow areas influenced the prediction most. The heatmap shows where the model "
               "focused; it does not confirm the diagnosis.")