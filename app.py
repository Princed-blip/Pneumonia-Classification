
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import base64
from torchvision import models, transforms
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import matplotlib.pyplot as plt
import io

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="PulmoAI | Chest X-Ray Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM STYLING
# =========================================================

st.html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #f4f7fb;
}

[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stSidebar"] {
    background: #101d35;
}

[data-testid="stSidebar"] * {
    color: #e8eef8;
}

[data-testid="stSidebar"] .stMarkdown {
    color: #e8eef8;
}

.main .block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

.hero {
    background: linear-gradient(120deg, #101d35 0%, #173b57 60%, #087e83 100%);
    padding: 38px 42px;
    border-radius: 22px;
    color: white;
    margin-bottom: 28px;
    box-shadow: 0 12px 35px rgba(16, 29, 53, 0.12);
}

.hero-kicker {
    color: #75e0d5;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 12px;
}

.hero h1 {
    font-family: 'Manrope', sans-serif;
    font-size: 38px;
    font-weight: 800;
    color: white;
    margin: 0 0 12px 0;
}

.hero p {
    color: #d4e5ef;
    font-size: 15px;
    line-height: 1.7;
    max-width: 720px;
    margin: 0;
}

.section-heading {
    color: #172b49;
    font-family: 'Manrope', sans-serif;
    font-weight: 800;
    font-size: 21px;
    margin: 8px 0 5px 0;
}

.section-subtitle {
    color: #718096;
    font-size: 13px;
    margin-bottom: 18px;
}

.panel {
    background: white;
    padding: 24px;
    border-radius: 18px;
    border: 1px solid #e7edf4;
    box-shadow: 0 5px 20px rgba(31, 54, 82, 0.04);
    margin-bottom: 18px;
}

.panel-label {
    font-size: 12px;
    color: #087e83;
    font-weight: 800;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    margin-bottom: 8px;
}

.panel-title {
    color: #172b49;
    font-size: 18px;
    font-weight: 800;
    margin-bottom: 7px;
}

.panel-description {
    color: #78879a;
    font-size: 13px;
    line-height: 1.6;
}

.metric-card {
    background: white;
    padding: 20px;
    border: 1px solid #e7edf4;
    border-radius: 16px;
    box-shadow: 0 4px 15px rgba(31, 54, 82, 0.04);
    min-height: 112px;
}

.metric-label {
    color: #8290a2;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 1px;
    text-transform: uppercase;
}

.metric-value {
    color: #172b49;
    font-size: 25px;
    font-weight: 800;
    margin-top: 9px;
}

.result-card {
    padding: 25px;
    border-radius: 18px;
    margin: 12px 0 18px 0;
}

.normal-result {
    background: #e9f8f2;
    border: 1px solid #b8e8d2;
}

.pneumonia-result {
    background: #fff1ed;
    border: 1px solid #ffd1c5;
}

.result-label {
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 1.2px;
    margin-bottom: 8px;
}

.result-value {
    font-size: 30px;
    font-weight: 800;
    margin-bottom: 8px;
}

.result-description {
    font-size: 13px;
    line-height: 1.6;
}

.status-pill {
    display: inline-block;
    padding: 7px 12px;
    border-radius: 30px;
    background: #e3f5f2;
    color: #087e83;
    font-size: 11px;
    font-weight: 800;
}

.footer {
    text-align: center;
    color: #8794a5;
    font-size: 12px;
    padding: 25px 10px 5px 10px;
    border-top: 1px solid #e5ebf2;
    margin-top: 30px;
}

.stButton > button {
    background: linear-gradient(100deg, #087e83, #10a5a0);
    color: white;
    border: none;
    border-radius: 11px;
    padding: 13px 20px;
    font-weight: 800;
    min-height: 48px;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    background: linear-gradient(100deg, #06676c, #078b88);
    box-shadow: 0 6px 18px rgba(8, 126, 131, 0.2);
    transform: translateY(-1px);
}

[data-testid="stFileUploader"] {
    background: #f8fafc;
    border: 1px dashed #b8c7d7;
    border-radius: 14px;
    padding: 15px;
}

.stProgress > div > div > div {
    background-color: #0b9994;
}

div[data-testid="stExpander"] {
    border: 1px solid #e5ebf2;
    border-radius: 12px;
    background: white;
}
</style>
""")

# =========================================================
# MODEL CONFIGURATION
# =========================================================

MODEL_PATH = "best_model.pth"
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# IMAGE TRANSFORMATIONS
# =========================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    model = models.efficientnet_b3(weights=None)

    in_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 2)
    )

    try:
        checkpoint = torch.load(
            MODEL_PATH,
            map_location=DEVICE,
            weights_only=True
        )
    except TypeError:
        checkpoint = torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )

    # Support both raw state dictionaries and checkpoint dictionaries
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    # Remove DataParallel prefix if present
    if isinstance(checkpoint, dict):
        checkpoint = {
            key.replace("module.", ""): value
            for key, value in checkpoint.items()
        }

    model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()

    return model


# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict_image(model, image):
    image_rgb = image.convert("RGB")
    input_tensor = transform(image_rgb).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)

    confidence, predicted_class = torch.max(probabilities, dim=1)

    return (
        CLASS_NAMES[predicted_class.item()],
        confidence.item(),
        probabilities[0].cpu().numpy(),
        input_tensor
    )


# =========================================================
# GRAD-CAM FUNCTION
# =========================================================

def generate_gradcam(model, image, input_tensor, predicted_class):
    target_layers = [model.features[-1]]

    cam = GradCAM(
        model=model,
        target_layers=target_layers
    )

    targets = [
        ClassifierOutputTarget(predicted_class)
    ]

    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=targets
    )[0, :]

    rgb_image = np.array(
        image.convert("RGB").resize((224, 224))
    ).astype(np.float32) / 255.0

    visualization = show_cam_on_image(
        rgb_image,
        grayscale_cam,
        use_rgb=True
    )

    return visualization


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        """
        <div style="padding:15px 5px 25px 5px;">
            <div style="font-size:35px;">🫁</div>
            <div style="font-size:23px;font-weight:800;color:white;">
                PulmoAI
            </div>
            <div style="font-size:11px;color:#8fa8c4;letter-spacing:1px;">
                CHEST X-RAY INTELLIGENCE
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown("### 🏥 About the System")

    st.write(
        "PulmoAI is a deep-learning research prototype "
        "designed to classify chest X-ray images into "
        "Normal and Pneumonia categories."
    )

    st.markdown("---")

    st.markdown("### ⚙️ Model Information")

    st.write("**Architecture:** EfficientNet-B3")
    st.write("**Framework:** PyTorch")
    st.write("**Input size:** 224 × 224")
    st.write("**Explainability:** Grad-CAM")

    st.markdown("---")

    st.markdown("### 💻 System Status")

    st.success("Application ready")

    st.caption(f"Processing device: {DEVICE}")

    st.markdown("---")

    st.caption(
        "Developed by Michael Kosarachi\n\n"
        "Department of Biomedical Engineering\n\n"
        "University of Nigeria, Nsukka"
    )


# =========================================================
# HERO SECTION
# =========================================================

st.html("""
<div class="hero">
    <div class="hero-kicker">AI-POWERED MEDICAL IMAGING</div>
    <h1>Chest X-Ray Analysis</h1>
    <p>
        An intelligent chest imaging research tool that uses
        deep learning to classify X-ray images and generate
        visual explanations of model predictions.
    </p>
    <div style="margin-top:22px;">
        <span class="status-pill">● SYSTEM ONLINE</span>
    </div>
</div>
""")

# =========================================================
# MAIN CONTENT
# =========================================================

st.html("""
<div class="section-heading">Imaging Workspace</div>
<div class="section-subtitle">
    Upload a chest X-ray image and run the AI analysis when ready.
</div>
""")

upload_col, info_col = st.columns([1.5, 1], gap="large")

with upload_col:

    st.html("""
    <div class="panel">
        <div class="panel-label">01 / Image Input</div>
        <div class="panel-title">Upload Chest X-Ray</div>
        <div class="panel-description">
            Select a chest X-ray image in JPG, JPEG or PNG format.
            The image will be previewed before analysis.
        </div>
    </div>
    """)

    uploaded_file = st.file_uploader(
        "Choose an X-ray image",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear chest X-ray image."
    )

with info_col:

    st.html("""
    <div class="panel">
        <div class="panel-label">02 / AI Processing</div>
        <div class="panel-title">Intelligent Classification</div>
        <div class="panel-description">
            The EfficientNet-B3 model evaluates the uploaded image
            and estimates the probability of each class.
        </div>
        <hr style="border:0;border-top:1px solid #edf1f5;margin:18px 0;">
        <div style="color:#087e83;font-weight:700;font-size:13px;">
            ✓ Image preprocessing<br><br>
            ✓ Deep learning inference<br><br>
            ✓ Confidence estimation<br><br>
            ✓ Grad-CAM visualization
        </div>
    </div>
    """)

# =========================================================
# IMAGE PREVIEW AND ANALYSIS BUTTON
# =========================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.markdown("---")

    preview_col, action_col = st.columns([1.3, 1], gap="large")

    with preview_col:

        st.markdown("### Image Preview")

        st.image(
            image,
            caption="Uploaded chest X-ray",
            width="stretch"
        )

    with action_col:

        st.markdown("### Ready for Analysis?")

        st.write(
            "Review your uploaded image, then click the button "
            "below to begin the AI-powered classification."
        )

        st.info(
            "Analysis will only begin after you click "
            "**Analyze X-Ray**."
        )

        analyze_button = st.button(
            "🔍 Analyze X-Ray",
            width="stretch"
        )

        if analyze_button:

            st.session_state.analysis_result = None

            with st.spinner("Loading AI model and analyzing image..."):

                try:
                    model = load_model()

                    prediction, confidence, probabilities, input_tensor = (
                        predict_image(model, image)
                    )

                    predicted_index = CLASS_NAMES.index(prediction)

                    gradcam_image = generate_gradcam(
                        model,
                        image,
                        input_tensor,
                        predicted_index
                    )

                    st.session_state.analysis_result = {
                        "prediction": prediction,
                        "confidence": confidence,
                        "probabilities": probabilities,
                        "gradcam": gradcam_image,
                        "filename": uploaded_file.name,
                        "original_image": image.copy()
                    }


                    st.success("Analysis completed successfully!")

                except Exception as e:

                    st.error(
                        "An error occurred while analyzing the image."
                    )

                    with st.expander("Technical error details"):
                        st.exception(e)


# =========================================================
# DISPLAY ANALYSIS RESULTS
# =========================================================

if st.session_state.get("analysis_result") is not None:

    result = st.session_state.analysis_result

    prediction = result["prediction"]
    confidence = result["confidence"]
    probabilities = result["probabilities"]
    gradcam_image = result["gradcam"]
    original_image = result["original_image"]


    st.markdown("---")

    st.html("""
    <div class="section-heading">Analysis Results</div>
    <div class="section-subtitle">
        AI-generated classification and model confidence.
    </div>
    """)

    # RESULT CARD

    if prediction == "NORMAL":

        st.html(f"""
        <div class="result-card normal-result">
            <div class="result-label" style="color:#13865c;">
                MODEL CLASSIFICATION
            </div>
            <div class="result-value" style="color:#126b4c;">
                ✓ NORMAL
            </div>
            <div class="result-description" style="color:#39745d;">
                The model classified this image as Normal.
                This is an AI prediction, not a clinical diagnosis.
            </div>
        </div>
        """)

    else:

        st.html(f"""
        <div class="result-card pneumonia-result">
            <div class="result-label" style="color:#d45b3c;">
                MODEL CLASSIFICATION
            </div>
            <div class="result-value" style="color:#b9472e;">
                ⚠ PNEUMONIA
            </div>
            <div class="result-description" style="color:#965b4d;">
                The model classified this image as Pneumonia.
                This is an AI prediction, not a clinical diagnosis.
            </div>
        </div>
        """)

    # METRICS

    metric1, metric2, metric3 = st.columns(3)

    with metric1:
        st.html(f"""
        <div class="metric-card">
            <div class="metric-label">Predicted Class</div>
            <div class="metric-value" style="font-size:21px;">
                {prediction}
            </div>
        </div>
        """)

    with metric2:
        st.html(f"""
        <div class="metric-card">
            <div class="metric-label">Model Confidence</div>
            <div class="metric-value">
                {confidence * 100:.2f}%
            </div>
        </div>
        """)

    with metric3:
        st.html(f"""
        <div class="metric-card">
            <div class="metric-label">Model Architecture</div>
            <div class="metric-value" style="font-size:21px;">
                EfficientNet-B3
            </div>
        </div>
        """)

    st.markdown("### 📊 Class Probability Distribution")

    prob_col1, prob_col2 = st.columns(2)

    with prob_col1:
        st.write(f"**Normal:** {probabilities[0] * 100:.2f}%")
        st.progress(float(probabilities[0]))

    with prob_col2:
        st.write(f"**Pneumonia:** {probabilities[1] * 100:.2f}%")
        st.progress(float(probabilities[1]))

    # GRAD-CAM

    st.markdown("---")

    st.html("""
    <div class="section-heading">Model Explainability</div>
    <div class="section-subtitle">
        Grad-CAM visualization showing image regions that influenced
        the model's prediction.
    </div>
    """)

    cam_col1, cam_col2 = st.columns(2, gap="large")

    with cam_col1:

        st.markdown("#### Original X-Ray")

        st.image(
            original_image,
            width="stretch"
        )

        
    with cam_col2:

        st.markdown("#### Grad-CAM Heatmap")

        st.image(
            gradcam_image,
            width="stretch"
        )

    st.info(
        "Grad-CAM provides a visual explanation of regions "
        "that influenced the model's prediction. It does not "
        "represent a confirmed medical finding."
    )

    # DOWNLOAD REPORT

    st.markdown("---")

    st.markdown("### 📄 Analysis Summary")

    report = f"""
PULMOAI - CHEST X-RAY ANALYSIS REPORT

Image: {result['filename']}

Predicted Class: {prediction}

Model Confidence: {confidence * 100:.2f}%

Normal Probability: {probabilities[0] * 100:.2f}%

Pneumonia Probability: {probabilities[1] * 100:.2f}%

Model: EfficientNet-B3

Explainability Method: Grad-CAM

IMPORTANT DISCLAIMER:
This application is a research prototype.
Its output is not a medical diagnosis and should not
replace professional clinical assessment.
"""

    st.download_button(
        label="⬇️ Download Analysis Summary",
        data=report,
        file_name="pulmoai_analysis_report.txt",
        mime="text/plain",
        width="stretch"
    )


# =========================================================
# DISCLAIMER
# =========================================================

st.markdown("---")

st.warning(
    """
    **Medical Disclaimer**

    PulmoAI is an academic research prototype and is not intended
    for clinical diagnosis or treatment decisions. Its predictions
    should not replace assessment by a qualified healthcare professional.

    The model was trained using paediatric chest X-ray images
    from a single hospital. Its performance has not been established
    for adults or for images from other hospitals, scanners, or populations.
    """
)

# =========================================================
# FOOTER
# =========================================================

st.html("""
<div class="footer">
    <div style="font-weight:800;color:#52647a;font-size:14px;">
        PULMOAI
    </div>
    AI-Powered Chest X-Ray Research Application<br>
    Developed by Michael Kosarachi |
    Department of Biomedical Engineering, University of Nigeria, Nsukka
    <br><br>
    © 2026 PulmoAI. Academic Research Prototype.
</div>
""")
