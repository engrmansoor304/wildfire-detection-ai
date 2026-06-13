# ═══════════════════════════════════════════════════════════════
# 🔥 WILDFIRE DETECTION SYSTEM - STREAMLIT APP (FIXED)
# ═══════════════════════════════════════════════════════════════

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import cv2
from PIL import Image
import io
import os
import json
import time
import base64
import warnings
from datetime import datetime
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import models
from sklearn.metrics import roc_curve, auc
import folium
# Add this right after all imports
import tempfile
import sys

# Configure temporary directory for file uploads
tempfile.tempdir = "/app/temp"
os.makedirs(tempfile.tempdir, exist_ok=True)

# Increase file upload limit
st.set_option('deprecation.showfileUploaderEncoding', False)
from streamlit_folium import folium_static
# Load Groq API key from environment or use hardcoded (for testing only)
GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '')
# DEBUG - Remove after testing
if not GROQ_API_KEY:
    st.warning("⚠️ Groq API key not found. PDF reports will be disabled. Add your key in Settings → Variables and secrets")

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="🔥 Wildfire Detection AI",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.set_option('deprecation.showfileUploaderEncoding', False)

# ═══════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Space+Grotesk:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #0d0d0d; color: #ffffff; }
.stApp { background-color: #0d0d0d; }
#MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;} .stDeployButton {display: none;}
::-webkit-scrollbar {width: 6px;} ::-webkit-scrollbar-track {background: #0d0d0d;} ::-webkit-scrollbar-thumb {background: #e94560; border-radius: 10px;}
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0a0a0a 0%, #111827 100%); border-right: 1px solid #1f2937; }
[data-testid="stSidebar"] .stRadio label { color: #d1d5db !important; font-size: 14px !important; padding: 8px 0 !important; }
[data-testid="stSidebar"] .stTextInput input { background: #1a1a2e !important; border: 1px solid #e94560 !important; color: white !important; border-radius: 8px !important; }
.main-title { font-family: 'Space Grotesk', sans-serif; font-size: 3.5rem; font-weight: 800; text-align: center; background: linear-gradient(135deg, #e94560 0%, #f97316 50%, #e94560 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; animation: fireGlow 3s ease-in-out infinite alternate; line-height: 1.1; margin-bottom: 0.5rem; }
@keyframes fireGlow { 0% {filter: drop-shadow(0 0 10px rgba(233,69,96,0.5));} 100% {filter: drop-shadow(0 0 30px rgba(249,115,22,0.8));} }
.sub-title { text-align: center; color: #9ca3af; font-size: 1.1rem; font-weight: 400; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 2rem; }
.metric-card { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); border: 1px solid #e94560; border-radius: 16px; padding: 24px 20px; text-align: center; box-shadow: 0 0 20px rgba(233,69,96,0.15), 0 4px 20px rgba(0,0,0,0.4); transition: transform 0.3s ease, box-shadow 0.3s ease; margin: 8px 0; }
.metric-card:hover { transform: translateY(-4px); box-shadow: 0 0 40px rgba(233,69,96,0.3), 0 8px 30px rgba(0,0,0,0.5); }
.metric-value { font-family: 'Space Grotesk', sans-serif; font-size: 2.2rem; font-weight: 700; color: #e94560; line-height: 1; }
.metric-label { font-size: 0.8rem; color: #9ca3af; text-transform: uppercase; letter-spacing: 1.5px; margin-top: 6px; }
.model-card { background: #1a1a2e; border: 1px solid #374151; border-radius: 12px; padding: 20px; text-align: center; margin: 8px 0; transition: all 0.3s ease; }
.model-card:hover { border-color: #e94560; box-shadow: 0 0 20px rgba(233,69,96,0.2); }
.model-name { font-size: 0.95rem; font-weight: 600; color: #e5e7eb; margin-bottom: 8px; }
.model-acc { font-size: 1.8rem; font-weight: 700; font-family: 'Space Grotesk', sans-serif; }
.result-fire { background: linear-gradient(135deg, rgba(239,68,68,0.15) 0%, rgba(233,69,96,0.1) 100%); border: 2px solid #ef4444; border-radius: 16px; padding: 28px; text-align: center; box-shadow: 0 0 40px rgba(239,68,68,0.2); animation: pulseRed 2s ease-in-out infinite; }
@keyframes pulseRed { 0%, 100% {box-shadow: 0 0 40px rgba(239,68,68,0.2);} 50% {box-shadow: 0 0 60px rgba(239,68,68,0.4);} }
.result-safe { background: linear-gradient(135deg, rgba(34,197,94,0.15) 0%, rgba(16,185,129,0.1) 100%); border: 2px solid #22c55e; border-radius: 16px; padding: 28px; text-align: center; box-shadow: 0 0 40px rgba(34,197,94,0.2); }
.result-label { font-family: 'Space Grotesk', sans-serif; font-size: 2rem; font-weight: 800; margin-bottom: 8px; }
[data-testid="stFileUploader"] { background: #1a1a2e !important; border: 2px dashed #e94560 !important; border-radius: 16px !important; padding: 20px !important; }
.section-header { font-family: 'Space Grotesk', sans-serif; font-size: 1.5rem; font-weight: 700; color: white; border-left: 4px solid #e94560; padding-left: 16px; margin: 28px 0 16px 0; }
.tech-badge { display: inline-block; background: #1a1a2e; border: 1px solid #374151; border-radius: 20px; padding: 6px 14px; font-size: 0.8rem; color: #9ca3af; margin: 4px; font-weight: 500; }
.stButton > button { background: linear-gradient(135deg, #e94560 0%, #c2185b 100%); color: white; border: none; border-radius: 10px; padding: 12px 28px; font-size: 1rem; font-weight: 600; cursor: pointer; transition: all 0.3s ease; box-shadow: 0 4px 15px rgba(233,69,96,0.3); width: 100%; }
.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 8px 25px rgba(233,69,96,0.5); }
hr { border: none; border-top: 1px solid #1f2937; margin: 24px 0; }
.info-box { background: rgba(6,182,212,0.08); border: 1px solid #06b6d4; border-radius: 12px; padding: 16px 20px; margin: 12px 0; color: #e5e7eb; font-size: 0.9rem; line-height: 1.6; }
.step-card { background: #1a1a2e; border: 1px solid #374151; border-radius: 12px; padding: 20px; text-align: center; height: 140px; display: flex; flex-direction: column; justify-content: center; }
.debug-box { background: rgba(239,68,68,0.08); border: 1px solid #ef4444; border-radius: 8px; padding: 12px; margin: 8px 0; font-size: 0.8rem; color: #fca5a5; font-family: monospace; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════
# Add after other constants
TEMPERATURE = 1.0  # Adjust: >1 = more confident, <1 = less confident
CLASSES     = ["nowildfire", "wildfire"]
NUM_CLASSES = 2
IMG_SIZE    = 224
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEBUG_MODE  = False  # ← ADD THIS LINE (set to False to hide debug info)

MODEL_PATHS = {
    "MobileNetV2"    : "models/MobileNetV2_best.pth",
    "MobileNetV3"    : "models/MobileNetV3_best.pth",
    "EfficientNet-B0": "models/EfficientNet-B0_best.pth",
}

MODEL_STATS = {
    "MobileNetV2"    : {"val_acc": 95.76, "color": "#06b6d4"},
    "MobileNetV3"    : {"val_acc": 97.22, "color": "#e94560"},
    "EfficientNet-B0": {"val_acc": 97.03, "color": "#80ffdb"},
}

RESULTS_PATH = "results"
ASSETS_PATH  = "assets"

# ═══════════════════════════════════════════════════════════════
# VALIDATION TRANSFORM — matches training exactly
# ═══════════════════════════════════════════════════════════════
val_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    ),
    ToTensorV2()
])

# ═══════════════════════════════════════════════════════════════
# MODEL DEFINITIONS — must match training notebook exactly
# ═══════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════
# MODEL DEFINITIONS — FIXED to match your saved models
# ═══════════════════════════════════════════════════════════════




# ═══════════════════════════════════════════════════════════════
# MODEL DEFINITIONS — FIXED to match your saved models
# ═══════════════════════════════════════════════════════════════

class MobileNetV2Model(nn.Module):
    """
    FIXED: Matches your saved MobileNetV2 model
    """
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = models.mobilenet_v2(pretrained=False)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


class MobileNetV3Model(nn.Module):
    """
    FIXED: Matches your saved MobileNetV3 model
    """
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = models.mobilenet_v3_large(pretrained=False)
        in_features = self.backbone.classifier[3].in_features
        self.backbone.classifier[3] = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


class EfficientNetB0Model(nn.Module):
    """
    FIXED: Handles missing efficientnet_pytorch gracefully
    """
    def __init__(self, num_classes=2):
        super().__init__()
        try:
            from efficientnet_pytorch import EfficientNet
            self.backbone = EfficientNet.from_name("efficientnet-b0")
            in_features = self.backbone._fc.in_features
            self.backbone._fc = nn.Sequential(
                nn.Linear(in_features, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, num_classes)
            )
            self.loaded = True
        except ImportError:
            self.loaded = False
            raise ImportError("efficientnet_pytorch not installed. Run: pip install efficientnet-pytorch")

    def forward(self, x):
        if not self.loaded:
            raise RuntimeError("EfficientNet model not properly initialized")
        return self.backbone(x)


# ═══════════════════════════════════════════════════════════════
# LOAD MODELS - SIMPLE VERSION FOR HUGGING FACE
# ═══════════════════════════════════════════════════════════════

def load_all_models():
    loaded = {}
    load_errors = {}
    
    # Simple loading without complex caching
    for name, path in MODEL_PATHS.items():
        try:
            # Skip EfficientNet if causing issues
            if name == "EfficientNet-B0" and not os.path.exists(path):
                load_errors[name] = "File not found"
                loaded[name] = None
                continue
            
            # Load checkpoint
            checkpoint = torch.load(path, map_location=DEVICE)
            
            # Get state dict
            if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
                state_dict = checkpoint['model_state']
            else:
                state_dict = checkpoint
            
            # Create model
            if name == "MobileNetV2":
                from torchvision.models import mobilenet_v2
                model = mobilenet_v2(pretrained=False)
                in_features = model.classifier[1].in_features
                model.classifier = nn.Sequential(
                    nn.Dropout(0.2),
                    nn.Linear(in_features, 256),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(256, NUM_CLASSES)
                )
            elif name == "MobileNetV3":
                from torchvision.models import mobilenet_v3_large
                model = mobilenet_v3_large(pretrained=False)
                in_features = model.classifier[3].in_features
                model.classifier[3] = nn.Sequential(
                    nn.Linear(in_features, 256),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(256, NUM_CLASSES)
                )
            elif name == "EfficientNet-B0":
                try:
                    from efficientnet_pytorch import EfficientNet
                    model = EfficientNet.from_name("efficientnet-b0")
                    in_features = model._fc.in_features
                    model._fc = nn.Sequential(
                        nn.Linear(in_features, 256),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(256, NUM_CLASSES)
                    )
                except:
                    load_errors[name] = "EfficientNet library not available"
                    loaded[name] = None
                    continue
            
            # Load weights
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            
            loaded[name] = model
            load_errors[name] = None
            
        except Exception as e:
            load_errors[name] = str(e)
            loaded[name] = None
    
    return loaded, load_errors
# ═══════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════
# PREDICT FUNCTION — OPTIMIZED FOR BETTER CONFIDENCE
# ═══════════════════════════════════════════════════════════════
def predict_image(models_dict, img_pil):
    # Convert PIL to numpy RGB
    img_np = np.array(img_pil.convert("RGB"))
    
    # Apply validation transform
    transformed = val_transform(image=img_np)
    tensor = transformed["image"].unsqueeze(0).to(DEVICE)
    
    results = {}
    
    for name, model in models_dict.items():
        if model is None:
            results[name] = {
                "prediction": "unknown",
                "confidence": 0.0,
                "probs": {"nowildfire": 0.0, "wildfire": 0.0},
                "error": "Model not loaded"
            }
            continue
        
        try:
            model.eval()
            with torch.no_grad():
                # Forward pass
                logits = model(tensor)
                
                # Apply temperature scaling for better confidence calibration
                temperature = 1.2  # Slightly lower confidence (more conservative)
                probs = F.softmax(logits / temperature, dim=1)
                probs_np = probs.cpu().numpy()[0]
            
            pred_idx = int(np.argmax(probs_np))
            pred_class = CLASSES[pred_idx]
            confidence = float(probs_np[pred_idx]) * 100.0
            
            results[name] = {
                "prediction": pred_class,
                "confidence": confidence,
                "probs": {
                    CLASSES[i]: float(probs_np[i]) * 100.0
                    for i in range(NUM_CLASSES)
                },
                "error": None
            }
            
        except Exception as e:
            results[name] = {
                "prediction": "unknown",
                "confidence": 0.0,
                "probs": {"nowildfire": 0.0, "wildfire": 0.0},
                "error": str(e)
            }
    
    return results
# ═══════════════════════════════════════════════════════════════
# GRAD-CAM
# ═══════════════════════════════════════════════════════════════
def get_gradcam(model, img_pil, model_name):
    try:
        img_np = np.array(img_pil.convert("RGB").resize((IMG_SIZE, IMG_SIZE)))
        tensor = val_transform(image=img_np)["image"].unsqueeze(0).to(DEVICE)

        if "MobileNetV2" in model_name:
            target_layer = model.backbone.features[-1]
        elif "MobileNetV3" in model_name:
            target_layer = model.backbone.features[-1]
        else:
            try:
                target_layer = model.backbone._blocks[-1]
            except Exception:
                target_layer = list(model.backbone.children())[-3]

        gradients  = []
        activations = []

        def save_grad(grad):
            gradients.append(grad)

        def forward_hook(module, input, output):
            activations.append(output)
            output.register_hook(save_grad)

        handle = target_layer.register_forward_hook(forward_hook)

        model.zero_grad()
        out  = model(tensor)
        pred = out.argmax(dim=1)
        out[0, pred].backward()
        handle.remove()

        if not gradients or not activations:
            return None

        grads   = gradients[0].cpu().numpy()[0]
        acts    = activations[0].cpu().detach().numpy()[0]
        weights = grads.mean(axis=(1, 2))

        cam = np.zeros(acts.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * acts[i]

        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam = cam / cam.max()

        cam_resized = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
        return cam_resized

    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════
# CHART HELPERS
# ═══════════════════════════════════════════════════════════════
CHART_LAYOUT = dict(
    template      = "plotly_dark",
    paper_bgcolor = "#0d0d0d",
    plot_bgcolor  = "#111827",
    font          = dict(family="Inter", color="white", size=12),
    margin        = dict(l=20, r=20, t=50, b=20),
    hoverlabel    = dict(
        bgcolor    = "#1a1a2e",
        bordercolor= "#e94560",
        font_size  = 12,
        font_color = "white"
    )
)

SCENE_3D = dict(
    bgcolor = "#0d0d0d",
    xaxis   = dict(gridcolor="#1f2937", showbackground=True, backgroundcolor="#111827"),
    yaxis   = dict(gridcolor="#1f2937", showbackground=True, backgroundcolor="#111827"),
    zaxis   = dict(gridcolor="#1f2937", showbackground=True, backgroundcolor="#111827")
)


def chart_confidence_gauge(confidence, prediction):
    color = "#ef4444" if prediction == "wildfire" else "#22c55e"
    fig = go.Figure(go.Indicator(
        mode  = "gauge+number+delta",
        value = confidence,
        title = dict(text="🎯 Detection Confidence", font=dict(size=16, color="white")),
        number= dict(suffix="%", font=dict(size=40, color=color, family="Space Grotesk")),
        gauge = dict(
            axis      = dict(range=[0, 100], tickcolor="white", tickfont=dict(color="white")),
            bar       = dict(color=color, thickness=0.3),
            bgcolor   = "#1a1a2e",
            borderwidth=2,
            bordercolor="#374151",
            steps=[
                dict(range=[0,  40], color="rgba(34,197,94,0.15)"),
                dict(range=[40, 70], color="rgba(245,158,11,0.15)"),
                dict(range=[70,100], color="rgba(239,68,68,0.15)"),
            ],
            threshold=dict(line=dict(color=color, width=4), thickness=0.8, value=confidence)
        )
    ))
    fig.update_layout(**CHART_LAYOUT, height=320)
    return fig


def chart_model_bars(results):
    names  = [n for n in results if results[n]["error"] is None]
    confs  = [results[n]["confidence"] for n in names]
    colors = [MODEL_STATS[n]["color"] for n in names]
    accs   = [MODEL_STATS[n]["val_acc"] for n in names]

    fig = go.Figure()
    for name, conf, color, acc in zip(names, confs, colors, accs):
        fig.add_trace(go.Bar(
            name         = name,
            x            = [name],
            y            = [conf],
            marker       = dict(color=color, opacity=0.9, line=dict(color=color, width=2)),
            text         = f"{conf:.1f}%",
            textposition = "outside",
            textfont     = dict(color="white", size=13),
            hovertemplate= f"<b>{name}</b><br>Confidence: {conf:.1f}%<br>Val Accuracy: {acc}%<extra></extra>"
        ))
    fig.update_layout(
        **CHART_LAYOUT,
        title     = dict(text="🤖 Model Confidence Comparison", font=dict(size=15, color="white")),
        height    = 380,
        showlegend= True,
        xaxis_title="Model",
        yaxis_title="Confidence (%)",
        yaxis     = dict(range=[0, 115]),
        bargap    = 0.3
    )
    return fig


def chart_prob_breakdown(results):
    """Bar chart showing nowildfire vs wildfire probability per model."""
    names = [n for n in results if results[n]["error"] is None]
    nw_probs = [results[n]["probs"].get("nowildfire", 0) for n in names]
    wf_probs = [results[n]["probs"].get("wildfire",   0) for n in names]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name         = "No Wildfire",
        x            = names,
        y            = nw_probs,
        marker_color = "#22c55e",
        text         = [f"{v:.1f}%" for v in nw_probs],
        textposition = "auto",
    ))
    fig.add_trace(go.Bar(
        name         = "Wildfire",
        x            = names,
        y            = wf_probs,
        marker_color = "#ef4444",
        text         = [f"{v:.1f}%" for v in wf_probs],
        textposition = "auto",
    ))
    fig.update_layout(
        **CHART_LAYOUT,
        title    = dict(text="📊 Class Probability per Model", font=dict(size=15, color="white")),
        height   = 360,
        barmode  = "group",
        yaxis    = dict(range=[0, 110], title="Probability (%)"),
        xaxis    = dict(title="Model"),
    )
    return fig


def chart_gradcam_3d(cam_map):
    if cam_map is None:
        return None
    fig = go.Figure(data=[go.Surface(
        z         = cam_map,
        colorscale= "Inferno",
        showscale = True,
        colorbar  = dict(
         title    = "Activation",
         title_font= dict(color="white"),  # Changed from titlefont to title_font
         tickfont = dict(color="white")
        ),
            
            
            
        
        hovertemplate= "X: %{x}<br>Y: %{y}<br>Activation: %{z:.3f}<extra></extra>"
    )])
    fig.update_layout(
        **CHART_LAYOUT,
        title = dict(text="🌋 Grad-CAM 3D Activation Surface", font=dict(size=15, color="white")),
        height= 480,
        scene = dict(**SCENE_3D, xaxis_title="X", yaxis_title="Y", zaxis_title="Activation")
    )
    return fig


def chart_rgb_histogram(img_pil):
    img_np = np.array(img_pil.convert("RGB").resize((224, 224)))
    bins   = np.arange(0, 256, 8)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    fig = go.Figure()
    for ch, color, label in [(0, "#ef4444", "Red"), (1, "#22c55e", "Green"), (2, "#3b82f6", "Blue")]:
        hist, _ = np.histogram(img_np[:, :, ch].flatten(), bins=bins)
        fig.add_trace(go.Scatter(
            x    = bin_centers,
            y    = hist,
            name = label,
            mode = "lines",
            fill = "tozeroy",
            line = dict(color=color, width=2),
            hovertemplate=f"<b>{label}</b><br>Pixel: %{{x}}<br>Count: %{{y}}<extra></extra>"
        ))
    fig.update_layout(
        **CHART_LAYOUT,
        title      = dict(text="🌈 RGB Channel Distribution", font=dict(size=15, color="white")),
        height     = 340,
        xaxis_title= "Pixel Intensity",
        yaxis_title= "Count"
    )
    return fig


def chart_confusion_matrix_3d():
    try:
        cm_path = os.path.join(RESULTS_PATH, "confusion_matrix.csv")
        if not os.path.exists(cm_path):
            return None
        cm = pd.read_csv(cm_path, index_col=0).values.astype(float)
        fig = go.Figure(data=[go.Surface(
            z         = cm,
            x         = CLASSES,
            y         = CLASSES,
            colorscale= "Viridis",
            showscale = True,
            hovertemplate= "Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>"
        )])
        fig.update_layout(
            **CHART_LAYOUT,
            title = dict(text="🎯 3D Confusion Matrix", font=dict(size=15, color="white")),
            height= 500,
            scene = dict(**SCENE_3D, xaxis_title="Predicted", yaxis_title="Actual", zaxis_title="Count")
        )
        return fig
    except Exception:
        return None


def chart_roc_curve():
    try:
        pred_path = os.path.join(RESULTS_PATH, "test_predictions.csv")
        if not os.path.exists(pred_path):
            return None
        df     = pd.read_csv(pred_path)
        y_true = (df["true_label"] == "wildfire").astype(int)
        y_conf = df["confidence"]
        fpr, tpr, _ = roc_curve(y_true, y_conf)
        roc_auc     = auc(fpr, tpr)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"ROC (AUC={roc_auc:.3f})",
                                 mode="lines", line=dict(color="#06b6d4", width=3),
                                 fill="tozeroy", fillcolor="rgba(6,182,212,0.08)"))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], name="Random",
                                 mode="lines", line=dict(color="#6b7280", width=1.5, dash="dash")))
        fig.update_layout(**CHART_LAYOUT,
                          title=dict(text=f"📈 ROC Curve — AUC={roc_auc:.3f}", font=dict(size=15, color="white")),
                          height=420, xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        return fig
    except Exception:
        return None


def chart_radar():
    try:
        report_path = os.path.join(RESULTS_PATH, "classification_report.csv")
        if not os.path.exists(report_path):
            return None
        df   = pd.read_csv(report_path, index_col=0)
        cats = ["precision", "recall", "f1-score"]
        fig  = go.Figure()
        for i, cls in enumerate(CLASSES):
            if cls in df.index:
                vals = [df.loc[cls, c] for c in cats] + [df.loc[cls, cats[0]]]
                fig.add_trace(go.Scatterpolar(
                    r=vals, theta=cats + [cats[0]], fill="toself", name=cls,
                    line=dict(color=["#e94560", "#06b6d4"][i], width=2)
                ))
        fig.update_layout(**CHART_LAYOUT,
                          title=dict(text="🕸️ Per-Class Metrics Radar", font=dict(size=15, color="white")),
                          height=420,
                          polar=dict(radialaxis=dict(visible=True, range=[0,1], color="white",
                                                     gridcolor="#1f2937"),
                                     bgcolor="#111827",
                                     angularaxis=dict(color="white")))
        return fig
    except Exception:
        return None


def chart_sunburst():
    try:
        cm_path = os.path.join(RESULTS_PATH, "confusion_matrix.csv")
        if not os.path.exists(cm_path):
            return None
        cm = pd.read_csv(cm_path, index_col=0)
        labels, parents, values = ["All Predictions"], [""], [0]
        total = 0
        for cls in CLASSES:
            if cls in cm.index:
                row   = cm.loc[cls]
                corr  = int(row[cls]) if cls in row.index else 0
                wrong = int(row.sum()) - corr
                total += corr + wrong
                labels  += [cls, f"{cls} ✅", f"{cls} ❌"]
                parents += ["All Predictions", cls, cls]
                values  += [corr + wrong, corr, wrong]
        values[0] = total
        fig = go.Figure(go.Sunburst(
            labels=labels, parents=parents, values=values,
            branchvalues="total", maxdepth=3,
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percentParent:.1%}<extra></extra>"
        ))
        fig.update_layout(**CHART_LAYOUT,
                          title=dict(text="☀️ Prediction Breakdown", font=dict(size=15, color="white")),
                          height=450)
        return fig
    except Exception:
        return None




# ═══════════════════════════════════════════════════════════════
# PDF REPORT GENERATOR
# ═══════════════════════════════════════════════════════════════
def generate_pdf_report(groq_key, img_pil, predictions, ensemble, avg_conf, risk_level):
    try:
        from groq import Groq
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.units import inch
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                        Table, TableStyle, HRFlowable,
                                        PageBreak, Image as RLImage)
        from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY

        client = Groq(api_key=groq_key)

        pred_summary = "\n".join([
            f"- {n}: {v['prediction'].upper()} ({v['confidence']:.1f}%)"
            for n, v in predictions.items() if v["error"] is None
        ])

        def ask_groq(prompt):
            resp = client.chat.completions.create(
                model   = "llama-3.3-70b-versatile",
                messages= [
                    {"role": "system", "content": "You are a senior wildfire and remote sensing expert. Write professional, technical, detailed reports. Use formal language."},
                    {"role": "user",   "content": prompt}
                ],
                max_tokens =1200,
                temperature=0.3
            )
            return resp.choices[0].message.content.strip()

        pages = {}
        page_prompts = {
            "p1": f"Write a professional executive summary (3 paragraphs) for a wildfire detection report. Predictions: {pred_summary}. Ensemble: {ensemble.upper()} ({avg_conf:.1f}%). Risk: {risk_level}.",
            "p2": f"Write a detailed technical analysis (3 paragraphs) for wildfire detection from satellite imagery. Predictions: {pred_summary}. Cover preprocessing, inference, confidence interpretation.",
            "p3": f"Write model performance analysis (3 paragraphs) comparing MobileNetV2 (95.76%), MobileNetV3 (97.22%), EfficientNet-B0 (97.03%). Best: MobileNetV3.",
            "p4": f"Write a comprehensive risk assessment (3 paragraphs). Result: {ensemble.upper()}. Confidence: {avg_conf:.1f}%. Risk: {risk_level}.",
            "p5": f"Write environmental impact analysis (3 paragraphs) for {ensemble} scenario.",
            "p6": f"Write emergency response protocol (3 paragraphs) for {ensemble.upper()} with {avg_conf:.1f}% confidence.",
            "p7": f"Write prevention and mitigation recommendations (3 paragraphs) for wildfire risk.",
            "p8": f"Write conclusions (3 paragraphs). Key findings: {ensemble.upper()}, {avg_conf:.1f}% confidence, MobileNetV3 best at 97.22%.",
        }

        progress = st.progress(0)
        status   = st.empty()
        for i, (key, prompt) in enumerate(page_prompts.items()):
            status.text(f"Generating page {i+1}/8...")
            pages[key] = ask_groq(prompt)
            progress.progress((i + 1) / 8)

        status.text("Building PDF...")
        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4,
                                rightMargin=0.75*inch, leftMargin=0.75*inch,
                                topMargin=0.75*inch,  bottomMargin=0.75*inch)

        styles  = getSampleStyleSheet()
        s_title = ParagraphStyle("T", parent=styles["Title"], fontSize=26,
                                 textColor=colors.HexColor("#1a1a2e"),
                                 alignment=TA_CENTER, fontName="Helvetica-Bold")
        s_sub   = ParagraphStyle("S", parent=styles["Normal"], fontSize=12,
                                 textColor=colors.HexColor("#e94560"),
                                 alignment=TA_CENTER, fontName="Helvetica-Bold")
        s_h1    = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=15,
                                 textColor=colors.HexColor("#16213e"), fontName="Helvetica-Bold")
        s_body  = ParagraphStyle("B", parent=styles["Normal"], fontSize=10,
                                 leading=15, alignment=TA_JUSTIFY, fontName="Helvetica")
        s_alert_r = ParagraphStyle("AR", parent=styles["Normal"], fontSize=12,
                                   textColor=colors.white,
                                   backColor=colors.HexColor("#e94560"),
                                   alignment=TA_CENTER, fontName="Helvetica-Bold")
        s_alert_g = ParagraphStyle("AG", parent=styles["Normal"], fontSize=12,
                                   textColor=colors.white,
                                   backColor=colors.HexColor("#2d6a4f"),
                                   alignment=TA_CENTER, fontName="Helvetica-Bold")
        s_foot  = ParagraphStyle("F", parent=styles["Normal"], fontSize=8,
                                 textColor=colors.grey, alignment=TA_CENTER)

        def hr_line():
            return HRFlowable(width="100%", thickness=1.5,
                              color=colors.HexColor("#e94560"), spaceAfter=8, spaceBefore=4)

        def body_paras(text):
            out = []
            for p in text.strip().split("\n\n"):
                p = p.strip()
                if p:
                    out.append(Paragraph(p, s_body))
                    out.append(Spacer(1, 6))
            return out

        story = []
        story.append(Spacer(1, 0.4*inch))
        story.append(Paragraph("🔥 WILDFIRE DETECTION REPORT", s_title))
        story.append(Spacer(1, 0.15*inch))
        story.append(hr_line())
        story.append(Paragraph("AI-Powered Satellite Analysis", s_sub))
        story.append(hr_line())
        story.append(Spacer(1, 0.2*inch))

        if ensemble == "wildfire":
            story.append(Paragraph(f"⚠️ WILDFIRE DETECTED — HIGH RISK — {avg_conf:.1f}% CONFIDENCE", s_alert_r))
        else:
            story.append(Paragraph(f"✅ NO WILDFIRE — LOW RISK — {avg_conf:.1f}% CONFIDENCE", s_alert_g))

        story.append(Spacer(1, 0.2*inch))
        img_copy = img_pil.copy()
        img_copy.thumbnail((380, 280))
        img_buf = io.BytesIO()
        img_copy.save(img_buf, format="PNG")
        img_buf.seek(0)
        story.append(RLImage(img_buf, width=3.2*inch, height=2.4*inch))
        story.append(Spacer(1, 0.15*inch))

        meta = [
            ["Report Date",     datetime.now().strftime("%B %d, %Y %H:%M")],
            ["Ensemble Result", ensemble.upper()],
            ["Avg Confidence",  f"{avg_conf:.1f}%"],
            ["Risk Level",      risk_level],
            ["Best Model",      "MobileNetV3 (97.22%)"],
            ["Models Used",     "MobileNetV2, MobileNetV3, EfficientNet-B0"],
        ]
        mt = Table(meta, colWidths=[2*inch, 4.5*inch])
        mt.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (0,-1), colors.HexColor("#16213e")),
            ("TEXTCOLOR",  (0,0), (0,-1), colors.white),
            ("FONTNAME",   (0,0), (0,-1), "Helvetica-Bold"),
            ("FONTNAME",   (1,0), (1,-1), "Helvetica"),
            ("FONTSIZE",   (0,0), (-1,-1), 9),
            ("GRID",       (0,0), (-1,-1), 0.5, colors.grey),
            ("PADDING",    (0,0), (-1,-1), 7),
        ]))
        story.append(mt)
        story.append(PageBreak())

        page_titles = [
            "📋 EXECUTIVE SUMMARY", "🔬 TECHNICAL ANALYSIS",
            "🤖 MODEL PERFORMANCE",  "⚠️ RISK ASSESSMENT",
            "🌿 ENVIRONMENTAL IMPACT","🚨 EMERGENCY RESPONSE",
            "🛡️ PREVENTION & MITIGATION","📌 CONCLUSIONS",
        ]
        for i, (title, content) in enumerate(zip(page_titles, pages.values())):
            story.append(Paragraph(title, s_h1))
            story.append(hr_line())
            story.extend(body_paras(content))
            if i < len(page_titles) - 1:
                story.append(PageBreak())

        story.append(Spacer(1, 0.3*inch))
        story.append(hr_line())
        story.append(Paragraph(
            f"Generated by Wildfire Detection AI | Powered by PyTorch + Groq LLaMA3-70B | © {datetime.now().year}",
            s_foot
        ))

        doc.build(story)
        progress.progress(1.0)
        status.text("✅ Report ready!")
        buf.seek(0)
        return buf.getvalue()

    except Exception as e:
        st.error(f"Report generation failed: {e}")
        return None


# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='padding:24px 0 8px 0; text-align:center;'>
        <div style='font-size:2.5rem;'>🔥</div>
        <div style='font-family:Space Grotesk,sans-serif; font-size:1.4rem;
                    font-weight:700; color:#e94560;'>WILDFIRE AI</div>
        <div style='text-align:center; color:#6b7280; font-size:0.75rem;
                    letter-spacing:1px; text-transform:uppercase;'>Detection System v1.0</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    page = st.radio(
        label            = "Navigation",
        options          = ["🏠 Home", "🔍 Live Detection", "📊 Model Metrics",
                            "📄 Report Generator", "📁 Dataset Explorer"],
        label_visibility = "collapsed"
    )

    st.divider()

    st.markdown("<p style='color:#9ca3af;font-size:0.8rem;text-transform:uppercase;"
            "letter-spacing:1px;margin-bottom:8px;'>🔑 API Configuration</p>",
            unsafe_allow_html=True)

# Auto-load from environment or use default
default_key = os.environ.get('GROQ_API_KEY', '')

groq_key = st.text_input(
    label      = "Groq API Key",
    type       = "password",
    value      = "",  # Don't show the key
    placeholder= "gsk_... paste your key",
    help       = "Get free key at console.groq.com"
)

# If no key entered but we have a default from env, use it
if not groq_key and default_key:
    groq_key = default_key

KEY_READY = False
if groq_key and groq_key.startswith("gsk_"):
    st.success("✅ Key Connected")
    KEY_READY = True
elif groq_key:
    st.error("❌ Invalid key format")
else:
    st.warning("⚠️ Add key for PDF reports")
    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Best Acc", "97.22%")
        st.metric("Models",   "3")
    with c2:
        st.metric("AUC",    "0.995")
        st.metric("Images", "21K")

    st.divider()
    st.markdown("""
    <div style='text-align:center;color:#374151;font-size:0.72rem;padding:8px 0;'>
        © 2026 Wildfire Detection AI<br>PyTorch · Streamlit · Groq
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# LOAD MODELS AT STARTUP
# ═══════════════════════════════════════════════════════════════
all_models, load_errors = load_all_models()

# Show load status in sidebar
with st.sidebar:
    st.divider()
    st.markdown("<p style='color:#9ca3af;font-size:0.8rem;text-transform:uppercase;"
                "letter-spacing:1px;margin-bottom:8px;'>⚙️ Model Status</p>",
                unsafe_allow_html=True)
    for name in MODEL_PATHS:
        if all_models.get(name) is not None:
            st.markdown(f"<p style='color:#22c55e;font-size:0.8rem;margin:2px 0;'>✅ {name}</p>",
                        unsafe_allow_html=True)
        else:
            err = load_errors.get(name, "Unknown error")
            st.markdown(f"<p style='color:#ef4444;font-size:0.8rem;margin:2px 0;'>❌ {name}</p>",
                        unsafe_allow_html=True)
            st.markdown(f"<p style='color:#9ca3af;font-size:0.7rem;margin:0 0 4px 8px;"
                        f"word-break:break-all;'>{err[:80]}</p>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE: HOME
# ═══════════════════════════════════════════════════════════════
if page == "🏠 Home":

    st.markdown("<h1 class='main-title'>🔥 WILDFIRE DETECTION SYSTEM</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>AI-Powered Satellite Image Analysis</p>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, val, label in [
        (c1, "97.22%", "Best Accuracy"),
        (c2, "0.995",  "AUC Score"),
        (c3, "21,067", "Training Images"),
        (c4, "<1 sec", "Detection Speed"),
    ]:
        col.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{val}</div>
            <div class='metric-label'>{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>🔬 How It Works</div>", unsafe_allow_html=True)

    s1, s2, s3, s4 = st.columns(4)
    for col, icon, title, desc in [
        (s1, "📁", "Upload",  "Add your satellite image"),
        (s2, "🤖", "Analyse", "3 AI models run in parallel"),
        (s3, "🗳️", "Vote",   "Ensemble majority decision"),
        (s4, "📄", "Report",  "Download 8-page PDF"),
    ]:
        col.markdown(f"""
        <div class='step-card'>
            <div style='font-size:2rem;margin-bottom:8px;'>{icon}</div>
            <div style='font-weight:700;color:white;font-size:0.95rem;'>{title}</div>
            <div style='color:#9ca3af;font-size:0.8rem;margin-top:4px;'>{desc}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>🤖 Models Used</div>", unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    for col, name, acc, color, badge in [
        (m1, "MobileNetV2",     "95.76%", "#06b6d4", "⚡ Fast & Efficient"),
        (m2, "MobileNetV3",     "97.22%", "#e94560", "🏆 Best Performer"),
        (m3, "EfficientNet-B0", "97.03%", "#80ffdb", "🎯 High Precision"),
    ]:
        col.markdown(f"""
        <div class='model-card'>
            <div class='model-name'>{name}</div>
            <div class='model-acc' style='color:{color};'>{acc}</div>
            <div style='font-size:0.78rem;color:#6b7280;margin-top:6px;'>{badge}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>🛠 Tech Stack</div>", unsafe_allow_html=True)
    techs = ["PyTorch","Streamlit","Plotly","Groq LLaMA3","Albumentations",
             "Folium","OpenCV","ReportLab","MobileNetV3","EfficientNet","Grad-CAM","scikit-learn"]
    st.markdown("".join([f"<span class='tech-badge'>{t}</span>" for t in techs]),
                unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class='info-box'>
        🎯 <b>About this project:</b> This AI system detects wildfires from satellite imagery
        using an ensemble of 3 deep learning models achieving <b>97.22% accuracy</b> and
        <b>0.995 AUC score</b> on 21,067 satellite images.
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE: LIVE DETECTION
# ═══════════════════════════════════════════════════════════════
elif page == "🔍 Live Detection":

    st.markdown("<h2 class='section-header'>🔍 Live Wildfire Detection</h2>", unsafe_allow_html=True)

    uploaded = st.file_uploader(
        label="Upload a satellite image",
        type =["jpg", "jpeg", "png"],
        help ="Upload any satellite image to detect wildfire"
    )

    if uploaded is not None:
        img_pil = Image.open(uploaded).convert("RGB")

        col_img, col_info = st.columns([1, 1])
        with col_img:
            st.image(img_pil, caption=f"📸 {uploaded.name}", use_column_width=True)
        with col_info:
            st.markdown(f"""
            <div class='metric-card' style='text-align:left;'>
                <p style='color:#9ca3af;margin:0 0 12px 0;font-size:0.8rem;
                          text-transform:uppercase;letter-spacing:1px;'>📋 Image Info</p>
                <p style='margin:6px 0;'><span style='color:#6b7280;'>Name:</span>
                   <span style='color:white;'>{uploaded.name}</span></p>
                <p style='margin:6px 0;'><span style='color:#6b7280;'>Size:</span>
                   <span style='color:white;'>{img_pil.width} × {img_pil.height} px</span></p>
                <p style='margin:6px 0;'><span style='color:#6b7280;'>Format:</span>
                   <span style='color:white;'>{uploaded.type.split("/")[1].upper()}</span></p>
                <p style='margin:6px 0;'><span style='color:#6b7280;'>File size:</span>
                   <span style='color:white;'>{len(uploaded.getvalue())/1024:.1f} KB</span></p>
                <p style='margin:6px 0;'><span style='color:#6b7280;'>Device:</span>
                   <span style='color:white;'>{str(DEVICE).upper()}</span></p>
            </div>""", unsafe_allow_html=True)

        # Run predictions
        with st.spinner("🤖 AI models analysing image..."):
            t_start   = time.time()
            results   = predict_image(all_models, img_pil)
            t_elapsed = (time.time() - t_start) * 1000

        # Show any model errors prominently
        errors_found = {n: r["error"] for n, r in results.items() if r["error"]}
        if errors_found:
            st.markdown("<div class='section-header'>⚠️ Model Load / Inference Errors</div>",
                        unsafe_allow_html=True)
            for name, err in errors_found.items():
                st.markdown(f"""
                <div class='debug-box'>
                    ❌ <b>{name}</b>: {err}
                </div>""", unsafe_allow_html=True)
            st.warning("One or more models failed. Check that `efficientnet_pytorch` is installed "
                       "(`pip install efficientnet_pytorch`) and that model .pth files exist in the "
                       "`models/` folder.")

        # Only use successfully loaded models for ensemble
        valid_results = {n: r for n, r in results.items() if r["error"] is None}

        if not valid_results:
            st.error("❌ No models loaded successfully. Cannot make a prediction. "
                     "Check the Model Status panel in the sidebar for error details.")
            st.stop()

        votes    = [v["prediction"] for v in valid_results.values()]
        ensemble = max(set(votes), key=votes.count)
        avg_conf = float(np.mean([v["confidence"] for v in valid_results.values()]))
        risk     = "HIGH 🔴" if ensemble == "wildfire" else "LOW 🟢"
        is_fire  = ensemble == "wildfire"

        # Ensemble result banner
        st.markdown("<br>", unsafe_allow_html=True)
        css_class = "result-fire" if is_fire else "result-safe"
        color     = "#ef4444"    if is_fire else "#22c55e"
        label     = "⚠️ WILDFIRE DETECTED" if is_fire else "✅ NO WILDFIRE DETECTED"

        st.markdown(f"""
        <div class='{css_class}'>
            <div class='result-label' style='color:{color};'>
                {"🔥" if is_fire else "✅"} {label}
            </div>
            <div style='color:#d1d5db;font-size:1rem;margin-top:8px;'>
                Average Confidence: <b>{avg_conf:.1f}%</b>
                &nbsp;|&nbsp; Risk Level: <b>{risk}</b>
                &nbsp;|&nbsp; Speed: <b>{t_elapsed:.0f}ms</b>
                &nbsp;|&nbsp; Valid Models: <b>{len(valid_results)}/3</b>
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Individual model prediction cards
        st.markdown("<div class='section-header'>🤖 Individual Model Predictions</div>",
                    unsafe_allow_html=True)
        cols = st.columns(3)
        for col, (name, res) in zip(cols, results.items()):
            if res["error"]:
                col.markdown(f"""
                <div class='model-card'>
                    <div class='model-name'>{name}</div>
                    <div style='color:#ef4444;font-size:0.9rem;margin-top:8px;'>❌ Load Failed</div>
                    <div style='color:#6b7280;font-size:0.7rem;margin-top:4px;
                                word-break:break-all;'>{res["error"][:60]}</div>
                </div>""", unsafe_allow_html=True)
            else:
                pred_color = "#ef4444" if res["prediction"] == "wildfire" else "#22c55e"
                pred_emoji = "🔥" if res["prediction"] == "wildfire" else "✅"
                nw = res["probs"].get("nowildfire", 0)
                wf = res["probs"].get("wildfire",   0)
                col.markdown(f"""
                <div class='model-card'>
                    <div class='model-name'>{name}</div>
                    <div style='font-size:1.4rem;font-weight:700;color:{pred_color};
                                font-family:Space Grotesk;'>
                        {pred_emoji} {res["prediction"].upper()}
                    </div>
                    <div style='font-size:1.1rem;color:white;font-weight:600;margin-top:6px;'>
                        Confidence: {res["confidence"]:.1f}%
                    </div>
                    <div style='font-size:0.75rem;color:#9ca3af;margin-top:4px;'>
                        🟢 No Fire: {nw:.1f}% &nbsp;|&nbsp; 🔥 Fire: {wf:.1f}%
                    </div>
                    <div style='font-size:0.72rem;color:#6b7280;margin-top:4px;'>
                        Val Acc: {MODEL_STATS[name]["val_acc"]}%
                    </div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Charts
        st.markdown("<div class='section-header'>📊 Live Analysis Charts</div>",
                    unsafe_allow_html=True)

        ch1, ch2 = st.columns(2)
        with ch1:
            st.plotly_chart(chart_confidence_gauge(avg_conf, ensemble),
                            use_container_width=True, config={"displayModeBar": False})
        with ch2:
            st.plotly_chart(chart_model_bars(valid_results),
                            use_container_width=True, config={"displayModeBar": True})

        # Probability breakdown
        st.plotly_chart(chart_prob_breakdown(valid_results),
                        use_container_width=True, config={"displayModeBar": True})

        # Grad-CAM
        st.markdown("<div class='section-header'>🔥 Grad-CAM Explainability</div>",
                    unsafe_allow_html=True)

        best_model_obj = all_models.get("MobileNetV3")
        cam_map        = None
        if best_model_obj is not None:
            with st.spinner("Generating Grad-CAM..."):
                cam_map = get_gradcam(best_model_obj, img_pil, "MobileNetV3")

        gc1, gc2 = st.columns(2)
        with gc1:
            if cam_map is not None:
                img_resized = np.array(img_pil.convert("RGB").resize((IMG_SIZE, IMG_SIZE)))
                heatmap     = cv2.applyColorMap(np.uint8(255 * cam_map), cv2.COLORMAP_JET)
                heatmap     = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                overlay     = 0.6 * img_resized.astype(np.float32) + 0.4 * heatmap.astype(np.float32)
                overlay     = np.clip(overlay, 0, 255).astype(np.uint8)
                st.image(overlay, caption="🔥 Grad-CAM Heatmap Overlay", use_column_width=True)
            else:
                st.info("Grad-CAM not available for this model")

        with gc2:
            fig_3d = chart_gradcam_3d(cam_map)
            if fig_3d is not None:
                st.plotly_chart(fig_3d, use_container_width=True, config={"displayModeBar": True})

        # RGB Histogram
        st.markdown("<div class='section-header'>🌈 Image Pixel Analysis</div>",
                    unsafe_allow_html=True)
        st.plotly_chart(chart_rgb_histogram(img_pil),
                        use_container_width=True, config={"displayModeBar": True})

        # Save to session
        st.session_state["last_image"]    = img_pil
        st.session_state["last_results"]  = results
        st.session_state["last_ensemble"] = ensemble
        st.session_state["last_avg_conf"] = avg_conf
        st.session_state["last_risk"]     = risk

    else:
        st.markdown("""
        <div style='text-align:center;padding:60px 20px;color:#6b7280;'>
            <div style='font-size:4rem;margin-bottom:16px;'>🛰️</div>
            <div style='font-size:1.2rem;font-weight:600;color:#9ca3af;'>
                Upload a satellite image to begin detection
            </div>
            <div style='font-size:0.9rem;margin-top:8px;'>Supports JPG · PNG · JPEG</div>
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE: MODEL METRICS
# ═══════════════════════════════════════════════════════════════
elif page == "📊 Model Metrics":

    st.markdown("<h2 class='section-header'>📊 Model Performance Metrics</h2>",
                unsafe_allow_html=True)

    mc1, mc2, mc3, mc4 = st.columns(4)
    for col, val, lbl in [
        (mc1, "96.58%", "Test Accuracy"),
        (mc2, "0.995",  "AUC Score"),
        (mc3, "97.22%", "Best Val Acc"),
        (mc4, "3,161",  "Test Samples"),
    ]:
        col.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{val}</div>
            <div class='metric-label'>{lbl}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    fig_cm = chart_confusion_matrix_3d()
    if fig_cm:
        st.plotly_chart(fig_cm, use_container_width=True, config={"displayModeBar": True})
    else:
        st.info("confusion_matrix.csv not found in results/ folder")

    col_roc, col_rad = st.columns(2)
    with col_roc:
        fig_roc = chart_roc_curve()
        if fig_roc:
            st.plotly_chart(fig_roc, use_container_width=True, config={"displayModeBar": True})
        else:
            st.info("test_predictions.csv not found")
    with col_rad:
        fig_radar = chart_radar()
        if fig_radar:
            st.plotly_chart(fig_radar, use_container_width=True, config={"displayModeBar": True})
        else:
            st.info("classification_report.csv not found")

    fig_sun = chart_sunburst()
    if fig_sun:
        st.plotly_chart(fig_sun, use_container_width=True, config={"displayModeBar": True})

    

    report_path = os.path.join(RESULTS_PATH, "classification_report.csv")
    if os.path.exists(report_path):
        st.markdown("<div class='section-header'>📋 Classification Report</div>",
                    unsafe_allow_html=True)
        df_report = pd.read_csv(report_path, index_col=0)
        st.dataframe(df_report.style.background_gradient(cmap="RdYlGn", axis=None).format("{:.3f}"),
                     use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# PAGE: REPORT GENERATOR
# ═══════════════════════════════════════════════════════════════
elif page == "📄 Report Generator":

    st.markdown("<h2 class='section-header'>📄 AI Report Generator</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div class='info-box'>
        🤖 <b>Powered by Groq LLaMA3-70B</b> — Upload a satellite image, run predictions,
        then generate a professional 8-page PDF report. Add your Groq API key in the sidebar first.
    </div>""", unsafe_allow_html=True)

    uploaded_r = st.file_uploader(
        label="Upload satellite image for report",
        type=["jpg", "jpeg", "png"],
        key="report_upload"
    )

    if uploaded_r is not None:
        # ===== DEBUG CODE =====
        st.write("=== API KEY DEBUG ===")
        st.write(f"groq_key exists: {bool(groq_key)}")
        st.write(f"groq_key length: {len(groq_key) if groq_key else 0}")
        st.write(f"KEY_READY: {KEY_READY}")
        st.write(f"GROQ_API_KEY from env: {'SET' if os.environ.get('GROQ_API_KEY') else 'NOT SET'}")
        if groq_key:
            st.write(f"First 5 chars of key: {groq_key[:5]}...")
            st.write(f"Starts with gsk_: {groq_key.startswith('gsk_')}")
        # ===== End DEBUG =====
        
        img_pil_r = Image.open(uploaded_r).convert("RGB")
        st.image(img_pil_r, caption=f"📸 {uploaded_r.name}", use_column_width=True)

        with st.spinner("Running AI predictions..."):
            results_r = predict_image(all_models, img_pil_r)
            valid_r = {n: r for n, r in results_r.items() if r["error"] is None}
            votes_r = [v["prediction"] for v in valid_r.values()]
            ensemble_r = max(set(votes_r), key=votes_r.count) if votes_r else "unknown"
            avg_conf_r = float(np.mean([v["confidence"] for v in valid_r.values()])) if valid_r else 0.0
            risk_r = "HIGH 🔴" if ensemble_r == "wildfire" else "LOW 🟢"

        is_fire_r = ensemble_r == "wildfire"
        color_r = "#ef4444" if is_fire_r else "#22c55e"
        css_r = "result-fire" if is_fire_r else "result-safe"

        st.markdown(f"""
        <div class='{css_r}' style='margin:16px 0;'>
            <div class='result-label' style='color:{color_r};'>
                {"🔥 WILDFIRE DETECTED" if is_fire_r else "✅ NO WILDFIRE DETECTED"}
            </div>
            <div style='color:#d1d5db;margin-top:8px;'>
                Confidence: <b>{avg_conf_r:.1f}%</b> &nbsp;|&nbsp; Risk: <b>{risk_r}</b>
            </div>
        </div>""", unsafe_allow_html=True)

        # Test button for API key
        if st.button("🔑 Test Groq API Key"):
            try:
                from groq import Groq
                test_client = Groq(api_key=groq_key)
                test_response = test_client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": "Say OK"}],
                    max_tokens=5
                )
                st.success("✅ API Key is WORKING!")
                st.write(f"Response: {test_response.choices[0].message.content}")
            except Exception as e:
                st.error(f"❌ API Error: {e}")

        if not KEY_READY:
            st.warning("⚠️ Please add your Groq API key in the sidebar to generate the report.")
        else:
            if st.button("🚀 Generate 8-Page PDF Report"):
                pdf_bytes = generate_pdf_report(
                    groq_key=groq_key,
                    img_pil=img_pil_r,
                    predictions=results_r,
                    ensemble=ensemble_r,
                    avg_conf=avg_conf_r,
                    risk_level=risk_r
                )
                if pdf_bytes:
                    fname = f"Wildfire_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                    st.download_button(
                        label="⬇️ Download PDF Report",
                        data=pdf_bytes,
                        file_name=fname,
                        mime="application/pdf"
                    )
                    st.success(f"✅ Report ready: {fname}")
    else:
        st.markdown("""
        <div style='text-align:center;padding:60px 20px;color:#6b7280;'>
            <div style='font-size:4rem;'>📄</div>
            <div style='font-size:1.1rem;color:#9ca3af;margin-top:12px;'>
                Upload an image to generate your report
            </div>
        </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# PAGE: DATASET EXPLORER
# ═══════════════════════════════════════════════════════════════
elif page == "📁 Dataset Explorer":

    st.markdown("<h2 class='section-header'>📁 Dataset Explorer</h2>", unsafe_allow_html=True)

    dc1, dc2 = st.columns(2)
    with dc1:
        st.markdown("""
        <div class='metric-card' style='text-align:left;'>
            <p style='color:#9ca3af;font-size:0.8rem;text-transform:uppercase;letter-spacing:1px;'>
                📊 Dataset Statistics</p>
            <table style='width:100%;color:#d1d5db;font-size:0.95rem;'>
                <tr><td style='padding:6px 0;color:#6b7280;'>Total Images</td>
                    <td style='color:white;font-weight:600;'>21,067</td></tr>
                <tr><td style='padding:6px 0;color:#6b7280;'>No Wildfire</td>
                    <td style='color:#22c55e;font-weight:600;'>14,500</td></tr>
                <tr><td style='padding:6px 0;color:#6b7280;'>Wildfire</td>
                    <td style='color:#ef4444;font-weight:600;'>6,567</td></tr>
                <tr><td style='padding:6px 0;color:#6b7280;'>Train Set</td>
                    <td style='color:white;font-weight:600;'>14,746</td></tr>
                <tr><td style='padding:6px 0;color:#6b7280;'>Val Set</td>
                    <td style='color:white;font-weight:600;'>3,160</td></tr>
                <tr><td style='padding:6px 0;color:#6b7280;'>Test Set</td>
                    <td style='color:white;font-weight:600;'>3,161</td></tr>
            </table>
        </div>""", unsafe_allow_html=True)

    with dc2:
        fig_dist = go.Figure(data=[go.Bar(
            x      = ["No Wildfire", "Wildfire"],
            y      = [14500, 6567],
            marker = dict(color=["#22c55e", "#ef4444"], opacity=0.85,
                          line=dict(color=["#22c55e", "#ef4444"], width=2)),
            text         = ["14,500\n(68.8%)", "6,567\n(31.2%)"],
            textposition = "outside",
            textfont     = dict(color="white", size=13),
        )])
        fig_dist.update_layout(**CHART_LAYOUT,
                               title=dict(text="📊 Class Distribution",
                                          font=dict(size=15, color="white")),
                               height=320, yaxis=dict(range=[0, 18000]))
        st.plotly_chart(fig_dist, use_container_width=True, config={"displayModeBar": False})

    sample_path = os.path.join(ASSETS_PATH, "sample_images.png")
    if os.path.exists(sample_path):
        st.markdown("<div class='section-header'>🖼️ Sample Images</div>", unsafe_allow_html=True)
        st.image(sample_path, caption="Sample satellite images from dataset",
                 use_column_width=True)

    aug_path = os.path.join(RESULTS_PATH, "augmentation_demo.png")
    if os.path.exists(aug_path):
        st.markdown("<div class='section-header'>🔧 Augmentation Demo</div>", unsafe_allow_html=True)
        st.image(aug_path, caption="Original vs Augmented images", use_column_width=True)

    st.markdown("<div class='section-header'>🏆 Model Leaderboard</div>", unsafe_allow_html=True)
    board_df = pd.DataFrame([
        ("🥇", "MobileNetV3",     "97.22%", "97.15%", "23.5 mins"),
        ("🥈", "EfficientNet-B0", "97.03%", "96.74%", "24.2 mins"),
        ("🥉", "MobileNetV2",     "95.76%", "95.63%", "24.0 mins"),
    ], columns=["Rank", "Model", "Best Val Acc", "Final Val Acc", "Training Time"])
    st.dataframe(board_df.style.set_properties(
        **{"background-color": "#1a1a2e", "color": "white"}),
        use_container_width=True, hide_index=True)

    gc_path = os.path.join(RESULTS_PATH, "gradcam_heatmaps.png")
    if os.path.exists(gc_path):
        st.markdown("<div class='section-header'>🔥 Grad-CAM Results</div>", unsafe_allow_html=True)
        st.image(gc_path, caption="Grad-CAM: Correct vs Wrong Predictions",
                 use_column_width=True)

    for fname, title in [
        ("top5_correct.png", "✅ Top 5 Most Confident Correct"),
        ("top5_wrong.png",   "❌ Top 5 Most Confident Wrong"),
    ]:
        fpath = os.path.join(RESULTS_PATH, fname)
        if os.path.exists(fpath):
            st.markdown(f"<div class='section-header'>{title}</div>", unsafe_allow_html=True)
            st.image(fpath, use_column_width=True)