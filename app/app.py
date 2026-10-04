import os
import io
import h5py
import numpy as np
from PIL import Image

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
import timm

from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# -----------------------------------------------------------------------------
# Page
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="SugarScan | Sugarcane Leaf Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
#MainMenu, footer, header {visibility:hidden;}
.stApp {background:#f4fbf6;}
.block-container {padding-top:0.7rem; padding-bottom:0.6rem; max-width:1280px;}
section[data-testid="stSidebar"] {background:linear-gradient(180deg,#123524,#1f5b3f);}
section[data-testid="stSidebar"] * {color:#9fa2a0 !important;}
.title-card{background:linear-gradient(135deg,#184d35,#52b788); padding:.7rem 1rem; border-radius:18px; color:white; margin-bottom:.7rem;}
.title-card h1{font-size:1.6rem; margin:0; font-weight:800;}
.title-card p{margin:.2rem 0 0; color:#d8f3dc; font-size:.85rem;}
.small-card{background:white; border:1px solid #d8f3dc; border-radius:16px; padding:.75rem; box-shadow:0 4px 18px rgba(0,0,0,.05); margin-bottom:.65rem;}
.section-label{font-size:.74rem; text-transform:uppercase; letter-spacing:.08em; font-weight:800; color:#2d6a4f; margin-bottom:.45rem;}
.pred-box{border:2px solid #2d6a4f; background:#eaf8ee; border-radius:16px; padding:.8rem; text-align:center;}
.pred-disease{border-color:#c1121f; background:#ffe8e8;}
.pred-class{font-size:1.55rem; font-weight:800; color:#123524; margin:.05rem 0;}
.conf{font-size:2.25rem; font-weight:900; color:#184d35; line-height:1.1;}
.info{background:#f6fff8; border-left:4px solid #52b788; padding:.55rem .7rem; border-radius:8px; color:#1f5b3f; font-size:.85rem; margin-top:.55rem;}
.cam-caption{text-align:center; font-weight:800; color:#1f5b3f; background:#d8f3dc; border-radius:8px; padding:.22rem; font-size:.7rem; margin-top:.2rem;}
.warn{background:#fff3cd; padding:.7rem 1rem; border-radius:10px; color:#7a4d00; font-size:.9rem;}

/* Compact fixed image display */
[data-testid="stImage"] img{
    max-height:250px;
    object-fit:contain;
}

/* CAM row smaller */
.cam-section [data-testid="stImage"] img{
    max-height:180px;
    object-fit:contain;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Constants
# IMPORTANT: class order must be exactly ImageFolder dataset.classes from training
# -----------------------------------------------------------------------------
CLASS_NAMES = [
    "Banded Chlorosis", "Brown Spot", "BrownRust", "Dried Leaves",
    "Grassy Shoot", "Healthy Leaves", "Mosaic", "Pokkah Boeng",
    "RedRot", "Rust", "Sett Rot", "Viral Disease", "Yellow Leaf", "Smut"
]

DISEASE_INFO = {
    "Banded Chlorosis":"Nutrient deficiency causing yellow banding across leaf blades.",
    "Brown Spot":"Fungal infection producing brown oval lesions on leaves.",
    "BrownRust":"Orange-brown rust pustules scattered on the leaf surface.",
    "Dried Leaves":"Drying/desiccation symptoms due to stress or damaged leaf tissue.",
    "Grassy Shoot":"Phytoplasma infection causing weak, grassy tillering.",
    "Healthy Leaves":"No visible disease symptom detected.",
    "Mosaic":"Sugarcane mosaic symptom with yellow-green mottling.",
    "Pokkah Boeng":"Fusarium-related twisted or malformed young leaves.",
    "RedRot":"Red rot disease, commonly associated with stalk/leaf symptoms.",
    "Rust":"Yellow-orange rust pustules on the leaf surface.",
    "Sett Rot":"Seed-piece/sett rot symptoms caused by fungal infection.",
    "Viral Disease":"General viral symptom causing chlorosis or stunting.",
    "Yellow Leaf":"Sugarcane yellow leaf symptom; mid-rib yellowing can progress outward.",
    "Smut":"Smut disease, often associated with black whip-like fungal growth.",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = len(CLASS_NAMES)

# -----------------------------------------------------------------------------
# Architecture EXACTLY matching notebook: sugarcane-leaf-image-se-cbam-block.ipynb
# -----------------------------------------------------------------------------
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        scale = self.sigmoid(avg_out + max_out).unsqueeze(2).unsqueeze(3)
        return x * scale.expand_as(x)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        scale = torch.cat([avg_out, max_out], dim=1)
        scale = self.sigmoid(self.conv(scale))
        return x * scale.expand_as(x)

class CBAMBlock(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class XceptionWithAttention(nn.Module):
    def __init__(self, num_classes, attention="cbam"):
        super().__init__()
        base = timm.create_model("xception", pretrained=False, num_classes=0)
        self.backbone = base
        in_features = base.num_features
        self.attention = SEBlock(in_features) if attention == "se" else CBAMBlock(in_features)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(in_features, num_classes))
    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = self.attention(x)
        x = self.pool(x)
        return self.classifier(x)

class InceptionV3WithAttention(nn.Module):
    def __init__(self, num_classes, attention="cbam"):
        super().__init__()
        base = models.inception_v3(weights=None, aux_logits=True)
        base.aux_logits = False
        base.AuxLogits = None
        in_features = 2048
        # Name must be inception_blocks, not blocks. Otherwise 567 keys will not match.
        self.inception_blocks = nn.Sequential(
            base.Conv2d_1a_3x3, base.Conv2d_2a_3x3, base.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            base.Conv2d_3b_1x1, base.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            base.Mixed_5b, base.Mixed_5c, base.Mixed_5d,
            base.Mixed_6a, base.Mixed_6b, base.Mixed_6c, base.Mixed_6d, base.Mixed_6e,
            base.Mixed_7a, base.Mixed_7b, base.Mixed_7c,
        )
        self.attention = SEBlock(in_features) if attention == "se" else CBAMBlock(in_features)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(p=0.3), nn.Linear(in_features, num_classes))
    def forward(self, x):
        x = self.inception_blocks(x)
        x = self.attention(x)
        x = self.pool(x)
        return self.classifier(x)

class EnsembleModel(nn.Module):
    # Notebook used 0.6 and 0.4, not 0.5 and 0.5
    def __init__(self, model_a, model_b, weight_a=0.6, weight_b=0.4):
        super().__init__()
        self.model_a = model_a
        self.model_b = model_b
        self.weight_a = weight_a
        self.weight_b = weight_b
    def forward(self, x):
        out_a = torch.softmax(self.model_a(x), dim=1)
        out_b = torch.softmax(self.model_b(x), dim=1)
        return (self.weight_a * out_a) + (self.weight_b * out_b)

# -----------------------------------------------------------------------------
# Loading helpers
# -----------------------------------------------------------------------------
def _read_flat_h5(h5_path):
    state_dict = {}
    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            key = name.replace("/", ".").replace("|", ".")
            if key.startswith("model_a.") or key.startswith("model_b."):
                key = key.split(".", 1)[1]
            state_dict[key] = torch.tensor(obj[()])
    with h5py.File(h5_path, "r") as f:
        f.visititems(visit)
    return state_dict

def load_h5_weights(model, h5_path):
    state_dict = _read_flat_h5(h5_path)
    model_state = model.state_dict()
    matched = {k:v for k,v in state_dict.items() if k in model_state and tuple(v.shape) == tuple(model_state[k].shape)}
    missing = [k for k in model_state.keys() if k not in matched]
    extra = [k for k in state_dict.keys() if k not in model_state]
    model.load_state_dict(matched, strict=False)
    return model.to(DEVICE).eval(), len(matched), len(missing), len(extra)

@st.cache_resource(show_spinner=False)
def load_model(path_xception, path_inception):
    xc = XceptionWithAttention(NUM_CLASSES, "cbam")
    ic = InceptionV3WithAttention(NUM_CLASSES, "cbam")
    xc, m1, miss1, ex1 = load_h5_weights(xc, path_xception)
    ic, m2, miss2, ex2 = load_h5_weights(ic, path_inception)
    return EnsembleModel(xc, ic, 0.6, 0.4).to(DEVICE).eval(), (m1, miss1, ex1, m2, miss2, ex2)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🌿 SugarScan")
    st.caption("Sugarcane Leaf Disease Detection")
    st.divider()
    st.markdown("**Model paths**")
    path_xc = st.text_input("Xception CBAM", "fixed_xception_cbam.h5")
    path_ic = st.text_input("InceptionV3 CBAM", "fixed_inception_cbam.h5")
    show_debug = st.checkbox("Show loading debug", value=False)
    st.divider()
    uploaded_file = st.file_uploader("Upload leaf image", type=["jpg", "jpeg", "png"])
    st.divider()
    st.info("Best model: Ensemble\n\nXception(CW+CBAM) + InceptionV3(CW+CBAM)\n\nAccuracy: 97.20%")

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.markdown("""
<div class='title-card'>
  <h1>🌿 SugarScan</h1>
  <p>Explainable Sugarcane Leaf Disease Detection using Ensemble Xception + InceptionV3 with CBAM</p>
</div>
""", unsafe_allow_html=True)

if not uploaded_file:
    a,b,c = st.columns(3)
    a.markdown("<div class='small-card'><div class='section-label'>Model</div><b>Ensemble</b><br>Xception + InceptionV3</div>", unsafe_allow_html=True)
    b.markdown("<div class='small-card'><div class='section-label'>Attention</div><b>CBAM</b><br>Class-weight trained</div>", unsafe_allow_html=True)
    c.markdown("<div class='small-card'><div class='section-label'>Result</div><b>97.20%</b><br>Test Accuracy</div>", unsafe_allow_html=True)
    st.markdown("### Detectable classes")
    st.write(" · ".join(CLASS_NAMES))
    st.stop()

# -----------------------------------------------------------------------------
# Prediction
# -----------------------------------------------------------------------------
try:
    with st.spinner("Loading model weights..."):
        ensemble, debug = load_model(path_xc, path_ic)
except Exception as e:
    st.error(f"Model load failed: {e}")
    st.stop()

if show_debug:
    m1, miss1, ex1, m2, miss2, ex2 = debug
    st.markdown(f"<div class='warn'>Xception matched: {m1}, missing: {miss1}, extra: {ex1}<br>InceptionV3 matched: {m2}, missing: {miss2}, extra: {ex2}</div>", unsafe_allow_html=True)

img = Image.open(uploaded_file).convert("RGB")
tensor = transform(img).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    probs = ensemble(tensor).cpu().numpy()[0]
idx = int(np.argmax(probs))
pred_class = CLASS_NAMES[idx]
conf = float(probs[idx] * 100.0)
is_healthy = pred_class == "Healthy Leaves"

left, right = st.columns([1.05, .95], gap="large")
with left:
    st.markdown("<div class='small-card'><div class='section-label'>Input Image</div>", unsafe_allow_html=True)
    st.image(img, width=500)
    st.markdown("</div>", unsafe_allow_html=True)
with right:
    st.markdown("<div class='small-card'><div class='section-label'>Prediction</div>", unsafe_allow_html=True)
    extra_cls = "" if is_healthy else " pred-disease"
    status = "No Disease Detected" if is_healthy else "Disease Detected"
    icon = "✅" if is_healthy else "⚠️"
    st.markdown(f"""
    <div class='pred-box{extra_cls}'>
      <div style='font-weight:800;color:#2d6a4f'>{icon} {status}</div>
      <div class='pred-class'>{pred_class}</div>
      <div style='margin-top:.45rem;color:#40916c;font-weight:800;font-size:.7rem;text-transform:uppercase;letter-spacing:.08em'>Confidence</div>
      <div class='conf'>{conf:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"<div class='info'>📌 {DISEASE_INFO.get(pred_class, '')}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# XAI - use the last convolutional stage of Xception, better than attention block
# -----------------------------------------------------------------------------
st.markdown("<div class='small-card cam-section' style='margin-top:.45rem'><div class='section-label'>Grad-CAM Explainability</div>", unsafe_allow_html=True)
with st.spinner("Generating heatmaps..."):
    rgb = np.array(img.resize((224,224))).astype(np.float32) / 255.0
    xception = ensemble.model_a
    xception.eval()
    # More stable CAM target than the attention block
    target_layers = [xception.backbone.conv4]
    targets = [ClassifierOutputTarget(idx)]
    cams = []
    for label, CamClass in [("Grad-CAM", GradCAM), ("Grad-CAM++", GradCAMPlusPlus), ("Eigen-CAM", EigenCAM)]:
        cam = CamClass(model=xception, target_layers=target_layers)
        grayscale = cam(input_tensor=tensor, targets=targets)[0]
        vis = show_cam_on_image(rgb, grayscale, use_rgb=True)
        cams.append((label, vis))

cols = st.columns(4)
with cols[0]:
    st.image(img.resize((224,224)), width=220)
    st.markdown("<div class='cam-caption'>Original</div>", unsafe_allow_html=True)
for col, (label, vis) in zip(cols[1:], cams):
    with col:
        st.image(vis, width=220)
        st.markdown(f"<div class='cam-caption'>{label}</div>", unsafe_allow_html=True)
st.markdown(f"<div class='info'>🧪 Predicted: <b>{pred_class}</b> · Confidence: <b>{conf:.2f}%</b> · Model: Ensemble (Xception + InceptionV3, CW + CBAM)</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
