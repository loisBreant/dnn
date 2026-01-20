import streamlit as st
import sys
import os
import cv2
import numpy as np
import torch
from PIL import Image
import pandas as pd

# Add parent directory to path to import study module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from study.comparison import enhance_net_nopool, apply_clahe, apply_autogamma, get_metrics

# Model paths
MODEL_PATHS = {
    "Zero-DCE (Exp4)": os.path.join(os.path.dirname(os.path.dirname(__file__)), "snapshots/exp4/Epoch_Final.pth"),
    "Zero-DCE (Exp3)": os.path.join(os.path.dirname(os.path.dirname(__file__)), "snapshots/exp3/Epoch_Final.pth"),
    "Zero-DCE (Paper)": os.path.join(os.path.dirname(os.path.dirname(__file__)), "Zero-DCE/Zero-DCE_code/snapshots/Epoch99.pth")
}

# Page config
st.set_page_config(page_title="Low-Light Enhancer", page_icon="", layout="wide")

# Custom CSS for a modern look
st.markdown(u'''
<style>
    /* Global Styles */
    .reportview-container {
        background: #f0f2f6;
    }
    .main .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Headers */
    h1, h2, h3 {
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        font-weight: 600;
        color: #1E1E1E;
    }
    h1 {
        margin-bottom: 0.5rem;
    }
    
    /* Cards/Containers */
    .stCard {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #2563EB;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background-color: #1D4ED8;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Metrics Table Styling */
    div[data-testid="stDataFrame"] {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #E5E7EB;
    }
    
    /* Image Captions */
    .css-16idsys p {
        text-align: center;
        font-weight: 500;
        color: #4B5563;
        margin-top: 0.5rem;
    }
</style>
''', unsafe_allow_html=True)

# Header Section
st.title(" Low-Light Image Enhancement Studio")
st.markdown(u'''
    <div style='background-color: #EFF6FF; padding: 1rem; border-radius: 8px; border-left: 4px solid #2563EB; margin-bottom: 2rem;'>
        <p style='margin: 0; color: #1E3A8A;'>
            Upload a dark or low-light image to see how different algorithms enhance it. 
            We compare <strong>Gamma Correction</strong>, <strong>CLAHE</strong>, and three variants of <strong>Zero-DCE</strong>.
        </p>
    </div>
''', unsafe_allow_html=True)

# Load Model
@st.cache_resource
def load_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = enhance_net_nopool().to(device)
    if os.path.exists(model_path):
        try:
            # Try with weights_only=True (safer, newer torch versions)
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        except TypeError:
             # Fallback for older torch versions
            model.load_state_dict(torch.load(model_path, map_location=device))
        except RuntimeError as e: # Handle potential key mismatches if models differ slightly in saving (e.g. "module." prefix)
            state_dict = torch.load(model_path, map_location=device)
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k.replace("module.", "") # remove `module.`
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
            
        model.eval()
        return model, device
    else:
        st.error(f" Model weights not found at `{model_path}`. Please ensure the path is correct.")
        return None, None

# Pre-load all models
loaded_models = {}
for name, path in MODEL_PATHS.items():
    m, d = load_model(path)
    if m:
        loaded_models[name] = (m, d)

# Sidebar
with st.sidebar:
    st.header("Input Image")
    input_source = st.radio("Select input source:", ("Upload Image", "Camera"))
    
    uploaded_file = None
    if input_source == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    else:
        uploaded_file = st.camera_input("Take a picture")

    st.markdown("---")
    st.markdown("### About")
    st.info(
        "This demo compares traditional CV techniques with Deep Learning for low-light image enhancement.\n\n"
        "**Techniques:**\n"
        "- **Gamma**: Simple power-law transformation.\n"
        "- **CLAHE**: Contrast Limited Adaptive Histogram Equalization.\n"
        "- **Zero-DCE**: Zero-Reference Deep Curve Estimation (Neural Network).\n\n"
        "**Zero-DCE Models:**\n"
        f"- **Exp4**: Our trained model from Experiment 4.\n"
        f"- **Exp3**: Our trained model from Experiment 3.\n"
        f"- **Paper**: The original model checkpoint from the Zero-DCE paper."
    )


if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)
    
    if img_bgr is not None:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Image resizing logic
        max_dim = 500
        height, width = img_bgr.shape[:2]

        if height > max_dim or width > max_dim:
            scaling_factor = max_dim / max(height, width)
            new_width = int(width * scaling_factor)
            new_height = int(height * scaling_factor)
            
            img_bgr = cv2.resize(img_bgr, (new_width, new_height), interpolation=cv2.INTER_AREA)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) # Re-convert after resizing

            st.sidebar.info(f"Image resized to {new_width}x{new_height} to optimize processing time.")
        
        # Helper to format metrics
        def get_metric_dict(name, img_b):
            ent, grad = get_metrics(img_b)
            return {
                "Method": name,
                "Entropy": ent,
                "Gradient": grad
            }
        
        metrics = []
        metrics.append(get_metric_dict("Original", img_bgr))

        # Setup Layout
        st.subheader("Visual Comparison")
        tab1, tab2 = st.tabs(["Side-by-Side View", "Individual Results"])
        
        # --- TAB 1: SIDE-BY-SIDE ---
        with tab1:
            # Row 1: Original
            st.markdown("#### Original Input")
            st.image(img_rgb, caption="Original", width=400)
            
            st.markdown("---")
            
            # Row 2: Traditional
            st.markdown("#### Traditional Methods")
            col1, col2 = st.columns(2)
            gamma_placeholder = col1.empty()
            clahe_placeholder = col2.empty()
            
            st.markdown("---")
            
            # Row 3: Zero-DCE Models
            st.markdown("#### Zero-DCE Models")
            col_d1, col_d2, col_d3 = st.columns(3)
            dce_placeholders = {
                "Zero-DCE (Exp4)": col_d1.empty(),
                "Zero-DCE (Exp3)": col_d2.empty(),
                "Zero-DCE (Paper)": col_d3.empty()
            }
        
        # --- TAB 2: INDIVIDUAL ---
        with tab2:
            st.subheader("Deep Learning Results")
            for name in MODEL_PATHS.keys():
                st.markdown(f"**{name}**")
                st.empty() # Placeholder if needed, but we'll just show them below
            
            # We'll actually populate this tab dynamically below or just show them in order
            individual_placeholders = {}
            col_ind1, col_ind2 = st.columns(2)
            with col_ind1:
                st.markdown("### Traditional")
                ind_gamma = st.empty()
                ind_clahe = st.empty()
            with col_ind2:
                st.markdown("### Deep Learning")
                for name in MODEL_PATHS.keys():
                    individual_placeholders[name] = st.empty()

        # Metrics Placeholder
        st.markdown("---")
        st.subheader(" Performance Metrics")
        metrics_placeholder = st.empty()
        
        # Download Section
        st.subheader("Downloads")
        download_cols = st.columns(len(MODEL_PATHS))
        download_placeholders = {name: col.empty() for name, col in zip(MODEL_PATHS.keys(), download_cols)}

        # --- PROCESSING ---
        
        # 1. Gamma
        img_gamma_bgr = apply_autogamma(img_bgr)
        img_gamma_rgb = cv2.cvtColor(img_gamma_bgr, cv2.COLOR_BGR2RGB)
        
        gamma_placeholder.image(img_gamma_rgb, caption="Gamma Correction", use_container_width=True)
        ind_gamma.image(img_gamma_rgb, caption="Gamma", use_container_width=True)
        metrics.append(get_metric_dict("Gamma Correction", img_gamma_bgr))

        # 2. CLAHE
        img_clahe_bgr = apply_clahe(img_bgr)
        img_clahe_rgb = cv2.cvtColor(img_clahe_bgr, cv2.COLOR_BGR2RGB)
        
        clahe_placeholder.image(img_clahe_rgb, caption="CLAHE", use_container_width=True)
        ind_clahe.image(img_clahe_rgb, caption="CLAHE", use_container_width=True)
        metrics.append(get_metric_dict("CLAHE", img_clahe_bgr))

        # 3. Zero-DCE Models
        data_lowlight = Image.fromarray(img_rgb)
        data_lowlight = (np.asarray(data_lowlight)/255.0)
        # Assuming all models use the same device
        device = list(loaded_models.values())[0][1] if loaded_models else torch.device('cpu')
        data_lowlight = torch.from_numpy(data_lowlight).float().permute(2,0,1).unsqueeze(0).to(device)
        
        for name, (model, _) in loaded_models.items():
            with torch.no_grad():
                _, enhanced_image, _ = model(data_lowlight)
            
            res_deep = enhanced_image.squeeze().cpu().permute(1, 2, 0).numpy()
            res_deep = np.clip(res_deep * 255, 0, 255).astype('uint8')
            img_deep_rgb = res_deep
            img_deep_bgr = cv2.cvtColor(res_deep, cv2.COLOR_RGB2BGR)
            
            # Update UI - Tab 1
            if name in dce_placeholders:
                dce_placeholders[name].image(img_deep_rgb, caption=name, use_container_width=True)
            
            # Update UI - Tab 2
            if name in individual_placeholders:
                individual_placeholders[name].image(img_deep_rgb, caption=name, use_container_width=True)
                
            # Metrics
            metrics.append(get_metric_dict(name, img_deep_bgr))
            
            # Download
            is_success, buffer = cv2.imencode(".png", img_deep_bgr)
            if is_success:
                safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").lower()
                download_placeholders[name].download_button(
                    label=f"Download {name}",
                    data=buffer.tobytes(),
                    file_name=f"enhanced_{safe_name}.png",
                    mime="image/png"
                )

        # Finalize Metrics Display
        df_metrics = pd.DataFrame(metrics)
        col_m1, col_m2 = st.columns([2, 1])
        
        with metrics_placeholder.container():
            col_m1, col_m2 = st.columns([2, 1])
            with col_m1:
                st.dataframe(
                    df_metrics.style.format(subset=["Entropy", "Gradient"], formatter="{:.4f}")
                    .highlight_max(axis=0, subset=["Entropy", "Gradient"], props='color: white; background-color: #198754; font-weight: bold;')
                    .highlight_min(axis=0, subset=["Entropy", "Gradient"], props='color: white; background-color: #dc3545;')
                    , use_container_width=True
                )
            with col_m2:
                st.markdown(
                    """
                    <div style='font-size: 0.9rem; color: #666;'>
                    <strong>Metrics Explanation:</strong><br>
                    <ul style='padding-left: 1.2rem; margin-top: 0.5rem;'>
                        <li><strong>Entropy:</strong> Measures the amount of information or detail in the image. Higher is usually better for enhancement.</li>
                        <li><strong>Gradient:</strong> Average gradient magnitude. Indicates edge sharpness and texture.</li>
                    </ul>
                    </div>
                    """, unsafe_allow_html=True
                )

    else:
        st.error("Error decoding the image. Please try another file.")
else:
    # Placeholder when no image is uploaded
    st.info("Please upload an image from the sidebar to get started.")
