"""
app.py — Streamlit demonstration application for Member 5 Integration.

Academic research prototype only. NOT a diagnostic tool.

Launch from project root:
    streamlit run member5_integration/app.py

Models are loaded once at startup via st.cache_resource.
No retraining occurs during the app session.
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path
from PIL import Image

import streamlit as st

# ── Resolve paths ──────────────────────────────────────────────────────────────
_THIS  = Path(__file__).resolve().parent
_SRC   = _THIS / "src"
_M4SRC = _THIS.parent / "member4_ml_modelling" / "src"
for p in [str(_SRC), str(_M4SRC)]:
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(_THIS.parent / "member4_ml_modelling" / "outputs" / ".mplcache"),
)

from paths import M5_FUSION
from image_quality import check_image_quality
from model_adapters import (
    FullHandXGBAdapter,
    FullHandScratchCNNXGBAdapter,
    NailXGBAdapter,
    EyeXGBAdapter,
    EyeClassificationAdapter,
    FusedHandNailXGBAdapter,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Anemia Screening Research Demo",
    page_icon="🔬",
    layout="centered",
)

# ── MANDATORY DISCLAIMER — displayed prominently at top ────────────────────────
ACADEMIC_DISCLAIMER = (
    "For academic research only. This experimental result is not a diagnosis and "
    "must not replace laboratory complete blood count (CBC) testing or qualified medical advice."
)
st.error(f"**DISCLAIMER:** {ACADEMIC_DISCLAIMER}")

st.title("🔬 Hybrid AI-Based Non-Invasive Hemoglobin Estimation")
st.caption(
    "A Feasibility Study Using Image-Derived Features | "
    "College Research Prototype — Member 5 Integration"
)

st.markdown("""
This demonstration interface evaluates pre-trained regression and exploratory screening models
on uploaded images. All models are loaded once from disk (`st.cache_resource`) and **never retrained**
during user interaction.
""")

# ── Load models once at startup ────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading verified Member 4 & Member 5 models…")
def load_all_models():
    adapters = {}
    errors   = {}
    model_specs = [
        ("full_hand_reg",         FullHandXGBAdapter),
        ("full_hand_scratch_cnn", FullHandScratchCNNXGBAdapter),
        ("nail_reg",              NailXGBAdapter),
        ("eye_reg",               EyeXGBAdapter),
        ("eye_cls",               EyeClassificationAdapter),
    ]
    if (M5_FUSION / "fused_model.pkl").exists():
        model_specs.append(("fused_reg", FusedHandNailXGBAdapter))

    for name, cls in model_specs:
        try:
            a = cls()
            a.load()
            adapters[name] = a
        except Exception as exc:
            errors[name] = str(exc)
    return adapters, errors


adapters, load_errors = load_all_models()

if load_errors:
    st.warning("Some optional models could not be loaded:")
    for k, e in load_errors.items():
        st.write(f"- **{k}**: {e}")

# ── Sidebar: modality selection ────────────────────────────────────────────────
st.sidebar.header("Modality & Model Selection")

modality_options = []
if "full_hand_reg" in adapters:
    modality_options.append("Full-hand — Handcrafted XGBoost (Hb Regression)")
if "full_hand_scratch_cnn" in adapters:
    modality_options.append("Full-hand — Scratch-CNN 64-D + XGBoost (Hb Regression)")
if "nail_reg" in adapters:
    modality_options.append("Nail — Handcrafted XGBoost (Hb Regression)")
if "eye_reg" in adapters:
    modality_options.append("Eye / Conjunctiva — Handcrafted XGBoost (Hb Regression)")
if "eye_cls" in adapters:
    modality_options.append("Eye / Conjunctiva — Exploratory Screening Classifier")
if "fused_reg" in adapters:
    modality_options.append("Paired Full-hand + Nail — Multi-View Fusion (Hb Regression)")

if not modality_options:
    st.error("No models loaded. Check installation and model paths.")
    st.stop()

modality = st.sidebar.selectbox("Select modality & model", modality_options)

st.sidebar.markdown("---")
st.sidebar.markdown("**Important Scientific Guardrails:**")
st.sidebar.markdown(
    "- **Palm dataset:** Excluded from Hb estimation because the local Palm dataset has folder-derived labels only and no measured laboratory Hb values.\n"
    "- **Nail & Full-hand:** Originate from the same source hand photograph in the Figshare dataset (paired views, not independent clinical sensors).\n"
    "- **Screening scores:** Raw classifier probabilities are uncalibrated model scores, **not** calibrated medical risk."
)


def _save_temp_upload(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(uploaded_file, tmp)
        return Path(tmp.name)


def _render_quality_box(qr: dict, label_prefix: str = ""):
    title_str = f"Image Quality Assessment ({label_prefix})" if label_prefix else "Image Quality Assessment"
    st.subheader(title_str)
    st.caption(
        "Engineering image-quality checks only (readability, dimensions, aspect ratio, "
        "brightness/exposure, contrast, Laplacian blur score, saturation, and central framing). "
        "These checks do not clinically validate an image."
    )
    q_col1, q_col2, q_col3, q_col4 = st.columns(4)
    q_col1.metric("Dimensions", f"{qr.get('width', 'N/A')}×{qr.get('height', 'N/A')}")
    q_col2.metric("Brightness (0–255)", f"{qr.get('brightness_mean', 'N/A')}")
    q_col3.metric("Contrast (std)", f"{qr.get('contrast_std', 'N/A')}")
    q_col4.metric("Blur (Laplacian var)", f"{qr.get('blur_score', 'N/A')}")

    if qr["warnings"]:
        for w_msg in qr["warnings"]:
            st.warning(f"⚠ Quality Warning: {w_msg}")
    else:
        st.success("No engineering quality warnings triggered.")


# ── Mode 1: Paired Multi-View Fusion ──────────────────────────────────────────
if modality == "Paired Full-hand + Nail — Multi-View Fusion (Hb Regression)":
    st.markdown("---")
    st.subheader("Paired Full-hand + Nail Multi-View Upload")
    st.info(
        "**Scientific Requirement:** Nail and Full-hand in the Figshare cohort are paired views "
        "cropped from the **same source hand photograph** (`photo/<id>.jpg`). "
        "Do not upload arbitrary images from different individuals or unrelated photographs. "
        "Upload one Full-hand image and 1 to 3 Nail crop images from the same hand photograph; "
        "Nail crop features are aggregated via deterministic participant-level mean before concatenation."
    )

    up_full = st.file_uploader(
        "1. Upload Full-hand image (required)",
        type=["jpg", "jpeg", "png"],
        key="fusion_full",
    )
    up_nails = st.file_uploader(
        "2. Upload 1 to 3 paired Nail crop images from the SAME photograph (required)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="fusion_nails",
    )

    if up_full is not None and up_nails:
        if len(up_nails) > 3:
            st.warning("More than 3 Nail crops uploaded; using the first 3 crops in deterministic order.")
            up_nails = up_nails[:3]

        tmp_full = _save_temp_upload(up_full)
        tmp_nails = [_save_temp_upload(u) for u in up_nails]

        try:
            qr_full = check_image_quality(tmp_full)
            if not qr_full["is_readable"]:
                st.error(f"Full-hand file is unreadable: {'; '.join(qr_full['warnings'])}")
            else:
                st.image(Image.open(tmp_full).convert("RGB"), caption=f"Full-hand: {up_full.name}", width=240)
                _render_quality_box(qr_full, "Full-hand")

            nail_readable = True
            nail_cols = st.columns(len(tmp_nails))
            for idx, (np_path, u_file) in enumerate(zip(tmp_nails, up_nails)):
                qr_n = check_image_quality(np_path)
                if not qr_n["is_readable"]:
                    st.error(f"Nail crop #{idx+1} ({u_file.name}) is unreadable: {'; '.join(qr_n['warnings'])}")
                    nail_readable = False
                else:
                    nail_cols[idx].image(
                        Image.open(np_path).convert("RGB"),
                        caption=f"Nail #{idx+1}: {u_file.name} ({qr_n['width']}×{qr_n['height']})",
                        use_container_width=True,
                    )

            if qr_full["is_readable"] and nail_readable:
                fused_adapter: FusedHandNailXGBAdapter = adapters["fused_reg"]
                pred = fused_adapter.predict_paired(tmp_full, tmp_nails)
                st.markdown("---")
                st.subheader("Experimental Paired Fusion Result")
                if pred.get("hb_g_dl") is not None:
                    st.metric(
                        label=f"Experimental Hb Estimate — {pred['model_name']}",
                        value=f"{pred['hb_g_dl']:.2f} g/dL",
                    )
                    st.write(f"**Selected Modality:** `{pred['modality']}`")
                    st.write(f"**Model Name:** {pred['model_name']}")
                    st.write(f"**Model Provenance:** {pred['provenance']}")
                    for w in pred.get("warnings", []):
                        st.info(w)
                else:
                    st.error("Paired fusion prediction failed.")
                    for w in pred.get("warnings", []):
                        st.warning(w)
        finally:
            tmp_full.unlink(missing_ok=True)
            for p in tmp_nails:
                p.unlink(missing_ok=True)

# ── Mode 2: Single-Image Modalities ───────────────────────────────────────────
else:
    st.markdown("---")
    st.subheader("Upload an Image")

    col_note = {
        "Full-hand — Handcrafted XGBoost (Hb Regression)":
            "Upload a preprocessed Full-hand photograph (Figshare format).",
        "Full-hand — Scratch-CNN 64-D + XGBoost (Hb Regression)":
            "Upload a preprocessed Full-hand photograph (224×224 RGB input to Scratch-CNN + XGBoost).",
        "Nail — Handcrafted XGBoost (Hb Regression)":
            "Upload a close-up fingernail crop image.",
        "Eye / Conjunctiva — Handcrafted XGBoost (Hb Regression)":
            "Upload a conjunctiva / inner lower eyelid photograph.",
        "Eye / Conjunctiva — Exploratory Screening Classifier":
            "Upload a conjunctiva / inner lower eyelid photograph for exploratory threshold screening.",
    }
    st.info(col_note.get(modality, "Upload the relevant image."))

    uploaded = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png"],
        help="JPEG or PNG; minimum 64×64 pixels recommended.",
    )

    if uploaded is not None:
        tmp_path = _save_temp_upload(uploaded)
        try:
            qr = check_image_quality(tmp_path)
            if not qr["is_readable"]:
                st.error(f"Uploaded file cannot be read as a valid image: {'; '.join(qr['warnings'])}")
            else:
                pil_img = Image.open(tmp_path).convert("RGB")
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.image(pil_img, caption=f"Uploaded: {uploaded.name}", use_container_width=True)
                with col2:
                    st.markdown(f"**Image Dimensions:** {qr['width']} × {qr['height']} px (Channels: {qr['channels']})")
                    st.markdown(f"**Aspect Ratio:** {qr['aspect_ratio']}:1")
                    st.markdown(f"**Selected Modality:** {modality}")

                st.markdown("---")
                _render_quality_box(qr)

                st.markdown("---")
                st.subheader("Experimental Model Output")

                adapter_key_map = {
                    "Full-hand — Handcrafted XGBoost (Hb Regression)":        "full_hand_reg",
                    "Full-hand — Scratch-CNN 64-D + XGBoost (Hb Regression)": "full_hand_scratch_cnn",
                    "Nail — Handcrafted XGBoost (Hb Regression)":              "nail_reg",
                    "Eye / Conjunctiva — Handcrafted XGBoost (Hb Regression)": "eye_reg",
                    "Eye / Conjunctiva — Exploratory Screening Classifier":    "eye_cls",
                }
                adapter = adapters[adapter_key_map[modality]]
                pred = adapter.predict_image(tmp_path)

                if pred.get("hb_g_dl") is not None:
                    st.metric(
                        label=f"Experimental Hb Estimate ({pred['model_name']})",
                        value=f"{pred['hb_g_dl']:.2f} g/dL",
                    )
                elif pred.get("screening_label") is not None:
                    label_str = (
                        "Low-Hb / Anemic Screening Flag (1)"
                        if pred["screening_label"] == 1
                        else "Above-Threshold / Non-Anemic Flag (0)"
                    )
                    st.metric(
                        label=f"Experimental Screening Output ({pred['model_name']})",
                        value=label_str,
                    )
                    st.write(
                        f"**Uncalibrated Model Score:** `{pred['screening_score']:.4f}` "
                        "*(Note: Raw random-forest vote fraction; this is NOT a calibrated medical risk probability)*"
                    )
                    st.warning(
                        "**Inherited Label Rule:** Eye screening labels inherit Member 3's sex-specific cutoff "
                        "(`Hb < 13.0 g/dL` for males, `Hb < 12.0 g/dL` for females). This cohort includes "
                        "19 participants aged > 65 and lacks pregnancy/altitude/smoking adjustments."
                    )

                st.markdown(f"**Modality Key:** `{pred['modality']}`")
                st.markdown(f"**Model Name:** {pred['model_name']}")
                st.markdown(f"**Model Provenance:** {pred['provenance']}")

                for w_str in pred.get("warnings", []):
                    if w_str:
                        st.warning(w_str)
        finally:
            tmp_path.unlink(missing_ok=True)

# ── Fusion Summary Expander ────────────────────────────────────────────────────
st.markdown("---")
with st.expander("View Held-Out Test Comparison: Single-View vs. Paired Nail + Full-hand Fusion (n=38)"):
    fp = M5_FUSION / "fusion_comparison.csv"
    if fp.exists():
        import pandas as pd
        fc = pd.read_csv(fp)
        st.dataframe(fc[["label", "MAE_g_dl", "RMSE_g_dl", "R2", "n_test"]], use_container_width=True)
        plot_p = M5_FUSION / "fusion_comparison.png"
        if plot_p.exists():
            st.image(str(plot_p), use_container_width=True)
        st.caption(
            "Both Nail and Full-hand inputs are views cropped from the **same source photograph**. "
            "On the 38 held-out test participants, concatenating mean-aggregated Nail features with "
            "Full-hand features improved over Nail-only (MAE 1.6792 vs 1.8289 g/dL) but did not beat "
            "Full-hand-only (MAE 1.5819 g/dL)."
        )

st.markdown("---")
st.caption(ACADEMIC_DISCLAIMER)
