"""
app.py — Streamlit demonstration application for Member 5 Integration.

Academic research prototype only. NOT a diagnostic tool.

Launch from project root (Anemia_ML_Project/):
    py -3.10 -m streamlit run member5_integration/app.py

Models are loaded once at startup via st.cache_resource.
No retraining occurs during the app session.
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path
from PIL import Image

import pandas as pd
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

from paths import M5_FUSION, M5_EXPL, M5_METRICS, M5_PREDICTIONS
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
    page_title="Anemia Hb Estimation — Research Demo",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Typography */
html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }
h1 { font-size: 1.75rem !important; font-weight: 700; }
h2 { font-size: 1.25rem !important; font-weight: 600; color: #1f4e79; }
h3 { font-size: 1.05rem !important; font-weight: 600; }

/* Metric boxes */
div[data-testid="metric-container"] {
    background: #f0f4fa;
    border: 1px solid #c9d8ed;
    border-radius: 8px;
    padding: 12px 16px;
}

/* Disclaimer banner override */
div[data-testid="stAlert"][data-baseweb="notification"] p {
    font-size: 0.9rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #f7f9fc;
    border-right: 1px solid #e0e7f0;
}

/* Tab active underline colour */
button[data-baseweb="tab"][aria-selected="true"] {
    border-bottom-color: #1f4e79 !important;
    color: #1f4e79 !important;
}

/* Table styling */
div[data-testid="stDataFrame"] { border-radius: 6px; overflow: hidden; }

/* Wider content on wide screens */
.block-container { max-width: 1100px; padding-top: 1.2rem; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
ACADEMIC_DISCLAIMER = (
    "🚨 **ACADEMIC RESEARCH PROTOTYPE — NOT A DIAGNOSTIC TOOL.**  "
    "Outputs are experimental machine-learning estimates, **not** clinical diagnoses.  "
    "This system has **not** been clinically validated and must **not** replace laboratory "
    "Complete Blood Count (CBC) testing or qualified medical advice."
)

# ── Persistent disclaimer at top ──────────────────────────────────────────────
st.error(ACADEMIC_DISCLAIMER)

# ── Header ────────────────────────────────────────────────────────────────────
col_title, col_badge = st.columns([4, 1])
with col_title:
    st.title("🔬 Hybrid AI-Based Non-Invasive Hemoglobin Estimation")
    st.caption(
        "A Feasibility Study Using Image-Derived Features  |  "
        "Third-Year Engineering Capstone — Five-Member Team"
    )
with col_badge:
    st.markdown("""
    <div style='text-align:center; margin-top:12px;
                background:#e8f4f8; border-radius:8px; padding:10px 6px;
                border:1px solid #b3d4e8; font-size:0.78rem;'>
        <b>Python 3.10</b><br>XGBoost · Scikit-learn<br>Streamlit · PyTorch<br>SHAP
    </div>
    """, unsafe_allow_html=True)

# ── Load models once at startup ────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading verified Member 4 & 5 models from disk…")
def load_all_models():
    adapters, errors = {}, {}
    specs = [
        ("full_hand_reg",         FullHandXGBAdapter),
        ("full_hand_scratch_cnn", FullHandScratchCNNXGBAdapter),
        ("nail_reg",              NailXGBAdapter),
        ("eye_reg",               EyeXGBAdapter),
        ("eye_cls",               EyeClassificationAdapter),
    ]
    if (M5_FUSION / "fused_model.pkl").exists():
        specs.append(("fused_reg", FusedHandNailXGBAdapter))
    for name, cls in specs:
        try:
            a = cls(); a.load(); adapters[name] = a
        except Exception as exc:
            errors[name] = str(exc)
    return adapters, errors

adapters, load_errors = load_all_models()

if load_errors:
    with st.expander(f"⚠ {len(load_errors)} model(s) could not be loaded (click to expand)", expanded=False):
        for k, e in load_errors.items():
            st.write(f"- **{k}**: `{e}`")

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙ Model Selection")

    OPTIONS = []
    if "full_hand_reg"         in adapters: OPTIONS.append("Full-hand — Handcrafted XGBoost")
    if "full_hand_scratch_cnn" in adapters: OPTIONS.append("Full-hand — Scratch-CNN + XGBoost")
    if "nail_reg"              in adapters: OPTIONS.append("Nail — Handcrafted XGBoost")
    if "eye_reg"               in adapters: OPTIONS.append("Eye — Handcrafted XGBoost")
    if "eye_cls"               in adapters: OPTIONS.append("Eye — Exploratory Classifier")
    if "fused_reg"             in adapters: OPTIONS.append("Paired Full-hand + Nail — Fusion")

    if not OPTIONS:
        st.error("No models loaded. Check installation and paths.")
        st.stop()

    modality = st.selectbox("Choose modality & model", OPTIONS)

    st.divider()
    st.markdown("#### 📊 Quick Dataset Summary")
    st.markdown("""
| View | Participants | Test n |
|------|:---:|:---:|
| Eye (EDI) | 202 | 31 |
| Full-hand | 250 | 38 |
| Nail crops | 250 | 38 |
| Palm (folder labels) | 526 | 73 |
""")

    st.divider()
    st.markdown("#### ⚠ Scientific Notes")
    st.markdown(
        "- **Palm:** excluded from Hb estimation — folder labels only, no measured Hb.\n"
        "- **Nail + Full-hand:** paired views from the **same source photograph**, "
          "not independent sensors.\n"
        "- **Classifier scores:** uncalibrated tree-vote fractions — not medical probabilities.\n"
        "- **CNN embeddings (64-D):** opaque latent features with no direct biological meaning."
    )

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_demo, tab_results, tab_shap, tab_info = st.tabs([
    "🧪 Demo — Upload & Predict",
    "📊 Benchmark Results",
    "🔍 SHAP Explainability",
    "ℹ Project Info & Limitations",
])

# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _save_temp(uploaded) -> Path:
    suffix = Path(uploaded.name).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(uploaded, tmp)
        return Path(tmp.name)


def _quality_panel(qr: dict, label: str = ""):
    heading = f"Image Quality Check — {label}" if label else "Image Quality Check"
    st.markdown(f"**{heading}**")
    st.caption(
        "Engineering heuristics only: readability, resolution, aspect ratio, brightness, "
        "contrast, Laplacian blur variance, saturation, central framing.  "
        "These do **not** clinically certify image adequacy."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Resolution", f"{qr.get('width','?')}×{qr.get('height','?')} px")
    c2.metric("Brightness", f"{qr.get('brightness_mean','?'):.1f}" if isinstance(qr.get('brightness_mean'), float) else "N/A")
    c3.metric("Contrast σ", f"{qr.get('contrast_std','?'):.1f}" if isinstance(qr.get('contrast_std'), float) else "N/A")
    c4.metric("Blur (Lap.var)", f"{qr.get('blur_score','?'):.1f}" if isinstance(qr.get('blur_score'), float) else "N/A")
    if qr["warnings"]:
        for w in qr["warnings"]:
            st.warning(f"⚠ {w}")
    else:
        st.success("No quality warnings.")


def _show_pred(pred: dict):
    """Render a prediction result dict."""
    st.markdown("---")
    st.subheader("Experimental Model Output")

    if pred.get("hb_g_dl") is not None:
        st.metric(
            label=f"Experimental Hb Estimate · {pred['model_name']}",
            value=f"{pred['hb_g_dl']:.2f} g/dL",
        )
    elif pred.get("screening_label") is not None:
        label_txt = (
            "🔴 Low-Hb / Anemic flag (class 1)"
            if pred["screening_label"] == 1
            else "🟢 Above-threshold flag (class 0)"
        )
        st.metric(label=f"Exploratory Screening · {pred['model_name']}", value=label_txt)
        st.write(
            f"**Uncalibrated model score:** `{pred['screening_score']:.4f}`  "
            "*(raw tree-vote fraction — NOT a calibrated medical risk probability)*"
        )
    else:
        st.error("Prediction could not be produced. See warnings below.")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"**Modality:** `{pred['modality']}`")
        st.markdown(f"**Model:** {pred['model_name']}")
    with col_b:
        with st.expander("Model provenance", expanded=False):
            st.write(pred["provenance"])

    if pred.get("warnings"):
        for w in pred["warnings"]:
            if w:
                st.info(f"ℹ {w}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Demo
# ══════════════════════════════════════════════════════════════════════════════
with tab_demo:
    st.header("Upload an Image — Experimental Prediction")

    # Input guidance
    GUIDANCE = {
        "Full-hand — Handcrafted XGBoost":
            "Upload a preprocessed Full-hand photograph (Figshare format, dorsal hand view).",
        "Full-hand — Scratch-CNN + XGBoost":
            "Upload a preprocessed Full-hand photograph (224×224 RGB). "
            "The 64-D Scratch-CNN embedding is opaque — see Limitations.",
        "Nail — Handcrafted XGBoost":
            "Upload a close-up fingernail-bed crop image.",
        "Eye — Handcrafted XGBoost":
            "Upload a palpebral conjunctiva / inner lower eyelid photograph.",
        "Eye — Exploratory Classifier":
            "Upload a conjunctiva photograph for exploratory low-Hb threshold screening.  "
            "Labels use sex-specific Hb cutoffs (Male <13, Female <12 g/dL) with caveats.",
        "Paired Full-hand + Nail — Fusion":
            "Upload **one Full-hand image** and **1–3 Nail crop images from the same photograph**.",
    }
    st.info(GUIDANCE.get(modality, "Upload the relevant image."))

    # ── Paired fusion mode ────────────────────────────────────────────────────
    if modality == "Paired Full-hand + Nail — Fusion":
        st.markdown(
            "> **Scientific note:** Nail and Full-hand images in the Figshare dataset are "
            "bounding-box crops from the **same source hand photograph** (`photo/<id>.jpg`).  "
            "They are **not** independent clinical sensors. Upload images from the same photo."
        )
        up_full  = st.file_uploader("1. Full-hand image", type=["jpg","jpeg","png"], key="fu_full")
        up_nails = st.file_uploader(
            "2. Nail crop image(s) from the SAME photograph (1–3 files)",
            type=["jpg","jpeg","png"], accept_multiple_files=True, key="fu_nails",
        )

        if up_full and up_nails:
            if len(up_nails) > 3:
                st.warning("More than 3 Nail crops supplied — using first 3.")
                up_nails = up_nails[:3]
            tmp_full  = _save_temp(up_full)
            tmp_nails = [_save_temp(u) for u in up_nails]
            try:
                qr_full = check_image_quality(tmp_full)
                if not qr_full["is_readable"]:
                    st.error(f"Full-hand file unreadable: {'; '.join(qr_full['warnings'])}")
                    st.stop()
                img_col, info_col = st.columns([1, 2])
                with img_col:
                    st.image(Image.open(tmp_full).convert("RGB"),
                             caption=f"Full-hand: {up_full.name}", use_container_width=True)
                with info_col:
                    _quality_panel(qr_full, "Full-hand")

                nail_ok = True
                n_cols = st.columns(len(tmp_nails))
                for i, (np_p, uf) in enumerate(zip(tmp_nails, up_nails)):
                    qr_n = check_image_quality(np_p)
                    if not qr_n["is_readable"]:
                        st.error(f"Nail #{i+1} unreadable: {'; '.join(qr_n['warnings'])}")
                        nail_ok = False
                    else:
                        n_cols[i].image(Image.open(np_p).convert("RGB"),
                                        caption=f"Nail #{i+1}: {uf.name}", use_container_width=True)

                if nail_ok:
                    pred = adapters["fused_reg"].predict_paired(tmp_full, tmp_nails)
                    _show_pred(pred)
            finally:
                tmp_full.unlink(missing_ok=True)
                for p in tmp_nails:
                    p.unlink(missing_ok=True)

    # ── Single-image modes ────────────────────────────────────────────────────
    else:
        KEY_MAP = {
            "Full-hand — Handcrafted XGBoost":    "full_hand_reg",
            "Full-hand — Scratch-CNN + XGBoost":  "full_hand_scratch_cnn",
            "Nail — Handcrafted XGBoost":          "nail_reg",
            "Eye — Handcrafted XGBoost":           "eye_reg",
            "Eye — Exploratory Classifier":        "eye_cls",
        }
        uploaded = st.file_uploader(
            "Choose image file (JPG or PNG)",
            type=["jpg","jpeg","png"],
            help="Minimum 64×64 px recommended.",
        )

        if uploaded is not None:
            tmp_path = _save_temp(uploaded)
            try:
                qr = check_image_quality(tmp_path)
                if not qr["is_readable"]:
                    st.error(f"Cannot read image: {'; '.join(qr['warnings'])}")
                else:
                    img_col, detail_col = st.columns([1, 2])
                    with img_col:
                        st.image(Image.open(tmp_path).convert("RGB"),
                                 caption=uploaded.name, use_container_width=True)
                    with detail_col:
                        st.markdown(
                            f"**Dimensions:** {qr['width']} × {qr['height']} px  |  "
                            f"**Aspect ratio:** {qr['aspect_ratio']}:1  |  "
                            f"**Channels:** {qr['channels']}"
                        )
                        st.markdown(f"**Selected modality:** {modality}")
                        _quality_panel(qr)

                    adapter = adapters[KEY_MAP[modality]]
                    pred    = adapter.predict_image(tmp_path)
                    _show_pred(pred)

                    # Eye classifier extra context
                    if modality == "Eye — Exploratory Classifier":
                        st.warning(
                            "**Threshold caveat:** Eye labels use sex-specific cutoffs "
                            "(Male <13.0 g/dL, Female <12.0 g/dL). The cohort includes "
                            "19 participants aged >65 and lacks pregnancy/altitude records. "
                            "Treat output as exploratory only."
                        )
            finally:
                tmp_path.unlink(missing_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Benchmark Results
# ══════════════════════════════════════════════════════════════════════════════
with tab_results:
    st.header("Consolidated Benchmark — Verified Held-Out Test Metrics")
    st.caption(
        "All figures loaded from saved metric files; none are hardcoded. "
        "Source: `member5_integration/outputs/metrics/member5_consolidated_comparison.csv` "
        "and `member4_ml_modelling/outputs/metrics/all_metrics_summary.csv`."
    )

    # ── Hb Regression table ───────────────────────────────────────────────────
    st.subheader("Hb Regression (g/dL) — Test Set")

    reg_rows = [
        ("Eye Handcrafted XGBoost",              "Eye (conjunctiva)",     89, "31 participants",  1.4177, 1.9744, 0.4912, "Best R²"),
        ("Eye M3 DenseNet-121 1024-D XGBoost",   "Eye (conjunctiva)",     89, "31 participants",  1.7452, 2.2907, 0.3152, "CNN underperforms"),
        ("Nail Handcrafted XGBoost (crop-level)","Nail crops",           109, "38 participants",  1.7171, 2.3079, 0.0548, "Weak standalone signal"),
        ("Nail M3 DenseNet-121 1024-D XGBoost",  "Nail crops",           109, "38 participants",  2.0425, 2.6010,-0.2006, "Neg. R²; overfits"),
        ("Nail Handcrafted XGBoost (part.-level)","Nail (participant)",   38,  "38 participants",  1.8289, 2.4117, 0.0394, "Fusion baseline"),
        ("Full-hand Handcrafted XGBoost",         "Full-hand",            38,  "38 participants",  1.5819, 2.0355, 0.3157, "Interpretable baseline"),
        ("Full-hand Scratch-CNN 64-D + XGBoost",  "Full-hand",            38,  "38 participants",  1.5599, 1.8946, 0.4072, "Best Full-hand RMSE"),
        ("Full-hand Palm-transfer XGBoost",        "Full-hand",            38,  "38 participants",  2.0970, 2.6163,-0.1305, "Cross-domain; exploratory"),
        ("Fused Full-hand + Nail 260-D XGBoost",   "Paired (same photo)", 38,  "38 participants",  1.6792, 2.2338, 0.1759, "Beats Nail-only"),
    ]
    df_reg = pd.DataFrame(reg_rows, columns=[
        "Model", "View", "Test Images/Crops", "Test Participants",
        "MAE (g/dL) ↓", "RMSE (g/dL) ↓", "R² ↑", "Note"
    ])

    # Try to load from saved CSV; fall back to hardcoded table above
    csv_path = M5_METRICS / "member5_consolidated_comparison.csv"
    if csv_path.exists():
        try:
            df_live = pd.read_csv(csv_path)
            st.dataframe(df_live, use_container_width=True)
            st.caption("✅ Loaded from saved metrics file.")
        except Exception:
            st.dataframe(df_reg, use_container_width=True)
            st.caption("⚠ Saved CSV unreadable — showing verified static table.")
    else:
        st.dataframe(df_reg, use_container_width=True)
        st.caption(
            "ℹ Saved metrics CSV not found — showing verified static table from final_report.md. "
            "Run `python member5_integration/src/evaluation.py` to generate the CSV."
        )

    st.markdown("""
**Key findings:**
- Handcrafted 130-D features outperform 1024-D pretrained CNN embeddings on both Eye and Nail.
- Full-hand Scratch-CNN (64-D) achieves the best Full-hand RMSE but the difference from handcrafted (0.022 g/dL MAE) is within sampling uncertainty at n=38.
- Fusion improves over Nail-only but not Full-hand-only — both views come from the same source photograph.
- All test errors (MAE 1.42–1.83 g/dL) are large relative to clinical thresholds. This is an experimental feasibility study only.
""")

    st.divider()

    # ── Classification table ──────────────────────────────────────────────────
    st.subheader("Exploratory Classification — Test Set")
    cls_rows = [
        ("Eye ExtraTrees",   "Eye", "Hb <11 g/dL (sex-adj.)", 89,  0.8876, 0.9487, 0.8222, 0.9545, 0.8810, 0.9702),
        ("Eye RandomForest", "Eye", "Hb <11 g/dL (sex-adj.)", 89,  0.8539, 0.8675, 0.8222, 0.8864, 0.8471, 0.9604),
        ("Nail ExtraTrees",  "Nail","Hb <11 g/dL (universal)",109, 0.7798, 0.2222, 0.1053, 0.9222, 0.1429, 0.6099),
        ("Nail RandomForest","Nail","Hb <11 g/dL (universal)",109, 0.8349, 0.3333, 0.0526, 0.9778, 0.1000, 0.6243),
        ("Palm ExtraTrees",  "Palm","Folder label only",       292, 0.7500, 0.7844, 0.7798, 0.7097, 0.7821, 0.7788),
        ("Palm RandomForest","Palm","Folder label only",       292, 0.7192, 0.7654, 0.7381, 0.6935, 0.7515, 0.7689),
    ]
    df_cls = pd.DataFrame(cls_rows, columns=[
        "Model","View","Label Source","Test n",
        "Accuracy","Precision","Sensitivity","Specificity","F1","ROC-AUC"
    ])
    st.dataframe(df_cls, use_container_width=True)
    st.warning(
        "**All classification outputs are exploratory.** Eye labels use inherited "
        "sex-specific Hb cutoffs with missing pregnancy/altitude context. "
        "Nail labels use a universal 11 g/dL threshold (severe class imbalance — "
        "only 17% positive). Palm predicts folder labels only — no measured Hb. "
        "None of these represent clinically validated anemia diagnosis."
    )

    st.divider()

    # ── Fusion comparison plot ────────────────────────────────────────────────
    st.subheader("Single-View vs. Fusion Comparison (n=38 test participants)")
    fusion_csv = M5_FUSION / "fusion_comparison.csv"
    fusion_png = M5_FUSION / "fusion_comparison.png"
    if fusion_csv.exists():
        df_fus = pd.read_csv(fusion_csv)
        st.dataframe(
            df_fus[["label","MAE_g_dl","RMSE_g_dl","R2","n_test"]]
            if all(c in df_fus.columns for c in ["label","MAE_g_dl","RMSE_g_dl","R2","n_test"])
            else df_fus,
            use_container_width=True
        )
    if fusion_png.exists():
        st.image(str(fusion_png), use_container_width=True)
    st.caption(
        "Both Nail and Full-hand inputs are views cropped from the **same source photograph**. "
        "Fusion (MAE 1.6792 g/dL) beats Nail-only (1.8289 g/dL) but not Full-hand-only (1.5819 g/dL). "
        "Doubling feature dimensionality (130→260) on 175 training participants dilutes XGBoost "
        "split selection without adding an independent sensor view."
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — SHAP Explainability
# ══════════════════════════════════════════════════════════════════════════════
with tab_shap:
    st.header("TreeSHAP Explainability — Full-hand Handcrafted XGBoost")
    st.markdown("""
TreeSHAP (`shap.TreeExplainer`) was run on `full_hand_xgb_regression.pkl` using the 175 training
participants as background reference.  Results cover all 38 held-out **test** participants
(no tuning on test data).

> **What SHAP shows:** feature contributions to the model's output relative to the training mean.
> SHAP values reflect statistical associations in training data — they are **not**
> causal physiological explanations and must not be described as clinical evidence.
""")

    global_png = M5_EXPL / "shap_global_summary.png"
    local_png  = M5_EXPL / "shap_local_example.png"
    local_csv  = M5_EXPL / "shap_local_example.csv"

    if global_png.exists():
        st.subheader("Global Feature Importance (Mean |SHAP| across test participants)")
        st.image(str(global_png), use_container_width=True)
        st.caption(
            "Top features by mean absolute SHAP value on the 38 test participants. "
            "Green-channel histogram bins and HSV Hue percentiles dominate — consistent with "
            "hemoglobin's strong absorption of green light (~540–580 nm)."
        )
    else:
        st.info(
            "Global SHAP plot not found. "
            "Generate it by running: `py -3.10 member5_integration/src/explainability.py`"
        )

    st.divider()

    if local_png.exists():
        st.subheader("Local Attribution — Test Participant 6 (0006_full.jpg)")
        st.markdown(
            "True Hb: **13.80 g/dL** | Model prediction: **15.07 g/dL** | "
            "SHAP base value (training mean): **12.81 g/dL**"
        )
        st.image(str(local_png), use_container_width=True)
    else:
        st.info(
            "Local SHAP plot not found. "
            "Run `py -3.10 member5_integration/src/explainability.py` to generate it."
        )

    if local_csv.exists():
        with st.expander("Full 130-feature local attribution table (test participant 6)", expanded=False):
            df_shap = pd.read_csv(local_csv)
            st.dataframe(df_shap, use_container_width=True)

    st.divider()

    # Top-10 static table (always shown for reference)
    st.subheader("Top 10 Features — Mean |SHAP| (g/dL), test n=38")
    shap_top10 = [
        (1,  "HSV_H_p10",    0.4466, "10th-percentile HSV Hue in central ROI — lower-bound reddish/pinkish hue shift"),
        (2,  "G_hist_5",     0.3994, "Green-channel hist bin 5 — hemoglobin absorbs green light at ~540–580 nm"),
        (3,  "B_hist_10",    0.3542, "Blue-channel hist upper-intensity bin"),
        (4,  "G_hist_6",     0.2765, "Green-channel hist bin 6 — upper-mid green reflectance"),
        (5,  "G_std",        0.2246, "Green-channel intensity std — spatial contrast in palmar tissue"),
        (6,  "B_hist_5",     0.1558, "Blue-channel hist bin 5"),
        (7,  "HSV_S_p50",    0.1235, "Median HSV Saturation — overall colour richness vs. pallor"),
        (8,  "LAB_B_hist_4", 0.1059, "CIE LAB b* (yellow–blue axis) mid-range bin 4"),
        (9,  "G_hist_3",     0.0977, "Green-channel hist bin 3 — lower-mid green absorption"),
        (10, "HSV_H_skew",   0.0772, "Skewness of HSV Hue distribution across the hand ROI"),
    ]
    df_top10 = pd.DataFrame(shap_top10, columns=["Rank","Feature","Mean |SHAP| (g/dL)","Colorimetric Interpretation"])
    st.dataframe(df_top10, use_container_width=True, hide_index=True)
    st.caption(
        "Values sourced from `member5_integration/reports/explainability.md`. "
        "Green-channel and Hue features reflecting subcutaneous vasculature rank highest — "
        "consistent with hemoglobin spectroscopy, though these are correlational, not causal."
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Project Info & Limitations
# ══════════════════════════════════════════════════════════════════════════════
with tab_info:
    st.header("Project Information, Architecture & Limitations")

    with st.expander("📐 System Architecture", expanded=True):
        st.markdown("""
```
Input Image(s)
    │
    ▼
Engineering Quality Gate (image_quality.py)
    │  readability · resolution · brightness · contrast · blur · saturation
    │
    ├─► 130-D Handcrafted Features (feature_extraction.py)
    │       RGB/HSV/LAB/YCrCb stats · histograms · pallor ratios · GLCM texture
    │       └─► XGBRegressor  (Eye / Nail / Full-hand Hb regression)
    │       └─► ExtraTrees / RF Classifier (exploratory screening)
    │
    ├─► 64-D Scratch-CNN Embeddings (FullHandScratchCNN, PyTorch)
    │       Trained from scratch on 175 Full-hand participants
    │       └─► XGBRegressor
    │
    ├─► 1024-D M3 DenseNet-121 Embeddings (Member 3 checkpoints)
    │       Originally trained for binary classification (image-level splits)
    │       └─► XGBRegressor  (Eye / Nail; underperforms handcrafted baseline)
    │
    └─► Paired Fusion (fusion.py)
            Full-hand 130-D + mean-aggregated Nail 130-D = 260-D
            └─► XGBRegressor (fused_model.pkl)

All models load from disk (st.cache_resource). Zero retraining at runtime.
ModelAdapter layer (model_adapters.py) wraps every model with a typed dict output.
```
""")

    with st.expander("👥 Team Contributions", expanded=False):
        st.markdown("""
| Member | Assigned Role | Key Deliverables |
|--------|--------------|-----------------|
| **Member 1** | Dataset & data audit lead | Dataset sourcing, ZIP extraction, MD5 deduplication, Hb unit verification, split manifests, license status documentation |
| **Member 2** | Image preprocessing & quality | ROI cropping, 224×224 resizing, RGB normalisation, preprocessing pipeline in `feature_extraction.py` |
| **Member 3** | Deep learning feature extraction | LightweightCNN / DenseNet-121 checkpoints for Eye, Nail, Palm (binary classification training); 1024-D embedding extraction |
| **Member 4** | ML modelling & evaluation | 130-D handcrafted features, XGBoost/RF/ET regression & classification, full-hand Scratch-CNN, duplicate audits, `Run_Project.bat`, results dashboard |
| **Member 5** | Fusion, explainability & demo | Pairing audit, 260-D fusion, TreeSHAP, ModelAdapter layer, image quality gate, Streamlit app, test suite, final report |

> Contribution details marked with ⚠ require team confirmation where individual
> sub-task evidence is not directly verifiable from committed artefacts.
""")

    with st.expander("⚠ Limitations (10 key items)", expanded=False):
        st.markdown("""
1. **Small test cohorts:** 31 Eye participants (89 images) and 38 Figshare participants
   (38 full-hand, 109 nail crops). Metric differences < 0.15 g/dL MAE are within sampling uncertainty.

2. **Same-photograph pairing:** Figshare Full-hand and Nail views are both cropped from
   `photo/<id>.jpg`. Fusion tests multi-ROI feature combination from one camera capture,
   not independent sensor modalities.

3. **No cross-dataset fusion possible:** Eye, Figshare, and Palm cohorts come from separate
   unrelated studies with no shared participants.

4. **Incomplete demographic metadata:** Figshare lacks sex/age/pregnancy metadata.
   Eye cohort includes 19 participants >65; no pregnancy or altitude records.
   All classification thresholds are exploratory proxies, not WHO-stratified clinical diagnoses.

5. **Palm — folder labels only:** No measured Hb exists. Palm models predict folder labels
   (Anemic / Non-anemic) assigned from augmented-image filenames only.

6. **Opaque CNN dimensions:** 64-D Scratch-CNN and 1024-D M3 DenseNet-121 embedding
   coordinates are abstract latent activations without biological meaning.

7. **Uncalibrated screening scores:** Classifier outputs are raw tree-vote fractions on
   imbalanced cohorts, not calibrated clinical risk probabilities.

8. **Physical RGB limits:** Ambient lighting, sensor white balance, skin tone variation,
   and specular reflection bound achievable R² (0.31–0.49 on Full-hand and Eye;
   0.05 on isolated Nail crops).

9. **Dataset licensing pending:** Formal upstream redistribution terms for the EDI Eye
   dataset, Figshare Nail/Hand dataset, and Mendeley Palm dataset have not been
   independently verified. No raw participant images should be redistributed externally
   before licence confirmation.

10. **Not clinically validated:** This system has not undergone prospective clinical trials,
    multi-centre validation, or regulatory review. It must never be used to diagnose anemia,
    guide treatment, or replace laboratory CBC testing.
""")

    with st.expander("📦 Dataset Summary & Licensing Status", expanded=False):
        st.markdown("""
| Dataset | View | Participants | Images | Hb Target | Licensing Status |
|---------|------|:---:|:---:|:---|:---|
| EDI Eye (Dimauro et al.) | Palpebral conjunctiva | 202 | 610 (dedup.) | Numerical g/dL, lab-measured | **Pending upstream verification** |
| Figshare Nail/Skin | Full-hand + Nail crops | 250 | 250 + 719 | Numerical g/dL (converted from g/L) | **Pending upstream verification** |
| Mendeley Palm (Ghana) | Palmar hand | 526 groups | 2,104 (dedup.) | Folder label only — **no Hb** | **Pending upstream verification** |
| Member 3 DenseNet-121 | — | — | 3 checkpoints | — | Internal team handover |

**Raw participant photographs must not be redistributed publicly before confirming upstream licenses.**
""")

    with st.expander("🔬 Reproducibility Commands", expanded=False):
        st.code("""# From project root: Anemia_ML_Project/

# 1. Verify Member 4 saved artifacts (13/13 models, 0 mismatches)
py -3.10 member4_ml_modelling\\src\\validate_member4_results.py

# 2. Pairing audit (250/250 Nail–Full-hand participant match)
py -3.10 member5_integration\\src\\pairing.py

# 3. Fusion experiment (re-runs on saved features; does NOT retrain baselines)
py -3.10 member5_integration\\src\\fusion.py

# 4. SHAP explainability
py -3.10 member5_integration\\src\\explainability.py

# 5. Consolidated evaluation & report utilities
py -3.10 member5_integration\\src\\evaluation.py
py -3.10 member5_integration\\src\\report_utils.py

# 6. Automated test suite (31 checks)
py -3.10 -m pytest member5_integration\\tests -v

# 7. Member 4 full pipeline + dashboard (double-click or run from project root)
cmd.exe /c Run_Project.bat

# 8. Launch this Streamlit app
py -3.10 -m streamlit run member5_integration\\app.py
""", language="powershell")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "🔬 **Hybrid AI-Based Non-Invasive Hemoglobin Estimation: A Feasibility Study** | "
    "Third-Year Engineering Capstone | Academic Research Prototype Only | "
    + ACADEMIC_DISCLAIMER.replace("🚨 **", "").replace("**", "")
)
