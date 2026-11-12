"""
model_adapters.py — Consistent inference interface for all Member 4 trained models.

Each adapter:
 1. Loads its bundle once (call load() before predict_image()).
 2. Never retrains; raises if model file is missing.
 3. Checks feature dimension before prediction.
 4. Returns a typed dict.

Usage:
    adapter = FullHandXGBAdapter()
    adapter.load()
    result = adapter.predict_image("/path/to/image.jpg")
"""

import sys
import pickle
import warnings
import numpy as np
from pathlib import Path
from typing import Optional

# ── Resolve paths ──────────────────────────────────────────────────────────────
_THIS = Path(__file__).resolve().parent
_M5   = _THIS.parent
_ROOT = _M5.parent
_M4_SRC = _ROOT / "member4_ml_modelling" / "src"
if str(_M4_SRC) not in sys.path:
    sys.path.insert(0, str(_M4_SRC))

from paths import (
    EYE_XGB_REG_PKL, NAIL_XGB_REG_PKL, FULLHAND_XGB_REG_PKL,
    EYE_RF_REG_PKL,  NAIL_RF_REG_PKL,  FULLHAND_RF_REG_PKL,
    EYE_CLS_PKL, NAIL_CLS_PKL, PALM_CLS_PKL,
    FULLHAND_SCRATCH_XGB_PKL, FULLHAND_SCRATCH_PTH,
    M5_FUSION,
)

PREDICTION_RESULT_SCHEMA = {
    "modality":         str,
    "model_name":       str,
    "hb_g_dl":          (float, type(None)),
    "screening_label":  (int, type(None)),
    "screening_score":  (float, type(None)),
    "quality_issues":   list,
    "warnings":         list,
    "provenance":       str,
}


class ModelAdapter:
    """Required base interface for all Member 5 model adapters."""

    modality: str = "unknown"
    model_name: str = "unknown"
    provenance: str = "unknown"

    def predict_image(self, image_path) -> dict:
        raise NotImplementedError("Subclasses must implement predict_image(image_path) -> dict")


def _load_bundle(pkl_path: Path) -> dict:
    """Load a pickle bundle with schema validation."""
    if not pkl_path.exists():
        raise FileNotFoundError(f"Model bundle not found: {pkl_path}")
    with open(pkl_path, "rb") as f:
        bundle = pickle.load(f)
    if not isinstance(bundle, dict):
        raise ValueError(f"Expected dict bundle, got {type(bundle)}")
    return bundle


def _get_feat_cols(bundle: dict) -> list:
    """Return feature column list from either 'feat_cols' or 'feature_cols'."""
    cols = bundle.get("feat_cols", bundle.get("feature_cols"))
    if cols is None:
        raise KeyError("Bundle missing both 'feat_cols' and 'feature_cols' keys.")
    return cols


def _extract_and_align(image_path: Path, feat_cols: list) -> np.ndarray:
    """
    Extract handcrafted features and return a (1, n_features) array
    aligned to feat_cols order.
    """
    from feature_extraction import extract_features
    feats = extract_features(image_path)
    if feats is None:
        raise ValueError(f"Feature extraction returned None for {image_path}.")

    missing = [c for c in feat_cols if c not in feats]
    if missing:
        raise ValueError(
            f"Feature extraction produced a dict missing {len(missing)} required "
            f"features. First missing: {missing[:5]}"
        )

    vec = np.array([feats[c] for c in feat_cols], dtype=np.float32)
    if vec.shape[0] != len(feat_cols):
        raise ValueError(
            f"Feature vector length {vec.shape[0]} != expected {len(feat_cols)}."
        )
    return vec.reshape(1, -1)


# ══════════════════════════════════════════════════════════════════════════════
#  Base class
# ══════════════════════════════════════════════════════════════════════════════

class BaseHandcraftedAdapter(ModelAdapter):
    """Base for XGBoost / RF / ET regression adapters using handcrafted features."""

    modality:    str = "unknown"
    model_name:  str = "unknown"
    provenance:  str = "unknown"
    _pkl_path:   Path = None

    def __init__(self):
        self._bundle    = None
        self._scaler    = None
        self._model     = None
        self._feat_cols = None

    def load(self):
        """Load bundle from disk. Call once at startup."""
        self._bundle    = _load_bundle(self._pkl_path)
        self._scaler    = self._bundle["scaler"]
        # XGBoost bundles use "xgb"; RF/ET bundles use "rf"
        self._model     = self._bundle.get("xgb", self._bundle.get("rf"))
        if self._model is None:
            raise KeyError(f"Bundle {self._pkl_path.name} has neither 'xgb' nor 'rf' key.")
        self._feat_cols = _get_feat_cols(self._bundle)
        return self

    def predict_image(self, image_path) -> dict:
        if self._model is None:
            raise RuntimeError("Call .load() before predict_image().")
        path = Path(image_path)
        quality_issues = []
        warnings_out   = []

        # Quality pre-check
        from image_quality import check_image_quality
        qr = check_image_quality(path)
        if not qr["is_readable"]:
            return {
                "modality":        self.modality,
                "model_name":      self.model_name,
                "hb_g_dl":         None,
                "screening_label": None,
                "screening_score": None,
                "quality_issues":  qr["warnings"],
                "warnings":        ["Image is unreadable — prediction aborted."],
                "provenance":      self.provenance,
            }
        quality_issues.extend(qr["warnings"])

        # Feature extraction + alignment
        try:
            X = _extract_and_align(path, self._feat_cols)
        except Exception as exc:
            return {
                "modality":        self.modality,
                "model_name":      self.model_name,
                "hb_g_dl":         None,
                "screening_label": None,
                "screening_score": None,
                "quality_issues":  quality_issues,
                "warnings":        [f"Feature extraction failed: {exc}"],
                "provenance":      self.provenance,
            }

        X_scaled = self._scaler.transform(X)
        hb_pred  = float(self._model.predict(X_scaled)[0])

        # Clamp to physiologically plausible range (guard against extrapolation)
        HB_MIN, HB_MAX = 3.0, 20.0
        if hb_pred < HB_MIN or hb_pred > HB_MAX:
            warnings_out.append(
                f"Predicted Hb {hb_pred:.2f} g/dL is outside physiologically "
                f"expected range [{HB_MIN}, {HB_MAX}] — treat with caution."
            )
        hb_pred = float(np.clip(hb_pred, HB_MIN, HB_MAX))

        return {
            "modality":        self.modality,
            "model_name":      self.model_name,
            "hb_g_dl":         float(hb_pred),
            "screening_label": None,
            "screening_score": None,
            "quality_issues":  quality_issues,
            "warnings":        warnings_out,
            "provenance":      self.provenance,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  Concrete adapters — Hb Regression (XGBoost)
# ══════════════════════════════════════════════════════════════════════════════

class FullHandXGBAdapter(BaseHandcraftedAdapter):
    modality   = "full_hand"
    model_name = "Full-hand Handcrafted XGBoost Regressor"
    provenance = ("Trained on 175 Figshare participants (full-hand photos). "
                  "Participant-level split 175/37/38. "
                  "Hb targets: measured in g/L, converted to g/dL. "
                  "Test set n=38 participants. MAE=1.5819 g/dL, RMSE=2.0355 g/dL, R²=0.3157.")
    _pkl_path  = FULLHAND_XGB_REG_PKL


class NailXGBAdapter(BaseHandcraftedAdapter):
    modality   = "nail"
    model_name = "Nail Handcrafted XGBoost Regressor"
    provenance = ("Trained on nail-crop images from 250 Figshare participants (503 train / 107 val / 109 test crops). "
                  "One to three crops per participant; model sees one crop at inference. "
                  "Hb targets: measured g/dL. Test set n=109 nail crops across 38 participants. "
                  "MAE=1.7171 g/dL, RMSE=2.3079 g/dL, R²=0.0548.")
    _pkl_path  = NAIL_XGB_REG_PKL


class EyeXGBAdapter(BaseHandcraftedAdapter):
    modality   = "eye"
    model_name = "Eye Handcrafted XGBoost Regressor"
    provenance = ("Trained on 610 deduplicated conjunctiva images from 202 participants (India + Italy; 432/89/89 split). "
                  "Test set n=89 images. MAE=1.4177 g/dL, RMSE=1.9744 g/dL, R²=0.4912.")
    _pkl_path  = EYE_XGB_REG_PKL


class FullHandScratchCNNXGBAdapter(ModelAdapter):
    """
    Full-hand Scratch-CNN Embedding + XGBoost Regressor adapter.
    Safely imports FullHandScratchCNN and transform from train_fullhand_scratch_cnn.py.
    """
    modality   = "full_hand"
    model_name = "Full-hand Scratch-CNN (64-D) + XGBoost Regressor"
    provenance = ("Compact 4-stage CNN trained from scratch on 175 Full-hand training participants "
                  "(selected by validation MAE), followed by XGBoost on 64-D embeddings. "
                  "Test set n=38 participants. MAE=1.5599 g/dL, RMSE=1.8946 g/dL, R²=0.4072. "
                  "Note: 64-D CNN embedding dimensions are opaque latent features.")

    def __init__(self):
        self._bundle = None
        self._scaler = None
        self._xgb    = None
        self._emb_cols = None
        self._cnn    = None
        self._transform = None

    def load(self):
        import torch
        from train_fullhand_scratch_cnn import FullHandScratchCNN, transform

        self._bundle = _load_bundle(FULLHAND_SCRATCH_XGB_PKL)
        self._scaler = self._bundle["scaler"]
        self._xgb    = self._bundle["xgb"]
        self._emb_cols = self._bundle.get("embedding_columns")
        if not self._emb_cols or len(self._emb_cols) != 64:
            raise ValueError(f"Expected 64 embedding columns, got {self._emb_cols}")

        if not FULLHAND_SCRATCH_PTH.exists():
            raise FileNotFoundError(f"Scratch CNN checkpoint missing: {FULLHAND_SCRATCH_PTH}")
        saved = torch.load(FULLHAND_SCRATCH_PTH, map_location="cpu", weights_only=True)
        cnn = FullHandScratchCNN().cpu()
        cnn.load_state_dict(saved["state_dict"], strict=True)
        cnn.eval()
        self._cnn = cnn
        self._transform = transform
        return self

    def predict_image(self, image_path) -> dict:
        if self._cnn is None or self._xgb is None:
            raise RuntimeError("Call .load() before predict_image().")
        import torch
        import pandas as pd
        from image_quality import check_image_quality

        path = Path(image_path)
        qr = check_image_quality(path)
        if not qr["is_readable"]:
            return {
                "modality":        self.modality,
                "model_name":      self.model_name,
                "hb_g_dl":         None,
                "screening_label": None,
                "screening_score": None,
                "quality_issues":  qr["warnings"],
                "warnings":        ["Image is unreadable — prediction aborted."],
                "provenance":      self.provenance,
            }

        try:
            tensor = self._transform(str(path), augment=False).unsqueeze(0)
            with torch.inference_mode():
                emb = self._cnn.get_embedding(tensor).cpu().numpy().astype(np.float32)
            if emb.shape != (1, 64) or not np.isfinite(emb).all():
                raise ValueError(f"Invalid 64-D embedding shape/values: {emb.shape}")
            emb_df = pd.DataFrame(emb, columns=self._emb_cols)
            X_scaled = self._scaler.transform(emb_df)
            hb_pred = float(self._xgb.predict(X_scaled)[0])
        except Exception as exc:
            return {
                "modality":        self.modality,
                "model_name":      self.model_name,
                "hb_g_dl":         None,
                "screening_label": None,
                "screening_score": None,
                "quality_issues":  qr["warnings"],
                "warnings":        [f"Scratch-CNN inference failed: {exc}"],
                "provenance":      self.provenance,
            }

        warnings_out = [
            "64-D CNN embedding dimensions are opaque latent features with no direct biological meaning."
        ]
        HB_MIN, HB_MAX = 3.0, 20.0
        if hb_pred < HB_MIN or hb_pred > HB_MAX:
            warnings_out.append(
                f"Predicted Hb {hb_pred:.2f} g/dL outside [{HB_MIN}, {HB_MAX}] g/dL."
            )
        hb_pred = float(np.clip(hb_pred, HB_MIN, HB_MAX))

        return {
            "modality":        self.modality,
            "model_name":      self.model_name,
            "hb_g_dl":         hb_pred,
            "screening_label": None,
            "screening_score": None,
            "quality_issues":  qr["warnings"],
            "warnings":        warnings_out,
            "provenance":      self.provenance,
        }


class FusedHandNailXGBAdapter(ModelAdapter):
    """
    Paired Full-hand + Nail (crop-averaged) XGBoost Regressor adapter.
    Requires both a Full-hand image and 1..3 paired Nail crop images from the same photograph.
    """
    modality   = "fused_full_hand_and_nail"
    model_name = "Paired Full-hand + Nail (Mean-Aggregated) XGBoost Regressor"
    provenance = ("Trained on 175 paired Figshare participants (concatenated 130 Full-hand + "
                  "130 mean-aggregated Nail handcrafted features = 260-D). "
                  "Selected on 37 validation participants; evaluated on 38 test participants. "
                  "Test MAE=1.6792 g/dL, RMSE=2.2338 g/dL, R²=0.1759. "
                  "Both views originate from the same source hand photograph.")
    _pkl_path  = M5_FUSION / "fused_model.pkl"

    def __init__(self):
        self._bundle = None
        self._scaler = None
        self._xgb    = None
        self._full_cols = None
        self._nail_cols = None

    def load(self):
        self._bundle = _load_bundle(self._pkl_path)
        self._scaler = self._bundle["scaler"]
        self._xgb    = self._bundle["xgb"]
        self._full_cols = [c.removeprefix("full_") for c in self._bundle["full_feat_cols"]]
        self._nail_cols = [c.removeprefix("nail_") for c in self._bundle["nail_feat_cols"]]
        return self

    def predict_image(self, image_path) -> dict:
        return {
            "modality":        self.modality,
            "model_name":      self.model_name,
            "hb_g_dl":         None,
            "screening_label": None,
            "screening_score": None,
            "quality_issues":  [],
            "warnings":        ["Fused adapter requires paired Full-hand and Nail crop images via predict_paired(full_path, nail_paths)."],
            "provenance":      self.provenance,
        }

    def predict_paired(self, full_image_path, nail_image_paths: list) -> dict:
        if self._xgb is None:
            raise RuntimeError("Call .load() before predict_paired().")
        from image_quality import check_image_quality

        if not nail_image_paths:
            raise ValueError("At least one Nail crop image path is required for paired fusion.")

        all_warnings = []
        qr_full = check_image_quality(full_image_path)
        if not qr_full["is_readable"]:
            return {
                "modality":        self.modality,
                "model_name":      self.model_name,
                "hb_g_dl":         None,
                "screening_label": None,
                "screening_score": None,
                "quality_issues":  qr_full["warnings"],
                "warnings":        ["Full-hand image unreadable — fusion aborted."],
                "provenance":      self.provenance,
            }
        all_warnings.extend([f"Full-hand: {w}" for w in qr_full["warnings"]])

        nail_vecs = []
        for idx, np_path in enumerate(nail_image_paths):
            qr_n = check_image_quality(np_path)
            if not qr_n["is_readable"]:
                return {
                    "modality":        self.modality,
                    "model_name":      self.model_name,
                    "hb_g_dl":         None,
                    "screening_label": None,
                    "screening_score": None,
                    "quality_issues":  qr_n["warnings"],
                    "warnings":        [f"Nail crop #{idx+1} unreadable — fusion aborted."],
                    "provenance":      self.provenance,
                }
            all_warnings.extend([f"Nail #{idx+1}: {w}" for w in qr_n["warnings"]])
            nail_vecs.append(_extract_and_align(Path(np_path), self._nail_cols))

        X_full = _extract_and_align(Path(full_image_path), self._full_cols)
        X_nail_mean = np.mean(np.vstack(nail_vecs), axis=0, keepdims=True).astype(np.float32)
        X_fused = np.hstack([X_full, X_nail_mean])
        if X_fused.shape != (1, 260):
            raise ValueError(f"Expected (1, 260) fused vector, got {X_fused.shape}")

        X_scaled = self._scaler.transform(X_fused)
        hb_pred = float(np.clip(float(self._xgb.predict(X_scaled)[0]), 3.0, 20.0))

        return {
            "modality":        self.modality,
            "model_name":      self.model_name,
            "hb_g_dl":         hb_pred,
            "screening_label": None,
            "screening_score": None,
            "quality_issues":  all_warnings,
            "warnings": [
                f"Aggregated {len(nail_vecs)} Nail crop(s) via participant-level mean.",
                "Both Full-hand and Nail crops must originate from the same source photograph.",
            ],
            "provenance":      self.provenance,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  Classification adapters
# ══════════════════════════════════════════════════════════════════════════════

class BaseClassificationAdapter(ModelAdapter):
    modality:   str = "unknown"
    model_name: str = "unknown"
    provenance: str = "unknown"
    _pkl_path:  Path = None
    label_note: str = ""

    def __init__(self):
        self._bundle    = None
        self._scaler    = None
        self._model     = None
        self._feat_cols = None

    def load(self):
        self._bundle    = _load_bundle(self._pkl_path)
        self._scaler    = self._bundle["scaler"]
        self._model     = self._bundle.get("rf")   # classification bundles use 'rf'
        if self._model is None:
            raise KeyError(f"Bundle {self._pkl_path.name} missing 'rf' key.")
        self._feat_cols = _get_feat_cols(self._bundle)
        return self

    def predict_image(self, image_path) -> dict:
        if self._model is None:
            raise RuntimeError("Call .load() before predict_image().")
        path = Path(image_path)

        from image_quality import check_image_quality
        qr = check_image_quality(path)
        if not qr["is_readable"]:
            return {
                "modality":        self.modality,
                "model_name":      self.model_name,
                "hb_g_dl":         None,
                "screening_label": None,
                "screening_score": None,
                "quality_issues":  qr["warnings"],
                "warnings":        ["Image unreadable — aborted."],
                "provenance":      self.provenance,
            }

        try:
            X = _extract_and_align(path, self._feat_cols)
        except Exception as exc:
            return {
                "modality":        self.modality,
                "model_name":      self.model_name,
                "hb_g_dl":         None,
                "screening_label": None,
                "screening_score": None,
                "quality_issues":  qr["warnings"],
                "warnings":        [f"Feature extraction failed: {exc}"],
                "provenance":      self.provenance,
            }

        X_scaled = self._scaler.transform(X)
        label    = int(self._model.predict(X_scaled)[0])
        prob     = float(self._model.predict_proba(X_scaled)[0, 1])

        return {
            "modality":        self.modality,
            "model_name":      self.model_name,
            "hb_g_dl":         None,
            "screening_label": label,
            "screening_score": round(prob, 4),
            "quality_issues":  qr["warnings"],
            "warnings": [
                "This is an EXPERIMENTAL screening score, NOT a diagnosis or calibrated medical risk.",
                self.label_note,
            ],
            "provenance":      self.provenance,
        }


class EyeClassificationAdapter(BaseClassificationAdapter):
    modality   = "eye"
    model_name = "Eye Handcrafted RF Classifier"
    label_note = ("Labels derived using inherited WHO sex-specific Hb cutoffs "
                  "(Male <13 g/dL, Female <12 g/dL; missing pregnancy/altitude context and 19 subjects >65). "
                  "Test accuracy=85.39%, F1=0.8471, ROC-AUC=0.9604.")
    provenance = "Eye dataset, participant-level split, n=89 test images."
    _pkl_path  = EYE_CLS_PKL


class NailClassificationAdapter(BaseClassificationAdapter):
    modality   = "nail"
    model_name = "Nail Handcrafted RF Classifier"
    label_note = ("Labels derived using 11 g/dL universal Hb threshold "
                  "(no gender/age information available in Figshare dataset). "
                  "Severe class imbalance — 83% non-anemic. "
                  "F1=0.10; classification unreliable on this dataset.")
    provenance = "Figshare nail dataset, participant-level split, n=109 test crops."
    _pkl_path  = NAIL_CLS_PKL


# ══════════════════════════════════════════════════════════════════════════════
#  Registry
# ══════════════════════════════════════════════════════════════════════════════

AVAILABLE_ADAPTERS = {
    "full_hand_regression":          FullHandXGBAdapter,
    "nail_regression":               NailXGBAdapter,
    "eye_regression":                EyeXGBAdapter,
    "full_hand_scratch_cnn_xgb":     FullHandScratchCNNXGBAdapter,
    "fused_hand_nail_regression":    FusedHandNailXGBAdapter,
    "eye_classification":            EyeClassificationAdapter,
    "nail_classification":           NailClassificationAdapter,
}

