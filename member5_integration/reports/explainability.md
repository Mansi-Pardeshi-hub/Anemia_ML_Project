# Explainability Report — Handcrafted XGBoost Hb Regression

**Status:** SUCCESS (TreeSHAP)  
**Model Explained:** `Full-hand Handcrafted XGBoost Regressor (full_hand_xgb_regression.pkl)`  
**Method:** `shap.TreeExplainer (interventional, with XGBoost 3.x UBJ base_score string normalizer)`  
**Execution Note:** SHAP succeeded via shap.TreeExplainer (interventional, with XGBoost 3.x UBJ base_score string normalizer).

---

## 1. Rationale for Selecting the Handcrafted XGBoost Model

We perform feature-level explainability on the **handcrafted XGBoost regressor** (130 named colour, histogram, HSV/LAB, pallor/redness, and GLCM texture features) rather than the CNN-embedding models:
- Handcrafted features have explicit mathematical definitions (`R_mean`, `paleness_index`, `HSV_S_p50`, `LAB_A_mean`, `glcm_contrast_mean`).
- **CNN embedding dimensions (`embedding_000`..`embedding_1023` or `embedding_000`..`embedding_063`) are opaque latent activations** learned by convolutional layers. Assigning biological or clinical meaning to individual embedding indices is scientifically invalid.
- All feature ordering strictly follows the saved `bundle["feat_cols"]` contract.
- The SHAP background distribution is built **exclusively from the 175 training participants** (`split == "train"`).

---

## 2. Global Feature Importance (Training Background n=175)

Saved figure: `member5_integration/outputs/explainability/shap_global_summary.png`

| Rank | Feature Name | Mean \|SHAP\| (g/dL) |
|---:|---|---:|
| 1 | `HSV_H_p10` | 0.4466 |
| 2 | `G_hist_5` | 0.3994 |
| 3 | `B_hist_10` | 0.3542 |
| 4 | `G_hist_6` | 0.2765 |
| 5 | `G_std` | 0.2246 |
| 6 | `B_hist_5` | 0.1866 |
| 7 | `G_hist_7` | 0.1856 |
| 8 | `glcm_correlation_mean` | 0.1698 |
| 9 | `G_p75` | 0.1443 |
| 10 | `R_hist_6` | 0.1427 |

---

## 3. Local Explanation for a Single Held-Out Test Image

- **Participant ID:** `6` (`split == "test"`)
- **True Laboratory Hb:** `13.8 g/dL`
- **Model Predicted Hb:** `15.0721 g/dL`
- **Expected Base Value (Train Mean):** `12.8081 g/dL`
- **Saved waterfall plot:** `member5_integration/outputs/explainability/shap_local_example.png`
- **Saved full feature table:** `member5_integration/outputs/explainability/shap_local_example.csv`

### Top Positive Contributors (Increasing Predicted Hb) for Participant `6`

| Feature | Raw Value | SHAP Contribution |
|---|---:|---:|
| `G_hist_6` | 0.0368 | +0.4793 g/dL |
| `G_hist_5` | 0.0093 | +0.3322 g/dL |
| `HSV_H_p10` | 14.0000 | +0.2470 g/dL |
| `glcm_correlation_mean` | 0.9507 | +0.2156 g/dL |
| `G_p75` | 249.0000 | +0.1972 g/dL |

### Top Negative Contributors (Decreasing Predicted Hb) for Participant `6`

| Feature | Raw Value | SHAP Contribution |
|---|---:|---:|
| `R_hist_4` | 0.0026 | -0.2766 g/dL |
| `B_hist_5` | 0.0539 | -0.1993 g/dL |
| `G_hist_3` | 0.0037 | -0.1341 g/dL |
| `G_hist_4` | 0.0044 | -0.1164 g/dL |
| `R_hist_14` | 0.2161 | -0.0843 g/dL |

---

## 4. Scientific Limitations of Explainability

1. **Model Attribution, Not Physiological Causation:** SHAP values quantify how each image statistic shifts the XGBoost prediction relative to the training baseline. They do **not** prove that a colour channel or histogram bin reflects true subcutaneous hemoglobin concentration.
2. **Illumination and Camera Confounding:** Whole-image RGB/HSV/LAB statistics and histograms are sensitive to ambient lighting, white balance, background pixels, and skin tone.
3. **Opaque CNN Embeddings:** For the CNN-embedding models (`m3_eye_cnn_embedding_xgb.pkl`, `m3_nail_cnn_embedding_xgb.pkl`, `fullhand_scratch_cnn_embedding_xgb.pkl`), individual embedding coordinates have no standalone physical meaning.
4. **Academic Prototype Only:** These explanations are for model debugging and academic inspection, not clinical decision-making.
