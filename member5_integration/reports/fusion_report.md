# Fusion Report — Nail + Full-hand

**Status: FUSION SUCCESSFULLY EVALUATED**  
**Test participants: 38 (untouched until final evaluation)**

## Results

| Model | MAE (g/dL) | RMSE (g/dL) | R² | n_test |
|-------|-----------|------------|-----|--------|
| Full-hand only | 1.5819 | 2.0355 | 0.3157 | 38 |
| Nail-only (participant-level mean) | 1.8289 | 2.4117 | 0.0394 | 38 |
| **Fused (Full + Nail)** | **1.6792** | **2.2338** | **0.1759** | **38** |

## Interpretation

- Fusion **did not improve** MAE over the best single-modality model.
- Fusion **did not improve** R² over the best single-modality model.
- With only 38 test participants the metric differences should be treated with caution.
  Small-sample variability can easily account for differences of < 0.1 MAE.

## Methodology

- Nail crops aggregated to participant level by **mean of handcrafted features**.
- Concatenated Full-hand (130 features) + Nail (130 features) → 260-dimensional input.
- XGBoost trained on 175 participants; hyperparameters selected on 37 validation participants.
- Evaluated once on 38 test participants.

## Important Caveats

1. Both inputs are **image views from the same source photograph** — Nail crops and Full-hand images
   were extracted from the same Figshare hand photographs. This limits the information gain from fusion.
2. The improvement (if any) could also reflect the larger feature space rather than genuine multimodal benefit.
3. We do **not** claim this is a "multimodal" system in the clinical sensing sense; both modalities derive from the same image.
4. All results are research prototypes. No clinical validity has been established.
