"""Build a self-contained HTML view from the current verified metrics CSV."""

import csv
import html
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / "member4_ml_modelling" / "outputs" / "metrics" / "all_metrics_summary.csv"
OUTPUT = ROOT / "member4_ml_modelling" / "reports" / "results_dashboard.html"
CNN_COMPARISON = METRICS.parent / "cnn_embedding_experiment" / "cnn_embedding_comparison_summary.csv"


def cell(value):
    return html.escape("" if value is None else str(value))


def main():
    with METRICS.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    sections = []
    for task, title in (("regression", "Hb regression (g/dL)"), ("classification", "Exploratory classification")):
        selected = [r for r in rows if r.get("task", "").lower() == task]
        headers = ["model", "dataset", "n_test" if task == "regression" else "n_samples"]
        headers += (["MAE_g/dL", "RMSE_g/dL", "R2"] if task == "regression" else
                    ["accuracy", "precision", "recall", "specificity", "f1_score", "roc_auc"])
        out = [f"<section><h2>{title}</h2><div class='scroll'><table><thead><tr>"]
        out += [f"<th>{cell(h)}</th>" for h in headers]
        out.append("</tr></thead><tbody>")
        for row in selected:
            out.append("<tr>" + "".join(f"<td>{cell(row.get(h,''))}</td>" for h in headers) + "</tr>")
            caveat = row.get("caveats") or row.get("warning") or row.get("source")
            if caveat and "no verified CNN embedding matrix available" in caveat:
                caveat = "Handcrafted image features; CNN embedding results are reported in the separate paired-comparison section."
            if caveat:
                out.append(f"<tr class='caveat'><td colspan='{len(headers)}'>{cell(caveat)}</td></tr>")
        out.append("</tbody></table></div></section>")
        sections.append("".join(out))
    if CNN_COMPARISON.is_file():
        with CNN_COMPARISON.open(newline="", encoding="utf-8-sig") as stream:
            comparison = list(csv.DictReader(stream))
        headers = ["dataset", "model", "feature_type", "n_test", "MAE_g/dL", "RMSE_g/dL", "R2"]
        out = ["<section><h2>CNN embedding regression — paired held-out comparison</h2><div class='scroll'><table><thead><tr>"]
        out += [f"<th>{cell(h)}</th>" for h in headers]
        out.append("</tr></thead><tbody>")
        for row in comparison:
            out.append("<tr>" + "".join(f"<td>{cell(row.get(h,''))}</td>" for h in headers) + "</tr>")
        out.append("</tbody></table></div><p class='caveat'>Eye and Nail use Member 3 scratch-trained regression checkpoints. Full-hand includes two distinct comparisons: an exploratory Palm-classifier feature transfer (not Full-hand-trained) and a compact Member 4 CNN regressor trained from scratch on 175 Full-hand training participants, selected by validation MAE and scored on 38 held-out participants. All models are paired to identical held-out images against handcrafted XGBoost. Results remain small-sample academic experiments, not clinical validation.</p></section>")
        sections.append("".join(out))
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    page = """<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Member 4 verified results</title><style>
body{font:15px/1.5 system-ui,sans-serif;margin:0;background:#f4f6f8;color:#17212b}main{max-width:1200px;margin:32px auto;padding:0 18px}h1{margin-bottom:4px}section,.notice{background:#fff;border:1px solid #d9e0e7;border-radius:10px;padding:18px;margin:18px 0}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;padding:8px;border-bottom:1px solid #e5e9ed;white-space:nowrap}th{background:#edf2f6}.caveat td{white-space:normal;color:#46515b;font-size:13px;background:#fafbfc}.notice{border-left:5px solid #b7791f}small{color:#596773}
</style><main><h1>Member 4 — Verified model results</h1><small>Generated from the current all_metrics_summary.csv · STAMP_TOKEN</small>
<div class='notice'><b>Research prototype only.</b> Hb estimates and class outputs are not diagnoses or substitutes for CBC. Eye and Figshare thresholds are exploratory; Palm scores predict supplied folder labels. Member 3's original direct predictions lack IDs for row-wise comparison. Eye/Nail CNN embeddings and the separate Full-hand scratch-CNN experiment are image-keyed and compared on held-out images. No Member 3 Full-hand checkpoint or paired modalities were supplied; the Full-hand scratch model is a small-sample Member 4 experiment and no fusion is claimed.</div>
SECTIONS_TOKEN<p><small>Source: member4_ml_modelling/outputs/metrics/all_metrics_summary.csv. Regenerate with <code>python member4_ml_modelling/src/generate_results_dashboard.py</code>.</small></p></main></html>"""
    page = page.replace("STAMP_TOKEN", stamp).replace("SECTIONS_TOKEN", "".join(sections))
    OUTPUT.write_text(page, encoding="utf-8")
    print(f"Wrote {OUTPUT.relative_to(ROOT)} using {len(rows)} current metric rows")


if __name__ == "__main__":
    main()
