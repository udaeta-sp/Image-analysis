import os
import math
from pathlib import Path
import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

def load_tables(results_dir: Path):
    metrics = pd.read_csv(results_dir / "metrics.csv") if (results_dir / "metrics.csv").exists() else pd.DataFrame()
    qc = pd.read_csv(results_dir / "qc.csv") if (results_dir / "qc.csv").exists() else pd.DataFrame()
    return metrics, qc

def compute_summaries(metrics: pd.DataFrame):
    if metrics.empty:
        return dict(per_image={}, overall=None)

    # Per-image cluster summaries (area % and a*,b* medians)
    per_image = {}
    for image, df in metrics.groupby("image"):
        df = df.copy()
        # area percent by cluster
        area = df[["cluster","area_pct"]].set_index("cluster")["area_pct"].to_dict()
        # medians
        med = df.set_index("cluster")[["a_median","b_median","C_median","hue_median_deg"]].to_dict(orient="index")
        per_image[image] = dict(area_pct=area, medians=med)
    # Overall summary (mean area % per cluster index across images)
    overall = metrics.groupby(["cluster"])["area_pct"].mean().reset_index().sort_values("cluster")
    return dict(per_image=per_image, overall=overall)

def image_assets(results_dir: Path):
    figs = results_dir / "figures"
    if not figs.exists():
        return {}
    assets = {}
    for p in figs.glob("*_overlay.png"):
        base = p.stem.replace("_overlay","")
        hist = figs / f"{base}_ab_hist2d.png"
        assets[base] = dict(overlay=str(p.name), hist=str(hist.name) if hist.exists() else None)
    return assets

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Generate HTML report for biopigment_fullres results.")
    ap.add_argument("--results", default="results", help="Results directory containing metrics.csv, qc.csv, figures/")
    ap.add_argument("--out", default="report.html", help="Output HTML path")
    args = ap.parse_args()

    results_dir = Path(args.results)
    metrics, qc = load_tables(results_dir)
    summaries = compute_summaries(metrics)
    assets = image_assets(results_dir)

    env = Environment(
        loader=FileSystemLoader(str(Path(__file__).parent)),
        autoescape=select_autoescape()
    )
    tpl = env.get_template("report_template.html")
    html = tpl.render(
        metrics=metrics.to_dict(orient="records") if not metrics.empty else [],
        qc=qc.to_dict(orient="records") if not qc.empty else [],
        assets=assets,
        per_image=summaries["per_image"],
        overall=(summaries["overall"].to_dict(orient="records") if summaries["overall"] is not None else None),
        results_dir="results"
    )
    Path(args.out).write_text(html, encoding="utf-8")
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
