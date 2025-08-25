# biopigment_fullres (minimal scaffold)

**What it does**
- Loads full-resolution petri-dish photos.
- Detects the dish via Hough circles and erodes the rim.
- Converts to CIELAB, clusters dish pixels in a*–b* with K-means (fixed K).
- Saves per-cluster statistics and QC figures (overlay, 2D a*–b* histogram).
- Outputs `results/metrics.csv` and `results/qc.csv`.

**Folder layout**
biopigment_fullres/
analyze_images.py
config.yaml
requirements.txt
data/ # put your JPGs here (e.g., SAM_2673.JPG)
results/ # will be created

**Install (fresh venv recommended)**
pip install -r requirements.txt

**Run**
python analyze_images.py config.yaml

**Config notes**
- Works at **full resolution**. No resizing.
- `dish.min_radius_px` and `max_radius_px` tuned for 4608×3456 images.
- Set `clustering.K` to the number of color clusters (default 3).

**Outputs**
- `results/metrics.csv` — one row per image×cluster with L*, a*, b* stats, chroma (C*) and hue.
- `results/qc.csv` — dish detection diagnostics.
- `results/figures/*_overlay.png` — dish boundary + cluster pseudocolor overlay.
- `results/figures/*_ab_hist2d.png` — a*×b* density.
- `results/labels/*_labels.png` — 16-bit label image (−1 mapped to 65535).

**Next steps**
- Add Munsell card calibration.
- Add K auto-selection (silhouette/BIC) and longitudinal summaries.
- Add HTML report.
