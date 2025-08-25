import os, sys, json, math, glob
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from skimage import morphology, measure
from sklearn.cluster import KMeans
from scipy import stats as spstats
import matplotlib.pyplot as plt
import yaml
from tqdm import tqdm

def load_config(cfg_path):
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def iqr(x):
    q1 = np.percentile(x, 25)
    q3 = np.percentile(x, 75)
    return float(q3 - q1)

def ecdf_vals(x):
    xs = np.sort(x)
    ys = np.arange(1, len(xs)+1) / len(xs)
    return xs, ys

def dish_mask_from_hough(img_bgr, min_r, max_r, rim_erosion_pct):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        gray_blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=500,
        param1=100, param2=40, minRadius=min_r, maxRadius=max_r
    )
    if circles is None:
        return None, None
    circles = np.uint16(np.around(circles[0]))
    c = circles[np.argmax(circles[:,2])]
    x, y, r = int(c[0]), int(c[1]), int(c[2])
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.circle(mask, (x, y), r, 255, thickness=-1)
    rim = max(1, int(r * rim_erosion_pct / 100.0))
    eroded = cv2.erode(
        mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*rim+1, 2*rim+1))
    )
    return eroded, (x, y, r, rim)

def colony_mask_from_L(L, min_area_px, block_size, C):
    L_uint8 = np.clip((L/100.0)*255.0, 0, 255).astype(np.uint8)
    thr = cv2.adaptiveThreshold(
        L_uint8, 255, cv2.ADAPTIVE_THRESH_MEAN_,
        cv2.THRESH_BINARY_INV, block_size, C
    )
    thr = morphology.remove_small_objects(thr.astype(bool), min_size=min_area_px)
    thr = morphology.binary_closing(thr, morphology.disk(3))
    return thr.astype(np.uint8)*255

def rgb_to_lab(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:,:,0] * (100.0/255.0)
    a = lab[:,:,1] - 128.0
    b = lab[:,:,2] - 128.0
    return L, a, b

def cluster_ab(a, b, mask, K, seed):
    coords = np.column_stack([a[mask>0].ravel(), b[mask>0].ravel()])
    if coords.shape[0] < K:
        raise ValueError("Not enough pixels inside mask for clustering")
    km = KMeans(n_clusters=K, random_state=seed, n_init="auto")
    labels = km.fit_predict(coords)
    lab_img = np.full(a.shape, fill_value=-1, dtype=np.int32)
    lab_img[mask>0] = labels
    centers = km.cluster_centers_
    return lab_img, centers

def per_cluster_stats(L, a, b, labels, K):
    rows = []
    for k in range(K):
        sel = labels==k
        n = int(sel.sum())
        if n == 0:
            rows.append(dict(cluster=k, pixel_count=0, area_pct=0.0,
                             L_mean=np.nan,L_median=np.nan,L_sd=np.nan,L_iqr=np.nan,
                             a_mean=np.nan,a_median=np.nan,a_sd=np.nan,a_iqr=np.nan,
                             b_mean=np.nan,b_median=np.nan,b_sd=np.nan,b_iqr=np.nan,
                             C_mean=np.nan,C_median=np.nan,hue_mean_deg=np.nan,hue_median_deg=np.nan))
            continue
        Lk = L[sel]; ak = a[sel]; bk = b[sel]
        Ck = np.sqrt(ak**2 + bk**2)
        hue = np.degrees(np.arctan2(bk, ak))
        rows.append(dict(
            cluster=k,
            pixel_count=n,
            area_pct=100.0 * n / labels.size,
            L_mean=float(np.mean(Lk)),
            L_median=float(np.median(Lk)),
            L_sd=float(np.std(Lk, ddof=1)) if n>1 else 0.0,
            L_iqr=float(np.percentile(Lk,75) - np.percentile(Lk,25)) if n>1 else 0.0,
            a_mean=float(np.mean(ak)),
            a_median=float(np.median(ak)),
            a_sd=float(np.std(ak, ddof=1)) if n>1 else 0.0,
            a_iqr=float(np.percentile(ak,75) - np.percentile(ak,25)) if n>1 else 0.0,
            b_mean=float(np.mean(bk)),
            b_median=float(np.median(bk)),
            b_sd=float(np.std(bk, ddof=1)) if n>1 else 0.0,
            b_iqr=float(np.percentile(bk,75) - np.percentile(bk,25)) if n>1 else 0.0,
            C_mean=float(np.mean(Ck)),
            C_median=float(np.median(Ck)),
            hue_mean_deg=float(np.mean(hue)),
            hue_median_deg=float(np.median(hue))
        ))
    return rows

def save_overlay(image_bgr, labels, circle_info, out_path, dpi=200):
    K = np.max(labels)+1 if (labels>=0).any() else 0
    overlay = image_bgr.copy()
    if K>0:
        colors = []
        rng = np.random.default_rng(123)
        for _ in range(K):
            colors.append((int(rng.integers(0,255)), int(rng.integers(0,255)), int(rng.integers(0,255))))
        for k in range(K):
            m = (labels==k)
            overlay[m] = (0.6*overlay[m] + 0.4*np.array(colors[k])).astype(np.uint8)
    x,y,r,er = circle_info
    cv2.circle(overlay, (x,y), r, (255,255,255), 3)
    cv2.circle(overlay, (x,y), r-er, (0,0,0), 2)
    cv2.imwrite(out_path, overlay)

def hist2d_plot(a, b, mask, out_path, bins_a=64, bins_b=64, dpi=200):
    aa = a[mask>0].ravel()
    bb = b[mask>0].ravel()
    plt.figure()
    plt.hist2d(aa, bb, bins=[bins_a, bins_b])
    plt.xlabel("a*")
    plt.ylabel("b*")
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()

def run(config_path):
    cfg = load_config(config_path)
    img_glob = cfg["io"]["image_glob"]
    out_dir = Path(cfg["io"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = out_dir / "figures"
    labels_dir = out_dir / "labels"
    figures_dir.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)

    min_r = int(cfg["dish"]["min_radius_px"])
    max_r = int(cfg["dish"]["max_radius_px"])
    rim_pct = float(cfg["dish"]["rim_erosion_pct"])

    use_colony = bool(cfg["colony"]["use_colony_mask"])
    block = int(cfg["colony"]["thresh_block"])
    C = int(cfg["colony"]["thresh_C"])
    min_area_px = int(cfg["colony"]["min_area_px"])

    K = int(cfg["clustering"]["K"])
    seed = int(cfg["clustering"]["seed"])
    bins_a = int(cfg["distributions"]["bins_a"])
    bins_b = int(cfg["distributions"]["bins_b"])
    dpi = int(cfg["export"]["figures_dpi"])
    save_label_png = bool(cfg["export"]["save_label_png"])

    rows = []
    qc_rows = []

    images = sorted(glob.glob(img_glob))
    if not images:
        print(f"No images matched {img_glob}")
        return

    for ipath in tqdm(images, desc="Processing images"):
        img = cv2.imread(ipath, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Could not read {ipath}")
            continue
        h, w = img.shape[:2]
        mask_dish, circle_info = dish_mask_from_hough(img, min_r, max_r, rim_pct)
        qc = {"image": os.path.basename(ipath), "width": w, "height": h, "dish_found": bool(mask_dish is not None)}
        if mask_dish is None:
            qc_rows.append(qc)
            continue

        L, a, b = rgb_to_lab(img)

        if use_colony:
            col_mask = colony_mask_from_L(L, min_area_px, block, C)
            mask = cv2.bitwise_and(mask_dish, col_mask)
            qc["colony_mask_used"] = True
        else:
            mask = mask_dish
            qc["colony_mask_used"] = False

        try:
            labels, centers = cluster_ab(a, b, mask, K, seed)
        except Exception as e:
            qc["error"] = str(e)
            qc_rows.append(qc)
            continue

        stats_rows = per_cluster_stats(L, a, b, labels, K)
        for r in stats_rows:
            r.update({"image": os.path.basename(ipath), "K": K})
        rows.extend(stats_rows)

        base = Path(os.path.basename(ipath)).stem
        if save_label_png:
            out_overlay = figures_dir / f"{base}_overlay.png"
            save_overlay(img, labels, circle_info, str(out_overlay), dpi=dpi)
        out_hist2d = figures_dir / f"{base}_ab_hist2d.png"
        hist2d_plot(a, b, mask, str(out_hist2d), bins_a=bins_a, bins_b=bins_b, dpi=dpi)

        lab_out = labels_dir / f"{base}_labels.png"
        lab_to_save = labels.astype(np.int32).copy()
        lab_to_save[lab_to_save<0] = 65535
        cv2.imwrite(str(lab_out), lab_to_save.astype(np.uint16))

        x,y,r,er = circle_info
        qc.update({"dish_radius_px": int(r), "rim_erosion_px": int(er)})
        qc_rows.append(qc)

    if rows:
        df = pd.DataFrame(rows)[[
            "image","K","cluster","pixel_count","area_pct",
            "L_mean","L_median","L_sd","L_iqr",
            "a_mean","a_median","a_sd","a_iqr",
            "b_mean","b_median","b_sd","b_iqr",
            "C_mean","C_median","hue_mean_deg","hue_median_deg"
        ]]
        df.to_csv(out_dir / "metrics.csv", index=False)
    if qc_rows:
        pd.DataFrame(qc_rows).to_csv(out_dir / "qc.csv", index=False)

if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv)>1 else "config.yaml"
    run(cfg)
