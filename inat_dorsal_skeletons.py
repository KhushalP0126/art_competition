import argparse
import csv
import json
import math
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image, ImageDraw, ImageFont


GBIF_OCC_API = "https://api.gbif.org/v1/occurrence/search"
BUTTERFLY_FAMILY_KEYS = [9417, 7017, 5481, 5473, 6953, 1933999]  # Papilionidae..Riodinidae


def fetch_json(url, timeout=60):
    req = Request(url, headers={"User-Agent": "museum-butterfly-skeleton/1.0"})
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def download_image(url, timeout=60):
    req = Request(url, headers={"User-Agent": "museum-butterfly-skeleton/1.0"})
    with urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    return Image.open(BytesIO(data)).convert("RGB")


def rgb_to_gray(arr):
    return 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]


def sobel_like(gray):
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    return np.hypot(gx, gy)


def smooth2d(arr, iters=1):
    out = arr.astype(np.float32, copy=False)
    for _ in range(max(1, int(iters))):
        p = np.pad(out, 1, mode="edge")
        out = (
            p[:-2, :-2]
            + 2.0 * p[:-2, 1:-1]
            + p[:-2, 2:]
            + 2.0 * p[1:-1, :-2]
            + 4.0 * p[1:-1, 1:-1]
            + 2.0 * p[1:-1, 2:]
            + p[2:, :-2]
            + 2.0 * p[2:, 1:-1]
            + p[2:, 2:]
        ) / 16.0
    return out


def smooth3d(arr, iters=1):
    out = arr.astype(np.float32, copy=False)
    for _ in range(max(1, int(iters))):
        out = np.stack([smooth2d(out[..., c], 1) for c in range(out.shape[2])], axis=2)
    return out


def binary_dilate(mask, iters=1):
    m = mask
    for _ in range(iters):
        p = np.pad(m, 1, mode="constant")
        m = (
            p[:-2, :-2]
            | p[:-2, 1:-1]
            | p[:-2, 2:]
            | p[1:-1, :-2]
            | p[1:-1, 1:-1]
            | p[1:-1, 2:]
            | p[2:, :-2]
            | p[2:, 1:-1]
            | p[2:, 2:]
        )
    return m


def binary_erode(mask, iters=1):
    m = mask
    for _ in range(iters):
        p = np.pad(m, 1, mode="constant")
        m = (
            p[:-2, :-2]
            & p[:-2, 1:-1]
            & p[:-2, 2:]
            & p[1:-1, :-2]
            & p[1:-1, 1:-1]
            & p[1:-1, 2:]
            & p[2:, :-2]
            & p[2:, 1:-1]
            & p[2:, 2:]
        )
    return m


def transitions(p2, p3, p4, p5, p6, p7, p8, p9):
    seq = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
    t = np.zeros_like(p2, dtype=np.uint8)
    for i in range(8):
        t += ((seq[i] == 0) & (seq[i + 1] == 1)).astype(np.uint8)
    return t


def zhang_suen_thinning(mask, max_iters=120):
    img = mask.astype(np.uint8).copy()
    for _ in range(max_iters):
        changed = False
        p = np.pad(img, 1, mode="constant")
        p2 = p[:-2, 1:-1]
        p3 = p[:-2, 2:]
        p4 = p[1:-1, 2:]
        p5 = p[2:, 2:]
        p6 = p[2:, 1:-1]
        p7 = p[2:, :-2]
        p8 = p[1:-1, :-2]
        p9 = p[:-2, :-2]
        b = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
        a = transitions(p2, p3, p4, p5, p6, p7, p8, p9)
        m1 = (img == 1) & (b >= 2) & (b <= 6) & (a == 1) & ((p2 * p4 * p6) == 0) & ((p4 * p6 * p8) == 0)
        if m1.any():
            img[m1] = 0
            changed = True

        p = np.pad(img, 1, mode="constant")
        p2 = p[:-2, 1:-1]
        p3 = p[:-2, 2:]
        p4 = p[1:-1, 2:]
        p5 = p[2:, 2:]
        p6 = p[2:, 1:-1]
        p7 = p[2:, :-2]
        p8 = p[1:-1, :-2]
        p9 = p[:-2, :-2]
        b = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
        a = transitions(p2, p3, p4, p5, p6, p7, p8, p9)
        m2 = (img == 1) & (b >= 2) & (b <= 6) & (a == 1) & ((p2 * p4 * p8) == 0) & ((p2 * p6 * p8) == 0)
        if m2.any():
            img[m2] = 0
            changed = True
        if not changed:
            break
    return img.astype(bool)


def resize_max_dim(img, max_dim=1024):
    w, h = img.size
    m = max(w, h)
    if m <= max_dim:
        return img
    s = max_dim / m
    return img.resize((int(w * s), int(h * s)), Image.Resampling.LANCZOS)


def dorsal_score(arr):
    gray = rgb_to_gray(arr.astype(np.float32) / 255.0)
    edge = sobel_like(gray)
    thr = np.percentile(edge, 86)
    emask = edge > thr
    emask[:6, :] = False
    emask[-6:, :] = False
    emask[:, :6] = False
    emask[:, -6:] = False
    if emask.sum() < 200:
        return 0.0

    ys, xs = np.where(emask)
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    w = max(1, x1 - x0 + 1)
    h = max(1, y1 - y0 + 1)
    spread = min(1.0, w / (h + 1e-8) / 2.0)

    wts = edge[emask]
    xw = xs.astype(float)
    yw = ys.astype(float)
    mx = (xw * wts).sum() / (wts.sum() + 1e-8)
    my = (yw * wts).sum() / (wts.sum() + 1e-8)
    xzc = xw - mx
    yzc = yw - my
    cxx = (wts * xzc * xzc).sum() / (wts.sum() + 1e-8)
    cyy = (wts * yzc * yzc).sum() / (wts.sum() + 1e-8)
    cxy = (wts * xzc * yzc).sum() / (wts.sum() + 1e-8)
    angle = 0.5 * math.atan2(2 * cxy, cxx - cyy + 1e-8)
    horiz = abs(math.cos(angle))

    cx = int(round(mx))
    crop = gray[y0 : y1 + 1, x0 : x1 + 1]
    rel_mid = np.clip(cx - x0, 1, crop.shape[1] - 2)
    left = crop[:, :rel_mid]
    right = crop[:, rel_mid:][:, ::-1]
    ww = min(left.shape[1], right.shape[1])
    if ww < 20:
        symmetry = 0.0
        balance = 0.0
    else:
        left = left[:, :ww]
        right = right[:, :ww]
        diff = np.mean(np.abs(left - right))
        symmetry = max(0.0, 1.0 - diff / 0.35)
        lm = emask[:, :cx].sum()
        rm = emask[:, cx:].sum()
        balance = 1.0 - abs(lm - rm) / (lm + rm + 1e-8)

    return float(0.45 * symmetry + 0.25 * horiz + 0.2 * spread + 0.1 * balance)


def specimen_in_view(arr):
    rgb = arr.astype(np.float32) / 255.0
    gray = rgb_to_gray(rgb)
    edge = sobel_like(gray)
    sat = (rgb.max(axis=2) - rgb.min(axis=2)) / (rgb.max(axis=2) + 1e-6)

    fg = (edge > np.percentile(edge, 84)) | (sat > 0.22)
    fg = binary_dilate(fg, 1)
    fg = binary_erode(fg, 1)

    ys, xs = np.where(fg)
    h, w = gray.shape
    if xs.size < 500:
        return False, {
            "area": 0.0,
            "bbox_w": 0.0,
            "bbox_h": 0.0,
            "center_offset": 1.0,
            "border_ratio": 1.0,
        }

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    bbox_w = (x1 - x0 + 1) / w
    bbox_h = (y1 - y0 + 1) / h
    area = xs.size / (w * h)

    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    dx = (cx - (w - 1) / 2.0) / (w / 2.0)
    dy = (cy - (h - 1) / 2.0) / (h / 2.0)
    center_offset = float(np.hypot(dx, dy))

    b = 6
    border = np.zeros_like(fg, dtype=bool)
    border[:b, :] = True
    border[-b:, :] = True
    border[:, :b] = True
    border[:, -b:] = True
    border_ratio = float(fg[border].sum() / (fg.sum() + 1e-8))

    ok = (
        area > 0.03
        and bbox_w > 0.35
        and bbox_h > 0.25
        and center_offset < 0.45
        and border_ratio < 0.35
    )
    return ok, {
        "area": float(area),
        "bbox_w": float(bbox_w),
        "bbox_h": float(bbox_h),
        "center_offset": float(center_offset),
        "border_ratio": float(border_ratio),
    }


def build_feature_mask(arr):
    rgb = arr.astype(np.float32) / 255.0
    rgb = smooth3d(rgb, 1)
    gray = rgb_to_gray(rgb)
    gray_s = smooth2d(gray, 1)
    edge = sobel_like(gray_s)
    edge = edge / (edge.max() + 1e-8)

    cmax = rgb.max(axis=2)
    cmin = rgb.min(axis=2)
    sat = (cmax - cmin) / (cmax + 1e-6)
    value = cmax
    paper_like = (sat < 0.08) & (value > 0.72)
    edge_mask = (edge > np.percentile(edge, 86)) & (~paper_like)

    color_mask = (sat > 0.12) & (value > 0.08)
    color_context = binary_dilate(color_mask, 2)
    dark_lines = (gray_s < np.percentile(gray_s, 31)) & color_context
    mask = edge_mask & (color_mask | dark_lines)

    # Encourage the highly bilateral dorsal structure and suppress label text clutter.
    symmetric_seed = mask & np.fliplr(mask)
    if symmetric_seed.sum() > 50:
        mask = mask & binary_dilate(symmetric_seed, 14)

    mask = binary_dilate(mask, 2)
    mask = binary_erode(mask, 1)
    mask = binary_dilate(mask, 1)

    # Remove isolated high-frequency fragments.
    dense = smooth2d(mask.astype(np.float32), 1)
    mask = mask & (dense > 0.18)
    if mask.sum() < 120:
        mask = binary_dilate((edge > np.percentile(edge, 90)) & color_context, 1)
    return mask


def boost_color(rgb):
    gray = rgb_to_gray(rgb)[..., None]
    # Push away from grayscale so wing colors stay visible on skeleton strokes.
    boosted = gray + 1.6 * (rgb - gray)
    return np.clip(boosted, 0.0, 1.0)


def build_detail_map(arr, feature_mask=None):
    rgb = arr.astype(np.float32) / 255.0
    rgb = smooth3d(rgb, 1)
    gray = rgb_to_gray(rgb)
    edge = sobel_like(smooth2d(gray, 1))
    edge = edge / (edge.max() + 1e-8)
    cmax = rgb.max(axis=2)
    cmin = rgb.min(axis=2)
    sat = (cmax - cmin) / (cmax + 1e-8)
    sat = smooth2d(sat, 1)
    detail = np.clip(0.70 * edge + 0.30 * sat, 0, 1)
    detail = smooth2d(detail, 1)
    if feature_mask is not None and feature_mask.any():
        region = binary_dilate(feature_mask, 2)
        detail = detail * region.astype(np.float32)
    return detail


def to_skeleton(arr):
    mask = build_feature_mask(arr)
    skel = zhang_suen_thinning(mask)

    rgb = arr.astype(np.float32) / 255.0
    col = boost_color(rgb)

    # Dark indigo background to make colored lines pop.
    out = np.zeros((*skel.shape, 3), dtype=np.float32)
    out[..., 0] = 0.02
    out[..., 1] = 0.025
    out[..., 2] = 0.055

    glow1 = binary_dilate(skel, 1) & (~skel)
    glow2 = binary_dilate(skel, 2) & (~binary_dilate(skel, 1))

    out[glow2] = np.maximum(out[glow2], 0.25 * col[glow2] + np.array([0.03, 0.03, 0.06]))
    out[glow1] = np.maximum(out[glow1], 0.45 * col[glow1] + np.array([0.04, 0.04, 0.07]))
    out[skel] = np.maximum(out[skel], col[skel])

    out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


def iter_image_paths(image_dir):
    exts = (".jpg", ".jpeg", ".png", ".webp")
    return sorted([p for p in Path(image_dir).glob("*") if p.suffix.lower() in exts])


def hf_embed_images(image_paths, model_name, batch_size):
    def offline_embed(paths):
        valid = []
        feats = []
        for p in paths:
            try:
                arr = np.asarray(Image.open(p).convert("RGB").resize((96, 96), Image.Resampling.BILINEAR))
                rgb = arr.astype(np.float32) / 255.0
                gray = rgb_to_gray(rgb)
                edge = sobel_like(gray)
                edge = edge / (edge.max() + 1e-8)
                sat = (rgb.max(axis=2) - rgb.min(axis=2)) / (rgb.max(axis=2) + 1e-8)
                g_hist, _ = np.histogram(gray, bins=16, range=(0.0, 1.0), density=True)
                s_hist, _ = np.histogram(sat, bins=16, range=(0.0, 1.0), density=True)
                e_hist, _ = np.histogram(edge, bins=16, range=(0.0, 1.0), density=True)
                row_mean = gray.mean(axis=1)
                col_mean = gray.mean(axis=0)
                row_sig = np.array([x.mean() for x in np.array_split(row_mean, 8)], dtype=np.float32)
                col_sig = np.array([x.mean() for x in np.array_split(col_mean, 8)], dtype=np.float32)
                vec = np.concatenate([g_hist, s_hist, e_hist, row_sig, col_sig]).astype(np.float32)
                vec = vec / (np.linalg.norm(vec) + 1e-8)
                valid.append(p)
                feats.append(vec)
            except Exception:
                continue
        if not feats:
            return [], np.zeros((0, 1), dtype=np.float32)
        return valid, np.vstack(feats).astype(np.float32)

    try:
        import torch
        from transformers import AutoImageProcessor, AutoModel
    except Exception as e:
        print(f"HF imports unavailable ({e}); using offline image-stat embeddings.")
        return offline_embed(image_paths)

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Loading HF model: {model_name} on {device}")
    try:
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
        model.eval()
    except Exception as e:
        print(f"HF model load failed ({e}); using offline image-stat embeddings.")
        return offline_embed(image_paths)

    embeddings = []
    valid_paths = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        imgs = []
        kept_paths = []
        for p in batch_paths:
            try:
                imgs.append(Image.open(p).convert("RGB"))
                kept_paths.append(p)
            except Exception:
                continue
        if not imgs:
            continue

        inputs = processor(images=imgs, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                vec = outputs.pooler_output
            else:
                vec = outputs.last_hidden_state[:, 0, :]
        vec = vec.detach().cpu().numpy()
        norm = np.linalg.norm(vec, axis=1, keepdims=True) + 1e-8
        vec = vec / norm

        embeddings.append(vec)
        valid_paths.extend(kept_paths)
        print(f"Embedded {len(valid_paths)}/{len(image_paths)} images")

    if not embeddings:
        return [], np.zeros((0, 1), dtype=np.float32)
    return valid_paths, np.vstack(embeddings).astype(np.float32)


def normalize_pattern(arr, canvas_h, canvas_w):
    mask = build_feature_mask(arr)
    skel = zhang_suen_thinning(mask)
    if skel.sum() < 50 or mask.sum() < 120:
        return None, None, None, None

    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    skel_crop = skel[y0 : y1 + 1, x0 : x1 + 1]
    mask_crop = mask[y0 : y1 + 1, x0 : x1 + 1]

    rgb = arr.astype(np.float32) / 255.0
    col = boost_color(rgb)[y0 : y1 + 1, x0 : x1 + 1, :]
    detail = build_detail_map(arr, feature_mask=mask)[y0 : y1 + 1, x0 : x1 + 1]

    # Enforce canonical symmetry for an "ideal dorsal" aggregate.
    skel_crop = skel_crop | np.fliplr(skel_crop)
    mask_crop = mask_crop | np.fliplr(mask_crop)
    col = 0.5 * (col + np.fliplr(col))
    detail = 0.5 * (detail + np.fliplr(detail))

    h, w = skel_crop.shape
    tw = int(canvas_w * 0.82)
    th = int(canvas_h * 0.72)
    scale = min(tw / max(w, 1), th / max(h, 1))
    nw = max(2, int(round(w * scale)))
    nh = max(2, int(round(h * scale)))

    skel_img = Image.fromarray((skel_crop.astype(np.uint8) * 255), mode="L").resize(
        (nw, nh), resample=Image.Resampling.NEAREST
    )
    col_img = Image.fromarray((np.clip(col, 0, 1) * 255).astype(np.uint8), mode="RGB").resize(
        (nw, nh), resample=Image.Resampling.BILINEAR
    )
    det_img = Image.fromarray((np.clip(detail, 0, 1) * 255).astype(np.uint8), mode="L").resize(
        (nw, nh), resample=Image.Resampling.BILINEAR
    )
    mask_img = Image.fromarray((mask_crop.astype(np.uint8) * 255), mode="L").resize(
        (nw, nh), resample=Image.Resampling.NEAREST
    )

    skel_resized = np.asarray(skel_img) > 127
    col_resized = np.asarray(col_img).astype(np.float32) / 255.0
    det_resized = np.asarray(det_img).astype(np.float32) / 255.0
    mask_resized = np.asarray(mask_img) > 127

    canvas_s = np.zeros((canvas_h, canvas_w), dtype=bool)
    canvas_c = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    canvas_d = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    canvas_m = np.zeros((canvas_h, canvas_w), dtype=bool)
    ox = (canvas_w - nw) // 2
    oy = (canvas_h - nh) // 2
    canvas_s[oy : oy + nh, ox : ox + nw] = skel_resized
    canvas_c[oy : oy + nh, ox : ox + nw, :] = col_resized
    canvas_d[oy : oy + nh, ox : ox + nw] = det_resized
    canvas_m[oy : oy + nh, ox : ox + nw] = mask_resized
    return canvas_s, canvas_c, canvas_d, canvas_m


def aggregate_dorsal_patterns(
    image_dir,
    hf_model,
    hf_batch_size,
    top_k,
    canvas_w,
    canvas_h,
    min_dorsal_score=0.6,
):
    paths = iter_image_paths(image_dir)
    if not paths:
        raise RuntimeError(f"No images found in {image_dir}")
    print(f"Found {len(paths)} source images")

    valid_paths, emb = hf_embed_images(paths, hf_model, hf_batch_size)
    if emb.shape[0] == 0:
        raise RuntimeError("No embeddings generated")

    centroid = emb.mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
    sims = emb @ centroid
    order = np.argsort(-sims)
    k = min(top_k, len(order))
    chosen = order[:k]

    accum_line = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    accum_color = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    accum_color_support = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    accum_detail = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    accum_region = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    accum_weight = 0.0
    used = []

    for rank, idx in enumerate(chosen, start=1):
        p = valid_paths[idx]
        try:
            img = Image.open(p).convert("RGB")
            arr = np.asarray(resize_max_dim(img, 1024))
            in_view, _ = specimen_in_view(arr)
            if not in_view:
                continue
            score = dorsal_score(arr)
            if score < min_dorsal_score:
                continue
            s, c, d, m = normalize_pattern(arr, canvas_h, canvas_w)
            if s is None:
                continue
            quality = max(0.0, min(1.0, score))
            weight = max(0.02, float(sims[idx])) * (0.25 + 0.75 * quality * quality)
            sf = s.astype(np.float32)
            mf = m.astype(np.float32)
            support = np.maximum(sf, mf)
            accum_line += weight * sf
            accum_color += (weight * c) * support[..., None]
            accum_color_support += weight * support
            accum_detail += weight * d * mf
            accum_region += weight * mf
            accum_weight += weight
            used.append({"name": p.name, "sim": float(sims[idx]), "dorsal_score": float(score), "weight": weight})
            if rank % 10 == 0:
                print(f"Aggregated {len(used)} patterns")
        except Exception:
            continue

    if not used:
        raise RuntimeError("No usable dorsal patterns after filtering")

    prob = accum_line / max(1e-8, accum_weight)
    prob = np.clip(smooth2d(prob, 1), 0, 1)
    color = accum_color / (accum_color_support[..., None] + 1e-8)
    color = np.clip(color, 0, 1)
    detail = np.clip(accum_detail / (accum_region + 1e-8), 0, 1)
    detail = np.clip(smooth2d(detail, 1), 0, 1)
    return prob, color, detail, used


def build_ideal_dorsal_pattern(
    image_dir,
    output_path,
    hf_model,
    hf_batch_size,
    top_k,
    canvas_w,
    canvas_h,
    consensus_threshold,
):
    prob, color, _, used = aggregate_dorsal_patterns(
        image_dir=image_dir,
        hf_model=hf_model,
        hf_batch_size=hf_batch_size,
        top_k=top_k,
        canvas_w=canvas_w,
        canvas_h=canvas_h,
        min_dorsal_score=0.6,
    )

    core = prob >= consensus_threshold
    g1 = binary_dilate(core, 1) & (~core)
    g2 = binary_dilate(core, 2) & (~binary_dilate(core, 1))
    g3 = binary_dilate(core, 4) & (~binary_dilate(core, 2))
    g4 = binary_dilate(core, 7) & (~binary_dilate(core, 4))

    strength = np.clip((prob - consensus_threshold) / (1.0 - consensus_threshold + 1e-8), 0, 1)
    yy, xx = np.mgrid[0:canvas_h, 0:canvas_w].astype(np.float32)
    xx = xx / max(canvas_w - 1, 1)
    yy = yy / max(canvas_h - 1, 1)
    # Vivid field for stylized coloring.
    vivid = np.stack(
        [
            0.55 + 0.45 * np.sin(2 * np.pi * (0.85 * xx + 0.30 * yy) + 0.1),
            0.55 + 0.45 * np.sin(2 * np.pi * (0.65 * xx - 0.55 * yy) + 2.2),
            0.55 + 0.45 * np.sin(2 * np.pi * (0.95 * xx + 0.20 * yy) + 4.1),
        ],
        axis=-1,
    )
    # Blend observed butterfly colors with vivid palette.
    styled_color = np.clip(0.45 * color + 0.55 * vivid, 0, 1)

    # Competition-style background: deep gradient with subtle movement.
    bg = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    bg[..., 0] = 0.04 + 0.05 * (1 - yy) + 0.02 * np.sin(2 * np.pi * xx)
    bg[..., 1] = 0.03 + 0.04 * (1 - yy) + 0.01 * np.cos(2 * np.pi * (xx + yy))
    bg[..., 2] = 0.10 + 0.08 * (1 - yy) + 0.02 * np.sin(2 * np.pi * (0.7 * xx - 0.2 * yy))
    bg = np.clip(bg, 0, 1)

    out = bg.copy()

    # Soft wing fill from consensus area.
    fill = prob >= max(0.08, consensus_threshold * 0.55)
    fill_grad = np.stack(
        [
            0.95 - 0.30 * yy + 0.05 * np.cos(2 * np.pi * xx),
            0.43 + 0.33 * (1 - yy) + 0.04 * np.sin(2 * np.pi * (0.8 * xx + 0.2 * yy)),
            0.14 + 0.20 * xx + 0.05 * np.cos(2 * np.pi * (0.5 * xx - 0.4 * yy)),
        ],
        axis=-1,
    )
    fill_grad = np.clip(fill_grad, 0, 1)
    fill_alpha = np.clip(prob / (consensus_threshold + 1e-8), 0, 1) * 0.55
    out[fill] = np.clip((1 - fill_alpha[fill, None]) * out[fill] + fill_alpha[fill, None] * fill_grad[fill], 0, 1)

    # Layered neon glow around skeleton structure.
    out[g4] = np.maximum(out[g4], 0.14 * styled_color[g4] + np.array([0.02, 0.02, 0.05], dtype=np.float32))
    out[g3] = np.maximum(out[g3], 0.22 * styled_color[g3] + np.array([0.03, 0.03, 0.06], dtype=np.float32))
    out[g2] = np.maximum(out[g2], 0.35 * styled_color[g2] + np.array([0.03, 0.03, 0.06], dtype=np.float32))
    out[g1] = np.maximum(out[g1], 0.58 * styled_color[g1] + np.array([0.04, 0.04, 0.07], dtype=np.float32))
    out[core] = np.maximum(out[core], styled_color[core] * (0.70 + 0.30 * strength[core, None]))

    # Final symmetrization for an idealized dorsal template.
    out = 0.5 * (out + np.fliplr(out))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(out * 255, 0, 255).astype(np.uint8), mode="RGB").save(output_path)

    heat = np.clip(prob * 255, 0, 255).astype(np.uint8)
    heat_path = output_path.with_name(output_path.stem + "_consensus.png")
    Image.fromarray(heat, mode="L").save(heat_path)

    meta_path = output_path.with_name(output_path.stem + "_selected.csv")
    with meta_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_name", "embedding_similarity", "dorsal_score"])
        for row in sorted(used, key=lambda x: -x["sim"]):
            w.writerow([row["name"], f"{row['sim']:.6f}", f"{row['dorsal_score']:.6f}"])

    print(f"Ideal dorsal pattern: {output_path}")
    print(f"Consensus map: {heat_path}")
    print(f"Selected image list: {meta_path}")


def build_mathematical_outline(
    image_dir,
    output_path,
    hf_model,
    hf_batch_size,
    top_k,
    canvas_w,
    canvas_h,
    consensus_threshold,
):
    prob, color, detail, used = aggregate_dorsal_patterns(
        image_dir=image_dir,
        hf_model=hf_model,
        hf_batch_size=hf_batch_size,
        top_k=top_k,
        canvas_w=canvas_w,
        canvas_h=canvas_h,
        min_dorsal_score=0.6,
    )

    yy, xx = np.mgrid[0:canvas_h, 0:canvas_w].astype(np.float32)
    xn = xx / max(canvas_w - 1, 1)
    yn = yy / max(canvas_h - 1, 1)

    # Dark but colorful technical-paper background.
    out = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    out[..., 0] = 0.030 + 0.030 * (1 - yn) + 0.015 * np.sin(2 * np.pi * (0.7 * xn + 0.2 * yn))
    out[..., 1] = 0.018 + 0.028 * (1 - yn) + 0.014 * np.cos(2 * np.pi * (0.4 * xn + 0.6 * yn))
    out[..., 2] = 0.070 + 0.055 * (1 - yn) + 0.018 * np.sin(2 * np.pi * (0.9 * xn - 0.2 * yn))
    out = np.clip(out, 0, 1)

    rainbow = np.stack(
        [
            0.52 + 0.48 * np.sin(2 * np.pi * (0.95 * xn + 0.25 * yn) + 0.0),
            0.52 + 0.48 * np.sin(2 * np.pi * (0.65 * xn - 0.55 * yn) + 2.1),
            0.52 + 0.48 * np.sin(2 * np.pi * (1.15 * xn + 0.10 * yn) + 4.2),
        ],
        axis=-1,
    )
    rainbow = np.clip(rainbow, 0, 1)

    # Multi-level mathematical outlines.
    levels = [
        max(0.06, consensus_threshold * 0.30),
        max(0.10, consensus_threshold * 0.50),
        max(0.14, consensus_threshold * 0.70),
        consensus_threshold,
        min(0.92, consensus_threshold * 1.35),
    ]
    level_colors = [
        np.array([0.30, 0.72, 1.00], dtype=np.float32),
        np.array([0.62, 0.48, 1.00], dtype=np.float32),
        np.array([1.00, 0.56, 0.94], dtype=np.float32),
        np.array([1.00, 0.74, 0.48], dtype=np.float32),
        np.array([1.00, 0.95, 0.68], dtype=np.float32),
    ]
    prev = np.zeros_like(prob, dtype=bool)
    for lev, lc in zip(levels, level_colors):
        m = prob >= lev
        ring = binary_dilate(m, 1) & (~m)
        ring = ring & (~prev)
        prev = prev | ring
        out[ring] = np.maximum(out[ring], lc)

    # Keep interiors strongly colorful with inherited specimen tint + iridescent field.
    wing_fill = prob >= max(0.07, consensus_threshold * 0.5)
    wing_tint = np.clip(0.45 * color + 0.55 * rainbow, 0, 1)
    wing_strength = np.clip(prob / (consensus_threshold + 1e-8), 0, 1)
    out[wing_fill] = np.clip(
        (0.28 + 0.22 * (1 - wing_strength[wing_fill, None])) * out[wing_fill]
        + (0.72 - 0.22 * (1 - wing_strength[wing_fill, None])) * wing_tint[wing_fill],
        0,
        1,
    )

    # Feature-driven internal detail from raw specimen texture maps.
    d1 = detail > 0.36
    d2 = detail > 0.50
    d3 = detail > 0.64
    d1_ring = binary_dilate(d1, 1) & (~d1)
    d2_ring = binary_dilate(d2, 1) & (~d2)
    c_cyan = np.array([0.30, 0.90, 1.00], dtype=np.float32)
    c_magenta = np.array([1.00, 0.45, 0.95], dtype=np.float32)
    c_gold = np.array([1.00, 0.86, 0.35], dtype=np.float32)
    out[d1_ring & wing_fill] = np.maximum(out[d1_ring & wing_fill], np.clip(0.55 * wing_tint[d1_ring & wing_fill] + 0.45 * c_cyan, 0, 1))
    out[d2_ring & wing_fill] = np.maximum(out[d2_ring & wing_fill], np.clip(0.50 * wing_tint[d2_ring & wing_fill] + 0.50 * c_magenta, 0, 1))
    out[d3 & wing_fill] = np.maximum(out[d3 & wing_fill], np.clip(0.45 * wing_tint[d3 & wing_fill] + 0.55 * c_gold, 0, 1))

    # Small feature-linked stipple points (not random-only).
    dots = (detail > 0.48) & wing_fill & (prob < max(0.20, consensus_threshold))
    dot_selector = np.sin(173 * xn + 117 * yn) > 0.96
    dots = dots & dot_selector
    out[dots] = np.maximum(out[dots], np.clip(0.75 * rainbow[dots] + np.array([0.20, 0.18, 0.14], dtype=np.float32), 0, 1))

    # Brighten the strongest structural lines.
    core = prob >= consensus_threshold
    core_glow = binary_dilate(core, 2) & (~core)
    core_glow2 = binary_dilate(core, 5) & (~binary_dilate(core, 2))
    out[core_glow2] = np.maximum(out[core_glow2], np.clip(0.30 * rainbow[core_glow2] + np.array([0.10, 0.10, 0.14], dtype=np.float32), 0, 1))
    out[core_glow] = np.maximum(out[core_glow], np.clip(0.72 * rainbow[core_glow] + np.array([0.26, 0.22, 0.18], dtype=np.float32), 0, 1))
    out[core] = np.maximum(out[core], np.clip(0.85 * rainbow[core] + np.array([0.38, 0.34, 0.30], dtype=np.float32), 0, 1))

    # White-hot highlights on densest shared lines for obvious contrast pop.
    hot = prob >= min(0.96, consensus_threshold * 1.45)
    out[hot] = np.maximum(out[hot], np.array([1.0, 0.98, 0.95], dtype=np.float32))

    # Mirror for strict dorsal symmetry.
    out = 0.5 * (out + np.fliplr(out))

    # Add technical annotation lines/text.
    img = Image.fromarray(np.clip(out * 255, 0, 255).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    core_mask = prob >= consensus_threshold
    if core_mask.any():
        ys, xs = np.where(core_mask)
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        width = x1 - x0 + 1
        height = y1 - y0 + 1
        ratio = width / max(height, 1)
    else:
        ratio = 0.0

    ann_y = int(canvas_h * 0.80)
    line_color = (150, 150, 140)
    text_color = (195, 190, 172)
    draw.line([(int(canvas_w * 0.12), ann_y), (int(canvas_w * 0.36), ann_y)], fill=line_color, width=1)
    draw.line([(int(canvas_w * 0.64), ann_y), (int(canvas_w * 0.88), ann_y)], fill=line_color, width=1)
    draw.text((int(canvas_w * 0.30), ann_y + 12), "MATHEMATICAL OUTLINE", fill=(225, 217, 190), font=font)
    draw.text(
        (int(canvas_w * 0.26), ann_y + 30),
        "f(x,y)=sin(phi*x)+phi^-1*sin(phi*y)",
        fill=text_color,
        font=font,
    )
    draw.text(
        (int(canvas_w * 0.26), ann_y + 45),
        "g(x,y)=|grad P(x,y)|,  phi=1.61803",
        fill=text_color,
        font=font,
    )
    draw.text(
        (int(canvas_w * 0.26), ann_y + 60),
        f"N={len(used)}   mean_d={np.mean([u['dorsal_score'] for u in used]):.3f}   w/h={ratio:.3f}",
        fill=text_color,
        font=font,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)

    heat = np.clip(prob * 255, 0, 255).astype(np.uint8)
    heat_path = output_path.with_name(output_path.stem + "_consensus.png")
    Image.fromarray(heat, mode="L").save(heat_path)

    meta_path = output_path.with_name(output_path.stem + "_selected.csv")
    with meta_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_name", "embedding_similarity", "dorsal_score"])
        for row in sorted(used, key=lambda x: -x["sim"]):
            w.writerow([row["name"], f"{row['sim']:.6f}", f"{row['dorsal_score']:.6f}"])

    print(f"Mathematical outline: {output_path}")
    print(f"Consensus map: {heat_path}")
    print(f"Selected image list: {meta_path}")


def ellipse_level_set(xn, yn, cx, cy, ax, ay, angle):
    ca = math.cos(angle)
    sa = math.sin(angle)
    dx = xn - cx
    dy = yn - cy
    xr = ca * dx + sa * dy
    yr = -sa * dx + ca * dy
    return (xr / max(ax, 1e-6)) ** 2 + (yr / max(ay, 1e-6)) ** 2


def build_parametric_shape_fields(xn, yn, wing_mask, coeff):
    fill = np.zeros_like(xn, dtype=np.float32)
    solid = np.zeros_like(xn, dtype=bool)
    dashed = np.zeros_like(xn, dtype=bool)

    h1 = float(coeff["h1"])
    h2 = float(coeff["h2"])
    k1 = float(coeff["k1"])
    k2 = float(coeff["k2"])
    r1 = float(coeff["r1"])
    r2 = float(coeff["r2"])
    q1 = float(coeff["q1"])
    q2 = float(coeff["q2"])

    side_params = [(-1.0, -abs(h1), k1, r1, q1), (1.0, abs(h2), k2, r2, q2)]
    for side, cx, ky, rr, q in side_params:
        upper = ellipse_level_set(
            xn,
            yn,
            cx,
            -0.24 + 0.20 * ky,
            0.24 + 0.55 * rr,
            0.16 + 0.42 * rr,
            0.12 * side,
        )
        middle = ellipse_level_set(
            xn,
            yn,
            cx * 0.94,
            0.02 + 0.18 * ky,
            0.21 + 0.48 * rr,
            0.14 + 0.35 * rr,
            0.07 * side,
        )
        lower = ellipse_level_set(
            xn,
            yn,
            cx * 0.88,
            0.36 + 0.14 * ky,
            0.20 + 0.44 * rr,
            0.18 + 0.38 * rr,
            -0.04 * side,
        )

        fill += 0.40 * np.clip(1.0 - upper, 0.0, 1.0)
        fill += 0.36 * np.clip(1.0 - middle, 0.0, 1.0)
        fill += 0.32 * np.clip(1.0 - lower, 0.0, 1.0)

        solid |= np.abs(upper - 1.0) < 0.020
        solid |= np.abs(lower - 1.0) < 0.020

        angle = np.arctan2(yn - (0.02 + 0.18 * ky), xn - (cx * 0.94))
        radius = np.sqrt((xn - (cx * 0.94)) ** 2 + (yn - (0.02 + 0.18 * ky)) ** 2)
        dashed_arc = np.sin(24.0 * angle + 7.0 * radius) > 0.08
        dashed |= (np.abs(middle - 1.0) < 0.017) & dashed_arc

        for t in (0.24, 0.36, 0.48, 0.62, 0.76):
            slope = side * (0.25 + 1.05 * t)
            intercept = -0.03 + 0.12 * (t - 0.5) + 0.30 * q
            width = 0.004 + 0.0018 * t
            vein = np.abs(yn - (slope * xn + intercept)) < width
            solid |= vein & ((xn * side) > 0.02)

    thorax = np.abs(xn) < 0.045
    fill += 0.35 * thorax.astype(np.float32)
    solid |= thorax & (np.abs(yn) < 0.72)

    wing = wing_mask.astype(bool)
    fill = np.clip(fill, 0.0, 1.0) * wing.astype(np.float32)
    solid = solid & wing
    dashed = dashed & wing & (~solid)
    return fill, solid, dashed


def render_math_reconstruction(arr):
    phi = (1 + np.sqrt(5)) / 2
    rgb = arr.astype(np.float32) / 255.0
    rgb_s = smooth3d(rgb, 1)
    h, w = rgb.shape[:2]

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xn = 2.0 * (xx / max(w - 1, 1)) - 1.0
    yn = 2.0 * (yy / max(h - 1, 1)) - 1.0

    gray = rgb_to_gray(rgb_s)
    edge = sobel_like(smooth2d(gray, 1))
    edge = edge / (edge.max() + 1e-8)
    cmax = rgb_s.max(axis=2)
    cmin = rgb_s.min(axis=2)
    sat = (cmax - cmin) / (cmax + 1e-8)

    mask = build_feature_mask(arr)
    detail = build_detail_map(arr, feature_mask=mask)

    mr = float(rgb_s[..., 0].mean())
    mg = float(rgb_s[..., 1].mean())
    mb = float(rgb_s[..., 2].mean())
    ms = float(sat.mean())
    a = 4.0 + 5.0 * mr
    b = 4.0 + 5.0 * mg
    c = 3.0 + 4.0 * mb
    d = 1.8 + 2.5 * ms

    # Explicit line equations.
    m1 = -0.9 + 0.8 * mr
    q1 = -0.05 + 0.18 * (mg - 0.5)
    m2 = 0.9 - 0.8 * mg
    q2 = -0.05 + 0.18 * (mr - 0.5)
    kx = 0.02 * (mb - 0.5)

    # Explicit circle equations.
    h1 = -0.36 + 0.05 * (mr - mg)
    h2 = 0.36 - 0.05 * (mr - mg)
    k1 = 0.06 * (mg - mr)
    k2 = -k1
    r1 = 0.38 + 0.08 * mr
    r2 = 0.38 + 0.08 * mg

    r = np.sqrt(xn * xn + yn * yn + 1e-8)
    theta = np.arctan2(yn, xn)
    l1 = np.abs(yn - (m1 * xn + q1))
    l2 = np.abs(yn - (m2 * xn + q2))
    l3 = np.abs(xn - kx)
    line_dist = np.minimum(np.minimum(l1, l2), l3)

    c1 = np.abs((xn - h1) ** 2 + (yn - k1) ** 2 - r1 * r1)
    c2 = np.abs((xn - h2) ** 2 + (yn - k2) ** 2 - r2 * r2)
    circle_dist = np.minimum(c1, c2)

    field = (
        np.sin(a * xn + b * yn + 2.0 * np.pi * detail)
        + np.cos((a + b) * r - c * theta)
        + 0.55 * np.sin((c + phi) * 3.0 * xn * yn + d * detail)
        + 0.95 * np.exp(-18.0 * line_dist)
        + 1.10 * np.exp(-30.0 * circle_dist)
    )
    g = sobel_like(smooth2d(field, 1))
    active = binary_dilate(mask, 2)
    if active.sum() < 50:
        active = np.ones_like(mask, dtype=bool)
    coeff = {
        "m1": m1,
        "q1": q1,
        "m2": m2,
        "q2": q2,
        "kx": kx,
        "h1": h1,
        "h2": h2,
        "k1": k1,
        "k2": k2,
        "r1": r1,
        "r2": r2,
        "a": a,
        "b": b,
        "c": c,
        "d": d,
    }
    shape_fill, shape_solid, shape_dashed = build_parametric_shape_fields(xn, yn, active, coeff)
    thr = np.percentile(g[active], 87)
    contour = g >= thr
    spine = zhang_suen_thinning(mask)
    line_guides = (line_dist < 0.016) & active
    circle_guides = (circle_dist < 0.013) & active
    math_lines = contour | spine | line_guides | circle_guides
    math_lines = binary_dilate(math_lines, 1)
    math_lines = math_lines & (binary_dilate(active, 1) | contour)

    rainbow = np.stack(
        [
            0.5 + 0.5 * np.sin(2 * np.pi * (0.9 * xn + 0.2 * yn) + 0.0),
            0.5 + 0.5 * np.sin(2 * np.pi * (0.7 * xn - 0.4 * yn) + 2.1),
            0.5 + 0.5 * np.sin(2 * np.pi * (1.1 * xn + 0.1 * yn) + 4.2),
        ],
        axis=-1,
    )
    tint = np.clip(0.55 * boost_color(rgb) + 0.45 * rainbow, 0, 1)

    out = np.zeros((h, w, 3), dtype=np.float32)
    out[..., 0] = 0.02 + 0.02 * (1 - (yn + 1) / 2)
    out[..., 1] = 0.02 + 0.015 * (1 - (yn + 1) / 2)
    out[..., 2] = 0.07 + 0.03 * (1 - (yn + 1) / 2)

    fill = binary_dilate(active, 1)
    fill_alpha = 0.18 + 0.20 * detail
    out[fill] = np.clip(
        (1 - fill_alpha[fill, None]) * out[fill] + fill_alpha[fill, None] * (0.55 * tint[fill] + 0.45 * rainbow[fill]),
        0,
        1,
    )

    shape_zone = shape_fill > 0.02
    shape_alpha = 0.18 + 0.52 * shape_fill
    shape_color = np.clip(0.62 * tint + np.array([0.22, 0.15, 0.07], dtype=np.float32), 0, 1)
    out[shape_zone] = np.clip(
        (1 - shape_alpha[shape_zone, None]) * out[shape_zone] + shape_alpha[shape_zone, None] * shape_color[shape_zone],
        0,
        1,
    )

    glow = binary_dilate(math_lines, 2) & (~math_lines)
    out[glow] = np.maximum(out[glow], np.clip(0.45 * tint[glow] + np.array([0.08, 0.08, 0.11], dtype=np.float32), 0, 1))
    out[math_lines] = np.maximum(out[math_lines], np.clip(0.88 * tint[math_lines] + np.array([0.12, 0.10, 0.08], dtype=np.float32), 0, 1))
    out[shape_solid] = np.minimum(out[shape_solid], np.array([0.09, 0.09, 0.11], dtype=np.float32))
    out[shape_dashed] = np.minimum(out[shape_dashed], np.array([0.14, 0.14, 0.16], dtype=np.float32))

    out = 0.5 * (out + np.fliplr(out))
    eq = (
        "L1:y=m1*x+q1; L2:y=m2*x+q2; L3:x=kx; "
        "C1:(x-h1)^2+(y-k1)^2=r1^2; C2:(x-h2)^2+(y-k2)^2=r2^2; "
        "E_i=((x-cx_i)^2/a_i^2)+((y-cy_i)^2/b_i^2); "
        "Q=sin(a*x+b*y+2*pi*D)+cos((a+b)*r-c*theta)+0.55*sin((c+phi)*3*x*y+d*D)"
        "+0.95*exp(-18*min(|L1|,|L2|,|L3|))+1.10*exp(-30*min(|C1|,|C2|))+sum(exp(-40*|E_i-1|))"
    )
    params = {
        "a": a,
        "b": b,
        "c": c,
        "d": d,
        "m1": m1,
        "q1": q1,
        "m2": m2,
        "q2": q2,
        "kx": kx,
        "h1": h1,
        "k1": k1,
        "r1": r1,
        "h2": h2,
        "k2": k2,
        "r2": r2,
        "detail_mean": float(detail.mean()),
    }
    return (np.clip(out, 0, 1) * 255).astype(np.uint8), eq, params


def robust_weighted_average(values, weights):
    if not values:
        return None
    v = np.asarray(values, dtype=np.float32)
    w = np.asarray(weights, dtype=np.float32)
    w = np.maximum(w, 1e-8)
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med))) + 1e-6
    scale = 1.4826 * mad
    z = np.abs(v - med) / scale
    rw = w * np.square(np.clip(1.0 - z / 3.0, 0.0, 1.0))
    if rw.sum() <= 1e-8:
        rw = w
    return float((rw * v).sum() / (rw.sum() + 1e-8))


def clamp_coefficients(coeff, defaults):
    bounds = {
        "m1": (-2.5, 2.5),
        "q1": (-0.6, 0.6),
        "m2": (-2.5, 2.5),
        "q2": (-0.6, 0.6),
        "kx": (-0.4, 0.4),
        "h1": (-0.9, 0.9),
        "h2": (-0.9, 0.9),
        "k1": (-0.6, 0.6),
        "k2": (-0.6, 0.6),
        "r1": (0.12, 1.2),
        "r2": (0.12, 1.2),
        "a": (2.0, 12.0),
        "b": (2.0, 12.0),
        "c": (2.0, 10.0),
        "d": (0.8, 6.0),
    }
    out = {}
    for key, default in defaults.items():
        val = coeff.get(key, default)
        if not np.isfinite(val):
            val = default
        lo, hi = bounds.get(key, (-np.inf, np.inf))
        out[key] = float(np.clip(val, lo, hi))
    return out


def build_math_reconstruction_dataset(input_dir, output_dir, min_dorsal_score):
    src_paths = iter_image_paths(input_dir)
    if not src_paths:
        raise RuntimeError(f"No images found in {input_dir}")

    outdir = Path(output_dir)
    recon_dir = outdir / "reconstructions"
    outdir.mkdir(parents=True, exist_ok=True)
    recon_dir.mkdir(parents=True, exist_ok=True)

    report = outdir / "equations.csv"
    fields = [
        "source_image",
        "status",
        "reconstruction_path",
        "dorsal_score",
        "view_ok",
        "equation",
        "a",
        "b",
        "c",
        "d",
        "m1",
        "q1",
        "m2",
        "q2",
        "kx",
        "h1",
        "k1",
        "r1",
        "h2",
        "k2",
        "r2",
        "detail_mean",
    ]
    accepted = 0
    with report.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, p in enumerate(src_paths, start=1):
            row = {k: "" for k in fields}
            row["source_image"] = p.name
            try:
                arr = np.asarray(resize_max_dim(Image.open(p).convert("RGB"), 1024))
                view_ok, _ = specimen_in_view(arr)
                score = dorsal_score(arr)
                row["dorsal_score"] = f"{score:.6f}"
                row["view_ok"] = "1" if view_ok else "0"
                if (not view_ok) or (score < min_dorsal_score):
                    row["status"] = "skipped"
                    w.writerow(row)
                    continue

                recon, eq, params = render_math_reconstruction(arr)
                out_path = recon_dir / f"{p.stem}_math.png"
                Image.fromarray(recon, mode="RGB").save(out_path)
                row["status"] = "saved"
                row["reconstruction_path"] = str(out_path)
                row["equation"] = eq
                row["a"] = f"{params['a']:.6f}"
                row["b"] = f"{params['b']:.6f}"
                row["c"] = f"{params['c']:.6f}"
                row["d"] = f"{params['d']:.6f}"
                row["m1"] = f"{params['m1']:.6f}"
                row["q1"] = f"{params['q1']:.6f}"
                row["m2"] = f"{params['m2']:.6f}"
                row["q2"] = f"{params['q2']:.6f}"
                row["kx"] = f"{params['kx']:.6f}"
                row["h1"] = f"{params['h1']:.6f}"
                row["k1"] = f"{params['k1']:.6f}"
                row["r1"] = f"{params['r1']:.6f}"
                row["h2"] = f"{params['h2']:.6f}"
                row["k2"] = f"{params['k2']:.6f}"
                row["r2"] = f"{params['r2']:.6f}"
                row["detail_mean"] = f"{params['detail_mean']:.6f}"
                w.writerow(row)
                accepted += 1
                if i % 20 == 0:
                    print(f"Math dataset processed {i}/{len(src_paths)}, saved {accepted}")
            except Exception:
                row["status"] = "error"
                w.writerow(row)

    print(f"Mathematical dataset saved: {recon_dir}")
    print(f"Equation report: {report}")
    print(f"Saved reconstructions: {accepted}")
    return recon_dir


def build_ideal_math_reconstruction(
    math_dataset_dir,
    output_path,
    hf_model,
    hf_batch_size,
    top_k,
    canvas_w,
    canvas_h,
    consensus_threshold,
    raw_image_dir=None,
):
    recon_dir = Path(math_dataset_dir) / "reconstructions"
    if not recon_dir.exists():
        raise RuntimeError(f"Missing reconstruction dataset dir: {recon_dir}")

    prob, color, detail, used = aggregate_dorsal_patterns(
        image_dir=str(recon_dir),
        hf_model=hf_model,
        hf_batch_size=hf_batch_size,
        top_k=top_k,
        canvas_w=canvas_w,
        canvas_h=canvas_h,
        min_dorsal_score=0.0,
    )
    raw_prob = None
    raw_color = None
    raw_detail = None
    if raw_image_dir:
        try:
            raw_prob, raw_color, raw_detail, _ = aggregate_dorsal_patterns(
                image_dir=str(raw_image_dir),
                hf_model=hf_model,
                hf_batch_size=hf_batch_size,
                top_k=max(top_k, 80),
                canvas_w=canvas_w,
                canvas_h=canvas_h,
                min_dorsal_score=0.64,
            )
            # Bias color/detail toward real specimen palette and raw pattern structure.
            prob = np.clip(0.58 * prob + 0.42 * raw_prob, 0, 1)
            color = np.clip(0.30 * color + 0.70 * raw_color, 0, 1)
            detail = np.clip(0.34 * detail + 0.66 * raw_detail, 0, 1)
        except Exception as e:
            print(f"Raw palette fusion skipped ({e})")
    report_path = Path(math_dataset_dir) / "equations.csv"
    eq_rows = []
    if report_path.exists():
        with report_path.open("r", newline="") as f:
            for row in csv.DictReader(f):
                if row.get("status") != "saved":
                    continue
                eq_rows.append(row)
    eq_by_name = {}
    for row in eq_rows:
        recon_name = Path(row.get("reconstruction_path", "")).name
        if recon_name:
            eq_by_name[recon_name] = row

    yy, xx = np.mgrid[0:canvas_h, 0:canvas_w].astype(np.float32)
    xn = 2.0 * (xx / max(canvas_w - 1, 1)) - 1.0
    yn = 2.0 * (yy / max(canvas_h - 1, 1)) - 1.0
    r = np.sqrt(xn * xn + yn * yn + 1e-8)
    theta = np.arctan2(yn, xn)
    wing = prob >= max(0.07, consensus_threshold * 0.50)
    wing = wing & (smooth2d(wing.astype(np.float32), 1) > 0.20)
    if wing.any():
        mr = float(color[..., 0][wing].mean())
        mg = float(color[..., 1][wing].mean())
        mb = float(color[..., 2][wing].mean())
    else:
        mr = float(color[..., 0].mean())
        mg = float(color[..., 1].mean())
        mb = float(color[..., 2].mean())

    # Use the learned per-image equation coefficients whenever available.
    coeff_keys = ["m1", "q1", "m2", "q2", "kx", "h1", "h2", "k1", "k2", "r1", "r2", "a", "b", "c", "d"]
    default_coeffs = {
        "m1": -0.8 + 0.7 * mr,
        "q1": -0.04 + 0.14 * (mg - 0.5),
        "m2": 0.8 - 0.7 * mg,
        "q2": -0.04 + 0.14 * (mr - 0.5),
        "kx": 0.02 * (mb - 0.5),
        "h1": -0.36 + 0.04 * (mr - mg),
        "h2": 0.36 - 0.04 * (mr - mg),
        "k1": 0.05 * (mg - mr),
        "k2": -0.05 * (mg - mr),
        "r1": 0.40 + 0.07 * mr,
        "r2": 0.40 + 0.07 * mg,
        "a": 6.5 + 1.0 * mr,
        "b": 5.5 + 1.0 * mg,
        "c": 4.5 + 0.8 * mb,
        "d": 2.8 + 1.0 * float(detail.mean()),
    }
    coeff_values = {k: [] for k in coeff_keys}
    coeff_weights = {k: [] for k in coeff_keys}
    for item in used:
        row = eq_by_name.get(item["name"])
        if row is None:
            continue
        # Similarity + dorsal quality weighting reduces outlier/noisy equations.
        sim = max(0.0, float(item.get("sim", 0.0)))
        dorsal = max(0.0, min(1.0, float(item.get("dorsal_score", 0.0))))
        wgt = max(0.001, sim * (0.35 + 0.65 * dorsal * dorsal))
        for k in coeff_keys:
            try:
                v = float(row.get(k, ""))
            except (TypeError, ValueError):
                continue
            coeff_values[k].append(v)
            coeff_weights[k].append(wgt)
    coeff = {}
    for k in coeff_keys:
        rv = robust_weighted_average(coeff_values[k], coeff_weights[k])
        if rv is None:
            coeff[k] = default_coeffs[k]
        else:
            coeff[k] = rv
    coeff = clamp_coefficients(coeff, default_coeffs)

    m1 = coeff["m1"]
    q1 = coeff["q1"]
    m2 = coeff["m2"]
    q2 = coeff["q2"]
    kx = coeff["kx"]
    h1 = coeff["h1"]
    h2 = coeff["h2"]
    k1 = coeff["k1"]
    k2 = coeff["k2"]
    r1 = max(0.12, abs(coeff["r1"]))
    r2 = max(0.12, abs(coeff["r2"]))
    a = coeff["a"]
    b = coeff["b"]
    c = coeff["c"]
    d = coeff["d"]

    line_dist = np.minimum(
        np.minimum(np.abs(yn - (m1 * xn + q1)), np.abs(yn - (m2 * xn + q2))),
        np.abs(xn - kx),
    )
    circle_dist = np.minimum(
        np.abs((xn - h1) ** 2 + (yn - k1) ** 2 - r1 * r1),
        np.abs((xn - h2) ** 2 + (yn - k2) ** 2 - r2 * r2),
    )

    eq_field = (
        np.sin((a + 1.2 * detail) * (2.2 * xn) + (b + 0.8 * prob) * (1.8 * yn))
        + np.cos((a + b + 1.5 * prob) * r - (c + 1.2 * detail) * theta)
        + 0.45 * np.sin((c + 1.8) * xn * yn + d * detail)
        + 0.80 * np.exp(-15.0 * line_dist)
        + 0.95 * np.exp(-24.0 * circle_dist)
    )
    eg = sobel_like(smooth2d(eq_field, 1))
    if wing.any():
        line_thr = np.percentile(eg[wing], 86)
    else:
        line_thr = np.percentile(eg, 86)
    math_lines = ((eg >= line_thr) | (line_dist < 0.017) | (circle_dist < 0.013)) & wing
    if raw_prob is not None and raw_detail is not None:
        rp_edge = sobel_like(smooth2d(raw_prob, 1))
        rp_edge = rp_edge / (rp_edge.max() + 1e-8)
        rd_hi = raw_detail > np.percentile(raw_detail[wing] if wing.any() else raw_detail, 74)
        raw_lines = ((rp_edge > np.percentile(rp_edge[wing] if wing.any() else rp_edge, 80)) | rd_hi) & wing
        raw_lines = binary_dilate(raw_lines, 1)
        math_lines = math_lines | raw_lines
    shape_fill, shape_solid, shape_dashed = build_parametric_shape_fields(xn, yn, wing, coeff)

    rainbow = np.stack(
        [
            0.5 + 0.5 * np.sin(2 * np.pi * (0.95 * (xn + 1) / 2 + 0.20 * (yn + 1) / 2)),
            0.5 + 0.5 * np.sin(2 * np.pi * (0.70 * (xn + 1) / 2 - 0.50 * (yn + 1) / 2) + 2.2),
            0.5 + 0.5 * np.sin(2 * np.pi * (1.10 * (xn + 1) / 2 + 0.10 * (yn + 1) / 2) + 4.1),
        ],
        axis=-1,
    )
    base_col = np.clip(0.5 * color + 0.5 * rainbow, 0, 1)

    out = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    out[..., 0] = 0.03 + 0.02 * (1 - (yn + 1) / 2)
    out[..., 1] = 0.025 + 0.02 * (1 - (yn + 1) / 2)
    out[..., 2] = 0.08 + 0.04 * (1 - (yn + 1) / 2)

    out[wing] = np.clip(0.30 * out[wing] + 0.70 * base_col[wing], 0, 1)
    shape_zone = shape_fill > 0.02
    shape_alpha = 0.16 + 0.52 * shape_fill
    shape_col = np.clip(0.62 * base_col + np.array([0.20, 0.13, 0.07], dtype=np.float32), 0, 1)
    out[shape_zone] = np.clip(
        (1 - shape_alpha[shape_zone, None]) * out[shape_zone] + shape_alpha[shape_zone, None] * shape_col[shape_zone],
        0,
        1,
    )
    t1 = detail > 0.38
    t2 = detail > 0.58
    out[t1 & wing] = np.maximum(out[t1 & wing], np.clip(0.60 * base_col[t1 & wing] + np.array([0.16, 0.12, 0.12], dtype=np.float32), 0, 1))
    out[t2 & wing] = np.maximum(out[t2 & wing], np.clip(0.70 * base_col[t2 & wing] + np.array([0.22, 0.16, 0.12], dtype=np.float32), 0, 1))

    glow = binary_dilate(math_lines, 2) & (~math_lines)
    core = prob >= consensus_threshold
    out[glow] = np.maximum(out[glow], np.clip(0.55 * base_col[glow] + np.array([0.12, 0.10, 0.12], dtype=np.float32), 0, 1))
    out[math_lines] = np.maximum(out[math_lines], np.clip(0.78 * base_col[math_lines] + np.array([0.20, 0.16, 0.12], dtype=np.float32), 0, 1))
    out[shape_solid] = np.minimum(out[shape_solid], np.array([0.08, 0.08, 0.10], dtype=np.float32))
    out[shape_dashed] = np.minimum(out[shape_dashed], np.array([0.13, 0.13, 0.15], dtype=np.float32))
    out[core] = np.maximum(out[core], np.clip(0.88 * base_col[core] + np.array([0.30, 0.26, 0.22], dtype=np.float32), 0, 1))

    out = 0.5 * (out + np.fliplr(out))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(np.clip(out * 255, 0, 255).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    ann_y = int(canvas_h * 0.80)
    draw.line([(int(canvas_w * 0.11), ann_y), (int(canvas_w * 0.35), ann_y)], fill=(160, 160, 150), width=1)
    draw.line([(int(canvas_w * 0.65), ann_y), (int(canvas_w * 0.89), ann_y)], fill=(160, 160, 150), width=1)
    draw.text((int(canvas_w * 0.28), ann_y + 10), "IDEAL MATHEMATICAL RECONSTRUCTION", fill=(235, 228, 205), font=font)
    draw.text(
        (int(canvas_w * 0.25), ann_y + 26),
        "Lines: L1(y=m1x+q1), L2(y=m2x+q2), L3(x=kx)",
        fill=(210, 205, 188),
        font=font,
    )
    draw.text(
        (int(canvas_w * 0.20), ann_y + 40),
        "Circles + Ellipse bands: C1/C2 and Ei((x-cx)^2/a^2+(y-cy)^2/b^2=1)",
        fill=(210, 205, 188),
        font=font,
    )
    draw.text(
        (int(canvas_w * 0.28), ann_y + 54),
        f"N={len(used)}  m1={m1:.3f} m2={m2:.3f} r1={r1:.3f} r2={r2:.3f}",
        fill=(210, 205, 188),
        font=font,
    )
    img.save(output_path)

    heat = np.clip(prob * 255, 0, 255).astype(np.uint8)
    heat_path = output_path.with_name(output_path.stem + "_consensus.png")
    Image.fromarray(heat, mode="L").save(heat_path)

    meta_path = output_path.with_name(output_path.stem + "_selected.csv")
    with meta_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_name", "embedding_similarity", "dorsal_score"])
        for row in sorted(used, key=lambda x: -x["sim"]):
            w.writerow([row["name"], f"{row['sim']:.6f}", f"{row['dorsal_score']:.6f}"])

    print(f"Ideal mathematical reconstruction: {output_path}")
    print(f"Consensus map: {heat_path}")
    print(f"Selected image list: {meta_path}")


def gather_candidates(institutions, pages_per_query, page_limit):
    seen_media = set()
    candidates = []
    for inst in institutions:
        for family_key in BUTTERFLY_FAMILY_KEYS:
            for offset in range(0, pages_per_query * page_limit, page_limit):
                params = {
                    "institutionCode": inst,
                    "familyKey": family_key,
                    "basisOfRecord": "PRESERVED_SPECIMEN",
                    "mediaType": "StillImage",
                    "limit": page_limit,
                    "offset": offset,
                }
                url = f"{GBIF_OCC_API}?{urlencode(params)}"
                data = fetch_json(url)
                batch = data.get("results", [])
                if not batch:
                    break
                for occ in batch:
                    media_list = occ.get("media") or []
                    for media in media_list:
                        murl = media.get("identifier")
                        if not murl or murl in seen_media:
                            continue
                        fmt = (media.get("format") or "").lower()
                        if "image" not in fmt and not murl.lower().endswith((".jpg", ".jpeg", ".png")):
                            continue
                        seen_media.add(murl)
                        candidates.append(
                            {
                                "occurrence_key": occ.get("key"),
                                "institutionCode": occ.get("institutionCode"),
                                "family": occ.get("family"),
                                "species": occ.get("species"),
                                "license": media.get("license"),
                                "url": murl,
                            }
                        )
    return candidates


def prune_non_dorsal_images(image_dir, dorsal_threshold):
    src_dir = Path(image_dir)
    paths = iter_image_paths(src_dir)
    if not paths:
        raise RuntimeError(f"No images found in {src_dir}")

    rejected_dir = src_dir.parent / f"{src_dir.name}_rejected_non_dorsal"
    rejected_dir.mkdir(parents=True, exist_ok=True)

    kept = 0
    moved = 0
    errors = 0
    for i, p in enumerate(paths, start=1):
        try:
            arr = np.asarray(resize_max_dim(Image.open(p).convert("RGB"), 1024))
            view_ok, _ = specimen_in_view(arr)
            score = dorsal_score(arr)
            if view_ok and score >= dorsal_threshold:
                kept += 1
                continue

            dst = rejected_dir / p.name
            if dst.exists():
                dst = rejected_dir / f"{p.stem}_{i}{p.suffix}"
            shutil.move(str(p), str(dst))
            moved += 1
        except Exception:
            errors += 1
        if i % 25 == 0:
            print(f"Prune progress {i}/{len(paths)} kept={kept} moved={moved} errors={errors}")

    print(f"Dorsal prune complete: kept={kept}, moved={moved}, errors={errors}")
    print(f"Rejected images moved to: {rejected_dir}")


def process_candidate(item, raw_dir, skel_dir, min_dim, dorsal_threshold):
    key = item["occurrence_key"]
    inst = (item.get("institutionCode") or "unknown").lower()
    stem = f"{inst}_{key}"
    raw_path = raw_dir / f"{stem}.jpg"
    skel_path = skel_dir / f"{stem}_skeleton.png"

    try:
        img = download_image(item["url"])
    except Exception as exc:
        return {"status": f"download_error:{type(exc).__name__}", "item": item}

    w, h = img.size
    if max(w, h) < min_dim:
        return {"status": "reject_low_res", "item": item, "width": w, "height": h}

    small = resize_max_dim(img, 1024)
    arr = np.asarray(small)
    in_view, vm = specimen_in_view(arr)
    if not in_view:
        return {
            "status": "reject_out_of_view",
            "item": item,
            "width": w,
            "height": h,
            "view_area": vm["area"],
            "view_bbox_w": vm["bbox_w"],
            "view_bbox_h": vm["bbox_h"],
            "view_center_offset": vm["center_offset"],
            "view_border_ratio": vm["border_ratio"],
        }

    score = dorsal_score(arr)
    if score < dorsal_threshold:
        return {
            "status": "reject_not_dorsal",
            "item": item,
            "width": w,
            "height": h,
            "dorsal_score": score,
            "view_area": vm["area"],
            "view_bbox_w": vm["bbox_w"],
            "view_bbox_h": vm["bbox_h"],
            "view_center_offset": vm["center_offset"],
            "view_border_ratio": vm["border_ratio"],
        }

    img.save(raw_path, quality=95)
    skel = to_skeleton(arr)
    skel.save(skel_path)
    return {
        "status": "accepted",
        "item": item,
        "width": w,
        "height": h,
        "dorsal_score": score,
        "raw_path": str(raw_path),
        "skeleton_path": str(skel_path),
        "view_area": vm["area"],
        "view_bbox_w": vm["bbox_w"],
        "view_bbox_h": vm["bbox_h"],
        "view_center_offset": vm["center_offset"],
        "view_border_ratio": vm["border_ratio"],
    }


def run(args):
    outdir = Path(args.outdir)
    raw_dir = outdir / "raw_images"
    skel_dir = outdir / "skeletons"
    outdir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    skel_dir.mkdir(parents=True, exist_ok=True)

    report_path = outdir / "report.csv"
    fieldnames = [
        "status",
        "occurrence_key",
        "institutionCode",
        "family",
        "species",
        "license",
        "width",
        "height",
        "dorsal_score",
        "view_area",
        "view_bbox_w",
        "view_bbox_h",
        "view_center_offset",
        "view_border_ratio",
        "raw_path",
        "skeleton_path",
        "url",
    ]

    candidates = gather_candidates(args.institutions, args.pages_per_query, args.page_limit)
    print(f"Candidate images from museum APIs: {len(candidates)}")
    if not candidates:
        print("No candidate images found.")
        return

    accepted = 0
    processed = 0
    write_header = not report_path.exists()
    with report_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = [
                ex.submit(process_candidate, item, raw_dir, skel_dir, args.min_dim, args.dorsal_threshold)
                for item in candidates
            ]
            for fut in as_completed(futures):
                result = fut.result()
                processed += 1
                item = result.get("item", {})
                row = {
                    "status": result.get("status", ""),
                    "occurrence_key": item.get("occurrence_key", ""),
                    "institutionCode": item.get("institutionCode", ""),
                    "family": item.get("family", ""),
                    "species": item.get("species", ""),
                    "license": item.get("license", ""),
                    "width": result.get("width", ""),
                    "height": result.get("height", ""),
                    "dorsal_score": f"{result.get('dorsal_score', ''):.4f}" if "dorsal_score" in result else "",
                    "view_area": f"{result.get('view_area', ''):.4f}" if "view_area" in result else "",
                    "view_bbox_w": f"{result.get('view_bbox_w', ''):.4f}" if "view_bbox_w" in result else "",
                    "view_bbox_h": f"{result.get('view_bbox_h', ''):.4f}" if "view_bbox_h" in result else "",
                    "view_center_offset": f"{result.get('view_center_offset', ''):.4f}" if "view_center_offset" in result else "",
                    "view_border_ratio": f"{result.get('view_border_ratio', ''):.4f}" if "view_border_ratio" in result else "",
                    "raw_path": result.get("raw_path", ""),
                    "skeleton_path": result.get("skeleton_path", ""),
                    "url": item.get("url", ""),
                }
                writer.writerow(row)

                if result.get("status") == "accepted":
                    accepted += 1
                if processed % 20 == 0:
                    print(f"Processed {processed}, accepted {accepted}")
                if accepted >= args.target_count:
                    break

    print(f"Done. accepted={accepted} target={args.target_count}")
    print(f"Raw images: {raw_dir}")
    print(f"Skeletons: {skel_dir}")
    print(f"Report: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Museum butterfly pipeline: fetch/filter/skeletonize and HF-based ideal dorsal aggregation."
    )
    parser.add_argument(
        "--build-ideal",
        action="store_true",
        help="Build an ideal dorsal pattern from existing local images using Hugging Face embeddings.",
    )
    parser.add_argument(
        "--build-math-outline",
        action="store_true",
        help="Build a dark mathematical-outline render from existing local images.",
    )
    parser.add_argument(
        "--build-math-dataset",
        action="store_true",
        help="Generate a new dataset of per-image mathematical reconstructions from raw images.",
    )
    parser.add_argument(
        "--build-ideal-math",
        action="store_true",
        help="Build an ideal mathematical reconstruction from the generated mathematical dataset.",
    )
    parser.add_argument("--ideal-input-dir", default="museum_butterfly_dorsal/raw_images")
    parser.add_argument("--ideal-output", default="museum_butterfly_dorsal/ideal_dorsal_pattern.png")
    parser.add_argument("--math-output", default="museum_butterfly_dorsal/mathematical_outline.png")
    parser.add_argument("--math-dataset-input", default="museum_butterfly_dorsal/raw_images")
    parser.add_argument("--math-dataset-dir", default="museum_butterfly_dorsal/mathematical_dataset")
    parser.add_argument("--ideal-math-output", default="museum_butterfly_dorsal/ideal_math_reconstruction.png")
    parser.add_argument(
        "--ideal-math-raw-ref-dir",
        default="museum_butterfly_dorsal/raw_images",
        help="Raw dorsal image directory used for ideal-math palette and line references.",
    )
    parser.add_argument("--math-min-dorsal-score", type=float, default=0.60)
    parser.add_argument(
        "--prune-non-dorsal",
        action="store_true",
        help="Move non-dorsal/non-view images out of an existing local image folder.",
    )
    parser.add_argument("--prune-input-dir", default="museum_butterfly_dorsal/raw_images")
    parser.add_argument("--prune-threshold", type=float, default=0.66)
    parser.add_argument("--hf-model", default="google/vit-base-patch16-224-in21k")
    parser.add_argument("--hf-batch-size", type=int, default=8)
    parser.add_argument("--ideal-top-k", type=int, default=50)
    parser.add_argument("--ideal-canvas-width", type=int, default=1400)
    parser.add_argument("--ideal-canvas-height", type=int, default=900)
    parser.add_argument("--ideal-consensus-threshold", type=float, default=0.22)

    parser.add_argument("--outdir", default="museum_butterfly_dorsal")
    parser.add_argument("--target-count", type=int, default=50)
    parser.add_argument("--min-dim", type=int, default=1200)
    parser.add_argument("--dorsal-threshold", type=float, default=0.62)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--pages-per-query", type=int, default=2)
    parser.add_argument("--page-limit", type=int, default=100)
    parser.add_argument(
        "--institutions",
        nargs="+",
        default=["nhmuk", "usnm", "am"],
        help="GBIF institutionCode values",
    )
    args = parser.parse_args()

    if args.prune_non_dorsal:
        prune_non_dorsal_images(args.prune_input_dir, args.prune_threshold)
        if not (args.build_ideal or args.build_math_outline or args.build_math_dataset or args.build_ideal_math):
            return

    if args.build_math_dataset:
        build_math_reconstruction_dataset(
            input_dir=args.math_dataset_input,
            output_dir=args.math_dataset_dir,
            min_dorsal_score=args.math_min_dorsal_score,
        )
        if not args.build_ideal_math:
            return

    if args.build_ideal_math:
        build_ideal_math_reconstruction(
            math_dataset_dir=args.math_dataset_dir,
            output_path=args.ideal_math_output,
            hf_model=args.hf_model,
            hf_batch_size=args.hf_batch_size,
            top_k=args.ideal_top_k,
            canvas_w=args.ideal_canvas_width,
            canvas_h=args.ideal_canvas_height,
            consensus_threshold=args.ideal_consensus_threshold,
            raw_image_dir=args.ideal_math_raw_ref_dir,
        )
    elif args.build_math_outline:
        build_mathematical_outline(
            image_dir=args.ideal_input_dir,
            output_path=args.math_output,
            hf_model=args.hf_model,
            hf_batch_size=args.hf_batch_size,
            top_k=args.ideal_top_k,
            canvas_w=args.ideal_canvas_width,
            canvas_h=args.ideal_canvas_height,
            consensus_threshold=args.ideal_consensus_threshold,
        )
    elif args.build_ideal:
        build_ideal_dorsal_pattern(
            image_dir=args.ideal_input_dir,
            output_path=args.ideal_output,
            hf_model=args.hf_model,
            hf_batch_size=args.hf_batch_size,
            top_k=args.ideal_top_k,
            canvas_w=args.ideal_canvas_width,
            canvas_h=args.ideal_canvas_height,
            consensus_threshold=args.ideal_consensus_threshold,
        )
    else:
        run(args)


if __name__ == "__main__":
    main()
