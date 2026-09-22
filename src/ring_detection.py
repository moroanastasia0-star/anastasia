"""Per-frame ring geometry detection and lower-water-surface exclusion.

The ring is not assumed to be a perfect circle -- we only use its connected
component to derive an approximate center/radius/bbox for building a
*dynamic* exclusion boundary that keeps the lower water surface (and its
ripples/reflections) out of the final alpha, while still keeping droplets
and splashes that visually belong to the suspended ring.

Two-tier masking
-----------------
In the source footage the ring's lowest point and the nearest lower-surface
ripple arc are sometimes genuinely touching in pixel space (no gap at all),
so plain connected-component analysis on the full-detail mask cannot split
them apart -- they are one blob. Splitting requires a *strong* morphological
opening, but that same strong opening would also erase the very things we
must keep: small droplets and thin filaments. The fix is to use two masks:
  - a heavily opened "geometry" mask, used only to sever the thin neck
    between ring and ripple and locate the ring's true silhouette/bbox;
  - the original fine-detail mask, used for the actual kept pixels/alpha.
The geometry mask's (cleaned) silhouette then defines a per-column cutoff
line that gates the fine mask.
"""
import numpy as np
import cv2


def _binary(alpha, bin_thresh=50):
    return (alpha >= bin_thresh).astype(np.uint8) * 255


def _clean_binary(alpha, bin_thresh=50, open_ksize=3, close_ksize=5):
    binary = _binary(alpha, bin_thresh)
    if open_ksize:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)
    if close_ksize:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
    return binary


def detect_ring_geometry(alpha, bin_thresh=50, geometry_open_ksize=13):
    """Find the main ring in a single frame's raw alpha.

    Uses a strongly-opened "geometry" mask (severs any thin bridge to the
    lower ripple surface) to reliably find the ring's silhouette, bbox,
    center and an approximate radius, plus a fine-detail mask (light
    cleanup only) that keeps droplets/filaments for the actual output.

    Returns a dict, or None if nothing found.
    """
    h, w = alpha.shape[:2]

    fine = _clean_binary(alpha, bin_thresh=bin_thresh, open_ksize=3, close_ksize=5)
    n_fine, labels_fine, stats_fine, centroids_fine = cv2.connectedComponentsWithStats(fine, connectivity=8)
    if n_fine <= 1:
        return None

    geo_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (geometry_open_ksize, geometry_open_ksize))
    geometry_mask = cv2.morphologyEx(_binary(alpha, bin_thresh), cv2.MORPH_OPEN, geo_k)
    n_geo, labels_geo, stats_geo, centroids_geo = cv2.connectedComponentsWithStats(geometry_mask, connectivity=8)
    if n_geo <= 1:
        # opening wiped everything out (e.g. very thin/faint splash-only
        # frame) -- fall back to the fine mask for geometry too
        labels_geo, stats_geo, centroids_geo, n_geo = labels_fine, stats_fine, centroids_fine, n_fine

    best_label, best_score = None, -1.0
    for lbl in range(1, n_geo):
        x, y, bw, bh, area = stats_geo[lbl]
        if area < 0.003 * w * h:
            continue
        aspect = bw / max(bh, 1)
        cy = centroids_geo[lbl][1]
        flatness_penalty = max(0.0, aspect - 2.2) * 1.5
        low_position_penalty = max(0.0, (cy / h) - 0.72) * 6.0
        score = area - flatness_penalty * (0.01 * w * h) - low_position_penalty * (0.01 * w * h)
        if score > best_score:
            best_score = score
            best_label = lbl

    if best_label is None:
        return None

    ring_geo_mask = (labels_geo == best_label)
    x, y, bw, bh, area = stats_geo[best_label]
    cx, cy = centroids_geo[best_label]
    radius = 0.25 * (bw + bh)

    # map the geometry-space ring back onto the fine-detail labels: any
    # fine component that overlaps the (dilated, to re-cover droplets that
    # the opening ate) geometry ring blob is considered part of the ring
    # for keep purposes.
    reach = cv2.dilate(ring_geo_mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (geometry_open_ksize + 8, geometry_open_ksize + 8)))
    overlapping_fine_labels = set(np.unique(labels_fine[(reach > 0) & (labels_fine > 0)]).tolist())

    return {
        "bbox": (int(x), int(y), int(bw), int(bh)),
        "center": (float(cx), float(cy)),
        "radius": float(radius),
        "area": int(area),
        "ring_geo_mask": ring_geo_mask,
        "fine_labels": labels_fine,
        "fine_stats": stats_fine,
        "fine_centroids": centroids_fine,
        "n_fine": n_fine,
        "ring_fine_labels": overlapping_fine_labels,
    }


def build_keep_mask(alpha, geometry, lower_margin=30.0,
                     proximity_factor=1.9, feather_px=9, hard_cutoff_margin=22.0):
    """Build a soft (0..1) geometric keep-mask for one frame: the ring blob
    plus any nearby droplets/splashes, excluding the lower water surface.

    lower_margin: how close (in px) a *separate* component's centroid may
        sit below the ring's local bottom silhouette and still be treated
        as an attached droplet rather than lower-surface ripple.
    proximity_factor: a component is kept if its centroid lies within
        `proximity_factor * radius` of the ring center (catches flying
        droplets), unless it also falls below the lower cutoff line.
    hard_cutoff_margin: margin (px) added below the ring's own *clean*
        (geometry-mask-derived) local bottom silhouette before hard-gating
        everything below to zero. This is what guarantees the lower water
        surface can never leak through, even when it is pixel-connected to
        the ring in the fine-detail mask.
    """
    h, w = alpha.shape[:2]
    if geometry is None:
        return np.zeros((h, w), dtype=np.float32)

    labels = geometry["fine_labels"]
    stats = geometry["fine_stats"]
    centroids = geometry["fine_centroids"]
    ring_fine_labels = geometry["ring_fine_labels"]
    cx, cy = geometry["center"]
    radius = max(geometry["radius"], 1.0)

    keep = np.zeros((h, w), dtype=np.uint8)
    ring_mask = np.isin(labels, list(ring_fine_labels)) if ring_fine_labels else np.zeros((h, w), dtype=bool)
    keep[ring_mask] = 255

    # per-column bottom profile of the ring's clean GEOMETRY silhouette
    # (not the fine mask, which may still be fused to the ripple) --
    # this is what the hard cutoff line follows.
    geo_mask = geometry["ring_geo_mask"]
    cols_with_ring = np.where(geo_mask.any(axis=0))[0]
    by, bh_bottom = geometry["bbox"][1], geometry["bbox"][1] + geometry["bbox"][3]
    bottom_profile = np.full(w, float(bh_bottom), dtype=np.float32)
    if len(cols_with_ring) > 0:
        ys_idx = np.where(geo_mask, np.arange(h)[:, None], -1)
        col_bottom = ys_idx.max(axis=0).astype(np.float32)
        valid = col_bottom >= 0
        bottom_profile[valid] = col_bottom[valid]
        first, last = cols_with_ring[0], cols_with_ring[-1]
        bottom_profile[:first] = bottom_profile[first]
        bottom_profile[last + 1:] = bottom_profile[last]
    # smooth the profile so the cutoff line follows the ring's gentle
    # natural undulation but not any single-column spike
    bottom_profile = cv2.GaussianBlur(bottom_profile.reshape(1, -1), (0, 0), sigmaX=6).ravel()
    local_cutoff = bottom_profile + hard_cutoff_margin

    for lbl in range(1, geometry["n_fine"]):
        if lbl in ring_fine_labels:
            continue
        lx, ly, lbw, lbh, larea = stats[lbl]
        lcx, lcy = centroids[lbl]
        if larea < 4:
            continue
        dist = np.hypot(lcx - cx, lcy - cy)
        col = int(np.clip(lcx, 0, w - 1))
        local_limit = bottom_profile[col] + lower_margin
        is_low_wide_band = (lbw > 2.5 * max(lbh, 1)) and (lcy > h * 0.72)
        if is_low_wide_band:
            continue
        if lcy > local_limit:
            continue
        if dist <= proximity_factor * radius:
            keep[labels == lbl] = 255

    transition = max(int(feather_px * 2), 1)
    row_idx = np.arange(h, dtype=np.float32)[:, None]
    row_gate = np.clip((local_cutoff[None, :] + transition - row_idx) / (2.0 * transition), 0.0, 1.0)
    keep = (keep.astype(np.float32) / 255.0) * row_gate

    if feather_px > 0:
        keep = cv2.GaussianBlur(keep, (0, 0), sigmaX=feather_px / 3.0)
    return keep
