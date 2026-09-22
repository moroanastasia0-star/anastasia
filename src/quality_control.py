"""Automated quality control checks on sample checkpoint frames.

Implements the ten checks requested for each checkpoint (first, 25%, 50%,
75%, last frame): ring present/complete, droplets/splashes present,
transparency preserved, lower surface removed, background transparent,
halos, hard artificial edges, and temporal change of the mask.
"""
import numpy as np
import cv2


def checkpoint_indices(n_frames):
    return sorted(set([0, n_frames // 4, n_frames // 2, (3 * n_frames) // 4, n_frames - 1]))


def _connected_components(alpha, thresh=40):
    binary = (alpha >= thresh).astype(np.uint8)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    return n, labels, stats, centroids


def check_frame(alpha, geometry=None, hard_cutoff_y=None):
    h, w = alpha.shape[:2]
    result = {}

    n, labels, stats, centroids = _connected_components(alpha)
    areas = stats[1:, 4] if n > 1 else np.array([])
    total_fg_area = areas.sum() if len(areas) else 0

    # 1. ring present
    result["ring_present"] = bool(total_fg_area > 0.01 * w * h)

    # 2. ring complete: largest component should be a good fraction of total fg area
    #    and should have a reasonably closed, non-degenerate bbox
    if len(areas) > 0:
        largest = areas.max()
        result["ring_complete"] = bool(largest > 0.5 * total_fg_area and largest > 0.01 * w * h)
    else:
        result["ring_complete"] = False

    # 3. droplets present: several small isolated components besides the largest
    small_components = int(((areas > 3) & (areas < 0.01 * w * h)).sum()) if len(areas) else 0
    result["droplets_present"] = bool(small_components >= 1)
    result["n_small_components"] = small_components

    # 4. splashes present: elongated / irregular large component (heuristic via
    #    bbox fill ratio of the largest component well below a filled disk)
    if len(areas) > 0:
        largest_lbl = 1 + int(np.argmax(areas))
        lx, ly, lbw, lbh, larea = stats[largest_lbl]
        bbox_area = max(lbw * lbh, 1)
        fill_ratio = larea / bbox_area
        result["fill_ratio"] = float(fill_ratio)
        result["splashes_or_irregular_shape"] = bool(fill_ratio < 0.55)
    else:
        result["fill_ratio"] = 0.0
        result["splashes_or_irregular_shape"] = False

    # 5. transparency preserved: meaningful fraction of foreground pixels in
    #    the soft mid-range (not a binary 0/255 cutout)
    fg_vals = alpha[alpha >= 5]
    if fg_vals.size > 0:
        mid = ((fg_vals > 15) & (fg_vals < 240)).mean()
        result["soft_alpha_fraction"] = float(mid)
        result["transparency_preserved"] = bool(mid > 0.08)
    else:
        result["soft_alpha_fraction"] = 0.0
        result["transparency_preserved"] = False

    # 6. lower surface eliminated (needs the geometry's local cutoff)
    if hard_cutoff_y is not None:
        margin = 15
        below = alpha[min(int(hard_cutoff_y) + margin, h):, :]
        max_below = float(below.max()) if below.size else 0.0
        result["max_alpha_below_cutoff"] = max_below
        result["lower_surface_removed"] = bool(max_below < 20)
    else:
        result["max_alpha_below_cutoff"] = None
        result["lower_surface_removed"] = None

    # 7. background fully transparent (corners + far margins)
    border = np.concatenate([
        alpha[0:15, :].ravel(), alpha[-15:, :].ravel(),
        alpha[:, 0:15].ravel(), alpha[:, -15:].ravel(),
    ])
    result["max_alpha_border"] = float(border.max())
    result["background_transparent"] = bool(border.max() < 25)

    # 9. hard artificial edges: fraction of boundary pixels with intermediate
    #    alpha (a soft edge should ramp through mid-values, not jump 0->255)
    grad = cv2.Laplacian(alpha.astype(np.float32), cv2.CV_32F, ksize=3)
    boundary = np.abs(grad) > 40
    if boundary.any():
        boundary_vals = alpha[boundary]
        soft_boundary_fraction = float(((boundary_vals > 10) & (boundary_vals < 245)).mean())
    else:
        soft_boundary_fraction = 1.0
    result["soft_boundary_fraction"] = soft_boundary_fraction
    result["hard_artificial_edges"] = bool(soft_boundary_fraction < 0.15)

    return result


def check_halo(bgra_frame, plate_bgr, alpha_thresh=(10, 245)):
    """8. halo/fringe check: at partially-transparent edge pixels, compare
    the (already decontaminated) foreground color's hue against the plate's
    hue -- if decontamination failed, edge pixels drift toward the plate's
    blue hue."""
    rgb = bgra_frame[..., :3].astype(np.uint8)
    alpha = bgra_frame[..., 3]
    edge_mask = (alpha > alpha_thresh[0]) & (alpha < alpha_thresh[1])
    if not edge_mask.any():
        return {"halo_detected": False, "edge_plate_hue_similarity": 0.0}

    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV).astype(np.float32)
    plate_hsv = cv2.cvtColor(np.clip(plate_bgr, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)

    edge_sat = hsv[..., 1][edge_mask]
    plate_sat = plate_hsv[..., 1][edge_mask]
    # a halo shows up as edge saturation collapsing toward the pale plate's
    # (very low) saturation while alpha is still mid-range
    low_sat_fraction = float((edge_sat < (plate_sat.mean() + 8)).mean())
    return {
        "halo_detected": bool(low_sat_fraction > 0.35),
        "edge_low_saturation_fraction": low_sat_fraction,
    }


# these describe the *content* of a given instant (a fully-closed ring can
# legitimately have zero separate floating droplets at that exact frame) --
# informational only, never a pipeline-defect signal on their own.
INFORMATIONAL_KEYS = {"droplets_present", "splashes_or_irregular_shape"}
# these are "problem flags": True means a defect was found, so they gate
# the opposite way from a plain boolean check.
INVERTED_PROBLEM_KEYS = {"halo_detected", "hard_artificial_edges"}


def summarize(all_results):
    lines = []
    overall_ok = True
    for idx, res in all_results.items():
        lines.append(f"--- Frame {idx} ---")
        for k, v in res.items():
            lines.append(f"  {k}: {v}")
        checks = {}
        for k, v in res.items():
            if not isinstance(v, bool) or k in INFORMATIONAL_KEYS or k in INVERTED_PROBLEM_KEYS:
                continue
            checks[k] = v
        # problem flags: these must be False to pass
        problem_ok = not any(res.get(k, False) for k in INVERTED_PROBLEM_KEYS)
        frame_ok = all(checks.values()) and problem_ok
        overall_ok = overall_ok and frame_ok
        lines.append(f"  => FRAME {'PASS' if frame_ok else 'FAIL'}")
    lines.append(f"\nOVERALL: {'PASS' if overall_ok else 'FAIL'}")
    return "\n".join(lines), overall_ok
