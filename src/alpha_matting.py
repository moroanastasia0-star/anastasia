"""Temporal consistency and edge-aware alpha refinement.

No optical flow is used for the alpha itself (we must not interpolate or
invent motion); temporal smoothing here is a light 3-tap blend across
neighboring frames, purely to reduce single-frame flicker/noise in the
soft alpha values, not to change timing or shape.
"""
import numpy as np
import cv2


def temporal_smooth_alpha(alpha_sequence, weights=(0.2, 0.6, 0.2)):
    """3-tap temporal smoothing of a list of HxW float32 alpha maps.
    Edges of the sequence are handled by renormalizing the available taps
    (no frame duplication / interpolation across time)."""
    n = len(alpha_sequence)
    if n <= 2:
        return [a.copy() for a in alpha_sequence]
    stack = np.stack(alpha_sequence, axis=0)
    out = np.empty_like(stack)
    w_prev, w_cur, w_next = weights
    out[0] = (w_cur * stack[0] + w_next * stack[1]) / (w_cur + w_next)
    out[-1] = (w_prev * stack[-2] + w_cur * stack[-1]) / (w_prev + w_cur)
    if n > 2:
        out[1:-1] = w_prev * stack[:-2] + w_cur * stack[1:-1] + w_next * stack[2:]
    return [out[i] for i in range(n)]


def _box(img, r):
    k = 2 * r + 1
    return cv2.boxFilter(img, ddepth=-1, ksize=(k, k), borderType=cv2.BORDER_REFLECT)


def guided_filter_alpha(alpha, guide_gray, radius=4, eps=1e-3):
    """Edge-aware smoothing of the alpha channel guided by frame luminance.
    Manual implementation of He et al.'s guided filter (fast box-filter
    form) since opencv-contrib/ximgproc is not assumed to be installed.
    Straightens/tightens the mask boundary against real image edges while
    keeping flat regions (transparency gradients) smooth."""
    I = guide_gray.astype(np.float32) / 255.0
    p = alpha.astype(np.float32) / 255.0

    mean_I = _box(I, radius)
    mean_p = _box(p, radius)
    corr_I = _box(I * I, radius)
    corr_Ip = _box(I * p, radius)

    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = _box(a, radius)
    mean_b = _box(b, radius)

    q = mean_a * I + mean_b
    return np.clip(q * 255.0, 0, 255).astype(np.float32)


def refine_alpha_edges(alpha, frame_bgr, radius=4, eps=1e-3, blend=0.6):
    """Blend the raw alpha with its guided-filter refinement (blend=1 -> use
    fully refined result). Keeping some of the raw alpha preserves the
    finest droplets/filaments that a smoothing filter could thin out."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    refined = guided_filter_alpha(alpha, gray, radius=radius, eps=eps)
    return np.clip(blend * refined + (1 - blend) * alpha, 0, 255).astype(np.float32)
