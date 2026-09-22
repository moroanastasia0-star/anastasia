"""Color spill / edge decontamination and compositing over solid backgrounds.

The source background is a known (estimated) pale icy-blue plate, so at any
partially-transparent pixel the observed color is a blend of the true water
color and that plate color: I = a*F + (1-a)*B. We invert this ("unmixing")
to recover F, which removes the blue/white halo that a naive straight-alpha
cut would otherwise carry at every soft edge.
"""
import numpy as np
import cv2


def decontaminate(frame_bgr, alpha, plate, min_alpha=0.06):
    """Recover the true foreground color at partial-alpha pixels by
    unmixing against the known background plate. Returns an RGB (BGR)
    float32 image with spill removed."""
    a = np.clip(alpha.astype(np.float32) / 255.0, 0.0, 1.0)[..., None]
    I = frame_bgr.astype(np.float32)
    B = plate.astype(np.float32)

    a_safe = np.maximum(a, min_alpha)
    F = (I - (1.0 - a) * B) / a_safe
    F = np.clip(F, 0, 255)

    # fully-opaque pixels: keep the observed color as-is (no need to unmix)
    opaque = (a >= 0.98)
    F = np.where(opaque, I, F)
    return F.astype(np.float32)


def extend_edge_colors(rgb, alpha, valid_thresh=5, band_px=18):
    """Push valid foreground color a little way into the fully-transparent
    region surrounding the mask, so that codec chroma blur/resizing never
    reveals background-plate color peeking through soft edges."""
    valid = (alpha > valid_thresh).astype(np.uint8)
    if valid.all() or not valid.any():
        return rgb
    band_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (band_px, band_px))
    extended_valid = cv2.dilate(valid, band_kernel)
    fill_mask = ((extended_valid > 0) & (valid == 0)).astype(np.uint8)
    if not fill_mask.any():
        return rgb
    out = cv2.inpaint(np.clip(rgb, 0, 255).astype(np.uint8), fill_mask, band_px // 2 + 2, cv2.INPAINT_TELEA)
    return out.astype(np.float32)


def make_rgba(frame_bgr, alpha, plate, feather_edge_px=18):
    """Full decontamination pipeline -> BGRA uint8 frame (straight alpha)."""
    fg = decontaminate(frame_bgr, alpha, plate)
    fg = extend_edge_colors(fg, alpha, band_px=feather_edge_px)
    bgra = np.dstack([np.clip(fg, 0, 255).astype(np.uint8), np.clip(alpha, 0, 255).astype(np.uint8)])
    return bgra


def composite_over(bgra, color_bgr):
    """Composite a BGRA frame over a solid color background (straight
    alpha). Returns a BGR uint8 image."""
    rgb = bgra[..., :3].astype(np.float32)
    a = bgra[..., 3:4].astype(np.float32) / 255.0
    bg = np.array(color_bgr, dtype=np.float32).reshape(1, 1, 3)
    out = rgb * a + bg * (1 - a)
    return np.clip(out, 0, 255).astype(np.uint8)


# common preview background colors (BGR)
BG_BLACK = (0, 0, 0)
BG_WHITE = (255, 255, 255)
BG_ICY_BLUE = (235, 210, 175)  # pale icy blue, matches source palette (BGR)
