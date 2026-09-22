"""Background plate estimation, foreground-candidate (difference matting) and
text-region isolation.

Strategy
--------
The scene background (icy-blue gradient + the almost static lower ripple
surface) is a smooth, low-frequency function of pixel position. We fit a
low-degree 2D polynomial to it using robust regression on border regions of
several frames (iteratively re-weighted least squares, so any occasional
water/droplet/text pixel sampled in those regions gets down-weighted as an
outlier). The resulting "clean plate" lets us key every frame against a
*per-pixel* expected background value (not a single flat color), which is
what correctly preserves the natural, partially-transparent look of the
water instead of producing a hard cut-out.
"""
import numpy as np
import cv2


def estimate_background_plate(frames_bgr, text_mask=None, std_thresh=7.0, inpaint_radius=25):
    """Reconstruct the smooth background plate (H,W,3 float32, BGR).

    The background (icy-blue gradient + the near-static lower ripple
    surface) never moves; the ring/droplets do. So the per-pixel temporal
    median already IS the correct background almost everywhere (including
    the ring's own "hole" and the ripple band). The only pixels that need
    to be reconstructed are the ones the ring occupies most of the time,
    plus the static text overlay -- both identified via a confidence mask
    and filled in by inpainting from the surrounding, already-correct,
    background pixels. This is far more accurate than extrapolating a
    global polynomial, which tends to overshoot in the frame interior.
    """
    stack = np.stack([f.astype(np.float32) for f in frames_bgr], axis=0)
    gray_stack = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in frames_bgr], axis=0)
    median_frame = np.median(stack, axis=0).astype(np.float32)
    std_map = gray_stack.std(axis=0)

    unknown = (std_map >= std_thresh)
    if text_mask is not None:
        unknown |= (text_mask > 0)
    # widen slightly so semi-transparent ring edges (which reduce std only
    # partially) are fully excluded from the "known background" set too
    unknown = cv2.dilate(unknown.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

    plate = cv2.inpaint(median_frame.astype(np.uint8), unknown, inpaint_radius, cv2.INPAINT_TELEA)
    plate = cv2.GaussianBlur(plate, (0, 0), sigmaX=3)
    return plate.astype(np.float32)


def foreground_alpha_raw(frame_bgr, plate, low_thresh=6.0, high_thresh=32.0):
    """Perceptual (Lab) color distance from the background plate, mapped to
    an 8-bit soft alpha via a smoothstep. Small deviations (near-transparent
    water) get low but non-zero alpha; strong deviations (opaque foam) get
    alpha near 255."""
    lab_frame = cv2.cvtColor(frame_bgr.astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_plate = cv2.cvtColor(np.clip(plate, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    dist = np.linalg.norm(lab_frame - lab_plate, axis=2)

    t = (dist - low_thresh) / max(high_thresh - low_thresh, 1e-6)
    t = np.clip(t, 0, 1)
    smooth = t * t * (3 - 2 * t)  # smoothstep
    alpha = (smooth * 255.0).astype(np.float32)
    return alpha, dist


def text_mask_from_frames(frames_bgr, value_thresh=185.0, chroma_thresh=28.0, std_thresh=4.0):
    """Detect the static, near-achromatic (dark gray/black) text overlay.

    Uses three independent, robust cues combined with AND:
      1. Low temporal std  -> text never moves (unlike water).
      2. Low channel spread (B-R almost 0) -> text is desaturated/gray,
         unlike the blue-tinted water and background.
      3. Low brightness (V) -> text is dark ink, unlike bright water/bg.
    """
    stack = np.stack([f.astype(np.float32) for f in frames_bgr], axis=0)
    mean_frame = stack.mean(axis=0)
    std_gray = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in frames_bgr], axis=0).std(axis=0)

    b = mean_frame[..., 0]
    r = mean_frame[..., 2]
    spread = np.abs(b - r)
    value = mean_frame.max(axis=2)

    mask = (std_gray < std_thresh) & (spread < chroma_thresh) & (value < value_thresh)
    mask = mask.astype(np.uint8) * 255

    # clean up small noise, keep the coherent text blob(s)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    return mask
