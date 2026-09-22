"""Export helpers: PNG RGBA sequences, alpha-channel video (WebM VP9 / ProRes
4444), and plain preview MP4s -- all encoded at the source video's exact
original frame rate and frame count (no retiming)."""
import os
import glob
import subprocess
import shutil
import cv2
import numpy as np


def write_png_sequence(rgba_frames, out_dir, prefix="frame"):
    os.makedirs(out_dir, exist_ok=True)
    for f in glob.glob(os.path.join(out_dir, f"{prefix}_*.png")):
        os.remove(f)
    paths = []
    for i, bgra in enumerate(rgba_frames):
        path = os.path.join(out_dir, f"{prefix}_{i:05d}.png")
        cv2.imwrite(path, bgra)
        paths.append(path)
    return paths


def write_bgr_sequence(bgr_frames, out_dir, prefix="frame"):
    os.makedirs(out_dir, exist_ok=True)
    for f in glob.glob(os.path.join(out_dir, f"{prefix}_*.png")):
        os.remove(f)
    paths = []
    for i, bgr in enumerate(bgr_frames):
        path = os.path.join(out_dir, f"{prefix}_{i:05d}.png")
        cv2.imwrite(path, bgr)
        paths.append(path)
    return paths


def _run_ffmpeg(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {' '.join(cmd)}\n{result.stderr[-3000:]}")
    return result


def encode_webm_alpha(png_dir, fps, out_path, prefix="frame"):
    """NOTE on alpha playback: WebM stores VP9 alpha as an extra
    BlockAdditional plane. FFmpeg's *native* vp9 decoder (picked by
    default when just probing/opening the file) ignores it and reports
    yuv420p; only the libvpx-based decoder honors it. This is a
    decoder-selection quirk of ffmpeg, not a flaw in the encoded file --
    real alpha-aware consumers (browsers, libvpx, most NLEs) decode it
    correctly. When WE need to read our own alpha back (QC, previews),
    always pass -c:v libvpx-vp9 on the input side; see decode_webm_alpha().
    """
    pattern = os.path.join(png_dir, f"{prefix}_%05d.png")
    cmd = [
        "ffmpeg", "-y", "-framerate", str(fps), "-i", pattern,
        "-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p",
        "-auto-alt-ref", "0", "-b:v", "0", "-crf", "22",
        "-r", str(fps),
        out_path,
    ]
    _run_ffmpeg(cmd)
    return out_path


def decode_webm_alpha_frame(webm_path, frame_index, fps, out_png_path):
    """Decode a single frame of an alpha WebM back to RGBA PNG, forcing the
    libvpx decoder so the alpha plane is actually read (see note above)."""
    timestamp = frame_index / float(fps)
    cmd = [
        "ffmpeg", "-y", "-c:v", "libvpx-vp9", "-ss", f"{timestamp:.6f}",
        "-i", webm_path, "-update", "1", "-frames:v", "1",
        "-pix_fmt", "rgba", out_png_path,
    ]
    _run_ffmpeg(cmd)
    return out_png_path


def encode_prores4444(png_dir, fps, out_path, prefix="frame"):
    pattern = os.path.join(png_dir, f"{prefix}_%05d.png")
    cmd = [
        "ffmpeg", "-y", "-framerate", str(fps), "-i", pattern,
        "-c:v", "prores_ks", "-profile:v", "4444", "-pix_fmt", "yuva444p10le",
        "-r", str(fps),
        out_path,
    ]
    try:
        _run_ffmpeg(cmd)
        return out_path
    except RuntimeError:
        return None


def encode_mp4(png_dir, fps, out_path, prefix="frame"):
    pattern = os.path.join(png_dir, f"{prefix}_%05d.png")
    cmd = [
        "ffmpeg", "-y", "-framerate", str(fps), "-i", pattern,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        "-r", str(fps),
        out_path,
    ]
    _run_ffmpeg(cmd)
    return out_path


def encode_mask_preview(alpha_frames, fps, out_dir, out_path, prefix="mask"):
    gray_frames = [np.repeat(a.astype(np.uint8)[..., None], 3, axis=2) for a in alpha_frames]
    write_bgr_sequence(gray_frames, out_dir, prefix=prefix)
    return encode_mp4(out_dir, fps, out_path, prefix=prefix)


def ffmpeg_available():
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
