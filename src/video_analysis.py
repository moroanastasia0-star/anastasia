"""Video probing and frame extraction utilities."""
import json
import os
import subprocess
import glob
import cv2
import numpy as np


def probe_video(video_path):
    """Return basic technical metadata about a video using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration,codec_name",
        "-show_entries", "format=duration",
        "-of", "json", video_path,
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(out.stdout)
    stream = data["streams"][0]

    def parse_rate(r):
        num, den = r.split("/")
        return float(num) / float(den) if float(den) != 0 else 0.0

    fps = parse_rate(stream.get("r_frame_rate", "0/1"))
    duration = float(data.get("format", {}).get("duration") or stream.get("duration") or 0.0)
    nb_frames = stream.get("nb_frames")
    if nb_frames is None or nb_frames == "N/A":
        nb_frames = int(round(duration * fps))
    else:
        nb_frames = int(nb_frames)

    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": fps,
        "duration": duration,
        "nb_frames": nb_frames,
        "codec": stream.get("codec_name", "unknown"),
    }


def extract_frames(video_path, out_dir, pattern="f_%05d.png"):
    """Extract every frame as PNG, preserving original order/timing (no re-encoding of timing)."""
    os.makedirs(out_dir, exist_ok=True)
    for f in glob.glob(os.path.join(out_dir, "f_*.png")):
        os.remove(f)
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vsync", "0", os.path.join(out_dir, pattern),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    files = sorted(glob.glob(os.path.join(out_dir, "f_*.png")))
    return files


def load_frames_bgr(frame_files):
    return [cv2.imread(f, cv2.IMREAD_COLOR) for f in frame_files]


def temporal_std_map(frames_bgr):
    """Per-pixel temporal standard deviation (grayscale), useful to find static vs dynamic regions."""
    stack = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in frames_bgr], axis=0)
    return stack.std(axis=0)
