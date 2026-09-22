"""End-to-end pipeline orchestrator.

VIDEO -> FRAME EXTRACTION -> VIDEO ANALYSIS -> BACKGROUND ESTIMATION ->
WATER REGION DETECTION -> RING REGION DETECTION -> LOWER WATER SURFACE
REMOVAL -> TEMPORAL CONSISTENCY -> EDGE REFINEMENT -> ALPHA MATTE ->
RGBA COMPOSITING -> TRANSPARENT VIDEO -> QUALITY CONTROL

The module is split into a cheap "prepare" step (frame extraction + text
mask + background plate -- done once per video) and a per-frame
"process_frame" step (segmentation + ring geometry + matting), so a UI can
prepare a video once and then re-process a handful of preview frames
instantly whenever the user tweaks a parameter, without re-running the
expensive whole-video steps every time.
"""
import os
import sys
import time
import json
import shutil

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import cv2

import video_analysis as va
import segmentation as seg
import ring_detection as rd
import alpha_matting as am
import compositing as comp
import export as exp
import quality_control as qc


DEFAULT_PARAMS = {
    "bin_thresh": 50,
    "geometry_open_ksize": 13,
    "lower_margin": 30.0,
    "proximity_factor": 1.9,
    "hard_cutoff_margin": 22.0,
    "feather_px": 9,
    "guided_radius": 4,
    "guided_eps": 1e-3,
    "guided_blend": 0.6,
    "edge_extend_px": 18,
    "low_thresh": 6.0,
    "high_thresh": 32.0,
}


def log(msg, progress_cb=None):
    print(f"[pipeline] {msg}", flush=True)
    if progress_cb:
        progress_cb(msg)


def prepare_state(video_path, frames_raw_dir=None, progress_cb=None):
    """Run the expensive, per-video-only steps once: probing, frame
    extraction, text detection and background plate estimation."""
    log("Analisi tecnica del video (risoluzione, fps, durata)...", progress_cb)
    info = va.probe_video(video_path)
    log(f"  {info['width']}x{info['height']} @ {info['fps']:.3f}fps, "
        f"{info['duration']:.3f}s, {info['nb_frames']} frame, codec={info['codec']}", progress_cb)

    log("Estrazione di tutti i frame (ordine e timing originali)...", progress_cb)
    if frames_raw_dir is None:
        frames_raw_dir = os.path.join(os.path.dirname(video_path), "_frames_raw")
    frame_files = va.extract_frames(video_path, frames_raw_dir)
    frames = va.load_frames_bgr(frame_files)
    log(f"  {len(frames)} frame estratti", progress_cb)

    log("Rilevamento testo statico sovraimpresso...", progress_cb)
    text_mask = seg.text_mask_from_frames(frames)

    log("Stima dello sfondo pulito (clean plate, robusta agli outlier)...", progress_cb)
    plate = seg.estimate_background_plate(frames, text_mask=text_mask)

    return {
        "video_path": video_path,
        "info": info,
        "frames": frames,
        "text_mask": text_mask,
        "plate": plate,
    }


def process_frame(state, idx, params=None):
    """Run steps 4-10 (segmentation -> ring geometry -> lower-surface
    exclusion -> edge refinement -> decontaminated RGBA) for one frame,
    using the cached per-video state. Fast enough for interactive use."""
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)

    frame = state["frames"][idx]
    plate = state["plate"]
    text_mask = state["text_mask"]

    alpha_raw, _ = seg.foreground_alpha_raw(frame, plate, low_thresh=p["low_thresh"], high_thresh=p["high_thresh"])
    alpha_water = alpha_raw.copy()
    alpha_water[text_mask > 0] = 0

    geometry = rd.detect_ring_geometry(alpha_water, bin_thresh=p["bin_thresh"],
                                        geometry_open_ksize=p["geometry_open_ksize"])
    keep = rd.build_keep_mask(alpha_water, geometry, lower_margin=p["lower_margin"],
                               proximity_factor=p["proximity_factor"], feather_px=p["feather_px"],
                               hard_cutoff_margin=p["hard_cutoff_margin"])
    alpha_final = alpha_water * keep
    alpha_text_final = np.maximum(alpha_final, alpha_raw * (text_mask > 0).astype(np.float32))

    alpha_final = am.refine_alpha_edges(alpha_final, frame, radius=p["guided_radius"],
                                         eps=p["guided_eps"], blend=p["guided_blend"])
    alpha_text_final = am.refine_alpha_edges(alpha_text_final, frame, radius=p["guided_radius"],
                                              eps=p["guided_eps"], blend=p["guided_blend"])

    rgba_water = comp.make_rgba(frame, alpha_final, plate, feather_edge_px=p["edge_extend_px"])
    rgba_text = comp.make_rgba(frame, alpha_text_final, plate, feather_edge_px=p["edge_extend_px"])

    return {
        "geometry": geometry,
        "alpha_water": alpha_final,
        "alpha_text": alpha_text_final,
        "rgba_water": rgba_water,
        "rgba_text": rgba_text,
    }


def run_pipeline_from_state(state, output_root, params=None, progress_cb=None):
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)

    t_start = time.time()
    os.makedirs(output_root, exist_ok=True)
    frames_out_dir = os.path.join(output_root, "frames", "water_ring_frames")
    frames_out_text_dir = os.path.join(output_root, "frames", "water_ring_with_text_frames")
    masks_dir = os.path.join(output_root, "masks", "mask_frames_png")
    output_dir = os.path.join(output_root, "output")
    for d in (frames_out_dir, frames_out_text_dir, masks_dir, output_dir):
        os.makedirs(d, exist_ok=True)

    frames = state["frames"]
    info = state["info"]
    n = len(frames)

    log("Segmentazione per-frame (differenza percettiva dal plate + geometria dell'anello)...", progress_cb)
    alpha_water_list, alpha_with_text_list, geometries = [], [], []
    rgba_water, rgba_text = [], []
    for i in range(n):
        out = process_frame(state, i, p)
        alpha_water_list.append(out["alpha_water"])
        alpha_with_text_list.append(out["alpha_text"])
        geometries.append(out["geometry"])
        if (i + 1) % 20 == 0 or i == n - 1:
            log(f"  frame {i + 1}/{n}", progress_cb)

    log("Coerenza temporale (leggero smoothing anti-flicker, nessuna interpolazione)...", progress_cb)
    alpha_water_list = am.temporal_smooth_alpha(alpha_water_list)
    alpha_with_text_list = am.temporal_smooth_alpha(alpha_with_text_list)

    log("Decontaminazione colore (rimozione aloni) e compositing RGBA...", progress_cb)
    plate = state["plate"]
    for i in range(n):
        rgba_water.append(comp.make_rgba(frames[i], alpha_water_list[i], plate, feather_edge_px=p["edge_extend_px"]))
        rgba_text.append(comp.make_rgba(frames[i], alpha_with_text_list[i], plate, feather_edge_px=p["edge_extend_px"]))

    log("Controllo qualita automatico sui frame campione...", progress_cb)
    idxs = qc.checkpoint_indices(n)
    qc_results = {}
    for idx in idxs:
        geom = geometries[idx]
        cutoff_y = None
        if geom is not None:
            bx, by, bw, bh = geom["bbox"]
            cutoff_y = by + bh + p["hard_cutoff_margin"]
        res = qc.check_frame(alpha_water_list[idx], geometry=geom, hard_cutoff_y=cutoff_y)
        halo_res = qc.check_halo(rgba_water[idx], plate)
        res.update(halo_res)
        qc_results[idx] = res
    qc_report, qc_pass = qc.summarize(qc_results)
    log(qc_report, progress_cb)

    log("Esportazione: PNG RGBA sequence...", progress_cb)
    exp.write_png_sequence(rgba_water, frames_out_dir, prefix="frame")
    exp.write_png_sequence(rgba_text, frames_out_text_dir, prefix="frame")

    fps = info["fps"]
    results = {"info": info, "qc_report": qc_report, "qc_pass": qc_pass, "params": p}

    log("Esportazione: WebM VP9 con alpha...", progress_cb)
    webm_path = os.path.join(output_dir, "water_ring_transparent.webm")
    exp.encode_webm_alpha(frames_out_dir, fps, webm_path)
    results["webm"] = webm_path

    log("Esportazione: ProRes 4444 (se disponibile)...", progress_cb)
    mov_path = os.path.join(output_dir, "water_ring_transparent.mov")
    mov_result = exp.encode_prores4444(frames_out_dir, fps, mov_path)
    results["mov"] = mov_result

    log("Esportazione: WebM/PNG per la versione con testo...", progress_cb)
    webm_text_path = os.path.join(output_dir, "water_ring_with_text.webm")
    exp.encode_webm_alpha(frames_out_text_dir, fps, webm_text_path)
    results["webm_with_text"] = webm_text_path

    log("Esportazione: preview compositate (nero / bianco / azzurro)...", progress_cb)
    scratch_dir = os.path.join(output_root, "_scratch_preview_png")
    black_dir = os.path.join(scratch_dir, "black")
    white_dir = os.path.join(scratch_dir, "white")
    blue_dir = os.path.join(scratch_dir, "blue")
    black_frames = [comp.composite_over(f, comp.BG_BLACK) for f in rgba_water]
    white_frames = [comp.composite_over(f, comp.BG_WHITE) for f in rgba_water]
    blue_frames = [comp.composite_over(f, comp.BG_ICY_BLUE) for f in rgba_water]
    exp.write_bgr_sequence(black_frames, black_dir, prefix="frame")
    exp.write_bgr_sequence(white_frames, white_dir, prefix="frame")
    exp.write_bgr_sequence(blue_frames, blue_dir, prefix="frame")
    results["preview_black"] = exp.encode_mp4(black_dir, fps, os.path.join(output_dir, "preview_black.mp4"))
    results["preview_white"] = exp.encode_mp4(white_dir, fps, os.path.join(output_dir, "preview_white.mp4"))
    results["preview_blue"] = exp.encode_mp4(blue_dir, fps, os.path.join(output_dir, "preview_blue.mp4"))

    log("Esportazione: mask_preview.mp4...", progress_cb)
    results["mask_preview"] = exp.encode_mask_preview(
        alpha_water_list, fps, masks_dir, os.path.join(output_dir, "mask_preview.mp4"))

    results["frames_dir"] = frames_out_dir
    results["frames_with_text_dir"] = frames_out_text_dir
    results["output_dir"] = output_dir
    results["elapsed_s"] = time.time() - t_start

    with open(os.path.join(output_dir, "qc_report.txt"), "w") as f:
        f.write(qc_report)
    with open(os.path.join(output_dir, "run_info.json"), "w") as f:
        json.dump({"info": info, "params": p, "qc_pass": qc_pass,
                   "elapsed_s": results["elapsed_s"]}, f, indent=2)

    if os.path.isdir(scratch_dir):
        shutil.rmtree(scratch_dir)

    log(f"Pipeline completata in {results['elapsed_s']:.1f}s", progress_cb)
    return results


def run_pipeline(video_path, output_root, params=None, progress_cb=None, keep_raw_frames=False):
    frames_raw_dir = os.path.join(output_root, "_frames_raw")
    state = prepare_state(video_path, frames_raw_dir=frames_raw_dir, progress_cb=progress_cb)
    results = run_pipeline_from_state(state, output_root, params=params, progress_cb=progress_cb)
    if not keep_raw_frames and os.path.isdir(frames_raw_dir):
        shutil.rmtree(frames_raw_dir)
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Water ring alpha extraction pipeline")
    parser.add_argument("video", help="Path to source video")
    parser.add_argument("--out", default=".", help="Output root directory (creates frames/, masks/, output/ inside it)")
    args = parser.parse_args()
    run_pipeline(args.video, args.out)
