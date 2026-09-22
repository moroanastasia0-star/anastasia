"""Interfaccia locale (Gradio) per l'estrazione dell'anello d'acqua con
canale alpha. Avvio: python ui/app.py (o tramite gli script di avvio).

Progettata per un'utente NON programmatrice: upload del video, anteprima
automatica, controlli manuali opzionali, esportazione con un click.
"""
import os
import sys
import glob
import shutil
import traceback

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, SRC_DIR)

import numpy as np
import cv2
import gradio as gr

import main as pipeline
import compositing as comp
import quality_control as qc

PROJECT_ROOT = os.path.dirname(SRC_DIR)
WORKDIR = os.path.join(PROJECT_ROOT, "_ui_workdir")
os.makedirs(WORKDIR, exist_ok=True)

# single-user local tool -> a simple in-process cache is enough, avoids
# forcing large frame lists through Gradio's session (de)serialization.
_STATE = {}


def _bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _bgra_to_rgba(img):
    return cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)


def build_params(mask_threshold, alpha_softness, edge_feather, lower_cutoff,
                  preserve_droplets, bg_removal_strength, ring_offset_y, ring_scale):
    p = dict(pipeline.DEFAULT_PARAMS)
    p["bin_thresh"] = int(mask_threshold)
    p["high_thresh"] = float(alpha_softness)
    p["feather_px"] = int(edge_feather)
    p["hard_cutoff_margin"] = float(lower_cutoff) + (float(ring_offset_y))
    p["proximity_factor"] = float(preserve_droplets) * float(ring_scale)
    p["low_thresh"] = float(bg_removal_strength)
    return p


def analyze_video(video_path):
    if not video_path:
        raise gr.Error("Carica prima un video.")
    frames_raw_dir = os.path.join(WORKDIR, "_frames_raw")
    try:
        state = pipeline.prepare_state(video_path, frames_raw_dir=frames_raw_dir)
    except Exception as e:
        raise gr.Error(f"Errore durante l'analisi del video: {e}")
    _STATE["pipeline"] = state
    info = state["info"]
    n = len(state["frames"])
    mid = n // 2
    preview = _bgr_to_rgb(state["frames"][mid])
    info_text = (
        f"Risoluzione: {info['width']}x{info['height']}\n"
        f"Frame rate: {info['fps']:.3f} fps\n"
        f"Durata: {info['duration']:.3f} s\n"
        f"Numero di frame: {info['nb_frames']}\n"
        f"Codec sorgente: {info['codec']}"
    )
    return info_text, preview, gr.update(maximum=max(n - 1, 0), value=mid, interactive=True)


def preview_frame(frame_idx, mask_threshold, alpha_softness, edge_feather, lower_cutoff,
                   preserve_droplets, bg_removal_strength, ring_offset_y, ring_scale):
    state = _STATE.get("pipeline")
    if state is None:
        raise gr.Error("Analizza prima il video (pulsante 1).")
    n = len(state["frames"])
    idx = int(np.clip(int(frame_idx), 0, n - 1))
    params = build_params(mask_threshold, alpha_softness, edge_feather, lower_cutoff,
                           preserve_droplets, bg_removal_strength, ring_offset_y, ring_scale)
    try:
        out = pipeline.process_frame(state, idx, params)
    except Exception as e:
        raise gr.Error(f"Errore nell'anteprima: {e}")

    original = _bgr_to_rgb(state["frames"][idx])
    alpha_img = np.repeat(out["alpha_water"].astype(np.uint8)[..., None], 3, axis=2)
    rgba = out["rgba_water"]
    black = _bgr_to_rgb(comp.composite_over(rgba, comp.BG_BLACK))
    white = _bgr_to_rgb(comp.composite_over(rgba, comp.BG_WHITE))
    blue = _bgr_to_rgb(comp.composite_over(rgba, comp.BG_ICY_BLUE))

    geom = out["geometry"]
    if geom:
        geom_text = (f"Ring rilevato: centro=({geom['center'][0]:.0f},{geom['center'][1]:.0f}) "
                      f"raggio~{geom['radius']:.0f}px bbox={geom['bbox']}")
    else:
        geom_text = "Nessun anello rilevato in questo frame."

    return original, alpha_img, black, white, blue, geom_text


def run_export(video_path, mask_threshold, alpha_softness, edge_feather, lower_cutoff,
                preserve_droplets, bg_removal_strength, ring_offset_y, ring_scale,
                progress=gr.Progress()):
    state = _STATE.get("pipeline")
    if state is None:
        if not video_path:
            raise gr.Error("Carica e analizza prima un video.")
        frames_raw_dir = os.path.join(WORKDIR, "_frames_raw")
        state = pipeline.prepare_state(video_path, frames_raw_dir=frames_raw_dir)
        _STATE["pipeline"] = state

    params = build_params(mask_threshold, alpha_softness, edge_feather, lower_cutoff,
                           preserve_droplets, bg_removal_strength, ring_offset_y, ring_scale)

    output_root = os.path.join(WORKDIR, "export")
    if os.path.isdir(output_root):
        shutil.rmtree(output_root)

    log_lines = []

    def cb(msg):
        log_lines.append(msg)
        progress(0, desc=msg[:80])

    try:
        results = pipeline.run_pipeline_from_state(state, output_root, params=params, progress_cb=cb)
    except Exception as e:
        tb = traceback.format_exc()
        raise gr.Error(f"Errore durante l'esportazione: {e}\n{tb[-1500:]}")

    files = []
    for key in ("webm", "mov", "webm_with_text", "preview_black", "preview_white",
                "preview_blue", "mask_preview"):
        val = results.get(key)
        if val and os.path.isfile(val):
            files.append(val)

    frames_zip = shutil.make_archive(os.path.join(output_root, "water_ring_frames"),
                                      "zip", results["frames_dir"])
    files.append(frames_zip)

    full_log = "\n".join(log_lines)
    status = "COMPLETATO (controllo qualita: PASS)" if results["qc_pass"] else \
             "COMPLETATO (controllo qualita: alcuni controlli da rivedere, vedi report)"
    report = f"{status}\n\nTempo totale: {results['elapsed_s']:.1f}s\n\n{results['qc_report']}"

    return report, files, results.get("preview_black")


CSS = """
#title {text-align: center;}
"""

with gr.Blocks(title="Water Ring Alpha Extractor") as demo:
    gr.Markdown("# Water Ring Alpha Extractor\nEstrazione dell'anello d'acqua con canale alpha da un video originale (nessuna generazione AI: solo segmentazione/matting reale).", elem_id="title")

    with gr.Row():
        video_input = gr.Video(label="1. Carica il video originale")
        with gr.Column():
            analyze_btn = gr.Button("1. Analizza Video", variant="primary")
            info_box = gr.Textbox(label="Informazioni tecniche", lines=6, interactive=False)

    preview_frame_img = gr.Image(label="Frame originale", interactive=False)

    gr.Markdown("## 2. Anteprima automatica (Auto Detect Ring)")
    frame_slider = gr.Slider(label="Frame da visualizzare", minimum=0, maximum=1, step=1, value=0, interactive=False)
    preview_btn = gr.Button("Genera anteprima su questo frame", variant="primary")
    geom_text = gr.Textbox(label="Geometria anello rilevata", interactive=False)

    with gr.Row():
        out_original = gr.Image(label="Originale", interactive=False)
        out_mask = gr.Image(label="Maschera Alpha", interactive=False)
    with gr.Row():
        out_black = gr.Image(label="Su sfondo nero", interactive=False)
        out_white = gr.Image(label="Su sfondo bianco", interactive=False)
        out_blue = gr.Image(label="Su sfondo azzurro (icy blue)", interactive=False)

    with gr.Accordion("Controlli manuali avanzati (usa solo se l'automatismo non e' soddisfacente)", open=False):
        gr.Markdown(
            "Questi controlli modificano l'algoritmo automatico. I valori di default funzionano bene "
            "nella maggior parte dei casi: modifica solo se noti problemi nell'anteprima."
        )
        mask_threshold = gr.Slider(10, 150, value=50, step=1, label="Mask threshold (sensibilita' rilevamento acqua)")
        alpha_softness = gr.Slider(10, 90, value=32, step=1, label="Alpha softness (morbidezza delle trasparenze)")
        edge_feather = gr.Slider(0, 25, value=9, step=1, label="Edge feather (sfumatura dei bordi)")
        lower_cutoff = gr.Slider(0, 80, value=22, step=1, label="Lower surface cutoff (margine di taglio della superficie inferiore)")
        preserve_droplets = gr.Slider(1.0, 3.0, value=1.9, step=0.05, label="Preserve droplets (quanto includere le goccioline vicine)")
        bg_removal_strength = gr.Slider(2, 20, value=6, step=0.5, label="Background removal strength (forza rimozione sfondo)")
        ring_offset_y = gr.Slider(-60, 60, value=0, step=1, label="Ring position Y (nudge verticale del taglio inferiore)")
        ring_scale = gr.Slider(0.7, 1.5, value=1.0, step=0.05, label="Ring scale (tolleranza raggio per goccioline/schizzi)")

    gr.Markdown("## 3. Esportazione completa")
    export_btn = gr.Button("Esporta tutti i file (PNG, WebM, ProRes, preview, maschera)", variant="primary")
    export_report = gr.Textbox(label="Report finale / Controllo qualita", lines=16, interactive=False)
    export_files = gr.Files(label="File esportati (clic per scaricare)")
    export_preview_video = gr.Video(label="Anteprima risultato (sfondo nero)")

    params_list = [mask_threshold, alpha_softness, edge_feather, lower_cutoff,
                   preserve_droplets, bg_removal_strength, ring_offset_y, ring_scale]

    analyze_btn.click(analyze_video, inputs=[video_input],
                       outputs=[info_box, preview_frame_img, frame_slider])
    preview_btn.click(preview_frame, inputs=[frame_slider] + params_list,
                       outputs=[out_original, out_mask, out_black, out_white, out_blue, geom_text])
    export_btn.click(run_export, inputs=[video_input] + params_list,
                      outputs=[export_report, export_files, export_preview_video])


if __name__ == "__main__":
    demo.queue().launch(inbrowser=True, show_error=True, css=CSS)
