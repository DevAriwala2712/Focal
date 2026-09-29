"""Offline TrustSR Wayanad showcase. Run: streamlit run demo/app.py"""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import numpy as np
import rasterio
import streamlit as st
from PIL import Image
from risk.common import load_config

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / 'demo' / 'assets'


@st.cache_data
def read_rgb(path: str, stretch_max: float, gamma: float) -> Image.Image:
    with rasterio.open(path) as src:
        rgb = src.read([1, 2, 3]).transpose(1, 2, 0)
    display = np.clip(rgb / stretch_max, 0, 1) ** gamma
    return Image.fromarray((display * 255).astype('uint8'), 'RGB')


@st.cache_data
def read_classes(path: str) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1)


def annotated(image: Image.Image, classes: np.ndarray, resolution: str, overlay: bool) -> Image.Image:
    if not overlay:
        return image
    if resolution == '10 m':
        if classes.shape != (image.height*4, image.width*4):
            raise ValueError('10 m and 2.5 m products are misaligned')
        blocks = classes.reshape(image.height, 4, image.width, 4).transpose(0, 2, 1, 3)
        coarse = np.full((image.height, image.width), 1, dtype=np.uint8)
        coarse[np.any(blocks == 0, axis=(2, 3))] = 0
        coarse[np.any((blocks == 2) | (blocks == 3), axis=(2, 3))] = 2
        classes = coarse
    if classes.shape != (image.height, image.width):
        raise ValueError('Overlay and imagery grids differ')
    tint = np.zeros((image.height, image.width, 4), dtype=np.uint8)
    tint[classes == 0] = (143, 152, 168, 175)
    tint[classes == 2] = (235, 74, 72, 160)
    tint[classes == 3] = (255, 184, 69, 145)
    return Image.alpha_composite(image.convert('RGBA'), Image.fromarray(tint, 'RGBA')).convert('RGB')


def png_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format='PNG', optimize=True)
    return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')


def swipe_html(before: Image.Image, after: Image.Image, *, pre_date: str, post_date: str) -> str:
    pre_url, post_url = png_url(before), png_url(after)
    return f'''<!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{{margin:0;background:#0a1523;color:#f5f7fa;font:14px system-ui}}
.frame{{position:relative;width:min(100%,600px);aspect-ratio:1;margin:auto;overflow:hidden;border:1px solid #5e7086;border-radius:10px}}
.frame img{{position:absolute;inset:0;width:100%;height:100%;object-fit:fill}}
#post{{clip-path:inset(0 50% 0 0)}}#line{{position:absolute;left:50%;top:0;bottom:0;width:2px;background:white;box-shadow:0 0 8px #001}}
.tag{{position:absolute;top:12px;background:#0a1523df;padding:6px 10px;border-radius:5px;font-weight:700}}
.left{{left:12px}}.right{{right:12px}}
input{{display:block;width:min(100%,600px);margin:18px auto;accent-color:#f5b85d}}
.hint{{text-align:center;color:#bac8d8}}</style></head><body>
<div class="frame"><img alt="Before satellite view" src="{pre_url}"><img id="post" alt="After satellite view" src="{post_url}"><div id="line"></div>
<span class="tag left">Before · {pre_date}</span><span class="tag right">After · {post_date}</span></div>
<input aria-label="Move before and after swipe" type="range" min="0" max="100" value="50" oninput="document.getElementById('post').style.clipPath='inset(0 '+(100-this.value)+'% 0 0)';document.getElementById('line').style.left=this.value+'%'">
<div class="hint">Drag the slider to compare the same georeferenced footprint.</div></body></html>'''


def main():
    st.set_page_config(page_title='TrustSR · Wayanad', page_icon='🛰️', layout='wide')
    st.html('''<style>.stApp{background:#0b1522;color:#eef2f7}.block-container{max-width:1220px;padding-top:2rem}
    h1,h2,h3{color:#f5f7fa}.stMetric{background:#172739;padding:14px;border-radius:12px}</style>''')
    st.title('TrustSR · Wayanad')
    cfg, _, _ = load_config(ROOT/'configs/pipeline.yaml')
    event_date = cfg['imagery']['event_date']
    st.caption(f'Sentinel-2 vegetation-change showcase · {event_date} event · {cfg["demo"]["tile_pixels"] * 10 / 1000:g} km central tile')
    manifest_path = Path(st.session_state.get('live_manifest', ASSETS/'manifest.json'))
    if not manifest_path.exists():
        st.error('Precomputed Wayanad demo assets are missing. Run the fetch and demo-tile scripts in the README.')
        return
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    evaluation_path = ROOT/'results/metrics.json'
    evaluation = json.loads(evaluation_path.read_text(encoding='utf-8')) if evaluation_path.is_file() else None
    expected = np.array([cfg['imagery']['longitude'], cfg['imagery']['latitude'], cfg['imagery']['aoi_size_m']])
    if manifest.get('source_aoi') is None or not np.allclose(manifest['source_aoi'], expected, rtol=0, atol=1e-6):
        st.error('The bundled tile was generated from the supplied coordinates, which NRSC places away from the landslide. Corrected event-site imagery is being rebuilt; this tile is hidden.')
        return
    files = {key: manifest_path.parent/Path(name).name for key, name in manifest['files'].items()}
    if any(not path.is_file() for path in files.values()):
        st.error('One or more precomputed COGs are missing from demo/assets.')
        return
    if st.sidebar.button('Run live on GPU', help='Reprocess the central tile from the full local stack'):
        if not (ROOT/cfg['fetch']['stack']).exists():
            st.sidebar.error('Full source stack is unavailable. Reproduce it with the fetch script first.')
        else:
            with st.spinner('Running model and trust gate on one tile…'):
                from scripts.run_demo_tile import main as run_live
                run_live()
            st.session_state['live_manifest'] = str(ROOT/'data/wayanad/demo/manifest.json')
            st.rerun()
    st.sidebar.header('View')
    resolution = st.sidebar.radio('Resolution', ['10 m', '2.5 m'], index=1, horizontal=True)
    overlay = st.sidebar.toggle('Show trust-gated change', value=True)
    suffix = '10m' if resolution == '10 m' else '2p5m'
    before = read_rgb(str(files[f'pre_{suffix}']), cfg['demo']['display_max_reflectance'], cfg['demo']['display_gamma'])
    after = read_rgb(str(files[f'post_{suffix}']), cfg['demo']['display_max_reflectance'], cfg['demo']['display_gamma'])
    classes = read_classes(str(files['change']))
    before = annotated(before, classes, resolution, overlay)
    after = annotated(after, classes, resolution, overlay)
    left, right = st.columns([1.7, 1], gap='large')
    with left:
        st.subheader('Before / after')
        st.iframe(swipe_html(before, after, pre_date=f"{manifest['pre_dates'][0]}–{manifest['pre_dates'][-1]}",
                             post_date=manifest['post_date']), height=680)
        st.caption(f"Pre view pools {', '.join(manifest['pre_dates'])}. Post view is {manifest['post_date']}, {manifest['post_event_lag_days']} days after the event.")
    with right:
        st.subheader('Change confidence')
        m = manifest['metrics']
        c1, c2 = st.columns(2)
        c1.metric('Observed support', f"{m['observed_pixels']:,} px")
        c2.metric('Inferred position', f"{m['inferred_pixels']:,} px")
        c1.metric('SR-only rejected', f"{m['unsupported_pixels']:,} px")
        c2.metric('Masked / no data', f"{m['no_data_pixels']:,} px")
        st.markdown('🔴 **Observed support** — 10 m parent also shows NDVI drop. Fine boundary remains model-inferred.  \n'
                    '🟠 **Inferred position** — changed parent, but fine-pixel signal is uncertain.  \n'
                    '⬜ **No data** — cloud, shadow, or invalid in any selected date.')
        st.info(f"This is vegetation disturbance, not verified landslide damage. The first clear post image is {manifest['post_event_lag_days']} days after the event.")
        if evaluation and evaluation.get('downstream_f1_trust') is not None:
            st.warning(f"Independent event test F1: {evaluation['downstream_f1_trust']:.3f} at the 10 m label scale. This low score means the current detector is not reliable for damage decisions. Wayanad itself has no matching ground-truth mask.")
        st.caption('A published study reported 507 buildings and 8.38 km of road impacted. Those figures use a different sensor/method and are context only, not TrustSR detections.')
    with st.expander('Provenance and evaluation'):
        st.json({'pre_dates': manifest['pre_dates'], 'post_date': manifest['post_date'],
                 'model': 'SEN2SR-lite RGBN x4, 50-step local fine-tune',
                 'checkpoint_sha256': manifest.get('checkpoint_sha256'),
                 'crs': manifest['grid_crs'], 'k': manifest['k'],
                 'parent_ndvi_drop_threshold': manifest['parent_drop_threshold']})
        if evaluation:
            st.json({'held_out_worldstrat': evaluation.get('held_out_worldstrat'),
                     'event_held_out_f1_10m': evaluation.get('downstream_f1_10m'),
                     'event_held_out_f1_raw_sr': evaluation.get('downstream_f1_2p5m'),
                     'event_held_out_f1_trust': evaluation.get('downstream_f1_trust'),
                     'calibration': evaluation.get('calibration')})
        st.warning('Independent 2.5 m landslide boundary truth is unavailable. The event test scores use 10 m masks and do not measure Wayanad accuracy.')


if __name__ == '__main__':
    main()
