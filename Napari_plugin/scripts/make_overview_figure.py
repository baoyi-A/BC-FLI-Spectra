# -*- coding: utf-8 -*-
"""One figure showing the whole SLIC / BC-FLIM-Spectra workflow on real data.

Every panel is a real output of the plugin on one acquisition
(J:/Mix16-N-P-260306-DCZ-2-1), not a schematic. Read-only on the source.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from matplotlib.lines import Line2D
import tifffile
from PIL import Image

D = r'J:/Mix16-N-P-260306-DCZ-2-1'
OUT = r'C:/Users/admin/AppData/Local/Temp/claude/D--PKU-STUDY-DeepLearining-BC-FLIM-python-code/4e06c3bb-2073-4cf2-94d5-33068a861186/scratchpad/pipeline_figure.png'

INK = '#12181F'
MUTED = '#6A7484'
LINE = '#D8DEE6'
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 8.5,
    'axes.edgecolor': LINE,
    'axes.labelcolor': INK,
    'text.color': INK,
    'xtick.color': MUTED,
    'ytick.color': MUTED,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})


def crop_ar(a, ar=4 / 3, zoom=1.0):
    """Centre-crop to an aspect ratio, optionally zooming in first."""
    h, w = a.shape[:2]
    if zoom > 1:
        ch, cw = int(h / zoom), int(w / zoom)
        y0, x0 = (h - ch) // 2, (w - cw) // 2
        a = a[y0:y0 + ch, x0:x0 + cw]
        h, w = a.shape[:2]
    if w / h > ar:
        nw = int(h * ar); x0 = (w - nw) // 2
        return a[:, x0:x0 + nw]
    nh = int(w / ar); y0 = (h - nh) // 2
    return a[y0:y0 + nh]


def load_rgb(path, side=900):
    im = Image.open(path).convert('RGB')
    s = side / max(im.size)
    if s < 1:
        im = im.resize((int(im.size[0] * s), int(im.size[1] * s)), Image.LANCZOS)
    return np.asarray(im)


def load_tif_rgb(path, side=900):
    a = np.asarray(tifffile.imread(path))
    if a.ndim == 2:
        a = np.stack([a] * 3, -1)
    if a.dtype != np.uint8:
        a = (255 * (a.astype(float) / max(a.max(), 1))).astype(np.uint8)
    im = Image.fromarray(a[..., :3])
    s = side / max(im.size)
    if s < 1:
        im = im.resize((int(im.size[0] * s), int(im.size[1] * s)), Image.LANCZOS)
    return np.asarray(im)


def load_mask(p):
    raw = np.load(p, allow_pickle=True)
    if isinstance(raw, np.ndarray) and raw.dtype == object and raw.shape == ():
        raw = raw.item()
    if isinstance(raw, dict):
        raw = raw.get('masks', raw)
    return np.asarray(raw)


# ------------------------------------------------------------------ data
render = load_rgb(os.path.join(D, 'TileScan_001_s1_fastflim_rgb.png'))
cls_color = load_tif_rgb(os.path.join(D, 'intensity', 'TileScan_001_s1-cls-color.tif'))
bgy = load_tif_rgb(os.path.join(D, 'FOV-1_bgy_render_rgb.tif'), side=760)
bio_mask = load_mask(os.path.join(D, 'FOV-1_seg_image_seg.npy'))
dc = pd.read_excel(os.path.join(D, 'clustered.xlsx'))
sig = pd.read_excel(os.path.join(D, 'signal_analysis.xlsx'), sheet_name='Normalized G-B')

# barcode class colours: the plugin's own map, gray at index 0
BASE = ['#808080', '#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4',
        '#42d4f4', '#f032e6', '#469990', '#9A6324', '#808000', '#000075',
        '#800000', '#aaffc3']

fig = plt.figure(figsize=(13.2, 7.4), dpi=190)
gs = fig.add_gridspec(2, 3, hspace=0.34, wspace=0.16,
                      left=0.030, right=0.984, top=0.855, bottom=0.075)


def panel(ax, letter, title, sub):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(LINE)
    ax.set_title(f'{letter}   {title}', loc='left', fontsize=11, fontweight='600',
                 pad=19, color=INK)
    ax.text(0, 1.018, sub, transform=ax.transAxes, fontsize=8.2, color=MUTED,
            va='bottom', ha='left')


# (a) one FLIM acquisition, rendered by lifetime
ax = fig.add_subplot(gs[0, 0]); ax.imshow(crop_ar(render))
panel(ax, 'a', 'One FLIM acquisition', 'lifetime as colour, four spectral channels')

# (b) segmentation
ax = fig.add_subplot(gs[0, 1])
n_mask = load_mask(os.path.join(D, 'intensity', 'TileScan_001_s1_sum_seg_n.npy'))
p_mask = load_mask(os.path.join(D, 'intensity', 'TileScan_001_s1_sum_seg_p.npy'))
big = load_rgb(os.path.join(D, 'TileScan_001_s1_fastflim_rgb.png'), side=2048)
zb = crop_ar(big, zoom=3.2)
def _edges(m, shape):
    m = np.asarray(Image.fromarray(m.astype(np.int32), mode='I').resize(
        (shape[1], shape[0]), Image.NEAREST))
    e = np.zeros(m.shape, bool)
    e[:-1] |= m[:-1] != m[1:]
    e[:, :-1] |= m[:, :-1] != m[:, 1:]
    return e
zn = crop_ar(_edges(n_mask, big.shape), zoom=3.2)
zp = crop_ar(_edges(p_mask, big.shape), zoom=3.2)
img = zb.copy()
img[zp] = (255, 214, 90)
img[zn] = (255, 255, 255)
ax.imshow(img)
panel(ax, 'b', 'Cells segmented', 'nucleus white, cytoplasm gold  ·  detail view')

# (c) the 5-D fingerprint, shown as the phasor plane
ax = fig.add_subplot(gs[0, 2])
ax.add_patch(Arc((0.5, 0), 1, 1, theta1=0, theta2=180, color=LINE, lw=1.1))
tag = dc['cluster_tag'].fillna('').astype(str)
nrows = dc[dc['Localization'].astype(str).str.upper() == 'N']
for k in range(1, 11):
    sel = nrows[nrows['cluster_local'] == k]
    if len(sel):
        ax.scatter(sel['G'], sel['S'], s=7, c=BASE[k], linewidths=0, alpha=.9)
out = nrows[nrows['cluster_local'] == 0]
ax.scatter(out['G'], out['S'], s=6, c='#B8BEC8', linewidths=0, alpha=.7)
ax.set_xlim(0.15, 0.72); ax.set_ylim(0.40, 0.53)
ax.set_xlabel('G', fontsize=8.5); ax.set_ylabel('S', fontsize=8.5)
ax.set_xticks([0.2, 0.4, 0.6]); ax.set_yticks([0.42, 0.46, 0.50])
ax.tick_params(labelsize=7.5)
for sp in ax.spines.values():
    sp.set_color(LINE)
ax.set_title('c   Every cell becomes five numbers', loc='left', fontsize=11,
             fontweight='600', pad=19, color=INK)
ax.text(0, 1.018, 'phasor G, S plus three spectral ratios', transform=ax.transAxes,
        fontsize=8.2, color=MUTED, va='bottom', ha='left')

# (d) barcode identity per cell
ax = fig.add_subplot(gs[1, 0]); ax.imshow(crop_ar(cls_color))
panel(ax, 'd', 'Each cell gets its barcode', '10 barcodes separated in one dish')

# (e) registration onto the biosensor channel
ax = fig.add_subplot(gs[1, 1])
ax.imshow(bgy)
cls_full = np.asarray(tifffile.imread(os.path.join(D, 'intensity', 'TileScan_001_s1-cls.tif')))
if cls_full.ndim > 2:
    cls_full = np.squeeze(cls_full)
al = np.rot90(cls_full, k=3)
al = np.asarray(Image.fromarray(al.astype(np.uint16)).resize(
    (bio_mask.shape[1], bio_mask.shape[0]), Image.NEAREST))
al_s = np.asarray(Image.fromarray(al.astype(np.uint16)).resize(
    (bgy.shape[1], bgy.shape[0]), Image.NEAREST))
rgba = np.zeros((*al_s.shape, 4))
for k in range(1, 15):
    m = al_s == k
    if m.any():
        c = matplotlib.colors.to_rgb(BASE[k])
        rgba[m] = (*c, 0.48)
ax.clear()
zb2 = crop_ar(bgy, zoom=2.6)
zr = crop_ar(rgba, zoom=2.6)
ax.imshow(zb2); ax.imshow(zr)
panel(ax, 'e', 'Aligned to the biosensor channel',
      'FLIM 2048 px onto confocal 1024 px  ·  99% pure  ·  detail view')

# (f) per-class biosensor response
ax = fig.add_subplot(gs[1, 2])
frames = [c for c in sig.columns if isinstance(c, (int, np.integer))]
for k in range(1, 11):
    sub = sig[sig['Class'] == k]
    if len(sub) < 4:
        continue
    y = sub[frames].to_numpy(float).mean(0)
    ax.plot(frames, y, color=BASE[k], lw=1.35, alpha=.95)
ax.axvline(24, color=MUTED, lw=.8, ls=(0, (3, 3)))
ax.text(25.5, ax.get_ylim()[1], 'stimulus', fontsize=7.5, color=MUTED,
        va='top', ha='left')
ax.set_xlabel('frame', fontsize=8.5); ax.set_ylabel('normalised G−B', fontsize=8.5)
ax.tick_params(labelsize=7.5)
for sp in ax.spines.values():
    sp.set_color(LINE)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.set_title('f   One readout per barcode', loc='left', fontsize=11,
             fontweight='600', pad=19, color=INK)
ax.text(0, 1.018, 'each line is one barcode population, same dish',
        transform=ax.transAxes, fontsize=8.2, color=MUTED, va='bottom', ha='left')

fig.text(0.028, 0.955, 'SLIC  ·  one mixed dish, ten barcoded populations, one readout each',
         fontsize=15, fontweight='600', color=INK)
fig.text(0.028, 0.915,
         'Every panel is real output of the napari plugin on a single acquisition — '
         'no schematic, no simulation.',
         fontsize=9.5, color=MUTED)

fig.savefig(OUT, dpi=190, facecolor='white', bbox_inches='tight', pad_inches=0.22)
print('wrote', OUT)
im = Image.open(OUT)
print('size', im.size)
im.resize((1400, int(1400 * im.size[1] / im.size[0])), Image.LANCZOS).save(
    OUT.replace('.png', '_small.png'))
print('preview', OUT.replace('.png', '_small.png'))
