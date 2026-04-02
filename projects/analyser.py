#diffraction.py
#pip install pillow matplotlib numpy

import sys
import os
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


#actual part
def load_center_row(path):
    img = Image.open(path).convert("RGB")
    arr = np.array(img)
    row = arr[arr.shape[0] // 2]   

    #downsample
    step = max(1, len(row) // 500)
    row = row[::step]
    xs = np.arange(len(row)) * step

    r, g, b = row[:,0].astype(float), row[:,1].astype(float), row[:,2].astype(float)
    brightness = (r + g + b) / 3
    sat = np.abs(r-g) + np.abs(r-b) + np.abs(g-b)

    is_white = (brightness > 200) & (sat < 50)
    intensity = np.where(is_white, brightness + 100, r)

    return xs, intensity


def normalise(intensity):
    lo, hi = intensity.min(), intensity.max()
    if hi == lo:
        return np.full_like(intensity, 200.0)
    norm = (intensity - lo) / (hi - lo)
    #scale
    return 50 + norm * 250


def get_peaks(profile, xs):
    threshold = profile.max() * 0.3
    px, py = [], []
    for i in range(1, len(profile)-1):
        if profile[i] > profile[i-1] and profile[i] > profile[i+1] and profile[i] > threshold:
            px.append(xs[i])
            py.append(profile[i])
    return np.array(px), np.array(py)


def analyse(path):
    xs, raw = load_center_row(path)
    profile = normalise(raw)
    px, py = get_peaks(profile, xs)
    return xs, profile, raw.mean(), px, py


# plot
def make_plot(results, save_path=None):
    n = len(results)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4*rows))
    fig.patch.set_facecolor('#1a1a1a')

    #iterable
    if n == 1:
        axes = [axes]
    elif rows == 1:
        axes = list(axes)
    else:
        axes = [ax for row in axes for ax in row]

    for i, (name, xs, profile, raw_avg, px, py) in enumerate(results):
        ax = axes[i]
        ax.set_facecolor('#111111')

        ax.fill_between(xs, profile, alpha=0.2, color='#ff3333')
        ax.plot(xs, profile, color='#ff4444', linewidth=1.2, label='intensity')

        if len(px):
            ax.scatter(px, py, color='yellow', zorder=5, s=25, label=f'{len(px)} peaks')

        ax.set_ylim(0, 350)
        ax.set_xlabel('pixel position', color='#aaaaaa', fontsize=8)
        ax.set_ylabel('intensity', color='#aaaaaa', fontsize=8)
        ax.set_title(name, color='#dddddd', fontsize=9, pad=6)
        ax.tick_params(colors='#666666', labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor('#333333')
        ax.grid(color='#222222', linewidth=0.5)
        ax.legend(fontsize=7, facecolor='#1a1a1a', labelcolor='#aaaaaa', edgecolor='#333333')

        stats_str = (
            f"max {profile.max():.0f}  min {profile.min():.0f}  "
            f"avg {profile.mean():.0f}  raw avg {raw_avg:.0f}\n"
            f"contrast {(profile.max()-profile.min())/profile.max()*100:.1f}%"
        )
        ax.text(0.02, 0.97, stats_str, transform=ax.transAxes,
                fontsize=6.5, va='top', color='#888888',
                bbox=dict(facecolor='#1a1a1a', edgecolor='#333333', pad=3, alpha=0.8))

    #hide subs
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#1a1a1a')
        print(f"saved to {save_path}")
    else:
        plt.show()


#files
def get_images(inputs):
    exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    paths = []
    for item in inputs:
        if os.path.isdir(item):
            for f in sorted(os.listdir(item)):
                if os.path.splitext(f)[1].lower() in exts:
                    paths.append(os.path.join(item, f))
        elif os.path.isfile(item):
            paths.append(item)
        else:
            print(f"can't find: {item}")
    return paths


#main
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='laser diffraction analyser')
    parser.add_argument('inputs', nargs='+', help='image file(s) or folder')
    parser.add_argument('--save', default=None, help='save chart to file')
    args = parser.parse_args()

    files = get_images(args.inputs)
    if not files:
        print("no images found")
        sys.exit(1)

    results = []
    for path in files:
        name = os.path.basename(path)
        print(f"  {name}...", end=' ')
        try:
            xs, profile, raw_avg, px, py = analyse(path)
            print(f"ok  ({len(px)} peaks, contrast {(profile.max()-profile.min())/profile.max()*100:.1f}%)")
            results.append((name, xs, profile, raw_avg, px, py))
        except Exception as e:
            print(f"failed: {e}")

    if results:
        make_plot(results, save_path=args.save)
