#!/usr/bin/env python3
"""Assemble training snapshot grids into animated GIFs.

Usage:
    python3 make_gif.py                  # process both output/ddpm and output/flow
    python3 make_gif.py --dir output/ddpm
    python3 make_gif.py --dir output/flow
"""

import argparse
import glob
import os
import re

from PIL import Image, ImageDraw, ImageFont


def load_font(size=20):
    for path in [
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/SFMono-Regular.otf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    ]:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


def make_gif(src_dir, out_path, duration_ms=200):
    pattern = os.path.join(src_dir, "grid_step_*.png")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No grid_step_*.png found in {src_dir}")
        return

    frames = []
    font = load_font(22)

    for f in files:
        match = re.search(r"grid_step_(\d+)\.png", f)
        if not match:
            continue
        step = int(match.group(1))
        label = f"Step {step:,}"

        img = Image.open(f).convert("RGB")
        draw = ImageDraw.Draw(img)

        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        pad = 6
        draw.rectangle([4, 4, 4 + tw + 2 * pad, 4 + th + 2 * pad], fill=(0, 0, 0))
        draw.text((4 + pad, 4 + pad), label, fill=(255, 255, 255), font=font)

        frames.append(img.quantize(colors=256, method=Image.Quantize.MEDIANCUT))

    if not frames:
        print(f"No valid frames in {src_dir}")
        return

    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    print(f"Wrote {out_path} ({len(frames)} frames, {duration_ms}ms each)")


def main():
    parser = argparse.ArgumentParser(description="Build training GIFs")
    parser.add_argument("--dir", help="Single directory to process")
    parser.add_argument("--duration", type=int, default=200, help="ms per frame")
    args = parser.parse_args()

    dirs = [args.dir] if args.dir else ["output/ddpm", "output/flow"]

    for d in dirs:
        name = os.path.basename(d)
        out = os.path.join("output", f"{name}_training.gif")
        make_gif(d, out, args.duration)


if __name__ == "__main__":
    main()
