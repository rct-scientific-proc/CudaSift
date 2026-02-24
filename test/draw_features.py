"""Draw SIFT features on an image and save the result.

Usage:
    python draw_features.py <image> <features.json> [output.png]

If no output path is given, saves to <image_stem>_features.png next to the input.
"""

import argparse
import json
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw


def load_features(json_path: str) -> list[dict]:
    with open(json_path, "r") as f:
        data = json.load(f)
    return data["keypoints"]


def draw_features(
    img: Image.Image,
    keypoints: list[dict],
    *,
    radius_scale: float = 3.0,
    color: str = "lime",
    orientation_color: str = "red",
    line_width: int = 1,
) -> Image.Image:
    out = img.convert("RGB")
    draw = ImageDraw.Draw(out)

    for kp in keypoints:
        x = kp["x"]
        y = kp["y"]
        s = kp["scale"] * radius_scale
        orient = kp.get("orientation", 0.0)

        # Circle at keypoint location
        bbox = [x - s, y - s, x + s, y + s]
        draw.ellipse(bbox, outline=color, width=line_width)

        # Orientation tick
        dx = s * math.cos(orient)
        dy = s * math.sin(orient)
        draw.line([(x, y), (x + dx, y + dy)], fill=orientation_color, width=line_width)

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw SIFT features on an image.")
    parser.add_argument("image", help="Path to the input image")
    parser.add_argument("features", help="Path to the JSON features file")
    parser.add_argument("output", nargs="?", default=None, help="Output image path (default: <image>_features.png)")
    parser.add_argument("--scale", type=int, default=1, help="Upscale image by this factor before drawing (default: 1)")
    args = parser.parse_args()

    img = Image.open(args.image)
    if args.scale > 1:
        img = img.resize((img.width * args.scale, img.height * args.scale), Image.NEAREST)
    keypoints = load_features(args.features)

    result = draw_features(img, keypoints)

    if args.output:
        out_path = args.output
    else:
        p = Path(args.image)
        out_path = str(p.with_stem(p.stem + "_features").with_suffix(".png"))

    result.save(out_path)
    print(f"Saved {len(keypoints)} features to {out_path}")


if __name__ == "__main__":
    main()
