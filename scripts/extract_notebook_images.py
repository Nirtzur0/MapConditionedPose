#!/usr/bin/env python3
import argparse
import base64
import json
from pathlib import Path


def extract_images(notebook_path, output_dir, prefix, clean, manifest_path):
    output_dir.mkdir(parents=True, exist_ok=True)

    if clean:
        for path in output_dir.glob(f"{prefix}*.png"):
            path.unlink()

    nb = json.loads(notebook_path.read_text())
    images = []
    img_index = 0

    for cell_index, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        for output_index, output in enumerate(cell.get("outputs", [])):
            data = output.get("data", {})
            if not isinstance(data, dict) or "image/png" not in data:
                continue
            png_data = data["image/png"]
            if isinstance(png_data, list):
                png_data = "".join(png_data)
            raw = base64.b64decode(png_data)
            filename = f"{prefix}cell{cell_index:02d}_img{img_index:02d}.png"
            out_path = output_dir / filename
            out_path.write_bytes(raw)
            images.append(
                {
                    "file": str(out_path),
                    "cell_index": cell_index,
                    "output_index": output_index,
                }
            )
            img_index += 1

    if manifest_path:
        manifest_path.write_text(json.dumps(images, indent=2))

    return images


def main():
    parser = argparse.ArgumentParser(
        description="Extract embedded PNG outputs from a Jupyter notebook."
    )
    parser.add_argument(
        "--notebook",
        type=Path,
        default=Path("notebooks/visualization_presentation.ipynb"),
        help="Path to the input notebook.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/figures/visualization_presentation"),
        help="Directory to write extracted PNGs.",
    )
    parser.add_argument(
        "--prefix",
        default="viz_",
        help="Filename prefix for extracted images.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing images with the prefix before writing new ones.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("docs/figures/visualization_presentation/manifest.json"),
        help="Optional path to write a JSON manifest of extracted images.",
    )
    args = parser.parse_args()

    images = extract_images(
        notebook_path=args.notebook,
        output_dir=args.output_dir,
        prefix=args.prefix,
        clean=args.clean,
        manifest_path=args.manifest,
    )

    print(f"Extracted {len(images)} images to {args.output_dir}")


if __name__ == "__main__":
    main()
