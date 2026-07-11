"""Render a standalone SVG to PDF or PNG with headless Microsoft Edge."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_svg", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--edge", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_svg = args.input_svg.resolve()
    output = args.output.resolve()
    edge = args.edge or find_edge()
    svg = input_svg.read_text(encoding="utf-8")
    width, height = svg_dimensions(svg)

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="render_svg_") as temp_dir:
        temp = Path(temp_dir)
        html = temp / "figure.html"
        profile = temp / "edge-profile"
        html.write_text(
            "<!doctype html><html><head><meta charset='utf-8'>"
            f"<style>@page{{size:{width}px {height}px;margin:0}}"
            "html,body{margin:0;padding:0;overflow:hidden;background:white}"
            f"svg{{display:block;width:{width}px;height:{height}px}}</style>"
            f"</head><body>{svg}</body></html>",
            encoding="utf-8",
        )
        common = [
            str(edge),
            "--headless=new",
            "--disable-gpu",
            "--hide-scrollbars",
            f"--user-data-dir={profile}",
        ]
        if output.suffix.lower() == ".pdf":
            command = common + [
                "--no-pdf-header-footer",
                f"--print-to-pdf={output}",
                html.as_uri(),
            ]
        elif output.suffix.lower() == ".png":
            command = common + [
                "--force-device-scale-factor=1",
                f"--window-size={width},{height}",
                f"--screenshot={output}",
                html.as_uri(),
            ]
        else:
            raise ValueError("Output extension must be .pdf or .png")
        subprocess.run(command, check=True, capture_output=True, text=True)

    if not output.exists() or output.stat().st_size == 0:
        raise RuntimeError(f"Renderer did not create {output}")
    print(f"rendered {input_svg} -> {output}")


def find_edge() -> Path:
    executable = shutil.which("msedge")
    if executable:
        return Path(executable)
    for variable in ("PROGRAMFILES(X86)", "PROGRAMFILES"):
        root = os.environ.get(variable)
        if root:
            candidate = Path(root) / "Microsoft" / "Edge" / "Application" / "msedge.exe"
            if candidate.exists():
                return candidate
    raise FileNotFoundError("Microsoft Edge was not found; pass --edge explicitly")


def svg_dimensions(svg: str) -> tuple[int, int]:
    width_match = re.search(r'<svg[^>]*\bwidth="([0-9.]+)"', svg)
    height_match = re.search(r'<svg[^>]*\bheight="([0-9.]+)"', svg)
    if not width_match or not height_match:
        raise ValueError("SVG must declare numeric width and height attributes")
    return round(float(width_match.group(1))), round(float(height_match.group(1)))


if __name__ == "__main__":
    main()
