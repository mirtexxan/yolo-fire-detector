"""Remove image background and save a transparent PNG."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def sample_border_key_color(rgb: np.ndarray) -> np.ndarray:
    """Estimate the background color from image borders."""
    h, w = rgb.shape[:2]
    border = max(1, min(h, w) // 20)
    strips = [
        rgb[:border, :, :].reshape(-1, 3),
        rgb[-border:, :, :].reshape(-1, 3),
        rgb[:, :border, :].reshape(-1, 3),
        rgb[:, -border:, :].reshape(-1, 3),
    ]
    samples = np.concatenate(strips, axis=0)
    return np.median(samples, axis=0).astype(np.float32)


def remove_with_chroma_key(
    input_path: Path,
    output_path: Path,
    low_threshold: float = 18.0,
    high_threshold: float = 110.0,
) -> None:
    """Remove a near-solid green-screen style background and suppress green spill."""
    rgb = np.array(Image.open(input_path).convert("RGB"), dtype=np.uint8)
    key_color = sample_border_key_color(rgb)

    rgb_float = rgb.astype(np.float32)
    distance = np.linalg.norm(rgb_float - key_color[None, None, :], axis=2)

    alpha = np.clip((distance - low_threshold) / max(1.0, high_threshold - low_threshold), 0.0, 1.0)

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0].astype(np.float32)
    saturation = hsv[:, :, 1].astype(np.float32)
    value = hsv[:, :, 2].astype(np.float32)
    greenish_mask = (hue >= 35) & (hue <= 95) & (saturation >= 35) & (value >= 20)
    green_dominant_mask = greenish_mask & (rgb_float[:, :, 1] >= rgb_float[:, :, 0] + 12.0) & (
        rgb_float[:, :, 1] >= rgb_float[:, :, 2] + 12.0
    )
    greenish = greenish_mask.astype(np.float32)

    # Push likely green spill further toward transparency while keeping clearly non-green pixels intact.
    alpha *= 1.0 - 0.85 * greenish * (1.0 - alpha)
    alpha[green_dominant_mask] = 0.0
    alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=1.0)
    alpha = np.clip(alpha, 0.0, 1.0)

    foreground = rgb_float.copy()
    red_blue_max = np.maximum(foreground[:, :, 0], foreground[:, :, 2])
    spill_strength = greenish * alpha
    foreground[:, :, 1] = foreground[:, :, 1] * (1.0 - spill_strength) + red_blue_max * spill_strength
    foreground[:, :, 1] = np.where(
        greenish_mask,
        np.minimum(foreground[:, :, 1], red_blue_max + 6.0),
        foreground[:, :, 1],
    )
    foreground[alpha < 0.08] = 0.0

    rgba = np.dstack((np.clip(foreground, 0, 255).astype(np.uint8), (alpha * 255.0).astype(np.uint8)))
    Image.fromarray(rgba, mode="RGBA").save(output_path, format="PNG")


def remove_with_grabcut(input_path: Path, output_path: Path) -> None:
    """Fallback method using OpenCV GrabCut."""
    bgr = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Unable to read image: {input_path}")

    h, w = bgr.shape[:2]
    margin_x = max(8, int(w * 0.04))
    margin_y = max(8, int(h * 0.04))

    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    rect = (margin_x, margin_y, max(1, w - 2 * margin_x), max(1, h - 2 * margin_y))

    cv2.grabCut(bgr, mask, rect, bgd_model, fgd_model, 8, cv2.GC_INIT_WITH_RECT)
    fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    fg_mask = cv2.GaussianBlur(fg_mask, (0, 0), sigmaX=1.2)

    rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = fg_mask
    cv2.imwrite(str(output_path), rgba)


def remove_with_rembg(input_path: Path, output_path: Path, model: str = "isnet-general-use") -> None:
    """Preferred method based on rembg and ONNX segmentation."""
    from rembg import new_session, remove

    with input_path.open("rb") as in_file:
        data = in_file.read()

    session = new_session(model)
    result: object = remove(data, session=session)

    if isinstance(result, (bytes, bytearray, memoryview)):
        output_path.write_bytes(result)
        return

    if isinstance(result, Image.Image):
        result.save(output_path, format="PNG")
        return

    if isinstance(result, np.ndarray):
        Image.fromarray(result).save(output_path, format="PNG")
        return

    raise TypeError(f"Unsupported rembg output type: {type(result)!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rimuove lo sfondo e salva PNG con alpha")
    parser.add_argument("--input", required=True, help="Path immagine input")
    parser.add_argument("--output", required=True, help="Path output PNG")
    parser.add_argument(
        "--method",
        choices=["auto", "rembg", "grabcut", "chroma"],
        default="auto",
        help="Metodo di rimozione sfondo",
    )
    parser.add_argument(
        "--model",
        default="isnet-general-use",
        help="Modello rembg (usato con method rembg/auto)",
    )
    parser.add_argument("--low-threshold", type=float, default=18.0, help="Soglia bassa per chroma key")
    parser.add_argument("--high-threshold", type=float, default=110.0, help="Soglia alta per chroma key")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.method == "grabcut":
        remove_with_grabcut(input_path, output_path)
        print(f"Background removed with grabcut: {output_path}")
        return

    if args.method == "chroma":
        remove_with_chroma_key(
            input_path,
            output_path,
            low_threshold=args.low_threshold,
            high_threshold=args.high_threshold,
        )
        print(f"Background removed with chroma key: {output_path}")
        return

    if args.method == "rembg":
        remove_with_rembg(input_path, output_path, model=args.model)
        print(f"Background removed with rembg ({args.model}): {output_path}")
        return

    try:
        remove_with_rembg(input_path, output_path, model=args.model)
        print(f"Background removed with rembg ({args.model}): {output_path}")
    except Exception as ex:  # pragma: no cover
        print(f"rembg failed ({ex}), falling back to grabcut")
        remove_with_grabcut(input_path, output_path)
        print(f"Background removed with grabcut: {output_path}")


if __name__ == "__main__":
    main()
