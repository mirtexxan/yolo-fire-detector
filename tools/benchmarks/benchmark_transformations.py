"""Micro-benchmark for dataset transformations and sample generation."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import json
from pathlib import Path
import random
import statistics
import sys
import tempfile
import time
from typing import Any, Callable

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from generator import generate_negative_sample, generate_positive_sample
from settings import DatasetGenerationSettings, ImageTransformSettings
from transformations import (
    add_gaussian_blur,
    add_motion_blur,
    add_noise,
    add_occlusion_from_background,
    add_shadow,
    adjust_brightness_contrast,
    alpha_composite,
    augment_background,
    augment_fire,
    background_blobs,
    background_checker,
    background_flat_color,
    background_gradient,
    background_lines,
    background_mixed,
    background_noise,
    color_shift_hsv,
    generate_random_background,
    perspective_warp_keep_canvas,
    resize_fire_with_alpha,
    rotate_image_keep_canvas,
)
from utils import load_fire_image, make_output_folders, save_sample
DEFAULT_FIRE_PATH = PROJECT_ROOT / "base_fire_images" / "fire.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark trasformazioni dataset")
    parser.add_argument("--iterations", type=int, default=40, help="Numero di iterazioni per benchmark")
    parser.add_argument("--warmup", type=int, default=5, help="Numero di warmup per benchmark")
    parser.add_argument("--image-size", type=int, default=640, help="Dimensione immagine di test")
    parser.add_argument("--seed", type=int, default=42, help="Seed per benchmark ripetibile")
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="Config YAML da confrontare. Ripetibile, es. --config configs/presets/default.yaml --config configs/presets/fast-debug.yaml",
    )
    parser.add_argument("--output-json", type=str, default="", help="Path opzionale per salvare risultati JSON")
    return parser.parse_args()


@contextmanager
def patched_settings(cls: type, overrides: dict[str, Any]):
    original = {key: getattr(cls, key) for key in overrides}
    try:
        for key, value in overrides.items():
            setattr(cls, key, value)
        yield
    finally:
        for key, value in original.items():
            setattr(cls, key, value)


def benchmark_case(name: str, func: Callable[[], Any], iterations: int, warmup: int) -> dict[str, Any]:
    gc.collect()
    for _ in range(warmup):
        func()

    samples_ms: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        samples_ms.append((time.perf_counter() - start) * 1000.0)

    return {
        "name": name,
        "avg_ms": round(statistics.mean(samples_ms), 3),
        "median_ms": round(statistics.median(samples_ms), 3),
        "min_ms": round(min(samples_ms), 3),
        "max_ms": round(max(samples_ms), 3),
    }


def print_results(title: str, rows: list[dict[str, Any]]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    print(f"{'case':40} {'avg ms':>10} {'median':>10} {'min':>10} {'max':>10}")
    for row in sorted(rows, key=lambda item: item["avg_ms"], reverse=True):
        print(
            f"{row['name'][:40]:40} "
            f"{row['avg_ms']:10.3f} {row['median_ms']:10.3f} {row['min_ms']:10.3f} {row['max_ms']:10.3f}"
        )


def to_setting_overrides(payload: dict[str, Any]) -> dict[str, Any]:
    return {key.upper(): value for key, value in payload.items()}


def load_profile_config(config_path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config YAML non valida: {config_path}")
    return payload


def benchmark_profile(
    *,
    profile_name: str,
    config_payload: dict[str, Any],
    iterations: int,
    warmup: int,
    default_image_size: int,
    default_seed: int,
) -> dict[str, Any]:
    dataset_cfg = config_payload.get("dataset", {})
    if not isinstance(dataset_cfg, dict):
        dataset_cfg = {}

    image_transform_overrides = config_payload.get("image_transform_overrides", {})
    if not isinstance(image_transform_overrides, dict):
        image_transform_overrides = {}

    dataset_settings_overrides = config_payload.get("dataset_settings_overrides", {})
    if not isinstance(dataset_settings_overrides, dict):
        dataset_settings_overrides = {}

    image_size = int(dataset_cfg.get("image_size", default_image_size))
    seed = int(dataset_cfg.get("seed", default_seed))

    with patched_settings(ImageTransformSettings, to_setting_overrides(image_transform_overrides)), patched_settings(
        DatasetGenerationSettings,
        to_setting_overrides(dataset_settings_overrides),
    ):
        random.seed(seed)
        np.random.seed(seed)

        fire = load_fire_image(str(DEFAULT_FIRE_PATH))
        background = generate_random_background(image_size)
        resized_fire = resize_fire_with_alpha(fire, 0.35)
        fire_h, fire_w = resized_fire.shape[:2]
        fire_x = max(0, (image_size - fire_w) // 2)
        fire_y = max(0, (image_size - fire_h) // 2)

        rows = [
            benchmark_case("generate_random_background", lambda: generate_random_background(image_size), iterations, warmup),
            benchmark_case("augment_fire_current", lambda: augment_fire(fire), iterations, warmup),
            benchmark_case("augment_background", lambda: augment_background(background), iterations, warmup),
            benchmark_case("generate_negative_sample", lambda: generate_negative_sample(image_size), iterations, warmup),
            benchmark_case("generate_positive_sample", lambda: generate_positive_sample(fire, image_size), iterations, warmup),
        ]

        with tempfile.TemporaryDirectory(prefix=f"fire-bench-{profile_name}-") as temp_dir:
            make_output_folders(temp_dir)
            sample_image = generate_random_background(image_size)
            rows.append(
                benchmark_case(
                    "save_sample_jpg_txt",
                    lambda: save_sample(sample_image, "", temp_dir, random.randint(0, 999999), 0.8),
                    max(10, iterations // 2),
                    min(3, warmup),
                )
            )

    return {
        "profile": profile_name,
        "image_size": image_size,
        "seed": seed,
        "rows": rows,
    }


def print_profile_comparison(profiles: list[dict[str, Any]]) -> None:
    if len(profiles) < 2:
        return

    baseline = profiles[0]
    baseline_map = {row["name"]: row for row in baseline["rows"]}

    print("\nprofile comparison")
    print("------------------")
    for profile in profiles:
        print(f"Profile {profile['profile']} (image_size={profile['image_size']}, seed={profile['seed']})")
        print(f"{'case':32} {'avg ms':>10} {'vs baseline':>12}")
        for row in profile["rows"]:
            baseline_row = baseline_map.get(row["name"])
            if baseline_row is None:
                ratio = "n/a"
            else:
                ratio = f"{row['avg_ms'] / baseline_row['avg_ms']:.2f}x"
            print(f"{row['name'][:32]:32} {row['avg_ms']:10.3f} {ratio:>12}")
        print()


def main() -> int:
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    fire = load_fire_image(str(DEFAULT_FIRE_PATH))
    background = generate_random_background(args.image_size)
    resized_fire = resize_fire_with_alpha(fire, 0.35)
    fire_h, fire_w = resized_fire.shape[:2]
    fire_x = max(0, (args.image_size - fire_w) // 2)
    fire_y = max(0, (args.image_size - fire_h) // 2)
    composited = alpha_composite(background, resized_fire, fire_x, fire_y)

    background_rows = [
        benchmark_case("background_flat_color", lambda: background_flat_color(args.image_size), args.iterations, args.warmup),
        benchmark_case("background_noise", lambda: background_noise(args.image_size), args.iterations, args.warmup),
        benchmark_case("background_gradient", lambda: background_gradient(args.image_size), args.iterations, args.warmup),
        benchmark_case("background_blobs", lambda: background_blobs(args.image_size), args.iterations, args.warmup),
        benchmark_case("background_lines", lambda: background_lines(args.image_size), args.iterations, args.warmup),
        benchmark_case("background_checker", lambda: background_checker(args.image_size), args.iterations, args.warmup),
        benchmark_case("background_mixed", lambda: background_mixed(args.image_size), args.iterations, args.warmup),
        benchmark_case("generate_random_background", lambda: generate_random_background(args.image_size), args.iterations, args.warmup),
    ]

    fire_rows = [
        benchmark_case("rotate_image_keep_canvas", lambda: rotate_image_keep_canvas(fire, 25.0), args.iterations, args.warmup),
        benchmark_case("perspective_warp_keep_canvas", lambda: perspective_warp_keep_canvas(fire, ImageTransformSettings.PERSPECTIVE_SHIFT), args.iterations, args.warmup),
        benchmark_case("adjust_brightness_contrast", lambda: adjust_brightness_contrast(fire, 1.1, 12), args.iterations, args.warmup),
        benchmark_case("color_shift_hsv", lambda: color_shift_hsv(fire, 10), args.iterations, args.warmup),
        benchmark_case("add_gaussian_blur", lambda: add_gaussian_blur(fire, 5), args.iterations, args.warmup),
        benchmark_case("add_motion_blur", lambda: add_motion_blur(fire, 7), args.iterations, args.warmup),
        benchmark_case("add_noise", lambda: add_noise(fire, 15), args.iterations, args.warmup),
        benchmark_case("resize_fire_with_alpha", lambda: resize_fire_with_alpha(fire, 0.35), args.iterations, args.warmup),
        benchmark_case("augment_fire_current", lambda: augment_fire(fire), args.iterations, args.warmup),
    ]

    with patched_settings(ImageTransformSettings, {"SHADOW_PROB": 1.0}):
        forced_shadow = benchmark_case(
            "add_shadow_forced_on",
            lambda: add_shadow(background, fire_x, fire_y, fire_w, fire_h),
            args.iterations,
            args.warmup,
        )

    with patched_settings(ImageTransformSettings, {"OCCLUSION_PROB": 1.0}):
        forced_occlusion = benchmark_case(
            "add_occlusion_forced_on",
            lambda: add_occlusion_from_background(composited, fire_x, fire_y, fire_w, fire_h),
            args.iterations,
            args.warmup,
        )

    compositing_rows = [
        benchmark_case("alpha_composite", lambda: alpha_composite(background, resized_fire, fire_x, fire_y), args.iterations, args.warmup),
        benchmark_case("augment_background", lambda: augment_background(background), args.iterations, args.warmup),
        forced_shadow,
        forced_occlusion,
    ]

    pipeline_rows = [
        benchmark_case("generate_negative_sample", lambda: generate_negative_sample(args.image_size), args.iterations, args.warmup),
        benchmark_case("generate_positive_sample", lambda: generate_positive_sample(fire, args.image_size), args.iterations, args.warmup),
    ]

    with tempfile.TemporaryDirectory(prefix="fire-bench-") as temp_dir:
        make_output_folders(temp_dir)
        sample_image = generate_random_background(args.image_size)
        save_rows = [
            benchmark_case(
                "save_sample_jpg_txt",
                lambda: save_sample(sample_image, "", temp_dir, random.randint(0, 999999), 0.8),
                max(10, args.iterations // 2),
                min(3, args.warmup),
            )
        ]

    all_rows = {
        "backgrounds": background_rows,
        "fire": fire_rows,
        "compositing": compositing_rows,
        "pipelines": pipeline_rows,
        "io": save_rows,
    }

    profile_results: list[dict[str, Any]] = []
    if args.config:
        for raw_path in args.config:
            config_path = (PROJECT_ROOT / raw_path).resolve() if not Path(raw_path).is_absolute() else Path(raw_path)
            payload = load_profile_config(config_path)
            profile_results.append(
                benchmark_profile(
                    profile_name=config_path.stem,
                    config_payload=payload,
                    iterations=args.iterations,
                    warmup=args.warmup,
                    default_image_size=args.image_size,
                    default_seed=args.seed,
                )
            )

    print(f"Benchmark trasformazioni - image_size={args.image_size}, iterations={args.iterations}, warmup={args.warmup}")
    for title, rows in all_rows.items():
        print_results(title, rows)

    if profile_results:
        print_profile_comparison(profile_results)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "microbenchmarks": all_rows,
            "profiles": profile_results,
        }
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nRisultati JSON scritti in: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())