"""Final YOLO11m training script for ingredient detection.

This recipe is the main final candidate:
- based on the strongest YOLO11 direction (`ftbest_11`)
- keeps the stable FT-A2 backbone
- optionally adds recall-first checkpoint selection
- optionally adds light Albumentations for camera-style corruption

Rationale for the final upgrade:
- many inference misses come from small or thin objects such as garlic, chili,
  lime, bell_pepper, and green_onion
- some missed items in demo images are out-of-vocabulary (oil, flour, spices),
  so the training recipe should focus on small-object fidelity instead
- use the standard 640 training resolution for better stability with this
  dataset's source image sizes
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict

import albumentations as A
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils.downloads import attempt_download_asset

LOGGER = logging.getLogger("train")

DEFAULT_PROJECT = Path("runs")
DEFAULT_MODEL_CACHE = Path("model_weights")
DEFAULT_DEVICE = "2"

RUN_CONFIG: Dict[str, Any] = {
    "start_weights": "yolo11m.pt",
    "epochs": 300,
    "patience": 50,
    "imgsz": 640,
    "batch": 40,
    "workers": 12,
    "seed": 42,
    "amp": True,
    "cache": True,
    "optimizer": "SGD",
    "lr0": 0.007,
    "lrf": 0.01,
    "cos_lr": True,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,
    "box": 7.5,
    "cls": 0.6,
    "cls_pw": 0.25,
    "dfl": 1.5,
    "mosaic": 1.0,
    "close_mosaic": 20,
    "hsv_h": 0.02,
    "hsv_s": 0.8,
    "hsv_v": 0.5,
    "translate": 0.15,
    "scale": 0.65,
    "fliplr": 0.5,
    "flipud": 0.0,
    "mixup": 0.05,
    "cutmix": 0.05,
    "copy_paste": 0.0,
    "bgr": 0.0,
    "degrees": 4.0,
    "shear": 2.0,
    "perspective": 0.0003,
    "auto_augment": None,
    "erasing": 0.0,
}


def build_light_albumentations() -> list[A.BasicTransform]:
    return [
        A.RandomBrightnessContrast(brightness_limit=0.10, contrast_limit=0.10, p=0.15),
        A.OneOf(
            [
                A.ImageCompression(quality_range=(75, 92), p=1.0),
                A.GaussNoise(std_range=(0.01, 0.03), p=1.0),
                A.MotionBlur(blur_limit=3, p=1.0),
                A.GaussianBlur(blur_limit=3, sigma_limit=(0.5, 1.2), p=1.0),
            ],
            p=0.10,
        ),
    ]


class RecallFitnessTrainer(DetectionTrainer):
    """Select best.pt using recall-first fitness for HITL ingredient extraction."""

    def validate(self):
        saved_best = self.best_fitness
        self.best_fitness = None

        metrics, fitness = super().validate()

        try:
            precision = float(metrics.get("metrics/precision(B)", 0.0))
            recall = float(metrics.get("metrics/recall(B)", 0.0))
            map50 = float(metrics.get("metrics/mAP50(B)", 0.0))
            map50_95 = float(metrics.get("metrics/mAP50-95(B)", 0.0))

            custom_fitness = 0.5 * recall + 0.25 * map50 + 0.25 * map50_95

            self.best_fitness = saved_best
            if self.best_fitness is None or self.best_fitness < custom_fitness:
                self.best_fitness = custom_fitness

            LOGGER.info(
                "Val Epoch=%d P=%.4f R=%.4f mAP50=%.4f mAP50-95=%.4f Fitness*=%.4f",
                self.epoch + 1,
                precision,
                recall,
                map50,
                map50_95,
                custom_fitness,
            )
            fitness = custom_fitness
        except Exception as exc:
            self.best_fitness = saved_best
            LOGGER.warning("Custom fitness failed, fallback to default: %s", exc)

        return metrics, fitness


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train final YOLO11m ingredient detector")
    parser.add_argument("--data", type=Path, required=True, help="Path to YOLO data.yaml")
    parser.add_argument("--name", type=str, required=True, help="Run name")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--batch", type=int, default=None, help="Override default batch size")
    parser.add_argument("--workers", type=int, default=RUN_CONFIG["workers"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-path", type=Path, default=None)
    parser.add_argument("--use-light-albu", action="store_true")
    parser.add_argument("--use-recall-trainer", action="store_true")
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def ensure_paths(args: argparse.Namespace) -> None:
    if not args.data.exists():
        raise FileNotFoundError(f"data.yaml not found: {args.data}")
    DEFAULT_PROJECT.mkdir(parents=True, exist_ok=True)
    DEFAULT_MODEL_CACHE.mkdir(parents=True, exist_ok=True)


def ensure_model_cached(model_name: str, cache_dir: Path) -> Path:
    local_path = (cache_dir / model_name).resolve()

    if local_path.exists():
        LOGGER.info("Model already cached: %s", local_path)
        return local_path

    LOGGER.info("Downloading model weights: %s", model_name)
    try:
        downloaded = Path(attempt_download_asset(model_name, repo="ultralytics/assets"))
        if not downloaded.exists():
            raise FileNotFoundError(f"Downloaded file not found: {downloaded}")
        if downloaded.resolve() != local_path.resolve():
            shutil.move(str(downloaded), str(local_path))
        LOGGER.info("Weights cached at: %s", local_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to cache '{model_name}': {exc}") from exc

    return local_path


def resolve_start_weights(weights_ref: str) -> Path:
    candidate = Path(weights_ref)
    if candidate.exists():
        return candidate.resolve()
    if "/" in weights_ref or "\\" in weights_ref:
        raise FileNotFoundError(f"start_weights path not found: {weights_ref}")
    return ensure_model_cached(weights_ref, DEFAULT_MODEL_CACHE)


def resolve_resume_checkpoint(args: argparse.Namespace) -> Path | None:
    if args.resume_path is not None:
        checkpoint = args.resume_path.resolve()
        if not checkpoint.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {checkpoint}")
        return checkpoint

    if args.resume:
        checkpoint = (DEFAULT_PROJECT / args.name / "weights" / "last.pt").resolve()
        if not checkpoint.exists():
            raise FileNotFoundError(f"resume requested but checkpoint not found: {checkpoint}")
        return checkpoint

    return None


def build_train_params(args: argparse.Namespace) -> Dict[str, Any]:
    batch_size = int(args.batch if args.batch is not None else RUN_CONFIG["batch"])
    train_params: Dict[str, Any] = {
        "data": str(args.data.resolve()),
        "epochs": int(RUN_CONFIG["epochs"]),
        "imgsz": int(RUN_CONFIG["imgsz"]),
        "batch": batch_size,
        "device": args.device,
        "workers": int(args.workers),
        "patience": int(RUN_CONFIG["patience"]),
        "seed": int(RUN_CONFIG["seed"]),
        "project": str(DEFAULT_PROJECT.resolve()),
        "name": args.name,
        "amp": bool(RUN_CONFIG["amp"]),
        "cache": bool(RUN_CONFIG["cache"]),
        "optimizer": str(RUN_CONFIG["optimizer"]),
        "lr0": float(RUN_CONFIG["lr0"]),
        "lrf": float(RUN_CONFIG["lrf"]),
        "cos_lr": bool(RUN_CONFIG["cos_lr"]),
        "momentum": float(RUN_CONFIG["momentum"]),
        "weight_decay": float(RUN_CONFIG["weight_decay"]),
        "warmup_epochs": float(RUN_CONFIG["warmup_epochs"]),
        "warmup_momentum": float(RUN_CONFIG["warmup_momentum"]),
        "warmup_bias_lr": float(RUN_CONFIG["warmup_bias_lr"]),
        "box": float(RUN_CONFIG["box"]),
        "cls": float(RUN_CONFIG["cls"]),
        "cls_pw": float(RUN_CONFIG["cls_pw"]),
        "dfl": float(RUN_CONFIG["dfl"]),
        "mosaic": float(RUN_CONFIG["mosaic"]),
        "close_mosaic": int(RUN_CONFIG["close_mosaic"]),
        "hsv_h": float(RUN_CONFIG["hsv_h"]),
        "hsv_s": float(RUN_CONFIG["hsv_s"]),
        "hsv_v": float(RUN_CONFIG["hsv_v"]),
        "translate": float(RUN_CONFIG["translate"]),
        "scale": float(RUN_CONFIG["scale"]),
        "fliplr": float(RUN_CONFIG["fliplr"]),
        "flipud": float(RUN_CONFIG["flipud"]),
        "mixup": float(RUN_CONFIG["mixup"]),
        "cutmix": float(RUN_CONFIG["cutmix"]),
        "copy_paste": float(RUN_CONFIG["copy_paste"]),
        "bgr": float(RUN_CONFIG["bgr"]),
        "degrees": float(RUN_CONFIG["degrees"]),
        "shear": float(RUN_CONFIG["shear"]),
        "perspective": float(RUN_CONFIG["perspective"]),
        "auto_augment": RUN_CONFIG["auto_augment"],
        "erasing": float(RUN_CONFIG["erasing"]),
    }

    if args.use_light_albu:
        train_params["augmentations"] = build_light_albumentations()

    return train_params


def run_training(args: argparse.Namespace) -> Dict[str, Any]:
    resume_checkpoint = resolve_resume_checkpoint(args)
    weights_path = resume_checkpoint or resolve_start_weights(str(RUN_CONFIG["start_weights"]))

    LOGGER.info("Loading model from: %s", weights_path)
    model = YOLO(str(weights_path))
    train_params = build_train_params(args)

    if resume_checkpoint is not None:
        train_params["resume"] = True
        LOGGER.info("Resuming from checkpoint: %s", resume_checkpoint)

    LOGGER.info("Starting training with params: %s", train_params)
    trainer = RecallFitnessTrainer if args.use_recall_trainer else None
    result = model.train(trainer=trainer, **train_params) if trainer else model.train(**train_params)

    metrics = getattr(result, "results_dict", {})
    save_dir = str(getattr(result, "save_dir", ""))

    return {
        "run_name": args.name,
        "save_dir": save_dir,
        "model": str(RUN_CONFIG["start_weights"]),
        "resume": str(resume_checkpoint) if resume_checkpoint else None,
        "weights": str(weights_path),
        "data": str(args.data.resolve()),
        "epochs": int(RUN_CONFIG["epochs"]),
        "imgsz": int(RUN_CONFIG["imgsz"]),
        "batch": int(train_params["batch"]),
        "device": str(train_params["device"]),
        "workers": int(train_params["workers"]),
        "patience": int(RUN_CONFIG["patience"]),
        "use_light_albu": bool(args.use_light_albu),
        "use_recall_trainer": bool(args.use_recall_trainer),
        "metrics": metrics,
    }


def save_summary(summary: Dict[str, Any], args: argparse.Namespace) -> Path:
    output_path = (DEFAULT_PROJECT / f"{args.name}_summary.json").resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return output_path


def main() -> None:
    args = parse_args()
    configure_logging()
    ensure_paths(args)

    try:
        summary = run_training(args)
    except Exception as exc:
        LOGGER.error("Training failed: %s", exc, exc_info=True)
        raise SystemExit(1) from exc

    summary_path = save_summary(summary, args)
    LOGGER.info("Saved training summary to %s", summary_path)


if __name__ == "__main__":
    main()
