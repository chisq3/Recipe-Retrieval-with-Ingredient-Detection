"""Evaluate YOLO detection metrics on a dataset split.

This script reports Ultralytics/YOLO validation metrics only:
precision, recall, mAP50, and mAP50-95. It intentionally does not compute the
older image-level class-presence metrics, so the output can be used directly for
paper-style object-detection comparison.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence

from ultralytics import YOLO
import yaml


LOGGER = logging.getLogger("evaluate_yolo")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate standard YOLO metrics on a dataset split")
    parser.add_argument("--model", type=Path, required=True, help="Path to YOLO weights (.pt)")
    parser.add_argument("--data", type=Path, required=True, help="Path to YOLO data.yaml")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--batch", type=int, default=40)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", type=Path, default=None, help="Optional Ultralytics validation output directory")
    parser.add_argument("--name", type=str, default=None, help="Optional Ultralytics validation run name")
    parser.add_argument("--report-path", type=Path, default=Path("evaluation_report_PR.json"))
    parser.add_argument("--plots", action="store_true", help="Save Ultralytics validation plots")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level), format="%(asctime)s | %(levelname)s | %(message)s")


def load_class_names(data_yaml: Path) -> List[str]:
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Invalid data.yaml format")

    names = config.get("names")
    if isinstance(names, dict):
        return [str(names[index]) for index in sorted(names)]
    if isinstance(names, list) and names:
        return [str(name) for name in names]
    raise ValueError("data.yaml must contain a non-empty 'names' list or dict")


def to_float(value: Any) -> float:
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def to_float_list(values: Any) -> List[float]:
    if values is None:
        return []
    if hasattr(values, "tolist"):
        values = values.tolist()
    try:
        return [to_float(value) for value in list(values)]
    except TypeError:
        return [to_float(values)]


def aligned_metric(values: Any, class_indices: Any, num_classes: int) -> List[float | None]:
    """Align Ultralytics per-class arrays to the full class-id order."""
    vals = to_float_list(values)
    if len(vals) == num_classes:
        return vals

    aligned: List[float | None] = [None] * num_classes
    for class_index, value in zip(to_float_list(class_indices), vals):
        class_id = int(class_index)
        if 0 <= class_id < num_classes:
            aligned[class_id] = value
    return aligned


def aligned_count(values: Any, num_classes: int) -> List[int]:
    counts = [0] * num_classes
    for class_id, value in enumerate(to_float_list(values)):
        if class_id < num_classes:
            counts[class_id] = int(value)
    return counts


def validation_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "data": str(args.data.resolve()),
        "split": args.split,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "device": args.device,
        "plots": args.plots,
        "verbose": False,
    }
    if args.project is not None:
        kwargs["project"] = str(args.project.resolve())
    if args.name:
        kwargs["name"] = args.name
    return kwargs


def run_validation(args: argparse.Namespace) -> Any:
    if not args.model.exists():
        raise FileNotFoundError(f"Model file not found: {args.model}")

    LOGGER.info("Loading model: %s", args.model)
    model = YOLO(str(args.model.resolve()))

    LOGGER.info("Running Ultralytics model.val() on split='%s'", args.split)
    return model.val(**validation_kwargs(args))


def build_overall_metrics(result: Any) -> Dict[str, float]:
    box = getattr(result, "box", None)
    if box is None:
        raise RuntimeError("Validation result does not contain box metrics")

    return {
        "precision": to_float(getattr(box, "mp", 0.0)),
        "recall": to_float(getattr(box, "mr", 0.0)),
        "mAP50": to_float(getattr(box, "map50", 0.0)),
        "mAP50_95": to_float(getattr(box, "map", 0.0)),
        "fitness": to_float(getattr(result, "fitness", 0.0)),
    }


def build_per_class_metrics(result: Any, class_names: Sequence[str]) -> List[Dict[str, Any]]:
    box = getattr(result, "box", None)
    if box is None:
        raise RuntimeError("Validation result does not contain box metrics")

    num_classes = len(class_names)
    class_indices = getattr(box, "ap_class_index", [])
    precision = aligned_metric(getattr(box, "p", []), class_indices, num_classes)
    recall = aligned_metric(getattr(box, "r", []), class_indices, num_classes)
    ap50 = aligned_metric(getattr(box, "ap50", []), class_indices, num_classes)
    ap = aligned_metric(getattr(box, "ap", []), class_indices, num_classes)

    nt_per_image = aligned_count(getattr(result, "nt_per_image", []), num_classes)
    nt_per_class = aligned_count(getattr(result, "nt_per_class", []), num_classes)

    return [
        {
            "class_id": class_id,
            "class_name": class_name,
            "images": nt_per_image[class_id],
            "instances": nt_per_class[class_id],
            "precision": precision[class_id],
            "recall": recall[class_id],
            "mAP50": ap50[class_id],
            "mAP50_95": ap[class_id],
        }
        for class_id, class_name in enumerate(class_names)
    ]


def build_report(args: argparse.Namespace, result: Any, class_names: Sequence[str]) -> Dict[str, Any]:
    return {
        "config": {
            "model": str(args.model.resolve()),
            "data": str(args.data.resolve()),
            "split": args.split,
            "imgsz": args.imgsz,
            "device": args.device,
            "batch": args.batch,
            "workers": args.workers,
        },
        "metric_definitions": {
            "precision": "bbox-level mean precision from Ultralytics model.val()",
            "recall": "bbox-level mean recall from Ultralytics model.val()",
            "mAP50": "bbox-level mean average precision at IoU=0.50",
            "mAP50_95": "bbox-level mean average precision over IoU=0.50:0.95",
            "fitness": "Ultralytics validation fitness score",
        },
        "overall": build_overall_metrics(result),
        "per_class": build_per_class_metrics(result, class_names),
    }


def write_report(report: Dict[str, Any], report_path: Path) -> None:
    report_path = report_path.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    LOGGER.info("Saved report: %s", report_path)


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    class_names = load_class_names(args.data.resolve())
    result = run_validation(args)
    report = build_report(args, result, class_names)
    write_report(report, args.report_path)

    overall = report["overall"]
    LOGGER.info(
        "YOLO metrics | P=%.4f R=%.4f mAP50=%.4f mAP50-95=%.4f fitness=%.4f",
        overall["precision"],
        overall["recall"],
        overall["mAP50"],
        overall["mAP50_95"],
        overall["fitness"],
    )


if __name__ == "__main__":
    main()
