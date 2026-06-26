#!/usr/bin/env python3
"""Run YOLO inference on a single image and export annotated output."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

LOGGER = logging.getLogger("inference")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed argparse namespace.
    """
    parser = argparse.ArgumentParser(description="Single-image inference for ingredient detection")
    parser.add_argument("--model", type=Path, required=True, help="Path to YOLO weights (.pt)")
    parser.add_argument("--image", type=Path, required=True, help="Input image path")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=768, help="Inference image size")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: 'auto' (default), 'cpu', or GPU id like '0'",
    )
    parser.add_argument(
        "--output-image",
        type=Path,
        default=None,
        help="Output image path (default: <image_stem>_pred<ext> next to image)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional JSON output path (if provided, JSON will be written)",
    )
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Disable drawing class labels and confidence on boxes",
    )
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    return parser.parse_args()


def configure_logging(level: str) -> None:
    """Configure logging.

    Args:
        level: Logging level string.
    """
    logging.basicConfig(level=getattr(logging, level), format="%(asctime)s | %(levelname)s | %(message)s")


def resolve_device(device_arg: str) -> str:
    """Resolve runtime device with safe CPU fallback.

    Args:
        device_arg: CLI value for --device.

    Returns:
        'cpu' or GPU device string accepted by Ultralytics.
    """
    dev = device_arg.strip().lower()
    if dev != "auto":
        return device_arg

    try:
        import torch

        if torch.cuda.is_available():
            return "0"
    except Exception:
        pass
    return "cpu"


def build_detection_json(result: Any, image_path: Path) -> Dict[str, Any]:
    """Convert Ultralytics result into required JSON schema.

    Args:
        result: Ultralytics prediction result for one image.
        image_path: Input image path.

    Returns:
        Dictionary matching required detection schema.
    """
    names = getattr(result, "names", {})
    detections: List[Dict[str, Any]] = []
    unique_classes = set()

    boxes = getattr(result, "boxes", None)
    if boxes is not None and len(boxes) > 0:
        xywhn = boxes.xywhn.tolist() if boxes.xywhn is not None else []
        cls_ids = boxes.cls.tolist() if boxes.cls is not None else []
        confs = boxes.conf.tolist() if boxes.conf is not None else []

        for bbox, cls_id_raw, conf in zip(xywhn, cls_ids, confs):
            cls_id = int(cls_id_raw)
            class_name = str(names.get(cls_id, str(cls_id)))
            unique_classes.add(cls_id)

            detections.append(
                {
                    "class_id": cls_id,
                    "class_name": class_name,
                    "confidence": float(conf),
                    "bbox": {
                        "x_center": float(bbox[0]),
                        "y_center": float(bbox[1]),
                        "width": float(bbox[2]),
                        "height": float(bbox[3]),
                        "format": "yolo_normalized",
                    },
                }
            )

    return {
        "image_path": str(image_path.resolve()),
        "detections": detections,
        "total_unique_ingredients": len(unique_classes),
        # List tên nguyên liệu duy nhất — dùng trực tiếp làm input cho RAG pipeline
        "unique_ingredient_names": sorted({d["class_name"] for d in detections}),
    }


def save_annotated_image(result: Any, output_path: Path, draw_labels: bool) -> None:
    """Save annotated image with bounding boxes.

    Args:
        result: Ultralytics prediction result for one image.
        output_path: Path to write annotated image.
        draw_labels: Whether to draw class labels and confidence.
    """
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("Missing dependency: opencv-python (cv2)") from exc

    annotated = result.plot(labels=draw_labels, conf=draw_labels)
    if not cv2.imwrite(str(output_path), annotated):
        raise RuntimeError(f"Failed to write annotated image: {output_path}")


def main() -> None:
    """Run one-image inference and write annotated output."""
    args = parse_args()
    configure_logging(args.log_level)

    model_path = args.model.resolve()
    image_path = args.image.resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("Please install ultralytics: pip install ultralytics") from exc

    LOGGER.info("Loading model: %s", model_path)
    model = YOLO(str(model_path))

    device = resolve_device(args.device)
    LOGGER.info("Using device: %s", device)

    LOGGER.info("Running inference for image: %s", image_path)
    results = model.predict(
        source=str(image_path),
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=device,
        verbose=False,
        stream=False,
    )

    if not results:
        raise RuntimeError("No result returned by model.predict")

    output = build_detection_json(results[0], image_path)
    output_image_path = (
        args.output_image.resolve()
        if args.output_image
        else image_path.with_name(f"{image_path.stem}_pred{image_path.suffix}")
    )

    save_annotated_image(results[0], output_image_path, draw_labels=not args.no_labels)
    LOGGER.info("Saved annotated image: %s", output_image_path)

    if args.output_json:
        output_json_path = args.output_json.resolve()
        output_json_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
        LOGGER.info("Saved inference JSON: %s", output_json_path)

    LOGGER.info("Detections: %d boxes, %d unique classes", len(output["detections"]), output["total_unique_ingredients"])


if __name__ == "__main__":
    main()
