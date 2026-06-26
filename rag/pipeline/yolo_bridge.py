#!/usr/bin/env python3
"""Bridge: YOLO ingredient detection -> canonical ingredient terms for the RAG pipeline.

Runs the trained YOLO detector on an image, deduplicates detections (max confidence
per class), and maps the 47 class labels to the corpus's canonical ingredient
vocabulary. Verified against the runtime corpus: 46/47 labels match directly via the
underscore->space rule; the only exception is captured in YOLO_LABEL_ALIASES.

This module is the only place the YOLO half is coupled to RAG; the RAG core is
untouched. Its output (a canonical ingredient term list) feeds the existing
`available_ingredients` contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# YOLO class label -> canonical corpus term. Only the exceptions are listed; every
# other label uses the underscore->space rule (verified 46/47 against the corpus).
YOLO_LABEL_ALIASES = {
    "mint_leaves": "mint",
}


def to_canonical(label: str) -> str:
    """Map a YOLO class label to the corpus's canonical ingredient term."""
    key = str(label).strip().lower()
    if key in YOLO_LABEL_ALIASES:
        return YOLO_LABEL_ALIASES[key]
    return key.replace("_", " ")


def load_yolo(model_path: Path | str) -> Any:
    """Load the YOLO model once (so the demo can cache it across queries)."""
    from ultralytics import YOLO

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"YOLO checkpoint not found: {path}")
    return YOLO(str(path))


def detect_ingredients(
    model: Any,
    image_path: Path | str,
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 768,
    device: str = "cpu",
) -> list[dict[str, Any]]:
    """Detect ingredients in one image.

    Returns deduped detections (max confidence per class), sorted high->low:
    `[{"label": "green_onion", "canonical": "green onion", "confidence": 0.91}, ...]`.
    """
    results = model.predict(
        source=str(Path(image_path)),
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        verbose=False,
        stream=False,
    )
    best_conf: dict[str, float] = {}
    if results:
        result = results[0]
        names = getattr(result, "names", {})
        boxes = getattr(result, "boxes", None)
        if boxes is not None and len(boxes) > 0:
            for cls_id, conf_value in zip(boxes.cls.tolist(), boxes.conf.tolist()):
                label = str(names.get(int(cls_id), str(int(cls_id))))
                value = float(conf_value)
                if label not in best_conf or value > best_conf[label]:
                    best_conf[label] = value
    detections = [
        {"label": label, "canonical": to_canonical(label), "confidence": round(value, 3)}
        for label, value in best_conf.items()
    ]
    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections
