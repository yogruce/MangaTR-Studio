"""
MangaTR-Studio - text_detection.py
(Stage 1-4: Comic Text Detector + Duplicate Suppression + Line-Level EasyOCR + Clean JSON)
"""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import torch

from app.config import StudioConfig, ROOT_DIR
from app.engines.detector import MangaTextDetector
from app.engines.ocr_engine import MangaOCREngine
from app.utils.json_utils import save_json


INPUT_PATH = ROOT_DIR / "input" / "test.jpg"
OUTPUT_DIR = ROOT_DIR / "output" / "detection"
REGION_DIR = OUTPUT_DIR / "test_regions"


def draw_detection(
    image,
    box,
    line_boxes,
    index: int,
    mask_conf: float,
    ocr_conf: float,
    text: str,
) -> None:
    x1, y1, x2, y2 = box

    # Draw subtle orange inner boxes for individual detected lines
    for lb in line_boxes:
        lx1, ly1, lx2, ly2 = lb
        cv2.rectangle(image, (lx1, ly1), (lx2, ly2), (0, 165, 255), 1)

    # Draw outer green block bounding box
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 185, 0), 2)

    label = f"#{index:02d} ocr:{ocr_conf:.2f}"
    cv2.putText(
        image,
        label,
        (x1, max(20, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (230, 60, 0),
        2,
        cv2.LINE_AA,
    )

    if text:
        preview = text.replace("\n", " ")
        if len(preview) > 42:
            preview = preview[:39] + "..."
        cv2.putText(
            image,
            preview,
            (x1, min(image.shape[0] - 8, y2 + 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (0, 90, 230),
            1,
            cv2.LINE_AA,
        )


def main() -> None:
    print("=" * 74)
    print("MangaTR-Studio - Text Detection + Duplicate Cleanup + Line-Level OCR")
    print("=" * 74)

    config = StudioConfig()
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input image not found: {INPUT_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if REGION_DIR.exists():
        shutil.rmtree(REGION_DIR)
    REGION_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Input : {INPUT_PATH}")
    print(f"Model : {config.detector.model_path}")
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")

    print("\n[1/3] Comic Text Detector + Duplicate Temizleme başlatılıyor...")
    detector = MangaTextDetector(config.detector, device=device)

    image = cv2.imread(str(INPUT_PATH), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {INPUT_PATH}")

    print(f"Image : {image.shape[1]}x{image.shape[0]}")
    mask, mask_refined, blocks = detector.detect(image)
    print(f"Unique logical blocks (after deduplication & mask recovery): {len(blocks)}")

    print("\n[2/3] EasyOCR (Line-Level + Border-Clean) başlatılıyor...")
    ocr_engine = MangaOCREngine(config.ocr, device=device)
    ocr_results = ocr_engine.recognize_blocks(image, blocks, mask_refined=mask_refined)
    print("EasyOCR tamamlandı.\n")

    records = []
    visualization = image.copy()

    for blk, ocr_res in zip(blocks, ocr_results):
        index = blk.id
        region_path = REGION_DIR / f"region_{index:03d}.png"
        if ocr_res.crop_image is not None and ocr_res.crop_image.size > 0:
            cv2.imwrite(str(region_path), ocr_res.crop_image)

        line_records = [
            {
                "line_index": lr.index,
                "bbox": lr.bbox,
                "text": lr.text,
                "confidence": lr.confidence,
            }
            for lr in ocr_res.lines
        ]

        records.append(
            {
                "id": index,
                "bbox": blk.bbox,
                "language": blk.language,
                "vertical": blk.vertical,
                "text": ocr_res.text,
                "raw_ocr_text": ocr_res.raw_text,
                "confidence": ocr_res.ocr_confidence,
                "ocr_confidence": ocr_res.ocr_confidence,
                "mask_confidence": blk.mask_confidence,
                "font_size": blk.font_size,
                "line_count": len(line_records),
                "lines": line_records,
                "line_polygons": blk.lines,
                "recovered_from_mask": blk.recovered_from_mask,
                "merged_duplicates": blk.merged_duplicates,
                "reading_order_weight": blk.reading_order_weight,
                "region_image": str(region_path.relative_to(ROOT_DIR)),
            }
        )

        draw_detection(
            visualization,
            blk.bbox,
            blk.line_bboxes,
            index,
            blk.mask_confidence,
            ocr_res.ocr_confidence,
            ocr_res.text,
        )
        print(
            f"[{index:02d}] bbox={blk.bbox} lines={len(line_records)} "
            f"mask_conf={blk.mask_confidence:.3f} ocr_conf={ocr_res.ocr_confidence:.3f} "
            f"text={ocr_res.text!r}"
        )

    print("\n[3/3] Çıktılar kaydediliyor...")
    stem = INPUT_PATH.stem
    json_path = OUTPUT_DIR / f"{stem}.json"
    detected_path = OUTPUT_DIR / f"{stem}_detected.jpg"
    mask_path = OUTPUT_DIR / f"{stem}_mask.png"
    refined_path = OUTPUT_DIR / f"{stem}_mask_refined.png"

    output_payload = {
        "image": str(INPUT_PATH.relative_to(ROOT_DIR)),
        "image_width": int(image.shape[1]),
        "image_height": int(image.shape[0]),
        "device": device,
        "reading_order": config.detector.reading_order,
        "detector": {
            "model": str(config.detector.model_path.relative_to(ROOT_DIR)),
            "input_size": config.detector.input_size,
            "conf_thresh": config.detector.conf_thresh,
            "nms_thresh": config.detector.nms_thresh,
            "iou_dup_thresh": config.detector.iou_dup_thresh,
        },
        "ocr": {
            "engine": "EasyOCR (Line-Level + Block Fusion)",
            "languages": config.ocr.languages,
            "gpu": ocr_engine.gpu,
        },
        "blocks": records,
    }

    save_json(json_path, output_payload, indent=2)
    cv2.imwrite(str(detected_path), visualization)
    cv2.imwrite(str(mask_path), mask)
    cv2.imwrite(str(refined_path), mask_refined)

    print("\nTamamlandı.")
    print(f"JSON          : {json_path}")
    print(f"Visualization : {detected_path}")
    print(f"Mask          : {mask_path}")
    print(f"Refined mask  : {refined_path}")
    print(f"Regions       : {REGION_DIR}")
    print(f"Unique Blocks : {len(records)}")


if __name__ == "__main__":
    main()
