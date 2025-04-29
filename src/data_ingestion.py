import os
import shutil
import argparse
import yaml
import json
import logging
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

def ingest_local(src_root, dest_root):
    """
    Copy the entire directory tree from src_root → dest_root,
    preserving train/ and test/ subfolders.
    If src_root == dest_root, skip copying.
    """
    abs_src = os.path.abspath(src_root)
    abs_dest = os.path.abspath(dest_root)
    if abs_src == abs_dest:
        logger.info(f"Source and destination are identical ({abs_src}); skipping copy.")
        return

    if not os.path.isdir(src_root):
        raise FileNotFoundError(f"Source folder not found: {src_root}")

    shutil.rmtree(dest_root, ignore_errors=True)
    shutil.copytree(src_root, dest_root)
    logger.info(f"Ingested data from {src_root} → {dest_root}")


def validate_structure(data_root, classes, min_per_class, allowed_exts, max_size, report_path):
    report = {"errors": [], "warnings": [], "stats": {}}

    for split in ("train", "test"):
        split_dir = os.path.join(data_root, split)
        if not os.path.isdir(split_dir):
            report["errors"].append(f"Missing split folder: {split}/")
            continue

        report["stats"][split] = {}
        for cls in classes:
            cls_dir = os.path.join(split_dir, cls)
            if not os.path.isdir(cls_dir):
                report["errors"].append(f"Missing class folder: {split}/{cls}/")
                continue

            files = [
                f for f in os.listdir(cls_dir)
                if os.path.splitext(f)[1].lower() in allowed_exts
            ]
            report["stats"][split][cls] = len(files)

            if len(files) < min_per_class:
                report["errors"].append(
                    f"{split}/{cls}: only {len(files)} files, expected ≥ {min_per_class}"
                )

            for fname in files:
                path = os.path.join(cls_dir, fname)
                try:
                    with Image.open(path) as img:
                        w, h = img.size
                        if w > max_size[0] or h > max_size[1]:
                            report["warnings"].append(
                                f"{split}/{cls}/{fname}: size {w}×{h} > {max_size[0]}×{max_size[1]}"
                            )
                except Exception as e:
                    report["errors"].append(
                        f"{split}/{cls}/{fname}: cannot open ({e})"
                    )

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    if report["errors"]:
        logger.error(f"Validation FAILED—see report at {report_path}")
        raise RuntimeError("Data validation errors encountered")
    else:
        logger.info(f"Validation passed—report at {report_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Ingest and validate 3D-print defect data")
    p.add_argument(
        "--config", type=str, default="config.yaml",
        help="YAML config defining source path and validation rules"
    )
    p.add_argument(
        "--data-root", type=str, default="data/",
        help="Destination root for ingested data (must end up with data/train/ and data/test/)"
    )
    p.add_argument(
        "--report-out", type=str,
        default="metrics/data_validation_report.json",
        help="Path to write validation report JSON"
    )
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))
    src_cfg = cfg["data_source"]
    val_cfg = cfg["validation"]

    # 1) ingest
    if src_cfg["type"] == "local":
        ingest_local(src_cfg["path"], args.data_root)
    else:
        raise NotImplementedError(
            f"Ingestion for source type '{src_cfg['type']}' not implemented"
        )

    # 2) validate
    validate_structure(
        data_root       = args.data_root,
        classes         = val_cfg["classes"],
        min_per_class   = val_cfg["min_per_class"],
        allowed_exts    = val_cfg["allowed_extensions"],
        max_size        = val_cfg["max_image_size"],
        report_path     = args.report_out
    )
