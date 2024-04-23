# Copyright (C) 2021-2024, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import os

from doctr.file_utils import CLASS_NAME

os.environ["USE_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import time
from pathlib import Path

import tensorflow as tf
from keras import mixed_precision
from tqdm import tqdm

gpu_devices = tf.config.experimental.list_physical_devices("GPU")
if any(gpu_devices):
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)

from doctr import datasets
from doctr import transforms as T
from doctr.datasets import DataLoader
from doctr.models import detection
from doctr.utils.metrics import LocalizationConfusion


def evaluate(model, val_loader, batch_transforms, val_metric):
    # Reset val metric
    val_metric.reset()
    # Validation loop
    val_loss, batch_cnt = 0, 0
    for images, targets in tqdm(val_loader):
        images = batch_transforms(images)
        targets = [{CLASS_NAME: t["boxes"]} for t in targets]
        out = model(images, targets, training=False, return_preds=True)
        # Compute metric
        loc_preds = out["preds"]
        for target, loc_pred in zip(targets, loc_preds):
            for boxes_gt, boxes_pred in zip(target.values(), loc_pred.values()):
                # Remove scores
                val_metric.update(gts=boxes_gt, preds=boxes_pred[:, :-1])

        val_loss += out["loss"].numpy()
        batch_cnt += 1

    val_loss /= batch_cnt
    recall, precision, mean_iou = val_metric.summary()
    return val_loss, recall, precision, mean_iou


def main(args):
    print(args)

    # AMP
    if args.amp:
        mixed_precision.set_global_policy("mixed_float16")

    input_shape = (args.size, args.size, 3) if isinstance(args.size, int) else None

    # Load docTR model
    model = detection.__dict__[args.arch](
        pretrained=not isinstance(args.resume, str),
        assume_straight_pages=not args.rotation,
        input_shape=input_shape,
    )

    # Resume weights
    if isinstance(args.resume, str):
        print(f"Resuming {args.resume}")
        model.load_weights(args.resume).expect_partial()

    input_shape = model.cfg["input_shape"] if input_shape is None else input_shape
    mean, std = model.cfg["mean"], model.cfg["std"]

    st = time.time()
    ds = datasets.__dict__[args.dataset](
        train=True,
        download=True,
        use_polygons=args.rotation,
        sample_transforms=T.Resize(input_shape[:2]),
    )
    # Monkeypatch
    subfolder = ds.root.split("/")[-2:]
    ds.root = str(Path(ds.root).parent.parent)
    ds.data = [(os.path.join(*subfolder, name), target) for name, target in ds.data]
    _ds = datasets.__dict__[args.dataset](
        train=False,
        download=True,
        use_polygons=args.rotation,
        sample_transforms=T.Resize(input_shape[:2]),
    )
    subfolder = _ds.root.split("/")[-2:]
    ds.data.extend([(os.path.join(*subfolder, name), target) for name, target in _ds.data])

    test_loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        drop_last=False,
        shuffle=False,
    )
    print(f"Test set loaded in {time.time() - st:.4}s ({len(ds)} samples in " f"{len(test_loader)} batches)")

    batch_transforms = T.Normalize(mean=mean, std=std)

    # Metrics
    metric = LocalizationConfusion(use_polygons=args.rotation)

    print("Running evaluation")
    val_loss, recall, precision, mean_iou = evaluate(model, test_loader, batch_transforms, metric)
    print(
        f"Validation loss: {val_loss:.6} (Recall: {recall:.2%} | Precision: {precision:.2%} | "
        f"Mean IoU: {mean_iou:.2%})"
    )


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="docTR evaluation script for text detection (TensorFlow)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("arch", type=str, help="text-detection model to evaluate")
    parser.add_argument("--dataset", type=str, default="FUNSD", help="Dataset to evaluate on")
    parser.add_argument("-b", "--batch_size", type=int, default=2, help="batch size for evaluation")
    parser.add_argument("--size", type=int, default=None, help="model input size, H = W")
    parser.add_argument("--rotation", dest="rotation", action="store_true", help="inference with rotated bbox")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume")
    parser.add_argument("--amp", dest="amp", help="Use Automatic Mixed Precision", action="store_true")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
